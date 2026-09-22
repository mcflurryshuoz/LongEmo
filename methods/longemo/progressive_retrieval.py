"""Progressive disclosure over a frozen event graph.

The graph is queried as a database.  Only a small set of anchor events is
returned initially; the answerer can request validated pages of related
events.  This keeps the question context focused while preserving a complete
navigation trace for auditing.
"""
from __future__ import annotations

import json
import math

from .retrieval import bm25, event_bounds, term_match, validate_plan
from .stream_retrieval import _records


SCOPES = {"same_target", "same_subject", "relations", "time_range"}
DIRECTIONS = {"before", "after", "both"}


def validate_expand_request(value):
    if not isinstance(value, dict):
        raise ValueError("retrieve_more must be an object")
    anchors = value.get("anchor_ids")
    if not isinstance(anchors, list) or not anchors or not all(isinstance(x, str) for x in anchors):
        raise ValueError("retrieve_more.anchor_ids must be a nonempty string list")
    if value.get("scope", "same_target") not in SCOPES:
        raise ValueError("invalid progressive retrieval scope")
    if value.get("direction", "both") not in DIRECTIONS:
        raise ValueError("invalid progressive retrieval direction")
    if "relation_types" in value and (not isinstance(value["relation_types"], list) or
                                       not all(isinstance(x, str) for x in value["relation_types"])):
        raise ValueError("relation_types must be a string list")
    if "time_range" in value and value["time_range"] is not None:
        interval = value["time_range"]
        if (not isinstance(interval, list) or len(interval) != 2 or
                any(type(x) not in (int, float) or not math.isfinite(x) for x in interval) or
                interval[0] < 0 or interval[1] <= interval[0]):
            raise ValueError("invalid progressive retrieval time range")
    page_size = value.get("page_size", 8)
    if type(page_size) is not int or not 1 <= page_size <= 16:
        raise ValueError("page_size must be between 1 and 16")
    return value


def _score_events(memory, question, plan, dense_scores):
    bounds = plan.get("time_range") or [0, memory["duration"]]
    events = [event for event in memory.get("events", [])
              if event_bounds(event)[1] >= bounds[0] and event_bounds(event)[0] <= bounds[1]]
    records = _records(memory, events)
    query = question + " " + " ".join(plan["entity_terms"] + plan["target_terms"] + plan["query_terms"])
    lexical = bm25([json.dumps(record, ensure_ascii=False) for record in records], query)
    entity_hits = [term_match(plan["entity_terms"], json.dumps(record["people"], ensure_ascii=False))
                   for record in records]
    target_hits = [term_match(plan["target_terms"], " ".join(
        str(state.get("target", "")) + " " + str(state.get("emotion", ""))
        for state in record.get("states", []))) for record in records]
    semantic = [score + 3 * person + 2 * target
                for score, person, target in zip(lexical, entity_hits, target_hits)]
    if any(event["id"] not in dense_scores or not math.isfinite(dense_scores[event["id"]]) for event in events):
        raise ValueError("missing/nonfinite event embedding score")
    dense_ranked = sorted(range(len(events)), key=lambda i: (-dense_scores[events[i]["id"]],
                                                               event_bounds(events[i]), events[i]["id"]))
    semantic_ranked = sorted(range(len(events)), key=lambda i: (-semantic[i],
                                                                 event_bounds(events[i]), events[i]["id"]))
    fused = {}
    for route in (semantic_ranked, dense_ranked):
        for rank, i in enumerate(route, 1):
            if rank > 12:
                break
            fused[i] = fused.get(i, 0.0) + 1.0 / (60 + rank)
    combined = {events[i]["id"]: (fused.get(i, 0.0), semantic[i], dense_scores[events[i]["id"]])
                for i in range(len(events))}
    return events, records, combined, bounds


def _relation_neighbors(memory, ids, relation_types=None):
    allowed = set(relation_types or [])
    result = set()
    for relation in memory.get("relations", []):
        if allowed and relation.get("type") not in allowed:
            continue
        if relation.get("source") in ids:
            result.add(relation.get("target"))
        if relation.get("target") in ids:
            result.add(relation.get("source"))
    return {event_id for event_id in result if event_id}


def _stream_ids_for(index, anchors, scope):
    if scope == "relations" or scope == "time_range":
        return set()
    streams = set()
    for event_id in anchors:
        for stream_id in index.get("event_to_streams", {}).get(event_id, []):
            if scope == "same_target":
                streams.add(stream_id)
            elif scope == "same_subject":
                subject = stream_id.split(":", 2)[1] if ":" in stream_id else ""
                for candidate, value in index.get("streams", {}).items():
                    if candidate.split(":", 2)[1] == subject:
                        streams.add(candidate)
    return streams


def _stream_candidates(index, stream_ids, bounds):
    events = index.get("_events_by_id", {})
    ids = set()
    for stream_id in stream_ids:
        stream = index.get("streams", {}).get(stream_id, {})
        for event_id in stream.get("event_ids", []):
            event = events.get(event_id)
            if event and event_bounds(event)[1] >= bounds[0] and event_bounds(event)[0] <= bounds[1]:
                ids.add(event_id)
    return ids


def _timeline(records):
    return [{"id": record["id"], "span": list(event_bounds(record)),
             "states": [{"subject": state.get("subject"), "target": state.get("target"),
                          "emotion": state.get("emotion")} for state in record.get("states", [])]}
            for record in records]


def _fit_payload(events, budget_chars):
    selected = []
    for record in events:
        candidate = {"events": selected + [record], "timeline": _timeline(selected + [record])}
        if len(json.dumps(candidate, ensure_ascii=False)) > budget_chars:
            break
        selected.append(record)
    return {"events": selected, "timeline": _timeline(selected)}


def initial_disclosure(memory, question, plan, *, dense_scores, stream_index,
                       budget_chars=24000, anchor_k=4, neighbor_count=1, trace=None):
    """Return a small anchor packet and an auditable disclosure state."""
    validate_plan(plan)
    events, records, scores, bounds = _score_events(memory, question, plan, dense_scores)
    by_id = {event["id"]: event for event in events}
    by_record = {record["id"]: record for record in records}
    seeds = sorted(events, key=lambda event: (-scores[event["id"]][0],
                                               -scores[event["id"]][1], event_bounds(event), event["id"]))[:anchor_k]
    anchor_ids = [event["id"] for event in seeds]
    selected_ids = set(anchor_ids)
    # One immediate temporal neighbor per side is useful for local changes, but
    # never opens the entire stream before the answerer asks for it.
    subject_events = {}
    for event in events:
        for state in event.get("states", []):
            subject_events.setdefault(state.get("subject"), []).append(event)
    for subject in subject_events:
        subject_events[subject].sort(key=lambda event: (event_bounds(event), event["id"]))
    for anchor in seeds:
        subjects = {state.get("subject") for state in anchor.get("states", [])}
        for subject in subjects:
            chain = subject_events.get(subject, [])
            pos = next((i for i, event in enumerate(chain) if event["id"] == anchor["id"]), None)
            if pos is not None:
                for offset in (-neighbor_count, neighbor_count):
                    if 0 <= pos + offset < len(chain):
                        selected_ids.add(chain[pos + offset]["id"])
    ordered = sorted((by_record[event_id] for event_id in selected_ids),
                     key=lambda record: (-scores[record["id"]][0], event_bounds(record), record["id"]))
    # Keep the first disclosure deliberately small.  Neighbor discovery is a
    # navigation aid; it must not silently turn the initial packet into a
    # whole-person timeline.
    ordered = ordered[:anchor_k + 4]
    payload = _fit_payload(ordered, budget_chars)
    state = {"anchor_ids": anchor_ids, "revealed_ids": [record["id"] for record in payload["events"]],
             "seen_pages": [], "bounds": bounds, "event_count": len(events),
             "scores": {key: list(value) for key, value in scores.items()},
             "plan": plan, "question": question}
    coverage = {"candidate_events": len(events), "anchor_events": len(anchor_ids),
                "revealed_events": len(state["revealed_ids"]), "requested_expansions": 0,
                "omitted_events": max(0, len(selected_ids) - len(state["revealed_ids"]))}
    payload["coverage"] = coverage
    if trace is not None:
        trace.update(algorithm="progressive_anchor_stream_v1", anchor_ids=anchor_ids,
                     initial_ids=state["revealed_ids"], coverage=coverage)
    return payload, state


def expand_disclosure(memory, state, request, *, stream_index, budget_chars=24000, trace=None):
    """Expand only from validated anchors and return the next page."""
    request = validate_expand_request(request)
    events = memory.get("events", [])
    by_id = {event["id"]: event for event in events}
    anchors = [event_id for event_id in request["anchor_ids"] if event_id in by_id]
    if not anchors:
        raise ValueError("retrieve_more contains no known anchor")
    bounds = request.get("time_range") or state.get("bounds") or [0, memory["duration"]]
    scope = request.get("scope", "same_target")
    candidate_ids = set()
    if scope in ("same_target", "same_subject"):
        candidate_ids.update(_stream_candidates(stream_index, _stream_ids_for(stream_index, anchors, scope), bounds))
    if scope == "relations":
        candidate_ids.update(_relation_neighbors(memory, set(anchors), request.get("relation_types")))
    if scope == "time_range":
        candidate_ids.update(event_id for event_id, event in by_id.items()
                             if event_bounds(event)[1] >= bounds[0] and event_bounds(event)[0] <= bounds[1])
    candidate_ids.update(_relation_neighbors(memory, set(anchors), request.get("relation_types")))
    candidate_ids.difference_update(state.get("revealed_ids", []))
    anchor_end = max(event_bounds(by_id[event_id])[1] for event_id in anchors)
    anchor_start = min(event_bounds(by_id[event_id])[0] for event_id in anchors)
    direction = request.get("direction", "both")
    if direction == "before":
        candidate_ids = {event_id for event_id in candidate_ids if event_bounds(by_id[event_id])[1] <= anchor_start}
    elif direction == "after":
        candidate_ids = {event_id for event_id in candidate_ids if event_bounds(by_id[event_id])[0] >= anchor_end}
    records = _records(memory, [by_id[event_id] for event_id in candidate_ids])
    scores = state.get("scores", {})
    def rank(record):
        start, end = event_bounds(record)
        distance = min(abs(start - anchor_end), abs(end - anchor_start))
        score = scores.get(record["id"], [0.0, 0.0, 0.0])
        direction_bias = 0 if direction == "both" else (0 if (direction == "before" and end <= anchor_start) or
                                                          (direction == "after" and start >= anchor_end) else 1)
        return (direction_bias, distance, -score[0], -score[1], start, record["id"])
    records.sort(key=rank)
    page = records[:request.get("page_size", 8)]
    payload = _fit_payload(page, budget_chars)
    page_ids = [record["id"] for record in payload["events"]]
    state.setdefault("seen_pages", []).append({"request": request, "event_ids": page_ids})
    state.setdefault("revealed_ids", []).extend(page_ids)
    state["requested_expansions"] = len(state["seen_pages"])
    payload["coverage"] = {"candidate_events": len(candidate_ids),
                            "revealed_events": len(state["revealed_ids"]),
                            "page_events": len(page_ids),
                            "has_more": len(records) > len(page_ids),
                            "requested_expansions": state["requested_expansions"]}
    if trace is not None:
        trace.setdefault("expansions", []).append({"request": request, "returned_ids": page_ids,
                                                    "candidate_events": len(candidate_ids)})
    return payload, state
