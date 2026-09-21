"""Full-event-graph seed recall followed by chronological stream expansion."""
from __future__ import annotations

import json
import math

from .retrieval import bm25, event_bounds, term_match, validate_plan
from .stream_index import stream_event_ids


def _records(memory, events):
    people = {person["id"]: person for person in memory.get("entities", [])}
    observations = {observation["id"]: observation for observation in memory.get("observations", [])}
    result = []
    for event in events:
        record = dict(event)
        if record.get("summary_stale"):
            record.pop("summary", None)
            record.pop("updates", None)
        refs = set(event.get("observation_refs", []))
        refs.update(ref for state in event.get("states", []) for ref in state.get("evidence_refs", []))
        record["observations"] = [observations[ref] for ref in sorted(refs) if ref in observations]
        subjects = {state.get("subject") for state in event.get("states", [])}
        subjects.update(observation.get("subject") for observation in record["observations"])
        record["people"] = [people[person] for person in sorted(subjects) if person in people]
        record["relations"] = [relation for relation in memory.get("relations", [])
                                if event["id"] in (relation.get("source"), relation.get("target"))]
        result.append(record)
    return result


def _timeline_entry(event):
    return {"id": event["id"], "span": list(event_bounds(event)),
            "states": [{"subject": state.get("subject"), "target": state.get("target"),
                        "emotion": state.get("emotion")} for state in event.get("states", [])]}


def retrieve_stream(memory, question, plan, *, budget_chars=48000, top_k=12,
                    dense_scores=None, stream_index=None, page_size=64, trace=None):
    """Retrieve from every event, then read relevant chronological streams.

    The event graph remains the recall universe.  Adjacent events are added
    only after seed selection through derived streams and explicit relations.
    """
    if budget_chars < 1000 or top_k < 1 or page_size < 1:
        raise ValueError("invalid stream retrieval configuration")
    if dense_scores is None:
        raise ValueError("event stream retrieval requires dense embedding scores")
    validate_plan(plan)
    if stream_index is None:
        raise ValueError("event stream index is required")
    bounds = plan.get("time_range") or [0, memory["duration"]]
    all_events = list(memory.get("events", []))
    events = [event for event in all_events
              if event_bounds(event)[1] >= bounds[0] and event_bounds(event)[0] <= bounds[1]]
    records = _records(memory, events)
    by_id = {event["id"]: event for event in events}
    query = question + " " + " ".join(plan["entity_terms"] + plan["target_terms"] + plan["query_terms"])
    lexical = bm25([json.dumps(record, ensure_ascii=False) for record in records], query)
    entity_hits = [term_match(plan["entity_terms"], json.dumps(record["people"], ensure_ascii=False))
                   for record in records]
    target_hits = [term_match(plan["target_terms"], " ".join(
        str(state.get("target", "")) + " " + str(state.get("emotion", ""))
        for state in record.get("states", []))) for record in records]
    semantic = [score + 3 * person + 2 * target
                for score, person, target in zip(lexical, entity_hits, target_hits)]
    semantic_ranked = sorted(range(len(events)), key=lambda i: (-semantic[i], event_bounds(events[i])[0], events[i]["id"]))
    dense_ranked = sorted(range(len(events)), key=lambda i: (-dense_scores.get(events[i]["id"], float("-inf")),
                                                              event_bounds(events[i])[0], events[i]["id"]))
    if any(events[i]["id"] not in dense_scores or not math.isfinite(dense_scores[events[i]["id"]]) for i in range(len(events))):
        raise ValueError("missing/nonfinite event embedding score")
    fused = {}
    for route in (semantic_ranked[:top_k], dense_ranked[:top_k]):
        for rank, i in enumerate(route, 1):
            fused[i] = fused.get(i, 0.0) + 1.0 / (60 + rank)
    seeds = sorted(fused, key=lambda i: (-fused[i], -semantic[i], event_bounds(events[i])[0], events[i]["id"]))
    seed_ids = [events[i]["id"] for i in seeds]
    stream_ids = sorted({stream_id for event_id in seed_ids
                         for stream_id in stream_index.get("event_to_streams", {}).get(event_id, [])})

    # Expand explicit graph relations by two hops, retaining only events in the
    # requested scope.  The closure is a navigation set, not a causal claim.
    relation_ids = set(seed_ids)
    for _ in range(2):
        for relation in memory.get("relations", []):
            if relation.get("source") in relation_ids or relation.get("target") in relation_ids:
                relation_ids.update((relation.get("source"), relation.get("target")))
    relation_ids.intersection_update(by_id)

    pages = []
    stream_ids_seen = set(stream_ids)
    stream_event_ids_all = set()
    page = 0
    while True:
        ids, total = stream_event_ids(stream_index, stream_ids, bounds=bounds, page=page, page_size=page_size)
        if not ids:
            break
        pages.append({"page": page, "returned": len(ids), "total": total})
        stream_event_ids_all.update(ids)
        page += 1
        if len(ids) < page_size:
            break
    selected_ids = stream_event_ids_all | relation_ids | set(seed_ids)
    selected_ids.intersection_update(by_id)
    ordered_ids = sorted(selected_ids, key=lambda event_id: (event_bounds(by_id[event_id]), event_id))
    ordered_records = [next(record for record in records if record["id"] == event_id) for event_id in ordered_ids]
    timeline = [_timeline_entry(by_id[event_id]) for event_id in ordered_ids]
    result = {"events": [], "timeline": [], "coverage": {
        "scope": bounds, "candidate_events": len(events), "seed_events": len(seed_ids),
        "stream_ids": stream_ids, "stream_events": len(stream_event_ids_all),
        "relation_events": len(relation_ids), "stream_pages": pages,
    }}
    # Timeline entries are compact but still budgeted.  Keep the chronological
    # stream visible to the answerer before adding full evidence records.
    for entry in timeline:
        candidate = {**result, "timeline": result["timeline"] + [entry]}
        if len(json.dumps(candidate, ensure_ascii=False)) > budget_chars * 0.25:
            break
        result["timeline"].append(entry)
    included = []
    for record in ordered_records:
        candidate = {**result, "events": result["events"] + [record]}
        if len(json.dumps(candidate, ensure_ascii=False)) > budget_chars - 256:
            continue
        result["events"].append(record)
        included.append(record["id"])
    omitted = [event_id for event_id in ordered_ids if event_id not in included]
    result["coverage"].update(returned_events=len(included), timeline_events=len(result["timeline"]),
                               omitted_event_ids=omitted,
                               memory_source_windows=len(memory.get("completed_windows", [])))
    if len(json.dumps(result, ensure_ascii=False)) > budget_chars:
        result["coverage"]["omitted_event_count"] = len(result["coverage"].pop("omitted_event_ids"))
    if trace is not None:
        trace.update(algorithm="semantic_dense_rrf_graph_stream_v1", rrf_constant=60,
                     recall_k=top_k, semantic=[{"id": events[i]["id"], "score": semantic[i],
                     "bm25": lexical[i], "entity_match": entity_hits[i], "target_match": target_hits[i]}
                     for i in semantic_ranked[:top_k]],
                     dense=[{"id": events[i]["id"], "cosine": dense_scores[events[i]["id"]]}
                            for i in dense_ranked[:top_k]],
                     fused=[{"id": events[i]["id"], "rrf": fused[i]} for i in seeds],
                     seed_event_ids=seed_ids, stream_ids=stream_ids,
                     stream_pages=pages, stream_event_ids=sorted(stream_event_ids_all),
                     relation_event_ids=sorted(relation_ids), selected_event_ids=included)
    return result

