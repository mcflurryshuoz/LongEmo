"""Structured semantic + dense event recall, rank fusion and graph expansion."""
from __future__ import annotations

from collections import Counter
import json
import math
import re


def tokens(value):
    value = str(value).lower()
    western = re.findall(r"[a-z0-9]+", value)
    cjk = re.findall(r"[\u4e00-\u9fff]+", value)
    return western + [s[i:i+2] for s in cjk for i in range(max(1, len(s)-1))]


def bm25(documents, query):
    counts = [Counter(tokens(doc)) for doc in documents]
    lengths = [sum(c.values()) for c in counts]
    average = sum(lengths) / max(1, len(lengths)) or 1
    df = Counter(t for count in counts for t in count)
    scores = []
    for count, length in zip(counts, lengths):
        score = 0.0
        for term in set(tokens(query)):
            frequency = count[term]
            idf = math.log(1 + (len(counts) - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * length / average))
        scores.append(score)
    return scores


def event_bounds(event):
    return min(s[0] for s in event["spans"]), max(s[1] for s in event["spans"])


def validate_plan(value):
    if not isinstance(value, dict) or value.get("mode") not in ("trajectory", "comparison", "count", "causal", "local"):
        raise ValueError("invalid retrieval mode")
    for field in ("entity_terms", "target_terms", "query_terms"):
        if not isinstance(value.get(field), list) or not all(isinstance(t, str) for t in value[field]):
            raise ValueError(f"{field} must contain strings")
    interval = value.get("time_range")
    if interval is not None and (not isinstance(interval, list) or len(interval) != 2 or
        any(type(x) not in (int, float) or not math.isfinite(x) for x in interval) or interval[0] < 0 or interval[1] <= interval[0]):
        raise ValueError("invalid explicit question time range")


def term_match(terms, text):
    words = set(tokens(text))
    return sum(len(set(tokens(term)) & words) / max(1, len(set(tokens(term)))) for term in terms)


def retrieve(memory, question, plan, *, mode="graph", budget_chars=48000, top_k=12, dense_scores=None, trace=None):
    if mode not in ("graph", "flat") or budget_chars < 1000 or top_k < 1:
        raise ValueError("invalid retrieval configuration")
    if mode == "graph" and dense_scores is None:
        raise ValueError("event graph retrieval requires dense embedding scores; no silent fallback")
    people = {p["id"]: p for p in memory["entities"]}
    observations = {o["id"]: o for o in memory["observations"]}
    bounds = plan.get("time_range") or [0, memory["duration"]]
    events = [e for e in memory["events"] if event_bounds(e)[1] >= bounds[0] and event_bounds(e)[0] <= bounds[1]]
    records = []
    for event in events:
        # Both conditions receive exactly the same underlying fields, including
        # relation facts. Only candidate selection/order differs.
        record = dict(event)
        if record.get("summary_stale"):
            record.pop("summary", None)
            record.pop("updates", None)
        refs = set(event["observation_refs"]) | {ref for state in event["states"] for ref in state["evidence_refs"]}
        record["observations"] = [observations[r] for r in sorted(refs)]
        record["people"] = [people[p] for p in sorted({s["subject"] for s in event["states"]} |
            {o["subject"] for o in record["observations"]})]
        record["relations"] = [r for r in memory["relations"] if event["id"] in (r["source"], r["target"])]
        records.append(record)
    query = question + " " + " ".join(plan["entity_terms"] + plan["target_terms"] + plan["query_terms"])
    lexical = bm25([json.dumps(record, ensure_ascii=False) for record in records], query)
    entity_hits = [term_match(plan["entity_terms"], json.dumps(r["people"], ensure_ascii=False)) for r in records]
    target_hits = [term_match(plan["target_terms"], " ".join(s["target"]+" "+s["emotion"] for s in r["states"])) for r in records]
    semantic = [score + 3*person + 2*target for score, person, target in zip(lexical, entity_hits, target_hits)]
    semantic_ranked = sorted(range(len(events)), key=lambda i: (-semantic[i], event_bounds(events[i])[0]))
    global_scope = plan["mode"] in ("trajectory", "comparison", "count", "causal")
    neighbors = set()
    dense_ranked, fused = [], {}
    ranked = semantic_ranked
    if mode == "graph":
        if any(e["id"] not in dense_scores or not math.isfinite(dense_scores[e["id"]]) for e in events):
            raise ValueError("missing/nonfinite event embedding score")
        dense_ranked = sorted(range(len(events)), key=lambda i: (-dense_scores[events[i]["id"]], event_bounds(events[i])[0]))
        for route in (semantic_ranked[:top_k], dense_ranked[:top_k]):
            for rank, i in enumerate(route, 1):
                fused[i] = fused.get(i, 0.0) + 1.0/(60+rank)
        seeds = sorted(fused, key=lambda i: (-fused[i], -semantic[i], event_bounds(events[i])[0]))
        selected_ids = {events[i]["id"] for i in seeds}
        neighbors = {r[key] for r in memory["relations"] if r["source"] in selected_ids or r["target"] in selected_ids
                     for key in ("source", "target")}
        ranked = seeds + [i for i in semantic_ranked if i not in fused]
        # Each person's adjacent events are navigation links, never new causal claims.
        for subject in {s["subject"] for i in seeds for s in events[i]["states"]}:
            chain = sorted([i for i,e in enumerate(events) if any(s["subject"] == subject for s in e["states"])],
                           key=lambda i: event_bounds(events[i])[0])
            for pos, i in enumerate(chain):
                if i in seeds:
                    neighbors.update(events[j]["id"] for j in chain[max(0,pos-1):pos+2])
        neighbor_order = [i for i in ranked if events[i]["id"] in neighbors and i not in seeds]
        diverse = []
        if global_scope:
            # Soft person matching prioritizes relevant storylines without losing
            # evidence when the cast index or question identity is uncertain.
            bins = {}
            coverage_rank = sorted(ranked, key=lambda i: (-entity_hits[i], ranked.index(i)))
            for i in coverage_rank:
                relative = (event_bounds(events[i])[0] - bounds[0]) / max(1, bounds[1]-bounds[0])
                bins.setdefault(min(11, max(0, int(relative*12))), []).append(i)
            diverse = [bins[bucket][0] for bucket in sorted(bins)]
        order = seeds[:min(4, top_k)] + neighbor_order[:4] + diverse + seeds + neighbor_order + ranked
    else:
        order = ranked
    if trace is not None:
        trace.update(algorithm="semantic_dense_rrf_graph_v1" if mode == "graph" else "semantic_only",
            rrf_constant=60, recall_k=top_k,
            semantic=[{"id": events[i]["id"], "score": semantic[i], "bm25": lexical[i],
                       "entity_match": entity_hits[i], "target_match": target_hits[i]} for i in semantic_ranked[:top_k]],
            dense=[{"id": events[i]["id"], "cosine": dense_scores[events[i]["id"]]} for i in dense_ranked[:top_k]],
            fused=[{"id": events[i]["id"], "rrf": fused[i]} for i in ranked if i in fused],
            graph_neighbor_ids=sorted(neighbors))
    order = list(dict.fromkeys(order))
    result = {"events": [], "timeline": [], "coverage": {"scope": bounds, "candidate_events": len(events)}}
    if mode == "graph" and global_scope:
        timeline = [{"id": e["id"], "span": list(event_bounds(e)),
            "states": [{"subject": s["subject"], "target": s["target"], "emotion": s["emotion"]} for s in e["states"]]}
            for e in sorted(events, key=lambda e: event_bounds(e)[0])]
        # Explicitly report any truncation of the lightweight index.
        for entry in timeline:
            candidate = {**result, "timeline": result["timeline"] + [entry]}
            if len(json.dumps(candidate, ensure_ascii=False)) > budget_chars * 0.25:
                break
            result["timeline"].append(entry)
    included = []
    for index in order:
        candidate = {**result, "events": result["events"] + [records[index]]}
        if len(json.dumps(candidate, ensure_ascii=False)) <= budget_chars - 256:
            result["events"].append(records[index])
            included.append(index)
        if mode == "flat" and len(included) >= top_k:
            break
    # Keep an explicit equal character cap; a top-k baseline can legitimately
    # use less. The report records actual sizes instead of claiming equal usage.
    if mode == "graph":
        result["events"].sort(key=event_bounds)
    result["coverage"].update(returned_events=len(included), timeline_events=len(result["timeline"]),
        omitted_event_ids=[events[i]["id"] for i in range(len(events)) if i not in included],
        memory_source_windows=len(memory["completed_windows"]))
    # Omitted IDs may themselves overflow a very small budget: keep a count.
    if len(json.dumps(result, ensure_ascii=False)) > budget_chars:
        result["coverage"]["omitted_event_count"] = len(result["coverage"].pop("omitted_event_ids"))
    if trace is not None:
        trace["selected_event_ids"] = [r["id"] for r in result["events"]]
    return result
