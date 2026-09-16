"""BM25 baseline and temporal/relational retrieval over identical evidence."""
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


def retrieve(memory, question, plan, *, mode="graph", budget_chars=48000, top_k=12):
    if mode not in ("graph", "flat") or budget_chars < 1000 or top_k < 1:
        raise ValueError("invalid retrieval configuration")
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
    scores = bm25([json.dumps(record, ensure_ascii=False) for record in records], query)
    ranked = sorted(range(len(events)), key=lambda i: (-scores[i], event_bounds(events[i])[0]))
    order = ranked[:top_k]
    global_scope = plan["mode"] in ("trajectory", "comparison", "count", "causal")
    if mode == "graph":
        selected_ids = {events[i]["id"] for i in order}
        neighbors = {r[key] for r in memory["relations"] if r["source"] in selected_ids or r["target"] in selected_ids
                     for key in ("source", "target")}
        order += [i for i in ranked if events[i]["id"] in neighbors]
        if global_scope:
            # Allocate coverage across time before exhausting the budget with
            # near-duplicate highly similar events from a single scene.
            bins = {}
            for i in ranked:
                relative = (event_bounds(events[i])[0] - bounds[0]) / max(1, bounds[1]-bounds[0])
                bins.setdefault(min(11, max(0, int(relative * 12))), []).append(i)
            diverse = []
            while any(bins.values()):
                for bucket in sorted(bins):
                    if bins[bucket]:
                        diverse.append(bins[bucket].pop(0))
            order = ranked[:min(3, top_k)] + diverse + order
        order += ranked
    else:
        order += ranked
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
    return result
