"""Derived chronological stream indexes for a frozen event memory.

The event memory remains the only source of truth.  This module builds a
reproducible navigation view after perception is complete; it never mutates
events, states, observations, or relations.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict


def event_bounds(event):
    spans = event.get("spans") or []
    if not spans:
        raise ValueError(f"event {event.get('id')} has no spans")
    return min(float(span[0]) for span in spans), max(float(span[1]) for span in spans)


def _normalise(value):
    value = re.sub(r"\s+", " ", str(value or "").strip().lower())
    return value


def _stream_id(subject, target):
    digest = hashlib.sha1(f"{subject}\x00{target}".encode("utf-8")).hexdigest()[:12]
    return f"stream:{subject}:{digest}"


def _event_subjects(event):
    subjects = {state.get("subject") for state in event.get("states", []) if state.get("subject")}
    return sorted(subjects)


def _event_targets(event):
    targets = {_normalise(state.get("target")) for state in event.get("states", [])}
    return sorted(target for target in targets if target) or ["__unknown__"]


def build_stream_index(memory):
    """Return an immutable derived index over all events in ``memory``.

    Exact target strings intentionally create candidate streams rather than
    making an irreversible semantic merge.  Relation and temporal links are
    retained separately so the retriever can widen a stream when evidence
    supports it.
    """
    events = list(memory.get("events", []))
    streams = defaultdict(list)
    event_to_streams = defaultdict(list)
    event_by_id = {event["id"]: event for event in events}
    for event in events:
        subjects = _event_subjects(event) or ["__unknown_subject__"]
        for subject in subjects:
            for target in _event_targets(event):
                stream_id = _stream_id(subject, target)
                streams[stream_id].append(event["id"])
                event_to_streams[event["id"]].append(stream_id)

    temporal_edges = []
    # A temporal edge is a navigation fact, never a causal claim.  Add links
    # only for consecutive events of the same subject so the graph stays
    # sparse and deterministic.
    subject_events = defaultdict(list)
    for event in events:
        for subject in _event_subjects(event):
            subject_events[subject].append(event)
    for subject, ordered in subject_events.items():
        ordered.sort(key=lambda event: (event_bounds(event), event["id"]))
        for left, right in zip(ordered, ordered[1:]):
            left_end = event_bounds(left)[1]
            right_start = event_bounds(right)[0]
            edge_type = "overlaps" if right_start <= left_end else "precedes"
            temporal_edges.append({"source": left["id"], "target": right["id"],
                                   "type": edge_type, "subject": subject,
                                   "evidence": [left["id"], right["id"]]})

    streams_json = {}
    for stream_id, ids in streams.items():
        unique = sorted(set(ids), key=lambda event_id: (event_bounds(event_by_id[event_id]), event_id))
        subjects, targets = stream_id.split(":", 2)[1:]
        streams_json[stream_id] = {"id": stream_id, "subject": subjects,
                                   "target_key": targets, "event_ids": unique,
                                   "count": len(unique)}
    return {
        "schema_version": 1,
        "video_id": memory.get("video_id"),
        "duration": memory.get("duration"),
        "memory_source_windows": len(memory.get("completed_windows", [])),
        "event_count": len(events),
        "streams": streams_json,
        "event_to_streams": {key: sorted(value) for key, value in event_to_streams.items()},
        "temporal_edges": temporal_edges,
    }


def stream_event_ids(index, stream_ids, *, bounds=None, page=0, page_size=64):
    """Read one deterministic page of chronological event IDs."""
    if page < 0 or page_size < 1:
        raise ValueError("invalid stream page")
    low, high = bounds if bounds is not None else (float("-inf"), float("inf"))
    events = index.get("_events_by_id") or {}
    ids = set()
    for stream_id in stream_ids:
        stream = index.get("streams", {}).get(stream_id)
        if not stream:
            continue
        for event_id in stream["event_ids"]:
            event = events.get(event_id)
            if event is None:
                continue
            start, end = event_bounds(event)
            if end >= low and start <= high:
                ids.add(event_id)
    ordered = sorted(ids, key=lambda event_id: (event_bounds(events[event_id]), event_id))
    start = page * page_size
    return ordered[start:start + page_size], len(ordered)


def attach_events(index, memory):
    """Attach a private lookup map for retrieval; JSON serialization omits it."""
    index = dict(index)
    index["_events_by_id"] = {event["id"]: event for event in memory.get("events", [])}
    return index


def public_metadata(index):
    """Return JSON-safe metadata for manifests and traces."""
    return {key: value for key, value in index.items() if not key.startswith("_")}

