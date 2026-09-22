"""Validated event-owned emotion states, provenance and versioned corrections."""
from __future__ import annotations

import copy
import math


def empty_memory(video_id, duration, video_sha):
    return {"schema_version": 2, "video_id": video_id, "duration": duration, "video_sha256": video_sha,
            "entities": [], "observations": [], "events": [], "relations": [], "sources": [],
            "state_history": [], "completed_windows": []}


def text(value, field, *, optional=False):
    if optional and value is None:
        return None
    if not isinstance(value, str) or (not optional and not value.strip()):
        raise ValueError(f"{field} must be nonempty text")
    return value.strip()


def span(value, bounds):
    if not isinstance(value, list) or len(value) != 2 or any(type(x) not in (int, float) or not math.isfinite(x) for x in value):
        raise ValueError("span must contain two finite absolute video seconds")
    start, end = value
    if end < start or start < bounds[0] - 0.25 or end > bounds[1] + 0.25:
        raise ValueError("span outside supplied audiovisual window")
    return [max(bounds[0], float(start)), min(bounds[1], float(end))]


def references(values, mapping, field):
    if not isinstance(values, list) or not values:
        raise ValueError(f"{field} needs supporting observation IDs")
    if any(not isinstance(x, str) or x not in mapping for x in values):
        raise ValueError(f"unknown reference in {field}")
    return list(dict.fromkeys(mapping[x] for x in values))


def text_list(value, field):
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{field} must contain text values")
    return list(dict.fromkeys(item.strip() for item in value))


def confidence(value, field):
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{field} must be between 0 and 1")
    return float(value)


SIGNAL_KINDS = {"explicit_score", "count", "ordinal", "superlative", "quote", "label"}
EVENT_TYPES = {"observation", "action", "interaction", "speech", "reaction", "transition", "outcome", "evaluation"}


def signals(value, field):
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    result = []
    for signal in value:
        if not isinstance(signal, dict) or signal.get("kind") not in SIGNAL_KINDS:
            raise ValueError(f"{field}.kind is invalid")
        kind = signal["kind"]
        value_text = text(signal.get("value"), f"{field}.value")
        evidence_text = text(signal.get("text"), f"{field}.text")
        result.append({"kind": kind, "value": value_text, "text": evidence_text})
    return result


def state_index(memory):
    """Derived view only: state objects remain inside their parent event."""
    return {s["id"]: (e, s) for e in memory["events"] for s in e["states"]}


def apply_window(memory, payload, *, window_id, core, media, metadata, allow_revisions=False):
    """Validate on a copy and publish atomically; failed outputs never partly mutate memory."""
    if not isinstance(payload, dict):
        raise ValueError("perception output must be an object")
    for field in ("entities", "observations", "events", "relations", "corrections"):
        if not isinstance(payload.get(field), list):
            raise ValueError(f"{field} must be a list")
    if window_id in memory["completed_windows"]:
        raise ValueError("window already committed")
    result = copy.deepcopy(memory)
    entities = {p["id"]: p for p in result["entities"]}
    entity_map = {p: p for p in entities}
    local_ids = set()
    for p in payload["entities"]:
        local = text(p.get("id"), "entity.id")
        if local in local_ids:
            raise ValueError("duplicate local entity ID")
        local_ids.add(local)
        name = text(p.get("name"), "name", optional=True)
        description = text(p.get("description"), "entity description")
        if local not in entities:
            canonical = f"P{len(result['entities']) + 1}"
            record = {"id": canonical, "name": name, "description": description, "source_refs": [window_id]}
            result["entities"].append(record)
            entities[canonical] = record
            entity_map[local] = canonical
        else:
            entities[local]["source_refs"].append(window_id)
            if name and not entities[local]["name"]:
                entities[local]["name"] = name
    observations = {}
    for observation in payload["observations"]:
        local = text(observation.get("id"), "observation.id")
        if local in observations:
            raise ValueError("duplicate local observation ID")
        subject = observation.get("subject")
        if subject not in entity_map:
            raise ValueError("observation subject must reference a declared entity")
        modality = observation.get("modality")
        if modality not in ("visual", "audio", "subtitle", "multimodal"):
            raise ValueError("unsupported observation modality")
        if modality == "audio" and not metadata["audio"]:
            raise ValueError("cannot claim audio evidence without audio input")
        if modality == "subtitle" and not metadata["subtitle_ids"]:
            raise ValueError("cannot claim subtitle evidence without subtitle input")
        canonical = f"O{len(result['observations'])+1}"
        observations[local] = canonical
        result["observations"].append({"id": canonical, "subject": entity_map[subject],
            "span": span(observation.get("span"), media), "cue": text(observation.get("cue"), "cue"),
            "modality": modality, "source_ref": window_id})
    event_map = {e["id"]: e["id"] for e in memory["events"]}
    existing = {e["id"]: e for e in result["events"]}
    local_events = set()
    state_number = len(state_index(result))
    for item in payload["events"]:
        local = text(item.get("id"), "event.id")
        if local in existing:
            raise ValueError("use new_event_1 style local IDs, not existing E IDs; link via continues_event")
        if local in local_events:
            raise ValueError("duplicate local event ID")
        local_events.add(local)
        interval = span(item.get("span"), media)
        if interval[1] < core[0] or interval[0] >= core[1]:
            raise ValueError("event must overlap core, not only padding")
        refs = references(item.get("observation_ids"), observations, "event.observation_ids")
        previous = item.get("continues_event")
        if previous is not None:
            if previous not in existing:
                raise ValueError("continues_event must refer to an existing event")
            event_type = text(item.get("event_type", existing[previous].get("event_type", "observation")),
                              "event.event_type")
            if event_type not in EVENT_TYPES:
                raise ValueError("event.event_type is invalid")
            event = existing[previous]
            event["event_type"] = event_type
            event["version"] += 1
            event["spans"].append(interval)
            event["observation_refs"] = list(dict.fromkeys(event["observation_refs"] + refs))
            event["window_ids"].append(window_id)
            event["updates"].append({"window": window_id, "summary": text(item.get("summary"), "event.summary")})
            action = text(item.get("action"), "event.action", optional=True)
            if action:
                event["actions"] = list(dict.fromkeys(event.get("actions", []) + [action]))
            event["objects"] = list(dict.fromkeys(event.get("objects", []) + text_list(item.get("objects"), "event.objects")))
            participant_locals = text_list(item.get("participants"), "event.participants")
            if any(p not in entity_map for p in participant_locals):
                raise ValueError("event.participants must reference declared entities")
            event["participants"] = list(dict.fromkeys(event.get("participants", []) +
                [entity_map[p] for p in participant_locals]))
            event["signals"] = event.get("signals", []) + signals(item.get("signals"), "event.signals")
            current_confidence = confidence(item.get("confidence"), "event.confidence")
            if current_confidence is not None:
                event["confidence"] = max(event.get("confidence", 0.0), current_confidence)
        else:
            action = text(item.get("action"), "event.action", optional=True)
            event_type = text(item.get("event_type", "observation"), "event.event_type")
            if event_type not in EVENT_TYPES:
                raise ValueError("event.event_type is invalid")
            objects = text_list(item.get("objects"), "event.objects")
            participant_locals = text_list(item.get("participants"), "event.participants")
            if any(p not in entity_map for p in participant_locals):
                raise ValueError("event.participants must reference declared entities")
            participants = [entity_map[p] for p in participant_locals]
            participants = list(dict.fromkeys(participants + [entity_map[s["subject"]] for s in item.get("states", [])
                                                               if s.get("subject") in entity_map]))
            event_signals = signals(item.get("signals"), "event.signals")
            event_confidence = confidence(item.get("confidence"), "event.confidence")
            event = {"id": f"E{len(result['events'])+1}", "version": 1, "spans": [interval],
                     "summary": text(item.get("summary"), "event.summary"), "observation_refs": refs,
                     "window_ids": [window_id], "states": [], "updates": [],
                     "event_type": event_type,
                     "actions": [action] if action else [], "objects": objects,
                     "participants": participants, "signals": event_signals,
                     "confidence": event_confidence}
            result["events"].append(event)
            existing[event["id"]] = event
        event_map[local] = event["id"]
        if not isinstance(item.get("states"), list):
            raise ValueError("event.states must be a list")
        for state in item["states"]:
            if state.get("subject") not in entity_map:
                raise ValueError("state.subject must be declared")
            state_number += 1
            event["states"].append({"id": f"S{state_number}", "version": 1,
                "subject": entity_map[state["subject"]], "span": interval,
                "target": text(state.get("target"), "emotion target"),
                "emotion": text(state.get("emotion"), "emotion"),
                "intensity": text(state.get("intensity"), "intensity"),
                "uncertainty": text(state.get("uncertainty", ""), "uncertainty", optional=True),
                "appraisal": text(state.get("appraisal"), "appraisal", optional=True),
                "evidence_refs": references(state.get("evidence_ids"), observations, "state.evidence_ids"),
                "known_at_window": window_id})
    for edge in payload["relations"]:
        if edge.get("type") not in ("temporal", "causal", "changes_to", "coexists_with"):
            raise ValueError("invalid relation type")
        if edge.get("source") not in event_map or edge.get("target") not in event_map:
            raise ValueError("relation endpoint not found")
        source, target = event_map[edge["source"]], event_map[edge["target"]]
        if source != target:
            result["relations"].append({"source": source, "target": target, "type": edge["type"],
                "evidence_refs": references(edge.get("evidence_ids"), observations, "relation.evidence_ids")})
    if payload["corrections"] and not allow_revisions:
        raise ValueError("corrections disabled")
    index = state_index(result)
    for correction in payload["corrections"]:
        key = correction.get("state_id")
        if key not in index:
            raise ValueError("correction state not found")
        event, state = index[key]
        if correction.get("expected_version") != state["version"]:
            raise ValueError("stale state revision")
        result["state_history"].append({"event_id": event["id"], "state": copy.deepcopy(state),
            "revised_at": window_id, "reason": text(correction.get("reason"), "correction reason")})
        state.update(version=state["version"]+1, emotion=text(correction.get("emotion"), "correction emotion"),
                     uncertainty=text(correction.get("uncertainty", ""), "uncertainty", optional=True))
        state["evidence_refs"] = list(dict.fromkeys(state["evidence_refs"] + references(correction.get("evidence_ids"), observations, "correction.evidence_ids")))
        event["version"] += 1
        event["summary_stale"] = True
        # Conservative invalidation: relations depending on this event must be
        # regenerated from evidence; no stale derived causal claim is returned.
        result["relations"] = [r for r in result["relations"] if event["id"] not in (r["source"], r["target"])]
    result["sources"].append({"id": window_id, "core": core, "media": media, "sampling": metadata})
    result["completed_windows"].append(window_id)
    return result
