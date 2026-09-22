"""Window-only audiovisual memory used for the no-event ablation.

This schema deliberately has no events, states, relations, or continuity links.
Every record remains owned by its fixed time window so retrieval can compare the
window-only representation with the event-graph method under identical media
sampling and question/answer calls.
"""
from __future__ import annotations

import copy
import math


def _text(value, field, optional=False):
    if optional and value is None:
        return None
    if not isinstance(value, str) or (not optional and not value.strip()):
        raise ValueError(f"{field} must be nonempty text")
    return value.strip()


def _list(value, field):
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(x, str) or not x.strip() for x in value):
        raise ValueError(f"{field} must contain text values")
    return list(dict.fromkeys(x.strip() for x in value))


def _span(value, bounds):
    if (not isinstance(value, list) or len(value) != 2 or
            any(type(x) not in (int, float) or not math.isfinite(x) for x in value)):
        raise ValueError("span must contain two finite seconds")
    start, end = value
    if end < start or start < bounds[0] - .25 or end > bounds[1] + .25:
        raise ValueError("span outside supplied audiovisual window")
    return [max(bounds[0], float(start)), min(bounds[1], float(end))]


def empty_memory(video_id, duration, video_sha):
    return {"schema_version": 1, "representation": "window_records", "video_id": video_id,
            "duration": duration, "video_sha256": video_sha, "entities": [],
            "observations": [], "windows": [], "sources": [], "completed_windows": []}


def apply_window(memory, payload, *, window_id, core, media, metadata):
    """Validate an entire window on a copy and publish it atomically."""
    if not isinstance(payload, dict) or window_id in memory["completed_windows"]:
        raise ValueError("invalid perception payload or duplicate window")
    for field in ("entities", "observations"):
        if not isinstance(payload.get(field), list):
            raise ValueError(f"{field} must be a list")
    result = copy.deepcopy(memory)
    entities = {p["id"]: p for p in result["entities"]}
    entity_map = {}
    for item in payload["entities"]:
        local = _text(item.get("id"), "entity.id")
        if local in entity_map:
            raise ValueError("duplicate local entity ID")
        name = _text(item.get("name"), "entity.name", optional=True)
        desc = _text(item.get("description"), "entity.description")
        if local in entities:
            entity_map[local] = local
            entities[local]["source_refs"].append(window_id)
            if name and not entities[local].get("name"):
                entities[local]["name"] = name
        else:
            canonical = f"P{len(result['entities']) + 1}"
            entity_map[local] = canonical
            record = {"id": canonical, "name": name, "description": desc, "source_refs": [window_id]}
            result["entities"].append(record)
            entities[canonical] = record
    observations = {}
    for item in payload["observations"]:
        local = _text(item.get("id"), "observation.id")
        if local in observations:
            raise ValueError("duplicate local observation ID")
        subject = item.get("subject")
        if subject not in entity_map:
            raise ValueError("observation subject must reference a declared entity")
        modality = item.get("modality")
        if modality not in ("visual", "audio", "subtitle", "multimodal"):
            raise ValueError("unsupported observation modality")
        if modality == "audio" and not metadata.get("audio"):
            raise ValueError("cannot claim audio evidence without audio input")
        if modality == "subtitle" and not metadata.get("subtitle_ids"):
            raise ValueError("cannot claim subtitle evidence without subtitle input")
        interval = _span(item.get("span"), media)
        if interval[1] < core[0] or interval[0] >= core[1]:
            raise ValueError("observation must overlap the owned core interval")
        canonical = f"O{len(result['observations']) + 1}"
        observations[local] = canonical
        result["observations"].append({"id": canonical, "subject": entity_map[subject],
            "span": interval, "cue": _text(item.get("cue"), "observation.cue"),
            "modality": modality, "source_ref": window_id})
    cues = []
    raw_cues = payload.get("emotion_cues", [])
    if not isinstance(raw_cues, list):
        raise ValueError("emotion_cues must be a list")
    for cue in raw_cues:
        subject = cue.get("subject")
        if subject not in entity_map:
            raise ValueError("emotion cue subject must be declared")
        refs = cue.get("evidence_ids")
        if not isinstance(refs, list) or not refs or any(x not in observations for x in refs):
            raise ValueError("emotion cue needs supporting observation IDs")
        cues.append({"subject": entity_map[subject], "target": _text(cue.get("target"), "emotion target"),
                     "emotion": _text(cue.get("emotion"), "emotion"),
                     "intensity": _text(cue.get("intensity"), "intensity"),
                     "evidence_refs": [observations[x] for x in dict.fromkeys(refs)]})
    window = {"id": window_id, "core": list(core), "media": list(media),
              "summary": _text(payload.get("summary"), "window.summary"),
              "actions": _list(payload.get("actions"), "window.actions"),
              "objects": _list(payload.get("objects"), "window.objects"),
              "signals": _list(payload.get("signals"), "window.signals"),
              "participants": list(dict.fromkeys(entity_map[x] for x in _list(payload.get("participants"), "window.participants") if x in entity_map)),
              "emotion_cues": cues,
              "observation_ids": list(observations.values())}
    result["windows"].append(window)
    result["sources"].append({"id": window_id, "core": list(core), "media": list(media), "sampling": metadata})
    result["completed_windows"].append(window_id)
    return result
