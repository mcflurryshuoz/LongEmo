"""Semantic and dense retrieval over independent chronological window records."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .common import fingerprint
from .noevent_memory import person_index
from .retrieval import bm25, term_match


def _document(memory, window):
    people = {p["id"]: p for p in memory["entities"]}
    names = [f"{people[p]['name'] or p}: {people[p]['description']}" for p in window.get("participants", []) if p in people]
    cues = []
    for cue in window.get("emotion_cues", []):
        person = people.get(cue["subject"], {}).get("name") or cue["subject"]
        cues.append(f"{person} {cue['emotion']} about {cue['target']}; {cue['intensity']}")
    observations = {o["id"]: o for o in memory["observations"]}
    observed = [observations[o]["cue"] for o in window.get("observation_ids", []) if o in observations]
    return "\n".join(["People: " + "; ".join(names), "Summary: " + window.get("summary", ""),
                     "Actions: " + "; ".join(window.get("actions", [])),
                     "Objects: " + "; ".join(window.get("objects", [])),
                     "Signals: " + "; ".join(window.get("signals", [])),
                     "Emotion cues: " + "; ".join(cues), "Observed: " + "; ".join(observed)])


class WindowIndex:
    def __init__(self, memory, encoder, directory):
        import numpy as np
        self.encoder = encoder
        windows = memory["windows"]
        self.window_ids = [w["id"] for w in windows]
        documents = [_document(memory, w) for w in windows]
        signature = fingerprint({"documents": list(zip(self.window_ids, documents)), "encoder": encoder.config})
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (memory["video_id"] + "-windows-" + signature + ".npz")
        reused = path.exists()
        if reused:
            with np.load(path, allow_pickle=False) as stored:
                self.vectors = stored["vectors"]
                self.window_ids = stored["window_ids"].tolist()
            if self.vectors.shape != (len(self.window_ids), encoder.config["dimension"]):
                raise ValueError("invalid cached window embedding matrix")
        else:
            chunks, ids = [], []
            for window_id, document in zip(self.window_ids, documents):
                pieces = encoder.chunks(document)
                chunks.extend(pieces)
                ids.extend([window_id] * len(pieces))
            self.vectors = encoder.encode(chunks)
            self.window_ids = ids
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, vectors=self.vectors, window_ids=np.asarray(ids, dtype=str))
            temporary.replace(path)
        self.metadata = {"signature": signature, "video_id": memory["video_id"],
                         "windows": len(windows), "chunks": len(self.window_ids),
                         "encoder": encoder.config, "cache_reused": reused}

    def rank(self, question):
        vector = self.encoder.encode([question], query=True)[0]
        values = self.vectors @ vector
        scores = {}
        for window_id, score in zip(self.window_ids, values):
            scores[window_id] = max(scores.get(window_id, -1.0), float(score))
        return scores


def _window_record(window, observations, people):
    """Resolve every citation before budgeting the actual answer evidence."""
    record = dict(window)
    refs = list(dict.fromkeys(window["observation_ids"] +
        [ref for cue in window.get("emotion_cues", []) for ref in cue["evidence_refs"]]))
    if any(ref not in observations for ref in refs):
        raise ValueError("window evidence contains an unknown observation reference")
    record["observations"] = [observations[ref] for ref in refs]
    subjects = set(window.get("participants", [])) | {
        observation["subject"] for observation in record["observations"]} | {
        cue["subject"] for cue in window.get("emotion_cues", [])}
    if any(subject not in people for subject in subjects):
        raise ValueError("window evidence contains an unknown person reference")
    record["people"] = [people[subject] for subject in sorted(subjects)]
    return record


def _packet(records, *, available, global_scope, budget_chars, ranking):
    evidence_ids = list(dict.fromkeys([record["id"] for record in records] +
        [observation["id"] for record in records for observation in record["observations"]]))
    packet = {"representation": "window_records", "windows": records,
              "evidence_ids": evidence_ids,
              "timeline": [{"window_id": record["id"], "core": record["core"],
                            "summary": record["summary"]} for record in records],
              "coverage": {"selected_windows": len(records), "available_windows": available,
                           "global_scope": global_scope, "budget_characters": budget_chars,
                           "used_characters": 0}, "ranking": ranking}
    # Include all wrappers and the character counter itself, using the exact
    # JSON serialization the answer client receives. Never truncate JSON text.
    while True:
        used = len(json.dumps(packet, ensure_ascii=False))
        if packet["coverage"]["used_characters"] == used:
            return packet
        packet["coverage"]["used_characters"] = used


def retrieve(memory, question, plan, *, dense_scores, budget_chars=48000, top_k=12):
    if budget_chars < 1000 or top_k < 1:
        raise ValueError("invalid window retrieval configuration")
    if not isinstance(dense_scores, dict):
        raise ValueError("window retrieval requires dense embedding scores")
    windows = list(memory["windows"])
    bounds = plan.get("time_range") or [0, memory["duration"]]
    windows = [w for w in windows if w["core"][1] >= bounds[0] and w["core"][0] <= bounds[1]]
    if any(w["id"] not in dense_scores or not math.isfinite(dense_scores[w["id"]]) for w in windows):
        raise ValueError("missing/nonfinite window embedding score")
    documents = [_document(memory, w) for w in windows]
    query = question + " " + " ".join(plan["entity_terms"] + plan["target_terms"] + plan["query_terms"])
    lexical = bm25(documents, query)
    semantic = [score + 2 * term_match(plan["entity_terms"], doc) + term_match(plan["target_terms"], doc)
                for score, doc in zip(lexical, documents)]
    semantic_rank = sorted(range(len(windows)), key=lambda i: (-semantic[i], windows[i]["core"][0]))
    dense_rank = sorted(range(len(windows)), key=lambda i: (-dense_scores.get(windows[i]["id"], -1), windows[i]["core"][0]))
    fused = {}
    for route in (semantic_rank[:top_k], dense_rank[:top_k]):
        for rank, i in enumerate(route, 1):
            fused[i] = fused.get(i, 0.0) + 1.0 / (60 + rank)
    seeds = sorted(fused, key=lambda i: (-fused[i], -semantic[i], windows[i]["core"][0]))
    global_scope = plan["mode"] in ("trajectory", "comparison", "count", "causal")
    ranked = seeds + [i for i in semantic_rank if i not in fused]
    if global_scope:
        selected = ranked
    else:
        selected = ranked[:max(top_k, 4)]
    selected = sorted(selected, key=lambda i: windows[i]["core"][0])
    observations = {observation["id"]: observation for observation in memory["observations"]}
    people = {person["id"]: person for person in person_index(memory)}
    ranking = {"semantic_top": [windows[i]["id"] for i in semantic_rank[:top_k]],
               "dense_top": [windows[i]["id"] for i in dense_rank[:top_k]]}
    metadata = {"available": len(windows), "global_scope": global_scope,
                "budget_chars": budget_chars, "ranking": ranking}
    chosen = []
    packet = _packet(chosen, **metadata)
    if packet["coverage"]["used_characters"] > budget_chars:
        raise ValueError("window retrieval metadata exceeds the evidence budget")
    for i in selected:
        record = _window_record(windows[i], observations, people)
        candidate = _packet(chosen + [record], **metadata)
        if candidate["coverage"]["used_characters"] > budget_chars:
            continue
        chosen.append(record)
        packet = candidate
    if selected and not chosen:
        raise ValueError("no complete window evidence fits the configured budget")
    return packet
