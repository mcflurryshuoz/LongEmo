"""Import a validated one-request probe into an unstarted noevent continuation.

Only a new continuation cache is written.  The parent and the diagnostic are
immutable.  Importing audio populates its exact cache; importing vision applies
the original validator and advances one window, so the builder skips that HTTP
request.  A seed intent without a completed receipt is an audit stop.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import uuid

from experiments.zyf import noevent_retry_probe as probe


def regular(path):
    path = Path(path).absolute()
    if any(item.is_symlink() for item in (path, *path.parents)) or not path.is_file():
        raise ValueError("seed input must be an ordinary file without symlink ancestors")
    return path


def read(path):
    return probe.read(regular(path))


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def target_folder(run, video):
    return Path(run) / "noevent/videos" / video / "memory" / video


def validate_probe(probe_dir, prepared):
    folder = Path(probe_dir).absolute()
    names = ("summary.json", "request_started.json", "response.json", "validated_payload.json")
    hashes = {name: probe.sha(regular(folder / name)) for name in names}
    summary, started, response, artifact = (read(folder / name) for name in names)
    request_hash = prepared["failed"]["request_hash"]
    if (summary.get("status") != "validated" or summary.get("reusable") is not True or
            summary.get("parent_files_unchanged") is not True or summary.get("official_result") is not False or
            summary.get("parse_status") != "ok" or summary.get("validate_status") != "ok" or
            summary.get("artifact_sha256") != hashes["validated_payload.json"]):
        raise ValueError("probe has no immutable successful validation receipt")
    if any(value.get("request_hash") != request_hash for value in (summary, started, response, artifact)):
        raise ValueError("probe request differs from the reconstructed failed request")
    if started.get("maximum_http_requests") != 1 or summary.get("maximum_http_requests") != 1:
        raise ValueError("probe does not prove a single request budget")
    wire = json.dumps(prepared["adapter"].build_payload(prepared["client"], prepared["messages"]), allow_nan=False).encode()
    if started.get("wire_sha256") != hashlib.sha256(wire).hexdigest():
        raise ValueError("probe wire payload differs from the reconstructed request")
    if (type(summary.get("http_status")) is not int or not 200 <= summary["http_status"] < 300 or
            summary["http_status"] != response.get("http_status") or response.get("body_redacted") is not False):
        raise ValueError("probe response is not an intact successful HTTP response")
    body = response["body_text"].encode("utf-8")
    body_sha = hashlib.sha256(body).hexdigest()
    if body_sha != response.get("body_sha256") or body_sha != artifact.get("response_sha256"):
        raise ValueError("probe response body hash changed")
    raw = json.loads(body)
    if probe.refused(raw):
        raise ValueError("refusal cannot be imported as a successful observation")
    parsed = prepared["adapter"].parse_response(prepared["client"], raw)
    payload = prepared["parse_json"](parsed["content"])
    if payload != artifact.get("payload") or artifact.get("official_result") is not False:
        raise ValueError("validated payload differs from the original HTTP response")
    for key, value in prepared["artifact"].items():
        if artifact.get(key) != value:
            raise ValueError("probe model, media, or checkpoint provenance differs")
    prepared["validate"](payload)
    return artifact, {str(folder / name): value for name, value in hashes.items()}


def verify_seed_receipt(run, video, checkpoint, *, allow_memory_growth=False):
    """Return known seed hashes/provenance; never repair an interrupted import."""
    root = Path(run) / "probe_seeds" / video
    if not root.exists():
        return None
    receipt = read(root / "receipt.json")
    intent = read(root / "intent.json")
    if (receipt.get("schema_version") != 1 or receipt.get("video_id") != video or
            receipt.get("checkpoint_receipt_sha256") != digest(checkpoint) or
            receipt.get("intent_sha256") != probe.sha(root / "intent.json") or
            intent.get("checkpoint_receipt_sha256") != digest(checkpoint)):
        raise ValueError("seed receipt differs from frozen inheritance")
    window = receipt.get("window_id", "")
    stage = receipt.get("stage")
    if (stage not in ("audio", "visual") or not re.fullmatch(r"W[0-9]{5}", window) or
            receipt.get("source") != "validated_one_request_probe" or receipt.get("official_result") is not False or
            receipt.get("no_api_calls") is not True):
        raise ValueError("seed receipt identity is invalid")
    output = ("windows/" if stage == "visual" else "audio/") + window + ".json"
    if set(receipt.get("target_files_after", {})) != {"memory.json", output}:
        raise ValueError("seed receipt contains unexpected target paths")
    for path, expected in receipt["probe_source_hashes"].items():
        if probe.sha(regular(path)) != expected:
            raise ValueError("seed diagnostic source changed")
    target = target_folder(run, video)
    for relative, expected in receipt["target_files_after"].items():
        if relative == "memory.json" and allow_memory_growth:
            memory = read(target / relative)
            if receipt["stage"] == "visual" and receipt["window_id"] not in memory["completed_windows"]:
                raise ValueError("seed window disappeared from the later memory")
        elif probe.sha(regular(target / relative)) != expected:
            raise ValueError("seed output changed before it could be reused")
    return receipt


def exclusive_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Match the core builder's serialization for downstream audio source hashes.
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def replace_known_memory(path, value, expected):
    path = regular(path)
    temporary = path.with_name(path.name + ".seed-" + uuid.uuid4().hex)
    exclusive_json(temporary, value)
    if probe.sha(path) != expected:
        raise ValueError("target memory changed during the seed import")
    os.replace(temporary, path)


def apply_seed(parent_run, continuation_run, video_id, probe_dir, repo):
    parent, run, repo = (Path(value).absolute() for value in (parent_run, continuation_run, repo))
    if (run == parent or parent in run.parents or run in parent.parents or
            run == repo or repo in run.parents or any(x.is_symlink() for x in (run, *run.parents))):
        raise ValueError("continuation must be an isolated ordinary directory")
    config = read(run / "configuration.json")
    if (read(run / "prepared.json")["configuration_sha256"] != probe.sha(run / "configuration.json") or
            config.get("condition") != "noevent" or config.get("execution_conditions") != ["noevent"] or
            config.get("parent_run") != str(parent) or config.get("core_repo") != str(repo)):
        raise ValueError("seed target is not the matching frozen noevent continuation")
    checkpoint = config["checkpoints"][video_id]
    if read(run / "inheritance" / (video_id + ".json")) != checkpoint:
        raise ValueError("continuation inheritance receipt changed")
    if any((run / "tasks" / stage / ("noevent-" + video_id) / "task.json").exists() for stage in ("build", "answer", "score")):
        raise ValueError("probe cannot seed an already-claimed continuation task")
    with (run / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any((run / "tasks" / stage / ("noevent-" + video_id) / "task.json").exists() for stage in ("build", "answer", "score")):
            raise ValueError("a continuation task was claimed while waiting for the seed lock")
        existing = verify_seed_receipt(run, video_id, checkpoint)
        if existing:
            return existing
        prepared = probe.prepare(parent, video_id, repo)  # Media reconstruction only; no HTTP.
        artifact, source_hashes = validate_probe(probe_dir, prepared)
        if (checkpoint["parent_manifest_sha256"] != artifact["parent_manifest_sha256"] or
                checkpoint["committed_files"].get("memory.json") != artifact["parent_memory_sha256"]):
            raise ValueError("continuation cloned a different parent checkpoint")
        source = Path(checkpoint["source"])
        for relative, expected in checkpoint["files"].items():
            if probe.sha(regular(source / relative)) != expected:
                raise ValueError("original checkpoint changed since it was cloned")
        target = target_folder(run, video_id)
        copied = {**checkpoint["committed_files"], **checkpoint["pending_audio_strict_validation"]}
        for relative, expected in copied.items():
            if probe.sha(regular(target / relative)) != expected:
                raise ValueError("unstarted target is not the original validated clone")
        if (target / "manifest.json").exists() or (target / "calls.jsonl").exists():
            raise ValueError("target builder has already initialized")
        memory_path, window = target / "memory.json", artifact["window_id"]
        memory = read(memory_path)
        if len(memory["completed_windows"]) != checkpoint["completed_windows"] or window in memory["completed_windows"]:
            raise ValueError("target seed is not the first unfinished window")
        if artifact["stage"] == "audio":
            outputs = {"audio/" + window + ".json": {"input_fingerprint": artifact["audio_input_fingerprint"],
                "model": artifact["model_configuration"], "result": artifact["payload"]}}
            updated = None
        elif artifact["stage"] == "visual":
            trace = artifact["sampling"].get("audio_observer") or {}
            audio_path = target / "audio" / (window + ".json")
            if probe.sha(regular(audio_path)) != trace.get("source_sha256"):
                raise ValueError("seed vision does not reference the target's exact pending audio cache")
            runner = sys.modules["methods.longemo.noevent_runner"]
            updated = runner.apply_window(memory, artifact["payload"], window_id=window,
                                          core=artifact["core"], media=artifact["interval"], metadata=artifact["sampling"])
            total = math.ceil(memory["duration"] / config["profile"]["window_seconds"])
            updated["complete"] = len(updated["completed_windows"]) == total
            outputs = {"windows/" + window + ".json": {"input": artifact["context"],
                "sampling": artifact["sampling"], "perception": artifact["payload"]}}
        else:
            raise ValueError("unknown probe stage")
        if any((target / name).exists() or (target / name).is_symlink() for name in outputs):
            raise ValueError("seed output would overwrite an unknown existing cache")
        seed_root = run / "probe_seeds" / video_id
        seed_root.mkdir(parents=True, exist_ok=False, mode=0o700)
        intent = {"schema_version": 1, "video_id": video_id, "window_id": window, "stage": artifact["stage"],
                  "started_unix": time.time(), "checkpoint_receipt_sha256": digest(checkpoint),
                  "probe_source_hashes": source_hashes, "no_api_calls": True}
        exclusive_json(seed_root / "intent.json", intent)
        for name, value in outputs.items():
            exclusive_json(target / name, value)
        if updated is not None:
            replace_known_memory(memory_path, updated, artifact["parent_memory_sha256"])
        after = {name: probe.sha(target / name) for name in outputs}
        after["memory.json"] = probe.sha(memory_path)
        if any(probe.sha(regular(path)) != expected for path, expected in prepared["protected"].items()):
            raise ValueError("parent changed during the import")
        receipt = {"schema_version": 1, "video_id": video_id, "window_id": window, "stage": artifact["stage"],
                   "checkpoint_receipt_sha256": digest(checkpoint), "intent_sha256": probe.sha(seed_root / "intent.json"),
                   "probe_source_hashes": source_hashes, "target_files_before": copied, "target_files_after": after,
                   "source": "validated_one_request_probe", "request_hash": artifact["request_hash"],
                   "model_configuration": artifact["model_configuration"], "official_result": False, "no_api_calls": True}
        exclusive_json(seed_root / "receipt.json", receipt)
        return verify_seed_receipt(run, video_id, checkpoint)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("parent-run", "continuation-run", "video-id", "probe-dir", "repo"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    try:
        result = apply_seed(args.parent_run, args.continuation_run, args.video_id, args.probe_dir, args.repo)
    except Exception as exc:
        print(json.dumps({"status": "seed_rejected_or_needs_audit", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({key: result[key] for key in ("video_id", "window_id", "stage", "source", "no_api_calls")}))
    return 0


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
