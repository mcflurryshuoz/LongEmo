"""Fixed V100/V114/V91 audio429 probes: one additional HTTP request per case.

The three old attempts are immutable and exhausted. Each separate probe proves
all three original same-message requests, preserves its source visual profile,
and freezes the existing 243 first scores. No audio is imported and no builder,
answerer, or judge starts here. A global lock serializes these diagnostic calls.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from experiments.zyf import noevent_continuation as base
from experiments.zyf import noevent_length_continuation as lineage
from experiments.zyf import noevent_retry_probe as probe


SPECS_SHA = "c213425b298bdda99edcce1aa8161fa05ad150dfaf1591994c2169f827e2018e"
CORE_CONTINUATION_SHA = "1e9776267db1f38330a3bca2c66fb031ce6ec3b925abb11531a5c31748052192"
CASES = {"G2_V000100": ("W00039", 38, 76, 8192), "G2_V000114": ("W00057", 56, 64, 8192),
         "G2_V000091": ("W00025", 24, 66, 16384)}
FIRST_SCORES = 243
EXHAUSTED_MESSAGE = ("Resource exhausted. Please try again later. Please refer to "
    "https://cloud.google.com/vertex-ai/generative-ai/docs/error-code-429 for more details.")


def load_case(path, video):
    if base.sha(base.regular(path)) != SPECS_SHA:
        raise ValueError("fixed audio429 evidence specification changed")
    values = base.read(path)["cases"]
    if {value["video"] for value in values} != set(CASES) or len(values) != len(CASES) or video not in CASES:
        raise ValueError("only the three explicitly selected audio429 cases are supported")
    value = next(value for value in values if value["video"] == video)
    if tuple(value[key] for key in ("window", "completed_windows", "total_windows", "visual_max_tokens")) != CASES[video]:
        raise ValueError("fixed video, window or source visual profile differs")
    return value


def check_final(row, case):
    expected = case["ledger_rows"][-1]
    if (row != expected or row.get("attempt") != 3 or row.get("http_status") != 429
            or row.get("status") != "error" or row.get("failure_stage") != "generate"
            or row.get("error_type") != "ServiceError" or row.get("will_retry") is not False
            or row.get("retryable") is not True or row.get("retry_reason") != "temporary_http_error"
            or row.get("request_hash") != row.get("initial_request_hash")
            or probe.is_refusal(row.get("service_error_code"))):
        raise ValueError("only the exact third recorded audio429 request is eligible")


def source_lineage(source, config):
    """Read-only validation; never write observations into an older run."""
    base.verify(source, config)
    recovery = config.get("length_recovery")
    if not recovery:
        return {}
    if recovery["driver_sha256"] != base.sha(base.regular(lineage.__file__)):
        raise ValueError("source length continuation implementation changed")
    parent = Path(recovery["source_run"])
    protected = {str(parent / "configuration.json"): recovery["source_configuration_sha256"],
                 str(parent / "prepared.json"): recovery["source_prepared_sha256"]}
    if base.read(parent / "prepared.json")["configuration_sha256"] != recovery["source_configuration_sha256"]:
        raise ValueError("source length parent preparation changed")
    for video, item in recovery["videos"].items():
        lineage.terminal_video(parent, video)
        protected.update(item["source_state_hashes"])
        protected[item["evidence"]["diagnostic_path"]] = item["evidence"]["diagnostic_sha256"]
        if base.read(base.regular(item["claim_path"])) != item["claim"]:
            raise ValueError("source length recovery claim changed")
        protected[item["claim_path"]] = base.sha(item["claim_path"])
    for path, expected in protected.items():
        if base.sha(base.regular(path)) != expected:
            raise ValueError("source length ancestry evidence changed")
    return protected


def verify_history(case, folder, client, messages, adapter, child):
    ledger = base.regular(folder / "audio/calls.jsonl")
    if base.sha(ledger) != case["audio_ledger_sha256"]:
        raise ValueError("original three-attempt audio ledger changed")
    all_rows = base.records(ledger)
    rows = [row for row in all_rows if row.get("purpose") == "audio_observer:" + case["window"]]
    if rows != case["ledger_rows"] or len(rows) != 3 or all_rows[-1] != rows[-1]:
        raise ValueError("exactly the fixed three terminal audio attempts are required")
    check_final(rows[-1], case)
    request_hash = base.digest(messages)
    if any(row.get("attempt") != i or row.get("request_hash") != request_hash or row.get("initial_request_hash") != request_hash
           or row.get("http_status") != 429 or row.get("failure_stage") != "generate"
           or row.get("status") != "error" or row.get("error_type") != "ServiceError"
           or row.get("retryable") is not True or row.get("retry_reason") != "temporary_http_error"
           or row.get("will_retry") is not (i < 3) for i, row in enumerate(rows, 1)):
        raise ValueError("audio429 history is not three unchanged, non-repair requests")
    wire = json.dumps(adapter.build_payload(client, messages), allow_nan=False).encode()
    wire_item = {"payload_sha256": hashlib.sha256(wire).hexdigest(), "bytes": len(wire)}
    source = Path(case["source_run"])
    expected_paths = {str(Path(item["path"]).resolve()) for item in case["diagnostics"]}
    actual_paths = set()
    for path in (source / "private_diagnostics/build" / case["video"]).glob("*.json"):
        if not path.name.endswith(".intent.json") and base.read(base.regular(path)).get("request_hash") == request_hash:
            actual_paths.add(str(path.resolve()))
    if len(expected_paths) != 3 or actual_paths != expected_paths:
        raise ValueError("missing, duplicated or additional same-hash private wrapper evidence")
    diagnostics = sorted(case["diagnostics"], key=lambda item: datetime.fromisoformat(item["started_at"]).timestamp())
    protected, evidence = {str(ledger): base.sha(ledger)}, []
    for row, expected in zip(rows, diagnostics):
        path = base.regular(expected["path"])
        if base.sha(path) != expected["sha256"]:
            raise ValueError("a pinned original audio429 diagnostic changed")
        value = base.read(path)
        if (value.get("request_hash") != request_hash or value.get("outcome") != "error"
                or value.get("error_type") != "ServiceError" or value.get("pid") != child["pid"]
                or value.get("pid") != expected["pid"] or value.get("model_configuration") != client.configuration()
                or value.get("model_configuration") != expected["model_configuration"]
                or value.get("serialized_requests") != [wire_item] or expected["serialized_requests"] != [wire_item]
                or value.get("started_at") != expected["started_at"] or value.get("elapsed_seconds") != expected["elapsed_seconds"]):
            raise ValueError("original audio429 model, wire bytes, PID or timing differs")
        started = datetime.fromisoformat(value["started_at"])
        if (started.tzinfo is None or not 0 <= started.timestamp() - row["time_unix"] < .02
                or abs(value["elapsed_seconds"] - row["elapsed_seconds"]) >= .1):
            raise ValueError("three wrapper records cannot be aligned to the original ledger attempts")
        responses = value.get("responses")
        if not isinstance(responses, list) or len(responses) != 1 or responses[0].get("http_status") != 429:
            raise ValueError("each old outer attempt must contain exactly one HTTP429 response")
        response = responses[0]
        body = response.get("body_text")
        if (not isinstance(body, str) or hashlib.sha256(body.encode()).hexdigest() != response.get("body_sha256")
                or response["body_sha256"] != expected["responses"][0]["body_sha256"]):
            raise ValueError("original audio429 response body is unavailable, altered or redacted")
        raw = json.loads(body)
        errors = raw.get("error") if isinstance(raw, dict) else None
        fixed_error = [{"error": {"code": 429, "message": EXHAUSTED_MESSAGE, "status": "RESOURCE_EXHAUSTED"}}]
        if errors != fixed_error or errors != expected["responses"][0]["error"] or probe.refused(raw):
            raise ValueError("response is not the inspected resource-exhausted error; no policy/billing retry is allowed")
        protected[str(path)] = expected["sha256"]
        evidence.append({"attempt": row["attempt"], "request_hash": request_hash, "diagnostic_path": str(path),
                         "diagnostic_sha256": expected["sha256"], "response_sha256": response["body_sha256"],
                         "http_status": 429, "wire_sha256": wire_item["payload_sha256"], "wire_bytes": len(wire)})
    return rows[-1], protected, evidence


def prepare(specs_path, video_id, repo, related_runs):
    case = load_case(specs_path, video_id)
    source, repo = Path(case["source_run"]).resolve(strict=True), Path(repo).resolve(strict=True)
    if str(source) != case["source_run"] or base.sha(base.regular(source / "configuration.json")) != case["configuration_sha256"]:
        raise ValueError("fixed source configuration changed")
    if base.sha(base.regular(base.__file__)) != CORE_CONTINUATION_SHA:
        raise ValueError("frozen original continuation helper changed")
    config = base.read(source / "configuration.json")
    protected = source_lineage(source, config)
    if config["core_repo"] != str(repo) or config["profile"]["visual"]["max_tokens"] != case["visual_max_tokens"]:
        raise ValueError("source core or visual budget differs from the fixed case")
    protected.update(lineage.terminal_video(source, video_id))
    task = base.read(source / "tasks/build" / ("noevent-" + video_id) / "task.json")
    command = base.stage_command(source, config, "build", video_id)
    if (task.get("command") != command or task.get("cwd") != str(repo)
            or task.get("signature") != base.digest({"command": command, "cwd": str(repo)})):
        raise ValueError("source task is not the exact frozen diagnostic build command")
    modules = probe.bind_repo(repo)
    runner, common, audio_module = (modules[name] for name in
        ("methods.longemo.noevent_runner", "methods.longemo.common", "methods.longemo.audio"))
    args = runner.parser().parse_args(command[command.index("--") + 1:])
    if args.tries != 3 or args.workers != 1:
        raise ValueError("original outer attempt or worker budget changed")
    baseline = Path(config["parent_run"])
    old = base.read(base.regular(baseline / "configuration.json"))
    checkpoint = base.checkpoint_receipt(source, repo, old, video_id)
    folder = Path(checkpoint["source"])
    manifest, memory = base.read(folder / "manifest.json"), base.read(folder / "memory.json")
    completed = case["completed_windows"]
    if (checkpoint["completed_windows"] != completed or memory.get("complete")
            or memory["completed_windows"] != [f"W{i:05d}" for i in range(1, completed + 1)]
            or (folder / "audio" / (case["window"] + ".json")).exists()):
        raise ValueError("source is not the fixed incomplete window-only checkpoint")
    questions = base.read(base.regular(source / "questions" / (video_id + ".json")))
    if [q["question_id"] for q in questions] != case["question_ids"]:
        raise ValueError("the fixed unscored question selection differs")
    for parent in (baseline, source):
        base.ensure_no_backend_attempts(parent, questions)
    related = sorted({str(Path(value).resolve(strict=True)) for value in [source, *related_runs]})
    if str(baseline) in related:
        raise ValueError("baseline must not be passed as a related continuation")
    scores = lineage.score_snapshot(related, base.read(baseline / "questions.json"), set(case["question_ids"]), config["protected_scores"])
    if len(config["protected_scores"]) + len(scores) != FIRST_SCORES:
        raise ValueError("the complete 243 first-score inventory is required before probing")
    visual, client = runner.client_for(args), runner.audio_client_for(args)
    expected_audio = case["diagnostics"][0]["model_configuration"]
    if (visual.configuration() != checkpoint["parent_model"] or not client
            or client.configuration() != checkpoint["parent_audio_observer"] or client.configuration() != expected_audio
            or client.base_url.rstrip("/") != probe.MATRIX or client.max_tokens != 4096):
        raise ValueError("original audio/visual model configuration changed")
    video, subtitle = Path(args.videos_dir) / (video_id + ".mp4"), Path(args.subtitles_dir) / (video_id + ".json")
    media_identity = base.verified_media(old, [video_id])
    info = runner.probe(video)
    if (info["duration"] != case["duration"] or memory["duration"] != info["duration"]
            or math.ceil(info["duration"] / args.window_seconds) != case["total_windows"]):
        raise ValueError("source video duration or total window count changed")
    core = [completed * args.window_seconds, min((completed + 1) * args.window_seconds, info["duration"])]
    interval = [max(0, core[0] - args.padding), min(info["duration"], core[1] + args.padding)]
    media, metadata = runner.window_input(video, *interval, subtitle_rows=runner.subtitles(subtitle), **runner._media_options(args))
    audio = [part for part in media if part["type"] == "input_audio"]
    if not audio:
        raise ValueError("source audio media could not be reconstructed")
    messages = [{"role": "system", "content": audio_module.AUDIO_PROMPT}, {"role": "user", "content": [
        {"type": "text", "text": json.dumps({"clip_start": interval[0], "clip_end": interval[1]})}] + audio}]
    adapter = modules["evaluation.inference.adapters"].API_ADAPTERS[client.api_format]
    failed, history_files, evidence = verify_history(case, folder, client, messages, adapter, task["child"])
    if any(row.get("purpose") == "window_perception:" + video_id + ":" + case["window"] for row in base.records(folder / "calls.jsonl")):
        raise ValueError("the failed audio window already has a visual attempt")

    def validate(value):
        if not isinstance(value, dict) or not isinstance(value.get("observations"), list):
            raise ValueError("audio observations must be a list")
        for observation in value["observations"]:
            audio_module.span(observation.get("span"), interval)
            for field in ("voice", "cue"):
                audio_module.text(observation.get(field), field)

    protected.update({str(folder / name): expected for name, expected in checkpoint["files"].items()})
    protected.update(history_files)
    for path in (source / "configuration.json", source / "prepared.json", source / "process.json", source / "questions.json",
                 source / "questions" / (video_id + ".json"), video, subtitle, Path(specs_path), Path(__file__),
                 Path(base.__file__), Path(probe.__file__), Path(lineage.__file__), Path(base.pilot.__file__)):
        protected[str(path.resolve())] = base.sha(base.regular(path))
    for item in config["protected_scores"].values():
        protected[item["path"]] = item["sha256"]
        protected[item["envelope"]["source"]] = item["official_source_sha256"]
    for item in scores.values():
        protected[item["path"]] = item["sha256"]
        protected[item["source"]] = item["source_sha256"]
    credentials = base.read(base.regular(args.credential_file))
    keys = [value for name, value in credentials.items() if any(term in name.lower() for term in ("key", "token", "secret", "password"))
            and isinstance(value, str)]
    artifact = {"video_id": video_id, "window_id": case["window"], "stage": "audio", "source_run": str(source),
        "source_configuration_sha256": case["configuration_sha256"], "visual_max_tokens": case["visual_max_tokens"],
        "context": runner.perception_context(memory, window_id=case["window"], core=core, media=interval),
        "sampling": metadata, "core": core, "interval": interval, "model_configuration": client.configuration(),
        "audio_input_fingerprint": common.fingerprint({"messages": messages, "model": client.configuration()}),
        "parent_memory_sha256": base.sha(folder / "memory.json"), "parent_manifest_sha256": base.sha(folder / "manifest.json"),
        "original_attempts": 3, "new_additional_requests": 1, "probe_is_not_a_completed_window": True}
    return {"case": case, "client": client, "messages": messages, "failed": failed, "fingerprint": common.fingerprint,
        "validate": validate, "parse_json": modules["evaluation.inference.prompts"].json_object, "adapter": adapter,
        "keys": keys + [client.api_key], "protected": protected, "artifact": artifact,
        "expected_wire_sha256": evidence[-1]["wire_sha256"], "expected_wire_bytes": evidence[-1]["wire_bytes"],
        "evidence": evidence, "checkpoint": checkpoint, "protected_score_count": FIRST_SCORES, "media_identity": media_identity}


@contextmanager
def narrow_eligibility(case):
    original = probe.check_eligible
    probe.check_eligible = lambda row: check_final(row, case)
    try:
        yield
    finally:
        probe.check_eligible = original


def execute_once(prepared, output, opener=probe.open_once):
    check_final(prepared["failed"], prepared["case"])
    wire = json.dumps(prepared["adapter"].build_payload(prepared["client"], prepared["messages"]), allow_nan=False).encode()
    if hashlib.sha256(wire).hexdigest() != prepared["expected_wire_sha256"] or len(wire) != prepared["expected_wire_bytes"]:
        raise ValueError("actual request body changed before the one authorized HTTP dispatch")
    with narrow_eligibility(prepared["case"]):
        return probe.execute_once(prepared, output, opener)


def run(stage, specs_path, video, repo, related_runs, output_dir, opener=probe.open_once):
    if stage not in ("preflight", "probe"):
        raise ValueError("only preflight or single-request probe is supported")
    case = load_case(specs_path, video)
    source, repo, output = Path(case["source_run"]).resolve(strict=True), Path(repo).resolve(strict=True), Path(output_dir).resolve()
    if any(output == root or root in output.parents for root in (source, repo)):
        raise ValueError("probe output must be outside the source and frozen checkout")
    registry = source.parents[1] / "recovery_claims/noevent_audio429"
    if any(path.is_symlink() for path in (registry, *registry.parents)):
        raise ValueError("single-request guard may not use symlink aliases")
    registry.mkdir(parents=True, exist_ok=True, mode=0o700)
    serial_path = registry / "serial.lock"
    if serial_path.exists() or serial_path.is_symlink():
        base.regular(serial_path)
    with serial_path.open("a") as serial:
        fcntl.flock(serial, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with base.stopped_parent(source):
            prepared = prepare(specs_path, video, repo, related_runs)
            if stage == "probe":
                output = probe.claim_output(source, video, output)
                claim = registry / (base.digest(str(source))[:20] + "-" + video + "-" + case["window"] + ".json")
                probe.write_private(claim, {"source_run": str(source), "video_id": video, "window_id": case["window"],
                    "output_dir": str(output), "maximum_http_requests": 1, "claimed_unix": time.time(),
                    "request_hash": prepared["failed"]["request_hash"], "driver_sha256": base.sha(__file__), "specs_sha256": SPECS_SHA})
            else:
                if output.exists() or any(path.is_symlink() for path in (output, *output.parents)):
                    raise ValueError("preflight output must be a new ordinary directory")
                output.mkdir(parents=True, mode=0o700)
            receipt = {"schema_version": 1, "status": "preflight_passed", "no_api_calls": True,
                "video_id": video, "window_id": case["window"], "source_run": str(source),
                "source_configuration_sha256": case["configuration_sha256"], "request_hash": prepared["failed"]["request_hash"],
                "completed_windows": case["completed_windows"], "total_windows": case["total_windows"],
                "original_attempts": 3, "maximum_additional_http_requests": 1, "visual_max_tokens": case["visual_max_tokens"],
                "protected_score_count": FIRST_SCORES, "protected_files": prepared["protected"], "evidence": prepared["evidence"],
                "artifact": prepared["artifact"], "resource_exhaustion_cause": "unspecified; no rate/billing inference"}
            probe.write_private(output / "preflight.json", receipt)
            return execute_once(prepared, output, opener) if stage == "probe" else receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("stage", choices=("preflight", "probe"))
    parser.add_argument("--video-id", choices=tuple(CASES), required=True)
    for name in ("specs", "repo", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--related-run", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        value = run(args.stage, args.specs, args.video_id, args.repo, args.related_run, args.output_dir)
    except Exception as exc:
        print(json.dumps({"status": "not_dispatched_or_needs_audit", "error_type": type(exc).__name__}))
        return 2
    print(json.dumps({key: value[key] for key in ("status", "video_id", "window_id", "request_hash", "protected_score_count", "http_status") if key in value}))
    return 0 if value.get("status") in ("preflight_passed", "validated") else 1


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
