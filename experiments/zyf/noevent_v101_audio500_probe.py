"""One additional, exact V101/W64 audio request after the recorded HTTP 500.

The old three-attempt budget is exhausted and immutable. This separate probe
does not import audio, advance a window, run a builder, answer, or judge. It can
only dispatch the third request after reconstructing both original JSON repairs
and matching its recorded message hash and serialized request-body hash.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from experiments.zyf import noevent_continuation as base
from experiments.zyf import noevent_retry_probe as probe
from experiments.zyf.noevent_length_continuation import score_snapshot, terminal_video


VIDEO = "G2_V000101"
WINDOW = "W00064"
SOURCE_NAME = "noevent_runtime24_continuation_20260923"
FINAL_REQUEST = "c011c4308a00924f88b9e2159119b550b582de8b546c9715e2e95c97e0f4b543"
CORE_CONTINUATION_SHA = "1e9776267db1f38330a3bca2c66fb031ce6ec3b925abb11531a5c31748052192"
ORIGINAL_WIRE_SHA = "df69419b956457f08134189719eba9576d8669d7543789f061de326cc1cfcbcd"
ORIGINAL_WIRE_BYTES = 1030542
ORIGINAL_DIAGNOSTIC_SHAS = {
    "e88cc564428b22ec9af000877b60a72f6e441be37abb965f5c22598ce4bd78f2":
        "c96b9ca38c7bbbbf7332ae74d01dbbb0259a17ea16fb8b3c1da7d2eeae811443",
    "f0df2669616589f4bc7edf7a733e600b305accd6c3abdaa01375885423ceea9c":
        "19baafb456b853f5c1d0066360c8fc7bb97d18db86bfbdf7ac9e1cc822693fe5",
    FINAL_REQUEST: "4d5442664a2be385ab88ddef1f704485930a05d699359e697021e1568ebab748",
}


def check_eligible(row):
    if (row.get("purpose") != "audio_observer:" + WINDOW or row.get("status") != "error"
            or row.get("attempt") != 3 or row.get("failure_stage") != "generate"
            or row.get("http_status") != 500 or row.get("will_retry") is not False
            or row.get("retryable") is not True or row.get("retry_reason") != "temporary_http_error"
            or row.get("request_hash") != FINAL_REQUEST or probe.is_refusal(row.get("service_error_code"))):
        raise ValueError("only the fixed third V101/W64 temporary HTTP500 is eligible")


def replay_repairs(initial_messages, rows, audio_folder, common, parse_json, validate, key, raw_content=None):
    """Replay the unchanged LoggedClient repair rule using intact saved text."""
    if len(rows) != 3 or [row.get("attempt") for row in rows] != [1, 2, 3]:
        raise ValueError("the original audio request must have exactly three recorded attempts")
    check_eligible(rows[-1])
    messages = list(initial_messages)
    initial_hash = common.fingerprint(messages)
    protected = {}
    for index, row in enumerate(rows):
        if (row.get("purpose") != "audio_observer:" + WINDOW or row.get("initial_request_hash") != initial_hash
                or row.get("request_hash") != common.fingerprint(messages)):
            raise ValueError("original repair-chain request hash differs")
        if index == 2:
            break
        if (row.get("status") != "error" or row.get("failure_stage") != "validate"
                or row.get("error_type") not in ("ValueError", "JSONDecodeError")
                or row.get("retry_reason") != "schema_repair" or row.get("retryable") is not True
                or row.get("will_retry") is not True or probe.is_refusal(row.get("service_error_code"))):
            raise ValueError("the first two attempts must be explicit schema repairs")
        relative = row.get("diagnostic_path")
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("missing or unsafe original failed-content diagnostic")
        path = base.regular(audio_folder / relative)
        if base.sha(path) != row.get("diagnostic_sha256"):
            raise ValueError("original failed-content diagnostic hash changed")
        value = base.read(path)
        content = value.get("content")
        if (value.get("kind") != "failed_model_content" or value.get("content_redacted") is not False
                or not isinstance(content, str) or not content.strip()
                or hashlib.sha256(content.encode()).hexdigest() != value.get("response_sha256")
                or value.get("response_sha256") != row.get("response_sha256")):
            raise ValueError("repair input is missing, altered, or redacted")
        metadata = value.get("metadata") or {}
        for field in ("purpose", "attempt", "request_hash", "initial_request_hash", "error_type", "failure_stage", "retry_reason"):
            if metadata.get(field) != row.get(field):
                raise ValueError("saved repair metadata differs from its exact ledger row")
        protected[str(path)] = base.sha(path)
        if raw_content is not None:
            original_content, source_hashes = raw_content(row, messages)
            if original_content != content:
                raise ValueError("saved repair text differs from its intact original HTTP200 envelope")
            protected.update(source_hashes)
        try:
            validate(parse_json(content))
        except ValueError as exc:
            if type(exc).__name__ != row["error_type"]:
                raise ValueError("original validation exception type cannot be reproduced") from None
            error_text = str(exc).replace(key, "[REDACTED_API_KEY]") if key else str(exc)
            if error_text[:1000] != row.get("validation_error"):
                raise ValueError("original full validation error cannot be reproduced") from None
        else:
            raise ValueError("formerly failed content now validates; original repair is not reproducible")
        safe_content = content.replace(key, "[REDACTED_API_KEY]") if key else content
        messages += [{"role": "assistant", "content": safe_content},
                     {"role": "user", "content": "Repair the JSON to satisfy the schema. Validation error: " + error_text}]
    if common.fingerprint(messages) != FINAL_REQUEST:
        raise ValueError("final reconstructed audio request differs from the fixed original")
    return messages, protected


def wrapper_evidence(source, client, messages, adapter, row, http_status, outcome):
    wire = json.dumps(adapter.build_payload(client, messages), allow_nan=False).encode()
    wire_hash = hashlib.sha256(wire).hexdigest()
    matches = []
    for path in sorted((source / "private_diagnostics/build" / VIDEO).glob("*.json")):
        if path.name.endswith(".intent.json"):
            continue
        value = base.read(base.regular(path))
        if value.get("request_hash") != row["request_hash"]:
            continue
        if base.sha(path) != ORIGINAL_DIAGNOSTIC_SHAS.get(row["request_hash"]):
            raise ValueError("original private wrapper diagnostic differs from the inspected source")
        if (value.get("model_configuration") != client.configuration() or value.get("outcome") != outcome
                or (outcome == "error" and value.get("error_type") != row.get("error_type"))
                or value.get("serialized_requests") != [{"payload_sha256": wire_hash, "bytes": len(wire)}]):
            raise ValueError("recorded client configuration or actual wire payload differs")
        responses = value.get("responses")
        if not isinstance(responses, list) or len(responses) != 1 or responses[0].get("http_status") != http_status:
            raise ValueError("request must have exactly one recorded response with the expected HTTP status")
        response = responses[0]
        body = response.get("body_text")
        if not isinstance(body, str) or hashlib.sha256(body.encode()).hexdigest() != response.get("body_sha256"):
            raise ValueError("original response body is unavailable or redacted")
        raw = json.loads(body)
        if probe.refused(raw):
            raise ValueError("a provider refusal cannot be repaired or retried")
        matches.append({"diagnostic_path": str(path), "diagnostic_sha256": base.sha(path),
                        "response_sha256": response["body_sha256"], "wire_sha256": wire_hash,
                        "http_status": http_status, "raw": raw})
    if len(matches) != 1:
        raise ValueError("exactly one matching original private wrapper diagnostic is required")
    return matches[0]


def repair_content(source, client, messages, adapter, row):
    evidence = wrapper_evidence(source, client, messages, adapter, row, 200, "ok")
    raw = evidence["raw"]
    if not isinstance(raw, dict) or raw.get("error"):
        raise ValueError("original schema failure did not have a successful model response")
    content = adapter.parse_response(client, raw).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("original failed response text is unavailable")
    if row.get("response_sha256") and hashlib.sha256(content.encode()).hexdigest() != row["response_sha256"]:
        raise ValueError("parsed original response differs from the core failed-content hash")
    return content, {evidence["diagnostic_path"]: evidence["diagnostic_sha256"]}


def final_response_evidence(source, client, messages, adapter, last):
    evidence = wrapper_evidence(source, client, messages, adapter, last, 500, "error")
    wire = json.dumps(adapter.build_payload(client, messages), allow_nan=False).encode()
    if evidence["wire_sha256"] != ORIGINAL_WIRE_SHA or len(wire) != ORIGINAL_WIRE_BYTES:
        raise ValueError("final payload differs from the independently inspected original wire bytes")
    raw = evidence.pop("raw")
    errors = raw.get("error") if isinstance(raw, dict) else None
    if (not isinstance(errors, list) or len(errors) != 1 or not isinstance(errors[0], dict)
            or set(errors[0]) != {"error"} or not isinstance(errors[0]["error"], dict)):
        raise ValueError("original response does not have the inspected nested internal-server-error shape")
    detail = errors[0]["error"]
    if (detail.get("code") != 500 or detail.get("status") != "INTERNAL"
            or detail.get("message") != "Internal error encountered."):
        raise ValueError("original response does not establish the known internal server error")
    return {**evidence, "error_code": 500, "error_status": "INTERNAL"}


def prepare(source_run, repo, related_runs):
    source, repo = Path(source_run).resolve(strict=True), Path(repo).resolve(strict=True)
    if source.name != SOURCE_NAME:
        raise ValueError("this bounded probe only supports the fixed main24 source run")
    if base.sha(base.regular(base.__file__)) != CORE_CONTINUATION_SHA:
        raise ValueError("the original continuation helper changed")
    config = base.read(base.regular(source / "configuration.json"))
    base.verify(source, config)
    if config["core_repo"] != str(repo) or config["profile"]["visual"]["max_tokens"] != 8192:
        raise ValueError("original source core or visual configuration differs")
    states = terminal_video(source, VIDEO)
    task_path = source / "tasks/build" / ("noevent-" + VIDEO) / "task.json"
    task = base.read(task_path)
    command = base.stage_command(source, config, "build", VIDEO)
    if (task.get("command") != command or task.get("cwd") != str(repo)
            or task.get("signature") != base.digest({"command": command, "cwd": str(repo)})):
        raise ValueError("source build argv differs from the frozen diagnostic wrapper command")
    modules = probe.bind_repo(repo)
    runner, common, audio_module = (modules[name] for name in
        ("methods.longemo.noevent_runner", "methods.longemo.common", "methods.longemo.audio"))
    args = runner.parser().parse_args(command[command.index("--") + 1:])
    if args.tries != 3 or args.workers != 1:
        raise ValueError("original build attempt/worker budget differs")
    baseline = Path(config["parent_run"])
    old = base.read(base.regular(baseline / "configuration.json"))
    checkpoint = base.checkpoint_receipt(source, repo, old, VIDEO)
    if checkpoint["completed_windows"] != 63:
        raise ValueError("V101 must remain at exactly 63 completed windows")
    folder = Path(checkpoint["source"])
    manifest, memory = base.read(folder / "manifest.json"), base.read(folder / "memory.json")
    if memory.get("complete") or memory["completed_windows"] != [f"W{i:05d}" for i in range(1, 64)]:
        raise ValueError("V101 checkpoint is not the original incomplete prefix")
    if (folder / "audio" / (WINDOW + ".json")).exists():
        raise ValueError("W64 audio already exists; do not probe it again")
    questions = base.read(source / "questions" / (VIDEO + ".json"))
    for parent in (baseline, source):
        base.ensure_no_backend_attempts(parent, questions)
    all_questions = base.read(baseline / "questions.json")
    related = sorted({str(Path(value).absolute()) for value in (source, *related_runs)})
    if str(baseline) in related:
        raise ValueError("the original baseline must not be passed twice")
    scores = score_snapshot(related, all_questions, {q["question_id"] for q in questions}, config["protected_scores"])
    visual, client = runner.client_for(args), runner.audio_client_for(args)
    if (visual.configuration() != checkpoint["parent_model"] or not client
            or client.configuration() != checkpoint["parent_audio_observer"]
            or client.base_url.rstrip("/") != probe.MATRIX or client.model != "gemini-3.8-flash"
            or client.max_tokens != 4096 or client.timeout != 180 or client.options != {"reasoning_effort": "low"}):
        raise ValueError("original audio/visual model configuration differs")
    video, subtitle = Path(args.videos_dir) / (VIDEO + ".mp4"), Path(args.subtitles_dir) / (VIDEO + ".json")
    media_identity = base.verified_media(old, [VIDEO])
    info = runner.probe(video)
    if math.ceil(info["duration"] / args.window_seconds) != 75 or memory["duration"] != info["duration"]:
        raise ValueError("V101 no longer has the frozen 75-window media")
    core = [63 * args.window_seconds, min(64 * args.window_seconds, info["duration"])]
    interval = [max(0, core[0] - args.padding), min(info["duration"], core[1] + args.padding)]
    media, metadata = runner.window_input(video, *interval, subtitle_rows=runner.subtitles(subtitle), **runner._media_options(args))
    audio = [part for part in media if part["type"] == "input_audio"]
    if not audio:
        raise ValueError("original audio could not be reconstructed")
    initial = [{"role": "system", "content": audio_module.AUDIO_PROMPT}, {"role": "user", "content": [
        {"type": "text", "text": json.dumps({"clip_start": interval[0], "clip_end": interval[1]})}] + audio}]

    def validate(value):
        if not isinstance(value, dict) or not isinstance(value.get("observations"), list):
            raise ValueError("audio observations must be a list")
        for observation in value["observations"]:
            audio_module.span(observation.get("span"), interval)
            for field in ("voice", "cue"):
                audio_module.text(observation.get(field), field)

    ledger = folder / "audio/calls.jsonl"
    audio_rows = base.records(ledger)
    rows = [row for row in audio_rows if row.get("purpose") == "audio_observer:" + WINDOW]
    if not rows or audio_rows[-1] != rows[-1]:
        raise ValueError("W64 is not the terminal audio failure")
    visual_rows = base.records(folder / "calls.jsonl")
    if any(row.get("purpose") == "window_perception:" + VIDEO + ":" + WINDOW for row in visual_rows):
        raise ValueError("W64 already has a visual attempt")
    parse_json = modules["evaluation.inference.prompts"].json_object
    adapter = modules["evaluation.inference.adapters"].API_ADAPTERS[client.api_format]
    messages, repairs = replay_repairs(initial, rows, folder / "audio", common, parse_json, validate, client.api_key,
        raw_content=lambda row, messages: repair_content(source, client, messages, adapter, row))
    evidence = final_response_evidence(source, client, messages, adapter, rows[-1])
    protected = {str(folder / name): expected for name, expected in checkpoint["files"].items()}
    protected.update(states)
    protected.update(repairs)
    for path in (source / "configuration.json", source / "prepared.json", source / "questions.json",
                 source / "questions" / (VIDEO + ".json"), source / "process.json", Path(evidence["diagnostic_path"]),
                 video, subtitle, Path(__file__), Path(probe.__file__)):
        protected[str(path)] = base.sha(base.regular(path))
    for item in config["protected_scores"].values():
        protected[item["path"]] = item["sha256"]
        protected[item["envelope"]["source"]] = item["official_source_sha256"]
    for item in scores.values():
        protected[item["path"]] = item["sha256"]
        protected[item["source"]] = item["source_sha256"]
    credentials = base.read(base.regular(args.credential_file))
    keys = [value for name, value in credentials.items() if any(term in name.lower() for term in ("key", "token", "secret", "password"))
            and isinstance(value, str)]
    artifact = {"video_id": VIDEO, "window_id": WINDOW, "stage": "audio", "context": runner.perception_context(
        memory, window_id=WINDOW, core=core, media=interval), "sampling": metadata, "core": core, "interval": interval,
        "audio_input_fingerprint": common.fingerprint({"messages": initial, "model": client.configuration()}),
        "model_configuration": client.configuration(), "parent_memory_sha256": base.sha(folder / "memory.json"),
        "parent_manifest_sha256": base.sha(folder / "manifest.json"), "source_run": str(source),
        "original_attempts": 3, "new_additional_requests": 1, "probe_is_not_a_completed_window": True}
    return {"client": client, "messages": messages, "failed": rows[-1], "fingerprint": common.fingerprint,
            "validate": validate, "parse_json": parse_json, "adapter": adapter, "keys": keys + [client.api_key],
            "protected": protected, "artifact": artifact, "expected_wire_sha256": evidence["wire_sha256"],
            "evidence": evidence, "checkpoint": checkpoint, "protected_score_count": len(config["protected_scores"]) + len(scores),
            "media_identity": media_identity, "source_configuration_sha256": base.sha(source / "configuration.json")}


@contextmanager
def narrow_eligibility():
    # This separate single-threaded diagnostic process reuses only the proven
    # one-HTTP sender. The original helper file and all worker processes stay unchanged.
    original = probe.check_eligible
    probe.check_eligible = check_eligible
    try:
        yield
    finally:
        probe.check_eligible = original


def execute_once(prepared, output, opener=probe.open_once):
    check_eligible(prepared["failed"])
    wire = json.dumps(prepared["adapter"].build_payload(prepared["client"], prepared["messages"]), allow_nan=False).encode()
    if hashlib.sha256(wire).hexdigest() != prepared["expected_wire_sha256"]:
        raise ValueError("actual wire payload changed before the single diagnostic dispatch")
    with narrow_eligibility():
        return probe.execute_once(prepared, output, opener)


def run(stage, source_run, repo, related_runs, output_dir, opener=probe.open_once):
    source, repo = Path(source_run).resolve(strict=True), Path(repo).resolve(strict=True)
    output = Path(output_dir).resolve()
    if any(output == root or root in output.parents for root in (source, repo)):
        raise ValueError("diagnostic output must be outside the source run and frozen core")
    if stage not in ("preflight", "probe"):
        raise ValueError("only a preflight or single probe is supported")
    with base.stopped_parent(source):
        prepared = prepare(source, repo, related_runs)
        if stage == "probe":
            output = probe.claim_output(source, VIDEO, output)
            registry = source.parents[1] / "recovery_claims/noevent_audio500"
            registry.mkdir(parents=True, exist_ok=True, mode=0o700)
            if any(path.is_symlink() for path in (registry, *registry.parents)):
                raise ValueError("global single-request guard may not use symlink aliases")
            claim = registry / (base.digest(str(source))[:20] + "-" + VIDEO + "-" + WINDOW + ".json")
            probe.write_private(claim, {"source_run": str(source), "output_dir": str(output), "request_hash": FINAL_REQUEST,
                "maximum_http_requests": 1, "claimed_unix": time.time(), "driver_sha256": base.sha(__file__),
                "source_configuration_sha256": prepared["source_configuration_sha256"]})
        else:
            if output.exists() or any(path.is_symlink() for path in (output, *output.parents)):
                raise ValueError("preflight output must be a new ordinary directory")
            output.mkdir(parents=True, mode=0o700)
        receipt = {"schema_version": 1, "status": "preflight_passed", "no_api_calls": True,
                   "video_id": VIDEO, "window_id": WINDOW, "request_hash": FINAL_REQUEST,
                   "completed_windows": 63, "total_windows": 75, "original_attempts": 3, "maximum_additional_http_requests": 1,
                   "protected_score_count": prepared["protected_score_count"], "evidence": prepared["evidence"],
                   "protected_files": prepared["protected"], "artifact": prepared["artifact"]}
        probe.write_private(output / "preflight.json", receipt)
        return execute_once(prepared, output, opener) if stage == "probe" else receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("stage", choices=("preflight", "probe"))
    for name in ("source-run", "repo", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--related-run", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        value = run(args.stage, args.source_run, args.repo, args.related_run, args.output_dir)
    except Exception as exc:
        print(json.dumps({"status": "not_dispatched_or_needs_audit", "error_type": type(exc).__name__}))
        return 2
    print(json.dumps({key: value[key] for key in ("status", "video_id", "window_id", "request_hash", "protected_score_count", "http_status") if key in value}))
    return 0 if value.get("status") in ("preflight_passed", "validated") else 1


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
