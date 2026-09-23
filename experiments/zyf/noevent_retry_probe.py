"""One explicitly authorized diagnostic request for an unknown noevent failure.

This standalone tool never changes the parent run, submits an audio request while
probing vision, retries a request, or publishes a memory/answer/score.  Its output
directory and sibling guard are exclusive, including when a previous attempt's
outcome is unknown.  Run with ``python -B`` to keep the frozen checkout read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from urllib import error, request


MATRIX = "https://matrixllm.alipay.com/v1"
SAFE_TOKEN = re.compile(r"[A-Za-z0-9_./:@+-]{1,160}\Z")
REFUSALS = {"contentfilter", "contentpolicyviolation", "responsibleaipolicyviolation",
            "safety", "blockedprompt", "prohibitedcontent", "blockedinput"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("required input must be an existing ordinary file")
    return json.loads(path.read_text())


def write_private(path, value):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def safe(value):
    return value if isinstance(value, str) and SAFE_TOKEN.fullmatch(value) else None


def is_refusal(code):
    return re.sub(r"[^a-z0-9]", "", str(code or "").lower()) in REFUSALS


def redact(text, keys):
    for key in sorted({key for key in keys if isinstance(key, str) and key}, key=len, reverse=True):
        text = text.replace(key, "[REDACTED_API_KEY]")
    text = re.sub(r"(?i)Bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", text)
    return re.sub(r"\b(?:sk-|hf_|ghp_)[A-Za-z0-9_-]{12,}", "[REDACTED_CREDENTIAL]", text)


def claim_output(parent_run, video_id, output_dir):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", video_id):
        raise ValueError("invalid video identifier")
    output = Path(output_dir).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("diagnostic output already exists")
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if output.parent.is_symlink():
        raise ValueError("diagnostic parent must not be a symlink")
    run_id = hashlib.sha256(str(Path(parent_run).resolve()).encode()).hexdigest()[:20]
    guard = output.parent / (".noevent-probe-" + run_id + "-" + video_id + ".json")
    write_private(guard, {"schema_version": 1, "video_id": video_id,
                          "parent_run": str(Path(parent_run).resolve()),
                          "output_dir": str(output), "claimed_unix": time.time(), "pid": os.getpid(),
                          "maximum_http_requests": 1})
    output.mkdir(mode=0o700)
    return output


def bind_repo(repo):
    repo = Path(repo).resolve(strict=True)
    for name in ("methods", "evaluation"):
        module = sys.modules.get(name)
        if module is not None and getattr(module, "__file__", None):
            if repo not in Path(module.__file__).resolve().parents:
                raise ValueError("a different core checkout was already imported")
    sys.path.insert(0, str(repo))
    core = {name: importlib.import_module(name) for name in (
        "methods.longemo.noevent_runner", "methods.longemo.common", "methods.longemo.audio",
        "evaluation.inference.adapters", "evaluation.inference.prompts")}
    for module in core.values():
        if repo not in Path(module.__file__).resolve().parents:
            raise ValueError("core module did not resolve to the frozen checkout")
    return core


def parse_original_command(task, repo, runner):
    if task.get("status") != "error" or Path(task.get("cwd", "")).resolve() != Path(repo).resolve():
        raise ValueError("parent task is not a stopped failed task in the selected checkout")
    child = task.get("child") or {}
    if child.get("pid"):
        if type(child["pid"]) is not int or child["pid"] < 1 or not str(child.get("start_ticks", "")).isdigit():
            raise ValueError("parent child identity is ambiguous")
        stat = Path("/proc") / str(child["pid"]) / "stat"
        if stat.exists():
            fields = stat.read_text().rsplit(")", 1)[1].split()
            if fields[19] == str(child["start_ticks"]) and fields[0] != "Z":
                raise ValueError("parent build child is still alive")
    command = task.get("command")
    if not isinstance(command, list) or any(not isinstance(x, str) for x in command):
        raise ValueError("parent command must be an argv list")
    if "--api-key" in command or any(x.startswith("--api-key=") for x in command):
        raise ValueError("inline API credentials are not supported")
    index = command.index("-m")
    if command[index + 1:index + 3] != ["methods.longemo.noevent_runner", "build"]:
        raise ValueError("parent task is not the noevent builder")
    return runner.parser().parse_args(command[index + 2:])


def ledger_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    if path.is_symlink() or not path.is_file():
        raise ValueError("ledger must be an ordinary file")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def check_eligible(row):
    if is_refusal(row.get("service_error_code")):
        raise ValueError("an explicit refusal must not be probed")
    if (row.get("status") != "error" or row.get("failure_stage") != "generate" or
            row.get("attempt") != 1 or row.get("will_retry") is not False):
        raise ValueError("only a stopped first generate attempt is eligible")
    status = row.get("http_status")
    unknown_runtime = row.get("error_type") == "RuntimeError" and status is None and not row.get("service_error_code")
    unknown_428 = status == 428 and row.get("service_error_code") in (None, "", "428")
    if not (unknown_runtime or unknown_428):
        raise ValueError("only unknown RuntimeError or unknown HTTP428 is eligible")
    if not re.fullmatch(r"[a-f0-9]{64}", str(row.get("request_hash", ""))):
        raise ValueError("parent request hash is missing")
    if row.get("initial_request_hash", row["request_hash"]) != row["request_hash"]:
        raise ValueError("a repaired or changed request is not eligible")


def select_failure(folder, video_id, window_id):
    entries = []
    for stage, path in (("visual", folder / "calls.jsonl"), ("audio", folder / "audio/calls.jsonl")):
        rows = ledger_rows(path)
        if rows:
            row = rows[-1]
            if type(row.get("time_unix")) not in (int, float) or not math.isfinite(row["time_unix"]):
                raise ValueError("ledger has no finite request timestamp")
            entries.append((row["time_unix"], stage, path, row))
    if not entries:
        raise ValueError("parent failure ledger missing")
    _, stage, path, row = max(entries, key=lambda x: x[0])
    expected = "audio_observer:" + window_id if stage == "audio" else "window_perception:" + video_id + ":" + window_id
    if row.get("purpose") != expected:
        raise ValueError("terminal failure does not match the first incomplete window")
    relevant = [entry for entry in ledger_rows(path) if entry.get("purpose") == expected]
    if len(relevant) != 1:
        raise ValueError("the diagnostic is restricted to one prior attempt of this request")
    check_eligible(row)
    return stage, path, row


def prepare(parent_run, video_id, repo):
    parent = Path(parent_run).resolve(strict=True)
    repo = Path(repo).resolve(strict=True)
    modules = bind_repo(repo)
    runner = modules["methods.longemo.noevent_runner"]
    common = modules["methods.longemo.common"]
    audio_module = modules["methods.longemo.audio"]
    config_path = parent / "configuration.json"
    task_path = parent / "tasks/build" / ("noevent-" + video_id) / "task.json"
    config, task = read(config_path), read(task_path)
    if Path(config["repos"]["noevent"]).resolve() != repo or config["source_hashes"]["noevent"] != common.code_hash():
        raise ValueError("frozen noevent source differs")
    args = parse_original_command(task, repo, runner)
    folder = Path(args.output_dir).resolve() / video_id
    if parent not in folder.parents:
        raise ValueError("memory must belong to the parent run")
    manifest_path, memory_path = folder / "manifest.json", folder / "memory.json"
    manifest, memory = read(manifest_path), read(memory_path)
    if manifest.get("fingerprint") != common.fingerprint(manifest["configuration"]):
        raise ValueError("parent manifest fingerprint mismatch")
    if memory.get("build_fingerprint") != manifest["fingerprint"] or memory.get("representation") != "window_records" or memory.get("complete"):
        raise ValueError("parent memory is not the matching incomplete noevent checkpoint")
    visual = runner.client_for(args)
    audio = runner.audio_client_for(args)
    if visual.base_url.rstrip("/") != MATRIX or not audio or audio.base_url.rstrip("/") != MATRIX:
        raise ValueError("this diagnostic only supports the original Matrix configuration")
    if visual.model != config.get("perception_model") or audio.model != config.get("audio_model"):
        raise ValueError("parent perception model differs")
    video = Path(args.videos_dir) / (video_id + ".mp4")
    subtitle = Path(args.subtitles_dir) / (video_id + ".json")
    info = runner.probe(video)
    expected = dict(manifest["configuration"])
    expected.update(representation="window_records", video_sha256=sha(video), subtitles_sha256=sha(subtitle),
                    model=visual.configuration(), audio_observer=audio.configuration(),
                    window_seconds=args.window_seconds, padding=args.padding, media=runner._media_options(args),
                    code_hash=common.code_hash())
    if expected != manifest["configuration"] or memory.get("video_sha256") != expected["video_sha256"]:
        raise ValueError("original model, media, or builder configuration changed")
    frozen_media = config["media"][video_id]
    actual_media = {"video_sha256": expected["video_sha256"], "subtitles_sha256": expected["subtitles_sha256"],
                    "video_bytes": video.stat().st_size, "subtitles_bytes": subtitle.stat().st_size}
    if (any(frozen_media.get(key) != actual_media[key] for key in ("video_sha256", "subtitles_sha256")) or
            any(key in frozen_media and frozen_media[key] != actual_media[key] for key in ("video_bytes", "subtitles_bytes"))):
        raise ValueError("media differs from the parent cohort")
    count = math.ceil(info["duration"] / args.window_seconds)
    completed = memory["completed_windows"]
    if completed != [f"W{i+1:05d}" for i in range(len(completed))] or len(completed) >= count or memory["duration"] != info["duration"]:
        raise ValueError("checkpoint windows/duration are not a contiguous incomplete prefix")
    i, window_id = len(completed), f"W{len(completed)+1:05d}"
    core = [i * args.window_seconds, min((i + 1) * args.window_seconds, info["duration"])]
    interval = [max(0, core[0] - args.padding), min(info["duration"], core[1] + args.padding)]
    stage, ledger, failed = select_failure(folder, video_id, window_id)
    protected = [config_path, task_path, manifest_path, memory_path, ledger, subtitle]
    before = {str(path): sha(path) for path in protected}
    media, metadata = runner.window_input(video, *interval, subtitle_rows=runner.subtitles(subtitle), **runner._media_options(args))
    audio_parts = [part for part in media if part["type"] == "input_audio"]
    if not audio_parts:
        raise ValueError("original audio media is absent")
    audio_messages = [{"role": "system", "content": audio_module.AUDIO_PROMPT},
                      {"role": "user", "content": [{"type": "text", "text": json.dumps({"clip_start": interval[0], "clip_end": interval[1]})}] + audio_parts}]
    audio_signature = common.fingerprint({"messages": audio_messages, "model": audio.configuration()})
    pending_audio = folder / "audio" / (window_id + ".json")

    def validate_audio(value):
        if not isinstance(value, dict) or not isinstance(value.get("observations"), list):
            raise ValueError("audio observations must be a list")
        for observation in value["observations"]:
            audio_module.span(observation.get("span"), interval)
            for field in ("voice", "cue"):
                audio_module.text(observation.get(field), field)

    context = runner.perception_context(memory, window_id=window_id, core=core, media=interval)
    if stage == "audio":
        if pending_audio.exists():
            raise ValueError("failed audio unexpectedly already has a committed cache")
        client, messages, validate = audio, audio_messages, validate_audio
    else:
        cached = read(pending_audio)
        if cached.get("input_fingerprint") != audio_signature or cached.get("model") != audio.configuration():
            raise ValueError("pending audio input/model fingerprint mismatch")
        validate_audio(cached["result"])
        before[str(pending_audio)] = sha(pending_audio)
        replacement = {"type": "text", "text": "Timestamped audio observations from an independent audio model; "
                       "these may contain errors. Match voice identity cautiously using the frames and dialogue. " + json.dumps(cached["result"], ensure_ascii=False)}
        media = [part for part in media if part["type"] != "input_audio"] + [replacement]
        metadata.update(audio_representation="derived_timestamped_cues", audio_observer={
            "model": audio.model, "input_fingerprint": audio_signature, "source_sha256": sha(pending_audio)})
        messages = [{"role": "system", "content": runner.PERCEPTION_NOEVENT},
                    {"role": "user", "content": [{"type": "text", "text": json.dumps(context, ensure_ascii=False)}] + media}]
        client = visual

        def validate(value):
            runner.apply_window(memory, value, window_id=window_id, core=core, media=interval, metadata=metadata)

    if common.fingerprint(messages) != failed["request_hash"]:
        raise ValueError("reconstructed request hash differs; no HTTP request allowed")
    credentials = read(args.credential_file)
    keys = [value for key, value in credentials.items() if "KEY" in key and isinstance(value, str)]
    return {"client": client, "messages": messages, "failed": failed, "fingerprint": common.fingerprint,
            "validate": validate, "parse_json": modules["evaluation.inference.prompts"].json_object,
            "adapter": modules["evaluation.inference.adapters"].API_ADAPTERS[client.api_format],
            "keys": keys + [client.api_key], "protected": before,
            "artifact": {"video_id": video_id, "window_id": window_id, "stage": stage, "context": context,
                         "sampling": metadata, "core": core, "interval": interval,
                         "audio_input_fingerprint": audio_signature, "model_configuration": client.configuration(),
                         "parent_memory_sha256": before[str(memory_path)], "parent_manifest_sha256": before[str(manifest_path)]}}


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise error.HTTPError(req.full_url, code, "redirect refused", headers, fp)


def open_once(req, timeout):
    return request.build_opener(NoRedirect).open(req, timeout=timeout)


def response_metadata(raw):
    result = {"safe_code": None, "response_model": None, "finish_reason": None, "choices_count": None}
    if not isinstance(raw, dict):
        return result
    detail = raw.get("error")
    if isinstance(detail, dict):
        result["safe_code"] = safe(detail.get("code", detail.get("status")))
    result["response_model"] = safe(raw.get("modelVersion", raw.get("model")))
    choices = raw.get("choices") if "choices" in raw else raw.get("candidates")
    if isinstance(choices, list):
        result["choices_count"] = len(choices)
        if choices and isinstance(choices[0], dict):
            result["finish_reason"] = safe(choices[0].get("finish_reason", choices[0].get("finishReason")))
    return result


def refused(raw):
    if not isinstance(raw, dict):
        return False
    meta = response_metadata(raw)
    feedback = raw.get("promptFeedback")
    if (is_refusal(meta["safe_code"]) or
            (isinstance(feedback, dict) and feedback.get("blockReason"))):
        return True
    for choice in raw.get("choices", []) or []:
        if isinstance(choice, dict) and (choice.get("finish_reason") == "content_filter" or
                isinstance(choice.get("message"), dict) and choice["message"].get("refusal")):
            return True
    return meta["finish_reason"] in {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "IMAGE_SAFETY"}


def execute_once(prepared, output, opener=open_once):
    check_eligible(prepared["failed"])
    fingerprint = prepared["fingerprint"]
    if fingerprint(prepared["messages"]) != prepared["failed"]["request_hash"]:
        raise ValueError("request hash changed before dispatch")
    if any(sha(path) != expected for path, expected in prepared["protected"].items()):
        raise ValueError("parent files changed before dispatch")
    client, adapter = prepared["client"], prepared["adapter"]
    wire = json.dumps(adapter.build_payload(client, prepared["messages"]), allow_nan=False).encode()
    req = request.Request(adapter.endpoint(client), data=wire, headers=adapter.headers(client), method="POST")
    summary = {"schema_version": 1, "kind": "independent_one_request_diagnostic", "official_result": False,
               "video_id": prepared["artifact"]["video_id"], "window_id": prepared["artifact"]["window_id"],
               "stage": prepared["artifact"]["stage"], "model": safe(client.model),
               "request_hash": prepared["failed"]["request_hash"], "maximum_http_requests": 1,
               "parse_status": "not_attempted", "validate_status": "not_attempted", "http_status": None}
    write_private(output / "request_started.json", {**summary, "started_unix": time.time(), "pid": os.getpid(),
                                                   "wire_sha256": hashlib.sha256(wire).hexdigest()})
    started = time.monotonic()
    body = None
    try:
        try:
            with opener(req, client.timeout) as response:
                summary["http_status"] = response.status
                body = response.read()
        except error.HTTPError as exc:
            summary["http_status"] = exc.code
            with exc:
                body = exc.read()
        except Exception as exc:
            summary.update(status="transport_error", error_type=type(exc).__name__)
            return summary
        redacted_body = redact(body.decode("utf-8", errors="replace"), prepared["keys"])
        write_private(output / "response.json", {"schema_version": 1, "request_hash": summary["request_hash"],
            "http_status": summary["http_status"], "body_sha256": hashlib.sha256(body).hexdigest(),
            "body_text": redacted_body, "body_redacted": redacted_body != body.decode("utf-8", errors="replace")})
        try:
            raw = json.loads(body)
        except (ValueError, UnicodeError) as exc:
            summary.update(status="invalid_response_json", parse_status="error", error_type=type(exc).__name__)
            return summary
        summary.update(response_metadata(raw))
        if refused(raw):
            summary.update(status="content_filter", parse_status="refused", safe_code="content_filter")
            return summary
        if not 200 <= summary["http_status"] < 300:
            summary.update(status="http_error", error_type="HTTPError")
            return summary
        try:
            response = adapter.parse_response(client, raw)
            content = response.get("content")
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("empty or nontext response")
            payload = prepared["parse_json"](content)
            summary["parse_status"] = "ok"
        except Exception as exc:
            summary.update(status="parse_error", parse_status="error", error_type=type(exc).__name__)
            return summary
        try:
            prepared["validate"](payload)
            summary["validate_status"] = "ok"
        except Exception as exc:
            summary.update(status="validation_error", validate_status="error", error_type=type(exc).__name__)
            write_private(output / "validation.json", {"request_hash": summary["request_hash"],
                "error_type": type(exc).__name__, "error": redact(str(exc), prepared["keys"])})
            return summary
        artifact = {"schema_version": 1, "official_result": False, "request_hash": summary["request_hash"],
                    "response_sha256": hashlib.sha256(body).hexdigest(), **prepared["artifact"], "payload": payload}
        encoded = json.dumps(artifact, ensure_ascii=False, allow_nan=False)
        safe_encoded = redact(encoded, prepared["keys"])
        if safe_encoded != encoded:
            summary.update(status="validated_but_redacted", reusable=False)
            write_private(output / "validated_payload_redacted.json", json.loads(safe_encoded))
        else:
            write_private(output / "validated_payload.json", artifact)
            summary.update(status="validated", reusable=True, artifact_sha256=sha(output / "validated_payload.json"))
        return summary
    finally:
        summary["elapsed_seconds"] = time.monotonic() - started
        summary["parent_files_unchanged"] = all(sha(path) == expected for path, expected in prepared["protected"].items())
        if not summary["parent_files_unchanged"]:
            summary.update(status="parent_changed", reusable=False)
        # Even an allowlisted provider code/model can equal a credential.
        for key, value in list(summary.items()):
            if isinstance(value, str):
                summary[key] = redact(value, prepared["keys"])
        write_private(output / "summary.json", summary)


def run_probe(parent_run, video_id, output_dir, repo, opener=open_once):
    output_path = Path(output_dir).resolve()
    for protected_root in (Path(parent_run).resolve(), Path(repo).resolve()):
        if output_path == protected_root or protected_root in output_path.parents:
            raise ValueError("diagnostic output must be outside the parent run and frozen checkout")
    output = claim_output(parent_run, video_id, output_dir)
    try:
        prepared = prepare(parent_run, video_id, repo)
        return execute_once(prepared, output, opener)
    except Exception as exc:
        summary = {"schema_version": 1, "official_result": False, "video_id": video_id,
                   "status": "preflight_failed", "error_type": type(exc).__name__}
        if not (output / "summary.json").exists():
            write_private(output / "summary.json", summary)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-run", "video-id", "output-dir", "repo"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_probe(args.parent_run, args.video_id, args.output_dir, args.repo)
    except Exception as exc:
        print(json.dumps({"status": "not_dispatched_or_failed_locally", "error_type": type(exc).__name__}))
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status") == "validated" else 1


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
