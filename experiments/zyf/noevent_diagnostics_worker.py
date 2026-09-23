"""Run the unchanged noevent builder with private response diagnostics.

Client.generate itself, urllib transport, protocol parsing, inputs and retries
stay unchanged. A module-local JSON proxy observes the bytes already serialized
or read by the original client; it never reads ahead or issues a request. The
original return value or exception is propagated unchanged. No credentials,
request headers, input text, frames or audio are written to diagnostics.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import uuid


_TOKEN = re.compile(r"^[A-Za-z0-9_.:/@+-]{1,160}$")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def redact(text, keys):
    for key in sorted({value for value in keys if isinstance(value, str) and value}, key=len, reverse=True):
        text = text.replace(key, "[REDACTED_API_KEY]")
    text = re.sub(r"(?i)Bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", text)
    return re.sub(r"\b(?:sk-|hf_|ghp_)[A-Za-z0-9_-]{12,}", "[REDACTED_CREDENTIAL]", text)


def safe_token(value):
    return value if isinstance(value, str) and _TOKEN.fullmatch(value) else None


def response_metadata(value):
    if not isinstance(value, dict):
        return {"json_type": type(value).__name__}
    error = value.get("error")
    feedback = value.get("promptFeedback")
    choices, candidates = value.get("choices"), value.get("candidates")
    finish = []
    if isinstance(choices, list):
        finish.extend(safe_token(c.get("finish_reason")) for c in choices if isinstance(c, dict))
    if isinstance(candidates, list):
        finish.extend(safe_token(c.get("finishReason")) for c in candidates if isinstance(c, dict))
    return {"json_type": "dict", "response_model": safe_token(value.get("modelVersion", value.get("model"))),
            "choices_count": len(choices) if isinstance(choices, list) else None,
            "candidates_count": len(candidates) if isinstance(candidates, list) else None,
            "finish_reasons": [x for x in finish if x is not None],
            "block_reason": safe_token(feedback.get("blockReason")) if isinstance(feedback, dict) else None,
            "error_code": safe_token(error.get("code", error.get("status"))) if isinstance(error, dict) else None}


def private_directory(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("private diagnostics directory must not contain symlink aliases")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not path.is_dir():
        raise ValueError("private diagnostics path is not a directory")
    os.chmod(path, 0o700)
    return path


def private_write(path, value, keys):
    payload = redact(json.dumps(value, ensure_ascii=False, allow_nan=False), keys).encode() + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)


class _ReadTap:
    def __init__(self, stream):
        self.stream = stream
        self.parts = []

    def read(self, *args, **kwargs):
        # Delegating precisely the original read preserves its boundaries,
        # timeout, failure and returned value. No extra read drains a response.
        value = self.stream.read(*args, **kwargs)
        if isinstance(value, bytes):
            self.parts.append(value)
        elif isinstance(value, str):
            self.parts.append(value.encode())
        return value

    def __getattr__(self, name):
        return getattr(self.stream, name)


class _JSONTap:
    """Proxy only evaluation.clients.json, never the global json module."""
    def __init__(self, original, active):
        self.original, self.active = original, active

    def dumps(self, *args, **kwargs):
        value = self.original.dumps(*args, **kwargs)
        current = getattr(self.active, "record", None)
        if current is not None:
            body = value.encode()
            current["serialized_requests"].append({"payload_sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)})
        return value

    def load(self, stream, *args, **kwargs):
        current = getattr(self.active, "record", None)
        if current is None:
            return self.original.load(stream, *args, **kwargs)
        tap = _ReadTap(stream)
        status = getattr(stream, "status", getattr(stream, "code", None))
        record = {"http_status": status if type(status) is int else None}
        current["responses"].append(record)
        try:
            value = self.original.load(tap, *args, **kwargs)
            record.update(json_parsed=True, metadata=response_metadata(value))
            return value
        except BaseException as exc:
            record.update(json_parsed=False, parse_error_type=type(exc).__name__)
            raise
        finally:
            body = b"".join(tap.parts)
            record.update(body_sha256=hashlib.sha256(body).hexdigest(), body_bytes=len(body),
                          body_text=body.decode("utf-8", errors="replace"), body_text_encoding="utf-8; invalid bytes replaced")

    def __getattr__(self, name):
        return getattr(self.original, name)


@contextmanager
def capture_clients(client_module, directory, keys=()):
    """Observe all standard Client calls made by this isolated build process."""
    directory = private_directory(directory)
    original_json, original_generate = client_module.json, client_module.Client.generate
    active = threading.local()
    client_module.json = _JSONTap(original_json, active)

    def generate(client, messages):
        previous = getattr(active, "record", None)
        request_hash = fingerprint(messages)
        name = request_hash[:16] + "-" + uuid.uuid4().hex
        secrets = tuple(keys) + (client.api_key,)
        record = {"schema_version": 1, "request_hash": request_hash, "model_configuration": client.configuration(),
                  "started_at": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
                  "thread_id": threading.get_ident(), "serialized_requests": [], "responses": []}
        # Claim diagnostics before the API call, so a crash cannot masquerade
        # as an unattempted request. Failure here occurs before any request.
        private_write(directory / (name + ".intent.json"), record, secrets)
        active.record = record
        started = time.monotonic()
        try:
            result = original_generate(client, messages)
            record["outcome"] = "ok"
            return result
        except BaseException as exc:
            record.update(outcome="error", error_type=type(exc).__name__, error_message=str(exc))
            raise
        finally:
            active.record = previous
            record["elapsed_seconds"] = time.monotonic() - started
            # Post-request diagnostics may not replace the original outcome or
            # cause LoggedClient to retry an otherwise successful model call.
            try:
                private_write(directory / (name + ".json"), record, secrets)
            except Exception as exc:
                print(f"Private model diagnostics could not be saved ({type(exc).__name__}); request outcome unchanged.",
                      file=sys.stderr, flush=True)

    client_module.Client.generate = generate
    try:
        yield
    finally:
        client_module.Client.generate = original_generate
        client_module.json = original_json


def bind_core(repo):
    repo = Path(repo).resolve(strict=True)
    sys.path.insert(0, str(repo))
    modules = [importlib.import_module(name) for name in
               ("methods.longemo.noevent_runner", "methods.longemo.common", "evaluation.clients")]
    if any(not Path(module.__file__).resolve().is_relative_to(repo) for module in modules):
        raise ValueError("diagnostic wrapper imported a different core checkout")
    return modules


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--core-repo", required=True)
    p.add_argument("--core-sha256", required=True)
    p.add_argument("--diagnostics-dir", required=True)
    p.add_argument("builder_arguments", nargs=argparse.REMAINDER)
    args = p.parse_args(argv)
    builder = args.builder_arguments[1:] if args.builder_arguments[:1] == ["--"] else args.builder_arguments
    if not builder or builder[0] != "build":
        raise ValueError("the response diagnostic wrapper is limited to noevent build")
    runner, common, clients = bind_core(args.core_repo)
    if common.code_hash() != args.core_sha256:
        raise ValueError("frozen noevent core hash changed before the diagnostic build")
    parsed = runner.parser().parse_args(builder)
    if parsed.workers != 1:
        raise ValueError("each continuation build subprocess must own one video")
    credentials = json.loads(Path(parsed.credential_file).read_text()) if parsed.credential_file else {}
    keys = [value for key, value in credentials.items() if re.search(r"key|token|password|secret", key, re.I)
            and isinstance(value, str)]
    directory = private_directory(args.diagnostics_dir)
    with capture_clients(clients, directory, keys):
        result = runner.main(builder)
    if common.code_hash() != args.core_sha256:
        raise ValueError("frozen core files changed during the diagnostic build")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
