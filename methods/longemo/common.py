"""Reproducible I/O and bounded API calls; credentials never enter manifests."""
from __future__ import annotations

import hashlib
import json
import errno
import os
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
from urllib.error import HTTPError, URLError
from uuid import uuid4

from evaluation.clients import ServiceError
from evaluation.inference.prompts import json_object
from evaluation.io_utils import write_json

_LOCK = threading.Lock()
_TRANSIENT_HTTP = {408, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524}
_EXTENDED_TRANSIENT_HTTP = {408, 520, 521, 522, 523, 524}
_NON_RETRYABLE_CODES = {
    "contentfilter", "contentpolicyviolation", "responsibleaipolicyviolation",
    "safety", "blockedprompt", "prohibitedcontent", "insufficientquota",
    "insufficientcredits", "insufficientbalance", "billinghardlimitreached",
    "authenticationerror", "invalidapikey", "permissiondenied", "unauthorized", "forbidden",
}
_TRANSIENT_ERRNOS = {
    errno.ETIMEDOUT, errno.ECONNRESET, errno.ECONNABORTED, errno.ECONNREFUSED,
    errno.EPIPE, errno.ENETUNREACH, errno.EHOSTUNREACH,
}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def code_hash():
    root = Path(__file__).resolve().parents[2]
    files = list(Path(__file__).parent.glob("*.py"))
    files += [p for p in (root / "evaluation").rglob("*.py") if "emollm" not in p.parts]
    return fingerprint({str(p.relative_to(root)): file_hash(p) for p in sorted(files)})


def git_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def append_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, path.open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def manifest(path, configuration):
    """Refuse stale cache reuse instead of silently mixing configurations."""
    path = Path(path)
    value = {"fingerprint": fingerprint(configuration), "configuration": configuration}
    if path.exists():
        old = json.loads(path.read_text())
        # A documentation-only commit changes HEAD but not the hashed source.
        # Preserve the original manifest and provenance for an identical run.
        comparable = lambda c: {k:v for k,v in c.items() if k != "git_revision"}
        if comparable(old["configuration"]) != comparable(configuration):
            raise ValueError(f"configuration changed; use a new output directory: {path.parent}")
        return old["fingerprint"]
    write_json(path, value)
    return value["fingerprint"]


def token_usage(raw):
    raw = raw or {}
    prompt = raw.get("prompt_tokens", raw.get("promptTokenCount"))
    completion = raw.get("completion_tokens", raw.get("candidatesTokenCount"))
    total = raw.get("total_tokens", raw.get("totalTokenCount"))
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total,
            "thought_tokens": raw.get("thoughtsTokenCount"), "raw": raw}


def _service_metadata(exc):
    status = getattr(exc, "status_code", None)
    if status is None and isinstance(exc, HTTPError):
        status = exc.code
    status = status if type(status) is int else None
    code = getattr(exc, "code", None)
    code = code if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", code) else None
    return status, code


def _temporary_network_error(exc):
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    if isinstance(exc, socket.gaierror):
        return exc.errno == socket.EAI_AGAIN
    if isinstance(exc, URLError) and isinstance(exc.reason, BaseException):
        return _temporary_network_error(exc.reason)
    if isinstance(exc, OSError):
        return exc.errno in _TRANSIENT_ERRNOS
    return False


def _retry_decision(exc, *, structural_failure=False, empty_response=False, transport_failure=False):
    """Only evidenced temporary transport errors or nonempty schema failures retry."""
    status, code = _service_metadata(exc)
    normalized_code = re.sub(r"[^a-z0-9]", "", (code or "").lower())
    if normalized_code in _NON_RETRYABLE_CODES:
        return False, "policy_authentication_or_credit_refusal"
    if status in (401, 402, 403, 404) or (status is not None and status >= 400 and status not in _TRANSIENT_HTTP):
        return False, "permanent_http_error"
    # ServiceError historically omitted several explicit gateway failures from
    # its retry table. Extend only those known statuses; an explicit veto from
    # another transport (for example exhausted internal retries) remains final.
    extended_gateway = isinstance(exc, ServiceError) and status in _EXTENDED_TRANSIENT_HTTP
    if getattr(exc, "retryable", None) is False and not extended_gateway:
        return False, "explicit_nonretryable_error"
    if empty_response:
        return False, "empty_or_nontext_model_response"
    if structural_failure:
        return True, "schema_repair"
    if not transport_failure:
        return False, "unclassified_response_or_local_error"
    if status in _TRANSIENT_HTTP:
        return True, "temporary_http_error"
    if _temporary_network_error(exc):
        return True, "temporary_network_error"
    return False, "unclassified_error"


class LoggedClient:
    def __init__(self, client, ledger, tries=3):
        if type(tries) is not int or tries < 1:
            raise ValueError("tries must be a positive integer")
        self.client, self.ledger, self.tries = client, Path(ledger), tries

    def _redact_key(self, value):
        key = getattr(self.client, "api_key", None)
        return value.replace(key, "[REDACTED_API_KEY]") if isinstance(key, str) and key else value

    def _save_failed_content(self, content, record):
        """Write only model text and allowlisted metadata, never the request/raw envelope."""
        relative = Path("diagnostics") / self.ledger.stem / (
            f"{record['initial_request_hash'][:16]}-{record['attempt']}-{uuid4().hex}.json")
        path = self.ledger.parent / relative
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        safe_content = self._redact_key(content)
        metadata = {key: record[key] for key in (
            "purpose", "model", "response_model", "attempt", "time_unix", "request_hash",
            "initial_request_hash", "error_type", "failure_stage", "retry_reason") if key in record}
        metadata = {key: self._redact_key(value) if isinstance(value, str) else value
                    for key, value in metadata.items()}
        value = {"schema_version": 1, "kind": "failed_model_content", "metadata": metadata,
                 "content": safe_content, "content_redacted": safe_content != content,
                 "response_sha256": hashlib.sha256(content.encode()).hexdigest()}
        encoded = (json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n").encode()
        # Exclusive, private files keep diagnostics immutable even across
        # threads and later attempts of the same logical request.
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
        return {"diagnostic_path": relative.as_posix(), "diagnostic_sha256": file_hash(path),
                "response_sha256": value["response_sha256"]}

    def call(self, messages, *, purpose, validate=None):
        messages = list(messages)
        initial_request_id = fingerprint(messages)
        for attempt in range(1, self.tries + 1):
            response = None
            content = None
            phase = "generate"
            empty_response = False
            started = time.monotonic()
            record = {"purpose": purpose, "request_hash": fingerprint(messages),
                      "initial_request_hash": initial_request_id, "attempt": attempt,
                      "model": self.client.model, "time_unix": time.time()}
            try:
                response = self.client.generate(messages)
                phase = "response"
                if not isinstance(response, dict):
                    empty_response = True
                    raise RuntimeError("model returned no structured response")
                record["usage"] = token_usage(response.get("usage"))
                raw = response.get("raw_response") or {}
                if isinstance(raw, dict):
                    response_model = raw.get("modelVersion", raw.get("model"))
                    if isinstance(response_model, str) and re.fullmatch(r"[A-Za-z0-9_./:@+-]{1,160}", response_model):
                        record["response_model"] = self._redact_key(response_model)
                content = response.get("content")
                if not isinstance(content, str) or not content.strip():
                    empty_response = True
                    raise RuntimeError("model returned empty or nontext content")
                phase = "validate"
                value = json_object(content) if validate else content.strip()
                if validate:
                    validate(value)
                record.update(status="ok", elapsed_seconds=time.monotonic() - started)
                phase = "record"
                append_json(self.ledger, record)
                return value
            except Exception as exc:
                record.update(status="error", error_type=type(exc).__name__, failure_stage=phase,
                              elapsed_seconds=time.monotonic() - started)
                status, code = _service_metadata(exc)
                if status is not None:
                    record["http_status"] = status
                if code is not None:
                    record["service_error_code"] = self._redact_key(code)
                if isinstance(exc, ValueError):
                    record["validation_error"] = self._redact_key(str(exc))[:1000]
                structural_failure = phase == "validate" and isinstance(exc, ValueError) and validate is not None
                retryable, reason = _retry_decision(exc, structural_failure=structural_failure,
                                                   empty_response=empty_response, transport_failure=phase == "generate")
                record.update(retryable=retryable, retry_reason=reason,
                              will_retry=retryable and attempt < self.tries)
                if isinstance(content, str) and (phase == "validate" or empty_response):
                    try:
                        record.update(self._save_failed_content(content, record))
                    except OSError as diagnostic_error:
                        record.update(diagnostic_write_error=type(diagnostic_error).__name__, will_retry=False)
                append_json(self.ledger, record)
                if not record["will_retry"]:
                    raise RuntimeError(f"{purpose}: {type(exc).__name__} after {attempt} attempts") from None
                if structural_failure:
                    messages += [{"role": "assistant", "content": self._redact_key(content)},
                                 {"role": "user", "content": "Repair the JSON to satisfy the schema. Validation error: " + self._redact_key(str(exc))}]
                time.sleep(min(2 ** (attempt - 1), 4))


def usage_summary(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()] if Path(path).exists() else []
    result = {"attempts": len(rows), "successful_calls": sum(r["status"] == "ok" for r in rows),
              "missing_usage_attempts": sum(not r.get("usage", {}).get("raw") for r in rows),
              "elapsed_api_seconds": sum(r["elapsed_seconds"] for r in rows)}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "thought_tokens"):
        result[key] = sum(r.get("usage", {}).get(key) or 0 for r in rows)
    costs = [r.get("usage", {}).get("raw", {}).get("cost") for r in rows]
    result["provider_reported_cost_usd"] = sum(x for x in costs if isinstance(x, (int, float)))
    result["missing_cost_attempts"] = sum(x is None for x in costs)
    return result
