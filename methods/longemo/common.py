"""Reproducible I/O and bounded API calls; credentials never enter manifests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time

from evaluation.inference.prompts import json_object
from evaluation.io_utils import write_json

_LOCK = threading.Lock()


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


class LoggedClient:
    def __init__(self, client, ledger, tries=3):
        self.client, self.ledger, self.tries = client, Path(ledger), tries

    def call(self, messages, *, purpose, validate=None):
        messages = list(messages)
        request_id = fingerprint(messages)
        for attempt in range(1, self.tries + 1):
            response = None
            started = time.monotonic()
            record = {"purpose": purpose, "request_hash": request_id, "attempt": attempt,
                      "model": self.client.model, "time_unix": time.time()}
            try:
                response = self.client.generate(messages)
                record["usage"] = token_usage(response.get("usage"))
                raw = response.get("raw_response") or {}
                record["response_model"] = raw.get("modelVersion", raw.get("model"))
                text = response["content"]
                value = json_object(text) if validate else text.strip()
                if validate:
                    validate(value)
                elif not value:
                    raise ValueError("empty answer")
                record.update(status="ok", elapsed_seconds=time.monotonic() - started)
                append_json(self.ledger, record)
                return value
            except Exception as exc:
                record.update(status="error", error_type=type(exc).__name__, elapsed_seconds=time.monotonic() - started)
                if getattr(exc, "status_code", None) is not None:
                    record["http_status"] = exc.status_code
                    record["service_error_code"] = getattr(exc, "code", None)
                if isinstance(exc, ValueError):
                    record["validation_error"] = str(exc)[:1000]
                append_json(self.ledger, record)
                if attempt == self.tries or getattr(exc, "retryable", True) is False:
                    raise RuntimeError(f"{purpose}: {type(exc).__name__} after {attempt} attempts") from None
                if validate and isinstance(exc, ValueError) and response is not None:
                    messages += [{"role": "assistant", "content": response["content"]},
                                 {"role": "user", "content": "Repair the JSON to satisfy the schema. Validation error: " + str(exc)}]
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
