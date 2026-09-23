"""One exact failed embedding-batch probe; preserve old caches and first scores."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib import request, error


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def private_json(path, value):
    data = (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-run", type=Path, required=True)
    p.add_argument("--core-repo", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--request-hash", required=True)
    p.add_argument("--video-id", default="G2_V000026")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.core_repo.resolve()))
    from methods.longemo.noevent_runner import parser
    from methods.longemo.noevent_retrieval import _document
    from methods.longemo.embeddings import APIEncoder
    from methods.longemo.common import fingerprint, code_hash

    parent = args.parent_run.resolve()
    task_path = parent / "tasks" / "answer" / ("noevent-" + args.video_id) / "task.json"
    task = json.loads(task_path.read_text())
    if task.get("status") != "error":
        raise ValueError("probe requires the stopped failed answer task")
    command = task["command"]
    module_index = command.index("methods.longemo.noevent_runner")
    old = parser().parse_args(command[module_index + 1:])
    if old.command != "answer" or old.embedding_backend != "gemini":
        raise ValueError("only the original compatible Gemini embedding profile is supported")
    config = json.loads((parent / "configuration.json").read_text())
    if code_hash() != config["source_hashes"]["noevent"]:
        raise ValueError("frozen noevent core changed")
    memory_path = Path(old.memory_dir) / args.video_id / "memory.json"
    memory = json.loads(memory_path.read_text())
    if not memory.get("complete") or memory.get("representation") != "window_records":
        raise ValueError("complete window-only memory required")
    ledger = Path(old.embedding_cache_dir) / "api" / "calls.jsonl"
    output = args.output_dir.resolve()
    for source in (parent, Path(old.embedding_cache_dir).resolve()):
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("diagnostics must be isolated from the original run and cache")
    if (ledger.parent / (args.request_hash + ".json")).exists():
        raise ValueError("the failed batch now has an existing cache; do not probe it again")
    failures = [json.loads(s) for s in ledger.read_text().splitlines()]
    failures = [v for v in failures if v.get("request_hash") == args.request_hash]
    if not failures or any(v.get("status") != "error" or v.get("http_status") != 400 for v in failures):
        raise ValueError("expected an unclassified HTTP400 batch with no successful call")
    args.output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    credentials = json.loads(Path(old.credential_file).read_text())
    key = credentials["OPENROUTER_API_KEY"]
    encoder = APIEncoder(key, args.output_dir / "private_cache", model=old.embedding_model,
                         base_url=old.embedding_base_url, tries=1)
    chunks = [piece for w in memory["windows"] for piece in encoder.chunks(_document(memory, w))]
    matches = []
    for offset in range(0, len(chunks), 16):
        inputs = ["title: Emotional event | text: " + t for t in chunks[offset:offset + 16]]
        if fingerprint({"config": encoder.config, "inputs": inputs}) == args.request_hash:
            matches.append((offset, inputs))
    if len(matches) != 1:
        raise ValueError("failed input batch could not be reproduced exactly")
    offset, inputs = matches[0]
    payload = {"model": encoder.config["model"], "input": inputs,
               "dimensions": encoder.config["dimension"], "encoding_format": "float"}
    body = json.dumps(payload).encode()
    intent = {"schema_version": 1, "parent_run": str(parent), "video_id": args.video_id,
              "request_hash": args.request_hash, "payload_sha256": hashlib.sha256(body).hexdigest(),
              "memory_sha256": hashlib.sha256(memory_path.read_bytes()).hexdigest(),
              "parent_ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
              "batch_offset": offset, "batch_count": len(inputs),
              "input_characters": [len(t) for t in inputs], "request_bytes": len(body),
              "new_probe_max_requests": 1, "old_run_unchanged": True, "execute": args.execute}
    private_json(args.output_dir / "intent.json", intent)
    if not args.execute:
        print(json.dumps(intent))
        return
    private_json(args.output_dir / "attempt_started.json", {"time_unix": time.time(), "request_hash": args.request_hash})
    started = time.monotonic()
    req = request.Request(encoder.config["base_url"] + "/embeddings", data=body,
                          headers={"Content-Type": "application/json", "Authorization": "Bearer " + key}, method="POST")
    status, content, network_error = None, b"", None
    try:
        with request.build_opener(NoRedirect()).open(req, timeout=120) as response:
            status, content = response.status, response.read()
    except error.HTTPError as exc:
        status, content = exc.code, exc.read()
    except Exception as exc:
        network_error = type(exc).__name__
    raw_text = content.decode("utf-8", errors="replace")
    private_json(args.output_dir / "private_response.json", {
        "request_hash": args.request_hash, "http_status": status,
        "response_sha256": hashlib.sha256(content).hexdigest(), "body": raw_text.replace(key, "[REDACTED]")})
    result = {"request_hash": args.request_hash, "http_status": status,
              "elapsed_seconds": time.monotonic() - started, "network_error_type": network_error,
              "valid_vectors": False, "response_sha256": hashlib.sha256(content).hexdigest()}
    try:
        data = json.loads(raw_text)
        if status == 200:
            rows = sorted(data["data"], key=lambda x: x["index"])
            if [r["index"] for r in rows] != list(range(len(inputs))):
                raise ValueError("invalid embedding indices")
            vectors = [r["embedding"] for r in rows]
            encoder._validate(vectors, len(inputs))
            private_json(args.output_dir / "validated_vectors.json", {
                "signature": args.request_hash, "vectors": vectors, "model_returned": data.get("model"),
                "diagnostic_source": "one_new_exact_batch_probe"})
            result["valid_vectors"] = True
        else:
            detail = data.get("error", {})
            code = str(detail.get("code", "")) if isinstance(detail, dict) else ""
            if key not in code and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", code):
                result["service_error_code"] = code
    except Exception as exc:
        result["parse_or_validation_error_type"] = type(exc).__name__
    private_json(args.output_dir / "result.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
