"""Offline fixed-case history and one-request audio429 guards."""
from __future__ import annotations

from contextlib import nullcontext
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from evaluation.clients import Client
from evaluation.inference.adapters import openai
from evaluation.inference.prompts import json_object
from experiments.zyf import noevent_audio429_probe as diagnostic
from experiments.zyf import noevent_retry_probe as sender
from methods.longemo.common import fingerprint


class Response(io.BytesIO):
    status = 200


class Audio429Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "runtime/runs/length5"
        self.folder = self.source / "noevent/videos/G2_V000091/memory/G2_V000091"
        (self.folder / "audio").mkdir(parents=True)
        self.repo = self.root / "frozen-core"
        self.repo.mkdir()
        self.client = Client("gemini-3.8-flash", sender.MATRIX, "chat", "fixture-secret", 180, 4096, None,
                             {"reasoning_effort": "low"})
        self.messages = [{"role": "user", "content": "unchanged original audio request"}]
        request_hash = fingerprint(self.messages)
        self.wire = json.dumps(openai.build_payload(self.client, self.messages), allow_nan=False).encode()
        wire_item = {"payload_sha256": hashlib.sha256(self.wire).hexdigest(), "bytes": len(self.wire)}
        self.case = {"source_run": str(self.source), "configuration_sha256": "fixture-config-sha", "video": "G2_V000091",
            "window": "W00025", "completed_windows": 24, "total_windows": 66, "visual_max_tokens": 16384,
            "question_ids": ["Q1"], "ledger_rows": [], "diagnostics": []}
        self.error = [{"error": {"code": 429, "message": diagnostic.EXHAUSTED_MESSAGE, "status": "RESOURCE_EXHAUSTED"}}]
        for attempt in range(1, 4):
            t = 1700000000 + attempt * 10
            row = {"purpose": "audio_observer:W00025", "attempt": attempt, "model": "gemini-3.8-flash",
                "request_hash": request_hash, "initial_request_hash": request_hash, "time_unix": t,
                "status": "error", "error_type": "ServiceError", "failure_stage": "generate", "elapsed_seconds": 1.04,
                "http_status": 429, "retryable": True, "retry_reason": "temporary_http_error", "will_retry": attempt < 3}
            self.case["ledger_rows"].append(row)
            body = json.dumps({"error": self.error, "requestId": str(attempt)})
            body_sha = hashlib.sha256(body.encode()).hexdigest()
            started = datetime.fromtimestamp(t + .008, timezone.utc).isoformat()
            value = {"request_hash": request_hash, "outcome": "error", "error_type": "ServiceError", "pid": 1234,
                "model_configuration": self.client.configuration(), "serialized_requests": [wire_item], "started_at": started,
                "elapsed_seconds": 1.01, "responses": [{"http_status": 429, "body_text": body, "body_sha256": body_sha}]}
            path = self.source / "private_diagnostics/build/G2_V000091" / (str(attempt) + ".json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
            self.case["diagnostics"].append({"path": str(path), "sha256": sender.sha(path), "serialized_requests": [wire_item],
                "started_at": started, "elapsed_seconds": 1.01, "pid": 1234, "model_configuration": self.client.configuration(),
                "responses": [{"http_status": 429, "body_sha256": body_sha, "error": self.error}]})
        ledger = self.folder / "audio/calls.jsonl"
        ledger.write_text("".join(json.dumps(row) + "\n" for row in self.case["ledger_rows"]))
        self.case["audio_ledger_sha256"] = sender.sha(ledger)
        self.specs = self.root / "specs.json"
        values = []
        for video, (window, count, total, visual) in diagnostic.CASES.items():
            value = copy.deepcopy(self.case)
            value.update(video=video, window=window, completed_windows=count, total_windows=total, visual_max_tokens=visual)
            values.append(value)
        self.specs.write_text(json.dumps({"cases": values}))
        handle = patch.object(diagnostic, "SPECS_SHA", sender.sha(self.specs))
        handle.start()
        self.addCleanup(handle.stop)

    def prepared(self):
        row, protected, evidence = diagnostic.verify_history(self.case, self.folder, self.client, self.messages, openai, {"pid": 1234})
        return {"case": self.case, "messages": self.messages, "client": self.client, "adapter": openai,
            "failed": row, "fingerprint": fingerprint, "parse_json": json_object,
            "validate": lambda value: self.assertEqual(value, {"observations": []}), "protected": protected,
            "keys": [self.client.api_key], "expected_wire_sha256": hashlib.sha256(self.wire).hexdigest(),
            "expected_wire_bytes": len(self.wire), "evidence": evidence,
            "artifact": {"video_id": self.case["video"], "window_id": self.case["window"], "stage": "audio",
                "visual_max_tokens": 16384, "probe_is_not_a_completed_window": True,
                "audio_input_fingerprint": fingerprint({"messages": self.messages, "model": self.client.configuration()})}}

    def run_patched(self, output, opener):
        with patch.object(diagnostic.base, "stopped_parent", return_value=nullcontext()), \
             patch.object(diagnostic, "prepare", side_effect=lambda *args: self.prepared()):
            return diagnostic.run("probe", self.specs, self.case["video"], self.repo, [], output, opener)

    def test_three_same_hash_attempts_reconstruct_then_exactly_one_http_preserves16k_source(self):
        prepared = self.prepared()
        self.assertEqual(len(prepared["evidence"]), 3)
        raw = {"choices": [{"finish_reason": "stop", "message": {"content": '{"observations":[]}'}}]}
        opener = Mock(return_value=Response(json.dumps(raw).encode()))
        output = self.root / "output/probe"
        value = self.run_patched(output, opener)
        self.assertEqual(value["status"], "validated")
        self.assertEqual(opener.call_count, 1)
        self.assertEqual(opener.call_args.args[0].data, self.wire)
        artifact = sender.read(output / "validated_payload.json")
        self.assertEqual(artifact["visual_max_tokens"], 16384)
        self.assertTrue(artifact["probe_is_not_a_completed_window"])
        self.assertFalse((self.folder / "audio/W00025.json").exists())

    def test_missing_or_extra_same_hash_wrapper_prevents_dispatch(self):
        opener = Mock()
        path = Path(self.case["diagnostics"][0]["path"])
        data = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            self.run_patched(self.root / "missing", opener)
        path.write_bytes(data)
        path.with_name("extra.json").write_bytes(data)
        with self.assertRaisesRegex(ValueError, "additional"):
            self.run_patched(self.root / "extra", opener)
        opener.assert_not_called()

    def test_wire_pid_timing_and_policy_mismatch_cannot_be_relabelled_as429(self):
        for field, value in (("pid", 9999), ("elapsed_seconds", 20), ("serialized_requests", [])):
            with self.subTest(field=field):
                original = copy.deepcopy(self.case["diagnostics"][0])
                self.case["diagnostics"][0][field] = value
                with self.assertRaises(ValueError):
                    self.prepared()
                self.case["diagnostics"][0] = original
        path = Path(self.case["diagnostics"][0]["path"])
        original = sender.read(path)
        changed = copy.deepcopy(original)
        changed["responses"][0]["body_text"] = json.dumps({"error": [{"error": {"code": 429,
            "status": "insufficient_quota", "message": "billing rejected"}}]})
        changed["responses"][0]["body_sha256"] = hashlib.sha256(changed["responses"][0]["body_text"].encode()).hexdigest()
        path.write_text(json.dumps(changed))
        self.case["diagnostics"][0]["sha256"] = sender.sha(path)
        self.case["diagnostics"][0]["responses"][0]["body_sha256"] = changed["responses"][0]["body_sha256"]
        with self.assertRaisesRegex(ValueError, "resource-exhausted"):
            self.prepared()

    def test_original_three_attempt_budget_and_specs_are_immutable(self):
        case = diagnostic.load_case(self.specs, self.case["video"])
        self.assertEqual(case["visual_max_tokens"], 16384)
        self.specs.write_text(self.specs.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "specification changed"):
            diagnostic.load_case(self.specs, self.case["video"])
        self.case["ledger_rows"][-1]["attempt"] = 1
        with self.assertRaises(ValueError):
            self.prepared()

    def test_global_claim_and_serial_lock_prevent_parallel_or_second_request(self):
        opener = Mock()
        registry = self.source.parents[1] / "recovery_claims/noevent_audio429"
        registry.mkdir(parents=True)
        with (registry / "serial.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with self.assertRaises(BlockingIOError):
                self.run_patched(self.root / "blocked", opener)
        opener.assert_not_called()
        raw = {"choices": [{"finish_reason": "stop", "message": {"content": '{"observations":[]}'}}]}
        opener = Mock(side_effect=lambda *a, **k: Response(json.dumps(raw).encode()))
        self.run_patched(self.root / "one/first", opener)
        with self.assertRaises(FileExistsError):
            self.run_patched(self.root / "different-parent/second", opener)
        self.assertEqual(opener.call_count, 1)

    def test_new429_response_or_changed_protected_input_never_triggers_retry(self):
        prepared = self.prepared()
        output = self.root / "again429"
        output.mkdir()
        opener = Mock(side_effect=HTTPError(sender.MATRIX, 429, "resource exhausted", {}, io.BytesIO(json.dumps({"error": self.error}).encode())))
        value = diagnostic.execute_once(prepared, output, opener)
        self.assertEqual(value["status"], "http_error")
        self.assertEqual(value["http_status"], 429)
        self.assertEqual(opener.call_count, 1)
        path = Path(next(iter(prepared["protected"])))
        path.write_text(path.read_text() + " ")
        opener = Mock()
        with self.assertRaisesRegex(ValueError, "parent files changed"):
            diagnostic.execute_once(prepared, self.root / "changed", opener)
        opener.assert_not_called()


if __name__ == "__main__":
    unittest.main()
