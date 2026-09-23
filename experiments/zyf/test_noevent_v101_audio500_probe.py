"""Offline request-reconstruction and single-dispatch tests; no remote calls."""
from __future__ import annotations

from contextlib import nullcontext
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from evaluation.clients import Client
from evaluation.inference.adapters import openai
from evaluation.inference.prompts import json_object
from experiments.zyf import noevent_v101_audio500_probe as diagnostic
from experiments.zyf import noevent_retry_probe as sender
from methods.longemo import audio, common


class Response(io.BytesIO):
    status = 200


class Audio500Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "runtime/runs" / diagnostic.SOURCE_NAME
        self.source.mkdir(parents=True)
        self.repo = self.root / "frozen-core"
        self.repo.mkdir()
        self.folder = self.source / "audio"
        self.folder.mkdir()
        self.client = Client("gemini-3.8-flash", sender.MATRIX, "chat", "fixture-secret", 180, 4096, None,
                             {"reasoning_effort": "low"})
        self.initial = [{"role": "system", "content": audio.AUDIO_PROMPT}, {"role": "user", "content": [
            {"type": "text", "text": json.dumps({"clip_start": 1258, "clip_end": 1282})},
            {"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}}]}]
        self.rows, self.paths, self.pins = [], [], {}
        messages = copy.deepcopy(self.initial)
        initial_hash = common.fingerprint(messages)
        contents = ['{"observations":[{"span":[1258,1259],"voice":"","cue":"hello"}]}',
                    '{"observations": [broken}']
        for index, content in enumerate(contents, 1):
            try:
                self.validate(json_object(content))
            except ValueError as exc:
                error_type, error_text = type(exc).__name__, str(exc)
            row = {"purpose": "audio_observer:" + diagnostic.WINDOW, "attempt": index, "status": "error",
                "failure_stage": "validate", "error_type": error_type, "validation_error": error_text[:1000],
                "request_hash": common.fingerprint(messages), "initial_request_hash": initial_hash,
                "retry_reason": "schema_repair", "retryable": True, "will_retry": True}
            path = self.folder / ("diagnostics/" + str(index) + ".json")
            path.parent.mkdir(exist_ok=True)
            row["response_sha256"] = hashlib.sha256(content.encode()).hexdigest()
            value = {"kind": "failed_model_content", "content": content, "content_redacted": False,
                "response_sha256": row["response_sha256"], "metadata": {k: row[k] for k in
                    ("purpose", "attempt", "request_hash", "initial_request_hash", "error_type", "failure_stage", "retry_reason")}}
            path.write_text(json.dumps(value))
            row.update(diagnostic_path=str(path.relative_to(self.folder)), diagnostic_sha256=sender.sha(path))
            self.paths.append(path)
            self.rows.append(row)
            self.wrapper(row, messages, 200, "ok", {"choices": [{"finish_reason": "stop", "message": {"content": content}}]})
            messages += [{"role": "assistant", "content": content}, {"role": "user", "content":
                "Repair the JSON to satisfy the schema. Validation error: " + error_text}]
        final_hash = common.fingerprint(messages)
        self.rows.append({"purpose": "audio_observer:" + diagnostic.WINDOW, "attempt": 3, "status": "error",
            "failure_stage": "generate", "error_type": "ServiceError", "request_hash": final_hash,
            "initial_request_hash": initial_hash, "http_status": 500, "retry_reason": "temporary_http_error",
            "retryable": True, "will_retry": False})
        self.wrapper(self.rows[-1], messages, 500, "error", {"error": [{"error": {
            "code": 500, "status": "INTERNAL", "message": "Internal error encountered."}}]})
        self.messages = messages
        self.wire = json.dumps(openai.build_payload(self.client, messages), allow_nan=False).encode()
        for name, value in (("FINAL_REQUEST", final_hash), ("ORIGINAL_DIAGNOSTIC_SHAS", self.pins),
                            ("ORIGINAL_WIRE_SHA", hashlib.sha256(self.wire).hexdigest()), ("ORIGINAL_WIRE_BYTES", len(self.wire))):
            handle = patch.object(diagnostic, name, value)
            handle.start()
            self.addCleanup(handle.stop)

    @staticmethod
    def validate(value):
        if not isinstance(value, dict) or not isinstance(value.get("observations"), list):
            raise ValueError("audio observations must be a list")
        for observation in value["observations"]:
            audio.span(observation.get("span"), [1258, 1282])
            for field in ("voice", "cue"):
                audio.text(observation.get(field), field)

    def wrapper(self, row, messages, status, outcome, raw):
        body = json.dumps(raw)
        wire = json.dumps(openai.build_payload(self.client, messages), allow_nan=False).encode()
        value = {"request_hash": row["request_hash"], "model_configuration": self.client.configuration(),
            "outcome": outcome, "serialized_requests": [{"payload_sha256": hashlib.sha256(wire).hexdigest(), "bytes": len(wire)}],
            "responses": [{"http_status": status, "body_text": body, "body_sha256": hashlib.sha256(body.encode()).hexdigest()}]}
        if outcome == "error":
            value["error_type"] = row["error_type"]
        path = self.source / "private_diagnostics/build" / diagnostic.VIDEO / (row["request_hash"] + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        self.pins[row["request_hash"]] = sender.sha(path)

    def prepared(self):
        messages, protected = diagnostic.replay_repairs(self.initial, self.rows, self.folder, common, json_object,
            self.validate, self.client.api_key, raw_content=lambda row, messages: diagnostic.repair_content(
                self.source, self.client, messages, openai, row))
        evidence = diagnostic.final_response_evidence(self.source, self.client, messages, openai, self.rows[-1])
        protected[evidence["diagnostic_path"]] = evidence["diagnostic_sha256"]
        return {"messages": messages, "failed": self.rows[-1], "fingerprint": common.fingerprint,
            "client": self.client, "adapter": openai, "parse_json": json_object, "validate": self.validate,
            "keys": [self.client.api_key], "protected": protected, "expected_wire_sha256": evidence["wire_sha256"],
            "artifact": {"video_id": diagnostic.VIDEO, "window_id": diagnostic.WINDOW, "stage": "audio",
                "audio_input_fingerprint": common.fingerprint({"messages": self.initial, "model": self.client.configuration()}),
                "probe_is_not_a_completed_window": True}, "evidence": evidence, "protected_score_count": 235,
            "source_configuration_sha256": "fixture-source-sha"}

    def run_patched(self, output, opener, source=None):
        with patch.object(diagnostic.base, "stopped_parent", return_value=nullcontext()), \
             patch.object(diagnostic, "prepare", side_effect=lambda *args: self.prepared()):
            return diagnostic.run("probe", source or self.source, self.repo, [], output, opener)

    def test_repair_chain_and_wire_match_then_success_is_only_audio_artifact(self):
        prepared = self.prepared()
        self.assertEqual(prepared["messages"], self.messages)
        raw = {"choices": [{"finish_reason": "stop", "message": {"content": '{"observations":[]}'}}]}
        opener = Mock(return_value=Response(json.dumps(raw).encode()))
        output = self.root / "diagnostic/once"
        result = self.run_patched(output, opener)
        self.assertEqual(result["status"], "validated")
        self.assertEqual(opener.call_count, 1)
        self.assertEqual(opener.call_args.args[0].data, self.wire)
        artifact = sender.read(output / "validated_payload.json")
        self.assertTrue(artifact["probe_is_not_a_completed_window"])
        self.assertEqual(artifact["audio_input_fingerprint"], prepared["artifact"]["audio_input_fingerprint"])
        self.assertFalse((output / "memory.json").exists())
        self.assertFalse((self.source / "audio/W00064.json").exists())

    def test_missing_or_changed_repair_input_prevents_http(self):
        opener = Mock()
        self.paths[0].unlink()
        with self.assertRaises(ValueError):
            self.run_patched(self.root / "diagnostic/missing", opener)
        opener.assert_not_called()

    def test_wrong_repair_error_hash_wire_and500_shape_are_rejected(self):
        original = self.rows[0]["validation_error"]
        self.rows[0]["validation_error"] = "wrong"
        with self.assertRaisesRegex(ValueError, "validation error"):
            self.prepared()
        self.rows[0]["validation_error"] = original
        self.rows[-1]["request_hash"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "eligible"):
            self.prepared()
        self.rows[-1]["request_hash"] = diagnostic.FINAL_REQUEST
        with patch.object(diagnostic, "ORIGINAL_WIRE_SHA", "f" * 64), self.assertRaisesRegex(ValueError, "wire bytes"):
            self.prepared()
        self.wrapper(self.rows[-1], self.messages, 500, "error", {"error": [{"code": 500, "status": "INTERNAL"}]})
        with self.assertRaisesRegex(ValueError, "nested"):
            self.prepared()

    def test_alias_duplicate_and_different_output_parent_cannot_dispatch_twice(self):
        raw = {"choices": [{"finish_reason": "stop", "message": {"content": '{"observations":[]}'}}]}
        opener = Mock(side_effect=lambda *a, **k: Response(json.dumps(raw).encode()))
        self.run_patched(self.root / "first/once", opener)
        alias = self.source.parent / ".." / "runs" / diagnostic.SOURCE_NAME
        with self.assertRaises(FileExistsError):
            self.run_patched(self.root / "different-parent/again", opener, source=alias)
        self.assertEqual(opener.call_count, 1)

    def test_http500_or429_response_is_saved_once_without_retry(self):
        for status in (500, 429):
            with self.subTest(status=status):
                prepared = self.prepared()
                folder = self.root / ("http" + str(status))
                folder.mkdir()
                opener = Mock(side_effect=HTTPError("https://example.invalid", status, "failure", {}, io.BytesIO(b'{"error": {"code": "INTERNAL"}}')))
                result = diagnostic.execute_once(prepared, folder, opener)
                self.assertEqual(result["status"], "http_error")
                self.assertEqual(result["http_status"], status)
                self.assertEqual(opener.call_count, 1)
                self.assertFalse((folder / "validated_payload.json").exists())

    def test_changed_source_or_wire_fails_before_sender_and_restores_helper(self):
        prepared = self.prepared()
        original_check = sender.check_eligible
        protected_path = self.paths[0]
        protected_path.write_text(protected_path.read_text() + " ")
        folder = self.root / "source-change"
        folder.mkdir()
        opener = Mock()
        with self.assertRaisesRegex(ValueError, "parent files changed"):
            diagnostic.execute_once(prepared, folder, opener)
        opener.assert_not_called()
        self.assertIs(sender.check_eligible, original_check)
        prepared["expected_wire_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "wire payload"):
            diagnostic.execute_once(prepared, folder, opener)
        opener.assert_not_called()


if __name__ == "__main__":
    unittest.main()
