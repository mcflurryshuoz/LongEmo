import copy
import hashlib
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

from evaluation.clients import ServiceError
from methods.longemo.audio import AUDIO_PROMPT, bridge_audio
from methods.longemo.common import LoggedClient, file_hash


class SequenceClient:
    model = "mock-perception"
    api_key = "PRIVATE_API_KEY_123456"

    def __init__(self, results):
        self.results = list(results)
        self.requests = []

    def configuration(self):
        return {"model": self.model}

    def generate(self, messages):
        self.requests.append(copy.deepcopy(messages))
        value = self.results.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def response(content):
    return {"content": content, "usage": {"total_tokens": 7}, "raw_response": {
        "modelVersion": "mock-version", "headers": {"Authorization": SequenceClient.api_key},
        "request": {"private_media": "PRIVATE_INPUT_MEDIA"}, "opaque_provider_data": "DO_NOT_COPY"}}


def validate_ok(value):
    if not isinstance(value, dict) or value.get("ok") is not True:
        raise ValueError("ok must be true")


class AudioObservationReliabilityTests(unittest.TestCase):
    def test_silence_music_ambience_and_unknown_voice_are_valid(self):
        cases = [[], [{"span": [10, 11], "voice": "non-speech: music", "cue": "A soft melody plays."}],
                 [{"span": [10, 11], "voice": "non-speech: ambience", "cue": "A door closes."}],
                 [{"span": [10, 11], "voice": "unknown speaker: low, quiet voice", "cue": "Says hello softly."}]]
        for observations in cases:
            with self.subTest(observations=observations), tempfile.TemporaryDirectory() as tmp:
                client = SequenceClient([response(json.dumps({"observations": observations}))])
                content = [{"type": "text", "text": "frames"},
                           {"type": "input_audio", "input_audio": {"data": "PRIVATE_INPUT_MEDIA", "format": "wav"}}]
                output, _ = bridge_audio(content, [10, 12], client, Path(tmp) / "W1.json", 3)
                self.assertEqual(len(client.requests), 1)
                self.assertFalse(any(part["type"] == "input_audio" for part in output))
                self.assertEqual(json.loads((Path(tmp) / "W1.json").read_text())["result"]["observations"], observations)
                sent_prompt = client.requests[0][0]["content"]
                self.assertEqual(sent_prompt, AUDIO_PROMPT)
                self.assertIn("nonempty voice", sent_prompt)
                self.assertIn("non-speech: music", sent_prompt)
                self.assertIn("non-speech: ambience", sent_prompt)
                self.assertIn("genuine silence", sent_prompt)

    def test_voice_cue_and_span_validation_are_not_relaxed(self):
        valid = {"span": [10, 11], "voice": "non-speech: music", "cue": "A melody plays."}
        for changed in ({"voice": ""}, {"cue": ""}, {"span": [1, 2]}):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as tmp:
                client = SequenceClient([response(json.dumps({"observations": [{**valid, **changed}]}))])
                path = Path(tmp) / "W1.json"
                with self.assertRaisesRegex(RuntimeError, "after 1 attempts"):
                    bridge_audio([{"type": "input_audio", "input_audio": {"data": "x", "format": "wav"}}],
                                 [10, 12], client, path, 1)
                self.assertFalse(path.exists())
                self.assertEqual(len(client.requests), 1)


class LoggedClientReliabilityTests(unittest.TestCase):
    def ledger(self, directory):
        return [json.loads(line) for line in (Path(directory) / "calls.jsonl").read_text().splitlines()]

    def test_nonempty_json_and_schema_failures_are_preserved_before_bounded_repair(self):
        for failed_content in ('{"ok":', '{"ok": false}'):
            with self.subTest(content=failed_content), tempfile.TemporaryDirectory() as tmp, \
                 patch("methods.longemo.common.time.sleep") as sleep:
                client = SequenceClient([response(failed_content), response('{"ok": true}')])
                messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {
                    "url": "data:image/png;base64,PRIVATE_INPUT_MEDIA"}}]}]
                api = LoggedClient(client, Path(tmp) / "calls.jsonl", 3)
                self.assertEqual(api.call(messages, purpose="window:W1", validate=validate_ok), {"ok": True})
                rows = self.ledger(tmp)
                self.assertEqual([row["status"] for row in rows], ["error", "ok"])
                self.assertEqual(rows[0]["retry_reason"], "schema_repair")
                self.assertTrue(rows[0]["will_retry"])
                self.assertNotEqual(rows[0]["request_hash"], rows[1]["request_hash"])
                self.assertEqual(rows[0]["initial_request_hash"], rows[1]["initial_request_hash"])
                path = Path(tmp) / rows[0]["diagnostic_path"]
                saved = json.loads(path.read_text())
                self.assertEqual(saved["content"], failed_content)
                self.assertFalse(saved["content_redacted"])
                self.assertEqual(saved["response_sha256"], hashlib.sha256(failed_content.encode()).hexdigest())
                self.assertEqual(rows[0]["diagnostic_sha256"], file_hash(path))
                self.assertEqual(saved["metadata"]["request_hash"], rows[0]["request_hash"])
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
                stored = "\n".join(p.read_text() for p in Path(tmp).rglob("*") if p.is_file())
                for forbidden in ("PRIVATE_INPUT_MEDIA", client.api_key, "Authorization", "DO_NOT_COPY", "data:image"):
                    self.assertNotIn(forbidden, stored)
                self.assertEqual(client.requests[1][-2]["content"], failed_content)
                sleep.assert_called_once()

    def test_provider_echoed_credential_is_redacted_but_original_response_hash_is_kept(self):
        content = json.dumps({"ok": False, "echo": SequenceClient.api_key})
        with tempfile.TemporaryDirectory() as tmp:
            client = SequenceClient([response(content)])
            with self.assertRaises(RuntimeError):
                LoggedClient(client, Path(tmp) / "calls.jsonl", 1).call([], purpose="probe", validate=validate_ok)
            row = self.ledger(tmp)[0]
            saved = json.loads((Path(tmp) / row["diagnostic_path"]).read_text())
            self.assertTrue(saved["content_redacted"])
            self.assertNotIn(client.api_key, json.dumps(saved))
            self.assertIn("[REDACTED_API_KEY]", saved["content"])
            self.assertEqual(saved["response_sha256"], hashlib.sha256(content.encode()).hexdigest())

    def test_unknown_validator_errors_preserve_content_without_retrying(self):
        content = '{"ok": true}'
        errors = [TypeError("implementation error"), KeyError("missing internal field"),
                  RuntimeError("unexpected validator failure"), ConnectionResetError("local validation error")]
        for error in errors:
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as tmp, \
                 patch("methods.longemo.common.time.sleep") as sleep:
                client = SequenceClient([response(content)])

                def validate(value):
                    raise error

                with self.assertRaisesRegex(RuntimeError, "after 1 attempts"):
                    LoggedClient(client, Path(tmp) / "calls.jsonl", 3).call([], purpose="probe", validate=validate)
                row = self.ledger(tmp)[0]
                self.assertFalse(row["will_retry"])
                self.assertEqual(row["failure_stage"], "validate")
                saved = json.loads((Path(tmp) / row["diagnostic_path"]).read_text())
                self.assertEqual(saved["content"], content)
                self.assertEqual(saved["metadata"]["error_type"], type(error).__name__)
                self.assertEqual(len(client.requests), 1)
                sleep.assert_not_called()

    def test_authentication_credit_and_policy_refusals_never_retry(self):
        cases = [ServiceError(status) for status in (401, 402, 403, 404)]
        cases += [ServiceError(200, "content_filter"), ServiceError(500, "content_policy_violation"),
                  ServiceError(524, "ResponsibleAIPolicyViolation"), ServiceError(429, "insufficient_quota")]
        for error in cases:
            error.retryable = True  # A misleading transport flag must not override a known refusal.
            with self.subTest(error=str(error)), tempfile.TemporaryDirectory() as tmp, \
                 patch("methods.longemo.common.time.sleep") as sleep:
                client = SequenceClient([error, response('{"ok": true}')])
                with self.assertRaisesRegex(RuntimeError, "after 1 attempts"):
                    LoggedClient(client, Path(tmp) / "calls.jsonl", 3).call([], purpose="probe", validate=validate_ok)
                self.assertEqual(len(client.requests), 1)
                self.assertFalse(self.ledger(tmp)[0]["will_retry"])
                sleep.assert_not_called()

    def test_unknown_failures_and_empty_responses_do_not_retry(self):
        unknown = RuntimeError("opaque provider error")
        unknown.retryable = True  # A flag alone is not evidence of a temporary error.
        cases = [unknown, ValueError("local request configuration error"), URLError("unknown transport issue"),
                 response(""), response(" \n"), response(None), response([]), None]
        for value in cases:
            with self.subTest(value=str(value)), tempfile.TemporaryDirectory() as tmp, \
                 patch("methods.longemo.common.time.sleep") as sleep:
                client = SequenceClient([value, response('{"ok": true}')])
                with self.assertRaisesRegex(RuntimeError, "after 1 attempts"):
                    LoggedClient(client, Path(tmp) / "calls.jsonl", 3).call([], purpose="probe", validate=validate_ok)
                self.assertEqual(len(client.requests), 1)
                self.assertFalse(self.ledger(tmp)[0]["will_retry"])
                sleep.assert_not_called()

    def test_explicit_temporary_failures_retry_within_the_original_attempt_limit(self):
        cases = [ServiceError(429), ServiceError(503), ServiceError(524), TimeoutError("request timed out"),
                 ConnectionResetError("peer reset connection"), URLError(TimeoutError("request timed out"))]
        for error in cases:
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as tmp, \
                 patch("methods.longemo.common.time.sleep") as sleep:
                client = SequenceClient([error, response('{"ok": true}')])
                result = LoggedClient(client, Path(tmp) / "calls.jsonl", 3).call([], purpose="probe", validate=validate_ok)
                self.assertEqual(result, {"ok": True})
                self.assertEqual(len(client.requests), 2)
                self.assertTrue(self.ledger(tmp)[0]["will_retry"])
                sleep.assert_called_once()
        with tempfile.TemporaryDirectory() as tmp, patch("methods.longemo.common.time.sleep") as sleep:
            client = SequenceClient([TimeoutError("timeout")] * 3)
            with self.assertRaisesRegex(RuntimeError, "after 3 attempts"):
                LoggedClient(client, Path(tmp) / "calls.jsonl", 3).call([], purpose="probe", validate=validate_ok)
            self.assertEqual(len(client.requests), 3)
            self.assertEqual(sleep.call_count, 2)
            self.assertEqual([r["will_retry"] for r in self.ledger(tmp)], [True, True, False])

    def test_transport_exhaustion_veto_is_preserved(self):
        error = RuntimeError("transport exhausted its internal retry budget")
        error.status_code, error.retryable = 503, False
        with tempfile.TemporaryDirectory() as tmp, patch("methods.longemo.common.time.sleep") as sleep:
            client = SequenceClient([error])
            with self.assertRaisesRegex(RuntimeError, "after 1 attempts"):
                LoggedClient(client, Path(tmp) / "calls.jsonl", 3).call([], purpose="probe", validate=validate_ok)
            self.assertEqual(len(client.requests), 1)
            sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
