"""Transparent response capture tests using in-memory HTTP responses only."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from evaluation import clients
from experiments.zyf import noevent_diagnostics_worker as worker


class Response(io.BytesIO):
    status = 200

    def __init__(self, body):
        super().__init__(body)
        self.read_count = 0

    def read(self, *args, **kwargs):
        self.read_count += 1
        return super().read(*args, **kwargs)


class PrivateDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name).resolve() / "private"
        self.client = clients.Client("gemini-3.8-flash", "https://example.invalid/v1", "chat", "fixture-key-123456", 180, 8192)
        self.messages = [{"role": "user", "content": "a window"}]
        self.original_json, self.original_generate = clients.json, clients.Client.generate

    def tearDown(self):
        self.assertIs(clients.json, self.original_json)
        self.assertIs(clients.Client.generate, self.original_generate)
        self.temporary.cleanup()

    def records(self):
        return [json.loads(p.read_text()) for p in self.folder.glob("*.json") if not p.name.endswith(".intent.json")]

    @staticmethod
    def body(content="ok", finish="stop"):
        return json.dumps({"model": "gemini-3.8-flash", "choices": [{"finish_reason": finish, "message": {"content": content}}]}).encode()

    def test_identical_wire_request_result_and_single_response_read(self):
        response = Response(self.body())
        observed = []

        def opener(req, **kwargs):
            observed.append((req, kwargs))
            return response

        expected = json.dumps(self.client.payload(self.messages), allow_nan=False).encode()
        with patch.object(clients.request, "urlopen", side_effect=opener), redirect_stdout(io.StringIO()):
            with worker.capture_clients(clients, self.folder):
                result = self.client.generate(self.messages)
        self.assertEqual(result["content"], "ok")
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0][0].data, expected)
        self.assertEqual(observed[0][0].get_header("Authorization"), "Bearer " + self.client.api_key)
        self.assertEqual(observed[0][1], {"timeout": 180})
        self.assertEqual(response.read_count, 1)
        record = self.records()[0]
        self.assertEqual(record["request_hash"], worker.fingerprint(self.messages))
        self.assertEqual(record["serialized_requests"], [{"payload_sha256": hashlib.sha256(expected).hexdigest(), "bytes": len(expected)}])
        self.assertEqual(record["responses"][0]["body_sha256"], hashlib.sha256(self.body()).hexdigest())
        self.assertEqual(record["responses"][0]["metadata"]["finish_reasons"], ["stop"])
        self.assertEqual(stat.S_IMODE(self.folder.stat().st_mode), 0o700)
        self.assertTrue(all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in self.folder.iterdir()))

    def test_original_length_runtime_error_now_has_matched_raw_evidence(self):
        with patch.object(clients.request, "urlopen", return_value=Response(self.body("cut off", "length"))), redirect_stdout(io.StringIO()):
            with worker.capture_clients(clients, self.folder):
                with self.assertRaisesRegex(RuntimeError, "output exceeded the token limit"):
                    self.client.generate(self.messages)
        record = self.records()[0]
        self.assertEqual(record["error_type"], "RuntimeError")
        self.assertEqual(record["responses"][0]["http_status"], 200)
        self.assertEqual(record["responses"][0]["metadata"]["finish_reasons"], ["length"])
        self.assertIn("cut off", record["responses"][0]["body_text"])

    def test_http_error_body_is_captured_without_changing_service_exception(self):
        body = json.dumps({"error": {"code": "policy_check", "message": "Data leak protection rejected"}}).encode()
        stream = Response(body)
        failure = HTTPError(self.client._url(), 428, "Precondition", {}, stream)
        with patch.object(clients.request, "urlopen", side_effect=failure) as opened, redirect_stdout(io.StringIO()):
            with worker.capture_clients(clients, self.folder):
                with self.assertRaises(clients.ServiceError) as caught:
                    self.client.generate(self.messages)
            opened.assert_called_once()
        self.assertEqual((caught.exception.status_code, caught.exception.code), (428, "policy_check"))
        record = self.records()[0]
        self.assertEqual(record["responses"][0]["http_status"], 428)
        self.assertIn("Data leak protection rejected", record["responses"][0]["body_text"])
        self.assertEqual(stream.read_count, 1)
        failure.close()

    def test_non_json_body_and_transport_exception_are_not_reclassified(self):
        with patch.object(clients.request, "urlopen", return_value=Response(b"<html>gateway failed</html>")), redirect_stdout(io.StringIO()):
            with worker.capture_clients(clients, self.folder):
                with self.assertRaises(json.JSONDecodeError):
                    self.client.generate(self.messages)
        record = self.records()[0]
        self.assertFalse(record["responses"][0]["json_parsed"])
        self.assertEqual(record["responses"][0]["body_text"], "<html>gateway failed</html>")
        failure = TimeoutError("test timeout")
        with patch.object(clients.request, "urlopen", side_effect=failure) as opened, redirect_stdout(io.StringIO()):
            with worker.capture_clients(clients, self.folder):
                with self.assertRaises(TimeoutError) as caught:
                    self.client.generate(self.messages)
            opened.assert_called_once()
        self.assertIs(caught.exception, failure)

    def test_private_body_redaction_does_not_change_the_returned_model_content(self):
        value = self.client.api_key + " Bearer other-token-987654 " + "sk-1234567890abcdefghijklmn"
        raw = self.body(value)
        with patch.object(clients.request, "urlopen", return_value=Response(raw)), redirect_stdout(io.StringIO()):
            with worker.capture_clients(clients, self.folder):
                result = self.client.generate(self.messages)
        self.assertEqual(result["content"], value)
        for path in self.folder.iterdir():
            text = path.read_text()
            self.assertNotIn(self.client.api_key, text)
            self.assertNotIn("other-token-987654", text)
            self.assertNotIn("sk-1234567890abcdefghijklmn", text)
        self.assertEqual(self.records()[0]["responses"][0]["body_sha256"], hashlib.sha256(raw).hexdigest())

    def test_thread_local_calls_do_not_exchange_bodies_or_fingerprints(self):
        def opener(req, **kwargs):
            payload = json.loads(req.data)
            return Response(self.body(payload["messages"][0]["content"]))

        messages = [[{"role": "user", "content": value}] for value in ("first", "second")]
        with patch.object(clients.request, "urlopen", side_effect=opener) as opened, redirect_stdout(io.StringIO()):
            with worker.capture_clients(clients, self.folder):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    results = list(pool.map(self.client.generate, messages))
            self.assertEqual(opened.call_count, 2)
        self.assertEqual([value["content"] for value in results], ["first", "second"])
        by_hash = {record["request_hash"]: record for record in self.records()}
        for message in messages:
            body = json.loads(by_hash[worker.fingerprint(message)]["responses"][0]["body_text"])
            self.assertEqual(body["choices"][0]["message"]["content"], message[0]["content"])

    def test_post_request_disk_failure_does_not_trigger_an_extra_model_attempt(self):
        original = worker.private_write

        def write(path, value, keys):
            if not path.name.endswith(".intent.json"):
                raise OSError("disk unavailable")
            return original(path, value, keys)

        with patch.object(worker, "private_write", side_effect=write), patch.object(clients.request, "urlopen", return_value=Response(self.body())) as opened:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with worker.capture_clients(clients, self.folder):
                    result = self.client.generate(self.messages)
            opened.assert_called_once()
        self.assertEqual(result["content"], "ok")
        self.assertEqual(len(list(self.folder.glob("*.intent.json"))), 1)


if __name__ == "__main__":
    unittest.main()
