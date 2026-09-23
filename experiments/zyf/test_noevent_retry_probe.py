import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from experiments.zyf import noevent_retry_probe as probe
from evaluation.clients import Client
from evaluation.inference.adapters import openai
from methods.longemo.common import fingerprint
from methods.longemo import audio, common, noevent_runner as runner
from methods.longemo.noevent_memory import apply_window, empty_memory, perception_context


class Response(io.BytesIO):
    status = 200


class Tests(unittest.TestCase):
    def prepared(self):
        messages = [{"role": "user", "content": "unchanged original media request"}]
        failed = {"status": "error", "failure_stage": "generate", "attempt": 1,
                  "will_retry": False, "error_type": "RuntimeError", "request_hash": fingerprint(messages)}
        client = Client("gemini-3.8-flash", probe.MATRIX, "chat", "test-secret-key", 180, 8192)
        return {"messages": messages, "failed": failed, "fingerprint": fingerprint,
                "client": client, "adapter": openai, "parse_json": json.loads,
                "validate": lambda p: self.assertEqual(p, {"observations": []}),
                "keys": [client.api_key], "protected": {},
                "artifact": {"video_id": "G2_V000031", "window_id": "W00012", "stage": "visual",
                             "context": {"owned_core": [220, 240]}, "sampling": {"audio": True}}}

    def valid_response(self):
        return {"model": "gemini-flash-returned", "choices": [{"finish_reason": "stop",
                "message": {"content": '{"observations": []}'}}]}

    def execute(self, prepared, raw=None, opener=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        folder = Path(temporary.name) / "probe"
        folder.mkdir(mode=0o700)
        opener = opener or Mock(return_value=Response(json.dumps(raw or self.valid_response()).encode()))
        result = probe.execute_once(prepared, folder, opener)
        return result, folder, opener

    def test_one_request_validated_payload_is_private_not_a_formal_memory(self):
        prepared = self.prepared()
        result, folder, opener = self.execute(prepared)
        self.assertEqual(result["status"], "validated")
        self.assertTrue(result["parent_files_unchanged"])
        self.assertEqual(opener.call_count, 1)
        req = opener.call_args.args[0]
        self.assertEqual(req.full_url, probe.MATRIX + "/chat/completions")
        self.assertEqual(json.loads(req.data), prepared["client"].payload(prepared["messages"]))
        artifact = json.loads((folder / "validated_payload.json").read_text())
        self.assertEqual(artifact["request_hash"], prepared["failed"]["request_hash"])
        self.assertEqual(artifact["payload"], {"observations": []})
        self.assertFalse(artifact["official_result"])
        self.assertFalse((folder / "memory.json").exists())
        for path in folder.iterdir():
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_hash_change_refuses_before_network(self):
        prepared = self.prepared()
        prepared["messages"][0]["content"] = "changed"
        opener = Mock()
        with self.assertRaises(ValueError):
            self.execute(prepared, opener=opener)
        opener.assert_not_called()

    def test_prior_refusal_refuses_before_network(self):
        for code in ("content_filter", "content_policy_violation", "SAFETY"):
            with self.subTest(code=code):
                prepared = self.prepared()
                prepared["failed"]["service_error_code"] = code
                opener = Mock()
                with self.assertRaises(ValueError):
                    self.execute(prepared, opener=opener)
                opener.assert_not_called()

    def test_changed_parent_refuses_before_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "parent.json"
            path.write_text("old")
            prepared = self.prepared()
            prepared["protected"] = {str(path): probe.sha(path)}
            path.write_text("changed")
            opener = Mock()
            with self.assertRaises(ValueError):
                self.execute(prepared, opener=opener)
            opener.assert_not_called()

    def test_duplicate_output_and_sibling_guard_prevent_dispatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "probe"
            with patch.object(probe, "prepare", return_value=self.prepared()):
                opener = Mock(return_value=Response(json.dumps(self.valid_response()).encode()))
                probe.run_probe("/parent/run", "G2_V000031", output, "/repo", opener)
                with self.assertRaises(FileExistsError):
                    probe.run_probe("/parent/run", "G2_V000031", output, "/repo", opener)
                with self.assertRaises(FileExistsError):
                    probe.run_probe("/parent/run", "G2_V000031", Path(temporary) / "different", "/repo", opener)
                self.assertEqual(opener.call_count, 1)
                self.assertEqual(output.stat().st_mode & 0o777, 0o700)

    def test_output_cannot_modify_original_run_or_checkout(self):
        opener = Mock()
        for output in ("/parent/run/probe", "/repo/probe"):
            with self.subTest(output=output), self.assertRaises(ValueError):
                probe.run_probe("/parent/run", "V", output, "/repo", opener)
        opener.assert_not_called()

    def test_parser_failure_keeps_original_response_evidence_without_retry(self):
        raw = self.valid_response()
        raw["choices"][0]["finish_reason"] = "length"
        result, folder, opener = self.execute(self.prepared(), raw)
        self.assertEqual(result["status"], "parse_error")
        self.assertEqual(result["finish_reason"], "length")
        self.assertEqual(result["error_type"], "RuntimeError")
        self.assertEqual(opener.call_count, 1)
        self.assertIn('"length"', json.loads((folder / "response.json").read_text())["body_text"])

    def test_http428_keeps_redacted_body_and_never_retries(self):
        body = b'{"error":{"message":"private test-secret-key","code":"PRECONDITION"}}'
        failure = HTTPError("https://example.invalid", 428, "opaque", {}, io.BytesIO(body))
        result, folder, opener = self.execute(self.prepared(), opener=Mock(side_effect=failure))
        self.assertEqual(result["status"], "http_error")
        self.assertEqual(result["http_status"], 428)
        self.assertEqual(result["safe_code"], "PRECONDITION")
        self.assertEqual(opener.call_count, 1)
        evidence = json.loads((folder / "response.json").read_text())
        self.assertTrue(evidence["body_redacted"])
        self.assertNotIn("test-secret-key", evidence["body_text"])
        self.assertNotIn("private", json.dumps(result))

    def test_new_refusal_never_becomes_success(self):
        raw = self.valid_response()
        raw["choices"][0]["message"]["refusal"] = "private refusal text"
        result, folder, opener = self.execute(self.prepared(), raw)
        self.assertEqual(result["status"], "content_filter")
        self.assertEqual(opener.call_count, 1)
        self.assertFalse((folder / "validated_payload.json").exists())
        self.assertNotIn("private refusal", json.dumps(result))

    def test_timeout_is_one_attempt_and_guard_remains(self):
        result, folder, opener = self.execute(self.prepared(), opener=Mock(side_effect=TimeoutError()))
        self.assertEqual(result["status"], "transport_error")
        self.assertEqual(opener.call_count, 1)
        self.assertTrue((folder / "request_started.json").exists())
        self.assertTrue((folder / "summary.json").exists())

    def test_validation_failure_stays_private_and_not_reusable(self):
        prepared = self.prepared()
        prepared["validate"] = Mock(side_effect=ValueError("private semantic validation failure"))
        result, folder, opener = self.execute(prepared)
        self.assertEqual(result["status"], "validation_error")
        self.assertEqual(result["validate_status"], "error")
        self.assertNotIn("private semantic", json.dumps(result))
        self.assertTrue((folder / "validation.json").exists())
        self.assertFalse((folder / "validated_payload.json").exists())
        self.assertEqual(opener.call_count, 1)

    def test_unknown_eligibility_is_narrow(self):
        good = self.prepared()["failed"]
        probe.check_eligible(good)
        probe.check_eligible({**good, "http_status": 428, "error_type": "ServiceError"})
        for code in (None, "", "428"):
            probe.check_eligible({**good, "http_status": 428, "error_type": "ServiceError", "service_error_code": code})
        for change in ({"attempt": 2}, {"failure_stage": "validate"}, {"will_retry": True},
                       {"http_status": 524}, {"http_status": 400}, {"service_error_code": "KnownOtherCode"},
                       {"http_status": 428, "service_error_code": "PRECONDITION"},
                       {"http_status": 428, "service_error_code": "content_filter"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                probe.check_eligible({**good, **change})

    def test_real_numeric_428_code_from_parent_ledger_is_eligible(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            row = {**self.prepared()["failed"], "http_status": 428, "error_type": "ServiceError",
                   "service_error_code": "428", "time_unix": 100, "purpose": "window_perception:V:W00002"}
            (folder / "calls.jsonl").write_text(json.dumps(row) + "\n")
            stage, _, selected = probe.select_failure(folder, "V", "W00002")
            self.assertEqual(stage, "visual")
            self.assertEqual(selected, row)

    def test_terminal_audio_failure_selects_audio_and_rejects_wrong_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "audio").mkdir()
            visual = {"status": "ok", "time_unix": 1, "purpose": "window_perception:V:W00001"}
            audio = {**self.prepared()["failed"], "time_unix": 2, "purpose": "audio_observer:W00002"}
            (folder / "calls.jsonl").write_text(json.dumps(visual) + "\n")
            (folder / "audio/calls.jsonl").write_text(json.dumps(audio) + "\n")
            stage, _, row = probe.select_failure(folder, "V", "W00002")
            self.assertEqual(stage, "audio")
            self.assertEqual(row, audio)
            with self.assertRaises(ValueError):
                probe.select_failure(folder, "V", "W00003")

    def fixture(self, root, stage):
        repo = Path(runner.__file__).resolve().parents[2]
        parent = root / "run"
        folder = parent / "noevent/memory/V"
        folder.mkdir(parents=True)
        (folder / "audio").mkdir()
        media_dir = root / "media"
        media_dir.mkdir()
        video, subtitle = media_dir / "V.mp4", media_dir / "V.json"
        video.write_bytes(b"fixture video")
        subtitle.write_text("[]")
        credential = root / "private.json"
        credential.write_text('{"MODEL_API_KEY":"fixture-key"}')
        argv = ["build", "--data-path", str(parent / "questions.json"), "--videos-dir", str(media_dir),
                "--subtitles-dir", str(media_dir), "--output-dir", str(folder.parent),
                "--credential-file", str(credential), "--with-audio", "--model", "gemini-3.8-flash",
                "--base-url", probe.MATRIX, "--audio-model", "gemini-3.8-flash", "--audio-base-url", probe.MATRIX,
                "--window-seconds", "20", "--padding", "2"]
        args = runner.parser().parse_args(argv)
        visual, audio_client = runner.client_for(args), runner.audio_client_for(args)
        config = {"representation": "window_records", "video_sha256": probe.sha(video),
                  "subtitles_sha256": probe.sha(subtitle), "model": visual.configuration(),
                  "audio_observer": audio_client.configuration(), "window_seconds": args.window_seconds,
                  "padding": args.padding, "media": runner._media_options(args), "code_hash": "frozen", "git_revision": "old"}
        build_hash = fingerprint(config)
        manifest = {"configuration": config, "fingerprint": build_hash}
        memory = empty_memory("V", 40.0, probe.sha(video))
        payload = {"entities": [], "observations": [], "summary": "An ordinary room.", "emotion_cues": []}
        memory = apply_window(memory, payload, window_id="W00001", core=[0.0, 20.0], media=[0, 22.0], metadata={"audio": True})
        memory["build_fingerprint"] = build_hash
        (folder / "manifest.json").write_text(json.dumps(manifest))
        (folder / "memory.json").write_text(json.dumps(memory))
        cohort = {"repos": {"noevent": str(repo)}, "source_hashes": {"noevent": "frozen"},
                  "media": {"V": {"video_sha256": probe.sha(video), "subtitles_sha256": probe.sha(subtitle),
                                   "video_bytes": video.stat().st_size, "subtitles_bytes": subtitle.stat().st_size}},
                  "perception_model": "gemini-3.8-flash", "audio_model": "gemini-3.8-flash"}
        (parent / "configuration.json").write_text(json.dumps(cohort))
        task_dir = parent / "tasks/build/noevent-V"
        task_dir.mkdir(parents=True)
        (task_dir / "task.json").write_text(json.dumps({"status": "error", "cwd": str(repo),
            "command": ["python", "-u", "-m", "methods.longemo.noevent_runner"] + argv}))
        content = [{"type": "input_audio", "input_audio": {"format": "wav", "data": "AAAA"}}]
        interval, core = [18.0, 40.0], [20.0, 40.0]
        audio_messages = [{"role": "system", "content": audio.AUDIO_PROMPT}, {"role": "user", "content": [
            {"type": "text", "text": json.dumps({"clip_start": interval[0], "clip_end": interval[1]})}] + content}]
        if stage == "audio":
            messages = audio_messages
            ledger = folder / "audio/calls.jsonl"
            purpose = "audio_observer:W00002"
        else:
            result = {"observations": []}
            cache = {"input_fingerprint": fingerprint({"messages": audio_messages, "model": audio_client.configuration()}),
                     "model": audio_client.configuration(), "result": result}
            (folder / "audio/W00002.json").write_text(json.dumps(cache))
            replacement = {"type": "text", "text": "Timestamped audio observations from an independent audio model; "
                "these may contain errors. Match voice identity cautiously using the frames and dialogue. " + json.dumps(result, ensure_ascii=False)}
            context = perception_context(memory, window_id="W00002", core=core, media=interval)
            messages = [{"role": "system", "content": runner.PERCEPTION_NOEVENT}, {"role": "user", "content": [
                {"type": "text", "text": json.dumps(context, ensure_ascii=False)}, replacement]}]
            ledger = folder / "calls.jsonl"
            purpose = "window_perception:V:W00002"
        ledger.write_text(json.dumps({"status": "error", "attempt": 1, "will_retry": False,
            "failure_stage": "generate", "error_type": "RuntimeError", "time_unix": 100,
            "request_hash": fingerprint(messages), "purpose": purpose}) + "\n")
        return parent, folder, repo, content

    def test_rebuild_audio_and_visual_requests_exactly_without_requesting_audio(self):
        for stage in ("audio", "visual"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as temporary:
                parent, folder, repo, content = self.fixture(Path(temporary), stage)
                before = {str(p): probe.sha(p) for p in parent.rglob("*") if p.is_file()}
                with patch.object(common, "code_hash", return_value="frozen"), \
                     patch.object(runner, "probe", return_value={"duration": 40.0}), \
                     patch.object(runner, "window_input", return_value=(content, {"audio": True, "subtitle_ids": []})), \
                     patch.object(Client, "generate", side_effect=AssertionError("must not call generate")):
                    prepared = probe.prepare(parent, "V", repo)
                    self.assertEqual(prepared["artifact"]["stage"], stage)
                    self.assertEqual(fingerprint(prepared["messages"]), prepared["failed"]["request_hash"])
                self.assertEqual(before, {str(p): probe.sha(p) for p in parent.rglob("*") if p.is_file()})

    def test_visual_missing_pending_audio_is_rejected_without_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent, folder, repo, content = self.fixture(Path(temporary), "visual")
            (folder / "audio/W00002.json").unlink()
            with patch.object(common, "code_hash", return_value="frozen"), \
                 patch.object(runner, "probe", return_value={"duration": 40.0}), \
                 patch.object(runner, "window_input", return_value=(content, {"audio": True, "subtitle_ids": []})), \
                 patch.object(Client, "generate") as generate:
                with self.assertRaises(ValueError):
                    probe.prepare(parent, "V", repo)
                generate.assert_not_called()

    def test_frozen_media_byte_count_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent, folder, repo, content = self.fixture(Path(temporary), "audio")
            cfg_path = parent / "configuration.json"
            config = json.loads(cfg_path.read_text())
            config["media"]["V"]["video_bytes"] += 1
            cfg_path.write_text(json.dumps(config))
            with patch.object(common, "code_hash", return_value="frozen"), \
                 patch.object(runner, "probe", return_value={"duration": 40.0}), \
                 patch.object(Client, "generate") as generate:
                with self.assertRaisesRegex(ValueError, "media differs"):
                    probe.prepare(parent, "V", repo)
                generate.assert_not_called()

    def test_allowlisted_error_metadata_also_redacts_credentials(self):
        result, _, _ = self.execute(self.prepared(), {"error": {"code": "test-secret-key"}, "model": "test-secret-key"})
        self.assertNotIn("test-secret-key", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
