"""Offline seed import, real audio-cache reuse, and first-score guards."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evaluation.clients import Client
from evaluation.inference.adapters import openai
from evaluation.inference.prompts import json_object
from experiments.zyf import noevent_continuation as base
from experiments.zyf import noevent_v101_continuation as continuation
from experiments.zyf import test_noevent_continuation as fixtures
from experiments.zyf.matched_pilot import digest, memory_root, read, sha, write
from methods.longemo import audio
from methods.longemo.common import fingerprint
from methods.longemo.noevent_memory import apply_window


class V101ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.NoeventContinuationTests()
        self.f.setUp()
        self.addCleanup(self.f.tearDown)
        self.client = Client("gemini-3.8-flash", "https://matrixllm.alipay.com/v1", "chat", "fixture-secret", 180, 4096,
                             None, {"reasoning_effort": "low"})
        self.f.args.visual_max_tokens = 8192
        self.f.questions.append(self.f.question("Q2", "V2"))
        self.f.config["questions_sha256"] = digest(self.f.questions)
        write(self.f.parent / "configuration.json", self.f.config)
        write(self.f.parent / "questions.json", self.f.questions)
        manifest = read(self.f.source / "manifest.json")
        manifest["configuration"]["audio_observer"] = self.client.configuration()
        manifest["fingerprint"] = digest(manifest["configuration"])
        write(self.f.source / "manifest.json", manifest)
        memory = read(self.f.source / "memory.json")
        memory.update(duration=60, build_fingerprint=manifest["fingerprint"])
        write(self.f.source / "memory.json", memory)
        self.source_config = self.f.prepared()
        self.source = self.f.run
        self.folder = memory_root(self.source, "noevent", "V1") / "V1"
        write(self.folder / "manifest.json", manifest)
        payload = {"entities": [], "observations": [], "summary": "The person stands up."}
        sampling = {"audio": False, "subtitle_ids": []}
        memory = apply_window(read(self.folder / "memory.json"), payload, window_id="W00002", core=[20, 40], media=[18, 42], metadata=sampling)
        memory.update(complete=False, build_fingerprint=manifest["fingerprint"])
        write(self.folder / "memory.json", memory)
        write(self.folder / "windows/W00002.json", {"input": {"video_id": "V1", "window_id": "W00002",
            "core_interval": [20, 40], "media_interval": [18, 42], "cast": []}, "sampling": sampling, "perception": payload})
        self.f.jsonl(self.folder / "audio/calls.jsonl", [{"purpose": "audio_observer:W00003", "status": "error",
                                                       "http_status": 500, "attempt": 3}])
        write(self.source / "tasks/build/noevent-V1/task.json", {"status": "error", "returncode": 1,
            "child": {"pid": 987654329, "start_ticks": "1", "host": socket.gethostname()}})
        write(self.source / "pipelines/noevent/V1.json", {"status": "build_failed"})
        write(self.source / "process.json", {"pid": 987654328, "start_ticks": "1", "host": socket.gethostname()})
        checkpoint = base.checkpoint_receipt(self.source, self.f.repo, self.f.config, "V1")
        self.content = [{"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}}]
        self.interval = [38, 60]
        messages = [{"role": "system", "content": audio.AUDIO_PROMPT}, {"role": "user", "content": [
            {"type": "text", "text": json.dumps({"clip_start": 38, "clip_end": 60})}] + self.content}]
        artifact = {"video_id": "V1", "window_id": "W00003", "stage": "audio", "source_run": str(self.source),
            "audio_input_fingerprint": fingerprint({"messages": messages, "model": self.client.configuration()}),
            "model_configuration": self.client.configuration(), "parent_memory_sha256": sha(self.folder / "memory.json"),
            "parent_manifest_sha256": sha(self.folder / "manifest.json")}
        self.prepared = {"checkpoint": checkpoint, "artifact": artifact, "failed": {"request_hash": continuation.diagnostic.FINAL_REQUEST},
            "messages": messages, "client": self.client, "adapter": openai, "parse_json": json_object,
            "validate": lambda value: self.assertEqual(value, {"observations": []}), "protected": {},
            "media_identity": base.verified_media(self.f.config, ["V1"])}
        self.probe = self.f.root / "successful-probe"
        self.probe.mkdir()
        raw = {"choices": [{"finish_reason": "stop", "message": {"content": '{"observations":[]}'}}]}
        body = json.dumps(raw)
        body_sha = hashlib.sha256(body.encode()).hexdigest()
        write(self.probe / "response.json", {"request_hash": continuation.diagnostic.FINAL_REQUEST, "http_status": 200,
              "body_redacted": False, "body_sha256": body_sha, "body_text": body})
        wire = json.dumps(openai.build_payload(self.client, messages), allow_nan=False).encode()
        write(self.probe / "request_started.json", {"request_hash": continuation.diagnostic.FINAL_REQUEST,
              "maximum_http_requests": 1, "wire_sha256": hashlib.sha256(wire).hexdigest()})
        write(self.probe / "validated_payload.json", {**artifact, "request_hash": continuation.diagnostic.FINAL_REQUEST,
              "payload": {"observations": []}, "response_sha256": body_sha, "official_result": False})
        write(self.probe / "summary.json", {"status": "validated", "reusable": True, "parent_files_unchanged": True,
              "official_result": False, "parse_status": "ok", "validate_status": "ok", "http_status": 200,
              "request_hash": continuation.diagnostic.FINAL_REQUEST, "maximum_http_requests": 1,
              "artifact_sha256": sha(self.probe / "validated_payload.json")})
        first = self.f.parent / "accepted/noevent/Q0.json"
        original_score = read(first)["source"]
        write(self.probe / "preflight.json", {"status": "preflight_passed", "no_api_calls": True,
              "video_id": "V1", "window_id": "W00003", "request_hash": continuation.diagnostic.FINAL_REQUEST,
              "completed_windows": 2, "protected_score_count": 1, "artifact": {"source_run": str(self.source)},
              "protected_files": {str(first): sha(first), original_score: sha(original_score)}})
        self.related = self.f.root / "sibling"
        self.related.mkdir()
        self.run = self.f.root / "runtime/runs/audio-seed"
        self.args = SimpleNamespace(run=str(self.run), source_run=str(self.source), core_repo=str(self.f.repo),
            probe_dir=str(self.probe), selection=str(self.f.selection), credential_file=str(self.f.credentials),
            related_run=[str(self.related)], python=sys.executable)
        for name, value in (("VIDEO", "V1"), ("WINDOW", "W00003"), ("COMPLETED_WINDOWS", 2),
                            ("PROBE_FIRST_SCORES", 1), ("EXPECTED_QIDS", ["Q1"])):
            handle = patch.object(continuation, name, value)
            handle.start()
            self.addCleanup(handle.stop)
        handle = patch.object(continuation.diagnostic, "prepare", return_value=self.prepared)
        handle.start()
        self.addCleanup(handle.stop)

    def add_sibling_score(self):
        row = {"question_id": "Q2", "video_id": "V2", "status": "ok", "prediction": "first", "score": 1, "max_score": 1, "normalized_score": 1}
        source = self.related / "scores/first.jsonl"
        self.f.jsonl(source, [row])
        write(self.related / "accepted/noevent/Q2.json", {"score": row, "source": str(source)})

    def test_prepare_imports_audio_without_advancing_window_and_real_bridge_makes_no_call(self):
        before = base.inventory(self.folder)
        _, config = continuation.prepare(self.args)
        target = memory_root(self.run, "noevent", "V1") / "V1"
        self.assertEqual(read(target / "memory.json")["completed_windows"], ["W00001", "W00002"])
        self.assertEqual(base.inventory(self.folder), before)
        with patch.object(audio.LoggedClient, "call", side_effect=AssertionError("must reuse seeded audio")) as call:
            content, trace = audio.bridge_audio(self.content, self.interval, self.client, target / "audio/W00003.json")
        call.assert_not_called()
        self.assertEqual(trace["input_fingerprint"], self.prepared["artifact"]["audio_input_fingerprint"])
        self.assertEqual(config["profile"]["visual"]["max_tokens"], 8192)
        self.assertEqual(config["workers"], 1)
        self.assertEqual(config["question_workers"], 2)
        self.assertEqual(continuation.prepare(self.args)[1], config)

    def test_new_sibling_first_score_is_allowed_then_frozen_against_deletion(self):
        _, config = continuation.prepare(self.args)
        self.add_sibling_score()
        self.assertIn("Q2", continuation.verify(self.run, config))
        self.assertEqual(continuation.prepare(self.args)[1], config)
        (self.related / "accepted/noevent/Q2.json").unlink()
        with self.assertRaisesRegex(ValueError, "disappeared"):
            continuation.verify(self.run, config)

    def test_probe_original_score_and_audio_cache_cannot_change(self):
        _, config = continuation.prepare(self.args)
        path = self.f.parent / "accepted/noevent/Q0.json"
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        with self.assertRaises(ValueError):
            continuation.verify(self.run, config)
        path.write_bytes(original)
        target = memory_root(self.run, "noevent", "V1") / "V1/audio/W00003.json"
        value = read(target)
        value["input_fingerprint"] = "f" * 64
        write(target, value)
        with self.assertRaisesRegex(ValueError, "seed output changed"):
            continuation.verify(self.run, config)

    def test_task_reentry_and_second_owner_never_start_a_worker(self):
        _, config = continuation.prepare(self.args)
        command = base.stage_command(self.run, config, "build", "V1")
        write(self.run / "tasks/build/noevent-V1/task.json", {"status": "error", "signature": digest({"command": command, "cwd": config["core_repo"]})})
        with patch.object(base.pilot.subprocess, "Popen") as spawn:
            self.assertEqual(base.run_stage(self.run, config, "build", "V1", {})["status"], "error")
        spawn.assert_not_called()
        other = copy.copy(self.args)
        other.run = str(self.f.root / "runtime/runs/other-owner")
        with self.assertRaisesRegex(ValueError, "already owns"):
            continuation.prepare(other)

    def test_wrong_probe_or_selection_is_rejected_before_seed(self):
        value = read(self.probe / "validated_payload.json")
        value["model_configuration"]["max_tokens"] = 8192
        write(self.probe / "validated_payload.json", value)
        with self.assertRaisesRegex(ValueError, "validation receipt"):
            continuation.prepare(self.args)
        self.assertFalse(self.run.exists())


if __name__ == "__main__":
    unittest.main()
