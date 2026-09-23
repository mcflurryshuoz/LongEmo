"""Offline guards for a single bounded visual length recovery."""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.zyf import noevent_continuation as base
from experiments.zyf import noevent_seed_length_continuation as length
from experiments.zyf import test_noevent_continuation as fixtures
from experiments.zyf.matched_pilot import digest, memory_root, read, sha, write
from methods.longemo.noevent_memory import apply_window


class SeedLengthContinuationTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.NoeventContinuationTests()
        self.f.setUp()
        self.addCleanup(self.f.tearDown)
        self.f.args.visual_max_tokens = 8192
        memory = read(self.f.source / "memory.json")
        memory["duration"] = 60
        write(self.f.source / "memory.json", memory)
        self.f.questions.append(self.f.question("Q2", "V2"))
        self.f.config["questions_sha256"] = digest(self.f.questions)
        write(self.f.parent / "questions.json", self.f.questions)
        write(self.f.parent / "configuration.json", self.f.config)
        self.source_config = self.f.prepared()
        self.source = self.f.run
        self.folder = memory_root(self.source, "noevent", "V1") / "V1"
        manifest = read(self.folder / "ancestry/parent/manifest.json")
        write(self.folder / "manifest.json", manifest)
        memory = read(self.folder / "memory.json")
        payload = {"entities": [], "observations": [], "summary": "The person stands up."}
        context = {"video_id": "V1", "window_id": "W00002", "core_interval": [20, 40],
                   "media_interval": [18, 42], "cast": []}
        sampling = {"audio": False, "subtitle_ids": []}
        memory = apply_window(memory, payload, window_id="W00002", core=[20, 40], media=[18, 42], metadata=sampling)
        memory.update(build_fingerprint=manifest["fingerprint"], complete=False)
        write(self.folder / "memory.json", memory)
        write(self.folder / "windows/W00002.json", {"input": context, "sampling": sampling, "perception": payload})
        self.request_hash = "b" * 64
        self.f.jsonl(self.folder / "calls.jsonl", [{"purpose": "window_perception:V1:W00003", "status": "error",
            "failure_stage": "generate", "error_type": "RuntimeError", "will_retry": False, "request_hash": self.request_hash}])
        write(self.source / "tasks/build/noevent-V1/task.json", {"status": "error", "returncode": 1,
            "child": {"pid": 987654329, "start_ticks": "1", "host": socket.gethostname()}})
        write(self.source / "pipelines/noevent/V1.json", {"status": "build_failed"})
        self.diagnostic = self.source / "private_diagnostics/build/V1/one.json"
        self.raw = {"choices": [{"finish_reason": "length", "message": {"content": "truncated"}}],
                    "usage": {"completion_tokens": 328, "reasoning_tokens": 7860}}
        self.write_diagnostic(self.raw)
        self.related = self.f.root / "other"
        self.related.mkdir()
        self.run = self.f.root / "runtime/runs/length"
        self.args = SimpleNamespace(run=str(self.run), source_run=str(self.source), baseline_run=str(self.f.parent),
            core_repo=str(self.f.repo), selection=str(self.f.selection), credential_file=str(self.f.credentials),
            related_run=[str(self.related)], workers=5, question_workers=2, python=sys.executable)
        source_patch = patch.object(length, "source_hash", return_value="frozen-core")
        source_patch.start()
        self.addCleanup(source_patch.stop)
        original = base.protected_scores
        protected_patch = patch.object(base, "protected_scores", side_effect=lambda p, q, n: original(p, q, 1))
        protected_patch.start()
        self.addCleanup(protected_patch.stop)

    def write_diagnostic(self, raw):
        body = json.dumps(raw)
        write(self.diagnostic, {"request_hash": self.request_hash, "outcome": "error", "error_type": "RuntimeError",
            "model_configuration": read(self.folder / "manifest.json")["configuration"]["model"],
            "responses": [{"http_status": 200, "body_text": body, "body_sha256": hashlib.sha256(body.encode()).hexdigest()}]})

    def receipt(self):
        return base.checkpoint_receipt(self.source, self.f.repo, self.f.config, "V1")

    def add_score(self, qid="Q2", vid="V2", value=1):
        row = {"question_id": qid, "video_id": vid, "status": "ok", "prediction": "first answer",
               "score": value, "max_score": 1, "normalized_score": value}
        source = self.related / "scores" / (qid + ".jsonl")
        self.f.jsonl(source, [row])
        write(self.related / "accepted/noevent" / (qid + ".json"), {"source": str(source), "score": row})

    def test_prepare_clones_latest_stopped_video_while_source_coordinator_lock_is_held(self):
        before = base.inventory(self.folder)
        with (self.source / "coordinator.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            _, config = length.prepare(self.args)
        self.assertEqual(config["checkpoints"]["V1"]["completed_windows"], 2)
        self.assertEqual(config["length_recovery"]["videos"]["V1"]["original_completed_windows"], 1)
        self.assertEqual(config["profile"]["visual"]["max_tokens"], 16384)
        self.assertEqual(config["profile"]["max_tokens"], 8192)
        self.assertEqual(base.inventory(self.folder), before)
        cloned = memory_root(self.run, "noevent", "V1") / "V1"
        self.assertEqual(sha(cloned / "windows/W00002.json"), before["windows/W00002.json"])
        self.add_score()
        self.assertEqual(length.prepare(self.args)[1], config)
        self.assertEqual(length.verify(self.run, config)["Q2"]["score"]["score"], 1)
        self.assertEqual(base.stage_command(self.run, config, "score", "V1").count("--tries"), 1)

    def test_exact_length_response_excludes_policy_refusal_and_altered_body(self):
        receipt = self.receipt()
        self.assertEqual(length.length_evidence(self.source, "V1", receipt)["finish_reason"], "length")
        for change in ({"error": {"code": "blocked"}}, {"promptFeedback": {"blockReason": "SAFETY"}}):
            self.write_diagnostic({**self.raw, **change})
            with self.assertRaisesRegex(ValueError, "exclusively"):
                length.length_evidence(self.source, "V1", receipt)
        raw = copy.deepcopy(self.raw)
        raw["choices"][0]["message"]["refusal"] = "blocked"
        self.write_diagnostic(raw)
        with self.assertRaisesRegex(ValueError, "exclusively"):
            length.length_evidence(self.source, "V1", receipt)
        self.write_diagnostic(self.raw)
        diagnostic = read(self.diagnostic)
        diagnostic["responses"][0]["body_text"] += " "
        write(self.diagnostic, diagnostic)
        with self.assertRaisesRegex(ValueError, "altered"):
            length.length_evidence(self.source, "V1", receipt)

    def test_newly_observed_score_cannot_later_disappear_or_change(self):
        _, config = length.prepare(self.args)
        self.add_score()
        length.verify(self.run, config)
        accepted = self.related / "accepted/noevent/Q2.json"
        original = accepted.read_bytes()
        accepted.unlink()
        with self.assertRaisesRegex(ValueError, "disappeared"):
            length.verify(self.run, config)
        accepted.write_bytes(original)
        self.add_score(value=0)
        with self.assertRaisesRegex(ValueError, "changed"):
            length.verify(self.run, config)

    def test_claim_exclusive_and_selected_score_overlap_rejected(self):
        path = self.f.root / "claim.json"
        length.claim_video(path, {"owner": "one"})
        length.claim_video(path, {"owner": "one"})
        with self.assertRaisesRegex(ValueError, "already owns"):
            length.claim_video(path, {"owner": "two"})
        self.add_score("Q1", "V1")
        with self.assertRaisesRegex(ValueError, "overlap"):
            length.prepare(self.args)

    def test_source_preparation_and_full_model_configuration_are_frozen(self):
        receipt = self.receipt()
        value = read(self.diagnostic)
        value["model_configuration"]["model"] = "different-model"
        write(self.diagnostic, value)
        with self.assertRaisesRegex(ValueError, "8192-token"):
            length.length_evidence(self.source, "V1", receipt)
        write(self.source / "prepared.json", {"configuration_sha256": "wrong"})
        with self.assertRaisesRegex(ValueError, "preparation receipt"):
            length.prepare(self.args)

    def test_live_or_ambiguous_child_and_backend_task_are_rejected(self):
        task = self.source / "tasks/build/noevent-V1/task.json"
        value = read(task)
        value["returncode"] = None
        write(task, value)
        with self.assertRaisesRegex(ValueError, "terminal"):
            length.terminal_video(self.source, "V1")
        value["returncode"] = 1
        write(task, value)
        with patch.object(length, "assert_identity_stopped", side_effect=ValueError("live child")):
            with self.assertRaisesRegex(ValueError, "live child"):
                length.terminal_video(self.source, "V1")
        write(self.source / "tasks/answer/noevent-V1/task.json", {"status": "error"})
        with self.assertRaisesRegex(ValueError, "backend work"):
            length.terminal_video(self.source, "V1")

    def test_frozen_seed_ancestor_is_guarded_without_touching_source(self):
        marker = self.f.root / "seed-ancestor.json"
        write(marker, {"original": True})
        ancestry = {"protected_files": {str(marker): sha(marker)}, "inherited_window_sources": {}}
        with patch.object(length, "audio_seed_ancestry", return_value=ancestry):
            _, config = length.prepare(self.args)
        before = base.inventory(self.folder)
        length.verify(self.run, config)
        self.assertEqual(base.inventory(self.folder), before)
        write(marker, {"original": False})
        with self.assertRaisesRegex(ValueError, "ancestry changed"):
            length.verify(self.run, config)

    def test_seed_source_requires_proven_receipt_before_length_prepare(self):
        config = copy.deepcopy(self.source_config)
        config["audio_seed_recovery"] = {"case": {"video": "V1", "window": "W00002"}}
        with self.assertRaises(ValueError):
            length.audio_seed_ancestry(self.source, config)


if __name__ == "__main__":
    unittest.main()
