"""Offline continuation checks; no API client or remote connection is used."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.zyf import noevent_continuation as continuation
from experiments.zyf.matched_pilot import digest, memory_root, read, sha, write
from methods.longemo.noevent_memory import apply_window, empty_memory


class NoeventContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.parent = self.root / "runtime/runs/parent"
        self.run = self.root / "runtime/runs/new-noevent"
        self.repo = Path(__file__).resolve().parents[2]
        self.video = "V1"
        self.questions = [self.question("Q0", "V0"), self.question("Q1", "V1")]
        self.video_dir = self.root / "runtime/data/videos"
        self.sub_dir = self.root / "runtime/data/subtitles"
        self.video_dir.mkdir(parents=True)
        self.sub_dir.mkdir(parents=True)
        (self.video_dir / "V1.mp4").write_bytes(b"fixture-video-not-decoded")
        write(self.sub_dir / "V1.json", [])
        self.config = {
            "mode": "full", "questions_sha256": digest(self.questions), "repos": {"noevent": str(self.repo)},
            "source_hashes": {"noevent": "frozen-core"}, "media": {"V1": {
                "video_sha256": sha(self.video_dir / "V1.mp4"), "subtitles_sha256": sha(self.sub_dir / "V1.json")}},
            "model": "gpt-6-astra", "base_url": "https://example.invalid/v1", "timeout": 1800,
            "max_tokens": 8192, "perception_model": "gemini-3.8-flash", "perception_base_url": "https://example.invalid/v1",
            "perception_timeout": 1800, "audio_model": "gemini-3.8-flash", "audio_base_url": "https://example.invalid/v1",
            "embedding_model": "google/gemini-embedding-2", "embedding_base_url": "https://example.invalid/embedding",
            "window_seconds": 20, "padding": 2, "fps": 1, "max_frames": 24, "max_pixels": 200704,
            "evidence_chars": 48000, "stage_tries": {"build": 3, "plan": 3, "answer": 3, "score": 1},
            "videos_dir": str(self.video_dir), "subtitles_dir": str(self.sub_dir),
        }
        write(self.parent / "configuration.json", self.config)
        write(self.parent / "questions.json", self.questions)
        write(self.parent / "process.json", {"pid": 987654321, "start_ticks": "1", "host": socket.gethostname()})
        row = {"question_id": "Q0", "video_id": "V0", "status": "ok", "prediction": "first answer",
               "max_score": 1, "score": 1, "normalized_score": 1}
        source = self.parent / "scores/noevent/V0/run_first/scores.jsonl"
        self.jsonl(source, [row])
        write(self.parent / "accepted/noevent/Q0.json", {"score": row, "source": str(source),
                                                       "prediction_sha256": digest("first answer")})
        self.source = memory_root(self.parent, "noevent", "V1") / "V1"
        model = {"model": "gemini-3.8-flash", "max_tokens": 8192}
        observer = {"model": "gemini-3.8-flash", "max_tokens": 4096}
        build_config = {"representation": "window_records", **self.config["media"]["V1"],
                        "model": model, "audio_observer": observer, "window_seconds": 20, "padding": 2,
                        "media": {"fps": 1, "max_frames": 24, "max_pixels": 200704, "with_audio": True},
                        "code_hash": "frozen-core", "git_revision": "test"}
        write(self.source / "manifest.json", {"configuration": build_config, "fingerprint": digest(build_config)})
        memory = empty_memory("V1", 40, self.config["media"]["V1"]["video_sha256"])
        payload = {"entities": [], "observations": [], "summary": "A person is sitting."}
        context = {"video_id": "V1", "window_id": "W00001", "core_interval": [0, 20], "media_interval": [0, 22], "cast": []}
        metadata = {"audio": False, "subtitle_ids": []}
        memory = apply_window(memory, payload, window_id="W00001", core=[0, 20], media=[0, 22], metadata=metadata)
        memory.update(build_fingerprint=digest(build_config), complete=False)
        write(self.source / "memory.json", memory)
        write(self.source / "windows/W00001.json", {"input": context, "sampling": metadata, "perception": payload})
        self.jsonl(self.source / "calls.jsonl", [{"purpose": "window_perception:V1:W00002", "status": "error",
                                                "attempt": 1, "error_type": "RuntimeError", "failure_stage": "generate"}])
        self.selection = self.root / "selection.json"
        write(self.selection, {"schema_version": 1, "condition": "noevent", "question_ids": ["Q1"], "reason": "user-authorized new output budget"})
        self.credentials = self.root / "private.json"
        write(self.credentials, {"MODEL_API_KEY": "not-a-real-key"})
        self.args = SimpleNamespace(run=str(self.run), parent_run=str(self.parent), core_repo=str(self.repo),
                                    selection=str(self.selection), credential_file=str(self.credentials), visual_max_tokens=16384,
                                    protected_count=1, workers=4, question_workers=2, python=sys.executable)
        self.hash_patch = patch.object(continuation, "source_hash", return_value="frozen-core")
        self.hash_patch.start()

    def tearDown(self):
        self.hash_patch.stop()
        self.temporary.cleanup()

    @staticmethod
    def question(qid, vid):
        return {"question_id": qid, "video_id": vid, "granularity": "episode", "type": "emotional reasoning",
                "question": "How does the person feel?", "rubric": {"scores": {"0": "incorrect", "1": "correct"}}}

    @staticmethod
    def jsonl(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def prepared(self):
        return continuation.prepare(self.args)[1]

    def test_prepare_replays_windows_preserves_parent_and_archives_old_budget(self):
        before = continuation.inventory(self.source)
        old_score = sha(self.parent / "accepted/noevent/Q0.json")
        config = self.prepared()
        destination = memory_root(self.run, "noevent", "V1") / "V1"
        self.assertEqual(config["execution_conditions"], ["noevent"])
        self.assertEqual(config["profile"]["visual"]["max_tokens"], 16384)
        self.assertEqual(config["profile"]["max_tokens"], 8192)
        self.assertEqual(continuation.inventory(self.source), before)
        self.assertEqual(sha(self.parent / "accepted/noevent/Q0.json"), old_score)
        self.assertEqual(sha(destination / "windows/W00001.json"), before["windows/W00001.json"])
        self.assertFalse((destination / "manifest.json").exists())
        self.assertTrue((destination / "ancestry/parent/manifest.json").is_file())
        self.assertTrue((destination / "ancestry/parent/calls.jsonl").is_file())
        continuation.verify(self.run, config)

    def test_selection_rejects_duplicate_protected_unknown_and_other_conditions(self):
        original = read(self.selection)
        variants = [{**original, "question_ids": ["Q1", "Q1"]}, {**original, "question_ids": ["Q0"]},
                    {**original, "question_ids": ["unknown"]}, {**original, "condition": "method"},
                    {**original, "condition": "base"}, {**original, "question_ids": ["../Q1"]}]
        for value in variants:
            with self.subTest(value=value):
                write(self.selection, value)
                with self.assertRaises(ValueError):
                    self.prepared()

    def test_parent_first_score_and_source_mutation_are_rejected(self):
        config = self.prepared()
        envelope = self.parent / "accepted/noevent/Q0.json"
        original = envelope.read_bytes()
        value = read(envelope)
        value["score"]["score"] = 0
        write(envelope, value)
        with self.assertRaisesRegex(ValueError, "first-score inventory"):
            continuation.verify(self.run, config)
        envelope.write_bytes(original)
        source = Path(config["protected_scores"]["Q0"]["envelope"]["source"])
        source.write_text(source.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "official score source"):
            continuation.verify(self.run, config)

    def test_original_filter_and_backend_attempts_are_not_retried(self):
        error_path = self.source / "calls.jsonl"
        error = read(self.selection)
        self.jsonl(error_path, [{"purpose": "window_perception:V1:W00002", "status": "error", "service_error_code": "content_filter"}])
        with self.assertRaisesRegex(ValueError, "refusal"):
            self.prepared()
        self.jsonl(error_path, [{"purpose": "window_perception:V1:W00002", "status": "error", "error_type": "RuntimeError"}])
        write(self.parent / "tasks/plan/noevent-V1/task.json", {"status": "error"})
        with self.assertRaisesRegex(ValueError, "backend task"):
            self.prepared()
        self.assertEqual(read(self.selection), error)

    def test_inherited_blocked_memory_uses_parent_snapshot_instead_of_rebuilding(self):
        target = self.parent / "inheritance/parent/noevent/videos/V1/memory/V1"
        target.parent.mkdir(parents=True)
        self.source.rename(target)
        write(self.parent / "inheritance/parent_index.json", {"branches": {"noevent": {"videos": {
            "V1": {"state": "blocked", "memory_dir": str(target)}}}}})
        config = self.prepared()
        self.assertEqual(config["checkpoints"]["V1"]["completed_windows"], 1)
        self.assertEqual(config["checkpoints"]["V1"]["source"], str(target))

    def test_forged_checkpoint_symlink_or_event_schema_is_rejected(self):
        memory_path = self.source / "memory.json"
        original = memory_path.read_bytes()
        memory = read(memory_path)
        memory["representation"] = "event_graph"
        write(memory_path, memory)
        with self.assertRaises(ValueError):
            self.prepared()
        memory_path.write_bytes(original)
        window = self.source / "windows/W00001.json"
        content = read(window)
        content["perception"]["events"] = []
        write(window, content)
        with self.assertRaisesRegex(ValueError, "graph fields"):
            self.prepared()
        del content["perception"]["events"]
        write(window, content)
        (self.source / "unknown-link").symlink_to(window)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.prepared()

    def test_success_or_error_task_reentry_never_starts_a_second_process(self):
        config = self.prepared()
        command = continuation.stage_command(self.run, config, "build", "V1")
        task_path = self.run / "tasks/build/noevent-V1/task.json"
        signature = digest({"command": command, "cwd": config["core_repo"]})
        for status in ("ok", "error"):
            write(task_path, {"signature": signature, "status": status})
            with patch.object(continuation.pilot.subprocess, "Popen") as spawn:
                result = continuation.run_stage(self.run, config, "build", "V1", {})
            self.assertEqual(result["status"], status)
            spawn.assert_not_called()
        write(task_path, {"signature": signature, "status": "running"})
        with patch.object(continuation.pilot.subprocess, "Popen") as spawn:
            with self.assertRaisesRegex(ValueError, "in-flight"):
                continuation.run_stage(self.run, config, "build", "V1", {})
            spawn.assert_not_called()

    def test_frozen_profile_selection_and_copied_paths_cannot_change(self):
        config = self.prepared()
        self.args.visual_max_tokens = 32768
        with self.assertRaises(ValueError):
            self.prepared()
        self.args.visual_max_tokens = 16384
        target = memory_root(self.run, "noevent", "V1") / "V1/windows/W00001.json"
        target.write_text("{}")
        with self.assertRaisesRegex(ValueError, "unstarted copied checkpoint"):
            self.prepared()
        write(self.run / "configuration.json", {**config, "condition": "method"})
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            continuation.verify(self.run, read(self.run / "configuration.json"))

    def test_no_base_method_commands_and_judge_always_one_attempt(self):
        config = self.prepared()
        for condition in ("base", "method"):
            forbidden = {**config, "condition": condition}
            with self.assertRaisesRegex(ValueError, "only noevent"):
                continuation.stage_command(self.run, forbidden, "answer", "V1")
        build = continuation.stage_command(self.run, config, "build", "V1")
        answer = continuation.stage_command(self.run, config, "answer", "V1")
        judge = continuation.stage_command(self.run, config, "score", "V1")
        self.assertEqual(build[build.index("--max-tokens") + 1], "16384")
        self.assertEqual(answer[answer.index("--max-tokens") + 1], "8192")
        self.assertEqual(judge[judge.index("--tries") + 1], "1")
        self.assertIn("methods.longemo.noevent_runner", answer)

    def test_written_first_judgment_is_imported_without_api_and_duplicate_rejected(self):
        config = self.prepared()
        q = self.questions[1]
        prediction = {"question_id": "Q1", "video_id": "V1", "status": "ok", "prediction": "new answer"}
        write(self.run / "score_questions/V1.json", [q])
        write(self.run / "score_inputs/V1.json", [prediction])
        write(self.run / "tasks/score/noevent-V1/task.json", {"status": "running", "child": {
            "pid": 987654322, "start_ticks": "1", "host": socket.gethostname()}})
        row = {**prediction, "score": 1, "max_score": 1, "normalized_score": 1}
        source = self.run / "scores/noevent/V1/run_first/scores.jsonl"
        self.jsonl(source, [row])
        with patch.object(continuation.pilot.subprocess, "Popen") as spawn:
            continuation.accept_scores(self.run, config, "V1")
            before = sha(self.run / "accepted/noevent/Q1.json")
            continuation.accept_scores(self.run, config, "V1")
            self.assertEqual(sha(self.run / "accepted/noevent/Q1.json"), before)
            spawn.assert_not_called()
        report = continuation.report(self.run)
        self.assertEqual(report["combined"]["scored"], 2)
        self.assertEqual(report["continuation"]["scored"], 1)
        self.assertEqual(report["original"]["scored"], 1)
        self.jsonl(self.run / "scores/noevent/V1/run_second/scores.jsonl", [row])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            continuation.accept_scores(self.run, config, "V1")

    def test_pending_audio_is_deferred_to_strict_fingerprint_check(self):
        observer = read(self.source / "manifest.json")["configuration"]["audio_observer"]
        write(self.source / "audio/W00002.json", {"model": observer, "input_fingerprint": "a" * 64, "result": {"observations": []}})
        config = self.prepared()
        target = memory_root(self.run, "noevent", "V1") / "V1"
        self.assertTrue((target / "audio/W00002.json").exists())
        self.assertIn("audio/W00002.json", config["checkpoints"]["V1"]["pending_audio_strict_validation"])
        with self.assertRaisesRegex(ValueError, "separate exact-input"):
            continuation.apply_probe_seed(None)

    def test_report_cannot_overwrite_the_parent(self):
        self.prepared()
        with self.assertRaisesRegex(ValueError, "must not be overwritten"):
            continuation.report(self.run, self.parent / "report")

    def test_changed_video_is_rejected_before_any_new_request(self):
        config = self.prepared()
        (self.video_dir / "V1.mp4").write_bytes(b"different-video")
        with self.assertRaisesRegex(ValueError, "media identity changed"):
            continuation.verify(self.run, config)
        with self.assertRaisesRegex(ValueError, "media changed"):
            self.prepared()

    def test_per_video_rubric_and_same_budget_profile_are_checked(self):
        self.args.visual_max_tokens = 8192
        config = self.prepared()
        self.assertEqual(config["profile"]["visual"]["max_tokens"], 8192)
        report = continuation.report(self.run)
        self.assertNotIn("enlarged", report["comparison_scope"])
        self.assertIn("保持原视觉输出上限 8192", (self.run / "report/report.md").read_text())
        video_questions = self.run / "questions/V1.json"
        changed = read(video_questions)
        changed[0]["rubric"]["scores"]["1"] = "A changed scoring rubric."
        write(video_questions, changed)
        with self.assertRaisesRegex(ValueError, "per-video question"):
            continuation.verify(self.run, config)

    def test_report_revalidates_question_and_score_even_if_source_hash_is_rewritten(self):
        config = self.prepared()
        prediction = {"question_id": "Q1", "video_id": "V1", "prediction": "happy", "status": "ok"}
        write(self.run / "score_inputs/V1.json", [prediction])
        source = self.run / "scores/noevent/V1/run_first/scores.jsonl"
        row = {**prediction, "score": 2, "max_score": 2, "normalized_score": 1}
        self.jsonl(source, [row])
        write(self.run / "accepted/noevent/Q1.json", {"score": row, "source": str(source),
              "source_sha256": sha(source), "prediction_sha256": digest("happy")})
        with self.assertRaisesRegex(ValueError, "frozen question, prediction or rubric"):
            continuation.report(self.run)

    def test_pipeline_keeps_window_ancestry_and_reaches_one_official_judgment(self):
        config = self.prepared()
        stages = []
        folder = memory_root(self.run, "noevent", "V1") / "V1"

        def offline_stage(run, cfg, stage, video, env, question_file=None):
            stages.append(stage)
            self.assertEqual(env["PYTHONPATH"], str(self.repo))
            self.assertNotIn("LONGEMO_ALLOW_AUDIO_CACHE_REUSE", env)
            if stage == "build":
                manifest = read(folder / "ancestry/parent/manifest.json")
                manifest["configuration"]["model"]["max_tokens"] = 16384
                manifest["fingerprint"] = digest(manifest["configuration"])
                write(folder / "manifest.json", manifest)
                memory = read(folder / "memory.json")
                payload = {"entities": [], "observations": [], "summary": "The clip continues."}
                metadata = {"audio": False, "subtitle_ids": []}
                memory = apply_window(memory, payload, window_id="W00002", core=[20, 40], media=[18, 40], metadata=metadata)
                memory.update(build_fingerprint=manifest["fingerprint"], complete=True)
                write(folder / "memory.json", memory)
                write(folder / "windows/W00002.json", {"input": {"video_id": "V1", "window_id": "W00002",
                      "core_interval": [20, 40], "media_interval": [18, 40], "cast": []}, "sampling": metadata, "perception": payload})
            elif stage == "answer":
                self.jsonl(run / "answers/noevent/V1/predictions.jsonl", [{"question_id": "Q1", "video_id": "V1",
                           "status": "ok", "prediction": "happy"}])
            else:
                self.assertEqual(read(question_file)[0]["question_id"], "Q1")
                write(run / "tasks/score/noevent-V1/task.json", {"status": "ok", "child": {
                    "pid": 987654322, "start_ticks": "1", "host": socket.gethostname()}})
                self.jsonl(run / "scores/noevent/V1/run_first/scores.jsonl", [{"question_id": "Q1", "video_id": "V1",
                           "status": "ok", "prediction": "happy", "score": 1, "max_score": 1, "normalized_score": 1}])
            return {"status": "ok"}

        with patch.object(continuation, "run_stage", side_effect=offline_stage):
            continuation.pipeline(self.run, config, "V1")
        self.assertEqual(stages, ["build", "answer", "score"])
        provenance = read(self.run / "noevent/frozen_memories/V1.json")
        self.assertEqual(provenance["windows"]["W00001"]["source"], "parent")
        self.assertEqual(provenance["windows"]["W00002"]["source"], "continuation")
        self.assertEqual(read(self.run / "pipelines/noevent/V1.json")["status"], "complete")
        self.assertEqual(continuation.report(self.run)["combined"]["scored"], 2)


if __name__ == "__main__":
    unittest.main()
