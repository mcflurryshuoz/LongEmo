"""Offline safeguards for narrowing an existing frozen run to noevent/method."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.zyf import full_suite as suite
from experiments.zyf.matched_pilot import digest, read, sha, write


class FullSuiteScopeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run = self.root / "runtime/runs/full"
        self.original = self.root / "old-deployment/experiments/zyf/full_suite.py"
        self.original.parent.mkdir(parents=True)
        self.original.write_text("# immutable original coordinator\n")
        for name in ("matched_pilot.py", "full_suite_inheritance.py"):
            self.original.with_name(name).write_bytes(Path(suite.__file__).with_name(name).read_bytes())
        self.questions = [{"question_id": "Q1", "video_id": "V1", "granularity": "episode", "question": "What?",
                           "rubric": {"scores": {"0": "wrong", "2": "right"}}}]
        self.metadata = self.root / "metadata"
        write(self.metadata / "questions.json", self.questions)
        write(self.run / "questions.json", self.questions)
        write(self.run / "questions/V1.json", self.questions)
        manifest = {"schema_version": 1, "videos": {"V1": {"video_sha256": "a" * 64}}}
        write(self.metadata / "media.json", manifest)
        write(self.run / "media_manifest.json", manifest)
        self.config = {
            "mode": "full", "question_count": 1, "video_count": 1, "questions_sha256": digest(self.questions),
            "media_manifest_sha256": sha(self.metadata / "media.json"), "media": manifest["videos"],
            "suite_sha256": sha(self.original), "suite_support_hashes": {
                name: sha(self.original.with_name(name)) for name in ("matched_pilot.py", "full_suite_inheritance.py")},
            "source_hashes": {"event": "event-core", "noevent": "noevent-core"},
            "repos": {"event": str(self.root / "method"), "noevent": str(self.original.parents[2])},
            "parent_run": str(self.root / "pilot"), "python": sys.executable,
            "videos_dir": str(self.root / "videos"), "subtitles_dir": str(self.root / "subtitles"),
            "execution": {"workers": 12, "question_workers": 2, "poll_seconds": 10,
                          "parent_wait_timeout": 86400, "media_wait_timeout": 86400, "producer_done": None},
            "stage_tries": {"build": 3, "plan": 3, "answer": 3, "score": 1},
            "model": "test", "base_url": "https://example.invalid", "timeout": 1800,
            "max_tokens": 8192, "embedding_model": "embedding", "embedding_base_url": "https://example.invalid",
            "evidence_chars": 48000, "progressive_rounds": 3, "progressive_anchor_k": 4,
        }
        write(self.run / "configuration.json", self.config)
        self.credentials = self.root / "runtime/private/answer.json"
        write(self.credentials, {"MODEL_API_KEY": "test-only"})
        write(self.run / "tasks/build/noevent-V0/task.json", {
            "status": "error", "command": [sys.executable, "build", "--credential-file", str(self.credentials)],
            "signature": "old-attempt", "child": {"pid": 987654321, "start_ticks": "12"}})
        self.args = SimpleNamespace(stage="run", execution_scope=str(self.run / "execution_scope.json"),
            parent_run=self.config["parent_run"], questions=str(self.metadata / "questions.json"),
            media_manifest=str(self.metadata / "media.json"), method_repo=self.config["repos"]["event"],
            noevent_repo=self.config["repos"]["noevent"], videos_dir=self.config["videos_dir"],
            subtitles_dir=self.config["subtitles_dir"], runtime=str(self.root / "runtime"), run_name="full",
            python=sys.executable, expected_questions=1, expected_videos=1, workers=12, question_workers=2,
            poll_seconds=10, parent_wait_timeout=86400, media_wait_timeout=86400, producer_done=None,
            credential_file=str(self.credentials))
        self.scope = {"schema_version": 1, "conditions": ["noevent", "method"],
            "previous_configuration_sha256": sha(self.run / "configuration.json"),
            "coordinator_file": str(Path(suite.__file__).resolve()), "coordinator_sha256": sha(suite.__file__),
            "reason": "user_requested_no_new_base", "created_at": "2026-09-23T01:40:00+08:00",
            "previous_coordinator": {"pid": 987654321, "start_ticks": "12"}}
        write(self.run / "execution_scope.json", self.scope)
        self.sources = patch.object(suite, "source_hash", side_effect=lambda p: "event-core" if str(p) == self.config["repos"]["event"] else "noevent-core")
        self.sources.start()
        self.stopped = patch.object(suite, "assert_identity_stopped")
        self.stopped.start()

    def tearDown(self):
        self.stopped.stop()
        self.sources.stop()
        self.temporary.cleanup()

    def prepared(self):
        return suite.prepare(self.args, self.run)[0]

    def test_narrow_resume_preserves_configuration_budgets_tasks_and_deadlines(self):
        write(self.run / "timing.json", {"started_unix": 123})
        write(self.run / "phases/noevent.json", {"started_unix": 124})
        paths = [self.run / x for x in ("configuration.json", "tasks/build/noevent-V0/task.json", "timing.json", "phases/noevent.json")]
        before = {str(p): sha(p) for p in paths}
        config = self.prepared()
        self.assertEqual(suite.execution_conditions(config), ("noevent", "method"))
        self.assertEqual({k: v for k, v in config.items() if k != "_execution_scope"}, self.config)
        self.assertEqual({str(p): sha(p) for p in paths}, before)
        self.assertEqual(config["stage_tries"], {"build": 3, "plan": 3, "answer": 3, "score": 1})
        suite.verify_runtime_sources(config)

    def test_scope_cannot_expand_or_change_the_conditions(self):
        for conditions in (["noevent", "base", "method"], ["method"], ["method", "noevent"], ["noevent", "method", "other"]):
            with self.subTest(conditions=conditions):
                write(self.run / "execution_scope.json", {**self.scope, "conditions": conditions})
                with self.assertRaisesRegex(ValueError, "only disable new base"):
                    self.prepared()

    def test_scope_requires_a_separate_pinned_coordinator_and_original_hashes(self):
        original_bytes = self.original.read_bytes()
        self.original.write_text("# modified original\n")
        with self.assertRaisesRegex(ValueError, "coordinator changed after"):
            self.prepared()
        self.original.write_bytes(original_bytes)
        write(self.run / "execution_scope.json", {**self.scope, "coordinator_sha256": "0" * 64})
        with self.assertRaisesRegex(ValueError, "continuation coordinator changed"):
            self.prepared()
        write(self.run / "execution_scope.json", self.scope)
        self.original.with_name("matched_pilot.py").write_text("# modified support\n")
        with self.assertRaisesRegex(ValueError, "support source changed"):
            self.prepared()

    def test_all_execution_limits_paths_and_question_data_must_stay_fixed(self):
        changes = {"workers": 13, "question_workers": 3, "poll_seconds": 11, "parent_wait_timeout": 86401,
                   "media_wait_timeout": 86401, "expected_questions": 2, "expected_videos": 2,
                   "method_repo": str(self.root / "different"), "noevent_repo": str(self.root / "different"),
                   "parent_run": str(self.root / "different"), "videos_dir": str(self.root / "different"),
                   "subtitles_dir": str(self.root / "different"), "runtime": str(self.root / "different"),
                   "credential_file": str(self.root / "different"), "producer_done": str(self.root / "different")}
        for key, value in changes.items():
            with self.subTest(key=key):
                changed = copy.copy(self.args)
                setattr(changed, key, value)
                with self.assertRaises(ValueError):
                    suite.prepare(changed, self.run)
        write(self.run / "configuration.json", {**self.config, "stage_tries": {"score": 2}})
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            self.prepared()

    def test_each_stage_rechecks_immutable_scope_and_never_launches_base(self):
        config = self.prepared()
        with patch.object(suite, "run_command") as command:
            for stage in ("answer", "score"):
                with self.assertRaisesRegex(ValueError, "new base"):
                    suite.stage_run(self.args, self.run, config, stage, "base-V1", ["unused"], self.root, {})
            command.assert_not_called()
        write(self.run / "execution_scope.json", {**self.scope, "created_at": "changed-after-start"})
        with self.assertRaisesRegex(ValueError, "scope changed"):
            suite.verify_runtime_sources(config)

    def test_scope_cannot_be_omitted_after_it_has_been_installed(self):
        self.args.execution_scope = None
        with self.assertRaisesRegex(ValueError, "pass --execution-scope"):
            suite.prepare(self.args, self.run)

    def test_previous_coordinator_must_have_stopped(self):
        with patch.object(suite, "assert_identity_stopped", side_effect=ValueError("still active")):
            with self.assertRaisesRegex(ValueError, "still active"):
                self.prepared()

    def test_method_pipeline_runs_no_base_and_retains_old_base_first_score(self):
        config = self.prepared()
        memory = suite.memory_root(self.run, "event", "V1") / "V1/memory.json"
        write(memory, {"complete": True})
        write(self.run / "event/frozen_memories/V1.json", {"memory_sha256": sha(memory)})
        write(self.run / "event/plans/Q1.json", {"plan": "unchanged"})
        base_score = {"question_id": "Q1", "score": 0, "normalized_score": 0}
        write(self.run / "accepted/base/Q1.json", {"score": base_score, "source": "first-base-reference"})
        base_sha = sha(self.run / "accepted/base/Q1.json")
        index = {"branches": {"event": {"videos": {"V1": {"state": "reusable", "plans": {"Q1": {"state": "success"}}}}}},
                 "conditions": {condition: {"questions": {}} for condition in suite.CONDITIONS}}
        calls = []
        def stage(args, run, cfg, kind, key, command, repo, env):
            calls.append((kind, key))
            self.assertEqual(key, "method-V1")
            output = Path(command[command.index("--output-dir") + 1])
            if kind == "answer":
                self.assertEqual(command[command.index("--retrieval") + 1], "progressive")
                self.assertEqual(command[command.index("--progressive-routing") + 1], "none")
                suite.write_rows(output / "predictions.jsonl", [{"question_id": "Q1", "status": "ok", "prediction": "answer"}])
            elif kind == "score":
                self.assertEqual(command[command.index("--tries") + 1], "1")
                suite.write_rows(output / "run_fixture/scores.jsonl", [{"question_id": "Q1", "status": "ok", "prediction": "answer",
                    "score": 1, "max_score": 2, "normalized_score": .5}])
            return {"status": "ok"}
        with patch.object(suite, "stage_run", side_effect=stage):
            suite.video_pipeline(self.args, self.run, config, index, "event", "V1")
        self.assertEqual(calls, [("answer", "method-V1"), ("score", "method-V1")])
        self.assertFalse((self.run / "stage_questions/answer/base-V1.json").exists())
        self.assertFalse((self.run / "stage_questions/score/base-V1.json").exists())
        self.assertEqual(sha(self.run / "accepted/base/Q1.json"), base_sha)
        self.assertEqual(read(self.run / "pipelines/event/V1.json")["scored"], {"method": 1})
        write(self.run / "accepted/noevent/Q1.json", {"score": {"question_id": "Q1", "normalized_score": .5}})
        summary = suite.summarize(self.run, config)
        self.assertEqual(summary["retained_reference_conditions"], ["base"])
        self.assertEqual(summary["execution_conditions"], ["noevent", "method"])
        self.assertTrue(summary["complete"])

    def test_completion_does_not_wait_for_missing_base_scores(self):
        questions = [self.questions[0], {**self.questions[0], "question_id": "Q2"}]
        write(self.run / "questions.json", questions)
        config = {**self.config, "question_count": 2, "questions_sha256": digest(questions),
                  "_execution_scope": {"manifest": self.scope, "sha256": sha(self.run / "execution_scope.json")}}
        for condition in ("noevent", "method"):
            for question in questions:
                write(self.run / "accepted" / condition / (question["question_id"] + ".json"),
                      {"score": {"question_id": question["question_id"], "normalized_score": .5}})
        base_path = self.run / "accepted/base/Q1.json"
        write(base_path, {"score": {"question_id": "Q1", "normalized_score": 0}, "source": "first-base-reference"})
        before = sha(base_path)
        summary = suite.summarize(self.run, config)
        self.assertEqual(summary["conditions"]["base"]["scored"], 1)
        self.assertEqual(summary["conditions"]["base"]["total"], 2)
        self.assertFalse((self.run / "accepted/base/Q2.json").exists())
        self.assertEqual(sha(base_path), before)
        self.assertEqual(summary["execution_conditions"], ["noevent", "method"])
        self.assertEqual(summary["retained_reference_conditions"], ["base"])
        self.assertTrue(summary["complete"])

    def test_plans_keep_the_original_command_path_and_terminal_tasks_are_not_repeated(self):
        config = self.prepared()
        memory = suite.memory_root(self.run, "noevent", "V1") / "V1/memory.json"
        write(memory, {"complete": True})
        write(self.run / "noevent/frozen_memories/V1.json", {"memory_sha256": sha(memory)})
        index = {"branches": {"noevent": {"videos": {"V1": {"state": "reusable", "plans": {}}}}},
                 "conditions": {condition: {"questions": {"Q1": {"answer_state": "blocked", "score_state": "blocked"}}}
                                for condition in suite.CONDITIONS}}
        calls = []
        def stage(args, run, cfg, kind, key, command, repo, env):
            calls.append((kind, command))
            self.assertEqual(kind, "plan")
            self.assertEqual(command[1], str(self.original.resolve()))
            return {"status": "error"}
        with patch.object(suite, "stage_run", side_effect=stage):
            suite.video_pipeline(self.args, self.run, config, index, "noevent", "V1")
        self.assertEqual(len(calls), 1)
        before = sha(self.run / "tasks/build/noevent-V0/task.json")
        write(self.run / "pipelines/noevent/V1.json", {"status": "build_failed"})
        with patch.object(suite, "video_pipeline") as pipeline:
            suite.run_phase(self.args, self.run, config, index, "noevent")
            pipeline.assert_not_called()
        self.assertEqual(sha(self.run / "tasks/build/noevent-V0/task.json"), before)


if __name__ == "__main__":
    unittest.main()
