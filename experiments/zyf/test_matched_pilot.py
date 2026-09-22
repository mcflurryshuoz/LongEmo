import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.zyf.matched_pilot import (accept_scores, assert_identity_stopped, common_args, digest,
    execute, freeze, inherit_parent, memory_root, parent_failure, parser, read, run_command, run_lock,
    select_questions, sha, stage_exit_code, summary, validate_parent_checkpoint, video_tasks, write)


class MatchedPilotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_selection_is_real_fixed_cohort_and_smoke_subset(self):
        rows = [{"question_id": f"Q{i}", "video_id": f"V{i % 21}", "granularity": "episode"} for i in range(60)]
        source = self.root / "questions.json"
        write(source, rows)
        self.assertEqual(select_questions(source, "pilot"), rows[:50])
        smoke = select_questions(source, "smoke", "V0")
        self.assertEqual([q["question_id"] for q in smoke], ["Q0", "Q21", "Q42"])
        self.assertEqual({q["video_id"] for q in smoke}, {"V0"})

    def test_frozen_selection_cannot_silently_change(self):
        path = self.root / "frozen.json"
        freeze(path, {"question_ids": ["Q1"]})
        freeze(path, {"question_ids": ["Q1"]})
        with self.assertRaisesRegex(ValueError, "frozen artifact changed"):
            freeze(path, {"question_ids": ["Q2"]})

    def test_lock_rejects_second_coordinator(self):
        with run_lock(self.root):
            with self.assertRaisesRegex(RuntimeError, "another coordinator"):
                with run_lock(self.root):
                    pass

    def test_finished_failure_is_never_automatically_retried(self):
        counter = self.root / "counter.txt"
        code = "from pathlib import Path; p=Path(" + repr(str(counter)) + "); p.write_text(p.read_text()+'x' if p.exists() else 'x'); raise SystemExit(1)"
        command = [sys.executable, "-c", code]
        first = run_command(self.root / "task", command, self.root)
        second = run_command(self.root / "task", command, self.root)
        self.assertEqual(first["status"], "error")
        self.assertEqual(second, first)
        self.assertEqual(counter.read_text(), "x")

    def test_ambiguous_interruption_requires_audit(self):
        command = [sys.executable, "-c", "pass"]
        folder = self.root / "task"
        run_command(folder, command, self.root)
        state = read(folder / "task.json")
        state["status"] = "running"
        write(folder / "task.json", state)
        self.assertEqual(run_command(folder, command, self.root)["status"], "needs_audit")

    def test_build_queue_interleaves_representations(self):
        items = video_tasks([f"V{i}" for i in range(21)])
        self.assertEqual(items[:4], [("event", "V0"), ("noevent", "V0"), ("event", "V1"), ("noevent", "V1")])
        self.assertEqual(sum(branch == "event" for branch, _ in items[:12]), 6)
        self.assertEqual(sum(branch == "noevent" for branch, _ in items[:12]), 6)

    def test_stage_attempt_budgets_keep_official_judge_at_one(self):
        config = {"model": "m", "base_url": "https://example.invalid/v1", "timeout": 1,
                  "stage_tries": {"build": 3, "plan": 2, "answer": 4, "score": 99}}
        for stage, expected in (("build", 3), ("plan", 2), ("answer", 4), ("score", 1)):
            command = common_args(config, stage)
            self.assertEqual(command[command.index("--tries") + 1], str(expected))

    def test_parent_live_or_unverifiable_identity_is_rejected(self):
        from experiments.zyf.matched_pilot import pid_identity
        current = pid_identity(__import__("os").getpid())
        with self.assertRaisesRegex(ValueError, "active|cannot prove"):
            assert_identity_stopped(current, "fixture")
        with self.assertRaisesRegex(ValueError, "another host"):
            assert_identity_stopped({**current, "host": "different-host"}, "fixture")
        with patch("experiments.zyf.matched_pilot.pid_identity", return_value={**current, "start_ticks": "new"}):
            assert_identity_stopped({**current, "start_ticks": "old"}, "fixture")

    def checkpoint_fixture(self, folder, branch="event"):
        from methods.longemo import memory as event_memory, noevent_memory
        module = event_memory if branch == "event" else noevent_memory
        config = {"window_seconds": 20, "padding": 2, "fps": 1, "max_frames": 24, "max_pixels": 200704,
                  "model": "fixture", "base_url": "https://example.invalid/v1", "audio_model": "fixture-audio",
                  "audio_base_url": "https://example.invalid/audio", "max_tokens": 8192, "timeout": 30}
        media = {"video_sha256": "video-sha", "subtitles_sha256": "subtitle-sha"}
        clients = {"model": {"model": "fixture", "options": {"a": 1}},
                   "audio_observer": {"model": "fixture-audio"}}
        old = {**media, **clients, "window_seconds": 20, "padding": 2,
               "media": {"fps": 1, "max_frames": 24, "max_pixels": 200704, "with_audio": True},
               "allow_revisions": False}
        write(folder / "manifest.json", {"configuration": old, "fingerprint": digest(old)})
        payload = {"entities": [], "observations": [], "events": [], "relations": [], "corrections": []}
        if branch == "noevent":
            payload = {"entities": [], "observations": [], "summary": "Quiet scene", "actions": [],
                       "objects": [], "signals": [], "participants": [], "emotion_cues": []}
        audio = {"model": clients["audio_observer"], "input_fingerprint": "audio-input", "result": {"observations": []}}
        write(folder / "audio/W00001.json", audio)
        sampling = {"audio_observer": {"source_sha256": sha(folder / "audio/W00001.json"), "input_fingerprint": "audio-input"}}
        context = {"video_id": "V0", "window_id": "W00001", "core_interval": [0, 20], "media_interval": [0, 22]}
        write(folder / "windows/W00001.json", {"input": context, "sampling": sampling, "perception": payload})
        memory = module.apply_window(module.empty_memory("V0", 40, "video-sha"), payload,
                                     window_id="W00001", core=[0, 20], media=[0, 22], metadata=sampling)
        memory["build_fingerprint"] = digest(old)
        write(folder / "memory.json", memory)
        return config, media, clients

    def test_checkpoint_replays_atomic_windows_and_rejects_tampering(self):
        repo = Path(__file__).resolve().parents[2]
        for branch in ("event", "noevent"):
            with self.subTest(branch=branch):
                folder = self.root / branch
                config, media, clients = self.checkpoint_fixture(folder, branch)
                result = validate_parent_checkpoint(folder, repo, branch, "V0", media, config, clients)
                self.assertEqual(result["completed_windows"], 1)
                self.assertEqual(set(result["files"]), {"memory.json", "windows/W00001.json", "audio/W00001.json"})
                with self.assertRaisesRegex(ValueError, "client configuration"):
                    validate_parent_checkpoint(folder, repo, branch, "V0", media, config,
                                               {**clients, "model": {"model": "different"}})
                changed = read(folder / "memory.json")
                changed["completed_windows"] = ["W00002"]
                write(folder / "memory.json", changed)
                with self.assertRaisesRegex(ValueError, "contiguous prefix"):
                    validate_parent_checkpoint(folder, repo, branch, "V0", media, config, clients)

    def test_parent_failure_distinguishes_refusal_http428_and_schema(self):
        folder = self.root / "failure"
        ledger = folder / "audio/calls.jsonl"
        ledger.parent.mkdir(parents=True)
        for row, expected in (({"http_status": 400, "service_error_code": "content_policy_violation"}, "blocked_content"),
                              ({"http_status": 428}, "blocked_http428"),
                              ({"http_status": 403}, "blocked_service"),
                              ({"error_type": "JSONDecodeError"}, "resume_once"),
                              ({"error_type": "RemoteDisconnected"}, "resume_once"),
                              ({"error_type": "RuntimeError"}, "blocked_unknown")):
            ledger.write_text(json.dumps({"status": "error", "purpose": "audio_observer:W00002", **row}) + "\n")
            self.assertEqual(parent_failure(folder)["status"], expected)

    def parent_fixture(self):
        parent, run = self.root / "parent", self.root / "child"
        repo = Path(__file__).resolve().parents[2]
        questions = [{"question_id": f"Q{i}", "video_id": f"V{i % 21}", "granularity": "episode"} for i in range(50)]
        selected = [q for q in questions if q["video_id"] == "V0"]
        write(parent / "questions.json", questions)
        write(run / "questions.json", selected)
        for branch in ("event", "noevent"):
            folder = memory_root(parent, branch, "V0") / "V0"
            config, media, clients = self.checkpoint_fixture(folder, branch)
            write(folder / "audio/W00002.json", {"pending": "must stay in history"})
            (folder / "calls.jsonl").write_text(json.dumps({"status": "error", "purpose": "perception:V0:W00002",
                "error_type": "JSONDecodeError", "time_unix": 1}) + "\n")
        config.update(media={"V0": media}, repos={"event": str(repo), "noevent": str(repo)})
        write(parent / "configuration.json", {**config, "mode": "pilot", "questions_sha256": digest(questions)})
        args = SimpleNamespace(parent_run=str(parent), python=sys.executable, credential_file="unused")
        return parent, run, args, config, clients

    def test_parent_import_copies_only_committed_windows_and_archives_pending_audio(self):
        parent, run, args, config, clients = self.parent_fixture()
        original = {str(p): sha(p) for p in parent.rglob("*") if p.is_file()}
        with patch("experiments.zyf.matched_pilot.stopped_parent"), patch(
                "experiments.zyf.matched_pilot.inspect_client_configs", return_value=clients):
            inherit_parent(args, run, config)
            inherit_parent(args, run, config)
        copied = memory_root(run, "event", "V0") / "V0"
        self.assertTrue((copied / "memory.json").exists())
        self.assertFalse((copied / "manifest.json").exists())
        self.assertFalse((copied / "audio/W00002.json").exists())
        self.assertTrue((run / "inheritance/history/event-V0/audio/W00002.json").exists())
        self.assertEqual(original, {str(p): sha(p) for p in parent.rglob("*") if p.is_file()})
        lineage = read(run / "inheritance/manifest.json")
        self.assertEqual(sum(x["completed_windows"] for x in lineage["videos"].values()), 2)

    def test_parent_refusal_blocks_both_representations_and_scores_reject_import(self):
        parent, run, args, config, clients = self.parent_fixture()
        ledger = memory_root(parent, "event", "V0") / "V0/calls.jsonl"
        ledger.write_text(json.dumps({"status": "error", "purpose": "perception:V0:W00002", "http_status": 400,
                                    "service_error_code": "content_policy_violation"}) + "\n")
        with patch("experiments.zyf.matched_pilot.stopped_parent"), patch(
                "experiments.zyf.matched_pilot.inspect_client_configs", return_value=clients):
            inherit_parent(args, run, config)
            self.assertTrue((run / "inheritance/blocked/event-V0.json").exists())
            self.assertTrue((run / "inheritance/blocked/noevent-V0.json").exists())
            scored = self.root / "other-child"
            write(scored / "questions.json", read(run / "questions.json"))
            write(parent / "accepted/base/Q0.json", {"score": 0})
            with self.assertRaisesRegex(ValueError, "scoring attempts"):
                inherit_parent(args, scored, config)

    def test_zero_exit_with_incomplete_build_or_plans_is_error_without_stopping_peer(self):
        for stage in ("build", "plan"):
            with self.subTest(stage=stage):
                run = self.root / stage
                q = {"question_id": "Q1", "video_id": "V1", "granularity": "episode",
                     "rubric": {"scores": {"0": "no", "4": "yes"}}}
                write(run / "questions/V1.json", [q])
                credentials = self.root / "credentials.json"
                write(credentials, {"MODEL_API_KEY": "test-only-not-a-real-key"})
                args = parser().parse_args([stage, "--method-repo", str(self.root), "--noevent-repo", str(self.root),
                    "--runtime", str(self.root), "--run-name", stage, "--questions", str(run / "questions/V1.json"),
                    "--credential-file", str(credentials), "--build-workers", "2", "--video-workers", "2"])
                config = {"media": {"V1": {}}, "repos": {"event": str(self.root), "noevent": str(self.root)},
                          "model": "test", "base_url": "https://example.invalid/v1", "timeout": 2,
                          "videos_dir": str(self.root), "subtitles_dir": str(self.root), "audio_model": "test-audio",
                          "audio_base_url": "https://example.invalid/v1beta"}
                if stage == "plan":
                    for branch in ("event", "noevent"):
                        memory = memory_root(run, branch, "V1") / "V1/memory.json"
                        write(memory, {"complete": True})
                        freeze(run / branch / "frozen_memories/V1.json", {"memory_sha256": sha(memory)})
                        write(run / "tasks/build" / (branch + "-V1") / "task.json", {"status": "ok"})

                def fake_command(folder, command, cwd, env):
                    value = lambda flag: command[command.index(flag) + 1]
                    branch = "noevent" if folder.name.startswith("noevent") else "event"
                    if stage == "build":
                        write(Path(value("--output-dir")) / "V1/memory.json", {"complete": branch == "event"})
                    elif branch == "event":
                        path = run / branch / "plans/Q1.json"
                        write(path, {"plan": "frozen"})
                        freeze(run / branch / "plans/frozen/V1.json", {"Q1": sha(path)})
                    state = {"status": "ok", "returncode": 0}
                    write(folder / "task.json", state)
                    return state

                with patch("experiments.zyf.matched_pilot.run_command", side_effect=fake_command):
                    execute(args, run, config, [q])
                result = read(run / "status.json")
                self.assertEqual(result["tasks"][stage + "/event-V1"], "ok")
                self.assertEqual(result["tasks"][stage + "/noevent-V1"], "error")
                self.assertEqual(stage_exit_code(stage, result), 1)
                failed = read(run / "tasks" / stage / "noevent-V1/task.json")
                self.assertEqual(failed["returncode"], 0)
                self.assertIn("validation_error", failed)
                self.assertIn("budget parameter=48000", result["score_semantics"])
                self.assertNotIn("<=48000", result["score_semantics"])

    def test_explicit_failed_stage_cannot_return_zero_even_if_scores_exist(self):
        result = {"complete": True, "videos": 1,
                  "tasks": {"plan/event-V1": "ok", "plan/noevent-V1": "error"}}
        self.assertEqual(stage_exit_code("plan", result), 1)

    def test_failed_full_video_gate_leaves_remaining_video_tasks_unstarted(self):
        run = self.root / "gate"
        questions = [{"question_id": f"Q{i}", "video_id": f"V{i}", "granularity": "episode",
                      "rubric": {"scores": {"0": "no", "1": "yes"}}} for i in range(2)]
        for q in questions:
            write(run / "questions" / (q["video_id"] + ".json"), [q])
        credentials = self.root / "credentials.json"
        write(credentials, {"MODEL_API_KEY": "test-only-not-a-real-key"})
        args = parser().parse_args(["all", "--method-repo", str(self.root), "--noevent-repo", str(self.root),
            "--runtime", str(self.root), "--run-name", "gate", "--questions", "unused",
            "--credential-file", str(credentials), "--gate-video", "V0"])
        config = {"gate_video": "V0", "media": {"V0": {}, "V1": {}},
                  "repos": {"event": str(self.root), "noevent": str(self.root)}, "model": "test",
                  "base_url": "https://example.invalid/v1", "timeout": 2, "videos_dir": str(self.root),
                  "subtitles_dir": str(self.root), "audio_model": "test", "audio_base_url": "https://example.invalid"}
        observed = []

        def failed(folder, command, cwd, env):
            observed.append(folder.name)
            write(folder / "task.json", {"status": "error", "returncode": 1})
            return {"status": "error", "returncode": 1}

        with patch("experiments.zyf.matched_pilot.run_command", side_effect=failed):
            execute(args, run, config, questions)
        self.assertCountEqual(observed, ["event-V0", "noevent-V0"])
        self.assertFalse(read(run / "gate.json")["passed"])
        self.assertEqual(read(run / "status.json")["builds"]["event-V1"]["task_status"], "pending")

    def test_first_zero_score_survives_later_higher_score(self):
        questions = [{"question_id": "Q1", "rubric": {"scores": {"0": "no", "4": "yes"}}}]
        prediction = {"question_id": "Q1", "status": "ok", "prediction": "answer"}
        output = self.root / "answers/method/V1/predictions.jsonl"
        output.parent.mkdir(parents=True)
        output.write_text(json.dumps(prediction) + "\n")
        row = {"question_id": "Q1", "status": "ok", "prediction": "answer", "max_score": 4, "score": 0, "normalized_score": 0}
        first = self.root / "scores/method/V1/run_1/scores.jsonl"
        first.parent.mkdir(parents=True)
        first.write_text(json.dumps(row) + "\n")
        self.assertEqual(accept_scores(self.root, "method", questions)[0]["score"], 0)
        later = self.root / "scores/method/V1/run_2/scores.jsonl"
        later.parent.mkdir(parents=True)
        later.write_text(json.dumps({**row, "score": 4, "normalized_score": 1}) + "\n")
        self.assertEqual(accept_scores(self.root, "method", questions)[0]["score"], 0)
        first.write_text(json.dumps({**row, "score": 2, "normalized_score": 0.5}) + "\n")
        with self.assertRaisesRegex(ValueError, "first accepted official score was modified"):
            accept_scores(self.root, "method", questions)

    def test_summary_defers_live_outputs_and_does_not_swallow_terminal_corruption(self):
        run = self.root / "concurrent"
        questions = [{"question_id": "Q1", "video_id": "V1", "rubric": {"scores": {"0": "no", "1": "yes"}}}]
        output_paths = [run / "answers/method/V1/predictions.jsonl",
                        run / "scores/method/V1/run_partial/scores.jsonl",
                        memory_root(run, "event", "V1") / "V1/memory.json"]
        tasks = [("answer", "method"), ("score", "method"), ("build", "event")]
        for path, (stage, condition) in zip(output_paths, tasks):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"currently_writing":')
            write(run / "tasks" / stage / (condition + "-V1") / "task.json", {"status": "running"})
        result = summary(run, questions)
        self.assertEqual(result["conditions"]["method"]["scored"], 0)
        self.assertIsNone(result["builds"]["event-V1"]["completed_windows"])
        self.assertTrue(result["builds"]["event-V1"]["active_output_deferred"])
        # A terminated task's malformed complete record must surface; no broad
        # JSONDecodeError catch is permitted merely because another task runs.
        output_paths[0].write_text('{"broken":}\n')
        write(run / "tasks/answer/method-V1/task.json", {"status": "error"})
        with self.assertRaises(json.JSONDecodeError):
            summary(run, questions)

    def test_offline_stage_chain_uses_cropped_questions_and_shared_event_inputs(self):
        run = self.root / "run"
        q = {"question_id": "Q1", "video_id": "V1", "granularity": "episode",
             "rubric": {"scores": {"0": "no", "4": "yes"}}}
        write(run / "questions/V1.json", [q])
        credentials = self.root / "credentials.json"
        write(credentials, {"MODEL_API_KEY": "test-only-not-a-real-key"})
        args = parser().parse_args(["all", "--method-repo", str(self.root), "--noevent-repo", str(self.root),
            "--runtime", str(self.root), "--run-name", "run", "--questions", str(run / "questions/V1.json"),
            "--credential-file", str(credentials), "--build-workers", "2", "--video-workers", "2"])
        config = {"gate_video": "V1", "media": {"V1": {}}, "repos": {"event": str(self.root), "noevent": str(self.root)},
                  "model": "test", "base_url": "https://example.invalid/v1", "timeout": 2,
                  "videos_dir": str(self.root), "subtitles_dir": str(self.root), "audio_model": "test-audio",
                  "audio_base_url": "https://example.invalid/v1beta", "embedding_model": "test-embedding",
                  "embedding_base_url": "https://example.invalid/v1"}
        commands = []

        def fake_command(folder, command, cwd, env):
            commands.append(command)
            value = lambda flag: command[command.index(flag) + 1]
            if "build" in command:
                write(Path(value("--output-dir")) / "V1/memory.json", {"complete": True})
            elif "_plan" in command:
                branch = value("--branch")
                plan = run / branch / "plans/Q1.json"
                write(plan, {"plan": "frozen"})
                freeze(run / branch / "plans/frozen/V1.json", {"Q1": sha(plan)})
            elif "answer" in command:
                path = Path(value("--output-dir")) / "predictions.jsonl"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"question_id": "Q1", "status": "ok", "prediction": "answer"}) + "\n")
            else:
                path = Path(value("--output-dir")) / "run_fixture/scores.jsonl"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"question_id": "Q1", "status": "ok", "prediction": "answer",
                    "max_score": 4, "score": 1, "normalized_score": 0.25}) + "\n")
            write(folder / "task.json", {"status": "ok"})
            return {"status": "ok"}

        with patch("experiments.zyf.matched_pilot.run_command", side_effect=fake_command):
            execute(args, run, config, [q])
        self.assertTrue(summary(run, [q])["complete"])
        self.assertTrue(read(run / "gate.json")["passed"])
        answers = [cmd for cmd in commands if "answer" in cmd]
        self.assertEqual(len(answers), 3)
        event_answers = [cmd for cmd in answers if "--retrieval" in cmd]
        self.assertEqual(event_answers[0][event_answers[0].index("--retrieval") + 1], "graph")
        self.assertEqual(event_answers[1][event_answers[1].index("--retrieval") + 1], "progressive")
        for flag in ("--memory-dir", "--plans-dir", "--embedding-cache-dir"):
            self.assertEqual(event_answers[0][event_answers[0].index(flag) + 1], event_answers[1][event_answers[1].index(flag) + 1])
        self.assertEqual(event_answers[1][event_answers[1].index("--progressive-routing") + 1], "none")
        for command in commands:
            self.assertNotIn("--limit", command)
            if "--data-path" in command:
                self.assertEqual(command[command.index("--data-path") + 1], str(run / "questions/V1.json"))


if __name__ == "__main__":
    unittest.main()
