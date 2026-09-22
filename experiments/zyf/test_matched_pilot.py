import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.zyf.matched_pilot import accept_scores, execute, freeze, memory_root, parser, read, run_command, run_lock, select_questions, sha, stage_exit_code, summary, video_tasks, write


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
        config = {"media": {"V1": {}}, "repos": {"event": str(self.root), "noevent": str(self.root)},
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
