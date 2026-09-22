import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.zyf.full_suite import (media_status, parent_wait, prepare, producer_finished,
    recover_stopped_scores, run_phase, summarize, verify_runtime_sources, video_pipeline, write_rows)
from experiments.zyf.matched_pilot import digest, freeze, memory_root, read, sha, write


class FullSuiteTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def questions(self, count=4):
        return [{"question_id": f"Q{i}", "video_id": "V1", "granularity": "episode", "type": "emotion trajectory",
                 "question": f"question {i}", "rubric": {"scores": {"0": "no", "2": "yes"}}} for i in range(1, count + 1)]

    def config(self):
        return {"model": "gpt-6-astra", "base_url": "https://example.invalid/v1", "timeout": 1800,
                "perception_model": "gemini-3.8-flash", "perception_base_url": "https://example.invalid/v1",
                "perception_timeout": 240, "audio_model": "gemini-3.8-flash", "audio_base_url": "https://example.invalid/v1",
                "embedding_model": "test-embedding", "embedding_base_url": "https://example.invalid/v1",
                "stage_tries": {"build": 3, "plan": 3, "answer": 3, "score": 1},
                "window_seconds": 20, "padding": 2, "fps": 1, "max_frames": 24, "max_pixels": 200704,
                "evidence_chars": 48000, "progressive_rounds": 3, "progressive_anchor_k": 4,
                "video_count": 1, "repos": {"event": str(self.root), "noevent": str(self.root)},
                "videos_dir": str(self.root / "videos"), "subtitles_dir": str(self.root / "subtitles"),
                "media": {"V1": {}}, "execution": {"poll_seconds": .01, "media_wait_timeout": 1,
                    "parent_wait_timeout": 1, "producer_done": None}}

    def test_prepare_freezes_full_cohort_before_missing_media_arrives(self):
        questions = self.questions()
        parent, run = self.root / "parent", self.root / "run"
        write(parent / "questions.json", questions[:2])
        video_hash, subtitle_hash = "a" * 64, "b" * 64
        old = {**self.config(), "questions_sha256": digest(questions[:2]), "question_count": 2,
               "source_hashes": {"event": "same", "noevent": "same"},
               "media": {"V1": {"video_sha256": video_hash, "subtitles_sha256": subtitle_hash}}}
        write(parent / "configuration.json", old)
        source = self.root / "questions.json"
        write(source, questions)
        manifest = self.root / "media.json"
        write(manifest, {"schema_version": 1, "videos": {"V1": {"video_sha256": video_hash,
            "subtitles_sha256": subtitle_hash, "video_bytes": 100, "subtitles_bytes": 3}}})
        args = SimpleNamespace(parent_run=str(parent), questions=str(source), expected_questions=4, expected_videos=1,
            media_manifest=str(manifest), method_repo=str(self.root), noevent_repo=str(self.root), python=sys.executable,
            videos_dir=str(self.root / "not-yet-uploaded"), subtitles_dir=str(self.root / "not-yet-uploaded-subtitles"),
            workers=12, question_workers=2, poll_seconds=1, parent_wait_timeout=30, media_wait_timeout=30, producer_done=None)
        with patch("experiments.zyf.full_suite.source_hash", return_value="same"):
            config, frozen = prepare(args, run)
            self.assertEqual(len(frozen), 4)
            self.assertEqual(config["question_count"], 4)
            self.assertEqual(media_status(config, "V1"), "waiting")
            args.expected_questions = 5
            with self.assertRaisesRegex(ValueError, "full episode cohort"):
                prepare(args, run)

    def test_media_missing_waits_integrity_mismatch_fails_and_done_is_bound_to_manifest(self):
        config = self.config()
        video = Path(config["videos_dir"]) / "V1.mp4"
        subtitle = Path(config["subtitles_dir"]) / "V1.json"
        video.parent.mkdir(); subtitle.parent.mkdir()
        expected_video, expected_subtitle = b"video", b"[]\n"
        import hashlib
        config["media"]["V1"] = {"video_sha256": hashlib.sha256(expected_video).hexdigest(), "video_bytes": len(expected_video),
            "subtitles_sha256": hashlib.sha256(expected_subtitle).hexdigest(), "subtitles_bytes": len(expected_subtitle)}
        self.assertEqual(media_status(config, "V1"), "waiting")
        video.write_bytes(b"wrong")
        self.assertEqual(media_status(config, "V1"), "integrity_failed")
        video.write_bytes(expected_video); subtitle.write_bytes(expected_subtitle)
        self.assertEqual(media_status(config, "V1"), "ready")
        marker = self.root / "done.json"
        config["media_manifest_sha256"] = "manifest"
        config["execution"]["producer_done"] = str(marker)
        self.assertFalse(producer_finished(config))
        write(marker, {"complete": True, "media_manifest_sha256": "wrong"})
        with self.assertRaisesRegex(ValueError, "frozen complete media manifest"):
            producer_finished(config)
        write(marker, {"complete": True, "media_manifest_sha256": "manifest"})
        self.assertTrue(producer_finished(config))

    def test_parent_wait_never_proceeds_while_snapshot_reports_active_workers(self):
        run = self.root / "run"; run.mkdir()
        config = self.config(); config["parent_run"] = str(self.root / "parent")
        fake = SimpleNamespace(snapshot_parent=lambda *a, **kw: None)
        with patch.dict(sys.modules, {"experiments.zyf.full_suite_inheritance": fake}), patch.object(
                fake, "snapshot_parent", side_effect=[ValueError("parent process is still active: coordinator"), {"verified": True}]) as audit, patch(
                "experiments.zyf.full_suite.time.sleep") as sleeping:
            result = parent_wait(SimpleNamespace(), run, config, self.questions())
        self.assertTrue(result["verified"])
        self.assertEqual(audit.call_count, 2)
        sleeping.assert_called_once()

    def test_runtime_sources_rechecked_after_wait_before_any_new_stage(self):
        config = {**self.config(), "suite_sha256": "suite", "suite_support_hashes": {"matched_pilot.py": "support"},
                  "source_hashes": {"event": "event-code", "noevent": "noevent-code"}}
        with patch("experiments.zyf.full_suite.sha", side_effect=["suite", "support"]), patch(
                "experiments.zyf.full_suite.source_hash", side_effect=["event-code", "noevent-code"]):
            verify_runtime_sources(config)
        with patch("experiments.zyf.full_suite.sha", side_effect=["suite", "support"]), patch(
                "experiments.zyf.full_suite.source_hash", return_value="edited-during-wait"):
            with self.assertRaisesRegex(ValueError, "source changed after preparation"):
                verify_runtime_sources(config)
        with patch("experiments.zyf.full_suite.sha", side_effect=["suite", "edited-support"]):
            with self.assertRaisesRegex(ValueError, "support source changed"):
                verify_runtime_sources(config)

    def test_crash_after_judge_write_imports_first_score_without_restarting_task(self):
        run = self.root / "run"
        questions = self.questions(1)
        write(run / "questions.json", questions); write(run / "questions/V1.json", questions)
        prediction = {"question_id": "Q1", "video_id": "V1", "status": "ok", "prediction": "first answer"}
        score = {"question_id": "Q1", "status": "ok", "score": 0, "max_score": 2,
                 "normalized_score": 0, "prediction": "first answer"}
        write_rows(run / "answers/noevent/V1/predictions.jsonl", [prediction])
        write_rows(run / "score_inputs/noevent/V1.jsonl", [prediction])
        write(run / "stage_questions/score/noevent-V1.json", questions)
        task = run / "tasks/score/noevent-V1/task.json"
        write(task, {"status": "running", "child": {"pid": 123, "start_ticks": 456}})
        score_file = run / "scores/noevent/V1/run_fixture/scores.jsonl"
        write_rows(score_file, [score])
        index = {"conditions": {c: {"questions": {}} for c in ("noevent", "base", "method")}}
        before = sha(task)
        with patch("experiments.zyf.full_suite.run_command") as command:
            recover_stopped_scores(run, self.config(), index)
            recover_stopped_scores(run, self.config(), index)
            command.assert_not_called()
        self.assertEqual(read(run / "accepted/noevent/Q1.json")["score"], score)
        self.assertEqual(sha(task), before)
        self.assertTrue(read(run / "offline_score_imports.json")["verified_stopped_children"])
        # The original successful row cannot be silently replaced or duplicated.
        write_rows(score_file, [score, score])
        with self.assertRaisesRegex(ValueError, "duplicate question ID"):
            recover_stopped_scores(run, self.config(), index)

    def test_orphan_judge_output_requires_frozen_invocation(self):
        run = self.root / "run"
        write(run / "questions.json", self.questions(1))
        write_rows(run / "scores/noevent/V1/run_fixture/scores.jsonl", [])
        with self.assertRaisesRegex(ValueError, "orphan score output"):
            recover_stopped_scores(run, self.config(), {"conditions": {}})

    def test_per_video_pipeline_keeps_first_scores_and_answers_and_locks_only_failed_qids(self):
        run = self.root / "run"
        questions = self.questions()
        write(run / "questions.json", questions); write(run / "questions/V1.json", questions)
        config = self.config(); write(run / "configuration.json", config)
        credentials = self.root / "credentials.json"; write(credentials, {"MODEL_API_KEY": "test-only"})
        args = SimpleNamespace(runtime=str(self.root), credential_file=str(credentials), python=sys.executable, question_workers=2)
        memory = memory_root(run, "noevent", "V1") / "V1/memory.json"
        write(memory, {"complete": True})
        freeze(run / "noevent/frozen_memories/V1.json", {"memory_sha256": sha(memory)})
        for qid in ("Q1", "Q2", "Q3"):
            write(run / "noevent/plans" / (qid + ".json"), {"plan": qid})
        prediction = lambda qid: {"question_id": qid, "video_id": "V1", "status": "ok", "prediction": "answer " + qid}
        first = {"question_id": "Q1", "status": "ok", "score": 0, "max_score": 2, "normalized_score": 0, "prediction": "answer Q1"}
        freeze(run / "accepted/noevent/Q1.json", {"score": first, "source": "parent-original", "prediction_sha256": digest("answer Q1")})
        parent_rows = {
            "Q1": {"answer_state": "success", "score_state": "success", "prediction": {"row": prediction("Q1")}},
            "Q2": {"answer_state": "success", "score_state": "unstarted", "prediction": {"row": prediction("Q2")}},
            "Q3": {"answer_state": "blocked", "score_state": "unstarted", "prediction": None},
            "Q4": {"answer_state": "unstarted", "score_state": "unstarted", "prediction": None}}
        index = {"branches": {"noevent": {"videos": {"V1": {"state": "reusable",
            "plans": {qid: {"state": "success" if qid != "Q4" else "unstarted"} for qid in parent_rows}}}}},
            "conditions": {"noevent": {"questions": parent_rows}}}
        calls = []

        def fake_stage(args, run, config, stage, key, command, repo, env):
            flag = lambda name: command[command.index(name) + 1]
            selected = read(flag("--questions") if stage == "plan" else flag("--data-path"))
            qids = [q["question_id"] for q in selected]
            calls.append((stage, qids))
            if stage == "plan":
                for qid in qids:
                    write(run / "noevent/plans" / (qid + ".json"), {"plan": qid})
            elif stage == "answer":
                write_rows(Path(flag("--output-dir")) / "predictions.jsonl", [prediction(qid) for qid in qids])
            elif stage == "score":
                self.assertEqual(flag("--tries"), "1")
                write_rows(Path(flag("--output-dir")) / "run_fixture/scores.jsonl", [
                    {"question_id": qid, "status": "ok", "score": 1, "max_score": 2, "normalized_score": .5,
                     "prediction": "answer " + qid} for qid in qids])
            return {"status": "ok"}

        with patch("experiments.zyf.full_suite.stage_run", side_effect=fake_stage):
            video_pipeline(args, run, config, index, "noevent", "V1")
        self.assertEqual(calls, [("plan", ["Q4"]), ("answer", ["Q4"]), ("score", ["Q2", "Q4"])])
        self.assertEqual(read(run / "accepted/noevent/Q1.json")["score"]["score"], 0)
        self.assertEqual(read(run / "pipelines/noevent/V1.json")["status"], "partial")
        self.assertEqual(read(run / "status.json")["conditions"]["noevent"]["scored"], 3)

    def test_scheduler_does_not_start_api_work_for_missing_or_parent_failed_media(self):
        run = self.root / "run"
        questions = [{**self.questions(1)[0], "question_id": "Q" + str(i), "video_id": "V" + str(i)} for i in range(1, 4)]
        write(run / "questions.json", questions)
        config = self.config(); config.update(media={"V1": {}, "V2": {}, "V3": {}}, video_count=3)
        index = {"branches": {"noevent": {"videos": {"V1": {"state": "unstarted"}, "V2": {"state": "unstarted"},
                                                           "V3": {"state": "blocked"}}}}}
        calls = []

        def completed(args, run, config, index, branch, video):
            calls.append(video)
            write(run / "pipelines" / branch / (video + ".json"), {"status": "complete"})

        with patch("experiments.zyf.full_suite.media_status", side_effect=lambda config, video: "ready" if video == "V1" else "waiting"), patch(
                "experiments.zyf.full_suite.producer_finished", return_value=True), patch(
                "experiments.zyf.full_suite.video_pipeline", side_effect=completed):
            run_phase(SimpleNamespace(workers=2), run, config, index, "noevent")
        self.assertEqual(calls, ["V1"])
        self.assertEqual(read(run / "pipelines/noevent/V2.json")["status"], "media_missing")
        self.assertEqual(read(run / "pipelines/noevent/V3.json")["status"], "blocked_parent_build")

    def test_unresolved_parent_document_embedding_blocks_new_answer_but_not_first_judgment(self):
        run = self.root / "run"
        questions = self.questions(2)
        write(run / "questions.json", questions); write(run / "questions/V1.json", questions)
        config = self.config(); write(run / "configuration.json", config)
        credentials = self.root / "credentials.json"; write(credentials, {"MODEL_API_KEY": "test-only"})
        args = SimpleNamespace(runtime=str(self.root), credential_file=str(credentials), python=sys.executable, question_workers=2)
        memory = memory_root(run, "noevent", "V1") / "V1/memory.json"
        write(memory, {"complete": True})
        freeze(run / "noevent/frozen_memories/V1.json", {"memory_sha256": sha(memory)})
        for q in questions:
            write(run / "noevent/plans" / (q["question_id"] + ".json"), {"plan": q["question_id"]})
        index = {"branches": {"noevent": {"videos": {"V1": {"state": "reusable",
            "embedding_guard": {"state": "needs_audit", "reason": "parent document request unresolved"},
            "plans": {q["question_id"]: {"state": "success"} for q in questions}}}}},
            "conditions": {"noevent": {"questions": {"Q1": {"answer_state": "success", "score_state": "unstarted",
                "prediction": {"row": {"question_id": "Q1", "status": "ok", "prediction": "first"}}},
                "Q2": {"answer_state": "unstarted", "score_state": "unstarted"}}}}}
        calls = []

        def fake_stage(args, run, config, stage, key, command, repo, env):
            calls.append(stage)
            self.assertEqual(stage, "score")
            selected = read(command[command.index("--data-path") + 1])
            self.assertEqual([q["question_id"] for q in selected], ["Q1"])
            write_rows(run / "scores/noevent/V1/run_fixture/scores.jsonl", [{"question_id": "Q1", "status": "ok",
                "prediction": "first", "score": 1, "max_score": 2, "normalized_score": .5}])
            return {"status": "ok"}

        with patch("experiments.zyf.full_suite.stage_run", side_effect=fake_stage):
            video_pipeline(args, run, config, index, "noevent", "V1")
        self.assertEqual(calls, ["score"])
        status = read(run / "pipelines/noevent/V1.json")
        self.assertEqual(status["embedding_blocked_answers"], {"noevent": ["Q2"]})
        self.assertEqual(status["scored"], {"noevent": 1})


if __name__ == "__main__":
    unittest.main()
