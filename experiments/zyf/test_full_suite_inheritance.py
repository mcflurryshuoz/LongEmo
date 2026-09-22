import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.zyf.full_suite_inheritance import snapshot_parent
from experiments.zyf.matched_pilot import digest, memory_root, read, sha, source_hash, write


class FullSuiteInheritanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.parent, self.child = self.root / "pilot", self.root / "full"
        self.repo = Path(__file__).resolve().parents[2]
        self.parent_questions = [self.question("Q1"), self.question("Q2")]
        self.full_questions = copy.deepcopy(self.parent_questions) + [self.question("Q3"), self.question("Q4", "V2")]
        self.config = {"mode": "pilot", "questions_sha256": digest(self.parent_questions),
            "repos": {"event": str(self.repo), "noevent": str(self.repo)},
            "source_hashes": {b: source_hash(self.repo) for b in ("event", "noevent")},
            "media": {"V1": {"video_sha256": "video-sha", "subtitles_sha256": "subtitle-sha"}},
            "window_seconds": 20, "padding": 2, "fps": 1, "max_frames": 24, "max_pixels": 200704,
            "model": "fixture", "base_url": "https://example.invalid/v1", "timeout": 100,
            "evidence_chars": 48000, "progressive_routing": "none"}
        write(self.parent / "configuration.json", self.config)
        write(self.parent / "questions.json", self.parent_questions)
        write(self.parent / "questions/V1.json", self.parent_questions)
        write(self.parent / "process.json", {"fixture": "the real lock is separately exercised"})
        for branch in ("event", "noevent"):
            self.memory(branch)

    def tearDown(self):
        self.temporary.cleanup()

    def question(self, qid, video="V1"):
        return {"question_id": qid, "video_id": video, "granularity": "episode",
                "type": "emotional trajectory", "question": "How did the emotion change? " + qid,
                "rubric": {"scores": {"0": "incorrect", "4": "correct"}}}

    def task(self, stage, key, status="ok"):
        write(self.parent / "tasks" / stage / key / "task.json", {"status": status})

    def memory(self, branch):
        from methods.longemo import memory as event_memory, noevent_memory
        module = event_memory if branch == "event" else noevent_memory
        folder = memory_root(self.parent, branch, "V1") / "V1"
        clients = {"model": {"model": "fixture"}, "audio_observer": {"model": "fixture-audio"}}
        config = {**self.config["media"]["V1"], **clients, "window_seconds": 20, "padding": 2,
                  "media": {"fps": 1, "max_frames": 24, "max_pixels": 200704, "with_audio": True},
                  "allow_revisions": False, "code_hash": self.config["source_hashes"][branch]}
        write(folder / "manifest.json", {"configuration": config, "fingerprint": digest(config)})
        payload = {"entities": [], "observations": [], "events": [], "relations": [], "corrections": []}
        if branch == "noevent":
            payload = {"entities": [], "observations": [], "summary": "Quiet scene", "actions": [],
                       "objects": [], "signals": [], "participants": [], "emotion_cues": []}
        audio = {"model": clients["audio_observer"], "input_fingerprint": "audio-input", "result": {"observations": []}}
        write(folder / "audio/W00001.json", audio)
        sampling = {"audio_observer": {"source_sha256": sha(folder / "audio/W00001.json"), "input_fingerprint": "audio-input"}}
        context = {"video_id": "V1", "window_id": "W00001", "core_interval": [0, 20], "media_interval": [0, 20]}
        write(folder / "windows/W00001.json", {"input": context, "sampling": sampling, "perception": payload})
        memory = module.apply_window(module.empty_memory("V1", 20, "video-sha"), payload,
            window_id="W00001", core=[0, 20], media=[0, 20], metadata=sampling)
        memory.update(build_fingerprint=digest(config), complete=True)
        write(folder / "memory.json", memory)
        write(self.parent / branch / "frozen_memories/V1.json", {"memory_sha256": sha(folder / "memory.json")})
        self.task("build", branch + "-V1")
        return memory

    def plan(self, branch, qid="Q1"):
        plan = {"mode": "trajectory", "entity_terms": [], "target_terms": [], "query_terms": []}
        write(self.parent / branch / "plans" / (qid + ".json"), {"input_fingerprint": "a" * 64, "plan": plan})

    def prediction(self, condition, qid="Q1"):
        branch = "noevent" if condition == "noevent" else "event"
        memory_path = memory_root(self.parent, branch, "V1") / "V1/memory.json"
        memory = read(memory_path)
        question = next(q for q in self.parent_questions if q["question_id"] == qid)
        row = {k: question[k] for k in ("question_id", "video_id", "granularity", "type", "question")}
        row.update(status="ok", prediction="A valid answer.", memory_fingerprint=memory["build_fingerprint"],
                   memory_sha256=digest(memory))
        folder = self.parent / "answers" / condition / "V1"
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "predictions.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        config = {"questions": [{k: q[k] for k in ("question_id", "video_id", "question", "type")}
                                for q in self.parent_questions],
                  "memories": {"V1": sha(memory_path)}, "code_hash": self.config["source_hashes"][branch],
                  "evidence_chars": 48000, "representation": "window_records",
                  "retrieval": "graph" if condition == "base" else "progressive",
                  "progressive_routing": "none", "hybrid_graph_tasks": []}
        write(folder / "manifest.json", {"configuration": config, "fingerprint": digest(config)})
        write(folder / "evaluation_questions.json", self.parent_questions)
        self.task("answer", condition + "-V1")
        return row

    def score(self, condition, qid="Q1", accept=False, run_name="run_1", score=0):
        question = next(q for q in self.parent_questions if q["question_id"] == qid)
        row = {k: question[k] for k in ("question_id", "video_id", "granularity", "type")}
        row.update(status="ok", prediction="A valid answer.", score=score, max_score=4, normalized_score=score / 4)
        path = self.parent / "scores" / condition / "V1" / run_name / "scores.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(row) + "\n")
        if accept:
            write(self.parent / "accepted" / condition / (qid + ".json"),
                  {"score": row, "source": str(path), "prediction_sha256": digest(row["prediction"])})
        self.task("score", condition + "-V1")
        return row

    def snapshot(self):
        with patch("experiments.zyf.full_suite_inheritance.stopped_parent") as stopped:
            result = snapshot_parent(self.parent, self.child, self.full_questions)
            stopped.assert_called_once_with(self.parent)
            return result

    def test_imports_zero_first_score_and_never_rescores_later_high_score(self):
        self.prediction("method")
        first = self.score("method", accept=True, score=0)
        self.score("method", run_name="run_2", score=4)
        index = self.snapshot()
        q = index["conditions"]["method"]["questions"]["Q1"]
        self.assertEqual(q["accepted"]["envelope"]["score"], first)
        self.assertEqual(q["score_state"], "success")
        self.assertTrue(Path(q["prediction"]["artifact"]["path"]).is_file())

    def test_successful_raw_judgment_before_accept_is_inherited_not_repeated(self):
        self.prediction("base")
        row = self.score("base")
        index = self.snapshot()
        q = index["conditions"]["base"]["questions"]["Q1"]
        self.assertIsNone(q["accepted"]["artifact"])
        self.assertEqual(q["accepted"]["envelope"]["score"], row)
        self.assertFalse((self.parent / "accepted/base/Q1.json").exists())

    def test_successful_prediction_without_started_judge_only_needs_first_score(self):
        self.prediction("noevent")
        index = self.snapshot()
        q = index["conditions"]["noevent"]["questions"]["Q1"]
        self.assertEqual((q["answer_state"], q["score_state"]), ("success", "unstarted"))
        self.assertFalse(q["attempted_score"])

    def test_claimed_missing_results_block_old_questions_but_not_new_questions(self):
        self.plan("event")
        self.task("plan", "event-V1", "error")
        self.task("answer", "method-V1", "needs_audit")
        self.task("score", "method-V1", "error")
        index = self.snapshot()
        plans = index["branches"]["event"]["videos"]["V1"]["plans"]
        self.assertEqual([plans[q]["state"] for q in ("Q1", "Q2", "Q3")], ["success", "blocked", "unstarted"])
        questions = index["conditions"]["method"]["questions"]
        self.assertEqual((questions["Q2"]["answer_state"], questions["Q2"]["score_state"]), ("blocked", "blocked"))
        self.assertEqual((questions["Q3"]["answer_state"], questions["Q3"]["score_state"]), ("unstarted", "unstarted"))

    def test_failed_build_is_not_given_a_fresh_budget(self):
        self.task("build", "noevent-V1", "error")
        index = self.snapshot()
        self.assertEqual(index["branches"]["noevent"]["videos"]["V1"]["state"], "blocked")
        self.assertEqual(index["branches"]["noevent"]["videos"]["V2"]["state"], "unstarted")

    def test_parent_late_addition_and_snapshot_tamper_are_rejected(self):
        self.prediction("base")
        self.snapshot()
        self.snapshot()
        write(self.parent / "base/late.json", {"not_in_allowlist": True})
        self.plan("event", "Q2")
        with self.assertRaisesRegex(ValueError, "parent artifacts"):
            self.snapshot()
        (self.parent / "event/plans/Q2.json").unlink()
        copied = self.child / "inheritance/parent/questions.json"
        copied.write_text("[]")
        with self.assertRaisesRegex(ValueError, "snapshot changed"):
            self.snapshot()

    def test_duplicate_prediction_and_modified_accepted_score_are_rejected(self):
        self.prediction("base")
        self.prediction("base")
        with self.assertRaisesRegex(ValueError, "duplicate question ID"):
            self.snapshot()
        path = self.parent / "answers/base/V1/predictions.jsonl"
        path.write_text(path.read_text().splitlines()[0] + "\n")
        self.score("base", accept=True)
        accepted_path = self.parent / "accepted/base/Q1.json"
        accepted = read(accepted_path)
        accepted["score"].update(score=4, normalized_score=1)
        write(accepted_path, accepted)
        with self.assertRaisesRegex(ValueError, "original official row"):
            self.snapshot()

    def test_changed_question_source_or_retrieval_manifest_is_rejected(self):
        full = copy.deepcopy(self.full_questions)
        self.full_questions[0]["question"] = "Changed question"
        with self.assertRaisesRegex(ValueError, "exact parent questions"):
            self.snapshot()
        self.full_questions = full
        self.prediction("method")
        path = self.parent / "answers/method/V1/manifest.json"
        manifest = read(path)
        manifest["configuration"]["progressive_routing"] = "task"
        manifest["fingerprint"] = digest(manifest["configuration"])
        write(path, manifest)
        with self.assertRaisesRegex(ValueError, "pure ablation"):
            self.snapshot()

    def test_lock_is_required_and_active_parent_prevents_any_snapshot(self):
        with patch("experiments.zyf.full_suite_inheritance.stopped_parent", side_effect=ValueError("parent coordinator still owns its lock")):
            with self.assertRaisesRegex(ValueError, "owns its lock"):
                snapshot_parent(self.parent, self.child, self.full_questions)
        self.assertFalse((self.child / "inheritance/parent_index.json").exists())

    def test_config_change_refused_and_parent_sources_checked(self):
        changed = {**self.config, "evidence_chars": 1000}
        with patch("experiments.zyf.full_suite_inheritance.stopped_parent"):
            with self.assertRaisesRegex(ValueError, "evidence_chars"):
                snapshot_parent(self.parent, self.child, self.full_questions, changed)
        with patch("experiments.zyf.full_suite_inheritance.source_hash", return_value="changed"):
            with self.assertRaisesRegex(ValueError, "frozen hash"):
                self.snapshot()

    def embedding_ledger(self, rows):
        path = self.parent / "event/embeddings/api/calls.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_unmapped_embedding_failure_blocks_only_parent_attempted_video_new_answers(self):
        self.prediction("base")
        self.embedding_ledger([{"purpose": "embedding_documents", "request_hash": "a" * 64,
            "status": "error", "error_type": "TimeoutError", "attempt": 3}])
        index = self.snapshot()
        branch = index["branches"]["event"]
        self.assertEqual(branch["embedding_guard"]["scope"], "parent_attempted_videos")
        self.assertEqual(branch["embedding_guard"]["blocked_videos"], ["V1"])
        self.assertEqual(branch["videos"]["V1"]["embedding_guard"]["state"], "needs_audit")
        self.assertEqual(branch["videos"]["V2"]["embedding_guard"]["state"], "clear")
        question = index["conditions"]["base"]["questions"]["Q1"]
        self.assertEqual((question["answer_state"], question["score_state"]), ("success", "unstarted"))
        self.assertEqual(index["branches"]["noevent"]["embedding_guard"]["state"], "clear")

    def test_embedding_success_resolves_earlier_failure_and_query_failure_is_not_document_failure(self):
        self.task("answer", "method-V1", "error")
        self.embedding_ledger([
            {"purpose": "embedding_documents", "request_hash": "a" * 64, "status": "error", "attempt": 1},
            {"purpose": "embedding_documents", "request_hash": "a" * 64, "status": "ok", "attempt": 2},
            {"purpose": "embedding_query", "request_hash": "b" * 64, "status": "error", "attempt": 3}])
        index = self.snapshot()
        self.assertEqual(index["branches"]["event"]["embedding_guard"]["state"], "clear")

    def test_unknown_embedding_result_does_not_block_untouched_parent_or_new_video(self):
        self.embedding_ledger([{"purpose": "embedding_documents", "request_hash": "a" * 64, "status": "unknown"}])
        index = self.snapshot()
        guard = index["branches"]["event"]["embedding_guard"]
        self.assertEqual(guard["state"], "needs_audit")
        self.assertEqual(guard["blocked_videos"], [])
        self.assertTrue(all(v["embedding_guard"]["state"] == "clear" for v in index["branches"]["event"]["videos"].values()))


if __name__ == "__main__":
    unittest.main()
