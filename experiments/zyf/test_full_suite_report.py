import json
from pathlib import Path
import tempfile
import unittest

from experiments.zyf.full_suite_report import (Reader, classify_failure, digest, embedding_failure,
    grouped_metrics, missing_status, paired_metrics, render, snapshot, source_hash, source_series, validate_score)


def write(path, value, jsonl=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(("".join(json.dumps(x) + "\n" for x in value) if jsonl else json.dumps(value)))


class FullSuiteReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, folder):
        q = {"question_id": "Q1", "video_id": "V1", "type": "emotion trajectory", "rubric": {"scores": {"0": "bad", "2": "good"}}}
        row = {"question_id": "Q1", "video_id": "V1", "type": q["type"], "status": "ok", "score": 0,
               "max_score": 2, "normalized_score": 0, "prediction": "private answer must not be exported"}
        source = folder / "scores/noevent/V1/run_fixture/scores.jsonl"; write(source, [row], True)
        accepted = {"score": row, "source": str(source), "prediction_sha256": digest(row["prediction"])}
        path = folder / "accepted/noevent/Q1.json"; write(path, accepted)
        prediction = {k: row[k] for k in ("question_id", "video_id", "type", "status", "prediction")}
        pred_path = folder / "answers/noevent/V1/predictions.jsonl"; write(pred_path, [prediction], True)
        memory = folder / "noevent/videos/V1/memory/V1/memory.json"; write(memory, {"complete": True})
        write(folder / "noevent/frozen_memories/V1.json", {"memory_sha256": Reader().sha(memory)})
        plan = folder / "noevent/plans/Q1.json"; write(plan, {"plan": "fixture"})
        write(folder / "noevent/plans/frozen/V1.json", {"Q1": Reader().sha(plan)})
        write(folder / "answers/noevent/V1/embedding_indexes.json", {"V1": {"signature": "same"}})
        write(folder / "answers/noevent/V1/traces/Q1.json", {})
        return q, row, source, path, accepted, prediction, pred_path

    def test_zero_first_score_validated_and_raw_sensitive_content_not_exported(self):
        run = self.root / "run"; q, row, source, path, *_ = self.fixture(run)
        result = validate_score(Reader(), run, {}, "noevent", path, q)
        self.assertEqual(result["normalized_score_100"], 0)
        self.assertEqual(result["origin"], "full_suite")
        self.assertNotIn(row["prediction"], json.dumps(result))
        write(source, [{**row, "score": 2, "normalized_score": 1}], True)
        with self.assertRaisesRegex(ValueError, "raw score"):
            validate_score(Reader(), run, {}, "noevent", path, q)

    def test_inherited_score_accepts_snapshot_source_and_preserves_first_zero(self):
        parent = self.root / "snapshot"; run = self.root / "full"
        q, row, source, _, value, prediction, pred_path = self.fixture(parent)
        accepted = run / "accepted/noevent/Q1.json"; write(accepted, value)
        artifact = lambda p: {"path": str(p), "source": str(p), "sha256": Reader().sha(p)}
        index = {"snapshot_dir": str(parent), "conditions": {"noevent": {"questions": {"Q1": {
            "accepted": {"envelope": value, "artifact": None, "score_artifact": artifact(source)},
            "prediction": {"row": prediction, "artifact": artifact(pred_path)}}}}}}
        result = validate_score(Reader(), run, index, "noevent", accepted, q)
        self.assertEqual(result["origin"], "parent_pilot"); self.assertEqual(result["score"], 0)
        write(accepted, {**value, "prediction_sha256": "bad"})
        with self.assertRaisesRegex(ValueError, "prediction hash"):
            validate_score(Reader(), run, index, "noevent", accepted, q)

    def test_pairing_intersects_question_ids_before_computing_means(self):
        qs = [{"question_id": "Q" + str(i)} for i in (1, 2, 3)]
        accepted = {"base": {"Q1": {"normalized_score_100": 0}, "Q2": {"normalized_score_100": 100}},
                    "method": {"Q1": {"normalized_score_100": 100}, "Q3": {"normalized_score_100": 0}}, "noevent": {}}
        all_scores = grouped_metrics(accepted, qs)
        self.assertEqual(all_scores["base"]["mean_scored"], 50)
        self.assertEqual(all_scores["method"]["mean_scored"], 50)
        pair = paired_metrics(accepted, qs, ("base", "method"))
        self.assertEqual(pair["count"], 1); self.assertEqual(pair["method_minus_base"], 100)
        self.assertIsNone(paired_metrics(accepted, qs, tuple(accepted))["method_minus_base"])

    def test_failure_evidence_does_not_infer_filter_from_http_or_runtime(self):
        self.assertEqual(classify_failure({"http_status": 200, "service_error_code": "content_filter"}), "content_filter")
        self.assertEqual(classify_failure({"http_status": 428}), "http_428_unclassified")
        self.assertEqual(classify_failure({"error_type": "RuntimeError", "failure_stage": "generate"}), "generate_runtime_error_unknown")
        self.assertEqual(classify_failure({"http_status": 400}), "other")

    def test_embedding_classification_requires_that_videos_terminal_stack(self):
        run = self.root / "run"; task = run / "tasks/answer/noevent-V1/task.json"
        output = task.parent / "output.log"; output.parent.mkdir(parents=True)
        output.write_text('WindowIndex(...)\n encoder.encode(chunks)\n ServiceError: HTTP 400\n')
        state = {"status": "error", "started_unix": 1, "finished_unix": 3}
        failure = embedding_failure(Reader(), run, "noevent", task, state)
        self.assertEqual(failure["category"], "embedding_failed")
        self.assertEqual(failure["http_status"], 400); self.assertFalse(failure["gpt_answer_started"])
        self.assertNotIn("ServiceError", json.dumps(failure))
        self.assertIsNone(embedding_failure(Reader(), run, "noevent", task, {"status": "running"}))
        output.write_text('unrelated HTTP 400')
        self.assertIsNone(embedding_failure(Reader(), run, "noevent", task, state))

    def test_parent_build_block_is_distinct_from_new_failure(self):
        q = {"question_id": "Q1", "video_id": "V1"}
        index = {"branches": {"noevent": {"videos": {"V1": {"state": "blocked"}}}}}
        info = {"status": "pending", "memory_complete": False}
        a = missing_status(Reader(), self.root, index, "noevent", q, info, {})
        b = missing_status(Reader(), self.root, {}, "noevent", q, {**info, "status": "build_failed"}, {})
        self.assertEqual(a["status"], "parent_build_blocked"); self.assertEqual(b["status"], "build_failed")

    def test_series_uses_only_explicit_series_episode_source(self):
        self.assertEqual(source_series("jiayouernv S01E22"), "Home with Kids")
        self.assertEqual(source_series("Friends S01E01"), "Friends")
        self.assertEqual(source_series("https://youtube.com/watch?v=Friends"), "unclassified")

    def test_active_ledger_partial_tail_is_ignored_but_complete_bad_record_rejected(self):
        path = self.root / "calls.jsonl"; path.write_text('{"status":"ok"}\n{"status":')
        self.assertEqual(Reader().rows(path, active=True), [{"status": "ok"}])
        path.write_text('{"status":"ok"}\n{"status":\n')
        with self.assertRaises(json.JSONDecodeError):
            Reader().rows(path, active=True)

    def test_full_read_only_snapshot_has_all_558_statuses_without_exporting_questions(self):
        run, repo = self.root / "run", self.root / "repo"
        secret = "QUESTION_AND_GOLD_MUST_NOT_APPEAR"
        qs = [{"question_id": f"Q{i}", "video_id": f"V{i % 141}", "type": "emotion trajectory", "question": secret,
               "gold": secret, "rubric": {"scores": {"0": secret, "1": secret}}, "source": {"from": "Friends S01E01"}}
              for i in range(558)]
        write(run / "questions.json", qs)
        write(repo / "methods/longemo/common.py", {"test": True})
        for name in ("full_suite.py", "matched_pilot.py", "full_suite_inheritance.py"):
            write(repo / "experiments/zyf" / name, {"test": name})
        r = Reader()
        cfg = {"question_count": 558, "video_count": 141, "questions_sha256": digest(qs), "window_seconds": 20,
               "repos": {"event": str(repo), "noevent": str(repo)}, "source_hashes": {b: source_hash(r, repo) for b in ("event", "noevent")},
               "suite_sha256": r.sha(repo / "experiments/zyf/full_suite.py"),
               "suite_support_hashes": {n: r.sha(repo / "experiments/zyf" / n) for n in ("matched_pilot.py", "full_suite_inheritance.py")},
               "media": {f"V{i}": {} for i in range(141)}}
        write(run / "configuration.json", cfg)
        before = {str(p): Reader().sha(p) for p in run.rglob("*") if p.is_file()}
        result = snapshot(run)
        self.assertEqual(len(result["questions"]), 558)
        self.assertEqual(result["conditions"]["noevent"]["scored"], 0)
        self.assertEqual(result["branches"]["event"]["counts"], {"pending": 141})
        self.assertNotIn(secret, json.dumps(result))
        self.assertIn("0/558", render(result))
        self.assertEqual(before, {str(p): Reader().sha(p) for p in run.rglob("*") if p.is_file()})


if __name__ == "__main__":
    unittest.main()
