import json
from pathlib import Path
import shutil
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

    def full_fixture(self):
        run, repo = self.root.resolve() / "run", self.root.resolve() / "repo"
        secret = "QUESTION_AND_GOLD_MUST_NOT_APPEAR"
        qs = [{"question_id": f"Q{i}", "video_id": f"V{i % 141}", "type": "emotion trajectory", "question": secret,
               "gold": secret, "rubric": {"scores": {"0": secret, "2": secret}}, "source": {"from": "Friends S01E01"}}
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
        return run, repo, cfg, secret

    def scope_fixture(self, run, repo):
        folder = self.root / "new_coordinator"
        shutil.copytree(repo / "experiments/zyf", folder)
        write(folder / "full_suite.py", {"test": "new scope coordinator"})
        scope = {"schema_version": 1, "conditions": ["noevent", "method"],
                 "previous_configuration_sha256": Reader().sha(run / "configuration.json"),
                 "coordinator_file": str(folder / "full_suite.py"),
                 "coordinator_sha256": Reader().sha(folder / "full_suite.py"),
                 "reason": "user_requested_no_new_base", "created_at": "2026-09-23T02:00:00+08:00",
                 "previous_coordinator": {"pid": 512357, "start_ticks": "2006414"}}
        write(run / "execution_scope.json", scope)
        return scope, folder

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

    def test_output_validation_classification_requires_validate_stage_and_preserves_provider_errors(self):
        for error_type in ("ValueError", "JSONDecodeError"):
            row = {"error_type": error_type, "failure_stage": "validate"}
            with self.subTest(error_type=error_type):
                self.assertEqual(classify_failure(row), "output_validation_failed")
                self.assertEqual(classify_failure({**row, "failure_stage": "generate"}), "other")
                self.assertEqual(classify_failure({**row, "http_status": 428}), "http_428_unclassified")
                self.assertEqual(classify_failure({**row, "service_error_code": "content_filter"}), "content_filter")
        self.assertEqual(classify_failure({"error_type": "RuntimeError", "failure_stage": "validate"}), "other")

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
        run, repo, cfg, secret = self.full_fixture()
        before = {str(p): Reader().sha(p) for p in run.rglob("*") if p.is_file()}
        result = snapshot(run)
        self.assertEqual(len(result["questions"]), 558)
        self.assertEqual(result["conditions"]["noevent"]["scored"], 0)
        self.assertEqual(result["branches"]["event"]["counts"], {"pending": 141})
        self.assertNotIn("execution_conditions", result)
        self.assertNotIn("retained_reference_conditions", result)
        self.assertNotIn(secret, json.dumps(result))
        self.assertIn("0/558", render(result))
        self.assertEqual(before, {str(p): Reader().sha(p) for p in run.rglob("*") if p.is_file()})
        result["branches"]["noevent"]["videos"][0]["final_failure"] = {"category": "output_validation_failed"}
        self.assertIn("输出结构校验失败 1视频", render(result))

    def test_scoped_snapshot_preserves_first_scores_and_excludes_unrequested_base(self):
        run, repo, _, _ = self.full_fixture()
        _, _, source, _, accepted, *_ = self.fixture(run)
        # An existing base first score remains fully validated and byte-identical.
        shutil.copytree(run / "noevent", run / "event")
        shutil.copytree(run / "answers/noevent", run / "answers/base")
        base_source = run / "scores/base/V1/run_fixture/scores.jsonl"
        base_source.parent.mkdir(parents=True)
        shutil.copyfile(source, base_source)
        write(run / "accepted/base/Q1.json", {**accepted, "source": str(base_source)})
        write(run / "answers/base/V1/traces/Q1.json", {"routing": {
            "effective_retrieval": "graph", "progressive_routing": "none", "hybrid_graph_tasks": []}})
        # Snapshot's video inventory also needs the media duration.
        for branch in ("noevent", "event"):
            memory = run / branch / "videos/V1/memory/V1/memory.json"
            write(memory, {"complete": True, "duration": 20})
            write(run / branch / "frozen_memories/V1.json", {"memory_sha256": Reader().sha(memory)})
        original = snapshot(run)
        self.scope_fixture(run, repo)
        before = {str(p): Reader().sha(p) for p in run.rglob("*") if p.is_file()}
        result = snapshot(run)
        self.assertEqual(result["execution_conditions"], ["noevent", "method"])
        self.assertEqual(result["retained_reference_conditions"], ["base"])
        self.assertEqual(result["conditions"], original["conditions"])
        self.assertEqual(result["questions"][1]["conditions"], original["questions"][1]["conditions"])
        self.assertEqual(result["missing_counts"]["base"], {"not_requested": 557})
        self.assertEqual(result["questions"][0]["conditions"]["noevent"]["status"], "waiting_for_media_or_phase")
        self.assertIn("本轮仅按 noevent → event（method）执行", render(result))
        self.assertIn("本轮队列完成仅跟进 noevent 和 method", render(result))
        self.assertNotIn("base / method共图", render(result))
        self.assertEqual(before, {str(p): Reader().sha(p) for p in run.rglob("*") if p.is_file()})

    def test_scope_rejects_unverified_coordinator_and_preserves_original_source_checks(self):
        run, repo, _, _ = self.full_fixture()
        scope, folder = self.scope_fixture(run, repo)
        mutations = [
            (run / "execution_scope.json", {**scope, "previous_configuration_sha256": "bad"}, "original configuration"),
            (run / "execution_scope.json", {**scope, "coordinator_file": str(repo / "experiments/zyf/full_suite.py")}, "distinct coordinator"),
            (folder / "full_suite.py", {"changed": True}, "coordinator source"),
            (folder / "matched_pilot.py", {"changed": True}, "scope support"),
            (repo / "methods/longemo/common.py", {"changed": True}, "method/evaluator source"),
            (repo / "experiments/zyf/full_suite.py", {"changed": True}, "frozen coordinator"),
            (repo / "experiments/zyf/full_suite_inheritance.py", {"changed": True}, "frozen support"),
        ]
        for path, value, message in mutations:
            with self.subTest(path=str(path), message=message):
                previous = path.read_bytes()
                try:
                    write(path, value)
                    with self.assertRaisesRegex(ValueError, message):
                        snapshot(run)
                finally:
                    path.write_bytes(previous)

    def test_scope_requires_exact_supported_schema_and_frozen_identity(self):
        run, repo, _, _ = self.full_fixture()
        scope, _ = self.scope_fixture(run, repo)
        for changes in ({"conditions": ["method"]}, {"reason": "unknown"}, {"extra": True},
                        {"schema_version": True}, {"previous_coordinator": {"pid": 512357}},
                        *({"previous_coordinator": {"pid": 512357, "start_ticks": ticks}}
                          for ticks in (True, "0", "-1", "2.0", "", 0)),
                        {"previous_coordinator": {"pid": True, "start_ticks": "2006414"}},
                        {"created_at": "2026-09-23T02:00:00"}):
            with self.subTest(changes=changes):
                write(run / "execution_scope.json", {**scope, **changes})
                with self.assertRaises(ValueError):
                    snapshot(run)
        write(run / "execution_scope.json", {**scope, "previous_coordinator": {"pid": 512357, "start_ticks": 2006414}})
        self.assertEqual(snapshot(run)["execution_conditions"], ["noevent", "method"])
        target = run / "scope_target.json"
        write(target, scope)
        (run / "execution_scope.json").unlink()
        (run / "execution_scope.json").symlink_to(target)
        with self.assertRaisesRegex(ValueError, "regular file"):
            snapshot(run)


if __name__ == "__main__":
    unittest.main()
