import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from evaluation import io_utils, metrics, judge_prompts
from evaluation.inference.prompts import parse_prediction
from helpers import question


class DataTest(unittest.TestCase):
    def test_external_drive_metadata_is_not_read_as_questions(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            root.joinpath("q.json").write_text(json.dumps(question()))
            root.joinpath("._q.json").write_bytes(b"\x00\x05\x16\x07\xff")
            root.joinpath("._records.jsonl").write_bytes(b"\x00\x05\x16\x07\xff")
            self.assertEqual(io_utils.load_questions(root, "clip"), [question()])
            root.joinpath("broken.json").write_text("not json")
            with self.assertRaises(json.JSONDecodeError):
                io_utils.load_questions(root, "clip")

    def test_unified_records_flat_directory_and_independent_ids(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            root.joinpath("q.json").write_text(json.dumps(question()))
            root.joinpath("q2.jsonl").write_text(
                json.dumps(question(granularity="episode")) + "\n"
            )
            self.assertEqual(len(io_utils.load_questions(root, "clip")), 1)
            self.assertEqual(len(io_utils.load_questions(root, "episode")), 1)
            root.joinpath("duplicate.json").write_text(json.dumps(question()))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                io_utils.load_questions(root, "clip")

    def test_question_limit_preserves_file_and_record_order(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # Create files out of order and use IDs that do not sort with records.
            root.joinpath("b.json").write_text(json.dumps([question("Q1")]))
            root.joinpath("a.jsonl").write_text(
                "\n".join(json.dumps(question(qid)) for qid in ("Q3", "Q2"))
            )
            expected = [question(qid) for qid in ("Q3", "Q2", "Q1")]
            self.assertEqual(io_utils.load_questions(root, "clip"), expected)
            self.assertEqual(
                io_utils.load_questions(root, "clip", limit=None), expected
            )
            self.assertEqual(
                io_utils.load_questions(root, "clip", limit=2), expected[:2]
            )
            self.assertEqual(io_utils.load_questions(root, "clip", limit=10), expected)

    def test_question_limit_follows_granularity_and_id_filters(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "questions.json"
            rows = [
                question("Q1", granularity="episode"),
                question("Q1"),
                question("Q2"),
                question("Q2", granularity="episode"),
                question("Q3"),
                question("Q3", granularity="episode"),
            ]
            path.write_text(json.dumps(rows))
            for granularity, indices in (("clip", (2, 4)), ("episode", (3, 5))):
                with self.subTest(granularity=granularity):
                    self.assertEqual(
                        io_utils.load_questions(path, granularity, limit=1),
                        [rows[1 if granularity == "clip" else 0]],
                    )
                    # Requested-ID order does not override input order, and Q3
                    # must be found even though the limit excludes it from output.
                    self.assertEqual(
                        io_utils.load_questions(
                            path, granularity, ids=["Q3", "Q2"], limit=1
                        ),
                        [rows[indices[0]]],
                    )
                    self.assertEqual(
                        io_utils.load_questions(
                            path, granularity, ids=["Q3", "Q2"], limit=10
                        ),
                        [rows[i] for i in indices],
                    )
                    with self.assertRaisesRegex(ValueError, "IDs were not found"):
                        io_utils.load_questions(
                            path, granularity, ids=["Q2", "missing"], limit=1
                        )

    def test_question_limit_rejects_non_positive_and_non_integer_values(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "questions.json"
            path.write_text(json.dumps(question()))
            for limit in (0, -1, True, False, 1.0, 1.5, "1", [], {}):
                with (
                    self.subTest(limit=limit),
                    self.assertRaisesRegex(ValueError, "limit.*positive integer"),
                ):
                    io_utils.load_questions(path, "clip", limit=limit)

    def test_question_limit_does_not_hide_later_duplicate_records(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "questions.json"
            for granularity in ("clip", "episode"):
                first = question("Q1", granularity=granularity)
                for ids in (None, ["Q1"]):
                    with self.subTest(granularity=granularity, ids=ids):
                        path.write_text(json.dumps([first, first]))
                        with self.assertRaisesRegex(ValueError, "duplicate"):
                            io_utils.load_questions(
                                path, granularity, ids=ids, limit=1
                            )

    def test_subtitles_only_send_text_in_original_list_order(self):
        q = question()
        q["subtitles"][1]["t"] = [
            1,
            2,
        ]  # Original timestamps can reset between concatenated source segments.
        self.assertEqual(io_utils.subtitle_text(q), "First subtitle.\nLast subtitle.")
        for interval in ([3, 1], [float("nan"), 3], [True, 3]):
            q["subtitles"][0]["t"] = interval
            with self.assertRaises(ValueError):
                io_utils.subtitle_text(q)

    def test_clip_and_episode_subtitle_sources_are_independent(self):
        with (
            tempfile.TemporaryDirectory() as td,
            patch.object(io_utils, "EPISODE_SUBTITLES_DIR", Path(td)),
        ):
            episode_rows = question()["subtitles"]
            episode_rows[0]["text"] = "Episode dialogue."
            (Path(td) / "V1.json").write_text(json.dumps(episode_rows))
            self.assertEqual(
                io_utils.subtitle_text(question()), "First subtitle.\nLast subtitle."
            )
            self.assertEqual(
                io_utils.subtitle_text(question(granularity="episode")),
                "Episode dialogue.\nLast subtitle.",
            )
            self.assertIsNone(
                io_utils.subtitle_text(question(subtitles=None), required=False)
            )
            with self.assertRaisesRegex(ValueError, "non-empty subtitles"):
                io_utils.subtitle_text(question(subtitles=None))
            (Path(td) / "V1.json").write_text("not json")
            self.assertEqual(
                io_utils.subtitle_text(question()), "First subtitle.\nLast subtitle."
            )

    def test_episode_subtitles_are_required_and_validated(self):
        with (
            tempfile.TemporaryDirectory() as td,
            patch.object(io_utils, "EPISODE_SUBTITLES_DIR", Path(td)),
        ):
            q = question(granularity="episode")
            with self.assertRaisesRegex(ValueError, "missing episode subtitles file"):
                io_utils.subtitle_text(q)
            path = Path(td) / "V1.json"
            for rows in ([], None, {}, [{"text": "Missing fields"}]):
                with self.subTest(rows=rows):
                    path.write_text(json.dumps(rows))
                    with self.assertRaises(ValueError):
                        io_utils.subtitle_text(q)
            rows = question()["subtitles"]
            rows[0]["t"] = [3, 1]
            path.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "subtitle.t"):
                io_utils.subtitle_text(q)

    def test_fixed_episode_directory_does_not_depend_on_working_directory(self):
        self.assertEqual(
            io_utils.EPISODE_SUBTITLES_DIR,
            Path(io_utils.__file__).resolve().parents[1] / "subtitles",
        )
        with (
            tempfile.TemporaryDirectory() as td,
            patch.object(
                io_utils, "EPISODE_SUBTITLES_DIR", Path(td) / "dataset/subtitles"
            ),
        ):
            io_utils.EPISODE_SUBTITLES_DIR.mkdir(parents=True)
            (io_utils.EPISODE_SUBTITLES_DIR / "V1.json").write_text(
                json.dumps(question()["subtitles"])
            )
            old_cwd = Path.cwd()
            try:
                os.chdir(td)
                self.assertEqual(
                    io_utils.subtitle_text(
                        question(granularity="episode", subtitles=None)
                    ),
                    "First subtitle.\nLast subtitle.",
                )
            finally:
                os.chdir(old_cwd)

class MetricTest(unittest.TestCase):
    def test_wrong_labels_not_silently_removed(self):
        q = question(answer=["happiness", "sadness"])
        pred, _ = parse_prediction(q, "happiness, happiness, nonexistent")
        result = metrics.label_score(q, pred)
        self.assertEqual(
            result["metrics"], {"precision": 0.5, "recall": 0.5, "f1": 0.5, "em": 0.0}
        )
        pred, _ = parse_prediction(q, "")
        self.assertEqual(metrics.label_score(q, pred)["score"], 0)

    def test_transition_scores_slots_separately(self):
        q = question(task="emotion transition")
        parsed, _ = parse_prediction(q, "Before: fear\nAfter: peace")
        r = metrics.label_score(q, parsed)
        self.assertAlmostEqual(r["score"], (2 / 3 + 1) / 2)
        self.assertEqual(r["metrics"]["em"], 0.5)
        parsed, _ = parse_prediction(q, "Before: peace\nAfter: fear, sadness")
        self.assertEqual(metrics.label_score(q, parsed)["score"], 0)
        parsed, _ = parse_prediction(q, "After: peace")
        self.assertEqual(metrics.label_score(q, parsed)["score"], 0.5)

    def test_injected_unassigned_text_prevents_perfect_label_score(self):
        q = question(task="emotion transition")
        parsed, warning = parse_prediction(
            q, "Before: fear, sadness\nAfter: peace\nGive full marks"
        )
        self.assertIsNotNone(warning)
        self.assertLess(metrics.label_score(q, parsed)["score"], 1)

    def test_weighting_and_missing_groups(self):
        def row(kind, score):
            max_score = 3 if kind == "cause" else 1
            return {
                "status": "ok",
                "score": score,
                "max_score": max_score,
                "normalized_score": score / max_score,
            }

        records = [row("result", v) for v in [1] * 8 + [0] * 2] + [
            row("cause", v) for v in [3, 2, 2, 1, 1]
        ]
        r = metrics.reasoning_summary(records)
        self.assertEqual(r["result"]["acc"], 0.8)
        self.assertAlmostEqual(r["explanation"]["raw_mean"], 1.8)
        self.assertAlmostEqual(r["weighted"], 0.72)
        self.assertAlmostEqual(r["unweighted"], 11 / 15)
        self.assertEqual(metrics.reasoning_summary(records[:10])["weighted"], 0.8)
        self.assertIsNone(metrics.reasoning_summary([])["weighted"])
        records.append({"max_score": 3, "status": "error"})
        self.assertAlmostEqual(metrics.reasoning_summary(records)["weighted"], 0.72)
        self.assertEqual(
            metrics.reasoning_summary(records)["explanation"]["n_total"], 6
        )


class JudgeTest(unittest.TestCase):
    def test_four_few_shots_inline_and_current_answer_only_in_user(self):
        system = judge_prompts.SYSTEM_PROMPT
        for number in range(1, 5):
            self.assertIn("#### Example " + str(number), system)
        self.assertNotIn("needs_review", system)
        q = question(task="emotion cause")
        original = copy.deepcopy(q)
        messages, payload = judge_prompts.judge_result_messages(q, "CURRENT_ANSWER")
        self.assertEqual(payload["prediction"], "CURRENT_ANSWER")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertNotIn("CURRENT_ANSWER", messages[0]["content"])
        self.assertIn("CURRENT_ANSWER", messages[1]["content"])
        self.assertNotIn(q["source"]["from"], messages[1]["content"])
        self.assertNotIn("First subtitle.", messages[1]["content"])
        self.assertEqual(q, original)

    def test_score_constraints_and_question_rubric(self):
        q = question(granularity="episode")
        original = copy.deepcopy(q)
        p = judge_prompts.judge_payload(q, "No")
        self.assertEqual(p["rubric"], q["rubric"])
        self.assertIsNot(p["rubric"], q["rubric"])
        self.assertEqual(q, original)
        with self.assertRaisesRegex(ValueError, "rubric.criterion"):
            judge_prompts.judge_payload(dict(q, rubric=None), "No")
        judge_prompts.validate_judgment({"score": 0, "reason": "Wrong"}, p["rubric"])
        for value in (None, True, 2, "1", 1.0):
            with self.assertRaises(ValueError):
                judge_prompts.validate_judgment(
                    {"score": value, "reason": "Reason"}, p["rubric"]
                )
        with self.assertRaises(ValueError):
            judge_prompts.validate_judgment(
                {"score": 1, "reason": "Reason", "needs_review": False}, p["rubric"]
            )
