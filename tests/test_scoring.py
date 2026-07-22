#!/usr/bin/env python3
"""Offline tests for the scoring pipeline.

They call run_scoring.score_items / score_emotion_records directly with
judge=None (open non-emotion questions become needs_judge; emotion grouping
falls back to exact matching) — no DeepSeek, no network. The CLI itself always
uses the real judge.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from benchmark.evaluation import emotion_grouping, run_scoring  # noqa: E402


def question(series: str, qid: str, **overrides):
    q = {
        "series": series,
        "qid": qid,
        "question_type": "single_choice",
        "question": "What emotion is most visible? Please select only one option. Respond with the option letter only.",
        "options": ["A. happy", "B. sad", "C. angry"],
        "answer": "A",
        "input_video": {"scope_kind": "event", "eps": ["S01E01"], "segments": [{"ep": "S01E01", "start": 1.0, "end": 5.0}]},
        "input_transcript": [{"ep": "S01E01", "t": [1.0, 2.0], "text": "hello"}],
    }
    q.update(overrides)
    return q


def predicted(q, pred_answer):
    return {
        **q,
        "pred_answer": pred_answer,
        "pred_info": {"model": "test-model", "reason": "r", "raw": "", "error": None},
    }


def evaluate(items):
    """Offline mirror of the CLI flow: route, exact-match grouping, aggregate."""
    res = run_scoring.score_items(items, judge=None)
    grouping = emotion_grouping.gpt_group_emotions(run_scoring.emotion_labels(res["emotion_slots"]), None)
    emotion_records = run_scoring.score_emotion_records(res["emotion_slots"], grouping)
    return res, emotion_records


class FormalScoringTest(unittest.TestCase):
    def test_full_run_scores_all_categories(self):
        items = [
            # single choice: correct / wrong
            predicted(question("friends", "q1"), "A"),
            predicted(question("friends", "q2", answer="B"), "A"),
            # multi select: P/R/F1 — exact match F1 1, half recall F1 2/3
            predicted(
                question(
                    "friends",
                    "q3",
                    question_type="multi_select",
                    question='Choose the emotions shown. Please select all correct options. Respond with option letters only, such as "B D".',
                    answer=["A", "C"],
                ),
                "C A",
            ),
            predicted(
                question(
                    "friends",
                    "q4",
                    question_type="multi_select",
                    question='Choose the emotions shown. Please select all correct options. Respond with option letters only, such as "B D".',
                    answer=["A", "C"],
                ),
                "A",
            ),
            # ranking: exact sequence — order matters
            predicted(
                question(
                    "friends",
                    "q5",
                    question_type="ranking",
                    question='Order the stages. Some options may be distractors. Respond with option letters only, such as "A B C".',
                    answer=["B", "A"],
                ),
                "B A",
            ),
            predicted(
                question(
                    "friends",
                    "q6",
                    question_type="ranking",
                    question='Order the stages. Some options may be distractors. Respond with option letters only, such as "A B C".',
                    answer=["B", "A"],
                ),
                "A B",
            ),
            # open emotion: direct (exact grouping offline) + transition (two slots)
            predicted(
                question(
                    "friends",
                    "q7",
                    question_type="open_ended",
                    question='What is the emotion? Answer with a JSON object like {"emotion":["..."]}.',
                    options=[],
                    answer={"emotion": ["nervous"]},
                    emotion_answer=True,
                ),
                '{"emotion": ["nervous"]}',
            ),
            predicted(
                question(
                    "friends",
                    "q8",
                    question_type="open_ended",
                    question='How does the emotion change? Answer with a JSON object like {"from_emotion":["..."],"to_emotion":["..."]}.',
                    options=[],
                    answer={"from_emotion": ["calm"], "to_emotion": ["angry"]},
                    emotion_answer=True,
                ),
                '{"from_emotion": ["calm"], "to_emotion": ["sad"]}',
            ),
            # open non-emotion: needs the judge, unscored offline
            predicted(
                question(
                    "friends",
                    "q9",
                    question_type="open_ended",
                    question="What happens? Answer with one short phrase.",
                    options=[],
                    answer="He drops the tray.",
                ),
                "He drops the tray.",
            ),
        ]
        res, emotion_records = evaluate(items)
        self.assertEqual(res["format_errors"], [])

        acc = run_scoring.accuracy_metrics(res["accuracy_records"])
        self.assertEqual(acc["n"], 5)
        self.assertEqual(acc["scored"], 4)  # q9 needs the judge
        self.assertEqual(acc["correct"], 2)  # q1, q5
        self.assertEqual(acc["score_methods"]["needs_judge"], 1)
        by_type = run_scoring.grouped_metrics(res["accuracy_records"], "question_type", "accuracy")
        self.assertEqual(by_type["single_choice"]["correct"], 1)
        self.assertEqual(by_type["ranking"]["correct"], 1)

        multi = run_scoring.micro_prf(res["multi_records"])
        self.assertEqual(multi["n"], 2)
        self.assertAlmostEqual(multi["macro"]["f1"], (1.0 + 2 / 3) / 2, places=6)

        # emotion: q7 = 1 slot (match), q8 = 2 slots (from matches, to does not)
        emo = run_scoring.micro_prf(emotion_records)
        self.assertEqual(emo["n"], 3)
        self.assertAlmostEqual(emo["macro"]["f1"], (1.0 + 1.0 + 0.0) / 3, places=6)

    def test_same_qid_across_series_do_not_collide(self):
        items = [
            predicted(question("friends", "q1"), "A"),
            predicted(question("frasier", "q1", answer="B"), "C"),
        ]
        res, _ = evaluate(items)
        records = res["accuracy_records"]
        self.assertEqual({r["key"] for r in records}, {"friends/q1", "frasier/q1"})
        by_series = run_scoring.grouped_metrics(records, "series", "accuracy")
        self.assertEqual(by_series["friends"]["correct"], 1)
        self.assertEqual(by_series["frasier"]["correct"], 0)

    def test_gold_format_errors_exclude_the_question(self):
        items = [predicted(question("friends", "q1", answer="Z"), "A")]
        res, _ = evaluate(items)
        self.assertEqual(len(res["format_errors"]), 1)
        self.assertEqual(res["accuracy_records"], [])

    def test_yes_no_is_deterministic_not_judged(self):
        yn = question(
            "friends",
            "q1",
            question_type="open_ended",
            question="Has Ross gotten over it? Answer Yes or No.",
            options=[],
            answer="No",
        )
        items = [
            predicted(yn, "No, he still misses her."),
            predicted({**yn, "qid": "q2"}, "Yes"),
            predicted({**yn, "qid": "q3"}, "Yes and no"),
        ]
        res, _ = evaluate(items)
        acc = run_scoring.accuracy_metrics(res["accuracy_records"])
        # judge=None, yet all three are scored: the Yes/No path is deterministic
        self.assertEqual(acc["score_methods"], {"exact_yes_no": 3})
        self.assertEqual(acc["correct"], 1)  # q1 commits to No; q2 wrong; q3 hedged

    def test_missing_prediction_is_unscored(self):
        res, _ = evaluate([question("friends", "q1")])
        acc = run_scoring.accuracy_metrics(res["accuracy_records"])
        self.assertEqual(acc["scored"], 0)
        self.assertEqual(acc["score_methods"], {"missing_prediction": 1})

    def test_prediction_error_is_unscored(self):
        item = predicted(question("friends", "q1"), None)
        item["pred_info"]["error"] = "APIStatusError: 413"
        res, _ = evaluate([item])
        acc = run_scoring.accuracy_metrics(res["accuracy_records"])
        self.assertEqual(acc["scored"], 0)
        self.assertEqual(acc["score_methods"], {"prediction_error": 1})


if __name__ == "__main__":
    unittest.main()
