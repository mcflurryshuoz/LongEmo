#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from benchmark.preprocess import check_question_contract  # noqa: E402


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


class CheckDatasetTest(unittest.TestCase):
    def test_valid_file_passes(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "unified.json"
            p.write_text(json.dumps([question("friends", "q1"), question("frasier", "q1")]), encoding="utf-8")
            report = check_question_contract.check_path(p)
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(report["count"], 2)
        self.assertEqual(report["counts"]["by_series"], {"friends": 1, "frasier": 1})

    def test_duplicate_series_qid_is_an_error(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "unified.json"
            p.write_text(json.dumps([question("friends", "q1"), question("friends", "q1")]), encoding="utf-8")
            report = check_question_contract.check_path(p)
        self.assertFalse(report["ok"])
        self.assertEqual(report["errors"][0]["category"], "qid")

    def test_schema_and_input_errors_are_reported(self):
        bad = question("friends", "q1")
        bad.pop("input_transcript")
        bad["input_video"] = ["S01E01"]  # old episode-scope shape, not an object
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "unified.json"
            p.write_text(json.dumps([bad]), encoding="utf-8")
            report = check_question_contract.check_path(p)
        self.assertFalse(report["ok"])
        categories = {e["category"] for e in report["errors"]}
        self.assertIn("schema", categories)
        self.assertIn("input_video", categories)
        self.assertIn("input_transcript", categories)

    def test_segment_and_transcript_shape_errors(self):
        bad = question(
            "friends",
            "q1",
            input_video={"scope_kind": "event", "eps": ["S01E01"], "segments": [{"ep": "S01E02", "start": 5.0, "end": 1.0}]},
            input_transcript=[{"ep": "S01E01", "t": [2.0, 1.0], "text": ""}],
        )
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "unified.json"
            p.write_text(json.dumps([bad]), encoding="utf-8")
            report = check_question_contract.check_path(p)
        reasons = " | ".join(e["reason"] for e in report["errors"])
        self.assertIn("end <= start", reasons)
        self.assertIn("not listed in input_video.eps", reasons)
        self.assertIn("missing text", reasons)
        self.assertIn("end > start", reasons)

    def test_directory_mode_checks_series_against_directory(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "friends" / "grade_1").mkdir(parents=True)
            (root / "frasier" / "grade_1").mkdir(parents=True)
            (root / "friends" / "grade_1" / "S01E01.json").write_text(
                json.dumps([question("friends", "q1")]), encoding="utf-8"
            )
            # series says friends but the file lives under frasier/
            (root / "frasier" / "grade_1" / "S01E01.json").write_text(
                json.dumps([question("friends", "q2")]), encoding="utf-8"
            )
            report = check_question_contract.check_path(root)
        self.assertFalse(report["ok"])
        self.assertEqual(report["files"], 2)
        self.assertEqual(report["count"], 2)
        self.assertEqual([e["category"] for e in report["errors"]], ["series"])


if __name__ == "__main__":
    unittest.main()
