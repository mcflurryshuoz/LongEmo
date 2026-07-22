#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from benchmark.preprocess import prepare_clips  # noqa: E402


def question(series: str, qid: str):
    return {
        "series": series,
        "qid": qid,
        "question_type": "single_choice",
        "question": "What emotion is most visible? Please select only one option. Respond with the option letter only.",
        "options": ["A. happy", "B. sad", "C. angry"],
        "answer": "A",
        "input_video": {"scope_kind": "event", "eps": ["S01E01"], "segments": [{"ep": "S01E01", "start": 1.0, "end": 5.0}]},
        "input_transcript": [{"ep": "S01E01", "t": [1.0, 2.0], "text": "hello"}],
    }


def fake_cut_clip(video, out_path, start, end, **kwargs):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"x" * 2048)
    return out


class PrepareClipsTest(unittest.TestCase):
    def test_clip_names_include_series_and_are_found_by_key(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for series in ("friends", "frasier"):
                (root / "videos" / series).mkdir(parents=True)
                (root / "videos" / series / "S01E01.mp4").write_bytes(b"")
            videos = {
                "friends/S01E01": str(root / "videos" / "friends" / "S01E01.mp4"),
                "frasier/S01E01": str(root / "videos" / "frasier" / "S01E01.mp4"),
            }
            clip_dir = root / "clips"
            with mock.patch.object(prepare_clips.media, "cut_clip", side_effect=fake_cut_clip):
                # the same qid exists in two series; clips must not collide
                prepare_clips.cut_question_clips(question("friends", "q1"), "friends/q1", videos, clip_dir)
                prepare_clips.cut_question_clips(question("frasier", "q1"), "frasier/q1", videos, clip_dir)
            friends = prepare_clips.find_question_clips(clip_dir, "friends/q1")
            frasier = prepare_clips.find_question_clips(clip_dir, "frasier/q1")
        self.assertEqual(len(friends), 1)
        self.assertEqual(len(frasier), 1)
        self.assertTrue(friends[0].name.startswith("friends_q1_seg1_"))
        self.assertNotEqual(friends[0].name, frasier[0].name)

    def test_batch_clip_provider_returns_precut_clips_or_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            clip_dir = root / "clips"
            clip_dir.mkdir()
            (clip_dir / "friends_q1_seg1_1.0_5.0.mp4").write_bytes(b"x")
            args = argparse.Namespace(clips_dir=str(clip_dir), streaming=False)
            get_clips, release = prepare_clips.clip_provider(args, root / "questions.json")
            self.assertIsNone(release)
            clips = get_clips("friends/q1", question("friends", "q1"))
            self.assertEqual(len(clips), 1)
            self.assertTrue(clips[0]["clip_path"].endswith("friends_q1_seg1_1.0_5.0.mp4"))
            with self.assertRaises(FileNotFoundError):
                get_clips("frasier/q1", question("frasier", "q1"))

    def test_contract_gate_blocks_bad_questions(self):
        with tempfile.TemporaryDirectory() as td:
            qpath = Path(td) / "questions.json"
            bad = question("friends", "q1")
            bad["answer"] = "Z"
            qpath.write_text(json.dumps([bad]), encoding="utf-8")
            with self.assertRaises(SystemExit):
                prepare_clips.run_contract_check(qpath)


if __name__ == "__main__":
    unittest.main()
