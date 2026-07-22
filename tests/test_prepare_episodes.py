#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from benchmark.preprocess import prepare_episodes  # noqa: E402


def ranges(series="friends", **episodes):
    return {"series": series, "season": "S01", "episodes": episodes}


def episode(status="verified", segments=None):
    return {
        "duration": 1369.4,
        "status": status,
        "valid_segments": [
            {"start": 0.0, "end": 72.0, "label": "cold_open"},
            {"start": 117.0, "end": 1362.0, "label": "main_content"},
        ]
        if segments is None
        else segments,
    }


def make_args(**overrides):
    args = argparse.Namespace(
        ranges=None,
        ranges_root=None,
        video_root="/videos",
        out="/out",
        eps=None,
        verified_only=False,
        force=False,
        keep_going=False,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


class EpisodeJobsTest(unittest.TestCase):
    def test_jobs_carry_paths_segments_and_duration(self):
        jobs = prepare_episodes.episode_jobs(ranges(S01E02=episode()), make_args())
        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        self.assertEqual(job["source_video"], "/videos/friends/S01E02.mp4")
        self.assertEqual(job["out_video"], "/out/friends/S01E02.mp4")
        self.assertEqual(len(job["segments"]), 2)
        self.assertAlmostEqual(job["effective_duration"], 72.0 + 1245.0, places=1)

    def test_eps_filter_and_verified_only(self):
        r = ranges(S01E01=episode(status="draft"), S01E02=episode())
        self.assertEqual(len(prepare_episodes.episode_jobs(r, make_args())), 2)
        self.assertEqual(
            [j["ep"] for j in prepare_episodes.episode_jobs(r, make_args(verified_only=True))],
            ["S01E02"],
        )
        self.assertEqual(
            [j["ep"] for j in prepare_episodes.episode_jobs(r, make_args(eps=["S01E01"]))],
            ["S01E01"],
        )

    def test_bad_segments_are_rejected(self):
        with self.assertRaises(ValueError):
            prepare_episodes.episode_jobs(
                ranges(S01E01=episode(segments=[{"start": 10.0, "end": 5.0}])), make_args()
            )
        with self.assertRaises(ValueError):
            prepare_episodes.episode_jobs(ranges(S01E01=episode(segments=[])), make_args())


class FindRangesFilesTest(unittest.TestCase):
    def test_ranges_root_prefers_manual_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for series, names in (
                ("friends", ("valid_ranges.json", "valid_input_ranges.json")),
                ("frasier", ("valid_input_ranges.json",)),
            ):
                d = root / series / "s5_effective_inputs"
                d.mkdir(parents=True)
                for name in names:
                    (d / name).write_text(json.dumps(ranges(series=series)), encoding="utf-8")
            files = prepare_episodes.find_ranges_files(make_args(ranges_root=str(root)))
        names = {f.parent.parent.name: f.name for f in files}
        self.assertEqual(names, {"friends": "valid_ranges.json", "frasier": "valid_input_ranges.json"})


if __name__ == "__main__":
    unittest.main()
