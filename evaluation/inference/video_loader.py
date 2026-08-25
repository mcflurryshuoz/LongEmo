"""Locate prepared videos for model inference."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

from evaluation import contract


VideoGetter = Callable[[str, dict[str, Any]], list[dict[str, Any]]]


def add_input_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--clips-dir",
        default=None,
        help="Prepared 1_clip directory; each file is resolved by the question's exact video_name.",
    )
    parser.add_argument(
        "--episodes-dir",
        default=None,
        help="Prepared episode directory containing flat <series>_<episode>.mp4 files.",
    )


def question_granularity(questions: list[dict[str, Any]]) -> str:
    granularities = {contract.question_granularity(q) for q in questions}
    if len(granularities) != 1:
        raise ValueError("one inference run must use either 1_clip or 2_episode questions, not both")
    return str(granularities.pop())


def validate_input_args(args: argparse.Namespace, questions: list[dict[str, Any]]) -> None:
    granularity = question_granularity(questions)
    if granularity == "1_clip" and not args.clips_dir:
        raise ValueError("1_clip inference requires --clips-dir")
    if granularity == "2_episode" and not args.episodes_dir:
        raise ValueError("2_episode inference requires --episodes-dir")


def build_loader(
    args: argparse.Namespace,
    questions: list[dict[str, Any]],
) -> VideoGetter:
    """Dispatch each question to its prepared clip or effective episode."""
    validate_input_args(args, questions)
    granularity = question_granularity(questions)

    def get_videos(key: str, q: dict[str, Any]) -> list[dict[str, Any]]:
        if granularity == "1_clip":
            video_name = str(q.get("video_name") or "").strip()
            if not video_name:
                raise ValueError(f"{key}: 1_clip question has no video_name")
            path = Path(args.clips_dir) / video_name
            if not path.exists():
                raise FileNotFoundError(f"{key}: missing prepared clip: {path}")
            return [{"kind": "clip", "clip_path": str(path)}]

        if granularity == "2_episode":
            series = q["series"]
            inputs = []
            for episode in q.get("input_video") or []:
                path = Path(args.episodes_dir) / f"{series}_{episode}.mp4".lower()
                if not path.exists():
                    raise FileNotFoundError(f"{key}: missing prepared episode: {path}")
                inputs.append({"kind": "episode", "ep": str(episode), "clip_path": str(path)})
            return inputs
        raise ValueError(f"{key}: qid does not identify a released question granularity")

    return get_videos
