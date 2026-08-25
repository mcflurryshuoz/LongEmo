#!/usr/bin/env python3
"""Cut benchmark ``1_clip`` inputs from the original episodes.

The CLI cuts every selected question's input_video segment into --out using the
question's canonical ``video_name``. Questions that share one interval also
share one prepared file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from preprocess import io_utils, media  # noqa: E402


def select_questions(questions: list[dict[str, Any]], args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    wanted = set(args.qid or [])
    selected: list[tuple[str, dict[str, Any]]] = []
    for q in questions:
        key = io_utils.question_key(q)
        if wanted and not wanted & {key, q["qid"]}:
            continue
        selected.append((key, q))
    return selected


def cut_question_clip(
    q: dict[str, Any],
    video_root: str | Path,
    clip_dir: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Cut the question's single canonical video segment."""
    segment = q["input_video"]["segments"][0]
    series = q["series"]
    episode = segment["ep"]
    source = Path(video_root) / series / f"{episode}.mp4"
    if not source.exists():
        raise FileNotFoundError(f"missing source video: {source}")
    start, end = float(segment["start"]), float(segment["end"])
    out = clip_dir / q["video_name"]
    media.cut_clip(source, out, start, end, force=force)
    return {"ep": episode, "start": start, "end": end, "clip_path": str(out)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", required=True, help="Canonical g1_clip question JSON.")
    ap.add_argument("--out", required=True, help="Output directory for the cut clips.")
    ap.add_argument("--video-root", required=True, help="Root containing <series>/<episode>.mp4.")
    ap.add_argument("--qid", action="append", default=None, help="Only preprocess selected qid(s) or series/qid key(s).")
    ap.add_argument("--force", action="store_true", help="Re-cut existing clips.")
    ap.add_argument("--keep-going", action="store_true", help="Print errors and continue.")
    args = ap.parse_args()

    questions = io_utils.load_questions(args.questions)
    selected = select_questions(questions, args)
    clip_dir = Path(args.out)

    num_clips = 0
    errors: list[str] = []
    for pos, (key, q) in enumerate(selected, 1):
        print(f"[{pos}/{len(selected)}] {key}", flush=True)
        try:
            cut_question_clip(q, args.video_root, clip_dir, force=args.force)
            num_clips += 1
        except Exception as e:
            errors.append(f"{key}: {type(e).__name__}: {e}")
            if not args.keep_going:
                raise
    for err in errors:
        print(f"error: {err}", file=sys.stderr)
    print(json.dumps({"questions": len(selected), "clips": num_clips, "errors": len(errors), "out": str(clip_dir)}, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
