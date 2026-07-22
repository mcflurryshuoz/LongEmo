#!/usr/bin/env python3
"""Cut benchmark input clips (grade-1/2 questions) from the original episodes.

Batch CLI: cut every selected question's input_video segments into --out as
<series>_<qid>_seg<j>_<start>_<end>.mp4. Clip names are deterministic, so
inference finds pre-cut clips by name (--clips-dir) — no manifest file needed.

Streaming: model scripts call clip_provider(...) with --streaming, which cuts
one question's clips right before inference and deletes them right after, so
no intermediate clip files are kept.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from benchmark.common import io_utils, media  # noqa: E402
from benchmark.preprocess import check_question_contract  # noqa: E402


def select_questions(questions: list[dict[str, Any]], args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    wanted = set(args.qid or [])
    selected: list[tuple[str, dict[str, Any]]] = []
    for i, q in enumerate(questions, 1):
        key = io_utils.question_key(q, i)
        if wanted and not wanted & {key, io_utils.safe_qid(q, i), str(q.get("qid"))}:
            continue
        selected.append((key, q))
    if args.limit:
        selected = selected[: args.limit]
    return selected


def _segment_name(key: str, idx: int, start: float, end: float) -> str:
    # the qid already carries the episode id, so it is not repeated here
    return f"{key.replace('/', '_')}_seg{idx}_{start:.1f}_{end:.1f}.mp4"


def find_question_clips(clips_dir: str | Path, key: str) -> list[Path]:
    """Pre-cut clips of one question, in segment order (matched by name prefix)."""
    prefix = f"{key.replace('/', '_')}_seg"

    def seg_no(p: Path) -> int:
        m = re.match(re.escape(prefix) + r"(\d+)_", p.name)
        return int(m.group(1)) if m else 0

    return sorted(Path(clips_dir).glob(f"{prefix}*.mp4"), key=seg_no)


def run_contract_check(questions_path: str | Path) -> None:
    report = check_question_contract.check_path(questions_path)
    if not report.get("ok"):
        errors = report.get("errors") or []
        for e in errors[:10]:
            print(f"contract error: {e.get('qid')}: {e.get('reason')}", file=sys.stderr)
        raise SystemExit(f"question contract check failed: {len(errors)} error(s)")


def cut_question_clips(
    q: dict[str, Any],
    key: str,
    videos: dict[str, str],
    clip_dir: Path,
    *,
    series: str | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Cut every input_video segment of one question; returns the clip records."""
    segments = (q.get("input_video") or {}).get("segments") or []
    if not segments:
        raise ValueError("input_video.segments is empty; cannot cut a minimal input video")
    q_series = io_utils.question_series(q, default=series)
    if not q_series:
        raise ValueError(f"{key}: cannot infer series; add a 'series' field to the question or pass --series")
    clips: list[dict[str, Any]] = []
    for j, seg in enumerate(segments, 1):
        ep = str(seg["ep"])
        src = Path(videos.get(io_utils.video_key(q_series, ep)) or "")
        if not src.exists():
            raise FileNotFoundError(f"missing source video for {q_series}/{ep}: {src}")
        start, end = float(seg["start"]), float(seg["end"])
        if end <= start:
            raise ValueError(f"bad segment time: start={start}, end={end}")
        out = clip_dir / _segment_name(key, j, start, end)
        media.cut_clip(src, out, start, end, force=force)
        clips.append({"ep": ep, "start": start, "end": end, "clip_path": str(out)})
    return clips


# ---------------------------------------------------------------------------
# clip providers for model inference scripts
# ---------------------------------------------------------------------------

def add_clip_args(ap: argparse.ArgumentParser) -> None:
    """Clip-source options shared by every model inference script."""
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--clips-dir", default=None, help="Directory of pre-cut clips from a batch prepare_clips run.")
    mode.add_argument("--streaming", action="store_true", help="Cut each question's clips right before inference and delete them after; keeps no intermediate clips.")
    ap.add_argument("--keep-clips", action="store_true", help="Streaming: keep the cut clips in .stream_clips/ instead of deleting them after each question.")
    ap.add_argument("--series", default=None, help="Streaming: default series for questions without their own 'series' field.")
    ap.add_argument("--video-root", default=None, help="Streaming: root containing <series>/<episode>.mp4.")


def clip_provider(
    args: argparse.Namespace, questions_path: str | Path
) -> tuple[Callable[[str, dict[str, Any]], list[dict[str, Any]]], Callable[[list[dict[str, Any]]], None] | None]:
    """Return (get_clips, release_clips) for batch (--clips-dir) or --streaming mode."""
    if args.clips_dir:
        clips_dir = Path(args.clips_dir)

        def get_clips(key: str, q: dict[str, Any]) -> list[dict[str, Any]]:
            found = find_question_clips(clips_dir, key)
            expected = len((q.get("input_video") or {}).get("segments") or [])
            if len(found) != expected:
                raise FileNotFoundError(
                    f"{key}: expected {expected} pre-cut clip(s) in {clips_dir}, found {len(found)}; "
                    "re-run benchmark/preprocess/prepare_clips.py"
                )
            return [{"clip_path": str(p)} for p in found]

        return get_clips, None

    videos = io_utils.load_video_map(questions_path, series=args.series, video_root=args.video_root)
    stream_dir = Path(questions_path).parent / ".stream_clips"
    keep_clips = bool(getattr(args, "keep_clips", False))

    def get_clips(key: str, q: dict[str, Any]) -> list[dict[str, Any]]:
        return cut_question_clips(
            q,
            key,
            videos,
            stream_dir,
            series=args.series,
            # with --keep-clips a clip kept by an earlier run is reused as-is
            # (same reuse semantics as batch mode without --force)
            force=not keep_clips,
        )

    if keep_clips:
        return get_clips, None

    def release_clips(clips: list[dict[str, Any]]) -> None:
        for clip in clips:
            Path(clip.get("clip_path") or "").unlink(missing_ok=True)
        if stream_dir.exists() and not any(stream_dir.iterdir()):
            shutil.rmtree(stream_dir, ignore_errors=True)

    return get_clips, release_clips


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", required=True, help="Benchmark question file (episode or unified JSON).")
    ap.add_argument("--out", required=True, help="Output directory for the cut clips.")
    ap.add_argument("--series", default=None, help="Default series for questions without their own 'series' field.")
    ap.add_argument("--video-root", default=None, help="Root containing <series>/<episode>.mp4.")
    ap.add_argument("--qid", action="append", default=None, help="Only preprocess selected qid(s) or series/qid key(s).")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="Re-cut existing clips.")
    ap.add_argument("--keep-going", action="store_true", help="Print errors and continue.")
    args = ap.parse_args()

    run_contract_check(args.questions)
    questions = io_utils.load_questions(args.questions)
    videos = io_utils.load_video_map(args.questions, series=args.series, video_root=args.video_root)
    selected = select_questions(questions, args)
    clip_dir = Path(args.out)

    num_clips = 0
    errors: list[str] = []
    for pos, (key, q) in enumerate(selected, 1):
        print(f"[{pos}/{len(selected)}] {key}", flush=True)
        try:
            num_clips += len(
                cut_question_clips(q, key, videos, clip_dir, series=args.series, force=args.force)
            )
        except Exception as e:
            errors.append(f"{key}: {type(e).__name__}: {e}")
            if not args.keep_going:
                raise
    for err in errors:
        print(f"error: {err}", file=sys.stderr)
    print(json.dumps({"questions": len(selected), "clips": num_clips, "errors": len(errors), "out": str(clip_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
