#!/usr/bin/env python3
"""Build effective full-episode videos: cut opening/credits out of each episode.

For episode-level and longer granularities the model input is the whole episode
minus its opening theme / credits. The effective time ranges come from the
s5_effective_inputs files produced by the construction pipeline
(code/vebench/outputs/<series>/s5_effective_inputs/):

- valid_ranges.json        manually reviewed ranges (preferred when present)
- valid_input_ranges.json  auto-generated ranges

Each episode's valid_segments are cut from the source video and concatenated
into one mp4 at <out>/<series>/<ep>.mp4 — the same <root>/<series>/<ep>.mp4
layout as the original video root, so inference can point --video-root at the
output directory.

Usage:

  # one series, explicit ranges file
  python3 preprocess/prepare_episodes.py \
    --ranges vebench/outputs/friends/s5_effective_inputs/valid_ranges.json \
    --video-root /Volumes/Passport_2/datasets/new_video_dataset/videos \
    --out runs/effective_episodes

  # every series under a vebench outputs root (manual file preferred per series)
  python3 preprocess/prepare_episodes.py \
    --ranges-root vebench/outputs \
    --video-root ... --out runs/effective_episodes
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from benchmark.common import io_utils, media  # noqa: E402

RANGES_BASENAMES = ("valid_ranges.json", "valid_input_ranges.json")  # manual first


def find_ranges_files(args: argparse.Namespace) -> list[Path]:
    if args.ranges:
        return [Path(p) for p in args.ranges]
    root = Path(args.ranges_root)
    files: list[Path] = []
    for series_dir in sorted(p for p in root.iterdir() if (p / "s5_effective_inputs").is_dir()):
        for name in RANGES_BASENAMES:
            candidate = series_dir / "s5_effective_inputs" / name
            if candidate.exists():
                files.append(candidate)
                break
    if not files:
        raise SystemExit(f"no s5_effective_inputs ranges files found under {root}")
    return files


def episode_jobs(ranges: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    """One job per episode: source path, effective segments, output path."""
    series = str(ranges.get("series") or "")
    if not series:
        raise ValueError("ranges file has no 'series' field")
    wanted = set(args.eps or [])
    jobs = []
    for ep, info in sorted((ranges.get("episodes") or {}).items()):
        if wanted and ep not in wanted:
            continue
        status = str(info.get("status") or "")
        if args.verified_only and status != "verified":
            continue
        segments = []
        for seg in info.get("valid_segments") or []:
            start, end = float(seg["start"]), float(seg["end"])
            if end <= start:
                raise ValueError(f"{series}/{ep}: valid segment end <= start ({seg})")
            segments.append({"start": start, "end": end, "label": seg.get("label")})
        if not segments:
            raise ValueError(f"{series}/{ep}: no valid_segments")
        jobs.append(
            {
                "series": series,
                "ep": ep,
                "status": status,
                "segments": segments,
                "effective_duration": round(sum(s["end"] - s["start"] for s in segments), 1),
                "source_video": str(Path(args.video_root) / series / f"{ep}.mp4"),
                "out_video": str(Path(args.out) / series / f"{ep}.mp4"),
            }
        )
    return jobs


def process_job(job: dict[str, Any], *, force: bool) -> None:
    out = Path(job["out_video"])
    if out.exists() and out.stat().st_size > 1024 and not force:
        return
    src = Path(job["source_video"])
    if not src.exists():
        raise FileNotFoundError(f"missing source video: {src}")
    segments = job["segments"]
    if len(segments) == 1:
        media.cut_clip(src, out, segments[0]["start"], segments[0]["end"], force=True)
        return
    parts_dir = out.parent / ".parts"
    parts = []
    try:
        for j, seg in enumerate(segments, 1):
            part = parts_dir / f"{job['series']}_{job['ep']}_part{j}.mp4"
            media.cut_clip(src, part, seg["start"], seg["end"], force=True)
            parts.append(part)
        media.concat_clips(parts, out)
    finally:
        shutil.rmtree(parts_dir, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ranges", action="append", default=None, help="valid_ranges.json / valid_input_ranges.json (repeatable, one per series).")
    src.add_argument("--ranges-root", default=None, help="vebench outputs root; picks each series' s5_effective_inputs file (manual valid_ranges.json preferred).")
    ap.add_argument("--video-root", required=True, help="Root containing <series>/<episode>.mp4 source videos.")
    ap.add_argument("--out", required=True, help="Output root; effective episodes land at <out>/<series>/<ep>.mp4.")
    ap.add_argument("--eps", action="append", default=None, help="Only process selected episode id(s), e.g. S01E02.")
    ap.add_argument("--verified-only", action="store_true", help="Skip episodes whose ranges status is not 'verified' (auto files contain drafts).")
    ap.add_argument("--force", action="store_true", help="Re-cut existing outputs.")
    ap.add_argument("--keep-going", action="store_true", help="Print errors and continue.")
    args = ap.parse_args()

    jobs: list[dict[str, Any]] = []
    for path in find_ranges_files(args):
        ranges = io_utils.read_json(path)
        for job in episode_jobs(ranges, args):
            jobs.append({**job, "ranges_file": str(path)})

    done = 0
    errors: list[str] = []
    for pos, job in enumerate(jobs, 1):
        print(f"[{pos}/{len(jobs)}] {job['series']}/{job['ep']}", flush=True)
        try:
            process_job(job, force=args.force)
            done += 1
        except Exception as e:
            errors.append(f"{job['series']}/{job['ep']}: {type(e).__name__}: {e}")
            if not args.keep_going:
                raise
    for err in errors:
        print(f"error: {err}", file=sys.stderr)
    print(json.dumps({"episodes": done, "errors": len(errors), "out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
