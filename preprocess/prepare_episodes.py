#!/usr/bin/env python3
"""Build effective full-episode videos: cut opening/credits out of each episode.

For episode-level questions the model input is the whole episode minus its
opening theme / credits. In the released layout, each series directory contains
its source videos and ranges file:

  videos/sources/<series>/S01E01.mp4
  videos/sources/<series>/valid_ranges.json

- valid_ranges.json        manually reviewed ranges (preferred when present)
- valid_input_ranges.json  auto-generated ranges

Each episode's valid_segments are cut from the source video and concatenated
into one flat output file named <series>_<episode>.mp4. For example:

  videos/processed/episodes/friends_s01e01.mp4

Usage:

  # one series; ranges are discovered beside its local videos
  python3 preprocess/prepare_episodes.py \
    --series friends \
    --video-root videos/sources \
    --out videos/processed/episodes

  # every series under the released sources root
  python3 preprocess/prepare_episodes.py \
    --ranges-root videos/sources \
    --video-root videos/sources --out videos/processed/episodes
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from preprocess import io_utils, media  # noqa: E402

RANGES_BASENAMES = ("valid_ranges.json", "valid_input_ranges.json")  # manual first


def find_ranges_files(args: argparse.Namespace) -> list[Path]:
    if args.ranges:
        return [Path(p) for p in args.ranges]
    if not args.ranges_root:
        if not args.series:
            raise SystemExit("pass --series when --ranges and --ranges-root are omitted")
        series_dir = Path(args.video_root) / args.series
        for name in RANGES_BASENAMES:
            candidate = series_dir / name
            if candidate.exists():
                return [candidate]
        raise SystemExit(
            f"no valid_ranges.json or valid_input_ranges.json found in {series_dir}"
        )
    root = Path(args.ranges_root)
    files: list[Path] = []
    for series_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for name in RANGES_BASENAMES:
            candidate = series_dir / name
            if candidate.exists():
                files.append(candidate)
                break
    if not files:
        raise SystemExit(f"no per-series ranges files found under {root}")
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
                "out_video": str(Path(args.out) / f"{series}_{ep}.mp4".lower()),
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
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--ranges", action="append", default=None, help="valid_ranges.json / valid_input_ranges.json (repeatable, one per series).")
    src.add_argument("--ranges-root", default=None, help="Root containing <series>/valid_ranges.json files.")
    ap.add_argument("--series", default=None, help="Series name used to find <video-root>/<series>/valid_ranges.json when --ranges is omitted.")
    ap.add_argument("--video-root", required=True, help="Root containing <series>/<episode>.mp4 source videos.")
    ap.add_argument("--out", required=True, help="Output directory; effective episodes use flat <series>_<episode>.mp4 names.")
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
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
