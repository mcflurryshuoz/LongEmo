#!/usr/bin/env python3
"""Small ffmpeg helpers used by preprocessing and inference."""
from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path


def video_duration(path: str | Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


def cut_clip(
    video: str | Path,
    out_path: str | Path,
    start: float,
    end: float,
    *,
    force: bool = False,
) -> Path:
    """Extract [start, end] as an audio-visual mp4, keeping the source video's
    resolution, frame rate, and audio channels (re-encoded only for frame-exact
    cut points)."""
    out = Path(out_path)
    if out.exists() and out.stat().st_size > 1024 and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-i",
            str(video),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(out),
        ],
        check=True,
    )
    return out


def concat_clips(parts: list[str | Path], out_path: str | Path) -> Path:
    """Concatenate clips cut with cut_clip (same codec params) without re-encoding."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    list_file = out.with_suffix(".concat.txt")
    list_file.write_text("".join(f"file '{Path(p).resolve()}'\n" for p in parts), encoding="utf-8")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_file),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(out),
            ],
            check=True,
        )
    finally:
        list_file.unlink(missing_ok=True)
    return out


def data_url(path: str | Path, mime: str = "video/mp4") -> str:
    raw = Path(path).read_bytes()
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"
