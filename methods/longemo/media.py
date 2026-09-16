"""Timestamp-preserving window input. All temporary media are bounded to one window."""
from __future__ import annotations

import base64
import json
from pathlib import Path
import subprocess
import tempfile

from evaluation.inference.video_loader import sample_video_frames
from evaluation.io_utils import validate_subtitles


def probe(path):
    output = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries",
        "format=duration:stream=codec_type,start_time,duration", "-of", "json", str(path)], timeout=30)
    metadata = json.loads(output)
    video = next(s for s in metadata["streams"] if s["codec_type"] == "video")
    return {"duration": float(video.get("duration", metadata["format"]["duration"])),
            "start_time": float(video.get("start_time", 0)),
            "has_audio": any(s["codec_type"] == "audio" for s in metadata["streams"])}


def subtitles(path):
    rows = json.loads(Path(path).read_text())
    validate_subtitles(rows)
    return rows


def subtitle_window(rows, start, end):
    return [r for r in rows if r["t"][1] > start and r["t"][0] < end]


def render_subtitles(rows):
    return "\n".join(f"[{r['id']} {r['t'][0]:.2f}-{r['t'][1]:.2f}s {r['speaker'] or 'unknown speaker'}] {r['text']}" for r in rows)


def window_input(video, start, end, *, fps, max_frames, max_pixels, with_audio, subtitle_rows):
    frames, metadata = sample_video_frames({"path": str(video)}, fps=fps, max_frames=max_frames,
        max_pixels=max_pixels, start_seconds=start, end_seconds=end)
    content = list(frames)
    info = probe(video)
    if with_audio:
        if not info["has_audio"]:
            raise ValueError("audio requested but source has no audio stream")
        with tempfile.TemporaryDirectory(prefix="longemo-audio-") as directory:
            output = Path(directory) / "window.wav"
            # Preserve input stream offsets, align PTS to the video's origin, then
            # trim using absolute video seconds. Missing leading audio becomes silence.
            filt = (f"asetpts=PTS-({info['start_time']})/TB,aresample=16000:async=1:first_pts=0,"
                    f"apad,atrim=start={start}:end={end},asetpts=PTS-STARTPTS")
            subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-copyts",
                "-i", str(video), "-map", "0:a:0", "-af", filt, "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le", "-threads", "1", "-y", str(output)], check=True, timeout=180)
            content += [{"type": "text", "text": f"Audio window: {start:.3f} to {end:.3f} seconds. Audio-local zero corresponds to video {start:.3f}s."},
                        {"type": "input_audio", "input_audio": {"format": "wav", "data": base64.b64encode(output.read_bytes()).decode()}}]
    rows = subtitle_window(subtitle_rows, start, end)
    if rows:
        content.append({"type": "text", "text": "Timestamped subtitles (speaker may be unknown):\n" + render_subtitles(rows)})
    metadata.update(window=[start, end], audio=with_audio, subtitle_ids=[r["id"] for r in rows])
    return content, metadata
