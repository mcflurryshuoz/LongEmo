"""Read prepared videos and derive temporary frames or audio."""

from __future__ import annotations
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import wave


# Qwen2.5-Omni video defaults (qwen-omni-utils 0.0.9):
# https://github.com/QwenLM/Qwen2.5-Omni/blob/main/qwen-omni-utils/src/qwen_omni_utils/v2_5/vision_process.py
FPS = 2.0
MAX_FRAMES = 768
IMAGE_FACTOR = 28
VIDEO_MIN_PIXELS = 128 * IMAGE_FACTOR**2
VIDEO_MAX_PIXELS = 768 * IMAGE_FACTOR**2
VIDEO_TOTAL_PIXELS = 90316800
_SEEK_BATCH_SIZE = 4
_EXTRACTION_TIMEOUT = 300
_AUDIO_SAMPLE_RATE = 16000


def file_hash(path):
    """Stream a stable source checksum without loading the whole video."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_tool(name, purpose):
    executable = shutil.which(name)
    if not executable:
        raise RuntimeError(f"{name} is required for {purpose}; install it on PATH")
    return executable


def encode_video(video):
    """Encode the original MP4 as a Base64 video block for the API request."""
    path = Path(video["path"]).resolve()
    raw = path.read_bytes()
    if not raw:
        raise ValueError("video file is empty")
    return {
        "type": "video_url",
        "video_url": {
            "url": "data:video/mp4;base64," + base64.b64encode(raw).decode("ascii")
        },
    }

def _probe_audio(path, ffprobe, deadline):
    try:
        data = json.loads(
            _run_tool(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_entries",
                    "stream=index,start_time",
                    "-of",
                    "json",
                    str(path),
                ],
                deadline,
                timeout=30,
            )
        )
    except json.JSONDecodeError as exc:
        raise ValueError("ffprobe returned invalid audio metadata") from exc
    streams = data.get("streams", [])
    return streams[0] if streams else None


def has_audio_track(video):
    """Inspect the first audio stream without decoding or changing the video."""
    path = Path(video["path"]).resolve()
    ffprobe = _require_tool("ffprobe", "detecting video audio")
    return (
        _probe_audio(path, ffprobe, time.monotonic() + _EXTRACTION_TIMEOUT)
        is not None
    )


def extract_audio(video, *, required=False):
    """Return first-track PCM16 WAV and metadata, aligned to frame time zero.

    Preserve audio presentation timestamps relative to the first video stream:
    audio before video start is trimmed, and a delayed track begins with silence.
    Pad or trim to the video presentation duration, including its final frame.
    The source is never modified and extracted files exist only temporarily.
    """
    path = Path(video["path"]).resolve()
    ffprobe = _require_tool("ffprobe", "extracting video audio")
    deadline = time.monotonic() + _EXTRACTION_TIMEOUT
    if _probe_audio(path, ffprobe, deadline) is None:
        if required:
            raise ValueError("video has no audio track but audio is required")
        return None, {"audio": "no_audio_track"}
    ffmpeg = _require_tool("ffmpeg", "extracting video audio")
    duration, fps, start, _ = _probe_video(path, ffprobe, deadline)
    _, duration = _last_timestamp(path, ffprobe, duration, fps, start, deadline)
    samples = max(1, round(duration * _AUDIO_SAMPLE_RATE))
    with tempfile.TemporaryDirectory(prefix="longemobench-audio-") as temporary:
        output = Path(temporary) / "audio.wav"
        _run_tool(
            [
                ffmpeg,
                "-hide_banner",
                "-nostdin",
                "-loglevel",
                "error",
                "-copyts",
                "-threads",
                "1",
                "-i",
                str(path),
                "-map",
                "0:a:0",
                "-vn",
                "-sn",
                "-dn",
                "-af",
                f"asetpts=PTS-({start:.9f})/TB,"
                f"aresample={_AUDIO_SAMPLE_RATE}:async=1:first_pts=0,"
                f"apad=whole_len={samples},atrim=end_sample={samples},asetpts=N/SR/TB",
                "-ar",
                str(_AUDIO_SAMPLE_RATE),
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                "-threads",
                "1",
                "-f",
                "wav",
                str(output),
            ],
            deadline,
            timeout=_EXTRACTION_TIMEOUT,
        )
        raw = output.read_bytes()
        try:
            with wave.open(io.BytesIO(raw), "rb") as audio:
                if (
                    audio.getnchannels() != 1
                    or audio.getframerate() != _AUDIO_SAMPLE_RATE
                    or audio.getsampwidth() != 2
                    or audio.getcomptype() != "NONE"
                    or audio.getnframes() != samples
                    or len(audio.readframes(samples)) != samples * 2
                ):
                    raise ValueError(
                        "ffmpeg produced invalid or incomplete PCM16 WAV audio"
                    )
        except (wave.Error, EOFError) as exc:
            raise ValueError("ffmpeg produced invalid WAV audio") from exc
    metadata = {
        "audio": "separate_audio",
        "sample_rate": _AUDIO_SAMPLE_RATE,
        "channels": 1,
        "duration_seconds": samples / _AUDIO_SAMPLE_RATE,
        "video_start_seconds": start,
    }
    return {
        "type": "input_audio",
        "input_audio": {
            "data": base64.b64encode(raw).decode("ascii"),
            "format": "wav",
        },
    }, metadata


def _positive_integer(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_number(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"video has no valid positive {name}") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"video has no valid positive {name}")
    return number


def _run_tool(command, deadline, timeout=60):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RuntimeError("video media extraction timed out")
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=min(timeout, remaining),
        ).stdout
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("video media extraction timed out") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip()[-2000:]
        raise RuntimeError(f"{Path(command[0]).name} failed: {detail}") from exc
    except OSError as exc:
        raise RuntimeError(f"cannot run {Path(command[0]).name}: {exc}") from exc


def _probe_video(path, ffprobe, deadline):
    try:
        data = json.loads(
            _run_tool(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=duration,start_time,avg_frame_rate,r_frame_rate,nb_frames,width,height:format=duration",
                    "-of",
                    "json",
                    str(path),
                ],
                deadline,
                timeout=30,
            )
        )
    except json.JSONDecodeError as exc:
        raise ValueError("ffprobe returned invalid video metadata") from exc
    streams = data.get("streams", [])
    if not streams:
        raise ValueError("input has no video stream")
    stream = streams[0]
    # Some containers omit stream duration; do not replace an explicitly invalid one.
    duration = _positive_number(
        stream.get("duration", data.get("format", {}).get("duration")), "duration"
    )
    for dimension in ("width", "height"):
        value = stream.get(dimension)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"video has no valid {dimension}")
    fps = None
    for key in ("avg_frame_rate", "r_frame_rate"):
        try:
            numerator, separator, denominator = str(stream.get(key, "")).partition("/")
            candidate = float(numerator) / (float(denominator) if separator else 1)
            if math.isfinite(candidate) and candidate > 0:
                fps = candidate
                break
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
    if fps is None:
        raise ValueError("video has no valid positive frame rate")
    try:
        start = float(stream.get("start_time", 0))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("video has no valid start time") from exc
    if not math.isfinite(start):
        raise ValueError("video has no valid start time")
    try:
        frame_count = int(stream.get("nb_frames", 0))
    except (TypeError, ValueError, OverflowError):
        frame_count = 0
    return duration, fps, start, frame_count


def _last_timestamp(path, ffprobe, duration, fps, start, deadline):
    # Decoded presentation times locate the real last frame, including VFR clips.
    # ffprobe seeks to a preceding keyframe, so the interval may begin earlier.
    tail_start = max(start, start + duration - max(3.0, 8.0 / fps))
    try:
        data = json.loads(
            _run_tool(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-threads",
                    "1",
                    "-read_intervals",
                    f"{tail_start:.9f}%",
                    "-show_frames",
                    "-show_entries",
                    "frame=best_effort_timestamp_time,pts_time,duration_time",
                    "-of",
                    "json",
                    str(path),
                ],
                deadline,
                timeout=30,
            )
        )
    except json.JSONDecodeError as exc:
        raise ValueError("ffprobe returned invalid video timestamps") from exc
    timestamps = []
    for frame in data.get("frames", []):
        try:
            timestamp = (
                float(frame.get("best_effort_timestamp_time", frame.get("pts_time")))
                - start
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(timestamp) and timestamp >= -0.000001:
            try:
                frame_duration = _positive_number(
                    frame.get("duration_time"), "frame duration"
                )
            except ValueError:
                frame_duration = 1.0 / fps
            timestamps.append((max(0.0, timestamp), frame_duration))
    if not timestamps:
        raise ValueError("video has no usable frame timestamps near its end")
    last, frame_duration = max(timestamps)
    return last, max(duration, last + frame_duration)


def _probe_video_dimensions(path, ffprobe, deadline):
    """Read dimensions without changing the timing probe's audio-facing API."""
    try:
        data = json.loads(
            _run_tool(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height",
                    "-of",
                    "json",
                    str(path),
                ],
                deadline,
                timeout=30,
            )
        )
    except json.JSONDecodeError as exc:
        raise ValueError("ffprobe returned invalid video dimensions") from exc
    streams = data.get("streams", [])
    if not streams:
        raise ValueError("input has no video stream")
    dimensions = tuple(streams[0].get(name) for name in ("width", "height"))
    for name, value in zip(("width", "height"), dimensions):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"video has no valid {name}")
    return dimensions


def _frame_pixel_budget(frame_count, max_pixels=VIDEO_MAX_PIXELS):
    """Apply Qwen's total-video budget to the planned number of frames."""
    shared_budget = VIDEO_TOTAL_PIXELS * 2 / frame_count
    automatic_limit = max(
        min(VIDEO_MAX_PIXELS, shared_budget), int(VIDEO_MIN_PIXELS * 1.05)
    )
    return min(max_pixels, automatic_limit)


def _resize_dimensions(width, height, max_pixels):
    """Use Qwen's pixel-area sizing with dimensions on a 28-pixel grid.

    Rounding the minimum-size branch upward, or keeping a narrow dimension at
    one grid unit, can overshoot the area target just as in Qwen smart_resize.
    """
    if max(width, height) / min(width, height) > 200:
        raise ValueError("video aspect ratio must not exceed 200")
    dimensions = (width, height)
    resized = tuple(
        max(IMAGE_FACTOR, round(value / IMAGE_FACTOR) * IMAGE_FACTOR)
        for value in dimensions
    )
    area = math.prod(resized)
    if area > max_pixels:
        scale = math.sqrt(max_pixels / (width * height))
        rounding = math.floor
    elif area < VIDEO_MIN_PIXELS:
        scale = math.sqrt(VIDEO_MIN_PIXELS / (width * height))
        rounding = math.ceil
    else:
        return resized
    return tuple(
        max(IMAGE_FACTOR, rounding(value * scale / IMAGE_FACTOR) * IMAGE_FACTOR)
        for value in dimensions
    )


def sample_video_frames(
    video,
    *,
    fps=FPS,
    max_frames=MAX_FRAMES,
    max_pixels=VIDEO_MAX_PIXELS,
    total_pixels=None,
    start_seconds=0.0,
    end_seconds=None,
):
    """Sample video frames as timestamped JPEG request blocks with sampling metadata.

    Plan round(presentation duration * requested FPS) frames, with at least one
    frame and at most max_frames or the available source frames. The defaults
    use Qwen-Omni's 2 FPS, 768-frame cap, and per-frame/total pixel budgets.
    An explicit ``total_pixels`` instead caps the total sampled image area:
    it first limits the frame count using the minimum pixel budget, then
    divides the total budget among those frames without Qwen's multiplier.
    Frame dimensions are rounded to multiples of 28; small images are upscaled
    to the minimum pixel budget. Spread those
    samples uniformly from zero through the last presentation timestamp of the
    first video stream, never by seeking to exact EOF. Stream duration and source
    FPS locate a short decoded-frame probe near the end; frame PTS determines the
    actual endpoint even for variable frame rates. Presentation duration includes
    the final frame when reordered frames extend past the container's duration.
    Each accurate input seek
    returns the first available frame at or after the target (rounded down to
    microseconds). Four seeks share each ffmpeg process. Images use their source
    PTS, normalized to stream start, and repeat source frames are collapsed.
    ``start_seconds`` and ``end_seconds`` optionally restrict sampling to a
    presentation-time window (the returned timestamps remain relative to the
    complete video). A one-frame request samples the first frame in that
    window; short clips may yield fewer frames than requested. No audio is
    extracted. All JPEGs are temporary.
    """
    if isinstance(fps, bool) or not isinstance(fps, (int, float)):
        raise ValueError("fps must be a finite positive number")
    try:
        fps = float(fps)
    except OverflowError as exc:
        raise ValueError("fps must be a finite positive number") from exc
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a finite positive number")
    _positive_integer(max_frames, "max_frames")
    if (
        isinstance(max_pixels, bool)
        or not isinstance(max_pixels, int)
        or max_pixels < VIDEO_MIN_PIXELS
    ):
        raise ValueError(f"max_pixels must be an integer at least {VIDEO_MIN_PIXELS}")
    if total_pixels is not None and (
        isinstance(total_pixels, bool)
        or not isinstance(total_pixels, int)
        or total_pixels < VIDEO_MIN_PIXELS
    ):
        raise ValueError(f"total_pixels must be an integer at least {VIDEO_MIN_PIXELS}")
    try:
        interval_start = float(start_seconds)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("start_seconds must be finite and non-negative") from exc
    if not math.isfinite(interval_start) or interval_start < 0:
        raise ValueError("start_seconds must be finite and non-negative")
    if end_seconds is None:
        interval_end = None
    else:
        try:
            interval_end = float(end_seconds)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("end_seconds must be finite and greater than start_seconds") from exc
        if not math.isfinite(interval_end) or interval_end <= interval_start:
            raise ValueError("end_seconds must be finite and greater than start_seconds")
    path = Path(video["path"]).resolve()
    executables = {}
    for name in ("ffmpeg", "ffprobe"):
        executables[name] = shutil.which(name)
        if not executables[name]:
            raise RuntimeError(
                f"{name} is required for video frames; install it on PATH"
            )
    deadline = time.monotonic() + _EXTRACTION_TIMEOUT
    duration, source_fps, start, source_count = _probe_video(
        path, executables["ffprobe"], deadline
    )
    stream_duration = duration
    last, duration = _last_timestamp(
        path, executables["ffprobe"], duration, source_fps, start, deadline
    )
    # ``last`` is the last decodable presentation timestamp.  Sampling the
    # complete clip must retain the historical endpoint at ``last`` rather
    # than asking ffmpeg for a frame at the container duration (which often
    # lies after the final frame).
    interval_start = min(interval_start, last)
    full_clip = end_seconds is None
    interval_end = last if full_clip else min(interval_end, last)
    single_frame_clip = full_clip and last == 0
    if interval_end <= interval_start and not single_frame_clip:
        raise ValueError("requested frame interval does not overlap the video")
    interval_duration = (
        duration if single_frame_clip else interval_end - interval_start
    )
    # Compare before multiplying so even enormous finite FPS cannot overflow.
    count = (
        max_frames
        if fps >= max_frames / interval_duration
        else max(1, round(interval_duration * fps))
    )
    # Avoid duplicate work on clips with fewer frames than the requested cap.
    if source_count > 0:
        available = max(1, math.ceil(interval_duration * source_fps) + 1)
        count = min(count, available)
    elif interval_duration < count / source_fps:
        count = min(count, max(1, math.ceil(interval_duration * source_fps)))
    if single_frame_clip:
        count = 1
    if total_pixels is None:
        pixel_budget = _frame_pixel_budget(count, max_pixels)
    else:
        count = min(count, total_pixels // VIDEO_MIN_PIXELS)
        pixel_budget = min(max_pixels, total_pixels / count)
    width, height = _probe_video_dimensions(path, executables["ffprobe"], deadline)
    resized_width, resized_height = _resize_dimensions(width, height, pixel_budget)
    if total_pixels is not None:
        # Qwen's minimum-size/grid rounding can overshoot a tight area budget.
        while resized_width * resized_height > pixel_budget:
            if resized_width >= resized_height:
                resized_width -= IMAGE_FACTOR
            else:
                resized_height -= IMAGE_FACTOR
    # Leave one nominal frame interval before an arbitrary end so an accurate
    # seek can still land on an in-window PTS. VFR overshoot is filtered below.
    final_target = interval_end
    if end_seconds is not None and interval_end < last:
        final_target = max(interval_start, interval_end - 1.0 / source_fps)
    targets = (
        [
            interval_start
            + (final_target - interval_start) * i / (count - 1)
            for i in range(count)
        ]
        if count > 1
        else [interval_start]
    )
    blocks, timestamps = [], []
    with tempfile.TemporaryDirectory(prefix="longemobench-frames-") as temporary:
        directory = Path(temporary)
        for offset in range(0, count, _SEEK_BATCH_SIZE):
            batch = targets[offset : offset + _SEEK_BATCH_SIZE]
            command = [
                executables["ffmpeg"],
                "-hide_banner",
                "-nostdin",
                "-loglevel",
                "error",
                "-copyts",
            ]
            for target in batch:
                # Seek inside declared duration even if reordered final PTS
                # extends beyond it; select then advances to the real target.
                safe_target = target
                if target >= last - 1.0 / source_fps or target > max(0.0, stream_duration - 1.0 / source_fps):
                    safe_target = min(
                        target,
                        max(0.0, stream_duration - max(3.0, 8.0 / source_fps)),
                    )
                seek = math.floor(safe_target * 1_000_000) / 1_000_000
                command += ["-threads", "1", "-ss", f"{seek:.6f}", "-i", str(path)]
            for index in range(len(batch)):
                target_pts = (
                    start + math.floor(batch[index] * 1_000_000) / 1_000_000 - 0.000001
                )
                # ffprobe rounds PTS to six decimal places. Its last timestamp
                # can be up to half a microsecond beyond the real final PTS;
                # include that frame instead of filtering past EOF.
                command += [
                    "-map",
                    f"{index}:v:0",
                    "-frames:v",
                    "1",
                    "-an",
                    "-sn",
                    "-dn",
                    "-vf",
                    f"select='gte(t,{target_pts:.6f})',"
                    f"scale=w={resized_width}:h={resized_height}:flags=bicubic,setsar=1",
                    "-fps_mode",
                    "passthrough",
                    "-enc_time_base:v",
                    "1:1000000",
                    "-q:v",
                    "3",
                    "-pix_fmt",
                    "yuvj420p",
                    "-threads",
                    "1",
                    "-f",
                    "image2",
                    "-frame_pts",
                    "1",
                    str(directory / f"sample{offset + index}-%d.jpg"),
                ]
            _run_tool(command, deadline)
            for index in range(len(batch)):
                prefix = f"sample{offset + index}-"
                files = list(directory.glob(prefix + "*.jpg"))
                if len(files) != 1:
                    raise ValueError(
                        f"ffmpeg did not produce exactly one frame for sample {offset + index}"
                    )
                raw = files[0].read_bytes()
                if not raw.startswith(b"\xff\xd8") or not raw.endswith(b"\xff\xd9"):
                    raise ValueError(
                        f"ffmpeg produced an empty or invalid JPEG for sample {offset + index}"
                    )
                if os.environ.get("LONGEMO_FFMPEG_COMPAT") == "1":
                    # Ubuntu 20.04's FFmpeg 4.2 cannot set the output time
                    # base with -enc_time_base:v, so image2 frame PTS are not
                    # expressed in microseconds. The select filter still
                    # chooses the requested target; use that target for the
                    # local timestamp check in this explicitly scoped retry.
                    timestamp = round(batch[index] - start, 6)
                else:
                    try:
                        timestamp = round(
                            int(files[0].stem[len(prefix) :]) / 1_000_000 - start, 6
                        )
                    except ValueError as exc:
                        raise ValueError(
                            "ffmpeg produced a frame without a usable timestamp"
                        ) from exc
                if timestamp > interval_end + 0.000001:
                    # An accurate seek returns the first frame at/after the
                    # target. An arbitrary window end need not be an actual
                    # frame PTS; do not include or relabel that later frame.
                    continue
                if timestamp < interval_start - 0.000001:
                    raise ValueError(
                        "ffmpeg produced a frame outside the requested interval"
                    )
                timestamp = max(interval_start, min(interval_end, timestamp))
                if timestamps and timestamp < timestamps[-1]:
                    raise ValueError(
                        "ffmpeg produced frames out of chronological order"
                    )
                if timestamps and timestamp == timestamps[-1]:
                    continue
                timestamps.append(timestamp)
                blocks += [
                    {"type": "text", "text": f"Frame at {timestamp:.6f} seconds:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,"
                            + base64.b64encode(raw).decode("ascii")
                        },
                    },
                ]
    if not timestamps:
        raise ValueError("requested interval contains no sampled video frames")
    # Preserve the legacy full-clip duration/effective-FPS fields for a
    # default call.  Windowed callers get the interval duration while the
    # complete video duration is always available separately.
    windowed = start_seconds != 0 or end_seconds is not None
    reported_duration = interval_duration if windowed else duration
    metadata = {
        "sampling": "fps_capped_uniform_timestamp_seeks_v2",
        "timestamps_seconds": timestamps,
        "duration_seconds": reported_duration,
        "video_duration_seconds": duration,
        "interval_start_seconds": interval_start,
        "interval_end_seconds": interval_end,
        "stream_duration_seconds": stream_duration,
        "source_fps": source_fps,
        "requested_fps": fps,
        "max_frames": max_frames,
        "target_num_frames": count,
        "num_frames": len(timestamps),
        "effective_fps": len(timestamps) / reported_duration,
        "requested_max_pixels": max_pixels,
        "max_pixels": pixel_budget,
        "min_pixels": VIDEO_MIN_PIXELS,
        "total_pixels": VIDEO_TOTAL_PIXELS if total_pixels is None else total_pixels,
        "actual_total_pixels": resized_width * resized_height * len(timestamps),
        "resized_width": resized_width,
        "resized_height": resized_height,
    }
    return blocks, metadata
