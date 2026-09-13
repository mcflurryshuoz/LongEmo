"""Time-windowed evidence assembly for the Agentic video method.

The agentic controller decides *where* to look again.  Whether audio and
subtitles are available is fixed when a run starts and is therefore passed to
these helpers as already prepared blocks/rows; an inspection action cannot
turn either source on or off.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math
from typing import Any

try:  # ``LongEmoBenchv1`` package or repository-root execution
    from ...evaluation.inference.video_loader import (
        FPS,
        MAX_FRAMES,
        VIDEO_MAX_PIXELS,
        sample_video_frames,
    )
except ImportError:  # pragma: no cover - selected by the repository CLI layout
    from evaluation.inference.video_loader import (
        FPS,
        MAX_FRAMES,
        VIDEO_MAX_PIXELS,
        sample_video_frames,
    )


def sample_interval(
    video: Mapping[str, Any],
    *,
    start: float,
    end: float,
    fps: float = FPS,
    max_frames: int = MAX_FRAMES,
    max_pixels: int = VIDEO_MAX_PIXELS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Sample one presentation-time interval from ``video``.

    ``start`` and ``end`` are seconds relative to the complete video.  The
    returned frame blocks use the same canonical content format as the regular
    inference path and metadata retains absolute timestamps, so a controller
    can merge observations from multiple rounds without losing location.
    """

    blocks, metadata = sample_video_frames(
        video,
        fps=fps,
        max_frames=max_frames,
        max_pixels=max_pixels,
        start_seconds=start,
        end_seconds=end,
    )
    metadata = dict(metadata)
    metadata.update({"requested_start": float(start), "requested_end": float(end)})
    return blocks, metadata


# More descriptive aliases for callers that use "frames" terminology.
sample_frames = sample_interval
sample_video_interval = sample_interval


def select_subtitles(
    subtitles: Sequence[Mapping[str, Any]] | None,
    *,
    start: float,
    end: float,
) -> list[dict[str, Any]]:
    """Select subtitle rows that overlap a requested time interval.

    Rows are copied, preserving their IDs and original timestamps.  Boundary
    touching is considered overlap because a subtitle ending exactly when an
    inspection begins can still provide useful context.  This helper only
    filters the run's fixed subtitle input; it does not enable subtitles.
    """

    _validate_interval(start, end)
    if subtitles is None:
        return []
    selected: list[dict[str, Any]] = []
    for row in subtitles:
        if not isinstance(row, Mapping):
            raise ValueError("subtitle rows must be objects")
        timing = row.get("t")
        if (
            not isinstance(timing, Sequence)
            or isinstance(timing, (str, bytes))
            or len(timing) != 2
        ):
            raise ValueError("each subtitle requires a two-item t field")
        try:
            row_start, row_end = float(timing[0]), float(timing[1])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("subtitle timestamps must be finite numbers") from exc
        if (
            not math.isfinite(row_start)
            or not math.isfinite(row_end)
            or row_end < row_start
        ):
            raise ValueError("subtitle timestamps must be finite and ordered")
        if row_end >= start and row_start <= end:
            selected.append(dict(row))
    return selected


def subtitle_text(rows: Iterable[Mapping[str, Any]]) -> str:
    """Render selected subtitle rows as a compact, timestamped text block."""

    lines = []
    for row in rows:
        timing = row.get("t")
        speaker = row.get("speaker")
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        start, end = timing
        who = f"{speaker}: " if speaker else ""
        lines.append(f"[{float(start):.3f}-{float(end):.3f}] {who}{text}")
    return "\n".join(lines)


def compose_evidence(
    frame_blocks: Sequence[Mapping[str, Any]],
    *,
    audio_block: Mapping[str, Any] | None = None,
    subtitle_rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Combine a sampled frame block with fixed audio and subtitle evidence.

    ``audio_block`` is the block prepared at run initialization (or ``None``
    when audio was disabled).  Subtitle rows are similarly already filtered
    by the caller's fixed-input policy; this function does not accept action
    flags that could change those policies mid-run.
    """

    blocks = [dict(block) for block in frame_blocks]
    if audio_block is not None:
        blocks.append(dict(audio_block))
    if subtitle_rows:
        text = subtitle_text(subtitle_rows)
        if text:
            blocks.append({"type": "text", "text": "Subtitles:\n" + text})
    return blocks


def _validate_interval(start: float, end: float) -> None:
    try:
        start_value, end_value = float(start), float(end)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("interval bounds must be finite numbers") from exc
    if (
        not math.isfinite(start_value)
        or not math.isfinite(end_value)
        or start_value < 0
        or end_value <= start_value
    ):
        raise ValueError("interval requires 0 <= start < end")
