"""Validation for the small Agentic inspection/answer protocol."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any
import json
import re

from evaluation.inference.video_loader import VIDEO_MAX_PIXELS

MAX_INSPECT_FRAMES = 768
MAX_INSPECT_PIXELS = VIDEO_MAX_PIXELS
MAX_INSPECT_FPS = 30.0


def decode_action(text: str, *, max_ranges: int = 8) -> dict[str, Any]:
    """Decode the single JSON action returned by a model."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("model action must be a non-empty JSON object")
    candidate = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, re.S | re.I)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError("model action is not valid JSON") from exc
    return parse_action(value, max_ranges=max_ranges)


def parse_action(value: Any, *, max_ranges: int = 8) -> dict[str, Any]:
    """Validate and normalize one model action.

    Inspection actions may request only a time window and visual sampling
    budget.  Audio and subtitles are fixed run-level inputs and are rejected
    when included in an action.  Answer payloads are intentionally opaque: the
    benchmark's question-specific output rules handle their contents.
    """

    if not isinstance(value, Mapping):
        raise ValueError("agent action must be a JSON object")
    action = value.get("action")
    if action == "answer":
        if "answer" not in value:
            raise ValueError("answer action requires an answer field")
        return {"action": "answer", "answer": value["answer"]}
    if action != "inspect":
        raise ValueError("agent action must have action=inspect or action=answer")
    if set(value) - {"action", "ranges"}:
        raise ValueError("inspect action may contain only action and ranges")
    ranges = value.get("ranges")
    if not isinstance(ranges, list) or not ranges:
        raise ValueError("inspect action requires a non-empty ranges list")
    if len(ranges) > max_ranges:
        raise ValueError(f"inspect action allows at most {max_ranges} ranges")
    normalized = []
    for item in ranges:
        if not isinstance(item, Mapping):
            raise ValueError("each inspection range must be an object")
        required = {"start", "end", "fps", "max_frames", "max_pixels"}
        if set(item) != required:
            raise ValueError(
                "inspection ranges require start, end, fps, max_frames and max_pixels"
            )
        try:
            start, end, fps = float(item["start"]), float(item["end"]), float(item["fps"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("range times and fps must be finite numbers") from exc
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or not math.isfinite(fps)
            or start < 0
            or end <= start
            or fps <= 0
            or fps > MAX_INSPECT_FPS
        ):
            raise ValueError("range requires 0 <= start < end and fps > 0")
        max_frames = item["max_frames"]
        max_pixels = item["max_pixels"]
        if (
            isinstance(max_frames, bool)
            or not isinstance(max_frames, int)
            or max_frames < 1
            or max_frames > MAX_INSPECT_FRAMES
            or isinstance(max_pixels, bool)
            or not isinstance(max_pixels, int)
            or max_pixels < 1
            or max_pixels > MAX_INSPECT_PIXELS
        ):
            raise ValueError(
                "inspection requests exceed the allowed frame, pixel or FPS budget"
            )
        normalized.append(
            {
                "start": start,
                "end": end,
                "fps": fps,
                "max_frames": max_frames,
                "max_pixels": max_pixels,
            }
        )
    return {"action": "inspect", "ranges": normalized}
