"""The unified question format and task-specific scoring routes."""

from __future__ import annotations
import math
import re
from typing import Any
import json
from pathlib import Path

EPISODE_SUBTITLES_DIR = Path(__file__).resolve().parents[1] / "subtitles"

EMOTIC_26_LABELS = (
    "peace",
    "affection",
    "esteem",
    "anticipation",
    "engagement",
    "confidence",
    "happiness",
    "pleasure",
    "excitement",
    "surprise",
    "sympathy",
    "doubt/confusion",
    "disconnection",
    "fatigue",
    "embarrassment",
    "yearning",
    "disapproval",
    "aversion",
    "annoyance",
    "anger",
    "sensitivity",
    "sadness",
    "disquietment",
    "fear",
    "pain",
    "suffering",
)
LABEL_TASKS = {"contextual emotion", "emotion transition", "emotion influence"}
TASKS = {
    "clip": LABEL_TASKS | {"emotion trajectory", "emotion cause"},
    "episode": {
        "emotional intensity comparison",
        "emotion trajectory",
        "emotional reasoning",
    },
}


def norm_label(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def question_key(q: dict) -> str:
    """Return the stable question identifier used within one granularity run."""
    return q["question_id"]


def _nonempty(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")


def _time(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def validate_subtitles(rows: Any) -> None:
    if rows is None:
        return
    if not isinstance(rows, list):
        raise ValueError("subtitles must be a list or null")
    ids = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "t", "speaker", "text"}:
            raise ValueError("each subtitle requires id, t, speaker, text")
        _nonempty(row["id"], "subtitle.id")
        _nonempty(row["text"], "subtitle.text")
        if row["id"] in ids:
            raise ValueError("duplicate subtitle id")
        ids.add(row["id"])
        t = row["t"]
        if (
            not isinstance(t, list)
            or len(t) != 2
            or not all(_time(x) for x in t)
            or t[1] < t[0]
        ):
            raise ValueError("subtitle.t must contain valid start/end seconds")
        if row["speaker"] is not None and not isinstance(row["speaker"], str):
            raise ValueError("subtitle.speaker must be text or null")


def load_records(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.is_dir():
        files = sorted(
            p
            for p in path.iterdir()
            if p.suffix in {".json", ".jsonl"}
            and not p.name.startswith("._")
            and p.is_file()
        )
        if not files:
            raise ValueError(f"no JSON/JSONL files in {path}")
        return [record for file in files for record in load_records(file)]
    with path.open(encoding="utf-8") as stream:
        value = (
            [json.loads(line) for line in stream if line.strip()]
            if path.suffix == ".jsonl"
            else json.load(stream)
        )
    records = value if isinstance(value, list) else [value]
    if any(not isinstance(record, dict) for record in records):
        raise ValueError(f"{path}: records must be JSON objects")
    return records


def load_questions(path, granularity, ids=None, *, limit=None):
    """Load prepared questions and select by granularity, ID and count."""
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive integer or None")
    selected = []
    seen = set()
    for q in load_records(path):
        # Select by the explicit field; no legacy ID-based granularity inference.
        if q.get("granularity") not in TASKS:
            raise ValueError("every question requires clip or episode granularity")
        if q.get("granularity") != granularity:
            continue
        key = question_key(q)
        if key in seen:
            raise ValueError(f"duplicate question: {key}")
        seen.add(key)
        if not ids or q["question_id"] in ids:
            selected.append(q)
    if ids and set(ids) - {q["question_id"] for q in selected}:
        raise ValueError(
            "requested question IDs were not found in the selected granularity"
        )
    if not selected:
        raise ValueError(f"no {granularity} questions found")
    # Check duplicate and requested IDs before applying the count limit.
    return selected[:limit]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_records(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(x, ensure_ascii=False, allow_nan=False) + "\n" for x in records
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def subtitle_text(q, *, required=True):
    """Read clip subtitles from questions and episode subtitles from the dataset."""
    if q["granularity"] == "episode":
        path = EPISODE_SUBTITLES_DIR / (q["video_id"] + ".json")
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(
                f"{question_key(q)}: missing episode subtitles file: {path}"
            ) from exc
    else:
        rows = q["subtitles"]
    validate_subtitles(rows)
    if not rows:
        if not required:
            return None
        raise ValueError(f"{question_key(q)}: non-empty subtitles are required")
    # Input order follows the prepared video; original timestamps may reset across cuts.
    return "\n".join(row["text"].strip() for row in rows)
