#!/usr/bin/env python3
"""Shared I/O helpers for benchmark preprocessing, inference, and evaluation."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_questions(path: str | Path) -> list[dict[str, Any]]:
    """Load question/prediction records from JSONL or a JSON list."""
    p = Path(path)
    if p.suffix == ".jsonl":
        return read_jsonl(p)
    data = read_json(p)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    raise ValueError(f"Unsupported record file shape: {p}")


def safe_qid(item: dict[str, Any], idx: int) -> str:
    raw = str(item.get("qid") or f"q{idx:04d}")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_") or f"q{idx:04d}"


def question_key(item: dict[str, Any], idx: int) -> str:
    """Unique question key: "<series>/<qid>".

    qids repeat across series (e.g. g1_2_S01E12_q002 exists in several shows),
    so any index over a merged multi-series file must include the series.
    """
    qid = safe_qid(item, idx)
    series = str(item.get("series") or "").strip()
    return f"{series}/{qid}" if series else qid


def index_by_key(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for i, r in enumerate(records, 1):
        out[question_key(r, i)] = r
    return out


def option_lines(options: list[Any]) -> list[str]:
    lines: list[str] = []
    for i, opt in enumerate(options or []):
        label = chr(ord("A") + i)
        text = str(opt)
        if re.match(r"^\s*[A-Z]\s*[\.\)]", text):
            lines.append(text.strip())
        else:
            lines.append(f"{label}. {text.strip()}")
    return lines


def question_text(q: dict[str, Any]) -> str:
    text = str(q.get("question") or "").strip()
    opts = q.get("options") or []
    if opts:
        labels_present = bool(re.search(r"(^|\n)\s*[A-Z]\s*[\.\)]", text))
        if not labels_present:
            text = "\n".join([text, *option_lines(opts)])
    return text.strip()


def parse_jsonish(raw: str | None) -> Any:
    if not raw:
        return None
    text = str(raw).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def question_series(q: dict[str, Any], default: str | None = None) -> str | None:
    """Per-question series field, falling back to the CLI-level default."""
    return str(q.get("series") or default or "") or None


def video_key(series: str, ep: str) -> str:
    """Video-map key; includes series so episode ids never collide across shows."""
    return f"{series}/{ep}"


def load_video_map(
    questions_path: str | Path,
    *,
    series: str | None = None,
    video_root: str | Path | None = None,
) -> dict[str, str]:
    """Build "<series>/<ep>" -> video path from the questions themselves.

    Episode ids come from each question's input_video (eps + segments); the
    series comes from the question's own "series" field, falling back to
    --series. Paths follow <video_root>/<series>/<ep>.mp4. No pipeline
    intermediates (qgen_input.json etc.) are consulted.
    """
    root = Path(
        video_root
        or os.environ.get("P4_VIDEO_ROOT")
        or os.environ.get("SEG_VIDEO_ROOT")
        or "/Volumes/Passport_2/datasets/new_video_dataset/videos"
    )
    videos: dict[str, str] = {}
    questions = load_questions(questions_path)
    for i, q in enumerate(questions, 1):
        q_series = question_series(q, default=series)
        if not q_series:
            raise ValueError(
                f"{safe_qid(q, i)}: cannot infer series; add a 'series' field to the question or pass --series"
            )
        iv = q.get("input_video") or {}
        eps = [str(ep) for ep in iv.get("eps") or []]
        eps += [str(seg["ep"]) for seg in iv.get("segments") or [] if seg.get("ep")]
        for ep in eps:
            videos[video_key(q_series, ep)] = str(root / q_series / f"{ep}.mp4")
    return videos


