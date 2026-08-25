#!/usr/bin/env python3
"""Shared I/O helpers for benchmark preprocessing, inference, and evaluation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_questions(path: str | Path) -> list[dict[str, Any]]:
    """Load the benchmark's canonical JSON list."""
    data = read_json(path)
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError(f"question file must contain a JSON list of objects: {path}")
    return data


def question_key(item: dict[str, Any]) -> str:
    """Unique question key: "<series>/<qid>".

    qids repeat across series (e.g. g1_2_S01E12_q002 exists in several shows),
    so any index over a merged multi-series file must include the series.
    """
    return f"{item['series']}/{item['qid']}"
