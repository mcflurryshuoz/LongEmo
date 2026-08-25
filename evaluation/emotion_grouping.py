#!/usr/bin/env python3
"""Expand closed emotion answers into deterministic scoring slots."""
from __future__ import annotations

from typing import Any

from evaluation import contract


def extract_gold_slots(q: dict[str, Any]) -> dict[str, Any]:
    """Return one direct slot or the two directional transition slots."""
    answer = q.get("answer")
    if q.get("prompt_template") == "emotion_labels" and isinstance(answer, list):
        return {"emotion": answer}
    if q.get("prompt_template") == "emotion_before_after" and isinstance(answer, dict):
        return {field: answer[field] for field in contract.TRANSITION_FIELDS if answer.get(field)}
    return {}


def extract_pred_slots(q: dict[str, Any], pred_answer: Any) -> dict[str, Any]:
    if q.get("prompt_template") == "emotion_before_after":
        if isinstance(pred_answer, dict):
            return {field: pred_answer.get(field, []) for field in contract.TRANSITION_FIELDS}
        return {field: [] for field in contract.TRANSITION_FIELDS}
    return {"emotion": contract.emotion_terms(pred_answer)}


def emotion_slots(q: dict[str, Any], pred_answer: Any) -> list[dict[str, Any]]:
    """Expand one closed-emotion question into independently scored slots."""
    gold_slots = extract_gold_slots(q)
    if not gold_slots:
        return []
    pred_slots = extract_pred_slots(q, pred_answer)
    return [
        {
            "series": q.get("series"),
            "qid": q.get("qid"),
            "slot": slot,
            "question_type": q.get("question_type"),
            "gold_answer": q.get("answer"),
            "pred_answer": pred_answer,
            "gold_labels": contract.emotion_terms(gold_value),
            "pred_labels": contract.emotion_terms(pred_slots.get(slot, [])),
        }
        for slot, gold_value in gold_slots.items()
    ]
