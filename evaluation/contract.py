#!/usr/bin/env python3
"""Contracts for clip-level and episode-level benchmark questions."""
from __future__ import annotations

import re
from typing import Any

PROMPT_TEMPLATES = {
    "emotion_labels",
    "emotion_before_after",
    "ranking_letters",
    "single_choice_letter",
    "short_answer",
}
EMOTION_TEMPLATES = {"emotion_labels", "emotion_before_after"}
TRANSITION_FIELDS = ("before", "after")

EMOTIC_26_LABELS = (
    "peace", "affection", "esteem", "anticipation", "engagement", "confidence",
    "happiness", "pleasure", "excitement", "surprise", "sympathy", "doubt/confusion",
    "disconnection", "fatigue", "embarrassment", "yearning", "disapproval", "aversion",
    "annoyance", "anger", "sensitivity", "sadness", "disquietment", "fear", "pain", "suffering",
)
EMOTIC_26_SET = set(EMOTIC_26_LABELS)

def question_granularity(q: dict[str, Any]) -> str:
    """Read the released granularity from the question-id namespace."""
    qid = q["qid"]
    if qid.startswith("g1_"):
        return "1_clip"
    if qid.startswith("g3_"):
        return "2_episode"
    raise ValueError(f"qid does not identify a released granularity: {qid!r}")


def norm_label(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def emotion_terms(value: Any) -> list[str]:
    """Normalize a canonical list of emotion labels."""
    if not isinstance(value, list):
        return []

    out: list[str] = []
    for label in value:
        if not isinstance(label, str):
            continue
        term = label.strip(" \t\r\n\"'`.,;:，；。")
        if term:
            out.append(norm_label(term))
    return list(dict.fromkeys(out))


def option_label_set(q: dict[str, Any]) -> set[str]:
    labels: set[str] = set()
    for option in q.get("options") or []:
        match = re.match(r"^\s*([A-Z])\s*[\.\)]", str(option))
        if not match:
            raise ValueError(f"option is missing its letter label: {option!r}")
        labels.add(match.group(1))
    return labels


def letter_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not re.fullmatch(r"[A-Z]", item):
            return []
        out.append(item)
    return out


def is_emotion_question(q: dict[str, Any]) -> bool:
    return q.get("prompt_template") in EMOTION_TEMPLATES


def is_transition_question(q: dict[str, Any]) -> bool:
    return q.get("prompt_template") == "emotion_before_after"


def is_yes_no_question(q: dict[str, Any]) -> bool:
    if q.get("question_type") != "open_ended" or is_emotion_question(q):
        return False
    text = re.sub(r"\s+", " ", str(q.get("question") or "").strip())
    return bool(re.search(
        r"answer\s+(?:yes\s+or\s+no|[\"“]?yes[\"”]?\s+or\s+[\"“]?no[\"”]?)\.?$",
        text,
        re.I,
    ))


def _emotion_list_errors(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(v, str) and v for v in value):
        return [f"{field} must be a non-empty list of emotion labels"]
    invalid = sorted(set(value) - EMOTIC_26_SET)
    return [f"{field} contains labels outside the allowed 26-label set: {invalid}"] if invalid else []


def validate_prediction_format(q: dict[str, Any], pred_answer: Any) -> list[str]:
    """Validate the canonical prediction emitted by the prompt parser."""
    if pred_answer is None:
        return []
    template = q.get("prompt_template")
    labels = option_label_set(q)
    if template == "emotion_labels":
        return _emotion_list_errors(pred_answer, "prediction")
    if template == "emotion_before_after":
        if not isinstance(pred_answer, dict) or set(pred_answer) != set(TRANSITION_FIELDS):
            return ['prediction must be exactly {"before": [...], "after": [...]}']
        errors: list[str] = []
        for field in TRANSITION_FIELDS:
            errors.extend(_emotion_list_errors(pred_answer[field], f"prediction.{field}"))
        return errors
    if template == "ranking_letters":
        got = letter_list(pred_answer)
        if not got:
            return ["ranking prediction must be an ordered list of option letters"]
        return [] if set(got).issubset(labels) else ["ranking prediction contains letters outside options"]
    if q.get("question_type") == "single_choice":
        if not isinstance(pred_answer, str) or not re.fullmatch(r"[A-Z]", pred_answer):
            return ["single-choice prediction must be one option letter"]
        return [] if pred_answer in labels else ["single-choice prediction letter is outside options"]
    if not isinstance(pred_answer, str) or not pred_answer.strip():
        return ["open prediction must be a non-empty string"]
    if is_yes_no_question(q) and pred_answer not in {"Yes", "No"}:
        return ['Yes/No prediction must be exactly "Yes" or "No"']
    return []
