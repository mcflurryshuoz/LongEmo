"""Granularity-aware model prompts and output parsing."""
from __future__ import annotations

import re
from typing import Any

from evaluation import contract


_GENERAL_INSTRUCTIONS = (
    "Watch the entire video before answering. Use the visual, audio, dialogue,\n"
    "and temporal context provided by the video.\n\n"
    "Respond with the final answer only, using the format specified below."
)

_ANSWER_FORMATS = {
    "emotion_labels": (
        "- Select one or more emotion labels exclusively from the options listed above.\n"
        "- Copy every selected label exactly as written.\n"
        "- Do not introduce synonyms, paraphrases, explanations, or labels outside the list.\n"
        "- Output only the selected labels, separated by commas."
    ),
    "emotion_before_after": (
        "- For both stages, select one or more emotion labels exclusively from the options "
        "listed above.\n"
        "- Copy every selected label exactly as written.\n"
        "- Do not introduce synonyms, paraphrases, explanations, or labels outside the list.\n"
        "- Return exactly two lines in the following format:\n"
        "Before: <label>[, <label> ...]\n"
        "After: <label>[, <label> ...]"
    ),
    "ranking_letters": (
        "Return only the selected option letters in chronological order, separated by spaces."
    ),
    "single_choice_letter": "Return one option letter only.",
}


def _subtitle_block(record: dict[str, Any], with_transcript: bool) -> str | None:
    if not with_transcript or not record.get("input_transcript"):
        return None
    rows = [str(row.get("text") or "").strip() for row in record["input_transcript"]]
    rows = [row for row in rows if row]
    if not rows:
        return None
    return (
        "The subtitles below are listed in chronological order and contain no speaker labels:\n"
        + "\n".join(rows)
    )


def build_prompt(record: dict[str, Any], *, with_transcript: bool = True) -> str:
    """Build the exact user-message text for either released granularity."""
    granularity = contract.question_granularity(record)

    template = record.get("prompt_template")
    if granularity == "1_clip" and template not in contract.PROMPT_TEMPLATES:
        raise ValueError(f"1_clip question has invalid prompt_template: {template!r}")
    if granularity == "2_episode" and template is not None:
        raise ValueError("2_episode questions must not use prompt_template")

    parts = [_GENERAL_INSTRUCTIONS]
    if template in contract.EMOTION_TEMPLATES:
        parts.append(
            "Emotion label options (closed set; choose only from this list):\n"
            + ", ".join(contract.EMOTIC_26_LABELS)
            + "."
        )
    subtitles = _subtitle_block(record, with_transcript)
    if subtitles:
        parts.append(subtitles)
    parts.append("Question:\n" + str(record.get("question") or "").strip())

    options = record.get("options") or []
    if options:
        parts.append("Options:\n" + "\n".join(str(option) for option in options))

    answer_format = _ANSWER_FORMATS.get(template)
    if granularity == "2_episode" and record.get("question_type") == "single_choice":
        answer_format = _ANSWER_FORMATS["single_choice_letter"]
    if answer_format:
        parts.append("Answer format:\n" + answer_format)
    return "\n\n".join(parts).strip()


def _parse_emotions(text: str) -> tuple[list[str] | None, str | None]:
    values = [contract.norm_label(value) for value in text.split(",") if value.strip()]
    values = list(dict.fromkeys(values))
    if not values:
        return None, "no emotion label was returned"
    invalid = sorted(set(values) - contract.EMOTIC_26_SET)
    if invalid:
        return values, f"emotion labels outside the allowed set: {invalid}"
    return values, None


def parse_answer(record: dict[str, Any], raw_answer: str) -> tuple[Any, str | None]:
    """Parse final-only model text according to the active prompt contract."""
    text = str(raw_answer or "").strip()
    template = record.get("prompt_template")
    granularity = contract.question_granularity(record)

    if template == "emotion_labels":
        parsed, error = _parse_emotions(text)
        return (parsed if parsed is not None else text), error

    if template == "emotion_before_after":
        before = re.findall(r"^\s*Before\s*:\s*(.+?)\s*$", text, flags=re.I | re.M)
        after = re.findall(r"^\s*After\s*:\s*(.+?)\s*$", text, flags=re.I | re.M)
        if len(before) != 1 or len(after) != 1:
            return text, "expected exactly one Before line and one After line"
        before_values, before_error = _parse_emotions(before[0])
        after_values, after_error = _parse_emotions(after[0])
        if before_error or after_error:
            return text, "; ".join(error for error in (before_error, after_error) if error)
        return {"before": before_values, "after": after_values}, None

    if template == "ranking_letters":
        tokens = text.upper().split()
        allowed = contract.option_label_set(record)
        if not tokens or any(not re.fullmatch(r"[A-Z]", token) for token in tokens):
            return text, "expected option letters separated by spaces"
        if not set(tokens).issubset(allowed):
            return text, "ranking contains a letter outside the options"
        return tokens, None

    if template == "single_choice_letter" or (
        granularity == "2_episode" and record.get("question_type") == "single_choice"
    ):
        value = text.upper()
        if not re.fullmatch(r"[A-Z]", value):
            return text, "expected one option letter"
        if value not in contract.option_label_set(record):
            return text, "option letter is outside the options"
        return value, None

    if contract.is_yes_no_question(record):
        value = text.lower()
        if value not in {"yes", "no"}:
            return text, 'expected exactly "Yes" or "No"'
        return value.title(), None

    if not text:
        return text, "empty answer"
    return text, None
