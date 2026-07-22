#!/usr/bin/env python3
"""Answer-format contract for the final benchmark dataset.

Question schema (code/datasets/*/grade_1/*.json and the unified benchmark file):

    series, qid, question_type, question, options, answer,
    emotion_answer (true, only on open emotion questions),
    input_video, input_transcript

question_type is one of single_choice / multi_select / ranking / open_ended.
Gold answers:

- single_choice: one option letter, e.g. "B"
- multi_select: list of option letters, e.g. ["A", "C"]
- ranking: ordered list of option letters (an option may repeat)
- open_ended + emotion_answer: {"emotion": [...]} or
  {"from_emotion": [...], "to_emotion": [...]} (a transition is scored as two slots)
- open_ended (non-emotion): a short free-text string, judged by the LLM judge

Shared by the formal scorer (benchmark/evaluation/run_scoring.py) and the
dataset checker (benchmark/preprocess/check_question_contract.py).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from benchmark.common import io_utils  # noqa: E402


QUESTION_TYPES = {"single_choice", "multi_select", "ranking", "open_ended"}
CHOICE_TYPES = {"single_choice", "multi_select", "ranking"}
TRANSITION_FIELDS = ("from_emotion", "to_emotion")

# Canonical question tails; every finalized question ends with exactly one of
# these (verified against the full dataset), so the checker requires them
# verbatim instead of accepting paraphrases.
SINGLE_CHOICE_TAIL = "Please select only one option. Respond with the option letter only."
MULTI_SELECT_TAIL = 'Please select all correct options. Respond with option letters only, such as "B D".'
RANKING_TAIL = 'Respond with option letters only, such as "A B C".'
RANKING_DISTRACTOR_NOTE = "Some options may be distractors."
EMOTION_TAIL = 'Answer with a JSON object like {"emotion":["..."]}.'
TRANSITION_TAIL = 'Answer with a JSON object like {"from_emotion":["..."],"to_emotion":["..."]}.'


# ---------------------------------------------------------------------------
# label / text parsing
# ---------------------------------------------------------------------------

def norm_label(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    parsed = io_utils.parse_jsonish(value)
    return parsed if parsed is not None else value


def flatten_values(value: Any) -> list[Any]:
    value = parse_jsonish(value)
    if value is None:
        return []
    if isinstance(value, list):
        out: list[Any] = []
        for x in value:
            out.extend(flatten_values(x))
        return out
    if isinstance(value, dict):
        for key in ("answer", "emotion", *TRANSITION_FIELDS):
            if key in value:
                return flatten_values(value[key])
        return []
    return [value]


def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(flatten_text(v) for v in value.values())
    if isinstance(value, list):
        return " ".join(flatten_text(v) for v in value)
    return str(value)


def parse_transition(value: Any) -> tuple[str, str] | None:
    """Parse a free-text emotion transition like "from anxious to hurt" into (from, to)."""
    text = flatten_text(value)
    patterns = [
        r"from\s+(.+?)\s+to\s+(.+)$",
        r"从\s*(.+?)\s*(?:到|至|转为|变成|变到)\s*(.+)$",
        r"由\s*(.+?)\s*(?:到|至|转为|变成|变到)\s*(.+)$",
        r"(.+?)\s*(?:->|→)\s*(.+)$",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            return m.group(1).strip(" ，,.;；。"), m.group(2).strip(" ，,.;；。")
    return None


def emotion_terms(value: Any) -> list[str]:
    """Split a gold or predicted emotion expression into deduplicated short terms."""
    out: list[str] = []
    for raw in flatten_values(value):
        text = str(raw or "").strip()
        if not text:
            continue
        transition = parse_transition(text)
        if transition:
            parts = [transition[0], transition[1]]
        else:
            parts = re.split(r"[,，;；\n/]+", text)
            if len(parts) == 1 and re.search(r"\s+and\s+", text, re.I):
                parts = re.split(r"\s+and\s+", text, flags=re.I)
        for part in parts:
            term = part.strip(" \t\r\n\"'`.,;:，；。")
            term = re.sub(r"^(emotion|feeling|answer)\s*[:：]\s*", "", term, flags=re.I).strip()
            if term:
                out.append(term)
    seen = set()
    uniq = []
    for term in out:
        key = norm_label(term)
        if key not in seen:
            seen.add(key)
            uniq.append(term)
    return uniq


def robust_labels_from(value: Any) -> list[str]:
    """Extract option letters from a prediction, tolerating separators and wrappers."""
    if isinstance(value, list):
        out: list[str] = []
        for x in value:
            out.extend(robust_labels_from(x))
        return out
    if isinstance(value, dict):
        return robust_labels_from(value.get("answer"))
    text = str(value or "").strip()
    if not text:
        return []
    cleaned = re.sub(r"[\s,，;；/|>\-→]+", "", text).upper()
    if re.fullmatch(r"[A-Z]+", cleaned):
        return list(cleaned)
    return [m.group(1).upper() for m in re.finditer(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])", text)]


def option_label_set(q: dict[str, Any]) -> set[str]:
    labels = set()
    for i, opt in enumerate(q.get("options") or []):
        label = chr(ord("A") + i)
        m = re.match(r"^\s*([A-Z])\s*[\.\)]", str(opt))
        if m:
            label = m.group(1).upper()
        labels.add(label)
    return labels


def letter_list(value: Any) -> list[str]:
    """value as a list of single option letters, else []."""
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if not isinstance(item, str) or not re.fullmatch(r"[A-Z]", item.strip().upper()):
            return []
        out.append(item.strip().upper())
    return out


# ---------------------------------------------------------------------------
# question classification
# ---------------------------------------------------------------------------

def is_emotion_question(q: dict[str, Any]) -> bool:
    """True when the answer itself is an open emotion label set / transition.

    emotion_answer: true is the single routing signal; it only appears on
    open_ended emotion questions in the final dataset.
    """
    return q.get("question_type") == "open_ended" and q.get("emotion_answer") is True


def is_transition_question(q: dict[str, Any]) -> bool:
    answer = q.get("answer")
    return is_emotion_question(q) and isinstance(answer, dict) and any(f in answer for f in TRANSITION_FIELDS)


YES_NO_TAIL = "Answer Yes or No."


def is_yes_no_question(q: dict[str, Any]) -> bool:
    """Open questions ending with the canonical Yes/No tail: scored
    deterministically, never sent to the LLM judge."""
    return (
        q.get("question_type") == "open_ended"
        and not is_emotion_question(q)
        and _question_text(q).endswith(YES_NO_TAIL)
    )


def match_yes_no(pred_answer: Any) -> str | None:
    """"yes"/"no" when the prediction commits to exactly one side, else None
    (hedged, both-ways, or unparseable answers commit to nothing)."""
    text = norm_label(flatten_text(pred_answer))
    hits = {m.group(1) for m in re.finditer(r"\b(yes|no)\b", text)}
    return hits.pop() if len(hits) == 1 else None


# ---------------------------------------------------------------------------
# gold-answer format contract
# ---------------------------------------------------------------------------

def _question_text(q: dict[str, Any]) -> str:
    return re.sub(r"\s+", " ", str(q.get("question") or "").strip())


def _nonempty_str_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(str(v).strip() for v in value)


def validate_gold_format(q: dict[str, Any]) -> list[str]:
    """Strict per-question contract checks used by the formal evaluator.

    Finalized files must pass with zero errors; the scorer refuses to conclude
    a run that contains malformed gold answers.
    """
    errors: list[str] = []
    qtype = q.get("question_type")
    answer = q.get("answer")
    opts = q.get("options")
    labels = option_label_set(q)
    text = _question_text(q)

    if qtype not in QUESTION_TYPES:
        # never assume a default type: an unknown/missing type must land in
        # format_errors.json for review, not be scored as some other type
        errors.append(f"question_type must be one of {sorted(QUESTION_TYPES)}, got {qtype!r}")
        return errors
    if "emotion_answer" in q and (qtype != "open_ended" or q.get("emotion_answer") is not True):
        errors.append("emotion_answer may only appear on open_ended questions with value true")

    if qtype in CHOICE_TYPES:
        if not isinstance(opts, list) or len(opts) < 3:
            errors.append("choice questions must have at least three options")
        elif any(not isinstance(o, str) or not o.strip() for o in opts):
            errors.append("choice options must be non-empty strings")
        elif len(labels) != len(opts):
            errors.append("choice option labels must be unique")
    elif opts:
        errors.append("open questions must not carry options")

    if qtype == "single_choice":
        if not text.endswith(SINGLE_CHOICE_TAIL):
            errors.append(f"single-choice question must end with the canonical tail: {SINGLE_CHOICE_TAIL}")
        if not (isinstance(answer, str) and re.fullmatch(r"[A-Z]", answer.strip().upper())):
            errors.append("single-choice gold answer must be a single option letter")
        elif labels and answer.strip().upper() not in labels:
            errors.append("single-choice gold answer letter is outside the options")

    elif qtype == "multi_select":
        if not text.endswith(MULTI_SELECT_TAIL):
            errors.append(f"multi-select question must end with the canonical tail: {MULTI_SELECT_TAIL}")
        got = letter_list(answer)
        if not got:
            errors.append("multi-select gold answer must be a non-empty JSON list of option letters")
        elif len(set(got)) != len(got):
            errors.append("multi-select gold answer contains duplicate option letters")
        elif labels and not set(got).issubset(labels):
            errors.append("multi-select gold answer contains letters outside options")

    elif qtype == "ranking":
        if not text.endswith(RANKING_TAIL):
            errors.append(f"ranking question must end with the canonical tail: {RANKING_TAIL}")
        if RANKING_DISTRACTOR_NOTE.lower() not in text.lower():
            errors.append(f"ranking question must include the distractor note: {RANKING_DISTRACTOR_NOTE}")
        got = letter_list(answer)
        # the stem says "An option may be used more than once", so repeats are legal
        if len(got) < 2:
            errors.append("ranking gold answer must be a JSON list of at least two ordered option letters")
        elif labels and not set(got).issubset(labels):
            errors.append("ranking gold answer contains letters outside options")

    elif is_emotion_question(q):
        if not isinstance(answer, dict):
            errors.append("open emotion gold answer must be an object")
        elif set(answer) == {"emotion"}:
            if not text.endswith(EMOTION_TAIL):
                errors.append(f"emotion question must end with the canonical tail: {EMOTION_TAIL}")
            if not _nonempty_str_list(answer["emotion"]):
                errors.append("answer.emotion must be a non-empty list of emotion words")
        elif set(answer) == set(TRANSITION_FIELDS):
            if not text.endswith(TRANSITION_TAIL):
                errors.append(f"transition question must end with the canonical tail: {TRANSITION_TAIL}")
            for field in TRANSITION_FIELDS:
                if not _nonempty_str_list(answer[field]):
                    errors.append(f"answer.{field} must be a non-empty list of emotion words")
        else:
            errors.append(
                'open emotion gold answer must be exactly {"emotion": [...]} or '
                '{"from_emotion": [...], "to_emotion": [...]}'
            )

    else:  # open_ended, non-emotion
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s]
        if not sentences or not sentences[-1].lower().startswith("answer"):
            errors.append("open question must end with an Answer-format instruction sentence")
        if not isinstance(answer, str) or not answer.strip():
            errors.append("open non-emotion gold answer must be a non-empty string")
        elif is_yes_no_question(q) and answer.strip().lower() not in ("yes", "no"):
            errors.append('Yes/No gold answer must be exactly "Yes" or "No"')

    return errors


# ---------------------------------------------------------------------------
# prediction format contract
# ---------------------------------------------------------------------------

def validate_prediction_format(q: dict[str, Any], pred_answer: Any) -> list[str]:
    """Report model answer-format mismatches without changing the robust score.

    The formal metrics still parse common variants so model capability is not
    hidden by superficial formatting, but these issues make prompt/decoder
    compliance visible in the report.
    """
    issues: list[str] = []
    if pred_answer is None:
        return issues
    qtype = q.get("question_type")
    labels = option_label_set(q)

    if qtype == "single_choice":
        if not isinstance(pred_answer, str) or not re.fullmatch(r"\s*[A-Z]\s*", pred_answer):
            issues.append("single-choice prediction should be one option label string")
        elif labels and pred_answer.strip().upper() not in labels:
            issues.append("single-choice prediction label is outside options")

    elif qtype == "multi_select":
        parsed = robust_labels_from(pred_answer)
        if not isinstance(pred_answer, str) or not parsed:
            issues.append("multi-select prediction should be one space-separated option-label string")
        if parsed and len(set(parsed)) != len(parsed):
            issues.append("multi-select prediction contains duplicate option labels")
        if labels and parsed and not set(parsed).issubset(labels):
            issues.append("multi-select prediction contains labels outside options")

    elif qtype == "ranking":
        parsed = robust_labels_from(pred_answer)
        if not isinstance(pred_answer, str) or not parsed:
            issues.append("ranking prediction should be one space-separated ordered option-label string")
        # repeats are legal in ranking answers ("An option may be used more than once")
        if labels and parsed and not set(parsed).issubset(labels):
            issues.append("ranking prediction contains labels outside options")

    elif qtype == "open_ended" and is_yes_no_question(q):
        if not (isinstance(pred_answer, str) and pred_answer.strip().lower() in ("yes", "no")):
            issues.append('Yes/No prediction should be exactly "Yes" or "No"')

    elif is_emotion_question(q):
        fields = TRANSITION_FIELDS if is_transition_question(q) else ("emotion",)
        parsed = parse_jsonish(pred_answer)
        ok = isinstance(parsed, dict) and all(_nonempty_str_list(parsed.get(f)) for f in fields)
        if not ok:
            shape = ", ".join(f'"{f}": [...]' for f in fields)
            issues.append(f"emotion prediction should be a JSON object with {{{shape}}}")

    return issues
