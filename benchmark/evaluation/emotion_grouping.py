#!/usr/bin/env python3
"""OV-MER-style open-emotion scoring support: slot extraction + GPT-based grouping.

A transition question contributes two slots (from_emotion / to_emotion), so it is scored as
two emotion tasks; a direct emotion question contributes one "emotion" slot. Grouping follows
the OV-MER GPT-based strategy, implemented with the same DeepSeek judge model used for
open-answer accuracy.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from benchmark.common import contract, io_utils  # noqa: E402


# ---------------------------------------------------------------------------
# slot extraction
# ---------------------------------------------------------------------------

def extract_gold_slots(q: dict[str, Any]) -> dict[str, Any]:
    """Return slot -> gold expression for open questions answered with emotion labels.

    Gold answers are objects with exactly {"emotion":[...]} or
    {"from_emotion":[...],"to_emotion":[...]} (enforced by contract.validate_gold_format);
    anything else yields no slots and is reported as a format error by the scorer.
    """
    if not contract.is_emotion_question(q):
        return {}
    ans = q.get("answer")
    if not isinstance(ans, dict):
        return {}
    return {key: ans[key] for key in ("emotion", *contract.TRANSITION_FIELDS) if ans.get(key)}


def extract_pred_slots(q: dict[str, Any], pred_answer: Any, gold_slots: dict[str, Any]) -> dict[str, Any]:
    parsed_json = io_utils.parse_jsonish(pred_answer) if isinstance(pred_answer, str) else pred_answer
    from_slot, to_slot = contract.TRANSITION_FIELDS
    if from_slot in gold_slots or to_slot in gold_slots:
        if isinstance(parsed_json, dict) and (parsed_json.get(from_slot) or parsed_json.get(to_slot)):
            return {
                from_slot: parsed_json.get(from_slot, pred_answer),
                to_slot: parsed_json.get(to_slot, pred_answer),
            }
        parsed = contract.parse_transition(pred_answer)
        if parsed:
            return {from_slot: parsed[0], to_slot: parsed[1]}
        # Fallback: score the full predicted emotion set against each slot.
        return {slot: pred_answer for slot in gold_slots}
    if isinstance(parsed_json, dict) and parsed_json.get("emotion"):
        return {"emotion": parsed_json["emotion"]}
    return {slot: pred_answer for slot in gold_slots}


def emotion_slots(q: dict[str, Any], pred_answer: Any) -> list[dict[str, Any]]:
    """Expand one open-emotion question into scoring slots with parsed gold/pred labels."""
    gold_slots = extract_gold_slots(q)
    if not gold_slots:
        return []
    pred_slots = extract_pred_slots(q, pred_answer, gold_slots)
    slots = []
    for slot, gold_value in gold_slots.items():
        pred_value = pred_slots.get(slot, pred_answer)
        slots.append(
            {
                "series": q.get("series"),
                "qid": q.get("qid"),
                "slot": slot,
                "question_type": q.get("question_type"),
                "gold_answer": q.get("answer"),
                "pred_answer": pred_answer,
                "gold_labels": contract.emotion_terms(gold_value),
                "pred_labels": contract.emotion_terms(pred_value),
            }
        )
    return slots


# ---------------------------------------------------------------------------
# GPT-based grouping
# ---------------------------------------------------------------------------

def gpt_group_emotions(labels: list[str], judge: Any | None) -> dict[str, Any]:
    """Group all gold+predicted emotion labels by semantic equivalence (OV-MER GPT grouping)."""
    normalized = []
    seen = set()
    for label in labels:
        key = contract.norm_label(label)
        if key and key not in seen:
            seen.add(key)
            normalized.append(label.strip())
    if not normalized:
        return {"groups": [], "label_to_group": {}, "raw": None, "method": "none"}
    if judge is None:
        groups = [[x] for x in normalized]
        return {
            "groups": groups,
            "label_to_group": {contract.norm_label(x): i for i, x in enumerate(normalized)},
            "raw": None,
            "method": "exact_no_llm",
        }

    # OV-MER's GPT-based grouping prompt (arXiv:2410.01495), with the output
    # pinned to a parseable JSON object instead of a bare list of lists
    prompt = (
        "Please assume the role of an expert in the field of emotions. "
        "We provide a set of emotion labels. Please group the emotions, with each group "
        "containing emotions with the same meaning. Use every label exactly once. "
        "Directly output the results. Do not include any explanation. "
        'The output must be JSON in the format {"groups":[["label A","label B"],["label C"]]}.\n'
        f"Labels: {json.dumps(normalized, ensure_ascii=False)}"
    )
    raw = ""
    parsed = None
    finish_reason = None
    reasoning_excerpt = None
    last_error = None
    # DeepSeek reasoning models (deepseek-v4-flash) spend the whole budget in hidden reasoning
    # before emitting the final JSON: with a small cap they hit finish_reason=length with empty
    # content, which silently degrades to exact-match grouping (every label its own group -> all
    # open-emotion F1 = 0). Give reasoning + JSON generous room and escalate hard on truncation.
    for attempt in range(3):
        try:
            resp = judge.client.chat.completions.create(
                model=judge.model,
                temperature=judge.temperature,
                max_tokens=max(8000 * (attempt + 1), judge.max_tokens),
                messages=[{"role": "user", "content": prompt}],
            )
            choice = resp.choices[0]
            finish_reason = choice.finish_reason
            msg = choice.message
            raw = msg.content or ""
            reasoning_content = getattr(msg, "reasoning_content", None) or ""
            reasoning_excerpt = reasoning_content[:1200] or reasoning_excerpt
            parsed = io_utils.parse_jsonish(raw)
            # Salvage: if content is empty/non-JSON but the reasoning trace already contains the
            # final {"groups":[...]} object, parse it out of there rather than failing to exact-match.
            if not (isinstance(parsed, dict) and isinstance(parsed.get("groups"), list)) and reasoning_content:
                salvaged = io_utils.parse_jsonish(reasoning_content)
                if isinstance(salvaged, dict) and isinstance(salvaged.get("groups"), list):
                    parsed, raw = salvaged, raw or reasoning_content
            if isinstance(parsed, dict) and isinstance(parsed.get("groups"), list):
                break
            last_error = f"empty_or_non_json_grouping_response finish_reason={finish_reason}"
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
        time.sleep(1 * (attempt + 1))
    groups_raw = parsed.get("groups") if isinstance(parsed, dict) else None
    groups: list[list[str]] = []
    assigned = set()
    if isinstance(groups_raw, list):
        for group in groups_raw:
            if not isinstance(group, list):
                continue
            clean_group = []
            for item in group:
                key = contract.norm_label(item)
                if key in seen and key not in assigned:
                    clean_group.append(str(item).strip())
                    assigned.add(key)
            if clean_group:
                groups.append(clean_group)
    for label in normalized:
        key = contract.norm_label(label)
        if key not in assigned:
            groups.append([label])
            assigned.add(key)
    label_to_group: dict[str, int] = {}
    for i, group in enumerate(groups):
        for label in group:
            label_to_group[contract.norm_label(label)] = i
    method = "deepseek_gpt_grouping" if not last_error or groups_raw else "deepseek_grouping_fallback_exact"
    return {
        "groups": groups,
        "label_to_group": label_to_group,
        "raw": raw,
        "method": method,
        "finish_reason": finish_reason,
        "error": None if groups_raw else last_error,
        "reasoning_excerpt": reasoning_excerpt,
    }


def reusable_grouping(labels: list[str], path: str | Path | None) -> dict[str, Any] | None:
    """Load an existing grouping file if it covers every current label; else None."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    grouping = io_utils.read_json(p)
    if not isinstance(grouping, dict):
        return None
    label_to_group = grouping.get("label_to_group") or {}
    if not isinstance(label_to_group, dict):
        return None
    required = {contract.norm_label(x) for x in labels if contract.norm_label(x)}
    missing = sorted(x for x in required if x not in label_to_group)
    if missing:
        return None
    out = dict(grouping)
    method = str(out.get("method") or "reused_grouping")
    if not method.endswith("_reused"):
        out["method"] = f"{method}_reused"
    out["reused_from"] = str(p)
    return out
