#!/usr/bin/env python3
"""Formal benchmark scoring.

Input is one merged prediction file produced by the inference runner: the
benchmark question items, each carrying `pred_answer` plus a `pred_info:
{model, reason, raw, error}` block. The pred file is read-only here; all
results go to --out:

1. single_choice / ranking: accuracy (ranking requires the exact gold
   sequence). Open Yes/No questions are also deterministic; the remaining
   open questions join via the LLM judge with a binary 0/1 verdict.
2. multi_select: option-set precision/recall/F1.
3. open emotion questions (`emotion_answer: true`): OV-MER-style GPT grouping
   with DeepSeek, then cluster-set precision/recall/F1; a transition question
   is split into from_emotion/to_emotion, contributing two emotion slots.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from benchmark.common import contract, io_utils  # noqa: E402
from benchmark.evaluation import emotion_grouping, open_judge  # noqa: E402


# ---------------------------------------------------------------------------
# set metrics (open emotion)
# ---------------------------------------------------------------------------

def prf_counts(gold: set[str], pred: set[str]) -> dict[str, Any]:
    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f}


def micro_prf(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    ps, rs, f1s = [], [], []
    for r in records:
        counts["tp"] += r.get("tp", 0)
        counts["fp"] += r.get("fp", 0)
        counts["fn"] += r.get("fn", 0)
        ps.append(float(r.get("precision", 0.0)))
        rs.append(float(r.get("recall", 0.0)))
        f1s.append(float(r.get("f1", 0.0)))
    p = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0
    r = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    n = len(records)
    return {
        "n": n,
        "micro": {"tp": counts["tp"], "fp": counts["fp"], "fn": counts["fn"], "precision": p, "recall": r, "f1": f},
        "macro": {
            "precision": sum(ps) / n if n else None,
            "recall": sum(rs) / n if n else None,
            "f1": sum(f1s) / n if n else None,
        },
    }


def score_emotion_records(slots: list[dict[str, Any]], grouping: dict[str, Any]) -> list[dict[str, Any]]:
    label_to_group = grouping.get("label_to_group") or {}
    records = []
    for s in slots:
        gold_groups = {str(label_to_group.get(contract.norm_label(x), f"unmapped_gold:{contract.norm_label(x)}")) for x in s["gold_labels"]}
        pred_groups = {str(label_to_group.get(contract.norm_label(x), f"unmapped_pred:{contract.norm_label(x)}")) for x in s["pred_labels"]}
        rec = {"category": "open_emotion", **s, "gold_groups": sorted(gold_groups), "pred_groups": sorted(pred_groups)}
        rec.update(prf_counts(gold_groups, pred_groups))
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# accuracy scoring (objective + judged open)
# ---------------------------------------------------------------------------

def score_multi_select(q: dict[str, Any]) -> dict[str, Any]:
    pred_answer = q.get("pred_answer")
    rec = {
        "category": "multi_select",
        "series": q.get("series"),
        "qid": q.get("qid"),
        "question_type": q.get("question_type"),
        "gold_answer": q.get("answer"),
        "pred_answer": pred_answer,
    }
    if "pred_answer" not in q:
        rec["score_method"] = "missing_prediction"
        return rec
    if (q.get("pred_info") or {}).get("error"):
        rec.update({"score_method": "prediction_error", "error": q["pred_info"]["error"]})
        return rec
    gold = set(q["answer"])
    got = set(contract.robust_labels_from(pred_answer))
    rec.update({"gold_labels": sorted(gold), "pred_labels": sorted(got), "score_method": "set_prf"})
    rec.update(prf_counts(gold, got))
    return rec


def score_accuracy(q: dict[str, Any], judge: Any | None) -> dict[str, Any]:
    pred_answer = q.get("pred_answer")
    qtype = q.get("question_type")
    rec = {
        "category": "accuracy",
        "series": q.get("series"),
        "qid": q.get("qid"),
        "question_type": qtype,
        "gold_answer": q.get("answer"),
        "pred_answer": pred_answer,
    }
    if "pred_answer" not in q:
        rec.update({"correct": None, "score_method": "missing_prediction"})
    elif (q.get("pred_info") or {}).get("error"):
        rec.update({"correct": None, "score_method": "prediction_error", "error": q["pred_info"]["error"]})
    elif qtype == "single_choice":
        labels = contract.robust_labels_from(pred_answer)
        rec.update({"correct": bool(labels) and labels[0] == q["answer"].strip().upper(), "score_method": "exact_label"})
    elif qtype == "ranking":
        got = contract.robust_labels_from(pred_answer)
        rec.update({"correct": bool(got) and got == list(q["answer"]), "score_method": "exact_ranking"})
    elif contract.is_yes_no_question(q):
        picked = contract.match_yes_no(pred_answer)
        rec.update({"correct": picked == q["answer"].strip().lower(), "score_method": "exact_yes_no"})
    elif judge is None:
        rec.update({"correct": None, "score_method": "needs_judge"})
    else:
        rec.update(open_judge.judge_open_answer(q, pred_answer, judge))
    return rec


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def accuracy_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in records if r.get("correct") is not None]
    correct = [r for r in scored if r.get("correct") is True]
    return {
        "n": len(records),
        "scored": len(scored),
        "unscored": len(records) - len(scored),
        "correct": len(correct),
        "accuracy": len(correct) / len(scored) if scored else None,
        "score_methods": dict(Counter(str(r.get("score_method")) for r in records)),
    }


def grouped_metrics(records: list[dict[str, Any]], field: str, kind: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        grouped[str(r.get(field) or "?")].append(r)
    agg = accuracy_metrics if kind == "accuracy" else micro_prf
    return {k: agg(v) for k, v in sorted(grouped.items())}


def write_summary(path: Path, metrics: dict[str, Any]) -> None:
    acc = metrics["accuracy"]
    multi = metrics["multi_select"]
    emo = metrics["open_emotion"]
    lines = [
        "# Formal Evaluation",
        "",
        "## Accuracy (single choice / ranking / judged open QA)",
        f"- items: {acc['n']} (scored: {acc['scored']}, unscored: {acc['unscored']})",
        f"- accuracy: {acc['accuracy']:.4f}" if acc["accuracy"] is not None else "- accuracy: n/a",
    ]
    for qtype, m in metrics["accuracy_by_question_type"].items():
        acc_txt = f"{m['accuracy']:.4f}" if m["accuracy"] is not None else "n/a"
        lines.append(f"  - {qtype}: {acc_txt} ({m['correct']}/{m['scored']})")
    lines += [
        "",
        "## Multi-Select (option-set P/R/F1)",
        f"- items: {multi['n']} (unscored: {multi['unscored']})",
        f"- micro P/R/F1: {multi['micro']['precision']:.4f} / {multi['micro']['recall']:.4f} / {multi['micro']['f1']:.4f}",
        f"- macro F1: {multi['macro']['f1']:.4f}" if multi["macro"]["f1"] is not None else "- macro F1: n/a",
        "",
        "## Open Emotion (OV-MER grouping)",
        f"- slots: {emo['n']}",
        f"- micro P/R/F1: {emo['micro']['precision']:.4f} / {emo['micro']['recall']:.4f} / {emo['micro']['f1']:.4f}",
        f"- macro F1: {emo['macro']['f1']:.4f}" if emo["macro"]["f1"] is not None else "- macro F1: n/a",
        "",
        "## Format",
        f"- format errors: {len(metrics.get('format_errors') or [])}",
        f"- prediction format issues: {len(metrics.get('prediction_format_issues') or [])}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------

def score_items(items: list[dict[str, Any]], judge: Any | None) -> dict[str, list[dict[str, Any]]]:
    """Route every item to its scoring category.

    Returns emotion slots (ungrouped), multi-select / accuracy records, and the
    gold/prediction format issue lists. Directly testable without the CLI.
    """
    emotion_slots: list[dict[str, Any]] = []
    multi_records: list[dict[str, Any]] = []
    accuracy_records: list[dict[str, Any]] = []
    format_errors: list[dict[str, Any]] = []
    prediction_format_issues: list[dict[str, Any]] = []

    for i, q in enumerate(items, 1):
        pred_answer = q.get("pred_answer")
        base = {"series": q.get("series"), "qid": io_utils.safe_qid(q, i), "question_type": q.get("question_type")}
        gold_errors = contract.validate_gold_format(q)
        if gold_errors:
            format_errors.extend(
                {**base, "category": "gold_format", "reason": reason, "gold_answer": q.get("answer")}
                for reason in gold_errors
            )
            continue
        if "pred_answer" in q and not (q.get("pred_info") or {}).get("error"):
            prediction_format_issues.extend(
                {**base, "category": "prediction_format", "reason": reason, "pred_answer": pred_answer}
                for reason in contract.validate_prediction_format(q, pred_answer)
            )
        key = io_utils.question_key(q, i)
        if contract.is_emotion_question(q):
            slots = emotion_grouping.emotion_slots(q, pred_answer)
            for slot in slots:
                slot["key"] = key
            emotion_slots.extend(slots)
        elif q.get("question_type") == "multi_select":
            multi_records.append({**score_multi_select(q), "key": key})
        else:
            accuracy_records.append({**score_accuracy(q, judge), "key": key})

    return {
        "emotion_slots": emotion_slots,
        "multi_records": multi_records,
        "accuracy_records": accuracy_records,
        "format_errors": format_errors,
        "prediction_format_issues": prediction_format_issues,
    }


def emotion_labels(slots: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    for slot in slots:
        labels.extend(slot.get("gold_labels") or [])
        labels.extend(slot.get("pred_labels") or [])
    return labels


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="Merged pred_<model>.json from the inference runner (questions + prediction blocks).")
    ap.add_argument("--out", required=True, help="Output directory for scores/metrics/summary.")
    ap.add_argument("--judge-model", default=None, help="Judge model (default deepseek-v4-flash; env JUDGE_MODEL).")
    ap.add_argument("--judge-base-url", default=None, help="OpenAI-compatible judge endpoint (default https://api.deepseek.com; env JUDGE_BASE_URL).")
    ap.add_argument("--judge-api-key", default=None, help="Judge API key (env JUDGE_API_KEY).")
    ap.add_argument("--judge-timeout", type=float, default=180)
    ap.add_argument("--judge-temperature", type=float, default=0.0)
    ap.add_argument("--judge-max-tokens", type=int, default=4096)
    ap.add_argument("--reuse-grouping", default=None, help="Optional existing gpt_grouping.json to reuse when it covers all current emotion labels.")
    args = ap.parse_args()
    out_dir = Path(args.out)

    items = io_utils.load_questions(args.pred)
    judge = open_judge.OpenAICompatibleJudge.from_env_or_args(args)

    res = score_items(items, judge)
    multi_records = res["multi_records"]
    accuracy_records = res["accuracy_records"]
    format_errors = res["format_errors"]
    prediction_format_issues = res["prediction_format_issues"]

    all_emotion_labels = emotion_labels(res["emotion_slots"])
    # reuse order: the pinned file (--reuse-grouping), then the out dir's own
    # gpt_grouping.json; delete that file to force a fresh grouping
    grouping = emotion_grouping.reusable_grouping(all_emotion_labels, args.reuse_grouping)
    if grouping is None:
        grouping = emotion_grouping.reusable_grouping(all_emotion_labels, out_dir / "gpt_grouping.json")
    if grouping is None:
        grouping = emotion_grouping.gpt_group_emotions(all_emotion_labels, judge)
    emotion_records = score_emotion_records(res["emotion_slots"], grouping)

    multi_scored = [r for r in multi_records if "f1" in r]
    metrics = {
        "evaluation_mode": "formal",
        "formal_benchmark": True,
        "total_questions": len(items),
        "accuracy": accuracy_metrics(accuracy_records),
        "multi_select": {**micro_prf(multi_scored), "unscored": len(multi_records) - len(multi_scored)},
        "open_emotion": micro_prf(emotion_records),
        "accuracy_by_question_type": grouped_metrics(accuracy_records, "question_type", "accuracy"),
        "accuracy_by_series": grouped_metrics(accuracy_records, "series", "accuracy"),
        "multi_select_by_series": grouped_metrics(multi_scored, "series", "prf"),
        "open_emotion_by_slot": grouped_metrics(emotion_records, "slot", "prf"),
        "open_emotion_by_series": grouped_metrics(emotion_records, "series", "prf"),
        "grouping": {
            "method": grouping.get("method"),
            "num_groups": len(grouping.get("groups") or []),
            "model": judge.model,
            "reused_from": grouping.get("reused_from"),
        },
        "format_errors": format_errors,
        "prediction_format_issues": prediction_format_issues,
    }

    io_utils.write_jsonl(out_dir / "scores.jsonl", [*emotion_records, *multi_records, *accuracy_records])
    io_utils.write_json(out_dir / "metrics.json", metrics)
    io_utils.write_json(out_dir / "gpt_grouping.json", grouping)
    io_utils.write_json(out_dir / "format_errors.json", format_errors)
    io_utils.write_json(out_dir / "prediction_format_issues.json", prediction_format_issues)
    write_summary(out_dir / "summary.md", metrics)

    print(f"wrote {out_dir}")
    if format_errors:
        raise SystemExit(f"format errors: {len(format_errors)}; see {out_dir / 'format_errors.json'}")


if __name__ == "__main__":
    main()
