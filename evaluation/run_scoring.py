#!/usr/bin/env python3
"""Score canonical benchmark predictions.

Closed outputs are scored deterministically. Only non-Yes/No open answers use
the semantic judge. Prompt-template routing applies to ``1_clip``;
``2_episode`` is routed directly by question type.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any, Iterable

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from evaluation import contract, io_utils  # noqa: E402
from evaluation import emotion_grouping, open_judge  # noqa: E402


def prf_counts(gold: Iterable[str], pred: Iterable[str]) -> dict[str, Any]:
    gold_set = {contract.norm_label(value) for value in gold if contract.norm_label(value)}
    pred_set = {contract.norm_label(value) for value in pred if contract.norm_label(value)}
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if tp + fp else (1.0 if not gold_set else 0.0)
    recall = tp / (tp + fn) if tp + fn else (1.0 if not pred_set else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": gold_set == pred_set,
    }


def micro_prf(records: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [record for record in records if record.get("scored", True)]
    tp = sum(int(record.get("tp", 0)) for record in scored)
    fp = sum(int(record.get("fp", 0)) for record in scored)
    fn = sum(int(record.get("fn", 0)) for record in scored)
    denom = len(scored)
    if denom:
        precision = tp / (tp + fp) if tp + fp else (1.0 if not fn else 0.0)
        recall = tp / (tp + fn) if tp + fn else (1.0 if not fp else 0.0)
        micro_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    else:
        precision = recall = micro_f1 = None
    return {
        "n": len(records),
        "scored": denom,
        "micro": {"precision": precision, "recall": recall, "f1": micro_f1, "tp": tp, "fp": fp, "fn": fn},
        "macro": {
            "precision": sum(float(record.get("precision", 0.0)) for record in scored) / denom if denom else None,
            "recall": sum(float(record.get("recall", 0.0)) for record in scored) / denom if denom else None,
            "f1": sum(float(record.get("f1", 0.0)) for record in scored) / denom if denom else None,
        },
        "exact_match": (
            sum(bool(record.get("exact_match")) for record in scored) / denom if denom else None
        ),
    }


def score_closed_emotion_records(slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for slot in slots:
        record = dict(slot)
        if not slot.get("scored", True):
            record.update(
                {
                    "category": "closed_emotion",
                    "score_method": slot.get("score_method") or "missing_prediction",
                    "scored": False,
                }
            )
            records.append(record)
            continue
        record.update(prf_counts(slot.get("gold_labels") or [], slot.get("pred_labels") or []))
        record.update({"category": "closed_emotion", "score_method": "exact_label_set", "scored": True})
        records.append(record)
    return records


def _base_record(q: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": io_utils.question_key(q),
        "series": q.get("series"),
        "qid": q.get("qid"),
        "granularity": contract.question_granularity(q),
        "task_type": q.get("task_type"),
        "question_type": q.get("question_type"),
        "prompt_template": q.get("prompt_template"),
        "gold_answer": q.get("answer"),
        "pred_answer": q.get("pred_answer"),
    }


def score_accuracy(q: dict[str, Any], judge: Any | None) -> dict[str, Any]:
    """Score one non-emotion item, leaving unavailable predictions unscored."""
    pred = q.get("pred_answer")
    pred_info = q.get("pred_info") or {}
    if pred_info.get("error"):
        return {"correct": None, "scored": False, "score_method": "prediction_error"}
    if pred is None:
        return {"correct": None, "scored": False, "score_method": "missing_prediction"}

    template = q.get("prompt_template")
    qtype = q.get("question_type")
    if qtype == "single_choice":
        correct = pred == q.get("answer")
        return {"correct": correct, "scored": True, "score_method": "exact_single_choice"}
    if template == "ranking_letters" or qtype == "ranking":
        return {
            "correct": pred == q.get("answer"),
            "scored": True,
            "score_method": "exact_ranking",
        }
    if contract.is_yes_no_question(q):
        return {"correct": pred == q.get("answer"), "scored": True, "score_method": "exact_yes_no"}
    if judge is None:
        return {"correct": None, "scored": False, "score_method": "needs_judge"}
    judged = open_judge.judge_open_answer(q, pred, judge)
    judged["scored"] = judged.get("correct") is not None
    return judged


def accuracy_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [record for record in records if record.get("scored")]
    correct = sum(record.get("correct") is True for record in scored)
    methods = collections.Counter(str(record.get("score_method")) for record in records)
    return {
        "n": len(records),
        "scored": len(scored),
        "correct": correct,
        "accuracy": correct / len(scored) if scored else None,
        "score_methods": dict(sorted(methods.items())),
    }


def grouped_metrics(records: list[dict[str, Any]], field: str, metric: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        groups[str(record.get(field) or "unknown")].append(record)
    aggregate = accuracy_metrics if metric == "accuracy" else micro_prf
    return {name: aggregate(group) for name, group in sorted(groups.items())}


def question_score_records(
    items: list[dict[str, Any]],
    accuracy_records: list[dict[str, Any]],
    emotion_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize every original question to one score in the [0, 1] range."""
    accuracy_by_key = {record["key"]: record for record in accuracy_records}
    emotion_by_key: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in emotion_records:
        emotion_by_key[record["key"]].append(record)

    records: list[dict[str, Any]] = []
    for q in items:
        base = _base_record(q)
        key = base["key"]
        if contract.is_emotion_question(q):
            slots = emotion_by_key.get(key, [])
            component_scores = {
                str(slot.get("slot")): float(slot.get("f1", 0.0)) if slot.get("scored") else 0.0
                for slot in slots
            }
            score = sum(component_scores.values()) / len(component_scores) if component_scores else 0.0
            prediction_available = bool(slots) and all(slot.get("scored") for slot in slots)
            method = "mean_before_after_f1" if contract.is_transition_question(q) else "set_f1"
            records.append(
                {
                    **base,
                    "score": score,
                    "score_method": method,
                    "component_scores": component_scores,
                    "prediction_available": prediction_available,
                }
            )
            continue

        record = accuracy_by_key.get(key, {})
        records.append(
            {
                **base,
                "score": 1.0 if record.get("correct") is True else 0.0,
                "score_method": record.get("score_method") or "missing_score_record",
                "component_scores": {},
                "prediction_available": bool(record.get("scored")),
            }
        )
    return records


def question_score_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Average normalized question scores, including unavailable predictions as zero."""
    count = len(records)
    available = sum(bool(record.get("prediction_available")) for record in records)
    earned_points = sum(float(record.get("score", 0.0)) for record in records)
    return {
        "n": count,
        "prediction_available": available,
        "coverage": available / count if count else None,
        "earned_points": earned_points,
        "possible_points": count,
        "score": earned_points / count if count else None,
    }


def hierarchical_score_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Macro-average questions within tasks, tasks within granularities, then granularities."""
    by_granularity: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        by_granularity[str(record.get("granularity") or "unknown")].append(record)

    granularities: dict[str, Any] = {}
    for granularity, granularity_records in sorted(by_granularity.items()):
        by_task: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for record in granularity_records:
            by_task[str(record.get("task_type") or "unknown")].append(record)
        task_scores = {
            task: question_score_metrics(task_records)
            for task, task_records in sorted(by_task.items())
        }
        granularity_summary = question_score_metrics(granularity_records)
        granularity_summary["score"] = sum(
            float(task["score"]) for task in task_scores.values()
        ) / len(task_scores)
        granularity_summary["task_scores"] = task_scores
        granularities[granularity] = granularity_summary

    macro_score = (
        sum(float(granularity["score"]) for granularity in granularities.values()) / len(granularities)
        if granularities
        else None
    )
    return {
        "macro_score": macro_score,
        "aggregation": "question_macro_then_task_macro_then_granularity_macro",
        "granularity_scores": granularities,
    }


def grouped_question_scores(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        groups[str(record.get(field) or "unknown")].append(record)
    return {
        name: question_score_metrics(group)
        for name, group in sorted(groups.items())
    }


def score_items(items: list[dict[str, Any]], judge: Any | None) -> dict[str, Any]:
    accuracy_records: list[dict[str, Any]] = []
    emotion_slots: list[dict[str, Any]] = []
    prediction_format_issues: list[dict[str, Any]] = []

    for q in items:
        base = _base_record(q)
        if q.get("pred_answer") is not None:
            pred_errors = contract.validate_prediction_format(q, q.get("pred_answer"))
            parse_error = (q.get("pred_info") or {}).get("parse_error")
            if parse_error:
                pred_errors = [*pred_errors, str(parse_error)]
            if pred_errors:
                prediction_format_issues.append({**base, "errors": list(dict.fromkeys(pred_errors))})

        if contract.is_emotion_question(q):
            prediction_error = (q.get("pred_info") or {}).get("error")
            if prediction_error:
                scoring_state = {"scored": False, "score_method": "prediction_error"}
            elif q.get("pred_answer") is None:
                scoring_state = {"scored": False, "score_method": "missing_prediction"}
            else:
                scoring_state = {"scored": True, "score_method": "exact_label_set"}
            for slot in emotion_grouping.emotion_slots(q, q.get("pred_answer")):
                slot.update({key: base[key] for key in ("key", "granularity", "task_type", "prompt_template")})
                slot.update(scoring_state)
                emotion_slots.append(slot)
            continue
        accuracy_records.append({**base, **score_accuracy(q, judge)})

    closed_emotion_records = score_closed_emotion_records(emotion_slots)
    result = {
        "accuracy_records": accuracy_records,
        "emotion_slots": emotion_slots,
        "closed_emotion_records": closed_emotion_records,
        "prediction_format_issues": prediction_format_issues,
    }
    result["question_score_records"] = question_score_records(
        items, accuracy_records, closed_emotion_records
    )
    return result


def requires_judge(items: list[dict[str, Any]]) -> bool:
    """Return whether any predicted open item needs semantic judging."""
    for q in items:
        if contract.is_emotion_question(q):
            continue
        if q.get("pred_answer") is None or (q.get("pred_info") or {}).get("error"):
            continue
        if q.get("question_type") == "open_ended" and not contract.is_yes_no_question(q):
            return True
    return False


def _metrics(items: list[dict[str, Any]], result: dict[str, Any]) -> dict[str, Any]:
    accuracy_records = result["accuracy_records"]
    emotion_records = result["closed_emotion_records"]
    question_records = result["question_score_records"]
    hierarchy = hierarchical_score_metrics(question_records)
    question_summary = question_score_metrics(question_records)
    return {
        "evaluation_mode": "formal",
        "formal_benchmark": True,
        "total_questions": len(items),
        "overall_score": question_summary["score"],
        "earned_points": question_summary["earned_points"],
        "possible_points": question_summary["possible_points"],
        "aggregation": "sum_question_scores_over_total_questions",
        "macro_score": hierarchy["macro_score"],
        "macro_aggregation": hierarchy["aggregation"],
        "granularity_scores": hierarchy["granularity_scores"],
        "series_scores": grouped_question_scores(question_records, "series"),
        "task_type_scores": grouped_question_scores(question_records, "task_type"),
        "question_scores": question_summary,
        "accuracy": accuracy_metrics(accuracy_records),
        "closed_emotion": micro_prf(emotion_records),
        "accuracy_by_question_type": grouped_metrics(accuracy_records, "question_type", "accuracy"),
        "accuracy_by_series": grouped_metrics(accuracy_records, "series", "accuracy"),
        "accuracy_by_granularity": grouped_metrics(accuracy_records, "granularity", "accuracy"),
        "accuracy_by_task_type": grouped_metrics(accuracy_records, "task_type", "accuracy"),
        "closed_emotion_by_slot": grouped_metrics(emotion_records, "slot", "prf"),
        "closed_emotion_by_series": grouped_metrics(emotion_records, "series", "prf"),
        "prediction_format_issue_count": len(result["prediction_format_issues"]),
    }


def write_summary(path: str | Path, metrics: dict[str, Any]) -> None:
    acc = metrics["accuracy"]
    emotion = metrics["closed_emotion"]
    lines = [
        "# Evaluation Summary",
        "",
        f"- Questions: {metrics['total_questions']}",
        f"- Overall score: {metrics['overall_score']}",
        f"- Total points: {metrics['earned_points']}/{metrics['possible_points']}",
        f"- Task-macro score: {metrics['macro_score']}",
        f"- Prediction coverage: {metrics['question_scores']['coverage']}",
        f"- Accuracy: {acc['accuracy']} ({acc['correct']}/{acc['scored']} scored)",
        f"- Closed-emotion micro F1: {emotion['micro']['f1']}",
        f"- Closed-emotion exact match: {emotion['exact_match']}",
        f"- Prediction format issues: {metrics['prediction_format_issue_count']}",
    ]

    def add_score_table(title: str, groups: dict[str, Any]) -> None:
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Category | Questions | Points | Score | Coverage |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for name, values in groups.items():
            lines.append(
                f"| {name} | {values['n']} | {values['earned_points']:.2f}/"
                f"{values['possible_points']} | {values['score']:.2%} | "
                f"{values['coverage']:.2%} |"
            )

    add_score_table("Scores by task type", metrics["task_type_scores"])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", required=True, help="Prediction JSON file")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--model")
    ap.add_argument("--base-url")
    ap.add_argument("--api-key")
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--thinking", choices=("on", "off", "default"), default="default")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    items = io_utils.load_questions(args.pred)
    judge = None
    if requires_judge(items):
        try:
            judge = open_judge.OpenAICompatibleJudge.from_args(args)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    result = score_items(items, judge)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    io_utils.write_jsonl(out / "scores.jsonl", result["accuracy_records"])
    io_utils.write_jsonl(out / "scores_closed_emotion.jsonl", result["closed_emotion_records"])
    io_utils.write_jsonl(out / "scores_questions.jsonl", result["question_score_records"])
    io_utils.write_json(out / "prediction_format_issues.json", result["prediction_format_issues"])
    metrics = _metrics(items, result)
    if judge is not None:
        metrics["judge"] = {"model": args.model, "thinking": args.thinking}
    io_utils.write_json(out / "metrics.json", metrics)
    write_summary(out / "summary.md", metrics)
    print(json.dumps({"out": str(out), **metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
