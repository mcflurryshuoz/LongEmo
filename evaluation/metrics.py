"""Per-question label metrics, rubric scores and G2 reasoning aggregates."""

from __future__ import annotations
from collections import Counter, defaultdict
from .io_utils import norm_label


def prf(gold, prediction):
    gold, prediction = (
        {norm_label(x) for x in gold},
        {norm_label(x) for x in prediction},
    )
    correct = len(gold & prediction)
    precision = correct / len(prediction) if prediction else 0.0
    recall = correct / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "em": float(gold == prediction),
    }


def mean(values):
    return sum(values) / len(values) if values else None


def label_score(q, prediction):
    if q["type"] == "emotion transition":
        slots = {
            slot: prf(q["answer"][slot], prediction[slot])
            for slot in ("before", "after")
        }
        metrics = {
            name: mean([slots[slot][name] for slot in slots])
            for name in ("precision", "recall", "f1", "em")
        }
    else:
        slots, metrics = None, prf(q["answer"], prediction)
    return {
        "status": "ok",
        "score": metrics["f1"],
        "max_score": 1,
        "normalized_score": metrics["f1"],
        "metrics": metrics,
        "slots": slots,
        "reason": None,
        "error": None,
    }


def aggregate(records):
    valid = [r for r in records if r["status"] == "ok"]
    normalized = mean([r["normalized_score"] for r in valid])
    scales = {r["max_score"] for r in valid}
    result = {
        "n_total": len(records),
        "n_scored": len(valid),
        "coverage": len(valid) / len(records) if records else None,
        "statuses": dict(Counter(r["status"] for r in records)),
        "normalized_mean": normalized,
        "percent_score": normalized * 100 if normalized is not None else None,
    }
    if len(scales) == 1:
        result["raw_mean"] = mean([r["score"] for r in valid])
        result["max_score"] = valid[0]["max_score"]
        if scales == {1} and all(r.get("metrics") is None for r in valid):
            result["acc"] = result["raw_mean"]
    labeled = [r for r in valid if r.get("metrics") is not None]
    if labeled:
        result["labels"] = {
            name: mean([r["metrics"][name] for r in labeled])
            for name in ("precision", "recall", "f1", "em")
        }
        transitions = [r for r in labeled if r.get("slots") is not None]
        if transitions:
            result["before_after"] = {
                slot: {
                    name: mean([r["slots"][slot][name] for r in transitions])
                    for name in ("precision", "recall", "f1", "em")
                }
                for slot in ("before", "after")
            }
    return result


def reasoning_summary(records):
    groups = {
        "result": [r for r in records if r["max_score"] == 1],
        "explanation": [r for r in records if r["max_score"] > 1],
    }
    separate = {kind: aggregate(group) for kind, group in groups.items()}
    scores = {k: v["normalized_mean"] for k, v in separate.items() if v["n_scored"]}
    weights = {"result": 0.6, "explanation": 0.4}
    weighted = (
        sum(weights[k] * v for k, v in scores.items()) / sum(weights[k] for k in scores)
        if scores
        else None
    )
    unweighted = aggregate(records)["normalized_mean"]
    return {
        "result": separate["result"],
        "explanation": separate["explanation"],
        "weights": weights,
        "unweighted": unweighted,
        "unweighted_percent": unweighted * 100 if unweighted is not None else None,
        "weighted": weighted,
        "weighted_percent": weighted * 100 if weighted is not None else None,
    }


def summarize(records):
    granularities = {r["granularity"] for r in records}
    if len(granularities) != 1:
        raise ValueError("summarize one granularity at a time")
    grouped = defaultdict(list)
    for row in records:
        grouped[row["type"]].append(row)
    result = {
        "granularity": next(iter(granularities)),
        "overall_unweighted": aggregate(records),
        "tasks": {k: aggregate(v) for k, v in sorted(grouped.items())},
    }
    if result["granularity"] == "episode":
        reasoning = grouped.get("emotional reasoning", [])
        result["emotional_reasoning"] = reasoning_summary(reasoning)
    return result


def markdown_summary(metrics):
    def value(x):
        return "—" if x is None else f"{x:.4f}"

    rows = [
        "# Evaluation results",
        "",
        f"Granularity: `{metrics['granularity']}`",
        "",
        "| Task | Scored / total | Raw mean | Normalized | Percent |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, data in metrics["tasks"].items():
        rows.append(
            f"| {name} | {data['n_scored']} / {data['n_total']} | {value(data.get('raw_mean'))} | {value(data['normalized_mean'])} | {value(data['percent_score'])} |"
        )
    for name, data in metrics["tasks"].items():
        if "labels" in data:
            rows += [
                "",
                f"{name}: "
                + "; ".join(
                    f"{k.upper()}={value(v)}" for k, v in data["labels"].items()
                ),
                "",
            ]
    if "emotional_reasoning" in metrics:
        r = metrics["emotional_reasoning"]
        result, explanation = r["result"], r["explanation"]
        correct = (
            str(round(result["acc"] * result["n_scored"]))
            if result["n_scored"]
            else "—"
        )
        maximum = explanation.get("max_score")
        scale = f"0–{maximum}" if maximum is not None else "per-question scale"
        raw_mean = value(explanation.get("raw_mean"))
        if maximum is not None:
            raw_mean += f" / {maximum}"
        rows += [
            "",
            "G2 emotional reasoning:",
            "",
            "| Question group | Scored / total | Raw metric | Normalized | Percent |",
            "|---|---:|---|---:|---:|",
            f"| Result questions (score 0/1) | {result['n_scored']} / {result['n_total']} | {correct} / {result['n_scored']} | {value(result['normalized_mean'])} | {value(result['percent_score'])} |",
            f"| Explanation questions (rubric {scale}) | {explanation['n_scored']} / {explanation['n_total']} | Mean score: {raw_mean} | {value(explanation['normalized_mean'])} | {value(explanation['percent_score'])} |",
            "",
            f"- Unweighted: {value(r['unweighted'])}; percent: {value(r['unweighted_percent'])}",
            f"- Weighted (60% results / 40% explanations): {value(r['weighted'])}; percent: {value(r['weighted_percent'])}",
        ]
    overall = metrics["overall_unweighted"]
    rows += [
        "",
        f"Overall unweighted normalized mean: {value(overall['normalized_mean'])}",
        f"Scored: {overall['n_scored']} / {overall['n_total']}; statuses: `{overall['statuses']}`.",
        "",
        "Missing or blank predictions and evaluation failures are unscored, reported in coverage and status counts.",
    ]
    return "\n".join(rows) + "\n"
