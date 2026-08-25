#!/usr/bin/env python3
"""Shared runner for model-specific benchmark inference scripts.

The runner reads benchmark question items and writes a NEW merged file
(LongEmoBench/output/..._pred_<model>.json by default) that mirrors the input list, with
`pred_answer` plus a `pred_info` audit block added per
answered item. The original questions file is never modified.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from evaluation import contract, io_utils
from evaluation.inference import prompts


build_prompt = prompts.build_prompt


def add_thinking_arg(ap: Any) -> None:
    """Shared --thinking switch; each model script maps it to its provider's native parameter."""
    ap.add_argument(
        "--thinking",
        choices=("on", "off", "default"),
        default="default",
        help=(
            "Model-side reasoning switch: on/off maps to the provider's native thinking "
            "parameter; default leaves the model's own behavior untouched. Recorded in "
            "each prediction for run traceability."
        ),
    )


def select_questions(questions: list[dict[str, Any]], args: Any) -> list[tuple[str, dict[str, Any]]]:
    """Select (key, question) pairs; --qid accepts a bare qid or a series/qid key."""
    selected: list[tuple[str, dict[str, Any]]] = []
    wanted = set(getattr(args, "qid", None) or [])
    for q in questions:
        key = io_utils.question_key(q)
        if wanted and not wanted & {key, q["qid"]}:
            continue
        selected.append((key, q))
    return selected


def parse_prediction(record: dict[str, Any], raw: str) -> tuple[Any, str | None]:
    """Parse the model's final answer using the question's canonical format."""
    return prompts.parse_answer(record, raw)


def default_out_path(args: Any, questions_path: str | Path, questions: list[dict[str, Any]]) -> Path:
    """Default: LongEmoBench/output/<series>_<stem>_pred_<model>.json."""
    if getattr(args, "out", None):
        return Path(args.out)
    out_dir = Path(__file__).resolve().parents[2] / "output"
    question_path = Path(questions_path)
    stem = question_path.stem
    if stem == "all" and question_path.parent.name in {"g1_clip", "g2_episode"}:
        stem = f"{question_path.parent.name}_{stem}"
    series = {str(q.get("series")) for q in questions if q.get("series")}
    prefix = f"{series.pop()}_" if len(series) == 1 else ""
    return out_dir / f"{prefix}{stem}_pred_{args.model}.json"


def run_benchmark(
    *,
    args: Any,
    questions: list[dict[str, Any]],
    out_path: Path,
    ask: Callable[[dict[str, Any], list[dict[str, Any]]], str],
    load_inputs: Callable[[str, dict[str, Any]], list[dict[str, Any]]],
) -> None:
    """Answer selected questions and write the merged prediction file."""
    existing: dict[str, dict[str, Any]] = {}
    if out_path.exists():
        for item in io_utils.load_questions(out_path):
            if "pred_answer" in item:
                existing[io_utils.question_key(item)] = {
                    "pred_answer": item.get("pred_answer"),
                    "pred_info": item.get("pred_info"),
                }

    items = [dict(q) for q in questions]
    by_key = {io_utils.question_key(q): item for q, item in zip(questions, items)}
    for key, fields in existing.items():
        if key in by_key:
            by_key[key].update(fields)

    selected = select_questions(questions, args)
    answered = 0
    for i, (key, q) in enumerate(selected, 1):
        item = by_key[key]
        if "pred_answer" in item and not (item.get("pred_info") or {}).get("error") and not args.force:
            print(f"[{i}/{len(selected)}] {key}: cached", flush=True)
            continue
        print(f"[{i}/{len(selected)}] {key}: ask", flush=True)
        raw = ""
        error: str | None = None
        try:
            inputs = load_inputs(key, q)
            for k in range(args.tries):
                try:
                    raw = str(ask(q, inputs) or "")
                    if not raw.strip():
                        raise RuntimeError("model returned an empty response")
                    error = None
                    break
                except Exception as e:
                    error = f"{type(e).__name__}: {e}"
                    if k + 1 < args.tries:
                        time.sleep(2 * (k + 1))
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
        pred_answer, parse_error = parse_prediction(q, raw) if raw else (None, None)
        item["pred_answer"] = pred_answer
        item["pred_info"] = {
            "model": args.model,
            "granularity": contract.question_granularity(q),
            "input_mode": getattr(args, "input_mode", "video"),
            "prompt_template": q.get("prompt_template"),
            "thinking": getattr(args, "thinking", "default"),
            "with_transcript": bool(getattr(args, "with_transcript", True)),
            "raw": raw,
            "parse_error": parse_error,
            "error": error,
        }
        if error:
            print(f"[{i}/{len(selected)}] {key}: ERROR {error}", flush=True)
        else:
            answered += 1
        io_utils.write_json(out_path, items)

    io_utils.write_json(out_path, items)
    errors = sum(1 for it in items if (it.get("pred_info") or {}).get("error"))
    print(json.dumps({"questions": len(items), "answered": answered, "errors": errors, "out": str(out_path)}, ensure_ascii=False))
