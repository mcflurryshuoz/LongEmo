#!/usr/bin/env python3
"""Shared helpers for model-specific inference scripts.

The runner reads benchmark question items and writes a NEW merged file
(code/output/..._pred_<model>.json by default) that mirrors the input list, with
`pred_answer` plus a `pred_info: {model, reason, raw, error}` block added per
answered item. The original questions file is never modified.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from benchmark.common import io_utils


# The answer format itself is specified by each question's own final instruction
# (single authority; see docs/qa-granularity-*-guide.md tails). The contract only
# defines the harness-level wrapper and how the answer nests inside it. reason
# comes before answer so the model states its evidence before committing.
_RESPONSE_CONTRACT = (
    "Return ONLY a JSON object with exactly these keys in this order: reason, answer.",
    "Write reason first: one short sentence grounded in the visible/audible evidence.",
    "The question's final instruction specifies the answer format; follow it exactly and put that answer"
    ' (a string, or a JSON object if the question asks for one) as the value of "answer".',
    "Put no explanation inside answer.",
)


def build_prompt(record: dict[str, Any], *, with_transcript: bool = False) -> str:
    # The model sees only the standardized question (plus optional transcript);
    # internal fields like emotion_answer and the gold answer stay out of the prompt.
    has_transcript = with_transcript and bool(record.get("input_transcript"))
    evidence = "video and transcript" if has_transcript else "video"
    parts = [
        f"You answer video-based emotion benchmark questions from the provided {evidence}.",
        "",
        "Response contract:",
        *_RESPONSE_CONTRACT,
        "",
        "Question:",
        # question_text appends the option list, so choice questions always
        # reach the model with their options.
        io_utils.question_text(record),
    ]
    if has_transcript:
        # text only: speaker attribution must come from the video, and real SDH
        # subtitles carry no speaker labels anyway.
        rows = [str(t.get("text", "")) for t in record.get("input_transcript") or []]
        parts.extend([
            "",
            "Subtitles for the same video span, in time order. They carry no speaker labels and"
            " consecutive lines may be spoken by different people:",
            "\n".join(rows),
        ])
    return "\n".join(parts).strip()


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
    for i, q in enumerate(questions, 1):
        key = io_utils.question_key(q, i)
        if wanted and not wanted & {key, io_utils.safe_qid(q, i), str(q.get("qid"))}:
            continue
        selected.append((key, q))
    limit = getattr(args, "limit", None)
    if limit:
        selected = selected[:limit]
    return selected


def parse_prediction(raw: str) -> tuple[Any, str | None]:
    """Split a model response into (pred_answer, reason) per the answer contract.

    Self-hosted thinking variants (e.g. qwen3-omni-*-thinking on vLLM without a
    reasoning parser) may inline <think>...</think> before the answer JSON; the
    block is stripped here, while the stored raw keeps the full response.
    """
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    parsed = io_utils.parse_jsonish(text)
    if isinstance(parsed, dict) and "answer" in parsed:
        reason = parsed.get("reason")
        return parsed["answer"], str(reason).strip() if reason else None
    return text, None


def default_out_path(args: Any, questions_path: str | Path, questions: list[dict[str, Any]]) -> Path:
    """Default: code/output/<series>_<stem>_pred_<model>.json — never inside datasets/."""
    if getattr(args, "out", None):
        return Path(args.out)
    out_dir = Path(__file__).resolve().parents[2] / "output"
    stem = Path(questions_path).stem
    series = {str(q.get("series")) for q in questions if q.get("series")}
    prefix = f"{series.pop()}_" if len(series) == 1 else ""
    return out_dir / f"{prefix}{stem}_pred_{args.model}.json"


def run_benchmark(
    *,
    args: Any,
    questions: list[dict[str, Any]],
    out_path: Path,
    ask: Callable[[dict[str, Any], list[dict[str, Any]]], str],
    get_clips: Callable[[str, dict[str, Any]], list[dict[str, Any]]],
    release_clips: Callable[[list[dict[str, Any]]], None] | None = None,
) -> None:
    """Answer selected questions and write the merged prediction file.

    `get_clips` returns per-question clip dicts (already-cut in batch mode, cut
    on demand in streaming mode); `release_clips` deletes streaming clips so no
    intermediate result is kept.
    """
    existing: dict[str, dict[str, Any]] = {}
    if out_path.exists() and not args.force:
        for i, item in enumerate(io_utils.load_questions(out_path), 1):
            if "pred_answer" in item:
                existing[io_utils.question_key(item, i)] = {
                    "pred_answer": item.get("pred_answer"),
                    "pred_info": item.get("pred_info"),
                }

    items = [dict(q) for q in questions]
    by_key = {io_utils.question_key(q, i): item for i, (q, item) in enumerate(zip(questions, items), 1)}
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
        clips: list[dict[str, Any]] = []
        try:
            clips = get_clips(key, q)
            for k in range(args.tries):
                try:
                    raw = ask(q, clips)
                    error = None
                    break
                except Exception as e:
                    error = f"{type(e).__name__}: {e}"
                    time.sleep(2 * (k + 1))
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
        finally:
            if release_clips is not None and clips:
                release_clips(clips)
        pred_answer, reason = parse_prediction(raw) if raw else (None, None)
        item["pred_answer"] = pred_answer
        item["pred_info"] = {"model": args.model, "reason": reason, "raw": raw, "error": error}
        if error:
            print(f"[{i}/{len(selected)}] {key}: ERROR {error}", flush=True)
        else:
            answered += 1
        io_utils.write_json(out_path, items)

    io_utils.write_json(out_path, items)
    errors = sum(1 for it in items if (it.get("pred_info") or {}).get("error"))
    print(json.dumps({"questions": len(items), "answered": answered, "errors": errors, "out": str(out_path)}, ensure_ascii=False))
