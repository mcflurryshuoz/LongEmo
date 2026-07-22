#!/usr/bin/env python3
"""Check benchmark question files against the formal evaluation contract.

Accepts either one question file (an episode file or the final unified
benchmark JSON) or the datasets root directory (scans */grade_1/*.json, in
which case each question's `series` must match its directory).

Per question it checks the schema fields (series/qid/inputs) plus the
gold-answer format contract from common/contract.py; across questions it
checks that (series, qid) is unique — bare qids repeat across series.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from benchmark.common import contract, io_utils  # noqa: E402

REQUIRED_FIELDS = ("series", "qid", "question_type", "question", "options", "answer", "input_video", "input_transcript")


def _check_input_video(q: dict[str, Any], qid: str) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []

    def err(reason: str) -> None:
        errors.append({"qid": qid, "category": "input_video", "reason": reason})

    iv = q.get("input_video")
    if not isinstance(iv, dict):
        err("input_video must be an object with eps/segments")
        return errors
    eps = iv.get("eps") or []
    if not isinstance(eps, list) or not eps:
        err("input_video.eps must be a non-empty list")
        eps = []
    segments = iv.get("segments") or []
    if not isinstance(segments, list) or not segments:
        err("input_video.segments must be a non-empty list")
        segments = []
    for j, seg in enumerate(segments, 1):
        if not isinstance(seg, dict) or not all(k in seg for k in ("ep", "start", "end")):
            err(f"segment {j} must be an object with ep/start/end")
            continue
        try:
            if float(seg["end"]) <= float(seg["start"]):
                err(f"segment {j} end <= start")
        except (TypeError, ValueError):
            err(f"segment {j} start/end are not numeric")
        if eps and str(seg.get("ep")) not in set(map(str, eps)):
            err(f"segment {j} ep is not listed in input_video.eps")
    return errors


def _check_input_transcript(q: dict[str, Any], qid: str) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []

    def err(reason: str) -> None:
        errors.append({"qid": qid, "category": "input_transcript", "reason": reason})

    transcript = q.get("input_transcript")
    if not isinstance(transcript, list) or not transcript:
        err("input_transcript must be a non-empty list")
        return errors
    eps = set(map(str, (q.get("input_video") or {}).get("eps") or [])) if isinstance(q.get("input_video"), dict) else set()
    for j, row in enumerate(transcript, 1):
        if not isinstance(row, dict):
            err(f"transcript row {j} must be an object")
            continue
        if eps and str(row.get("ep")) not in eps:
            err(f"transcript row {j} ep is not listed in input_video.eps")
        if not str(row.get("text") or "").strip():
            err(f"transcript row {j} missing text")
        t = row.get("t")
        try:
            if not isinstance(t, list) or len(t) != 2 or float(t[1]) <= float(t[0]):
                err(f"transcript row {j} t must be numeric [start, end] with end > start")
        except (TypeError, ValueError):
            err(f"transcript row {j} t must be numeric [start, end] with end > start")
    return errors


def check_questions(questions: list[Any], *, expected_series: str | None = None, source: str | None = None) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    counts = {"by_series": Counter(), "by_question_type": Counter()}
    for idx, q in enumerate(questions, 1):
        if not isinstance(q, dict):
            errors.append({"qid": f"item{idx}", "category": "schema", "reason": "question item must be an object"})
            continue
        qid = io_utils.safe_qid(q, idx)
        counts["by_series"][str(q.get("series"))] += 1
        counts["by_question_type"][str(q.get("question_type"))] += 1
        for field in REQUIRED_FIELDS:
            if field == "options":
                if not isinstance(q.get("options"), list):
                    errors.append({"qid": qid, "category": "schema", "reason": "options must be a list (empty for open questions)"})
            elif field not in q or q[field] in (None, ""):
                errors.append({"qid": qid, "category": "schema", "reason": f"missing/empty {field!r}"})
        if expected_series and q.get("series") != expected_series:
            errors.append({"qid": qid, "category": "series", "reason": f"series {q.get('series')!r} does not match directory {expected_series!r}"})
        key = io_utils.question_key(q, idx)
        if key in seen:
            errors.append({"qid": qid, "category": "qid", "reason": f"duplicate (series, qid) also seen in {seen[key]}"})
        seen[key] = source or "input"
        errors.extend({"qid": qid, "category": "gold_format", "reason": r} for r in contract.validate_gold_format(q))
        errors.extend(_check_input_video(q, qid))
        errors.extend(_check_input_transcript(q, qid))
    return {
        "ok": not errors,
        "count": len(questions),
        "counts": {k: dict(v) for k, v in counts.items()},
        "errors": errors,
    }


def check_path(path: str | Path) -> dict[str, Any]:
    """Check one question file, or every */grade_1/*.json under a datasets dir."""
    p = Path(path)
    if p.is_file():
        return {**check_questions(io_utils.load_questions(p), source=str(p)), "source": str(p)}
    # prediction files (pred_<model>.json) live beside question files; skip them
    files = sorted(f for f in p.glob("*/grade_1/*.json") if not f.name.startswith("pred_"))
    if not files:
        raise SystemExit(f"no question files found under {p} (expected */grade_1/*.json)")
    merged: list[dict[str, Any]] = []
    reports = []
    for f in files:
        questions = io_utils.load_questions(f)
        report = check_questions(questions, expected_series=f.parent.parent.name, source=str(f))
        for e in report["errors"]:
            e["file"] = str(f)
        reports.append(report)
        merged.extend(questions)
    # cross-file duplicate check on the merged set
    seen: dict[str, int] = {}
    dup_errors = []
    for idx, q in enumerate(merged, 1):
        key = io_utils.question_key(q, idx)
        if key in seen:
            dup_errors.append({"qid": io_utils.safe_qid(q, idx), "category": "qid", "reason": f"duplicate (series, qid) across files: {key}"})
        seen[key] = idx
    errors = [e for r in reports for e in r["errors"]] + dup_errors
    counts = {"by_series": Counter(), "by_question_type": Counter()}
    for r in reports:
        for k in counts:
            counts[k].update(r["counts"][k])
    return {
        "ok": not errors,
        "source": str(p),
        "files": len(files),
        "count": len(merged),
        "counts": {k: dict(v) for k, v in counts.items()},
        "errors": errors,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", required=True, help="Question file (episode or unified benchmark JSON) or datasets root directory.")
    ap.add_argument("--out", default=None, help="Optional JSON report path.")
    args = ap.parse_args()

    report = check_path(args.questions)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text, end="")
    if not report["ok"]:
        raise SystemExit(f"question contract errors: {len(report['errors'])}")


if __name__ == "__main__":
    main()
