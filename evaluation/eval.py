"""Score labels locally and all other answers with the shared LLM judge."""

from __future__ import annotations
import argparse
import hashlib
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "evaluation"

from .io_utils import (
    load_questions,
    load_records,
    write_json,
)
from .io_utils import question_key
from .inference.prompts import parse_prediction, json_object
from .judge_prompts import (
    judge_result_messages,
    validate_judgment,
)
from .metrics import label_score, summarize, markdown_summary
from .inference.runner import model_args, init_client, retry, execute_tasks


def make_dir(base):
    """Create a run_YYYYMMDD_HHMM directory using local time."""
    name = datetime.now().strftime("run_%Y%m%d_%H%M")
    out = Path(base) / name
    index = 2
    while out.exists():
        out = Path(base) / f"{name}_{index}"
        index += 1
    out.mkdir(parents=True)
    return out


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument(
        "--data-path",
        required=True,
        help="Path to a question JSON/JSONL file or a flat question directory",
    )
    p.add_argument(
        "--predictions",
        required=True,
        help="Prediction JSON/JSONL (own runner or external methods)",
    )
    p.add_argument("-g", "--granularity", choices=("clip", "episode"), default="clip")
    p.add_argument(
        "--output-dir",
        help="Parent output directory; each evaluation creates a new timestamped subdirectory",
    )
    model_args(p)
    p.add_argument("--tries", type=int, default=3)
    p.add_argument("--workers", type=int, default=1)
    return p


def run(args):
    if args.tries < 1 or args.workers < 1:
        raise ValueError("tries and workers must be positive")
    questions = load_questions(args.data_path, args.granularity)
    predictions = {}
    for record in load_records(args.predictions):
        if record["question_id"] in predictions:
            raise ValueError(f"duplicate prediction: {record['question_id']}")
        value = record.get("prediction")
        if value is None or (isinstance(value, str) and not value.strip()):
            value = record.get("pred_answer")
        if isinstance(value, str) and not value.strip():
            value = None
        predictions[record["question_id"]] = value
    needs_judge = any(
        q.get("rubric") is not None
        and predictions.get(question_key(q)) is not None
        for q in questions
    )
    if needs_judge and not args.model:
        raise ValueError("--model is required for LLM-scored answers")
    client = init_client(args) if needs_judge else None
    records, tasks = {}, []
    for q in questions:
        key = question_key(q)
        records[key] = (q, predictions.get(key))
        base = {
            "question_id": q["question_id"],
            "video_id": q["video_id"],
            "granularity": q["granularity"],
            "type": q["type"],
            "max_score": max(int(s) for s in q["rubric"]["scores"])
            if q.get("rubric") is not None
            else 1,
        }
        tasks.append((key, base))

    out = make_dir(args.output_dir or f"output/{args.granularity}/scores")
    write_json(out / "evaluation_config.json", {
        "judge": client.configuration() if client is not None else None,
        "granularity": args.granularity,
        "tries": args.tries,
        "workers": args.workers,
        "source_hashes": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("eval.py", "judge_prompts.py", "metrics.py")},
    })
    print(f"Evaluation results: {out.resolve()}", flush=True)

    def score_answer(task):
        q, value = records[task[0]]
        if value is None:
            return {"status": "missing_prediction", "error": "no submitted answer"}
        if q.get("rubric") is None:
            parsed, warning = parse_prediction(q, value)
            return {
                **label_score(q, parsed),
                "prediction": parsed,
                "parse_error": warning,
            }
        # Open result answers, including Yes/No and numbers, always go to the judge.
        if not isinstance(value, str):
            raise ValueError("open-ended predictions must be text")
        messages, payload = judge_result_messages(q, value)

        attempts = []

        def generate_evaluation():
            response = client.generate(messages)
            attempts.append(response["raw_response"])
            judged = json_object(response["content"])
            validate_judgment(judged, payload["rubric"])
            return judged, response

        try:
            judged, response = retry(generate_evaluation, args.tries)
        except Exception as exc:
            return {
                "status": "error",
                "prediction": value,
                "evaluation_input": payload,
                "judge_responses": attempts,
                "error": f"{type(exc).__name__}: {exc}",
            }
        maximum = task[-1]["max_score"]
        return {
            "status": "ok",
            **judged,
            "max_score": maximum,
            "normalized_score": judged["score"] / maximum,
            "prediction": value,
            "evaluation_input": payload,
            "judge_response": response["raw_response"],
            "judge_responses": attempts,
            "usage": response["usage"],
            "error": None,
        }

    scored = execute_tasks(
        tasks, score_answer, out / "scores.jsonl", workers=args.workers, force=True
    )
    metrics = summarize(scored)
    write_json(out / "metrics.json", metrics)
    (out / "summary.md").write_text(markdown_summary(metrics), encoding="utf-8")
    return int(any(r["status"] != "ok" for r in scored))


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        return run(args)
    except (ValueError, OSError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
