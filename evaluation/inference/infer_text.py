#!/usr/bin/env python3
"""Run text-only inference for 2_episode questions."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from evaluation import contract, io_utils  # noqa: E402
from evaluation.inference import prompts, runner  # noqa: E402


def _client(base_url: str, api_key: str, timeout: float):
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)


def _thinking_options(base_url: str, thinking: str) -> dict[str, Any]:
    if thinking == "default":
        return {}
    host = (urlparse(base_url).hostname or "").lower()
    if host != "api.deepseek.com":
        raise ValueError("--thinking on/off is supported only for the official DeepSeek endpoint")
    return {
        "extra_body": {
            "thinking": {"type": "enabled" if thinking == "on" else "disabled"}
        }
    }


def _clock(seconds: float) -> str:
    total_ms = round(float(seconds) * 1000)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def load_subtitle_inputs(
    subtitles_dir: str | Path,
    key: str,
    question: dict[str, Any],
) -> list[dict[str, Any]]:
    """Load every episode named by one g2 question and validate its subtitle rows."""
    if contract.question_granularity(question) != "2_episode":
        raise ValueError(f"{key}: text inference supports only 2_episode questions")
    episodes = question.get("input_video")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"{key}: input_video must contain at least one episode")

    root = Path(subtitles_dir)
    series = str(question.get("series") or "")
    inputs: list[dict[str, Any]] = []
    for episode in episodes:
        ep = str(episode)
        path = root / series / f"{ep.lower()}.json"
        if not path.exists():
            raise FileNotFoundError(f"{key}: missing subtitles: {path}")
        rows = io_utils.read_json(path)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{key}: subtitle file must contain a non-empty list: {path}")

        previous_start = float("-inf")
        validated: list[dict[str, Any]] = []
        for position, row in enumerate(rows, 1):
            if not isinstance(row, dict):
                raise ValueError(f"{key}: subtitle row {position} is not an object: {path}")
            interval = row.get("t")
            if (
                not isinstance(interval, list)
                or len(interval) != 2
                or not all(isinstance(value, (int, float)) for value in interval)
            ):
                raise ValueError(f"{key}: subtitle row {position} has invalid t: {path}")
            start, end = float(interval[0]), float(interval[1])
            if end < start or start < previous_start:
                raise ValueError(f"{key}: subtitle row {position} has invalid time order: {path}")
            previous_start = start
            speaker = str(row.get("speaker") or "").strip()
            text = str(row.get("text") or "").strip()
            if not speaker or not text:
                raise ValueError(f"{key}: subtitle row {position} lacks speaker or text: {path}")
            validated.append({"t": [start, end], "speaker": speaker, "text": text})
        inputs.append({"kind": "subtitle", "ep": ep, "path": str(path), "rows": validated})
    return inputs


def transcript_text(inputs: list[dict[str, Any]]) -> str:
    """Format timestamped, speaker-attributed subtitle rows for a text model."""
    episodes: list[str] = []
    for item in inputs:
        lines = [f"Episode {item['ep']}:"]
        for row in item["rows"]:
            start, end = row["t"]
            lines.append(
                f"[{_clock(start)} - {_clock(end)}] {row['speaker']}: {row['text']}"
            )
        episodes.append("\n".join(lines))
    return "\n\n".join(episodes)


def text_out_path(
    args: argparse.Namespace,
    questions_path: str | Path,
    questions: list[dict[str, Any]],
) -> Path:
    path = runner.default_out_path(args, questions_path, questions)
    if args.out:
        return path
    return path.with_name(path.name.replace("_pred_", "_text_pred_", 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", required=True, help="Canonical 2_episode question JSON.")
    ap.add_argument(
        "--subtitles-dir",
        default="subtitles",
        help="Root containing <series>/<episode>.json subtitles; default: subtitles.",
    )
    ap.add_argument("--out", default=None, help="Prediction JSON path; the default name includes text.")
    ap.add_argument("--model", required=True, help="Text language model name.")
    ap.add_argument("--base-url", required=True, help="OpenAI-compatible API base URL.")
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    runner.add_thinking_arg(ap)
    ap.add_argument("--qid", action="append", default=None)
    ap.add_argument("--force", action="store_true", help="Re-ask questions that already have a prediction.")
    ap.add_argument("--tries", type=int, default=3)
    args = ap.parse_args()
    args.input_mode = "text"
    args.with_transcript = True

    questions = io_utils.load_questions(args.questions)
    try:
        granularities = {contract.question_granularity(q) for q in questions}
    except ValueError as exc:
        ap.error(str(exc))
    if granularities != {"2_episode"}:
        ap.error("text inference requires a file containing only 2_episode questions")

    client = _client(args.base_url, args.api_key, args.timeout)
    try:
        thinking_options = _thinking_options(args.base_url, args.thinking)
    except ValueError as exc:
        ap.error(str(exc))

    def load_inputs(key: str, q: dict[str, Any]) -> list[dict[str, Any]]:
        return load_subtitle_inputs(args.subtitles_dir, key, q)

    def ask(q: dict[str, Any], inputs: list[dict[str, Any]]) -> str:
        prompt = prompts.build_text_inference_prompt(q, transcript_text(inputs))
        request: dict[str, Any] = {
            "model": args.model,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        request.update(thinking_options)
        response = client.chat.completions.create(
            **request,
        )
        return response.choices[0].message.content or ""

    runner.run_benchmark(
        args=args,
        questions=questions,
        out_path=text_out_path(args, args.questions, questions),
        ask=ask,
        load_inputs=load_inputs,
    )


if __name__ == "__main__":
    main()
