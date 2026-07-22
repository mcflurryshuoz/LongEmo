#!/usr/bin/env python3
"""Run Gemini video QA inference on delivery question.json (OpenAI-compatible endpoint).

Clips come either from a batch prepare_clips run (--manifest) or are cut per
question and deleted right after (--streaming). Writes pred_<model>.json next
to the questions file (see common/inference_utils.run_benchmark).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from benchmark.common import io_utils, media  # noqa: E402
from benchmark.inference import inference_utils  # noqa: E402
from benchmark.preprocess import prepare_clips  # noqa: E402

DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _client(base_url: str, api_key: str, timeout: float):
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)


def _content(q: dict[str, Any], clips: list[dict[str, Any]], *, with_transcript: bool) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {"type": "text", "text": inference_utils.build_prompt(q, with_transcript=with_transcript)}
    ]
    for clip in clips:
        content.append({"type": "video_url", "video_url": {"url": media.data_url(clip["clip_path"], "video/mp4")}})
    return content


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default=None, help="Delivery question.json.")
    ap.add_argument("--out", default=None, help="Output pred json path. Default: code/output/<series>_<stem>_pred_<model>.json.")
    prepare_clips.add_clip_args(ap)
    ap.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"))
    ap.add_argument(
        "--base-url",
        default=os.environ.get("GEMINI_BASE_URL", DEFAULT_GEMINI_BASE_URL),
        help="Gemini OpenAI-compatible endpoint.",
    )
    ap.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    ap.add_argument("--timeout", type=float, default=float(os.environ.get("GEMINI_TIMEOUT", "180")))
    # generous cap: Gemini's maxOutputTokens includes hidden thinking tokens;
    # non-thinking answers just stop early, so the headroom is free.
    ap.add_argument("--max-tokens", type=int, default=int(os.environ.get("GEMINI_MAX_TOKENS", "4096")))
    ap.add_argument("--temperature", type=float, default=float(os.environ.get("GEMINI_TEMPERATURE", "0.0")))
    ap.add_argument("--with-transcript", action="store_true", help="Also feed the input_transcript subtitles to the model.")
    inference_utils.add_thinking_arg(ap)
    ap.add_argument("--qid", action="append", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="Re-ask questions that already have a prediction.")
    ap.add_argument("--tries", type=int, default=3)
    args = ap.parse_args()

    if not args.questions:
        ap.error("--questions is required")
    if not args.clips_dir and not args.streaming:
        ap.error("pass --clips-dir <pre-cut clips> (batch) or --streaming (cut per question, keep no clips)")
    if not args.api_key:
        raise SystemExit("Missing --api-key or GEMINI_API_KEY/GOOGLE_API_KEY for Gemini inference.")

    client = _client(args.base_url, args.api_key, args.timeout)
    questions = io_utils.load_questions(args.questions)
    get_clips, release_clips = prepare_clips.clip_provider(args, args.questions)

    def ask(q: dict[str, Any], clips: list[dict[str, Any]]) -> str:
        kwargs: dict[str, Any] = dict(
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            messages=[{"role": "user", "content": _content(q, clips, with_transcript=args.with_transcript)}],
        )
        if args.thinking != "default":
            # Gemini's OpenAI-compatible endpoint controls thinking via reasoning_effort;
            # thinking tokens are never returned in the message content. Note some pro
            # models reject "none" (their thinking cannot be disabled).
            kwargs["extra_body"] = {"reasoning_effort": "medium" if args.thinking == "on" else "none"}
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    inference_utils.run_benchmark(
        args=args,
        questions=questions,
        out_path=inference_utils.default_out_path(args, args.questions, questions),
        ask=ask,
        get_clips=get_clips,
        release_clips=release_clips,
    )


if __name__ == "__main__":
    main()
