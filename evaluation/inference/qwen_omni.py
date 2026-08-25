#!/usr/bin/env python3
"""Run Qwen-Omni inference for 1_clip and 2_episode questions."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from evaluation import io_utils, media  # noqa: E402
from evaluation.inference import runner  # noqa: E402
from evaluation.inference import video_loader  # noqa: E402

def _client(base_url: str, api_key: str, timeout: float):
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)


def _content(q: dict[str, Any], clips: list[dict[str, Any]], *, with_transcript: bool) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for clip in clips:
        content.append({"type": "video_url", "video_url": {"url": media.data_url(clip["clip_path"], "video/mp4")}})
    content.append({"type": "text", "text": runner.build_prompt(q, with_transcript=with_transcript)})
    return content


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", required=True, help="Canonical benchmark question JSON.")
    ap.add_argument("--out", default=None, help="Output path. Default: LongEmoBench/output/<series>_<stem>_pred_<model>.json.")
    video_loader.add_input_args(ap)
    ap.add_argument("--model", default="qwen3.5-omni-flash")
    ap.add_argument(
        "--base-url",
        required=True,
        help="OpenAI-compatible Qwen-Omni endpoint.",
    )
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--timeout", type=float, default=180)
    # generous cap: thinking output (inline or reasoning stream) counts against
    # max_tokens; non-thinking answers just stop early, so the headroom is free.
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument(
        "--no-transcript",
        dest="with_transcript",
        action="store_false",
        default=True,
        help="Do not include the question's speaker-free subtitles.",
    )
    runner.add_thinking_arg(ap)
    ap.add_argument("--qid", action="append", default=None)
    ap.add_argument("--force", action="store_true", help="Re-ask questions that already have a prediction.")
    ap.add_argument("--tries", type=int, default=3)
    args = ap.parse_args()

    questions = io_utils.load_questions(args.questions)
    try:
        get_videos = video_loader.build_loader(args, questions)
    except ValueError as exc:
        ap.error(str(exc))
    client = _client(args.base_url, args.api_key, args.timeout)

    def ask(q: dict[str, Any], clips: list[dict[str, Any]]) -> str:
        kwargs: dict[str, Any] = dict(
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            messages=[{"role": "user", "content": _content(q, clips, with_transcript=args.with_transcript)}],
        )
        if args.thinking != "default":
            kwargs["extra_body"] = {"enable_thinking": args.thinking == "on"}
        if args.thinking == "on":
            # DashScope requires streaming when thinking is enabled; the thinking
            # stream arrives as reasoning_content and is not part of the answer.
            chunks = client.chat.completions.create(stream=True, **kwargs)
            return "".join(
                (c.choices[0].delta.content or "") for c in chunks if c.choices
            )
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    runner.run_benchmark(
        args=args,
        questions=questions,
        out_path=runner.default_out_path(args, args.questions, questions),
        ask=ask,
        get_videos=get_videos,
    )


if __name__ == "__main__":
    main()
