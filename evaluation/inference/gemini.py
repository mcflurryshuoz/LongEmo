#!/usr/bin/env python3
"""Run Gemini inference for 1_clip and 2_episode benchmark questions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib import error, parse, request

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from evaluation import io_utils, media  # noqa: E402
from evaluation.inference import runner  # noqa: E402
from evaluation.inference import video_loader  # noqa: E402

DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _client(base_url: str, api_key: str, timeout: float):
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)


def _api_format(base_url: str) -> str:
    parsed = parse.urlparse(base_url)
    path = parsed.path.rstrip("/").lower()
    if "openai" in path or path.endswith("/v1"):
        return "openai"
    if path.endswith("/gemini") or parsed.netloc.lower() == "generativelanguage.googleapis.com":
        return "native"
    raise ValueError(
        f"cannot infer Gemini API format from --base-url {base_url!r}; "
        "use an OpenAI-compatible /v1 endpoint or a native /gemini endpoint"
    )


def _content(q: dict[str, Any], clips: list[dict[str, Any]], *, with_transcript: bool) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for clip in clips:
        content.append({"type": "video_url", "video_url": {"url": media.data_url(clip["clip_path"], "video/mp4")}})
    content.append({"type": "text", "text": runner.build_prompt(q, with_transcript=with_transcript)})
    return content


def _native_payload(
    q: dict[str, Any],
    clips: list[dict[str, Any]],
    *,
    with_transcript: bool,
    max_tokens: int,
    temperature: float,
    thinking: str,
) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    for clip in clips:
        data_url = media.data_url(clip["clip_path"], "video/mp4")
        parts.append(
            {
                "inline_data": {
                    "mime_type": "video/mp4",
                    "data": data_url.split(",", 1)[1],
                }
            }
        )
    parts.append({"text": runner.build_prompt(q, with_transcript=with_transcript)})

    generation_config: dict[str, Any] = {
        "maxOutputTokens": max_tokens,
        "temperature": temperature,
    }
    if thinking != "default":
        generation_config["thinkingConfig"] = {
            "thinkingBudget": -1 if thinking == "on" else 0,
        }
    return {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": generation_config,
    }


def _native_generate(
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout: float,
    payload: dict[str, Any],
) -> str:
    model_path = parse.quote(model, safe="")
    url = f"{base_url.rstrip('/')}/v1beta/models/{model_path}:generateContent"
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            result = json.load(response)
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"Gemini API HTTP {exc.code}: {detail}") from exc

    text = "".join(
        str(part.get("text") or "")
        for candidate in result.get("candidates") or []
        for part in (candidate.get("content") or {}).get("parts") or []
    )
    if not text:
        raise RuntimeError(f"Gemini API returned no text: {json.dumps(result, ensure_ascii=False)[:2000]}")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", required=True, help="Canonical benchmark question JSON.")
    ap.add_argument("--out", default=None, help="Output path. Default: LongEmoBench/output/<series>_<stem>_pred_<model>.json.")
    video_loader.add_input_args(ap)
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument(
        "--base-url",
        default=DEFAULT_GEMINI_BASE_URL,
        help="Provider base URL; the OpenAI-compatible or native Gemini protocol is inferred from it.",
    )
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--timeout", type=float, default=180)
    # generous cap: Gemini's maxOutputTokens includes hidden thinking tokens;
    # non-thinking answers just stop early, so the headroom is free.
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
        load_inputs = video_loader.build_loader(args, questions)
    except ValueError as exc:
        ap.error(str(exc))
    try:
        api_format = _api_format(args.base_url)
    except ValueError as exc:
        ap.error(str(exc))
    client = _client(args.base_url, args.api_key, args.timeout) if api_format == "openai" else None

    def ask(q: dict[str, Any], clips: list[dict[str, Any]]) -> str:
        if api_format == "native":
            return _native_generate(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                timeout=args.timeout,
                payload=_native_payload(
                    q,
                    clips,
                    with_transcript=args.with_transcript,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    thinking=args.thinking,
                ),
            )

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
        assert client is not None
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    runner.run_benchmark(
        args=args,
        questions=questions,
        out_path=runner.default_out_path(args, args.questions, questions),
        ask=ask,
        load_inputs=load_inputs,
    )


if __name__ == "__main__":
    main()
