"""Run the bounded coarse-to-fine Agentic video method over API models.

The method always sends sampled frames. Audio and subtitles are optional inputs
selected once at run start; inspect actions can request only additional visual
intervals and cannot change those inputs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.inference.runner import init_client, model_args, retry
from evaluation.inference.video_loader import (
    extract_audio,
    file_hash,
    sample_video_frames,
)
from evaluation.io_utils import (
    load_questions,
    load_records,
    question_key,
    subtitle_text,
    write_records,
)

from .evidence import sample_interval
from .prompts import SYSTEM_PROMPT, followup_prompt, initial_prompt
from .protocol import decode_action

SUPPORTED_FAMILIES = {"qwen_omni", "qwen_vl", "gpt", "claude", "gemini"}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--data-path", required=True, help="Question JSON/JSONL file or directory")
    p.add_argument("-g", "--granularity", choices=("clip", "episode"), default="clip")
    p.add_argument("--videos-dir", help="Directory containing <video_id>.mp4 files")
    p.add_argument("--output-dir", required=True, help="Output directory")
    p.add_argument("--with-audio", action="store_true", help="Provide the source video's audio")
    p.add_argument("--with-subtitle", action="store_true", help="Provide subtitles with the input")
    p.add_argument("--initial-fps", type=float, default=0.5)
    p.add_argument("--initial-max-frames", type=int, default=32)
    p.add_argument("--initial-max-pixels", type=int, default=200704)
    p.add_argument("--max-rounds", type=int, default=3)
    p.add_argument("--qid", action="append")
    p.add_argument("--limit", type=int)
    p.add_argument("--force", action="store_true")
    model_args(p)
    return p


def _user_content(text: str, frames: list[dict], audio: dict | None = None) -> list[dict]:
    content = [{"type": "text", "text": text}]
    content.extend(frames)
    if audio is not None:
        content.extend(
            [
                {"type": "text", "text": "Audio from the same video; time zero matches the video timeline."},
                audio,
            ]
        )
    return content


def _video_for(q: dict, videos_dir: str | None, data_path: str) -> dict:
    if videos_dir:
        root = Path(videos_dir)
    else:
        base = Path(data_path) if Path(data_path).is_dir() else Path(data_path).parent
        root = base / "videos"
    path = (root / f"{q['video_id']}.mp4").resolve()
    if not path.is_file() or not path.stat().st_size:
        raise ValueError(f"missing or empty prepared video: {path}")
    return {"path": str(path), "sha256": file_hash(path)}


def _run_one(
    client,
    q: dict,
    video: dict,
    *,
    with_audio: bool,
    with_subtitle: bool,
    initial_fps: float,
    initial_max_frames: int,
    initial_max_pixels: int,
    max_rounds: int,
    tries: int,
) -> dict:
    frames, meta = sample_video_frames(
        video,
        fps=initial_fps,
        max_frames=initial_max_frames,
        max_pixels=initial_max_pixels,
    )
    subtitle = subtitle_text(q, required=False) if with_subtitle else None
    question_text = initial_prompt(
        q, with_audio=with_audio, with_subtitle=with_subtitle
    )
    if subtitle:
        question_text += "\n\nSubtitles:\n" + subtitle
    audio = extract_audio(video, required=with_audio)[0] if with_audio else None
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _user_content(question_text, frames, audio)},
    ]
    trace = [{"round": 0, "action": "initial", "sampling": meta}]

    for round_index in range(1, max_rounds + 1):
        response = retry(lambda: client.generate(messages), tries)
        raw = response["content"]
        action = decode_action(raw)
        if action["action"] == "answer":
            return {
                "status": "ok",
                "prediction": action["answer"],
                "raw_answer": raw,
                "trace": trace,
                "usage": response.get("usage"),
                "raw_response": response.get("raw_response"),
            }
        if round_index == max_rounds:
            raise ValueError("model requested more inspections than max-rounds")
        evidence: list[dict] = []
        evidence_meta: list[dict] = []
        for item in action["ranges"]:
            blocks, info = sample_interval(video, **item)
            evidence.extend(blocks)
            evidence_meta.append(info)
        messages.append({"role": "assistant", "content": raw})
        messages.append(
            {
                "role": "user",
                "content": _user_content(followup_prompt(), evidence),
            }
        )
        trace.append(
            {"round": round_index, "action": "inspect", "ranges": evidence_meta}
        )
    raise AssertionError("agentic loop terminated without answer")


def run(args) -> int:
    if args.max_rounds < 1:
        raise ValueError("max-rounds must be positive")
    if args.initial_fps <= 0 or args.initial_max_frames < 1:
        raise ValueError("initial sampling settings must be positive")
    client = init_client(args)
    if args.model_family not in SUPPORTED_FAMILIES:
        raise ValueError(
            "Agentic video supports only Qwen Omni, Qwen VL, GPT, Claude and Gemini"
        )
    if args.with_audio and args.model_family in {"claude", "qwen_vl"}:
        raise ValueError(
            "the selected model adapter does not accept audio blocks; omit --with-audio"
        )
    if args.with_audio and args.api_format == "responses":
        raise ValueError(
            "audio input requires a chat-compatible endpoint for this method"
        )
    questions = load_questions(args.data_path, args.granularity, args.qid, limit=args.limit)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "predictions.jsonl"
    old = (
        {}
        if args.force or not destination.exists()
        else {x.get("question_id"): x for x in load_records(destination)}
    )
    results = []
    for q in questions:
        key = question_key(q)
        if key in old:
            results.append(old[key])
            continue
        base = {
            "question_id": q["question_id"],
            "video_id": q["video_id"],
            "granularity": args.granularity,
            "type": q.get("type"),
            "method": "agentic",
            "configuration": {
                "with_audio": args.with_audio,
                "with_subtitle": args.with_subtitle,
                "initial_fps": args.initial_fps,
                "initial_max_frames": args.initial_max_frames,
                "initial_max_pixels": args.initial_max_pixels,
                "max_rounds": args.max_rounds,
            },
        }
        try:
            video = _video_for(q, args.videos_dir, args.data_path)
            result = _run_one(
                client,
                q,
                video,
                with_audio=args.with_audio,
                with_subtitle=args.with_subtitle,
                initial_fps=args.initial_fps,
                initial_max_frames=args.initial_max_frames,
                initial_max_pixels=args.initial_max_pixels,
                max_rounds=args.max_rounds,
                tries=args.tries,
            )
        except Exception as exc:
            result = {
                "status": "error",
                "prediction": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append({**base, **result})
        write_records(destination, results)
    write_records(destination, results)
    return int(any(x.get("status") != "ok" for x in results))


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        return run(args)
    except (ValueError, OSError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
