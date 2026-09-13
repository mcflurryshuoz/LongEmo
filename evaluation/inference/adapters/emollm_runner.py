"""Batch utilities used by each official model's independent entry point.

This module does not choose or import model implementations. Each model entry
supplies its own runtime factory and runs in its own dependency environment.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from ...io_utils import (
    load_questions,
    question_key,
    subtitle_text,
)
from ..prompts import (
    build_messages,
    parse_prediction,
)
from ..runner import retry, execute_tasks


SOURCE_ROOT = Path(__file__).resolve().parent.parent / "emollm"
REPOSITORIES = {
    "affectgpt": "AffectGPT",
    "emotion_llama": "Emotion-LLaMA",
    "r1_omni": "R1-Omni",
}


def parser(model):
    p = argparse.ArgumentParser(
        description=f"Run official {REPOSITORIES[model]} on LongEmoBench clip questions."
    )
    p.add_argument(
        "--data-path",
        required=True,
        help="Path to a question JSON/JSONL file or a flat question directory",
    )
    p.add_argument(
        "--videos-dir",
        help="Prepared <video_id>.mp4 files; defaults to videos/ under the data directory",
    )
    p.add_argument(
        "--runtime-config",
        required=True,
        help="JSON with this model's local weight paths and generation options",
    )
    p.add_argument("-g", "--granularity", choices=("clip",), default="clip")
    p.add_argument(
        "--with-subtitle",
        action="store_true",
        help="Append subtitles to the prompt; default off",
    )
    p.add_argument("--qid", action="append")
    p.add_argument(
        "--limit", type=int, help="Run the first N selected questions; default all"
    )
    p.add_argument("--output-dir", help="Output directory")
    p.add_argument("--tries", type=int, default=1)
    p.add_argument("--force", action="store_true")
    return p


def read_config(path, model):
    config_file = Path(path).expanduser().resolve()
    config = json.loads(config_file.read_text())
    if not isinstance(config, dict):
        raise ValueError("runtime-config must contain a JSON object")
    # Explicit local paths in JSON are relative to that JSON, not the launch directory.
    # OmegaConf options remain upstream expressions and are never rewritten.
    for key, value in list(config.items()):
        if key.endswith(("_path", "_dir")) or key == "checkpoint":
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be a non-empty local path")
            asset = Path(value).expanduser()
            config[key] = str((config_file.parent / asset).resolve())
    if "repo_path" not in config:
        config.setdefault("repo_dir", str(SOURCE_ROOT / REPOSITORIES[model]))
    return config


def build_prompt(q, transcript):
    messages = build_messages(
        q, subtitle_text=transcript, prompt_mode="user"
    )
    return "\n\n".join(message["content"] for message in messages)


def run(model, factory, args):
    if args.tries < 1:
        raise ValueError("tries must be positive")
    data_path = Path(args.data_path)
    data_dir = data_path if data_path.is_dir() else data_path.parent
    videos_dir = Path(args.videos_dir) if args.videos_dir else data_dir / "videos"
    config = read_config(args.runtime_config, model)
    questions = load_questions(
        args.data_path, args.granularity, args.qid, limit=args.limit
    )
    out = Path(args.output_dir or f"output/clip/video/{model}").resolve()
    prepared, tasks = {}, []
    for q in questions:
        key = question_key(q)
        try:
            video = (videos_dir / f"{q['video_id']}.mp4").resolve()
            if not video.is_file() or not video.stat().st_size:
                raise ValueError(f"missing or empty prepared video: {video}")
            transcript = (
                subtitle_text(q, required=False) if args.with_subtitle else None
            )
            record = {
                "video": {"path": str(video)},
                "prompt": build_prompt(q, transcript),
            }
            if model == "affectgpt" and config.get("audio_dir"):
                audio = (Path(config["audio_dir"]) / f"{q['video_id']}.wav").resolve()
                if not audio.is_file() or not audio.stat().st_size:
                    raise ValueError(f"missing or empty audio sidecar: {audio}")
                record["audio"] = {"path": str(audio)}
        except (OSError, ValueError) as exc:
            record = {"error": str(exc)}
        prepared[key] = (q, record)
        base = {
            field: q[field]
            for field in ("question_id", "video_id", "granularity", "type")
        }
        tasks.append((key, base))

    runtime = None

    def generate_prediction(task):
        nonlocal runtime
        q, record = prepared[task[0]]
        if "error" in record:
            return {
                "status": "input_error",
                "prediction": None,
                "error": record["error"],
            }
        if runtime is None:
            runtime = factory(copy.deepcopy(config))
        raw = retry(
            lambda: runtime.generate(record["video"]["path"], record["prompt"]),
            args.tries,
        )
        if not isinstance(raw, str):
            raise TypeError("official runtime must return the generated answer as text")
        predicted, parse_error = parse_prediction(q, raw)
        return {
            "status": "ok",
            "prediction": predicted,
            "raw_answer": raw,
            "request": record,
            "parse_error": parse_error,
            "error": None,
            "media": copy.deepcopy(getattr(runtime, "metadata", {})),
        }

    # One runtime per process: the official stacks modify registries and CUDA state.
    results = execute_tasks(
        tasks, generate_prediction, out / "predictions.jsonl", workers=1, force=args.force
    )
    return int(any(row["status"] != "ok" for row in results))


def main(model, factory, argv=None):
    p = parser(model)
    args = p.parse_args(argv)
    try:
        return run(model, factory, args)
    except (OSError, ValueError) as exc:
        p.error(str(exc))
