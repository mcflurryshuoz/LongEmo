"""Run an InternVL checkpoint directly with its official Transformers-style API."""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "evaluation.inference"

from ...io_utils import load_questions, subtitle_text, write_records
from ..prompts import build_messages


def _dependencies():
    try:
        import numpy as np
        import torch
        from decord import VideoReader, cpu
        from PIL import Image
        from torchvision.transforms import Compose, Normalize, Resize, ToTensor
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - model environment specific
        raise RuntimeError(
            "InternVL Transformers inference requires torch, transformers, decord, "
            "Pillow and torchvision"
        ) from exc
    return np, torch, VideoReader, cpu, Image, Compose, Normalize, Resize, ToTensor, AutoModel, AutoTokenizer


def _prompt(question, with_subtitle):
    messages = build_messages(
        question,
        subtitle_text=subtitle_text(question, required=False) if with_subtitle else None,
        prompt_mode="system",
    )
    system, user = messages[0]["content"], messages[-1]["content"]
    return system + "\n\n" + user


def _sample_video(video_path, *, max_frames, input_size, runtime):
    np, torch, VideoReader, cpu, Image, Compose, Normalize, Resize, ToTensor, _, _ = runtime
    reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    count = len(reader)
    if count < 1:
        raise ValueError(f"empty video: {video_path}")
    indices = np.linspace(0, count - 1, min(max_frames, count), dtype=int).tolist()
    transform = Compose([
        Resize((input_size, input_size)),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    frames = [transform(Image.fromarray(reader[i].asnumpy()).convert("RGB")) for i in indices]
    values = torch.stack(frames)
    return values, len(indices)


def load_runtime(model_path, *, device_map="auto", dtype="bfloat16"):
    runtime = _dependencies()
    _, torch, _, _, _, _, _, _, _, AutoModel, AutoTokenizer = runtime
    torch_dtype = getattr(torch, dtype)
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map=device_map,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    return runtime, tokenizer, model


def generate(runtime, tokenizer, model, question, video_path, *, with_subtitle, max_frames, input_size, max_new_tokens):
    torch = runtime[1]
    pixel_values, frame_count = _sample_video(
        video_path, max_frames=max_frames, input_size=input_size, runtime=runtime
    )
    device = getattr(model, "device", None)
    pixel_values = pixel_values.to(dtype=torch.bfloat16)
    if device is not None and str(device) != "meta":
        pixel_values = pixel_values.to(device)
    prompt = "".join(f"Frame-{i + 1}: <image>\n" for i in range(frame_count)) + _prompt(question, with_subtitle)
    response = model.chat(
        tokenizer,
        pixel_values,
        prompt,
        dict(max_new_tokens=max_new_tokens, do_sample=False),
        num_patches_list=[1] * frame_count,
        history=None,
    )
    if isinstance(response, tuple):
        response = response[0]
    return str(response).strip(), {"representation": "frames", "audio": "not_supported", "num_frames": frame_count}


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--data-path", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("-g", "--granularity", choices=("clip", "episode"), default="clip")
    p.add_argument("--videos-dir")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--qid", action="append")
    p.add_argument("--limit", type=int)
    p.add_argument("--with-subtitle", action="store_true")
    p.add_argument("--max-frames", type=int, default=32)
    p.add_argument("--input-size", type=int, default=448)
    p.add_argument("--max-new-tokens", type=int, default=16384)
    p.add_argument("--device-map", default="auto")
    p.add_argument("--torch-dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    p.add_argument("--force", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.max_frames < 1 or args.input_size < 1 or args.max_new_tokens < 1:
        parser().error("max-frames, input-size and max-new-tokens must be positive")
    data = Path(args.data_path)
    videos_dir = Path(args.videos_dir) if args.videos_dir else (data if data.is_dir() else data.parent) / "videos"
    questions = load_questions(args.data_path, args.granularity, args.qid, limit=args.limit)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    runtime, tokenizer, model = load_runtime(args.model, device_map=args.device_map, dtype=args.torch_dtype)
    prediction_path = out / "predictions.jsonl"
    records = []
    for index, question in enumerate(questions, 1):
        qid = question["question_id"]
        video_path = videos_dir / f"{question['video_id']}.mp4"
        base = {"question_id": qid, "video_id": question["video_id"], "granularity": question["granularity"], "type": question["type"]}
        try:
            answer, media = generate(runtime, tokenizer, model, question, video_path, with_subtitle=args.with_subtitle,
                                     max_frames=args.max_frames, input_size=args.input_size, max_new_tokens=args.max_new_tokens)
            record = {**base, "status": "ok", "prediction": answer, "raw_answer": answer, "media": media, "error": None}
        except Exception as exc:  # preserve failures in long runs
            record = {**base, "status": "error", "prediction": None, "raw_answer": None, "media": None, "error": f"{type(exc).__name__}: {exc}"}
        records.append(record)
        write_records(prediction_path, records)
        print(f"[{index}/{len(questions)}] {qid}: {record['status']}", flush=True)
    write_records(prediction_path, records)
    return int(any(r["status"] != "ok" for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
