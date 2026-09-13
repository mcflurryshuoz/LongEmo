"""Run local multimodal checkpoints directly with Transformers.

This entry point is intentionally separate from ``evaluation.inference.run``:
It dispatches Qwen checkpoints to their official processor paths and dispatches
InternVL checkpoints to the official frame-based ``model.chat`` path. Omni
checkpoints receive a native
video and, by default, its embedded audio is enabled with
``use_audio_in_video=True``; VL checkpoints use visual video only and Audio
checkpoints receive an extracted WAV.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import json
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "evaluation.inference"

from ..io_utils import load_questions, subtitle_text, write_records
from .prompts import build_messages


OMNI_MODEL_CLASSES = (
    "Qwen3_5OmniMoeForConditionalGeneration",
    "Qwen3OmniMoeForConditionalGeneration",
    "Qwen2_5OmniForConditionalGeneration",
)
OMNI_PROCESSOR_CLASSES = (
    "Qwen3_5OmniMoeProcessor",
    "Qwen3OmniMoeProcessor",
    "Qwen2_5OmniProcessor",
)


def _dependencies(family):
    """Import heavy direct-inference dependencies only when this command runs."""
    try:
        import torch
        import transformers
        if family == "omni":
            from qwen_omni_utils import process_mm_info
        elif family == "vl":
            from qwen_vl_utils import process_vision_info as process_mm_info
        else:
            from qwen_omni_utils import process_audio_info as process_mm_info
    except ImportError as exc:  # pragma: no cover - depends on model environment
        package = {
            "omni": "qwen-omni-utils",
            "vl": "qwen-vl-utils",
            "audio": "qwen-omni-utils",
        }[family]
        raise RuntimeError(
            f"Qwen {family} Transformers inference requires torch, transformers and "
            f"{package}; install the versions required by the selected checkpoint"
        ) from exc
    return torch, transformers, process_mm_info


def detect_family(model_path):
    """Infer the native Qwen family from a checkpoint name."""
    name = str(model_path).lower().rsplit("/", 1)[-1]
    if "audio" in name:
        return "audio"
    if "-vl" in name or name.endswith("vl") or "_vl" in name:
        return "vl"
    if "omni" in name:
        return "omni"
    raise ValueError(
        "cannot infer a Qwen family from --model; use a checkpoint name containing "
        "Omni, VL, or Audio"
    )


def _resolve_class(module, names, label):
    for name in names:
        value = getattr(module, name, None)
        if value is not None:
            return value
    raise RuntimeError(
        f"installed transformers does not expose a supported Qwen Omni {label}; "
        f"expected one of {', '.join(names)}"
    )


def _dtype(torch, value):
    if value == "auto":
        return "auto"
    return getattr(torch, value)


def load_model(model_path, *, family, device_map="auto", dtype="auto", attn=None):
    torch, transformers, process_mm_info = _dependencies(family)
    if family == "omni":
        model_names = OMNI_MODEL_CLASSES
        processor_names = OMNI_PROCESSOR_CLASSES
    elif family == "vl":
        model_names = (
            "Qwen3VLForConditionalGeneration",
            "Qwen2_5_VLForConditionalGeneration",
            "Qwen2VLForConditionalGeneration",
        )
        processor_names = ()
    else:
        model_names = ("Qwen2AudioForConditionalGeneration",)
        processor_names = ()
    model_cls = _resolve_class(transformers, model_names, "model class")
    kwargs = {"torch_dtype": _dtype(torch, dtype), "device_map": device_map}
    if attn:
        kwargs["attn_implementation"] = attn
    model = model_cls.from_pretrained(model_path, **kwargs)
    processor = (
        _resolve_class(transformers, processor_names, "processor class").from_pretrained(model_path)
        if processor_names
        else transformers.AutoProcessor.from_pretrained(model_path)
    )
    model.eval()
    return {"torch": torch, "processor": processor, "model": model,
            "process_mm_info": process_mm_info, "family": family}


def _video_path(video_dir, question):
    path = Path(video_dir) / f"{question['video_id']}.mp4"
    if not path.is_file() or not path.stat().st_size:
        raise ValueError(f"missing or empty prepared video: {path}")
    return path.resolve()


def _question_text(question, with_subtitle):
    # Let the shared benchmark prompt builder provide the same task-specific
    # answer format as the API entry point, while omitting reference fields.
    messages = build_messages(
        question,
        subtitle_text=subtitle_text(question, required=False) if with_subtitle else None,
        prompt_mode="system",
    )
    return messages[0]["content"], messages[-1]["content"]


def _generate(runtime, question, video_path, *, with_subtitle, use_audio, max_new_tokens):
    torch = runtime["torch"]
    processor = runtime["processor"]
    model = runtime["model"]
    process_mm_info = runtime["process_mm_info"]
    family = runtime["family"]
    system_text, user_text = _question_text(question, with_subtitle)
    if family == "audio":
        with tempfile.TemporaryDirectory(prefix="longemobench-qwen-audio-") as temp:
            audio_path = Path(temp) / "input.wav"
            _extract_audio(video_path, audio_path)
            return _generate_audio(
                runtime,
                system_text,
                user_text,
                audio_path,
                max_new_tokens=max_new_tokens,
            )
    media_type = "video" if family in {"omni", "vl"} else "audio"
    native_messages = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [
            {"type": media_type, "video" if media_type == "video" else "audio": str(video_path)},
            {"type": "text", "text": user_text},
        ]},
    ]
    text = processor.apply_chat_template(
        native_messages, add_generation_prompt=True, tokenize=False
    )
    if family == "omni":
        audios, images, videos = process_mm_info(
            native_messages, use_audio_in_video=use_audio
        )
        inputs = processor(text=text, audio=audios, images=images, videos=videos,
                           return_tensors="pt", padding=True,
                           use_audio_in_video=use_audio)
    else:
        images, videos = process_mm_info(native_messages)
        inputs = processor(text=[text], images=images, videos=videos,
                           return_tensors="pt", padding=True)
    # ``BatchFeature.to`` moves tensors without converting input IDs to a
    # floating dtype.  With device_map=auto, the model handles placement.
    if getattr(model, "device", None) is not None and str(model.device) != "meta":
        inputs = inputs.to(model.device)
    generate_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
    if family == "omni":
        generate_kwargs.update({"use_audio_in_video": use_audio, "return_audio": False})
    with torch.inference_mode():
        generated = model.generate(**inputs, **generate_kwargs)
    if isinstance(generated, tuple):
        generated = generated[0]
    input_ids = inputs.input_ids
    trimmed = [out[len(src) :] for src, out in zip(input_ids, generated)]
    answer = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    return answer, {
        "representation": "native_video",
        "audio": "embedded_in_video" if family == "omni" and use_audio else "not_used",
        "family": family,
        "model_class": type(model).__name__,
        "processor_class": type(processor).__name__,
    }


def _extract_audio(video_path, output):
    """Extract a temporary WAV for Qwen-Audio's native processor."""
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video_path),
             "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-y", str(output)],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Qwen-Audio direct inference requires ffmpeg and an audio track") from exc


def _generate_audio(runtime, system_text, user_text, audio_path, *, max_new_tokens):
    torch = runtime["torch"]
    processor = runtime["processor"]
    model = runtime["model"]
    process_audio_info = runtime["process_mm_info"]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [
            {"type": "audio", "audio_url": str(audio_path)},
            {"type": "text", "text": user_text},
        ]},
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    audios = process_audio_info(messages)
    inputs = processor(text=text, audio=audios, return_tensors="pt", padding=True)
    if getattr(model, "device", None) is not None and str(model.device) != "meta":
        inputs = inputs.to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    input_ids = inputs.input_ids
    trimmed = [out[len(src):] for src, out in zip(input_ids, generated)]
    answer = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    return answer, {"representation": "audio", "audio": "extracted_wav", "family": "audio",
                    "model_class": type(model).__name__, "processor_class": type(processor).__name__}


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--data-path", required=True, help="Question JSON/JSONL file or directory")
    p.add_argument("--model", required=True, help="Local Qwen or InternVL checkpoint")
    p.add_argument("-g", "--granularity", choices=("clip", "episode"), default="clip")
    p.add_argument("--videos-dir", help="Directory containing <video_id>.mp4; defaults beside data-path")
    p.add_argument("--output-dir", required=True, help="Prediction output directory")
    p.add_argument("--qid", action="append", help="Question ID; may be repeated")
    p.add_argument("--limit", type=int, help="Run the first N selected questions")
    p.add_argument("--with-subtitle", action="store_true", help="Append clip subtitles to the question")
    p.add_argument("--max-new-tokens", type=int, default=16384)
    p.add_argument("--device-map", default="auto", help="Transformers device map, e.g. auto or cuda:0")
    p.add_argument("--torch-dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    p.add_argument("--attention", default=None, help="Optional attention implementation, e.g. flash_attention_2")
    p.add_argument("--force", action="store_true", help="Recompute predictions in the output directory")
    return p


def _model_argument(argv):
    values = list(argv) if argv is not None else __import__("sys").argv[1:]
    for value in values:
        if value.startswith("--model="):
            return value.split("=", 1)[1].lower()
    for i, value in enumerate(values[:-1]):
        if value == "--model":
            return values[i + 1].lower()
    return ""


def main(argv=None):
    if "internvl" in _model_argument(argv):
        from .adapters.internvl import main as internvl_main

        return internvl_main(argv)
    args = parser().parse_args(argv)
    if args.max_new_tokens < 1:
        parser().error("--max-new-tokens must be positive")
    if args.limit is not None and args.limit < 1:
        parser().error("--limit must be positive")
    data = Path(args.data_path)
    family = detect_family(args.model)
    videos_dir = Path(args.videos_dir) if args.videos_dir else (data if data.is_dir() else data.parent) / "videos"
    questions = load_questions(args.data_path, args.granularity, args.qid, limit=args.limit)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    runtime = load_model(args.model, family=family, device_map=args.device_map, dtype=args.torch_dtype, attn=args.attention)
    records = []
    prediction_path = out / "predictions.jsonl"
    old = {}
    if prediction_path.exists() and not args.force:
        for line in prediction_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                old[item.get("question_id")] = item
    for index, question in enumerate(questions, 1):
        qid = question["question_id"]
        path = _video_path(videos_dir, question)
        if qid in old and old[qid].get("status") == "ok" and not args.force:
            records.append(old[qid])
            continue
        base = {"question_id": qid, "video_id": question["video_id"], "granularity": question["granularity"], "type": question["type"]}
        try:
            use_audio = family in {"omni", "audio"}
            answer, media = _generate(runtime, question, path, with_subtitle=args.with_subtitle, use_audio=use_audio, max_new_tokens=args.max_new_tokens)
            record = {**base, "status": "ok", "prediction": answer, "raw_answer": answer, "media": media, "error": None}
        except Exception as exc:  # preserve per-question failures for long runs
            record = {**base, "status": "error", "prediction": None, "raw_answer": None, "media": None, "error": f"{type(exc).__name__}: {exc}"}
        records.append(record)
        write_records(prediction_path, records)
        print(f"[{index}/{len(questions)}] {qid}: {record['status']}", flush=True)
    write_records(prediction_path, records)
    return int(any(r["status"] != "ok" for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
