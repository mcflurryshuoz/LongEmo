"""Infer clip and episode questions from video, audio or subtitle text."""

from __future__ import annotations
import argparse
import math
import re
from pathlib import Path
from urllib.parse import urlparse

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "evaluation.inference"

from ..io_utils import load_questions, subtitle_text
from ..io_utils import question_key
from .prompts import (
    build_messages,
    text_messages,
)
from .prompts import parse_prediction
from .runner import model_args, init_client, retry, execute_tasks
from .video_loader import (
    FPS,
    MAX_FRAMES,
    VIDEO_MIN_PIXELS,
    VIDEO_MAX_PIXELS,
    encode_video,
    sample_video_frames,
    extract_audio,
)
from .adapters import qwen_audio


# Limits for independently transmitted image frames. Native-video models use
# their own video-token limits and do not use this table unless frames are
# explicitly requested.
MODEL_FRAME_CAPS = {
    "gpt": 1500,
    "claude": 600,
    "deepseek": 600,
    "seed": 1280,
}

# Sampling defaults, distinct from the API image-count limits above.
FRAME_DEFAULTS = {
    "fps": FPS,
    "max_frames": MAX_FRAMES,
    "frame_max_pixels": VIDEO_MAX_PIXELS,
    "total_pixels": None,
}
MODEL_FRAME_DEFAULTS = {
    "gpt": {
        "fps": 1.0,
        "max_frames": 128,
        "frame_max_pixels": 512 * 512,
        "total_pixels": 32 * 1024 * 1024,
    },
    # Largest successful AICodeMirror probe; uploads can still fail on retry.
    "claude": {
        "fps": 2.0,
        "max_frames": 80,
        "frame_max_pixels": 512 * 512,
        "total_pixels": 80 * VIDEO_MIN_PIXELS,
    },
    # Verified on DeepSeek's official API with 600 frames at 1008 x 560.
    "deepseek": {
        "fps": 2.0,
        "max_frames": 600,
        "frame_max_pixels": VIDEO_MAX_PIXELS,
        "total_pixels": 600 * VIDEO_MAX_PIXELS,
    },
}


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument(
        "--data-path",
        required=True,
        help="Path to a question JSON/JSONL file or a flat question directory",
    )
    p.add_argument("-g", "--granularity", choices=("clip", "episode"), default="clip")
    p.add_argument(
        "--modality",
        choices=("video", "audio", "text"),
        default="video",
        help="Input evidence: video, audio extracted from the video, or subtitles only",
    )
    p.add_argument(
        "--videos-dir",
        dest="videos_dir",
        help="Prepared <video_id>.mp4 files; default: videos under the data directory or next to the data file",
    )
    p.add_argument(
        "--output-dir", help="Output directory"
    )
    p.add_argument(
        "--prompt-mode",
        choices=("system", "user"),
        default="system",
        help="Place common instructions in a system or user message",
    )
    p.add_argument(
        "--with-audio",
        action="store_true",
        help="Attach a separate audio track when sampled frames are used",
    )
    p.add_argument(
        "--with-subtitle",
        action="store_true",
        help="Append subtitles to video or audio inputs; default off",
    )
    p.add_argument(
        "--sample-frames",
        action="store_true",
        help="Use timestamped sampled frames instead of native video",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Target sampling FPS; uses the model's sampling default when omitted",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum sampled frames across the complete video; uses the model default",
    )
    p.add_argument(
        "--frame-max-pixels",
        type=int,
        default=None,
        help="Per-frame pixel-area cap; uses the model default",
    )
    p.add_argument(
        "--total-pixels",
        type=int,
        default=None,
        help="Total pixel cap across sampled frames; GPT, Claude and DeepSeek use model budgets by default",
    )
    p.add_argument(
        "--qid", action="append", help="Select a question ID; may be repeated"
    )
    p.add_argument(
        "--limit",
        type=int,
        help="Run the first N questions after granularity and question-ID filtering; default all",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run selected questions instead of reusing successful predictions",
    )
    p.add_argument("--tries", type=int, default=3)
    p.add_argument("--workers", type=int, default=1)
    model_args(p)
    return p


def run(args):
    if not args.model:
        raise ValueError("--model is required")
    if args.modality != "text" and not args.videos_dir:
        data_path = Path(args.data_path)
        data_dir = data_path if data_path.is_dir() else data_path.parent
        args.videos_dir = str(data_dir / "videos")
    client = init_client(args)
    defaults = FRAME_DEFAULTS | MODEL_FRAME_DEFAULTS.get(args.model_family, {})
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.model_family == "qwen_audio" and args.modality != "audio":
        raise ValueError("Qwen2-Audio uses audio input; select --modality audio")
    if args.modality == "audio" and args.sample_frames:
        raise ValueError("--sample-frames requires --modality video")
    video_mode = None
    model_frame_cap = MODEL_FRAME_CAPS.get(args.model_family)
    max_frames = (
        min(args.max_frames, model_frame_cap)
        if model_frame_cap is not None
        else args.max_frames
    )
    if args.modality == "video":
        google_chat = (
            args.api_format == "chat"
            and urlparse(args.base_url).hostname == "generativelanguage.googleapis.com"
        )
        video_mode = (
            "frames"
            if args.sample_frames
            or args.granularity == "episode"
            or args.model_family
            in {"gpt", "claude", "glm", "internvl", "deepseek"}
            or args.api_format in {"responses", "anthropic"}
            or google_chat
            else "native"
        )
        if video_mode == "frames" and not (
            math.isfinite(args.fps)
            and args.fps > 0
            and args.max_frames >= 1
            and args.frame_max_pixels >= VIDEO_MIN_PIXELS
        ):
            raise ValueError(
                "frame sampling requires a finite positive fps, a positive max_frames, "
                f"and frame_max_pixels >= {VIDEO_MIN_PIXELS}"
            )
    # Native mode always sends the original video unchanged, including its
    # embedded soundtrack. A separate track is only meaningful for sampled
    # frames, where no audio is present in the image sequence.
    extra_audio = (
        args.modality == "video" and video_mode == "frames" and args.with_audio
    )
    questions = load_questions(
        args.data_path, args.granularity, args.qid, limit=args.limit
    )
    prompt_mode = args.prompt_mode
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", args.model)
    out = Path(args.output_dir or f"output/{args.granularity}/{args.modality}/{slug}")
    prepared, tasks = {}, []
    for q in questions:
        key = question_key(q)
        try:
            record = {"question": q, "subtitle_text": None, "video": None}
            if args.modality == "text" or args.with_subtitle:
                record["subtitle_text"] = subtitle_text(
                    q, required=args.modality == "text"
                )
            if args.modality != "text":
                path = (Path(args.videos_dir) / f"{q['video_id']}.mp4").resolve()
                if not path.is_file() or not path.stat().st_size:
                    raise ValueError(f"missing or empty prepared video: {path}")
                record["video"] = {"path": str(path)}
        except (OSError, ValueError) as exc:
            record = {"error": str(exc)}
        prepared[key] = record
        base = {
            "question_id": q["question_id"],
            "video_id": q["video_id"],
            "granularity": q["granularity"],
            "type": q["type"],
        }
        tasks.append((key, base))

    def generate_prediction(task):
        record = prepared[task[0]]
        if "error" in record:
            return {
                "status": "input_error",
                "prediction": None,
                "error": record["error"],
            }
        q = record["question"]
        media_metadata = None
        if args.modality == "text":
            messages = text_messages(
                q, record["subtitle_text"], prompt_mode=prompt_mode
            )
        else:
            video = record["video"]
            if args.modality == "audio":
                audio, audio_metadata = extract_audio(video, required=True)
                media = [
                    {
                        "type": "text",
                        "text": "Audio only; no video frames are provided.",
                    }
                ]
                if args.model_family == "qwen_audio":
                    audio_parts, audio_metadata = qwen_audio.segment_audio(
                        audio, audio_metadata
                    )
                    media.extend(audio_parts)
                else:
                    media.append(audio)
                media_metadata = {
                    "representation": "audio_only",
                    "audio": audio_metadata["audio"],
                    "audio_details": audio_metadata,
                }
            elif video_mode == "frames":
                media, media_metadata = sample_video_frames(
                    video,
                    fps=args.fps,
                    max_frames=max_frames,
                    max_pixels=args.frame_max_pixels,
                    total_pixels=args.total_pixels,
                )
                media_metadata.update({"representation": "frames", "audio": "not_sent"})
            else:
                media = [encode_video(video)]
                media_metadata = {
                    "representation": "native_video",
                    "audio": "in_video",
                }
            media_metadata["with_audio"] = extra_audio
            if extra_audio:
                audio, audio_metadata = extract_audio(video, required=False)
                media_metadata["audio"] = audio_metadata["audio"]
                media_metadata["audio_details"] = audio_metadata
                if audio is not None:
                    media += [
                        {
                            "type": "text",
                            "text": "Audio from the same video; time zero matches the video timeline.",
                        },
                        audio,
                    ]
            messages = build_messages(
                q,
                subtitle_text=record["subtitle_text"],
                prompt_mode=prompt_mode,
                media_content=media,
            )
        response = retry(lambda: client.generate(messages), args.tries)
        prediction, parse_error = parse_prediction(q, response["content"])
        # Store the prompt and media references, not another copy of the input video.
        recorded_messages = []
        frame_index = 0
        audio_index = 0
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                recorded_content = []
                for x in content:
                    if x["type"] == "text":
                        recorded_content.append(x)
                    elif x["type"] == "image_url":
                        timestamp = media_metadata["timestamps_seconds"][frame_index]
                        recorded_content.append(
                            {
                                "type": "frame_reference",
                                **record["video"],
                                "timestamp_seconds": timestamp,
                            }
                        )
                        frame_index += 1
                    elif x["type"] == "input_audio":
                        details = media_metadata["audio_details"]
                        if "segments" in details:
                            details = {
                                "sample_rate": details["sample_rate"],
                                "channels": details["channels"],
                                **details["segments"][audio_index],
                            }
                        audio_index += 1
                        recorded_content.append(
                            {
                                "type": "audio_reference",
                                **record["video"],
                                "audio_details": details,
                            }
                        )
                    else:
                        recorded_content.append(
                            {"type": "video_reference", **record["video"]}
                        )
                content = recorded_content
            recorded_messages.append({"role": message["role"], "content": content})
        return {
            "status": "ok",
            "prediction": prediction,
            "raw_answer": response["content"],
            "raw_response": response["raw_response"],
            "usage": response["usage"],
            "messages": recorded_messages,
            "parse_error": parse_error,
            "media": media_metadata,
            "error": None,
        }

    results = execute_tasks(
        tasks,
        generate_prediction,
        out / "predictions.jsonl",
        workers=args.workers,
        force=args.force,
    )
    return int(any(r["status"] != "ok" for r in results))


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        return run(args)
    except (ValueError, OSError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
