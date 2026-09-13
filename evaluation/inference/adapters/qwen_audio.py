"""Split Qwen2-Audio input into complete, ordered audio segments."""

from __future__ import annotations

import base64
import io
import wave


# Qwen2-Audio's official feature extractor processes 30 seconds at 16 kHz.
# https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct/blob/main/preprocessor_config.json
SEGMENT_SECONDS = 30


def segment_audio(block, metadata):
    """Send the entire waveform as ordered <=30-second parts in one request.

    Splitting prevents the feature extractor from truncating a long waveform.
    The serving engine must permit the resulting number of audio parts and
    enough context; an API rejection is never replaced by a shortened input.
    """
    audio = block["input_audio"]
    if audio.get("format") != "wav":
        raise ValueError("Qwen2-Audio segmentation requires PCM16 WAV audio")
    raw = base64.b64decode(audio["data"], validate=True)
    parts, segments = [], []
    with wave.open(io.BytesIO(raw), "rb") as source:
        rate, count = source.getframerate(), source.getnframes()
        if (
            rate != 16000
            or source.getnchannels() != 1
            or source.getsampwidth() != 2
            or source.getcomptype() != "NONE"
            or count < 1
        ):
            raise ValueError("Qwen2-Audio requires non-empty 16 kHz mono PCM16 audio")
        step = rate * SEGMENT_SECONDS
        for start in range(0, count, step):
            stop = min(start + step, count)
            frames = source.readframes(stop - start)
            if len(frames) != (stop - start) * 2:
                raise ValueError("incomplete audio waveform")
            output = io.BytesIO()
            with wave.open(output, "wb") as target:
                target.setnchannels(1)
                target.setsampwidth(2)
                target.setframerate(rate)
                target.writeframes(frames)
            segment = output.getvalue()
            timing = {"start_seconds": start / rate, "end_seconds": stop / rate}
            parts.extend(
                [
                    {
                        "type": "text",
                        "text": f"Audio at {start / rate:.3f}–{stop / rate:.3f} seconds:",
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(segment).decode("ascii"),
                            "format": "wav",
                        },
                    },
                ]
            )
            segments.append(timing)
    return parts, {
        **metadata,
        "segment_seconds": SEGMENT_SECONDS,
        "segments": segments,
    }
