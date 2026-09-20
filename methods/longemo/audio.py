"""Question-independent audio observations for image/text reasoning models."""
from __future__ import annotations

import json
import os
from pathlib import Path

from evaluation.clients import Client
from evaluation.io_utils import write_json
from .common import LoggedClient, fingerprint, file_hash
from .memory import span, text

AUDIO_PROMPT = """Observe only the supplied audio. No benchmark question or reference answer is provided.
Return JSON {observations:[{span:[start,end],voice:string,cue:string,uncertainty:string}]}.
Use absolute video seconds: add the supplied clip_start to times within the audio.
Transcribe relevant words faithfully and describe audible tone, volume, pace, laughter,
hesitation, interruptions and changes. Distinguish speech from audience laughter or music.
Voice is a local acoustic description, not an inferred actor/character identity.
Do not invent visuals, motives or psychological conclusions. Mark uncertain words or timing.
Cover the whole clip; an empty list is valid for silence. Observations are evidence hypotheses,
not reference annotations."""


def audio_client_for(args):
    is_gpt6 = "gpt-6" in (args.model or "")
    model = getattr(args, "audio_model", None)
    if is_gpt6 and args.with_audio and not model:
        raise ValueError("GPT-6 does not accept audio; --with-audio requires --audio-model")
    if not model:
        return None
    credentials = json.loads(Path(args.credential_file).read_text()) if args.credential_file else {}
    base_url = getattr(args, "audio_base_url", None)
    if model.startswith("gemini-"):
        if base_url and base_url.rstrip("/") == "https://matrixllm.alipay.com/v1":
            key = credentials.get("MODEL_API_KEY") or os.getenv("MODEL_API_KEY")
            if not key:
                raise ValueError("Matrix audio credential missing")
            return Client(model, base_url, "chat", key, 180, 4096, None,
                          {"reasoning_effort": "low"})
        if not base_url or not base_url.rstrip("/").endswith("/v1beta"):
            raise ValueError("native Gemini audio requires an explicit --audio-base-url ending in /v1beta")
        key = credentials.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("native Gemini audio credential missing")
        return Client(model, base_url, "gemini", key, 180, 4096, None,
                      {"generationConfig": {"thinkingConfig": {"thinkingLevel": "low"}}})
    if not model.startswith("google/"):
        raise ValueError("audio observer expects an explicit Gemini model ID")
    if base_url and base_url.rstrip("/") != "https://openrouter.ai/api/v1":
        raise ValueError("OpenRouter audio models cannot use a different provider URL")
    key = credentials.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("audio observer credential missing")
    return Client(model, "https://openrouter.ai/api/v1", "chat", key, 180, 4096, None,
                  {"reasoning": {"effort": "low"}})


def bridge_audio(content, interval, client, path, tries=3):
    audio = [part for part in content if part["type"] == "input_audio"]
    if not audio:
        return content, {"audio": "absent"}
    messages = [{"role": "system", "content": AUDIO_PROMPT}, {"role": "user", "content": [
        {"type": "text", "text": json.dumps({"clip_start": interval[0], "clip_end": interval[1]})}] + audio}]
    signature = fingerprint({"messages": messages, "model": client.configuration()})
    path = Path(path)

    def validate(value):
        if not isinstance(value, dict) or not isinstance(value.get("observations"), list):
            raise ValueError("audio observations must be a list")
        for observation in value["observations"]:
            span(observation.get("span"), interval)
            for field in ("voice", "cue"):
                text(observation.get(field), field)

    if path.exists():
        cached = json.loads(path.read_text())
        if cached["input_fingerprint"] != signature:
            if os.environ.get("LONGEMO_ALLOW_AUDIO_CACHE_REUSE") != "1":
                raise ValueError("audio observer cache configuration changed")
            # The isolated g450 retry uses an older FFmpeg compatibility shim;
            # its regenerated PCM may differ in container metadata while the
            # decoded clip and provider configuration remain unchanged.
        result = cached["result"]
        validate(result)
    else:
        api = LoggedClient(client, path.parent/"calls.jsonl", tries)
        result = api.call(messages, purpose="audio_observer:"+path.stem, validate=validate)
        write_json(path, {"input_fingerprint": signature, "model": client.configuration(), "result": result})
    replacement = {"type": "text", "text": "Timestamped audio observations from an independent audio model; "
                   "these may contain errors. Match voice identity cautiously using the frames and dialogue. " + json.dumps(result, ensure_ascii=False)}
    return [part for part in content if part["type"] != "input_audio"] + [replacement], {
        "model": client.model, "input_fingerprint": signature, "source_sha256": file_hash(path)}
