"""Bounded alternate-provider validation of unfinished media; no benchmark scores."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import time
from urllib.parse import urlparse

from evaluation.clients import Client, ServiceError
from evaluation.inference.adapters import gemini
from evaluation.io_utils import write_json
from experiments.zyf.blackai_continuation import read, atomic
from methods.longemo.audio import AUDIO_PROMPT
from methods.longemo.common import LoggedClient, file_hash, fingerprint
from methods.longemo.media import probe, subtitles, window_input
from methods.longemo.memory import apply_window, empty_memory, span, text
from methods.longemo.prompts import PERCEPTION

BASE = "https://api.aicodemirror.ai/api/gemini/v1beta"
MODEL = "gemini-3.7-flash"
PARENT = "blackai_gemini37_matrix_gpt6_compact_20260920"
ORIGINAL_HEADERS = gemini.headers


def mirror_headers(client):
    if urlparse(client.base_url).hostname != "api.aicodemirror.ai":
        return ORIGINAL_HEADERS(client)
    return {"Content-Type": "application/json", "Authorization": "Bearer " + client.api_key}


def probe_video(runtime, directory, key, vid):
    parent = runtime / "runs" / PARENT
    source = parent / "memory" / vid
    folder = directory / vid
    folder.mkdir(exist_ok=True)
    if (folder / "result.json").exists():
        return read(folder / "result.json")
    video = runtime / "data/episode/videos" / (vid + ".mp4")
    info = probe(video)
    memory = read(source / "memory.json") if (source / "memory.json").exists() else empty_memory(vid, info["duration"], file_hash(video))
    assert not memory.get("complete")
    i = next(i for i in range(math.ceil(info["duration"] / 20)) if f"W{i+1:05d}" not in memory["completed_windows"])
    wid = f"W{i+1:05d}"
    core = [i*20, min((i+1)*20, info["duration"])]
    interval = [max(0,core[0]-2),min(info["duration"],core[1]+2)]
    cached_audio = source / "audio" / (wid + ".json")
    stage = "visual" if cached_audio.exists() else "audio"
    result = {"video_id": vid, "window_id": wid, "stage": stage,
        "core_interval": core, "media_interval": interval, "provider": "AICodeMirror",
        "model": MODEL, "base_url": BASE, "attempts": 1, "time_unix": time.time(),
        "scope": "Independent diagnostic of first unfinished window; does not complete a video or add scores"}
    try:
        content, metadata = window_input(video, *interval, fps=1, max_frames=16,
            max_pixels=150528, with_audio=(stage == "audio"),
            subtitle_rows=subtitles(runtime / "data/prepared_subtitles" / (vid + ".json")))
        if stage == "audio":
            audio = [part for part in content if part["type"] == "input_audio"]
            messages = [{"role": "system", "content": AUDIO_PROMPT}, {"role": "user", "content": [
                {"type": "text", "text": json.dumps({"clip_start": interval[0], "clip_end": interval[1]})}] + audio}]
            client = Client(MODEL, BASE, "gemini", key, 180, 4096, None,
                {"generationConfig": {"thinkingConfig": {"thinkingLevel": "low"}}})
            def validate(payload):
                if not isinstance(payload, dict) or not isinstance(payload.get("observations"), list):
                    raise ValueError("audio observations must be a list")
                for observation in payload["observations"]:
                    span(observation.get("span"), interval)
                    for field in ("voice", "cue"): text(observation.get(field), field)
        else:
            old_audio = read(cached_audio)
            metadata.update(audio=True, audio_representation="derived_timestamped_cues",
                            audio_observer={"model": old_audio["model"]["model"],
                                            "source_sha256": file_hash(cached_audio)})
            content += [{"type": "text", "text": "Timestamped audio observations from an independent audio model; "
                "these may contain errors. Match voice identity cautiously using the frames and dialogue. " + json.dumps(old_audio["result"], ensure_ascii=False)}]
            result["audio_cache_sha256"] = file_hash(cached_audio)
            context = {"video_id": vid, "window_id": wid, "core_interval": core, "media_interval": interval,
                "corrections_enabled": False, "cast": memory["entities"], "preceding_events": memory["events"][-3:]}
            messages = [{"role": "system", "content": PERCEPTION}, {"role": "user", "content": [
                {"type": "text", "text": json.dumps(context, ensure_ascii=False)}] + content}]
            client = Client(MODEL, BASE, "gemini", key, 240, 8192, 1,
                {"generationConfig": {"thinkingConfig": {"thinkingLevel": "medium"}}})
            def validate(payload):
                apply_window(memory, payload, window_id=wid, core=core, media=interval,
                             metadata=metadata, allow_revisions=False)
        result["request_hash"] = fingerprint(messages)
        # Retain exact validation error class and safe response metadata; no key/payload in stdout.
        response = client.generate(messages)
        raw = response.get("raw_response") or {}
        result.update(response_model=raw.get("modelVersion"), finish_reason=response.get("finish_reason"),
                      usage=response.get("usage"), text_characters=len(response["content"]))
        from evaluation.inference.prompts import json_object
        payload = json_object(response["content"])
        write_json(folder / "payload.json", payload)
        validate(payload)
        result.update(status="ok", observations=len(payload.get("observations", [])), events=len(payload.get("events", [])))
    except Exception as exc:
        result.update(status="error", error_type=type(exc).__name__)
        if isinstance(exc, ServiceError):
            result.update(http_status=exc.status_code, service_error_code=exc.code)
        if isinstance(exc, (ValueError, RuntimeError)) and not isinstance(exc, ServiceError):
            result["error"] = str(exc).replace(key, "<redacted>")[:160]
    result["elapsed_seconds"] = round(time.time()-result["time_unix"], 3)
    atomic(folder / "result.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--credential-file", type=Path, required=True)
    p.add_argument("--video-id", action="append")
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    key = read(args.credential_file)["GEMINI_API_KEY"]
    states = read(args.runtime / "runs" / PARENT / "frontend_status.json")["videos"]
    targets = sorted(vid for vid,state in states.items() if state["status"] != "memory_complete")
    if args.video_id:
        assert set(args.video_id) <= set(targets)
        targets = [vid for vid in targets if vid in args.video_id]
    directory = args.runtime / "diagnostics/aicodemirror_remaining_20260921"
    directory.mkdir(parents=True, exist_ok=True)
    gemini.headers = mirror_headers
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        fs = {pool.submit(probe_video,args.runtime,directory,key,vid):vid for vid in targets}
        for f in as_completed(fs):
            result = f.result(); rows.append(result)
            print(json.dumps({k:v for k,v in result.items() if k != "usage"}), flush=True)
            atomic(directory / "summary.json", {"updated_unix":time.time(),"targets":targets,"results":rows})


if __name__ == "__main__": main()
