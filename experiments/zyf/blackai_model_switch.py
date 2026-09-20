"""Run the unresolved BlackAI media windows with a working model, preserving E11/E12."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from evaluation.io_utils import load_questions, load_records, write_json
from experiments.zyf.blackai_continuation import (
    BLACKAI, MATRIX, aggregate, atomic, backend, export_video, read,
    recorded_video_block,
)
from methods.longemo.common import file_hash, manifest, code_hash

NAME = "blackai_gemini37_matrix_gpt6_compact_20260920"
MODEL = "gemini-3.7-flash"
PARENT = "blackai_gemini38_matrix_gpt6_retry_20260920"
MAX_FRAMES = 16
MAX_PIXELS = 150528
MAX_TOKENS = 8192


def clone_checkpoint(source, target):
    """Record inherited 3.8 observations without relabelling them as 3.7."""
    receipt_path = target / "checkpoint_source.json"
    if target.exists():
        assert receipt_path.exists(), "unknown existing checkpoint"
        return read(receipt_path)
    memory = read(source / "memory.json") if (source / "memory.json").exists() else {"completed_windows": []}
    assert not memory.get("complete"), "only incomplete memories may continue"
    assert not any(p.is_symlink() for p in source.rglob("*"))
    files = {str(p.relative_to(source)): file_hash(p)
             for p in source.rglob("*") if p.is_file()}
    stage = target.with_name(target.name + ".staging")
    assert not stage.exists(), "unfinished checkpoint copy needs inspection"
    shutil.copytree(source, stage)
    assert files == {str(p.relative_to(stage)): file_hash(p)
                     for p in stage.rglob("*") if p.is_file()}
    # Preserve the old manifest as provenance. The runner writes a new manifest
    # for NEW windows; the experiment explicitly contains inherited 3.8 windows.
    (stage / "manifest.json").rename(stage / "inherited_manifest.json")
    completed = set(memory["completed_windows"])
    for p in (stage / "audio").glob("W*.json"):
        if p.stem not in completed:
            archive = stage / "inherited_pending_audio" / p.name
            archive.parent.mkdir(exist_ok=True)
            p.rename(archive)
    receipt = {"parent_run": PARENT, "files": files,
               "inherited_windows": memory["completed_windows"],
               "new_window_model": MODEL, "new_media": {
                   "max_frames": MAX_FRAMES, "max_pixels": MAX_PIXELS}}
    write_json(stage / "checkpoint_source.json", receipt)
    stage.rename(target)
    return receipt


def setup(runtime: Path, selection: Path, *, prepare_frontend=False):
    selected = read(selection)
    all_questions = load_questions(runtime / "data/questions.json", "episode")
    qmap = {q["question_id"]: q for q in all_questions}
    ids = selected["question_ids"]
    assert len(ids) == len(set(ids)) and set(ids) <= set(qmap)
    questions = [qmap[qid] for qid in ids]
    out = runtime / "runs" / NAME
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "protocol": NAME,
        "method_hash": code_hash(),
        "parent_run": PARENT,
        "source_fingerprint": selected["source_fingerprint"],
        "selection_sha256": file_hash(selection),
        "full_question_count": len(all_questions),
        "target_question_count": len(questions),
        "questions_sha256": file_hash(runtime / "data/questions.json"),
        "data_revision": read(runtime / "data/manifest.json")["revision"],
        "perception": {
            "model": MODEL, "base_url": BLACKAI, "audio_thinking": "low",
            "visual_thinking": "medium", "visual_max_tokens": MAX_TOKENS,
            "temperature": 1,
        },
        "answer": {"model": "gpt-6-astra", "base_url": MATRIX, "max_tokens": 8192},
        "embedding": {"model": "google/gemini-embedding-2",
                      "base_url": "https://openrouter.ai/api/v1"},
        "media": {"window_seconds": 20, "padding": 2, "fps": 1,
                  "max_frames": MAX_FRAMES, "max_pixels": MAX_PIXELS},
        "retrieval": {"top_k": 12, "evidence_chars": 48000, "max_inspections": 0},
        "scoring": "unchanged official scorer, earliest valid score retained",
        "scope": "95 unscored E11 questions: inherited Gemini 3.8 windows plus new Gemini 3.7 compact windows; mixed-model recovery, not a homogeneous benchmark run",
        "cache_policy": "Preserve inherited manifests and checksums; new audio for every unfinished window; no successful score replay",
    }
    manifest(out / "experiment_manifest.json", config)
    write_json(out / "questions.json", questions)
    vids = sorted({q["video_id"] for q in questions})
    old = runtime / "runs" / PARENT
    for vid in vids:
        write_json(out / "videos" / vid / "questions.json",
                   [q for q in questions if q["video_id"] == vid])
        source = old / "memory" / vid
        target = out / "memory" / vid
        if prepare_frontend:
            target.parent.mkdir(parents=True, exist_ok=True)
            clone_checkpoint(source, target)
    return out, questions


def canary(args, out):
    """Validate real audio and structured visual output before fan-out."""
    from evaluation.clients import Client
    from methods.longemo.audio import bridge_audio
    from methods.longemo.common import LoggedClient
    from methods.longemo.media import subtitles, window_input
    from methods.longemo.memory import apply_window
    from methods.longemo.prompts import PERCEPTION
    key = read(args.credential_file)["GEMINI_API_KEY"]
    vid = "G2_V000022"
    memory = read(out / "memory" / vid / "memory.json")
    window = "W00025"
    assert window not in memory["completed_windows"]
    directory = out / "canary"
    directory.mkdir(exist_ok=True)
    content, metadata = window_input(args.runtime / "data/episode/videos" / (vid + ".mp4"),
        478, 502, fps=1, max_frames=MAX_FRAMES, max_pixels=MAX_PIXELS, with_audio=True,
        subtitle_rows=subtitles(args.runtime / "data/prepared_subtitles" / (vid + ".json")))
    audio = Client(MODEL, BLACKAI, "gemini", key, 180, 4096, None,
        {"generationConfig": {"thinkingConfig": {"thinkingLevel": "low"}}})
    content, trace = bridge_audio(content, [478, 502], audio, directory / "audio/W00025.json", tries=3)
    metadata.update(audio_observer=trace, audio_representation="derived_timestamped_cues")
    context = {"video_id": vid, "window_id": window, "core_interval": [480, 500],
        "media_interval": [478, 502], "corrections_enabled": False,
        "cast": memory["entities"], "preceding_events": memory["events"][-3:]}
    visual = Client(MODEL, BLACKAI, "gemini", key, 240, MAX_TOKENS, 1,
        {"generationConfig": {"thinkingConfig": {"thinkingLevel": "medium"}}})
    def validate(payload):
        apply_window(memory, payload, window_id=window, core=[480, 500],
                     media=[478, 502], metadata=metadata, allow_revisions=False)
        assert payload.get("observations") and payload.get("events"), "canary needs real observations"
    payload = LoggedClient(visual, directory / "calls.jsonl", tries=3).call([
        {"role": "system", "content": PERCEPTION},
        {"role": "user", "content": [{"type": "text", "text": json.dumps(context)}] + content}],
        purpose="canary:" + vid + ":" + window, validate=validate)
    write_json(directory / "window.json", {"input": context, "sampling": metadata, "perception": payload})
    result = {"status": "ok", "time_unix": time.time(), "model": MODEL,
              "fingerprint": read(out / "experiment_manifest.json")["fingerprint"],
              "observations": len(payload["observations"]), "events": len(payload["events"])}
    atomic(directory / "result.json", result)
    print(json.dumps(result), flush=True)


def run_frontend(args, out, questions):
    env = dict(
        os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        LONGEMO_FFMPEG_COMPAT="1",
    )
    env.pop("LONGEMO_ALLOW_AUDIO_CACHE_REUSE", None)
    check = read(out / "canary/result.json")
    assert check["status"] == "ok" and check["fingerprint"] == read(out / "experiment_manifest.json")["fingerprint"]
    # Perception credentials must not be confused with Matrix's MODEL_API_KEY.
    credentials = read(args.credential_file)
    private = args.runtime / "private" / (NAME + "_perception.json")
    private.parent.mkdir(exist_ok=True)
    fd = os.open(private, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump({k: credentials[k] for k in ("GEMINI_API_KEY", "GOOGLE_GEMINI_BASE_URL")}, stream)
    compat = args.runtime / "tmp" / "ffmpeg_compat.py"
    if compat.exists():
        env["PATH"] = str(compat.parent) + os.pathsep + env.get("PATH", "")
        shim = compat.parent / "ffmpeg"
        if not shim.exists():
            shim.symlink_to(compat)
    setting = out / "perception_settings.json"
    write_json(setting, {"generationConfig": {"thinkingConfig": {"thinkingLevel": "medium"}}})
    vids = sorted({q["video_id"] for q in questions})
    state_path = out / "frontend_status.json"
    state = read(state_path) if state_path.exists() else {"videos": {}}

    def work(vid):
        folder = out / "videos" / vid
        root = folder / "build_memory"
        root.mkdir(exist_ok=True)
        memory = out / "memory" / vid
        link = root / vid
        if link.exists() or link.is_symlink():
            assert link.resolve() == memory.resolve()
        else:
            link.symlink_to(memory, target_is_directory=True)
        command = [
            sys.executable, "-u", "-m", "experiments.zyf.native_worker", "build",
            "--data-path", folder / "questions.json",
            "--videos-dir", args.runtime / "data/episode/videos",
            "--subtitles-dir", args.runtime / "data/prepared_subtitles",
            "--output-dir", root, "--window-seconds", 20, "--padding", 2,
            "--fps", 1, "--max-frames", MAX_FRAMES, "--max-pixels", MAX_PIXELS,
            "--with-audio", "--workers", 1, "--model", MODEL, "--base-url", BLACKAI,
            "--audio-model", MODEL, "--audio-base-url", BLACKAI,
            "--credential-file", private, "--thinking", "default",
            "--timeout", 240, "--tries", 3, "--max-tokens", MAX_TOKENS,
            "--temperature", 1, "--config", setting,
        ]
        write_json(folder / "build_command.json", {"command": list(map(str, command)),
                                                     "time_unix": time.time()})
        with (folder / "build.log").open("a") as log:
            rc = subprocess.run(list(map(str, command)), env=env, stdout=log,
                                stderr=subprocess.STDOUT).returncode
        complete = (memory / "memory.json").exists() and read(memory / "memory.json").get("complete")
        block = recorded_video_block(folder, memory)
        result = {"video_id": vid, "status": "memory_complete" if rc == 0 and complete
                  else block["status"] if block else "frontend_failed", "returncode": rc}
        if block:
            result["rejection"] = block["rejection"]
        for p in (memory / "calls.jsonl", memory / "audio/calls.jsonl"):
            rows = load_records(p) if p.exists() else []
            if rows and rows[-1].get("model") == MODEL and rows[-1].get("http_status") in (401, 402, 403):
                result.update(status="blocked_provider", http_status=rows[-1]["http_status"])
        export_video(out, vid, result)
        return result

    todo = [v for v in vids if v not in state["videos"]]
    active = {}; stop = False
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while todo or active:
            while todo and len(active) < args.workers and not stop:
                vid = todo.pop(0); active[pool.submit(work, vid)] = vid
            state.update(status="paused" if stop else "running", active_videos=list(active.values()),
                         queued_videos=list(todo), updated_unix=time.time())
            atomic(state_path, state)
            if not active: break
            done, _ = wait(active, timeout=10, return_when=FIRST_COMPLETED)
            for future in done:
                vid = active.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"video_id": vid, "status": "orchestration_error", "error_type": type(exc).__name__}
                state["videos"][vid] = result
                stop = stop or result["status"] in ("blocked_provider", "orchestration_error")
                print(json.dumps(result), flush=True)
    state.update(status="paused" if todo else "finished", queued_videos=todo,
                 active_videos=[], updated_unix=time.time())
    atomic(state_path, state)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("canary", "frontend", "backend"))
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--credential-file", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    assert 1 <= args.workers <= 8
    out, questions = setup(args.runtime, args.selection, prepare_frontend=False)
    with (out / (args.stage + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage in ("canary", "frontend"):
            setup(args.runtime, args.selection, prepare_frontend=True)
        atomic(out / (args.stage + "_process.json"), {"pid": os.getpid(), "started_unix": time.time()})
        if args.stage == "canary":
            canary(args, out)
        elif args.stage == "frontend":
            run_frontend(args, out, questions)
        elif args.stage == "backend":
            # The remote backend receives bundles; it needs no parent checkpoint.
            for r in ("matrix_gemini38_gpt6_full_v1", "blackai_gemini38_matrix_gpt6_remaining_v1"):
                p = args.runtime / "runs" / r / "scores.jsonl"
                assert p.exists(), "source score audit missing"
                successes = {row["question_id"] for row in load_records(p) if row.get("status") == "ok"}
                assert not successes.intersection(q["question_id"] for q in questions)
            inbox = out / "inbox"
            inbox.mkdir(exist_ok=True)
            for marker in (out / "outbox").glob("*.ready.json"):
                archive = out / "outbox" / read(marker)["archive"]
                shutil.copy2(archive, inbox / archive.name)
                shutil.copy2(marker, inbox / marker.name)
            backend(args, out, questions)


if __name__ == "__main__":
    main()
