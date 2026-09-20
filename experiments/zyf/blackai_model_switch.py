"""Run the unresolved BlackAI media windows with a working model, preserving E11/E12."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from methods.longemo.common import file_hash, manifest

NAME = "blackai_gemini35_matrix_gpt6_switch_20260920"
MODEL = "gemini-3.5-flash"
PARENT = "blackai_gemini38_matrix_gpt6_retry_20260920"


def setup(runtime: Path, selection: Path):
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
        "parent_run": PARENT,
        "source_fingerprint": selected["source_fingerprint"],
        "selection_sha256": file_hash(selection),
        "full_question_count": len(all_questions),
        "target_question_count": len(questions),
        "questions_sha256": file_hash(runtime / "data/questions.json"),
        "data_revision": read(runtime / "data/manifest.json")["revision"],
        "perception": {
            "model": MODEL, "base_url": BLACKAI, "audio_thinking": "low",
            "visual_thinking": "medium", "visual_max_tokens": 32768,
            "temperature": 1,
        },
        "answer": {"model": "gpt-6-astra", "base_url": MATRIX, "max_tokens": 8192},
        "embedding": {"model": "google/gemini-embedding-2",
                      "base_url": "https://openrouter.ai/api/v1"},
        "media": {"window_seconds": 20, "padding": 2, "fps": 1,
                  "max_frames": 24, "max_pixels": 200704},
        "retrieval": {"top_k": 12, "evidence_chars": 48000, "max_inspections": 0},
        "scoring": "unchanged official scorer, earliest valid score retained",
        "scope": "Unresolved E11 questions only; provider/model switch from Gemini 3.8 to 3.5",
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
        if not target.exists():
            shutil.copytree(source, target, symlinks=False)
    return out, questions


def run_frontend(args, out, questions):
    env = dict(
        os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        LONGEMO_FFMPEG_COMPAT="1", LONGEMO_ALLOW_AUDIO_CACHE_REUSE="1",
    )
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
            "--fps", 1, "--max-frames", 24, "--max-pixels", 200704,
            "--with-audio", "--workers", 1, "--model", MODEL, "--base-url", BLACKAI,
            "--audio-model", MODEL, "--audio-base-url", BLACKAI,
            "--credential-file", args.credential_file, "--thinking", "default",
            "--timeout", 240, "--tries", 3, "--max-tokens", 32768,
            "--temperature", 1, "--config", setting,
        ]
        write_json(folder / "build_command.json", {"command": list(map(str, command)),
                                                     "time_unix": time.time()})
        with (folder / "build.log").open("a") as log:
            rc = subprocess.run(command, env=env, stdout=log,
                                stderr=subprocess.STDOUT).returncode
        complete = (memory / "memory.json").exists() and read(memory / "memory.json").get("complete")
        block = recorded_video_block(folder, memory)
        result = {"video_id": vid, "status": "memory_complete" if rc == 0 and complete
                  else block["status"] if block else "frontend_failed", "returncode": rc}
        if block:
            result["rejection"] = block["rejection"]
        export_video(out, vid, result)
        return result

    todo = [v for v in vids if v not in state["videos"]]
    state.update(status="running", queued_videos=todo, updated_unix=time.time())
    atomic(state_path, state)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, v): v for v in todo}
        for future in as_completed(futures):
            vid = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"video_id": vid, "status": "orchestration_error",
                          "error_type": type(exc).__name__}
            state["videos"][vid] = result
            state.update(queued_videos=[v for v in todo if v not in state["videos"]],
                        updated_unix=time.time())
            atomic(state_path, state)
            print(json.dumps(result), flush=True)
    state.update(status="finished", queued_videos=[], updated_unix=time.time())
    atomic(state_path, state)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("frontend", "backend", "all"))
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--credential-file", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    assert args.workers > 0
    out, questions = setup(args.runtime, args.selection)
    with (out / (args.stage + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage in ("frontend", "all"):
            run_frontend(args, out, questions)
        if args.stage in ("backend", "all"):
            inbox = out / "inbox"
            inbox.mkdir(exist_ok=True)
            for marker in (out / "outbox").glob("*.ready.json"):
                archive = out / "outbox" / read(marker)["archive"]
                shutil.copy2(archive, inbox / archive.name)
                shutil.copy2(marker, inbox / marker.name)
            backend(args, out, questions)


if __name__ == "__main__":
    main()

