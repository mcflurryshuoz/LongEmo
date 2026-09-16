"""Reproducible pilot: shared memory, hybrid event-graph retrieval, and official scoring."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from evaluation.io_utils import load_records, write_json
from .common import code_hash, file_hash, manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--credential-file", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--judge-model", required=True)
    p.add_argument("--audio-model", default="google/gemini-3.8-flash")
    p.add_argument("--embedding-model", default="google/gemini-embedding-2")
    p.add_argument("--embedding-cache-dir", required=True)
    p.add_argument("--memory-dir", help="Reuse complete, frozen memory; no perception calls")
    p.add_argument("--window-seconds", type=float, default=20)
    p.add_argument("--build-workers", type=int, default=3)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--direct-baseline", action="store_true")
    p.add_argument("--with-inspection", action="store_true")
    args = p.parse_args()
    root = Path(args.data_root).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    questions = root / "pilot_questions.json"
    videos = root / "episode/videos"
    subtitle_dir = root / "prepared_subtitles"
    credentials = json.loads(Path(args.credential_file).read_text())
    env = os.environ.copy()
    env.update(MODEL_API_KEY=credentials["OPENROUTER_API_KEY"],
               OPENROUTER_API_KEY=credentials["OPENROUTER_API_KEY"], MODEL_BASE_URL="https://openrouter.ai/api/v1")
    manifest(out / "experiment_manifest.json", {"data_revision": json.loads((root/"manifest.json").read_text())["revision"],
        "questions_sha256": file_hash(questions), "code_hash": code_hash(), "model": args.model,
        "judge_model": args.judge_model, "window_seconds": args.window_seconds,
        "audio_model": args.audio_model, "embedding_model": args.embedding_model,
        "memory_dir": args.memory_dir, "reasoning_effort": "medium",
        "direct_baseline": args.direct_baseline, "with_inspection": args.with_inspection,
        "input_modalities": ["video", "audio", "subtitle"]})
    status = {"started_unix": time.time(), "stages": []}

    def run(name, module, arguments):
        entry = {"stage": name, "status": "running", "started_unix": time.time()}
        status["stages"].append(entry)
        write_json(out / "status.json", status)
        print("Starting", name, flush=True)
        with (out / (name + ".log")).open("a") as log:
            result = subprocess.run([sys.executable, "-u", "-m", module] + [str(x) for x in arguments],
                                    env=env, stdout=log, stderr=subprocess.STDOUT)
        entry.update(status="ok" if result.returncode == 0 else "error", exit_code=result.returncode, ended_unix=time.time())
        write_json(out / "status.json", status)
        if result.returncode:
            raise RuntimeError(f"{name} failed; inspect {out / (name+'.log')}; rerun safely after resolving errors")

    inference = ["--model", args.model, "--thinking", "default", "--timeout", "240", "--tries", "5",
                 "--credential-file", args.credential_file, "--audio-model", args.audio_model]
    memory_dir = Path(args.memory_dir).resolve() if args.memory_dir else out / "memory"
    if not args.memory_dir:
        run("build", "methods.longemo", ["build", "--data-path", questions, "--videos-dir", videos,
            "--subtitles-dir", subtitle_dir, "--output-dir", memory_dir, "--with-audio",
            "--window-seconds", args.window_seconds, "--padding", "2", "--fps", "1", "--max-frames", "24",
            "--max-pixels", "200704", "--max-tokens", "8192", "--workers", args.build_workers] + inference)
    methods = ["graph"]
    if args.with_inspection:
        methods.append("graph_inspect")
    if args.direct_baseline:
        methods.append("direct")
    for method in methods:
        run(method, "methods.longemo", ["answer", "--data-path", questions, "--memory-dir", memory_dir,
            "--plans-dir", out/"shared_plans", "--output-dir", out/method,
            "--retrieval", "direct" if method == "direct" else "graph",
            "--embedding-model", args.embedding_model, "--embedding-cache-dir", args.embedding_cache_dir,
            "--videos-dir", videos, "--subtitles-dir", subtitle_dir, "--with-audio",
            "--max-inspections", 1 if method == "graph_inspect" else 0, "--inspection-seconds", "60",
            "--workers", args.workers, "--max-tokens", "8192"] + inference)
    scores = {}
    for method in methods:
        predictions = out/method/"predictions.jsonl"
        rows = load_records(predictions)
        expected = {q["question_id"] for q in json.loads(questions.read_text())}
        if len(rows) != len(expected) or {r["question_id"] for r in rows} != expected or any(
            r.get("status") != "ok" or not isinstance(r.get("prediction"), str) or not r["prediction"].strip() for r in rows):
            raise ValueError("incomplete/duplicate/invalid submission; not scoring a partial result")
        score_dir = out/method/"scores"
        # Preserve completed judgments when resuming orchestration. A new
        # experiment directory is mandatory after any input/model/code change.
        candidates = sorted(score_dir.glob("run_*/metrics.json")) if score_dir.exists() else []
        complete = [path for path in candidates if json.loads(path.read_text())["overall_unweighted"]["coverage"] == 1]
        if not complete:
            run("score_" + method, "evaluation.eval", ["--data-path", questions, "--predictions", predictions,
                "-g", "episode", "--model", args.judge_model, "--output-dir", score_dir,
                "--workers", args.workers, "--tries", "3", "--timeout", "180", "--max-tokens", "8192"])
            complete = sorted(score_dir.glob("run_*/metrics.json"))
        metrics_path = complete[-1]
        scores[method] = json.loads(metrics_path.read_text())
        scores[method]["source_file"] = str(metrics_path)
    write_json(out / "comparison.json", scores)
    lines = ["# LongEmo pilot results", "", "This is a development pilot, not the full benchmark.", "",
             "| Method | Questions scored | Overall normalized (%) |", "|---|---:|---:|"]
    for method, result in scores.items():
        overall = result["overall_unweighted"]
        lines.append(f"| {method} | {overall['n_scored']}/{overall['n_total']} | {overall['percent_score']:.2f} |")
    lines += ["", "The main method fuses structured semantic and Gemini Embedding 2 event recall, expands the event graph and preserves temporal coverage.",
              "Direct uses 128 uniformly sampled frames plus subtitles and the identical independent audio-only observations. It never receives graph events/states. This is a system baseline, not a matched visual sampling ablation.",
              "The judge and score formulas are shared. Per-video construction costs and per-question call ledgers are retained."]
    (out/"comparison.md").write_text("\n".join(lines)+"\n")
    status.update(status="complete", ended_unix=time.time())
    write_json(out / "status.json", status)
    print(json.dumps({m:r["overall_unweighted"] for m,r in scores.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
