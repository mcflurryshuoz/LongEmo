"""Build and answer the window-only no-event ablation."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path

from evaluation.inference.runner import init_client, model_args
from evaluation.io_utils import load_questions, load_records, write_json, write_records
from .audio import audio_client_for, bridge_audio
from .common import LoggedClient, code_hash, file_hash, fingerprint, git_revision, manifest, usage_summary
from .media import probe, subtitles, window_input
from .noevent_memory import apply_window, empty_memory, perception_context
from .noevent_retrieval import WindowIndex, retrieve
from .prompts import ANSWER_NOEVENT, PERCEPTION_NOEVENT, PLANNER
from .retrieval import validate_plan
from .runner import _media_options, client_for, validate_answer


def _build_video(video_id, args, client):
    folder = Path(args.output_dir) / video_id
    (folder / "windows").mkdir(parents=True, exist_ok=True)
    video = Path(args.videos_dir) / (video_id + ".mp4")
    video_sha = file_hash(video)
    info = probe(video)
    subtitle_path = Path(args.subtitles_dir) / (video_id + ".json") if args.subtitles_dir else None
    rows = subtitles(subtitle_path) if subtitle_path else []
    audio_client = audio_client_for(args)
    config = {"representation": "window_records", "video_sha256": video_sha,
              "subtitles_sha256": file_hash(subtitle_path) if subtitle_path else None,
              "model": client.configuration(), "audio_observer": audio_client.configuration() if audio_client else None,
              "window_seconds": args.window_seconds, "padding": args.padding, "media": _media_options(args),
              "code_hash": code_hash(), "git_revision": git_revision()}
    build_hash = manifest(folder / "manifest.json", config)
    destination = folder / "memory.json"
    memory = json.loads(destination.read_text()) if destination.exists() else empty_memory(video_id, info["duration"], video_sha)
    if memory.get("representation") != "window_records":
        raise ValueError("memory directory belongs to another representation")
    memory["build_fingerprint"] = build_hash
    api = LoggedClient(client, folder / "calls.jsonl", args.tries)
    count = math.ceil(info["duration"] / args.window_seconds)
    for i in range(count):
        window_id = f"W{i+1:05d}"
        if window_id in memory["completed_windows"]:
            continue
        core = [i * args.window_seconds, min((i + 1) * args.window_seconds, info["duration"])]
        interval = [max(0, core[0] - args.padding), min(info["duration"], core[1] + args.padding)]
        media, metadata = window_input(video, *interval, subtitle_rows=rows, **_media_options(args))
        if audio_client:
            media, audio_trace = bridge_audio(media, interval, audio_client, folder / "audio" / (window_id + ".json"), args.tries)
            metadata["audio_representation"] = "derived_timestamped_cues"
            metadata["audio_observer"] = audio_trace
        # Strict ablation: the perception model sees only the current media
        # window and the accumulated person index.  It never receives prior
        # window summaries, so cross-window continuity cannot leak into the
        # window-only representation.
        context = perception_context(memory, window_id=window_id, core=core, media=interval)
        messages = [{"role": "system", "content": PERCEPTION_NOEVENT},
                    {"role": "user", "content": [{"type": "text", "text": json.dumps(context, ensure_ascii=False)}] + media}]

        def validate(payload):
            apply_window(memory, payload, window_id=window_id, core=core, media=interval, metadata=metadata)

        payload = api.call(messages, purpose=f"window_perception:{video_id}:{window_id}", validate=validate)
        memory = apply_window(memory, payload, window_id=window_id, core=core, media=interval, metadata=metadata)
        write_json(folder / "windows" / (window_id + ".json"), {"input": context, "sampling": metadata, "perception": payload})
        write_json(destination, memory)
        print(f"[{video_id}] window {i + 1}/{count}; people={len(memory['entities'])}", flush=True)
    memory["complete"] = len(memory["completed_windows"]) == count
    write_json(destination, memory)
    write_json(folder / "usage.json", usage_summary(api.ledger))
    return {"video_id": video_id, "status": "ok", "windows": count,
            "memory_sha256": file_hash(destination), "usage": usage_summary(api.ledger)}


def build(args):
    client = client_for(args)
    questions = load_questions(args.data_path, "episode")
    video_ids = sorted({q["video_id"] for q in questions})
    if args.video_id:
        if set(args.video_id) - set(video_ids):
            raise ValueError("unknown video ID")
        video_ids = [v for v in video_ids if v in args.video_id]
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_build_video, v, args, client): v for v in video_ids}
        for future in as_completed(futures):
            video_id = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({"video_id": video_id, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
                print(f"[{video_id}] failed: {type(exc).__name__}: {exc}", flush=True)
            write_json(Path(args.output_dir) / "build_results.json", sorted(results, key=lambda x: x["video_id"]))
    return int(any(r["status"] != "ok" for r in results))


def _answer_one(q, memory, args, client, output, index):
    qid = q["question_id"]
    public = {key: q[key] for key in ("question_id", "video_id", "granularity", "type", "question")}
    api = LoggedClient(client, output / "calls" / (qid + ".jsonl"), args.tries)
    planner_messages = [{"role": "system", "content": PLANNER}, {"role": "user", "content": json.dumps(
        {"question": public["question"], "duration": memory["duration"], "cast": memory["entities"]}, ensure_ascii=False)}]
    if args.plans_dir:
        plan_path = Path(args.plans_dir) / (qid + ".json")
        signature = fingerprint({"messages": planner_messages, "model": client.configuration()})
        if plan_path.exists():
            cached = json.loads(plan_path.read_text())
            if cached["input_fingerprint"] != signature:
                raise ValueError("shared retrieval plan input/model changed")
            plan = cached["plan"]
            validate_plan(plan)
        else:
            planner = LoggedClient(client, Path(args.plans_dir) / "calls" / (qid + ".jsonl"), args.tries)
            plan = planner.call(planner_messages, purpose=f"plan:{qid}", validate=validate_plan)
            write_json(plan_path, {"input_fingerprint": signature, "plan": plan})
    else:
        plan = api.call(planner_messages, purpose=f"plan:{qid}", validate=validate_plan)
    trace = {}
    evidence = retrieve(memory, public["question"], plan, dense_scores=index.rank(public["question"]),
                        budget_chars=args.evidence_chars, top_k=args.top_k)
    payload = {"question": public["question"], "evidence": evidence,
               "inspection_budget": {"requests_remaining": 0, "seconds_remaining": 0}}
    messages = [{"role": "system", "content": ANSWER_NOEVENT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    if len(json.dumps(evidence, ensure_ascii=False)) > args.evidence_chars:
        raise RuntimeError("serialized window evidence exceeds the configured budget")
    result = api.call(messages,
                      purpose=f"answer:{qid}", validate=validate_answer)
    valid_ids = set(evidence["evidence_ids"])
    trace.update({"question": public, "plan": plan, "retrieval": evidence, "answer": result,
                  "answer_input_characters": {"user": len(messages[1]["content"]),
                      "system": len(messages[0]["content"]),
                      "serialized_messages": len(json.dumps(messages, ensure_ascii=False))},
                  "invalid_evidence_ids": [x for x in result["evidence_ids"] if x not in valid_ids],
                  "usage": usage_summary(api.ledger)})
    write_json(output / "traces" / (qid + ".json"), trace)
    return {**public, "status": "ok", "prediction": result["answer"].strip(), "method": "noevent-window",
            "memory_fingerprint": memory["build_fingerprint"], "memory_sha256": fingerprint(memory), "usage": trace["usage"]}


def answer(args):
    client = client_for(args)
    questions = load_questions(args.data_path, "episode", args.qid, limit=args.limit)
    output = Path(args.output_dir)
    (output / "calls").mkdir(parents=True, exist_ok=True)
    (output / "traces").mkdir(parents=True, exist_ok=True)
    memories, hashes = {}, {}
    for video_id in sorted({q["video_id"] for q in questions}):
        memory_path = Path(args.memory_dir) / video_id / "memory.json"
        memory = json.loads(memory_path.read_text())
        if not memory.get("complete") or memory.get("representation") != "window_records":
            raise ValueError(f"incomplete/window memory for {video_id}")
        memories[video_id], hashes[video_id] = memory, file_hash(memory_path)
    from .embeddings import APIEncoder, Encoder
    if args.embedding_backend in ("gemini", "gemini-native"):
        credentials = json.loads(Path(args.credential_file).read_text()) if args.credential_file else {}
        native = args.embedding_backend == "gemini-native"
        key_name = "GEMINI_API_KEY" if native else "OPENROUTER_API_KEY"
        encoder = APIEncoder(credentials.get(key_name), Path(args.embedding_cache_dir) / "api",
                             model="gemini-embedding-2" if native else args.embedding_model,
                             base_url=args.embedding_base_url or "https://openrouter.ai/api/v1",
                             api_format="gemini" if native else "openai")
    else:
        encoder = Encoder(model=args.embedding_model, revision=args.embedding_revision)
    indexes = {v: WindowIndex(memory, encoder, args.embedding_cache_dir) for v, memory in memories.items()}
    write_json(output / "embedding_indexes.json", {v: i.metadata for v, i in indexes.items()})
    configuration = {"representation": "window_records", "embedding": encoder.config,
                    "model": client.configuration(), "questions": [{k: q[k] for k in ("question_id", "video_id", "question", "type")} for q in questions],
                    "memories": hashes, "evidence_chars": args.evidence_chars, "top_k": args.top_k,
                    "code_hash": code_hash(), "git_revision": git_revision()}
    manifest(output / "manifest.json", configuration)
    write_json(output / "evaluation_questions.json", questions)
    path = output / "predictions.jsonl"
    old = load_records(path) if path.exists() else []
    records = {r["question_id"]: r for r in old if r.get("status") == "ok"}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_answer_one, q, memories[q["video_id"]], args, client, output, indexes[q["video_id"]]): q
                   for q in questions if q["question_id"] not in records}
        for future in as_completed(futures):
            q = futures[future]
            try:
                records[q["question_id"]] = future.result()
            except Exception as exc:
                records[q["question_id"]] = {"question_id": q["question_id"], "video_id": q["video_id"],
                    "status": "error", "prediction": None, "error": f"{type(exc).__name__}: {exc}"}
            write_records(path, [records[q["question_id"]] for q in questions if q["question_id"] in records])
            print(f"[{q['question_id']}] {records[q['question_id']]['status']}", flush=True)
    write_records(path, [records[q["question_id"]] for q in questions])
    for v, expected in hashes.items():
        if file_hash(Path(args.memory_dir) / v / "memory.json") != expected:
            raise RuntimeError("shared window memory changed during answering")
    return int(any(r.get("status") != "ok" for r in records.values()))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    for name in ("build", "answer"):
        command = commands.add_parser(name)
        command.add_argument("--data-path", required=True)
        command.add_argument("--videos-dir", required=name == "build")
        command.add_argument("--subtitles-dir")
        command.add_argument("--output-dir", required=True)
        command.add_argument("--credential-file")
        command.add_argument("--with-audio", action="store_true")
        command.add_argument("--audio-model")
        command.add_argument("--audio-base-url")
        command.add_argument("--fps", type=float, default=1)
        command.add_argument("--max-frames", type=int, default=24)
        command.add_argument("--max-pixels", type=int, default=200704)
        command.add_argument("--tries", type=int, default=3)
        command.add_argument("--workers", type=int, default=1)
        model_args(command)
        if name == "build":
            command.add_argument("--window-seconds", type=float, default=20)
            command.add_argument("--padding", type=float, default=2)
            command.add_argument("--video-id", action="append")
        else:
            command.add_argument("--memory-dir", required=True)
            command.add_argument("--plans-dir")
            command.add_argument("--embedding-backend", choices=("gemini", "gemini-native", "local"), default="gemini")
            command.add_argument("--embedding-base-url")
            command.add_argument("--embedding-model", default="google/gemini-embedding-2")
            command.add_argument("--embedding-revision", default="d128750597153bb5987e10b1c3493a34e5a4502a")
            command.add_argument("--embedding-cache-dir", default=".cache/longemo-embeddings")
            command.add_argument("--evidence-chars", type=int, default=48000)
            command.add_argument("--top-k", type=int, default=12)
            command.add_argument("--qid", action="append")
            command.add_argument("--limit", type=int)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "build" and (args.window_seconds <= 0 or args.padding < 0):
        raise SystemExit("invalid window configuration")
    return build(args) if args.command == "build" else answer(args)


if __name__ == "__main__":
    raise SystemExit(main())
