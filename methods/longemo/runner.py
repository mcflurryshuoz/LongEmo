"""Build reusable emotional memory, then retrieve evidence and answer independently."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import time

from evaluation.inference.runner import init_client, model_args
from evaluation.io_utils import load_questions, write_json, write_records, load_records
from .common import LoggedClient, code_hash, file_hash, fingerprint, git_revision, manifest, usage_summary
from .media import probe, subtitles, window_input
from .memory import apply_window, empty_memory
from .prompts import ANSWER, PERCEPTION, PLANNER
from .audio import audio_client_for, bridge_audio
from .retrieval import retrieve, validate_plan


def client_for(args):
    from evaluation.azure_transport import is_azure
    if args.base_url and is_azure(args.base_url):
        # This route refreshes the existing Azure CLI credential; do not pick
        # an unrelated Gemini/OpenRouter secret from the shared resource file.
        args.api_key = ""
        return init_client(args)
    if args.credential_file:
        config = json.loads(Path(args.credential_file).read_text())
        if (args.model or "").startswith(("openai/", "google/")):
            args.api_key = args.api_key or config.get("OPENROUTER_API_KEY")
            args.base_url = args.base_url or "https://openrouter.ai/api/v1"
        else:
            args.api_key = args.api_key or config.get("MODEL_API_KEY") or config.get("GEMINI_API_KEY")
            args.base_url = args.base_url or config.get("MODEL_BASE_URL") or (config["GOOGLE_GEMINI_BASE_URL"].rstrip("/") + "/v1beta")
    if not args.model:
        raise ValueError("--model must be explicit")
    return init_client(args)


def _media_options(args):
    return {"fps": args.fps, "max_frames": args.max_frames, "max_pixels": args.max_pixels,
            "with_audio": args.with_audio}


def _build_video(video_id, args, client):
    folder = Path(args.output_dir) / video_id
    folder.mkdir(parents=True, exist_ok=True)
    video = Path(args.videos_dir) / (video_id + ".mp4")
    video_sha = file_hash(video)
    info = probe(video)
    subtitle_path = Path(args.subtitles_dir) / (video_id + ".json") if args.subtitles_dir else None
    rows = subtitles(subtitle_path) if subtitle_path else []
    audio_client = audio_client_for(args)
    config = {"audio_observer": audio_client.configuration() if audio_client else None, "method": "longemo-joint-perception-v1", "video_sha256": video_sha,
              "subtitles_sha256": file_hash(subtitle_path) if subtitle_path else None,
              "model": client.configuration(), "window_seconds": args.window_seconds, "padding": args.padding,
              "media": _media_options(args), "allow_revisions": args.allow_revisions,
              "code_hash": code_hash(), "git_revision": git_revision()}
    build_hash = manifest(folder / "manifest.json", config)
    destination = folder / "memory.json"
    memory = json.loads(destination.read_text()) if destination.exists() else empty_memory(video_id, info["duration"], video_sha)
    memory["build_fingerprint"] = build_hash
    api = LoggedClient(client, folder / "calls.jsonl", args.tries)
    count = math.ceil(info["duration"] / args.window_seconds)
    for i in range(count):
        window_id = f"W{i+1:05d}"
        if window_id in memory["completed_windows"]:
            continue
        core = [i * args.window_seconds, min((i+1) * args.window_seconds, info["duration"])]
        interval = [max(0, core[0]-args.padding), min(info["duration"], core[1]+args.padding)]
        media, metadata = window_input(video, *interval, subtitle_rows=rows, **_media_options(args))
        if audio_client:
            media, audio_trace = bridge_audio(media, interval, audio_client, folder/"audio"/(window_id+".json"), args.tries)
            metadata["audio_representation"] = "derived_timestamped_cues"
            metadata["audio_observer"] = audio_trace
        context = {"video_id": video_id, "window_id": window_id, "core_interval": core,
                   "media_interval": interval, "corrections_enabled": args.allow_revisions,
                   "cast": memory["entities"], "preceding_events": memory["events"][-3:]}
        messages = [{"role": "system", "content": PERCEPTION},
                    {"role": "user", "content": [{"type": "text", "text": json.dumps(context, ensure_ascii=False)}] + media}]

        def validate(payload):
            apply_window(memory, payload, window_id=window_id, core=core, media=interval,
                         metadata=metadata, allow_revisions=args.allow_revisions)

        payload = api.call(messages, purpose=f"perception:{video_id}:{window_id}", validate=validate)
        updated = apply_window(memory, payload, window_id=window_id, core=core, media=interval,
                               metadata=metadata, allow_revisions=args.allow_revisions)
        write_json(folder / "windows" / (window_id + ".json"), {"input": context, "sampling": metadata, "perception": payload})
        write_json(destination, updated)
        memory = updated
        print(f"[{video_id}] window {i+1}/{count}; people={len(memory['entities'])} events={len(memory['events'])}", flush=True)
    memory["complete"] = len(memory["completed_windows"]) == count
    write_json(destination, memory)
    write_json(folder / "usage.json", usage_summary(api.ledger))
    return {"video_id": video_id, "status": "ok", "windows": count, "events": len(memory["events"]),
            "memory_sha256": file_hash(destination), "usage": usage_summary(api.ledger)}


def build(args):
    client = client_for(args)
    # Read only video IDs from benchmark records. Neither questions nor answers
    # enter perception prompts, hashes, or generated memories.
    questions = load_questions(args.data_path, "episode")
    video_ids = sorted({q["video_id"] for q in questions})
    if args.video_id:
        if set(args.video_id) - set(video_ids):
            raise ValueError("unknown video ID")
        video_ids = [v for v in video_ids if v in args.video_id]
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_build_video, video_id, args, client): video_id for video_id in video_ids}
        for future in as_completed(futures):
            video_id = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({"video_id": video_id, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
                print(f"[{video_id}] failed: {type(exc).__name__}: {exc}", flush=True)
            write_json(Path(args.output_dir) / "build_results.json", sorted(results, key=lambda r: r["video_id"]))
    return int(any(r["status"] != "ok" for r in results))


def validate_answer(value):
    if not isinstance(value, dict) or not isinstance(value.get("answer"), str) or not value["answer"].strip():
        raise ValueError("answer must be nonempty text")
    for field in ("evidence_ids", "inspect"):
        if not isinstance(value.get(field), list):
            raise ValueError(f"{field} must be a list")
    if any(not isinstance(x, str) for x in value["evidence_ids"]):
        raise ValueError("evidence_ids must be strings")
    for item in value["inspect"]:
        if not isinstance(item, dict) or any(type(item.get(k)) not in (int, float) or not math.isfinite(item[k]) for k in ("start", "end")):
            raise ValueError("inspection times must be finite numbers")
        if item["start"] < 0 or item["end"] <= item["start"] or not isinstance(item.get("question"), str):
            raise ValueError("invalid inspection interval or question")


def _direct_one(q, memory, args, client, output):
    """Direct sampled frames + subtitles + the same audio-only observations.

    Graph events, emotions, planner and retrieved evidence are never read here.
    """
    qid = q["question_id"]
    video = Path(args.videos_dir)/(q["video_id"]+".mp4")
    rows = subtitles(Path(args.subtitles_dir)/(q["video_id"]+".json")) if args.subtitles_dir else []
    content, sampling = window_input(video, 0, memory["duration"], subtitle_rows=rows,
        fps=args.fps, max_frames=args.direct_frames, max_pixels=args.max_pixels, with_audio=False)
    audio_refs = {}
    if args.with_audio:
        observer = audio_client_for(args)
        if observer is None:
            raise ValueError("direct shared-audio baseline requires --audio-model")
        observations = []
        for window in memory["completed_windows"]:
            path = Path(args.memory_dir)/q["video_id"]/"audio"/(window+".json")
            record = json.loads(path.read_text())
            if record["model"] != observer.configuration():
                raise ValueError("direct baseline audio observer differs from memory frontend")
            observations.extend(record["result"]["observations"])
            audio_refs[window] = file_hash(path)
        content.append({"type": "text", "text": "Audio-only observations from an independent model; "
            "overlapping windows may repeat cues. They may contain errors; ground voice identities in frames/dialogue. "
            + json.dumps(observations, ensure_ascii=False)})
    from evaluation.inference.prompts import build_messages
    # The official ordinary inference prompt is reused with question-only fields.
    public = {k:q[k] for k in ("question_id","video_id","question","granularity","type")}
    messages = build_messages(public, media_content=content)
    api = LoggedClient(client, output/"calls"/(qid+".jsonl"), args.tries)
    prediction = api.call(messages, purpose="direct:"+qid)
    write_json(output/"traces"/(qid+".json"), {"sampling": sampling,"audio_sources": audio_refs,"usage": usage_summary(api.ledger)})
    return {"question_id": qid,"video_id":q["video_id"],"status":"ok","prediction":prediction,
            "method":"direct_shared_audio_frontend","usage":usage_summary(api.ledger)}


def _answer_one(q, memory, args, client, output, dense_index=None):
    if args.retrieval == "direct":
        return _direct_one(q, memory, args, client, output)
    qid = q["question_id"]
    api = LoggedClient(client, output / "calls" / (qid + ".jsonl"), args.tries)
    # Explicit allowlist: gold and question-specific rubric are never serialized.
    public = {key: q[key] for key in ("question_id", "video_id", "granularity", "type", "question")}
    planner_messages = [{"role": "system", "content": PLANNER}, {"role": "user", "content": json.dumps(
        {"question": public["question"], "duration": memory["duration"], "cast": memory["entities"]}, ensure_ascii=False)}]
    if getattr(args, "plans_dir", None):
        plan_path = Path(args.plans_dir) / (qid + ".json")
        plan_signature = fingerprint({"messages": planner_messages, "model": client.configuration()})
        if plan_path.exists():
            cached = json.loads(plan_path.read_text())
            if cached["input_fingerprint"] != plan_signature:
                raise ValueError("shared retrieval plan input/model changed")
            plan = cached["plan"]
            validate_plan(plan)
        else:
            planner = LoggedClient(client, Path(args.plans_dir)/"calls"/(qid+".jsonl"), args.tries)
            plan = planner.call(planner_messages, purpose=f"plan:{qid}", validate=validate_plan)
            write_json(plan_path, {"input_fingerprint": plan_signature, "plan": plan})
    else:
        plan = api.call(planner_messages, purpose=f"plan:{qid}", validate=validate_plan)
    retrieval_trace = {}
    dense_scores = dense_index.rank(public["question"]) if dense_index is not None else None
    evidence = retrieve(memory, public["question"], plan, mode=args.retrieval, budget_chars=args.evidence_chars,
                        top_k=args.top_k, dense_scores=dense_scores, trace=retrieval_trace)
    payload = {"question": public["question"], "evidence": evidence,
               "inspection_budget": {"requests_remaining": args.max_inspections, "seconds_remaining": args.inspection_seconds}}
    messages = [{"role": "system", "content": ANSWER}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    inspections, seconds_used = [], 0.0
    seen = set()
    result = None
    for round_index in range(args.max_inspections + 1):
        result = api.call(messages, purpose=f"answer:{qid}:round{round_index}", validate=validate_answer)
        if not result["inspect"] or round_index == args.max_inspections:
            break
        item = result["inspect"][0]
        start = min(item["start"], memory["duration"])
        end = min(item["end"], memory["duration"], start + args.inspection_seconds - seconds_used)
        key = (round(start, 3), round(end, 3))
        if end <= start or key in seen:
            break
        seen.add(key)
        video = Path(args.videos_dir) / (q["video_id"] + ".mp4")
        rows = subtitles(Path(args.subtitles_dir) / (q["video_id"] + ".json")) if args.subtitles_dir else []
        content, metadata = window_input(video, start, end, subtitle_rows=rows, **_media_options(args))
        audio_client = audio_client_for(args)
        if audio_client:
            content, audio_trace = bridge_audio(content, [start,end], audio_client,
                output/"inspection_audio"/(qid+f"-{round_index}.json"), args.tries)
            metadata["audio_representation"] = "derived_timestamped_cues"
            metadata["audio_observer"] = audio_trace
        seconds_used += end-start
        inspections.append({"request": item, "media": metadata})
        messages.append({"role": "assistant", "content": json.dumps(result, ensure_ascii=False)})
        instructions = {"neutral_observation_request": item["question"], "requests_remaining": args.max_inspections-round_index-1,
                        "seconds_remaining": args.inspection_seconds-seconds_used,
                        "instruction": "Reassess only with evidence; the supplied source is read-only and does not alter the shared memory."}
        messages.append({"role": "user", "content": content + [{"type": "text", "text": json.dumps(instructions)}]})
    valid_ids = {e["id"] for e in memory["events"]} | {o["id"] for o in memory["observations"]}
    trace = {"question": public, "plan": plan, "retrieval": evidence, "recall_trace": retrieval_trace, "inspection_trace": inspections,
             "answer": result, "invalid_evidence_ids": [i for i in result["evidence_ids"] if i not in valid_ids],
             "evidence_characters": len(json.dumps(evidence, ensure_ascii=False)), "usage": usage_summary(api.ledger)}
    write_json(output / "traces" / (qid + ".json"), trace)
    return {**public, "status": "ok", "prediction": result["answer"].strip(), "method": "longemo-"+args.retrieval,
            "memory_fingerprint": memory["build_fingerprint"], "memory_sha256": fingerprint(memory),
            "inspection_seconds": seconds_used, "usage": trace["usage"]}


def answer(args):
    client = client_for(args)
    questions = load_questions(args.data_path, "episode", args.qid, limit=args.limit)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    memories = {}
    hashes = {}
    for video_id in sorted({q["video_id"] for q in questions}):
        path = Path(args.memory_dir) / video_id / "memory.json"
        memory = json.loads(path.read_text())
        if not memory.get("complete"):
            raise ValueError(f"incomplete memory for {video_id}; complete perception before answering")
        if args.max_inspections or args.retrieval == "direct":
            if not args.videos_dir or file_hash(Path(args.videos_dir)/(video_id+".mp4")) != memory["video_sha256"]:
                raise ValueError("inspection video does not match memory source")
        memories[video_id], hashes[video_id] = memory, file_hash(path)
    indexes = {}
    encoder = None
    if args.retrieval == "graph":
        from .embeddings import Encoder, APIEncoder, EventIndex
        if args.embedding_backend in ("gemini", "gemini-native"):
            import os
            credentials = json.loads(Path(args.credential_file).read_text()) if args.credential_file else {}
            native_embedding = args.embedding_backend == 'gemini-native'
            if native_embedding and not args.embedding_base_url:
                raise ValueError('native Gemini embeddings require an explicit --embedding-base-url for the configured credential')
            key_name = 'GEMINI_API_KEY' if native_embedding else 'OPENROUTER_API_KEY'
            encoder = APIEncoder(credentials.get(key_name) or os.getenv(key_name),
                Path(args.embedding_cache_dir)/"api", model=args.embedding_model,
                base_url=args.embedding_base_url or 'https://openrouter.ai/api/v1',
                api_format='gemini' if native_embedding else 'openai')
        else:
            encoder = Encoder(model=args.embedding_model, revision=args.embedding_revision)
        for video_id, memory in memories.items():
            indexes[video_id] = EventIndex(memory, encoder, args.embedding_cache_dir)
        write_json(output/"embedding_indexes.json", {v:i.metadata for v,i in indexes.items()})
    configuration = {"embedding": encoder.config if encoder else None, "model": client.configuration(), "question_inputs": [
        {k: q[k] for k in ("question_id", "video_id", "question", "type")} for q in questions],
        "memories": hashes, "retrieval": args.retrieval, "evidence_chars": args.evidence_chars,
        "top_k": args.top_k, "max_inspections": args.max_inspections, "inspection_seconds": args.inspection_seconds,
        "direct_frames": args.direct_frames, "audio_model": args.audio_model,
        "inspection_media": _media_options(args), "inspection_subtitle_hashes": {
            v: file_hash(Path(args.subtitles_dir)/(v+".json")) for v in memories} if args.subtitles_dir else {},
        "code_hash": code_hash(), "git_revision": git_revision()}
    manifest(output / "manifest.json", configuration)
    write_json(output / "evaluation_questions.json", questions)
    path = output / "predictions.jsonl"
    old = load_records(path) if path.exists() else []
    if len({r["question_id"] for r in old}) != len(old):
        raise ValueError("duplicate IDs in existing prediction file")
    records = {r["question_id"]: r for r in old if r.get("status") == "ok"}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_answer_one, q, memories[q["video_id"]], args, client, output, indexes.get(q["video_id"])): q
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
    # Question-local evidence never mutates the shared memory, including across threads.
    for v, expected in hashes.items():
        if file_hash(Path(args.memory_dir) / v / "memory.json") != expected:
            raise RuntimeError("shared memory changed during question answering")
    return int(any(r["status"] != "ok" for r in records.values()))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    for name in ("build", "answer"):
        command = commands.add_parser(name)
        command.add_argument("--data-path", required=True)
        command.add_argument("--videos-dir", required=name == "build")
        command.add_argument("--subtitles-dir")
        command.add_argument("--output-dir", required=True)
        command.add_argument("--credential-file", help="Private JSON outside the repository; never included in manifests")
        command.add_argument("--with-audio", action="store_true")
        command.add_argument("--audio-model", help="Audio observer required when GPT-6 receives audio")
        command.add_argument("--audio-base-url", help="Explicit native Gemini audio endpoint; credentials stay in the private resource file")
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
            command.add_argument("--allow-revisions", action="store_true")
        else:
            command.add_argument("--memory-dir", required=True)
            command.add_argument("--plans-dir", help="Optional shared frozen plans for fair retrieval comparisons")
            command.add_argument("--retrieval", choices=("graph", "flat", "direct"), default="graph")
            command.add_argument("--direct-frames", type=int, default=128)
            command.add_argument("--embedding-backend", choices=("gemini", "gemini-native", "local"), default="gemini")
            command.add_argument("--embedding-base-url", help="Explicit embedding service URL; native Gemini uses /v1beta")
            command.add_argument("--embedding-model", default="google/gemini-embedding-2")
            command.add_argument("--embedding-revision", default="d128750597153bb5987e10b1c3493a34e5a4502a")
            command.add_argument("--embedding-cache-dir", default=".cache/longemo-embeddings")
            command.add_argument("--evidence-chars", type=int, default=48000)
            command.add_argument("--top-k", type=int, default=12)
            command.add_argument("--max-inspections", type=int, default=0)
            command.add_argument("--inspection-seconds", type=float, default=120)
            command.add_argument("--qid", action="append")
            command.add_argument("--limit", type=int)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.fps <= 0 or args.max_frames < 1 or args.workers < 1 or args.tries < 1:
        p.error("sampling and execution parameters must be positive")
    if args.command == "build" and (args.window_seconds <= 0 or args.padding < 0):
        p.error("invalid window configuration")
    if args.command == "answer" and (args.max_inspections < 0 or args.inspection_seconds <= 0):
        p.error("invalid inspection budget")
    return build(args) if args.command == "build" else answer(args)


if __name__ == "__main__":
    raise SystemExit(main())
