"""Explicit, resumable stages for the event/window three-condition pilot.

No service daemon or API preflight. A started task is never automatically
repeated, including after an ambiguous interruption; inspect its ledger first.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import threading
import time
import uuid


CONDITIONS = ("base", "method", "noevent")
STAGES = ("prepare", "build", "plan", "answer", "score", "status", "all")
_SUMMARY_LOCK = threading.RLock()


def read(path):
    return json.loads(Path(path).read_text())


def records(path):
    path = Path(path)
    if not path.exists():
        return []
    return ([json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if path.suffix == ".jsonl" else read(path))


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def freeze(path, value):
    """Never silently replace a frozen selection, config, plan, or judgment."""
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f"frozen artifact changed: {path}")
    else:
        write(path, value)


def select_questions(source, mode, video_id=None):
    rows = records(source)
    if not isinstance(rows, list) or len(rows) < 50:
        raise ValueError("the pilot source must contain at least 50 episode questions")
    ids = [row["question_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question IDs")
    pilot = rows[:50]
    videos = {row["video_id"] for row in pilot}
    if len(videos) != 21 or any(row.get("granularity") != "episode" for row in pilot):
        raise ValueError("the fixed first-50 pilot must contain 21 episode videos")
    if mode == "smoke":
        video_id = video_id or pilot[0]["video_id"]
        if video_id not in videos:
            raise ValueError("smoke video must belong to the frozen first-50 pilot")
        return [row for row in pilot if row["video_id"] == video_id]
    if video_id:
        raise ValueError("--video-id is only valid for smoke")
    return pilot


def pid_identity(pid):
    stat = Path(f"/proc/{pid}/stat")
    return {"pid": pid, "host": socket.gethostname(),
            "start_ticks": stat.read_text().rsplit(")", 1)[1].split()[19] if stat.exists() else None}


@contextmanager
def run_lock(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "coordinator.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another coordinator owns this run") from None
        write(folder / "process.json", {**pid_identity(os.getpid()), "started_unix": time.time()})
        yield


def run_command(folder, command, cwd, env=None):
    """Resume unstarted work; never repeat a recorded attempt or refusal."""
    state_path = folder / "task.json"
    signature = digest({"command": command, "cwd": str(cwd)})
    if state_path.exists():
        previous = read(state_path)
        if previous["signature"] != signature:
            raise ValueError(f"task configuration changed: {folder}")
        if previous["status"] == "running":
            return {**previous, "status": "needs_audit", "reason": "previous attempt may still have an in-flight request"}
        return previous
    state = {"signature": signature, "command": command, "cwd": str(cwd),
             "status": "running", "started_unix": time.time(), "coordinator": pid_identity(os.getpid())}
    write(state_path, state)  # Claim before spawning, including crash uncertainty.
    with (folder / "output.log").open("a") as log:
        try:
            child = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
            state["child"] = pid_identity(child.pid)
            write(state_path, state)
            code = child.wait()
            state.update(status="ok" if code == 0 else "error", returncode=code, finished_unix=time.time())
        except Exception as exc:
            state.update(status="needs_audit", error_type=type(exc).__name__)
            write(state_path, state)
            raise
    write(state_path, state)
    return state


def source_hash(repo):
    paths = list((repo / "methods/longemo").glob("*.py"))
    paths += list((repo / "evaluation").rglob("*.py"))
    return digest({str(p.relative_to(repo)): sha(p) for p in sorted(paths) if "emollm" not in p.parts})


def prepare(args, run):
    questions = select_questions(args.questions, args.mode, args.video_id)
    if args.gate_video and (args.mode != "pilot" or args.gate_video not in {q["video_id"] for q in questions}):
        raise ValueError("gate video must belong to the fixed-50 pilot")
    method, noevent = Path(args.method_repo).resolve(), Path(args.noevent_repo).resolve()
    if "--progressive-routing" not in (method / "methods/longemo/runner.py").read_text():
        raise ValueError("method checkout must support explicit --progressive-routing none")
    for name in ("eval.py", "judge_prompts.py", "metrics.py"):
        if sha(method / "evaluation" / name) != sha(noevent / "evaluation" / name):
            raise ValueError(f"official evaluator differs between checkouts: {name}")
    videos, subtitles = Path(args.videos_dir).resolve(), Path(args.subtitles_dir).resolve()
    media = {}
    for video_id in sorted({q["video_id"] for q in questions}):
        video, subtitle = videos / (video_id + ".mp4"), subtitles / (video_id + ".json")
        if not video.is_file() or not video.stat().st_size or not subtitle.is_file():
            raise ValueError(f"missing media or prepared subtitle: {video_id}")
        media[video_id] = {"video_sha256": sha(video), "subtitles_sha256": sha(subtitle)}
    config = {"mode": args.mode, "gate_video": args.gate_video,
              "questions_sha256": digest(questions), "question_count": len(questions),
              "video_count": len(media), "media": media, "videos_dir": str(videos), "subtitles_dir": str(subtitles),
              "repos": {"event": str(method), "noevent": str(noevent)},
              "source_hashes": {"event": source_hash(method), "noevent": source_hash(noevent)},
              "coordinator_sha256": sha(__file__), "python": args.python,
              "model": args.model, "base_url": args.base_url, "audio_model": args.audio_model,
              "audio_base_url": args.audio_base_url, "embedding_model": args.embedding_model,
              "embedding_base_url": args.embedding_base_url, "evidence_chars": 48000,
              "window_seconds": 20, "padding": 2, "fps": 1, "max_frames": 24, "max_pixels": 200704,
              "stage_tries": {"build": args.build_tries, "plan": args.plan_tries,
                              "answer": args.answer_tries, "score": 1},
              "max_tokens": 8192, "timeout": args.timeout,
              "parent_run": str(Path(args.parent_run).resolve()) if args.parent_run else None,
              "progressive_routing": "none", "progressive_rounds": 3, "progressive_anchor_k": 4}
    freeze(run / "configuration.json", config)
    freeze(run / "questions.json", questions)
    for video_id in media:
        freeze(run / "questions" / (video_id + ".json"), [q for q in questions if q["video_id"] == video_id])
    if args.parent_run:
        inherit_parent(args, run, config)
    return config, questions


def common_args(config, stage="build"):
    tries = 1 if stage == "score" else config.get("stage_tries", {}).get(stage, config.get("tries", 3))
    return ["--model", config["model"], "--base-url", config["base_url"], "--max-tokens", "8192",
            "--timeout", str(config["timeout"]), "--tries", str(tries)]


def memory_root(run, branch, video):
    # Each build command also writes build_results.json. Per-video roots avoid
    # competing subprocesses overwriting that file in a common output folder.
    return run / branch / "videos" / video / "memory"


def assert_identity_stopped(identity, label):
    """A stale PID is safe only when absent or its Linux start ticks differ."""
    if not identity or not identity.get("pid"):
        raise ValueError(f"missing process identity: {label}")
    if identity.get("host") != socket.gethostname():
        raise ValueError(f"cannot audit another host's process: {label}")
    current = pid_identity(identity["pid"])
    if current["start_ticks"] is not None:
        if identity.get("start_ticks") is None or str(current["start_ticks"]) == str(identity["start_ticks"]):
            raise ValueError(f"parent process is still active: {label}")
    elif not Path("/proc").exists():
        # Non-Linux offline tests can prove an absent PID, but cannot safely
        # distinguish reuse of a live PID without start ticks.
        try:
            os.kill(identity["pid"], 0)
        except ProcessLookupError:
            return
        raise ValueError(f"cannot prove parent process stopped: {label}")


@contextmanager
def stopped_parent(parent):
    with (parent / "coordinator.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("parent coordinator still owns its lock") from None
        assert_identity_stopped(read(parent / "process.json"), "coordinator")
        for path in sorted((parent / "tasks").glob("*/*/task.json")):
            task = read(path)
            if task.get("child"):
                assert_identity_stopped(task["child"], str(path.relative_to(parent)))
            elif task.get("status") in ("running", "needs_audit"):
                raise ValueError("parent task has no auditable child identity")
        yield


def parent_failure(folder):
    """Only documented schema/JSON failures and RemoteDisconnected may resume."""
    memory = read(folder / "memory.json") if (folder / "memory.json").exists() else {}
    completed = set(memory.get("completed_windows", []))
    errors = [row for path in (folder / "calls.jsonl", folder / "audio/calls.jsonl")
              for row in records(path) if row.get("status") == "error"
              and row.get("purpose", "").split(":")[-1] not in completed]
    if memory.get("complete"):
        return {"status": "complete", "evidence": []}
    refusals = [r for r in errors if str(r.get("service_error_code", "")).lower() in {
        "content_filter", "content_policy_violation", "safety", "prohibited_content", "blocked_input"}]
    if refusals:
        return {"status": "blocked_content", "evidence": refusals}
    if any(r.get("http_status") == 428 for r in errors):
        return {"status": "blocked_http428", "evidence": errors}
    if any(r.get("http_status") in (401, 402, 403) for r in errors):
        return {"status": "blocked_service", "evidence": errors}
    last = max(errors, key=lambda r: r.get("time_unix", 0), default=None)
    if last and last.get("error_type") in ("ValueError", "JSONDecodeError", "RemoteDisconnected"):
        return {"status": "resume_once", "evidence": [last]}
    return {"status": "blocked_unknown", "evidence": [last] if last else []}


def inspect_client_configs(args, config, repo):
    """Construct clients in an isolated process; never call generate or a service."""
    command = [args.python, str(Path(__file__).resolve()), "_clients", "--repo", str(repo),
               "--credential-file", args.credential_file]
    return json.loads(subprocess.check_output(command, input=json.dumps(config), text=True))


def child_clients(args):
    sys.path.insert(0, args.repo)
    from methods.longemo.runner import parser as method_parser, client_for
    from methods.longemo.audio import audio_client_for
    config = json.load(sys.stdin)
    if "azure" in config["base_url"] or "cognitiveservices" in config["base_url"]:
        raise ValueError("offline inheritance inspection does not refresh Azure credentials")
    options = method_parser().parse_args(["build", "--data-path", "unused", "--videos-dir", "unused",
        "--output-dir", "unused", "--credential-file", args.credential_file, *common_args(config, "build"),
        "--with-audio", "--audio-model", config["audio_model"], "--audio-base-url", config["audio_base_url"]])
    print(json.dumps({"model": client_for(options).configuration(),
                      "audio_observer": audio_client_for(options).configuration()}))
    return 0


def validate_parent_checkpoint(folder, repo, branch, video, expected_media, config, clients):
    manifest = read(folder / "manifest.json")
    old = manifest["configuration"]
    if digest(old) != manifest["fingerprint"]:
        raise ValueError("parent builder manifest fingerprint is invalid")
    for key in ("video_sha256", "subtitles_sha256"):
        if old.get(key) != expected_media[key]:
            raise ValueError(f"parent checkpoint media changed: {key}")
    for key in ("window_seconds", "padding"):
        if old.get(key) != config[key]:
            raise ValueError(f"parent sampling differs: {key}")
    media_options = {k: config[k] for k in ("fps", "max_frames", "max_pixels")}
    media_options["with_audio"] = True
    if old.get("media") != media_options or old.get("allow_revisions", False):
        raise ValueError("parent sampling or revision policy differs")
    if any(old.get(key) != clients[key] for key in ("model", "audio_observer")):
        raise ValueError("parent full model/client configuration differs")
    memory_path = folder / "memory.json"
    if not memory_path.exists():
        return {"files": {}, "completed_windows": 0, "complete": False}
    memory = read(memory_path)
    if memory.get("video_id") != video or memory.get("video_sha256") != expected_media["video_sha256"]:
        raise ValueError("parent memory belongs to different media")
    if memory.get("build_fingerprint") != manifest["fingerprint"]:
        raise ValueError("parent memory/manifest fingerprint mismatch")
    count = math.ceil(memory["duration"] / config["window_seconds"])
    windows = memory.get("completed_windows", [])
    if windows != [f"W{i+1:05d}" for i in range(len(windows))] or len(windows) > count:
        raise ValueError("parent completed windows must be an atomic contiguous prefix")
    if bool(memory.get("complete")) != (len(windows) == count):
        raise ValueError("parent complete marker disagrees with window count")
    module_path = Path(repo) / "methods/longemo" / ("memory.py" if branch == "event" else "noevent_memory.py")
    spec = importlib.util.spec_from_file_location("checkpoint_" + branch, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    replay = module.empty_memory(video, memory["duration"], expected_media["video_sha256"])
    files = {"memory.json": sha(memory_path)}
    for i, window in enumerate(windows):
        relative = "windows/" + window + ".json"
        artifact = read(folder / relative)
        context = artifact["input"]
        core = [i * config["window_seconds"], min((i + 1) * config["window_seconds"], memory["duration"])]
        interval = [max(0, core[0] - config["padding"]), min(memory["duration"], core[1] + config["padding"])]
        if (context.get("window_id") != window or context.get("video_id") != video or
                context.get("core_interval") != core or context.get("media_interval") != interval):
            raise ValueError("parent window ownership/timing is inconsistent")
        replay = module.apply_window(replay, artifact["perception"], window_id=window,
                                     core=core, media=interval, metadata=artifact["sampling"])
        files[relative] = sha(folder / relative)
        audio_path = folder / "audio" / (window + ".json")
        audio_trace = artifact["sampling"].get("audio_observer") or {}
        if audio_trace.get("source_sha256"):
            if not audio_path.is_file() or sha(audio_path) != audio_trace["source_sha256"]:
                raise ValueError("parent committed audio artifact/hash missing")
            cached_audio = read(audio_path)
            if (cached_audio.get("model") != clients["audio_observer"] or
                    cached_audio.get("input_fingerprint") != audio_trace.get("input_fingerprint")):
                raise ValueError("parent committed audio model/input fingerprint mismatch")
            files["audio/" + window + ".json"] = sha(audio_path)
    expected = {k: v for k, v in memory.items() if k not in ("build_fingerprint", "complete")}
    if replay != expected:
        raise ValueError("parent memory differs from replay of committed window artifacts")
    return {"files": files, "completed_windows": len(windows), "complete": bool(memory.get("complete"))}


def inherit_parent(args, run, config):
    """Copy only proven, stopped checkpoints into a distinct one-attempt run."""
    parent = Path(args.parent_run).resolve()
    if parent == run.resolve():
        raise ValueError("parent and new run must be different directories")
    receipt_path = run / "inheritance" / "manifest.json"
    with stopped_parent(parent):
        if receipt_path.exists():
            receipt = read(receipt_path)
            if receipt["parent_configuration_sha256"] != sha(parent / "configuration.json"):
                raise ValueError("parent configuration changed after inheritance")
            for item in receipt["videos"].values():
                source = Path(item["source"])
                for relative, expected in {**item["files"], **item["archived_files"]}.items():
                    if sha(source / relative) != expected:
                        raise ValueError("parent artifacts changed after inheritance")
            return
        old = read(parent / "configuration.json")
        parent_questions = read(parent / "questions.json")
        if old.get("mode") != "pilot" or len(parent_questions) != 50 or old.get("questions_sha256") != digest(parent_questions):
            raise ValueError("parent must be the frozen fixed-50 pilot")
        selected = {q["question_id"]: q for q in read(run / "questions.json")}
        if any(selected.get(q["question_id"], q) != q for q in parent_questions) or not set(selected) <= {q["question_id"] for q in parent_questions}:
            raise ValueError("new question subset differs from frozen parent")
        # Keep this protocol narrow: scored parents require a separate audited
        # import of per-condition predictions and first judgments, never replay.
        if (any((parent / "accepted").glob("*/*.json")) or
                any(records(p) for p in (parent / "scores").glob("*/*/run_*/scores.jsonl")) or
                any(records(p) for p in parent.glob("*/accepted_scores.jsonl"))):
            raise ValueError("parent contains scoring attempts; first-score import needs a separate audit")
        for key in ("model", "base_url", "audio_model", "audio_base_url", "window_seconds", "padding", "fps", "max_frames", "max_pixels", "max_tokens", "timeout"):
            if old.get(key) != config[key]:
                raise ValueError(f"parent protocol differs: {key}")
        clients = {branch: inspect_client_configs(args, config, repo) for branch, repo in config["repos"].items()}
        decisions = {}
        for branch, video in video_tasks(sorted(config["media"])):
            source = memory_root(parent, branch, video) / video
            if old["media"].get(video) != config["media"][video]:
                raise ValueError("parent frozen media hashes differ")
            decisions[branch + "-" + video] = parent_failure(source)
        for video in config["media"]:
            refusal = next((decisions[b + "-" + video] for b in ("event", "noevent")
                            if decisions[b + "-" + video]["status"] == "blocked_content"), None)
            if refusal:
                # The same provider has already refused this media. Changing
                # only the representation is not a reason to submit it again.
                for branch in ("event", "noevent"):
                    decisions[branch + "-" + video] = {**refusal, "scope": "same_provider_both_representations"}
        lineage = {"parent": str(parent), "parent_configuration_sha256": sha(parent / "configuration.json"),
                   "protocol": "one-new-bounded-task-per-video; committed-windows-only; no-score-import", "videos": {}}
        for branch, video in video_tasks(sorted(config["media"])):
            key = branch + "-" + video
            source = memory_root(parent, branch, video) / video
            checkpoint = validate_parent_checkpoint(source, config["repos"][branch], branch, video,
                                                     config["media"][video], config, clients[branch])
            decision = decisions[key]
            archived = [source / "manifest.json", source / "calls.jsonl",
                        *sorted((source / "audio").glob("*.json*")), *sorted((source / "windows").glob("*.json"))]
            archived_hashes = {str(path.relative_to(source)): sha(path) for path in archived
                               if path.is_file() and str(path.relative_to(source)) not in checkpoint["files"]}
            receipt = {"source": str(source), "source_manifest_sha256": sha(source / "manifest.json"),
                       "decision": decision, "archived_files": archived_hashes, **checkpoint}
            item_path = run / "inheritance" / "videos" / (key + ".json")
            destination = memory_root(run, branch, video) / video
            if item_path.exists():
                if read(item_path) != receipt:
                    raise ValueError("parent checkpoint changed during interrupted preparation")
                if any(sha(destination / name) != expected for name, expected in checkpoint["files"].items()):
                    raise ValueError("previously copied checkpoint changed before preparation completed")
            else:
                if destination.exists():
                    raise ValueError("refuse to overwrite unknown checkpoint destination")
                staging = destination.with_name(destination.name + ".import-" + uuid.uuid4().hex)
                staging.mkdir(parents=True)
                for relative, expected in checkpoint["files"].items():
                    target = staging / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source / relative, target)
                    if sha(target) != expected:
                        raise ValueError("checkpoint changed while copying")
                # Keep prior manifests, ledgers and pending audio outside the
                # builder cache. The new builder writes its own manifest/hash.
                history = run / "inheritance" / "history" / key
                history.mkdir(parents=True, exist_ok=True)
                for relative, expected in archived_hashes.items():
                    target = history / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source / relative, target)
                    if sha(target) != expected:
                        raise ValueError("parent history changed while copying")
                staging.replace(destination)
                freeze(item_path, receipt)
            if decision["status"].startswith("blocked"):
                freeze(run / "inheritance" / "blocked" / (key + ".json"), decision)
            lineage["videos"][key] = receipt
        freeze(receipt_path, lineage)


def verify_memory(run, branch, video):
    memory_path = memory_root(run, branch, video) / video / "memory.json"
    expected = read(run / branch / "frozen_memories" / (video + ".json"))
    if sha(memory_path) != expected["memory_sha256"]:
        raise ValueError("frozen perception memory was changed")


def child_plan(args):
    repo, run = Path(args.repo), Path(args.run)
    sys.path.insert(0, str(repo))
    from methods.longemo.runner import parser as method_parser, client_for
    from methods.longemo.common import LoggedClient, fingerprint
    from methods.longemo.prompts import PLANNER
    from methods.longemo.retrieval import validate_plan
    config = read(run / "configuration.json")
    verify_memory(run, args.branch, args.video)
    memory_folder = memory_root(run, args.branch, args.video)
    memory = read(memory_folder / args.video / "memory.json")
    if not memory.get("complete"):
        raise ValueError("cannot plan against incomplete memory")
    folder = run / args.branch / "plans"
    questions_path = run / "questions" / (args.video + ".json")
    options = method_parser().parse_args(["answer", "--data-path", str(questions_path), "--memory-dir", str(memory_folder),
        "--output-dir", str(folder), "--credential-file", args.credential_file, *common_args(config, "plan")])
    client = client_for(options)
    failed = False
    for q in records(questions_path):
        qid = q["question_id"]
        messages = [{"role": "system", "content": PLANNER}, {"role": "user", "content": json.dumps(
            {"question": q["question"], "duration": memory["duration"], "cast": memory["entities"]}, ensure_ascii=False)}]
        signature = fingerprint({"messages": messages, "model": client.configuration()})
        path = folder / (qid + ".json")
        if path.exists():
            cached = read(path)
            if cached["input_fingerprint"] != signature:
                raise ValueError("frozen plan input changed")
            validate_plan(cached["plan"])
            continue
        try:
            plan = LoggedClient(client, folder / "calls" / (qid + ".jsonl"),
                                config.get("stage_tries", {}).get("plan", config.get("tries", 3))).call(
                messages, purpose=f"plan:{qid}", validate=validate_plan)
            freeze(path, {"input_fingerprint": signature, "plan": plan})
        except Exception as exc:
            failed = True
            write(folder / "failures" / (qid + ".json"), {"status": "error", "error_type": type(exc).__name__})
    if failed:
        return 1
    freeze(folder / "frozen" / (args.video + ".json"),
           {q["question_id"]: sha(folder / (q["question_id"] + ".json")) for q in records(questions_path)})
    return 0


def verify_plans(run, branch, video):
    verify_memory(run, branch, video)
    folder = run / branch / "plans"
    frozen = read(folder / "frozen" / (video + ".json"))
    expected_ids = {q["question_id"] for q in records(run / "questions" / (video + ".json"))}
    if set(frozen) != expected_ids:
        raise ValueError("frozen plans do not cover the complete video question subset")
    if any(sha(folder / (qid + ".json")) != expected for qid, expected in frozen.items()):
        raise ValueError("frozen shared plans were changed")


def task_output_is_active(run, stage, condition, video):
    path = run / "tasks" / stage / (condition + "-" + video) / "task.json"
    return path.exists() and read(path).get("status") in ("running", "needs_audit")


def accept_scores(run, condition, questions):
    """Import the earliest valid official score; compare immutable accepted rows."""
    expected = {q["question_id"]: q for q in questions}
    predictions = {}
    for p in (run / "answers" / condition).glob("*/predictions.jsonl"):
        if task_output_is_active(run, "answer", condition, p.parent.name):
            continue
        predictions.update({r["question_id"]: r.get("prediction") for r in records(p) if r.get("status") == "ok"})
    accepted_dir = run / "accepted" / condition
    for path in sorted((run / "scores" / condition).glob("*/run_*/scores.jsonl"), key=lambda p: (p.stat().st_mtime_ns, str(p))):
        if task_output_is_active(run, "score", condition, path.parent.parent.name):
            continue
        for row in records(path):
            qid = row["question_id"]
            if qid not in expected:
                raise ValueError("score contains a question outside the frozen pilot")
            if row.get("status") != "ok":
                continue
            maximum = max(int(k) for k in expected[qid]["rubric"]["scores"])
            if (row.get("prediction") != predictions.get(qid) or row.get("max_score") != maximum or
                not isinstance(row.get("score"), (float, int)) or not 0 <= row["score"] <= maximum or
                not math.isclose(row.get("normalized_score", -1), row["score"] / maximum)):
                raise ValueError("judgment does not match the frozen prediction/rubric")
            target = accepted_dir / (qid + ".json")
            accepted = {"score": row, "source": str(path), "prediction_sha256": digest(predictions[qid])}
            if target.exists():
                first = read(target)
                if first["prediction_sha256"] != accepted["prediction_sha256"]:
                    raise ValueError("prediction changed after its first accepted score")
                if first["source"] == str(path) and first["score"] != row:
                    raise ValueError("first accepted official score was modified")
            else:
                freeze(target, accepted)
    rows = [read(accepted_dir / (q["question_id"] + ".json"))["score"] for q in questions
            if (accepted_dir / (q["question_id"] + ".json")).exists()]
    return rows


def summary(run, questions):
    # Video workers can finish concurrently; accepted scores and status must be
    # published as one coherent snapshot, never via colliding temp filenames.
    with _SUMMARY_LOCK:
        return _summary(run, questions)


def _summary(run, questions):
    result = {"questions": len(questions), "videos": len({q["video_id"] for q in questions}), "conditions": {}}
    for condition in CONDITIONS:
        scores = accept_scores(run, condition, questions)
        predictions = [r for p in (run / "answers" / condition).glob("*/predictions.jsonl")
                       if not task_output_is_active(run, "answer", condition, p.parent.name) for r in records(p)]
        result["conditions"][condition] = {"answers_ok": sum(r.get("status") == "ok" for r in predictions),
            "scored": len(scores), "total": len(questions),
            "mean_scored": 100 * sum(r["normalized_score"] for r in scores) / len(scores) if scores else None}
        (run / condition).mkdir(exist_ok=True)
        (run / condition / "accepted_scores.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in scores))
    result["tasks"] = {str(p.parent.relative_to(run / "tasks")): read(p)["status"] for p in (run / "tasks").glob("*/*/task.json")}
    result["builds"] = {}
    for branch, video in video_tasks(sorted({q["video_id"] for q in questions})):
        path = memory_root(run, branch, video) / video / "memory.json"
        active = task_output_is_active(run, "build", branch, video)
        # Do not race any child output. Running builders expose their live
        # checkpoints in memory.json; this aggregate only publishes terminal
        # task outputs, including failed tasks' committed prefixes.
        memory = read(path) if not active and path.exists() else {}
        result["builds"][branch + "-" + video] = {
            "completed_windows": None if active else len(memory.get("completed_windows", [])),
            "complete": bool(memory.get("complete")), "active_output_deferred": active,
            "task_status": result["tasks"].get("build/" + branch + "-" + video, "pending")}
    result["complete"] = all(value["scored"] == len(questions) for value in result["conditions"].values())
    result["score_semantics"] = "first valid scores; unscored excluded; evidence budget parameter=48000; actual input characters and cumulative tokens separately logged"
    write(run / "status.json", result)
    return result


def execute(args, run, config, questions):
    env = os.environ.copy()
    env["PATH"] = str(Path(args.runtime).resolve() / "bin") + os.pathsep + env.get("PATH", "")
    credentials = read(args.credential_file)
    if not credentials.get("MODEL_API_KEY"):
        raise ValueError("credential file lacks MODEL_API_KEY for the official judge")
    env["MODEL_API_KEY"] = credentials["MODEL_API_KEY"]
    videos = sorted(config["media"])
    script = str(Path(__file__).resolve())

    def completed(stage, key):
        path = run / "tasks" / stage / key / "task.json"
        return path.exists() and read(path)["status"] == "ok"

    def task(stage, branch, video, condition=None):
        repo = config["repos"][branch]
        question_path = str(run / "questions" / (video + ".json"))
        key = (condition or branch) + "-" + video
        inherited_block = run / "inheritance" / "blocked" / (branch + "-" + video + ".json")
        if stage == "build" and inherited_block.exists():
            state = {"status": "blocked", "inherited": read(inherited_block)}
            freeze(run / "tasks" / stage / key / "task.json", state)
            summary(run, questions)
            return
        memory = memory_root(run, branch, video)
        command = [args.python, "-u", "-m", "methods.longemo" if branch == "event" else "methods.longemo.noevent_runner"]
        if stage == "build":
            command += ["build", "--data-path", question_path, "--videos-dir", config["videos_dir"],
                "--subtitles-dir", config["subtitles_dir"], "--output-dir", str(memory),
                "--credential-file", args.credential_file, *common_args(config, "build"), "--workers", "1", "--with-audio",
                "--audio-model", config["audio_model"], "--audio-base-url", config["audio_base_url"],
                "--window-seconds", "20", "--padding", "2", "--fps", "1", "--max-frames", "24", "--max-pixels", "200704"]
        elif stage == "plan":
            if not completed("build", branch + "-" + video):
                return
            verify_memory(run, branch, video)
            command = [args.python, script, "_plan", "--repo", repo, "--run", str(run), "--branch", branch,
                       "--video", video, "--credential-file", args.credential_file]
        elif stage == "answer":
            if not completed("plan", branch + "-" + video):
                return
            verify_plans(run, branch, video)
            command += ["answer", "--data-path", question_path, "--memory-dir", str(memory),
                "--output-dir", str(run / "answers" / condition / video), "--plans-dir", str(run / branch / "plans"),
                "--credential-file", args.credential_file, *common_args(config, "answer"), "--workers", str(args.question_workers),
                "--embedding-backend", "gemini", "--embedding-model", config["embedding_model"],
                "--embedding-base-url", config["embedding_base_url"], "--embedding-cache-dir", str(run / branch / "embeddings"),
                "--evidence-chars", "48000"]
            if branch == "event":
                command += ["--retrieval", "graph" if condition == "base" else "progressive", "--progressive-routing", "none",
                            "--max-inspections", "0", "--progressive-rounds", "3", "--progressive-anchor-k", "4"]
        else:
            predictions = run / "answers" / condition / video / "predictions.jsonl"
            answer_state = run / "tasks" / "answer" / key / "task.json"
            if not predictions.exists() or not answer_state.exists():
                return
            answer_record = read(answer_state)
            if answer_record["status"] in ("running", "needs_audit") or answer_record.get("validation_error"):
                return
            command = [args.python, "-u", "-m", "evaluation.eval", "--data-path", question_path,
                "--predictions", str(predictions), "--granularity", "episode", "--output-dir", str(run / "scores" / condition / video),
                *common_args(config, "score"), "--workers", str(args.question_workers)]
        state = run_command(run / "tasks" / stage / key, command, repo, env)
        if state["status"] == "ok":
            try:
                if stage == "build":
                    memory_path = memory / video / "memory.json"
                    if not read(memory_path).get("complete"):
                        raise ValueError("build returned success with incomplete memory")
                    freeze(run / branch / "frozen_memories" / (video + ".json"), {"memory_sha256": sha(memory_path)})
                elif stage in ("plan", "answer"):
                    verify_plans(run, branch, video)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                # A zero process exit is insufficient evidence of completion.
                # Persist this task's failure and let independent videos finish.
                state = {**state, "status": "error", "validation_error": str(exc),
                         "validation_error_type": type(exc).__name__}
                write(run / "tasks" / stage / key / "task.json", state)
        print(f"{stage} {key}: {state['status']}", flush=True)
        summary(run, questions)

    gate = config.get("gate_video")
    gate_questions = [q for q in questions if q["video_id"] == gate]

    def gate_passed():
        return all(len(accept_scores(run, condition, questions)) >= len(gate_questions) and
                   all((run / "accepted" / condition / (q["question_id"] + ".json")).exists()
                       for q in gate_questions) for condition in CONDITIONS)

    if gate and args.stage != "all" and not gate_passed():
        raise ValueError("unfinished gate must run through all stages before separate stage execution")
    phases = [[gate], [video for video in videos if video != gate]] if gate and args.stage == "all" else [videos]
    stages = ("build", "plan", "answer", "score") if args.stage == "all" else (args.stage,)
    for selected_videos in phases:
        for stage in stages:
            if stage in ("prepare", "status"):
                continue
            def video_pipeline(item):
                branch, video = item
                if stage in ("build", "plan"):
                    task(stage, branch, video)
                elif branch == "event":
                    # Populate the shared event index first, avoiding competing
                    # writes to its cache; method then reuses exactly that index.
                    task(stage, branch, video, "base")
                    task(stage, branch, video, "method")
                else:
                    task(stage, branch, video, "noevent")
            with ThreadPoolExecutor(max_workers=args.build_workers if stage == "build" else args.video_workers) as pool:
                list(pool.map(video_pipeline, video_tasks(selected_videos)))
            summary(run, questions)
        if gate and selected_videos == [gate]:
            passed = gate_passed()
            write(run / "gate.json", {"video_id": gate, "passed": passed,
                  "required": "full video memory and first official scores in all three conditions",
                  "updated_unix": time.time()})
            if not passed:
                print("Full-video gate did not pass; remaining videos remain unstarted.", flush=True)
                return


def video_tasks(videos):
    """Interleave representations so both begin when a bounded pool starts."""
    return [(branch, video) for video in videos for branch in ("event", "noevent")]


def stage_exit_code(stage, result):
    if stage in ("prepare", "status"):
        return 0
    if stage == "all":
        return 0 if result["complete"] else 1
    expected = result["videos"] * (2 if stage in ("build", "plan") else 3)
    states = [state for key, state in result["tasks"].items() if key.startswith(stage + "/")]
    return 0 if len(states) == expected and all(state == "ok" for state in states) else 1


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=STAGES)
    p.add_argument("--method-repo", required=True)
    p.add_argument("--noevent-repo", required=True)
    p.add_argument("--runtime", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--questions", required=True)
    p.add_argument("--mode", choices=("smoke", "pilot"), default="pilot")
    p.add_argument("--video-id")
    p.add_argument("--gate-video", help="Complete this full pilot video through all three scores before starting remaining videos")
    p.add_argument("--parent-run", help="Stopped fixed-50 run to audit and inherit; never alters the parent")
    p.add_argument("--videos-dir")
    p.add_argument("--subtitles-dir")
    p.add_argument("--credential-file", required=True)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--model", default="gpt-6-astra")
    p.add_argument("--base-url", default="https://matrixllm.alipay.com/v1")
    p.add_argument("--audio-model", default="gemini-3.8-flash")
    p.add_argument("--audio-base-url", default="https://www.blackaicoding.com/v1beta")
    p.add_argument("--embedding-model", default="google/gemini-embedding-2")
    p.add_argument("--embedding-base-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--build-tries", type=int, default=3)
    p.add_argument("--plan-tries", type=int, default=3)
    p.add_argument("--answer-tries", type=int, default=3)
    p.add_argument("--build-workers", type=int, default=12,
                   help="total simultaneous builds across both representations; 12 targets about 6 per branch initially; respect combined provider limits")
    p.add_argument("--video-workers", type=int, default=6)
    p.add_argument("--question-workers", type=int, default=2)
    return p


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_clients":
        p = argparse.ArgumentParser()
        p.add_argument("--repo", required=True)
        p.add_argument("--credential-file", required=True)
        return child_clients(p.parse_args(argv[1:]))
    if argv and argv[0] == "_plan":
        p = argparse.ArgumentParser()
        for name in ("repo", "run", "branch", "video", "credential-file"):
            p.add_argument("--" + name, required=True)
        return child_plan(p.parse_args(argv[1:]))
    args = parser().parse_args(argv)
    if min(args.build_workers, args.video_workers, args.question_workers,
           args.build_tries, args.plan_tries, args.answer_tries) < 1:
        raise ValueError("concurrency and bounded tries must be positive")
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        raise ValueError("run name must be one directory name")
    runtime = Path(args.runtime).resolve()
    args.videos_dir = args.videos_dir or str(runtime / "data/episode/videos")
    args.subtitles_dir = args.subtitles_dir or str(runtime / "data/prepared_subtitles")
    args.credential_file = str(Path(args.credential_file).resolve())
    run = runtime / "runs" / args.run_name
    with run_lock(run):
        config, questions = prepare(args, run)
        if args.stage not in ("prepare", "status"):
            execute(args, run, config, questions)
        result = summary(run, questions)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return stage_exit_code(args.stage, result)


if __name__ == "__main__":
    raise SystemExit(main())
