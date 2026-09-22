"""Frozen full-dataset evaluation: noevent first, then a shared event graph.

Preparation is offline and can overlap a running pilot/data producer. Model
requests start only after a stopped-parent snapshot is audited. Each video then
flows through planning, answering and its first official judgments immediately.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import math
import os
from pathlib import Path
import shutil
import socket
import sys
import threading
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.zyf.matched_pilot import (
    assert_identity_stopped, common_args, digest, freeze, memory_root, read,
    records, run_command, run_lock, sha, source_hash, write,
)

CONDITIONS = ("noevent", "base", "method")
_STATUS_LOCK = threading.RLock()


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))
    temporary.replace(path)


def execution_conditions(config):
    scope = config.get("_execution_scope")
    return tuple(scope["manifest"]["conditions"]) if scope else CONDITIONS


def original_coordinator(config):
    return Path(config["repos"]["noevent"]) / "experiments/zyf/full_suite.py"


def _scope_manifest(path):
    value = read(path)
    required = {"schema_version", "conditions", "previous_configuration_sha256", "coordinator_file",
                "coordinator_sha256", "reason", "created_at", "previous_coordinator"}
    if (not isinstance(value, dict) or set(value) != required or type(value.get("schema_version")) is not int or value.get("schema_version") != 1 or
            value.get("conditions") != ["noevent", "method"] or
            value.get("reason") != "user_requested_no_new_base"):
        raise ValueError("execution scope must only disable new base work")
    for name in ("previous_configuration_sha256", "coordinator_sha256"):
        fingerprint = value[name]
        if (not isinstance(fingerprint, str) or len(fingerprint) != 64 or
                any(c not in "0123456789abcdef" for c in fingerprint)):
            raise ValueError("execution scope has an invalid SHA256: " + name)
    if not isinstance(value["created_at"], str) or not value["created_at"].strip():
        raise ValueError("execution scope requires its creation time")
    identity = value["previous_coordinator"]
    if (not isinstance(identity, dict) or set(identity) != {"pid", "start_ticks"} or
            type(identity.get("pid")) is not int or identity["pid"] <= 0 or
            not str(identity.get("start_ticks", "")).isdigit()):
        raise ValueError("execution scope requires an auditable previous coordinator identity")
    coordinator = value["coordinator_file"]
    if not isinstance(coordinator, str) or not Path(coordinator).is_absolute():
        raise ValueError("execution scope coordinator path must be absolute")
    return value


def prepare_scoped_resume(args, run):
    """Narrow scheduling without rewriting the frozen experiment or its budgets."""
    if args.stage != "run":
        raise ValueError("execution scope is only valid for a stopped-run continuation")
    path = Path(args.execution_scope)
    if (path.is_symlink() or path.resolve() != (run / "execution_scope.json").resolve() or
            not path.is_file()):
        raise ValueError("execution scope must be the run's regular execution_scope.json")
    scope = _scope_manifest(path)
    configuration_path = run / "configuration.json"
    if sha(configuration_path) != scope["previous_configuration_sha256"]:
        raise ValueError("frozen configuration changed before the scope continuation")
    config, questions = read(configuration_path), read(run / "questions.json")
    if config.get("mode") != "full" or "_execution_scope" in config:
        raise ValueError("execution scope requires an unchanged frozen full configuration")
    if (digest(questions) != config["questions_sha256"] or records(args.questions) != questions or
            len(questions) != config["question_count"] or args.expected_questions != config["question_count"] or
            len({q["video_id"] for q in questions}) != config["video_count"] or
            args.expected_videos != config["video_count"]):
        raise ValueError("scope continuation changed the frozen questions")
    if (sha(args.media_manifest) != config["media_manifest_sha256"] or
            read(run / "media_manifest.json") != read(args.media_manifest)):
        raise ValueError("scope continuation changed the media manifest")
    for name, expected in (("parent_run", config["parent_run"]), ("method_repo", config["repos"]["event"]),
                           ("noevent_repo", config["repos"]["noevent"]), ("videos_dir", config["videos_dir"]),
                           ("subtitles_dir", config["subtitles_dir"]), ("runtime", run.parent.parent)):
        if Path(getattr(args, name)).resolve() != Path(expected).resolve():
            raise ValueError("scope continuation changed the path: " + name)
    if (Path(args.python).resolve() != Path(config["python"]).resolve() or
            Path(args.run_name).name != run.name):
        raise ValueError("scope continuation changed the interpreter or run")
    for name in ("workers", "question_workers", "poll_seconds", "parent_wait_timeout", "media_wait_timeout"):
        if getattr(args, name) != config["execution"][name]:
            raise ValueError("scope continuation changed frozen execution settings: " + name)
    actual_done, expected_done = args.producer_done, config["execution"]["producer_done"]
    if (bool(actual_done) != bool(expected_done) or
            actual_done and Path(actual_done).resolve() != Path(expected_done).resolve()):
        raise ValueError("scope continuation changed the media completion marker")
    credential_paths = set()
    for task_path in (run / "tasks").glob("*/*/task.json"):
        command = read(task_path).get("command", [])
        if "--credential-file" in command:
            credential_paths.add(str(Path(command[command.index("--credential-file") + 1]).resolve()))
    if not credential_paths or credential_paths != {str(Path(args.credential_file).resolve())}:
        raise ValueError("scope continuation changed or cannot verify the original credential path")
    if config.get("stage_tries", {}).get("score") != 1:
        raise ValueError("official scoring budget must remain one attempt")
    assert_identity_stopped({**scope["previous_coordinator"], "host": socket.gethostname()},
                            "previous full-run coordinator")
    config = {**config, "_execution_scope": {"path": str(path.resolve()), "sha256": sha(path), "manifest": scope,
                                            "configuration_path": str(configuration_path.resolve())}}
    verify_runtime_sources(config)
    return config, questions


def prepare(args, run):
    if getattr(args, "execution_scope", None):
        return prepare_scoped_resume(args, run)
    if (run / "execution_scope.json").exists():
        raise ValueError("this run has a narrowed execution scope; pass --execution-scope")
    parent = Path(args.parent_run).resolve()
    if parent == run.resolve():
        raise ValueError("full evaluation needs a new run directory")
    old = read(parent / "configuration.json")
    questions = records(args.questions)
    ids = [q["question_id"] for q in questions]
    videos = sorted({q["video_id"] for q in questions})
    if (len(questions) != args.expected_questions or len(videos) != args.expected_videos or
            len(set(ids)) != len(ids) or any(q.get("granularity") != "episode" for q in questions)):
        raise ValueError("question selection does not match the declared full episode cohort")
    qmap = {q["question_id"]: q for q in questions}
    if any(qmap.get(q["question_id"]) != q for q in read(parent / "questions.json")):
        raise ValueError("parent pilot is not an unchanged subset of the full questions")
    media_manifest = read(args.media_manifest)
    if media_manifest.get("schema_version") != 1 or set(media_manifest.get("videos", {})) != set(videos):
        raise ValueError("media manifest must cover every frozen video exactly once")
    media = media_manifest["videos"]
    for video, row in media.items():
        for key in ("video_sha256", "subtitles_sha256"):
            if not isinstance(row.get(key), str) or len(row[key]) != 64 or any(c not in "0123456789abcdef" for c in row[key]):
                raise ValueError(f"invalid media SHA256: {video}/{key}")
        if any(type(row.get(key)) is not int or row[key] < 0 for key in ("video_bytes", "subtitles_bytes")) or row["video_bytes"] == 0:
            raise ValueError("media manifest requires file sizes")
        if video in old["media"] and any(old["media"][video][k] != row[k] for k in ("video_sha256", "subtitles_sha256")):
            raise ValueError("parent media differs from the frozen full manifest")
    repos = {"event": str(Path(args.method_repo).resolve()), "noevent": str(Path(args.noevent_repo).resolve())}
    hashes = {branch: source_hash(Path(repo)) for branch, repo in repos.items()}
    if hashes != old["source_hashes"]:
        raise ValueError("full evaluation must preserve the pilot's method/evaluator source hashes")
    config = {**old, "mode": "full", "gate_video": None, "questions_sha256": digest(questions),
              "question_count": len(questions), "video_count": len(videos), "media": media,
              "media_manifest_sha256": sha(args.media_manifest), "repos": repos, "source_hashes": hashes,
              "parent_run": str(parent), "parent_configuration_sha256": sha(parent / "configuration.json"),
              "videos_dir": str(Path(args.videos_dir).resolve()), "subtitles_dir": str(Path(args.subtitles_dir).resolve()),
              "suite_sha256": sha(__file__), "python": args.python,
              "suite_support_hashes": {name: sha(Path(__file__).with_name(name)) for name in
                                       ("matched_pilot.py", "full_suite_inheritance.py")},
              "execution": {"phase_order": ["noevent", "event"], "workers": args.workers,
                            "question_workers": args.question_workers, "poll_seconds": args.poll_seconds,
                            "parent_wait_timeout": args.parent_wait_timeout,
                            "media_wait_timeout": args.media_wait_timeout,
                            "producer_done": str(Path(args.producer_done).resolve()) if args.producer_done else None}}
    if config.get("stage_tries", {}).get("score", 1) != 1:
        raise ValueError("official scoring budget must remain one attempt")
    freeze(run / "configuration.json", config)
    freeze(run / "questions.json", questions)
    freeze(run / "media_manifest.json", media_manifest)
    for video in videos:
        freeze(run / "questions" / (video + ".json"), [q for q in questions if q["video_id"] == video])
    return config, questions


def parent_wait(args, run, config, questions):
    from experiments.zyf.full_suite_inheritance import snapshot_parent
    timing = run / "timing.json"
    if not timing.exists():
        freeze(timing, {"started_unix": time.time()})
    deadline = read(timing)["started_unix"] + config["execution"]["parent_wait_timeout"]
    while True:
        try:
            index = snapshot_parent(Path(config["parent_run"]), run, questions, config=config)
            write(run / "parent_status.json", {"status": "audited", "time_unix": time.time()})
            return index
        except ValueError as exc:
            # Only positive evidence of an active parent is a wait condition.
            # Corrupt or unverifiable provenance requires intervention.
            if not any(fragment in str(exc) for fragment in ("still active", "still owns its lock", "still running")):
                raise
            write(run / "parent_status.json", {"status": "waiting", "reason": str(exc), "time_unix": time.time()})
            if time.time() >= deadline:
                raise TimeoutError("parent did not finish within the frozen wait interval") from None
            time.sleep(min(config["execution"]["poll_seconds"], max(0.01, deadline - time.time())))


def verify_runtime_sources(config):
    """Recheck the actual new deployments after waiting and before each stage."""
    scope = config.get("_execution_scope")
    coordinator = Path(__file__)
    original = coordinator
    if scope:
        manifest = _scope_manifest(Path(scope["path"]))
        if sha(scope["path"]) != scope["sha256"] or manifest != scope["manifest"]:
            raise ValueError("execution scope changed after continuation preparation")
        if sha(scope["configuration_path"]) != manifest["previous_configuration_sha256"]:
            raise ValueError("frozen configuration changed during the scope continuation")
        if (Path(manifest["coordinator_file"]).resolve() != coordinator.resolve() or
                sha(coordinator) != manifest["coordinator_sha256"]):
            raise ValueError("scope continuation coordinator changed")
        original = original_coordinator(config)
        if original.resolve() == coordinator.resolve():
            raise ValueError("scope continuation requires a distinct coordinator deployment")
    if sha(original) != config["suite_sha256"]:
        raise ValueError("full-suite coordinator changed after configuration freezing")
    for name, expected in config["suite_support_hashes"].items():
        if (sha(original.with_name(name)) != expected or
                scope and sha(coordinator.with_name(name)) != expected):
            raise ValueError("full-suite support source changed: " + name)
    for branch, repo in config["repos"].items():
        if source_hash(Path(repo)) != config["source_hashes"][branch]:
            raise ValueError("method/evaluator source changed after preparation: " + branch)


def copy_verified_tree(source, destination):
    """Use a distinct destination; verify every copied file before publication."""
    source, destination = Path(source), Path(destination)
    inventory = {str(p.relative_to(source)): sha(p) for p in source.rglob("*") if p.is_file()}
    if destination.exists():
        actual = {str(p.relative_to(destination)): sha(p) for p in destination.rglob("*") if p.is_file()}
        if actual != inventory:
            raise ValueError("refuse to overwrite an unknown inherited cache directory")
        return
    staging = destination.with_name(destination.name + f".import-{os.getpid()}")
    if staging.exists():
        raise ValueError("interrupted cache import requires an audit")
    staging.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, staging)
    if {str(p.relative_to(staging)): sha(p) for p in staging.rglob("*") if p.is_file()} != inventory:
        raise ValueError("inherited files changed during copying")
    staging.replace(destination)


def materialize(run, config, index):
    """Copy immutable parent successes once; old per-video answer manifests stay historical."""
    marker = run / "inheritance" / "materialized.json"
    if marker.exists():
        return
    for branch in ("event", "noevent"):
        inherited = index["branches"][branch]
        cache = inherited.get("embeddings_dir")
        if cache and Path(cache).exists():
            copy_verified_tree(cache, run / branch / "embeddings")
        for video, value in inherited["videos"].items():
            if value["state"] == "reusable":
                destination = memory_root(run, branch, video) / video
                copy_verified_tree(value["memory_dir"], destination)
                freeze(run / branch / "frozen_memories" / (video + ".json"), {"memory_sha256": sha(destination / "memory.json")})
            for qid, plan in value.get("plans", {}).items():
                if plan["state"] == "success":
                    artifact = plan["artifact"]
                    if sha(artifact["path"]) != artifact["sha256"]:
                        raise ValueError("inherited plan snapshot changed")
                    freeze(run / branch / "plans" / (qid + ".json"), read(artifact["path"]))
    for condition in CONDITIONS:
        for qid, item in index["conditions"][condition]["questions"].items():
            if item.get("accepted"):
                freeze(run / "accepted" / condition / (qid + ".json"), item["accepted"]["envelope"])
    freeze(marker, {"parent_index_sha256": sha(run / "inheritance" / "parent_index.json"),
                    "completed_unix": time.time(), "successful_memory_hashes_preserved": True})


def verify_memory(run, branch, video):
    path = memory_root(run, branch, video) / video / "memory.json"
    value = read(path)
    if not value.get("complete") or sha(path) != read(run / branch / "frozen_memories" / (video + ".json"))["memory_sha256"]:
        raise ValueError("memory is incomplete or changed after freezing")
    return value


def child_plan(args):
    run, repo = Path(args.run), Path(args.repo)
    sys.path.insert(0, str(repo))
    from methods.longemo.runner import parser as model_parser, client_for
    from methods.longemo.common import LoggedClient, fingerprint
    from methods.longemo.prompts import PLANNER
    from methods.longemo.retrieval import validate_plan
    config = read(run / "configuration.json")
    memory = verify_memory(run, args.branch, args.video)
    questions = records(args.questions)
    options = model_parser().parse_args(["answer", "--data-path", args.questions,
        "--memory-dir", str(memory_root(run, args.branch, args.video)), "--output-dir", str(run / args.branch / "plans"),
        "--credential-file", args.credential_file, *common_args(config, "plan")])
    client = client_for(options)
    folder = run / args.branch / "plans"

    def one(q):
        qid = q["question_id"]
        messages = [{"role": "system", "content": PLANNER}, {"role": "user", "content": json.dumps(
            {"question": q["question"], "duration": memory["duration"], "cast": memory["entities"]}, ensure_ascii=False)}]
        signature = fingerprint({"messages": messages, "model": client.configuration()})
        destination = folder / (qid + ".json")
        try:
            if destination.exists():
                value = read(destination)
                if value["input_fingerprint"] != signature:
                    raise ValueError("inherited plan input differs")
                validate_plan(value["plan"])
            else:
                value = {"input_fingerprint": signature,
                         "plan": LoggedClient(client, folder / "calls" / (qid + ".jsonl"),
                                              config["stage_tries"]["plan"]).call(
                                                  messages, purpose="plan:" + qid, validate=validate_plan)}
                freeze(destination, value)
            return {"question_id": qid, "status": "ok", "sha256": sha(destination)}
        except Exception as exc:
            return {"question_id": qid, "status": "error", "error_type": type(exc).__name__}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        result = list(pool.map(one, questions))
    write_rows(Path(args.results), result)
    return int(any(row["status"] != "ok" for row in result))


def inherited_question(index, condition, qid):
    return index["conditions"][condition]["questions"].get(qid, {})


def prediction_inventory(run, index, condition, video):
    predictions = {}
    for q in records(run / "questions" / (video + ".json")):
        previous = inherited_question(index, condition, q["question_id"])
        if previous.get("prediction"):
            row = previous["prediction"]["row"]
            if row.get("status") == "ok":
                predictions[q["question_id"]] = row
    path = run / "answers" / condition / video / "predictions.jsonl"
    seen = set()
    for row in records(path):
        qid = row["question_id"]
        if qid in seen:
            raise ValueError("duplicate question ID in new predictions")
        seen.add(qid)
        if row.get("status") != "ok":
            continue
        if qid in predictions:
            raise ValueError("a previously successful parent prediction was attempted again")
        predictions[qid] = row
    return predictions


def accept_video_scores(run, index, condition, video, questions):
    qmap = {q["question_id"]: q for q in questions}
    predictions = prediction_inventory(run, index, condition, video)
    seen = set()
    for source in sorted((run / "scores" / condition / video).glob("run_*/scores.jsonl"), key=lambda p: (p.stat().st_mtime_ns, str(p))):
        for row in records(source):
            qid = row["question_id"]
            if qid not in qmap:
                raise ValueError("official output contains a question outside this video")
            if qid in seen:
                raise ValueError("duplicate question ID in new official scores")
            seen.add(qid)
            if row.get("status") != "ok":
                continue
            maximum = max(int(k) for k in qmap[qid]["rubric"]["scores"])
            if (row.get("prediction") != predictions.get(qid, {}).get("prediction") or row.get("max_score") != maximum or
                    type(row.get("score")) not in (int, float) or not 0 <= row["score"] <= maximum or
                    not math.isclose(row.get("normalized_score", -1), row["score"] / maximum)):
                raise ValueError("official score does not match its prediction and frozen rubric")
            envelope = {"score": row, "source": str(source), "prediction_sha256": digest(row["prediction"])}
            destination = run / "accepted" / condition / (qid + ".json")
            if destination.exists():
                old = read(destination)
                if old["source"] != str(source) or old["prediction_sha256"] != envelope["prediction_sha256"] or old["score"] != row:
                    raise ValueError("a question acquired a second/different official score")
            else:
                freeze(destination, envelope)


def recover_stopped_scores(run, config, index):
    """Import already-written first judgments after child-death verification.

    A coordinator crash between the scorer's atomic write and task completion
    must not discard valid judgments. Ambiguous tasks remain ambiguous; this
    function performs no model call and never changes their attempt records.
    """
    full = {q["question_id"]: q for q in records(run / "questions.json")}
    imported = []
    for condition in CONDITIONS:
        for video in config["media"]:
            score_files = list((run / "scores" / condition / video).glob("run_*/scores.jsonl"))
            if not score_files:
                continue
            key = condition + "-" + video
            task_path = run / "tasks/score" / key / "task.json"
            question_path = run / "stage_questions/score" / (key + ".json")
            input_path = run / "score_inputs" / condition / (video + ".jsonl")
            if not task_path.exists() or not question_path.exists() or not input_path.exists():
                raise ValueError("orphan score output lacks a frozen invocation")
            questions = records(question_path)
            if any(q != full.get(q["question_id"]) or q["video_id"] != video for q in questions):
                raise ValueError("recovered judge question subset differs from the frozen full cohort")
            predictions = prediction_inventory(run, index, condition, video)
            inputs = records(input_path)
            if (len({row["question_id"] for row in inputs}) != len(inputs) or
                    {row["question_id"] for row in inputs} != {q["question_id"] for q in questions} or
                    any(row.get("prediction") != predictions.get(row["question_id"], {}).get("prediction") for row in inputs)):
                raise ValueError("recovered judge input differs from its first successful predictions")
            accept_video_scores(run, index, condition, video, questions)
            imported.append({"condition": condition, "video_id": video,
                             "source_sha256": {str(path): sha(path) for path in score_files}})
    if imported:
        write(run / "offline_score_imports.json", {"verified_stopped_children": True, "imports": imported})


def summarize(run, config):
    with _STATUS_LOCK:
        rows = records(run / "questions.json")
        qids = {q["question_id"] for q in rows}
        result = {"run": run.name, "updated_unix": time.time(), "questions": len(rows), "videos": config["video_count"],
                  "phase_order": ["noevent", "event"], "conditions": {}, "pipelines": {}}
        for condition in CONDITIONS:
            accepted = [read(path)["score"] for path in sorted((run / "accepted" / condition).glob("*.json"))]
            if any(row["question_id"] not in qids for row in accepted) or len({r["question_id"] for r in accepted}) != len(accepted):
                raise ValueError("accepted scores do not match the full frozen cohort")
            result["conditions"][condition] = {"scored": len(accepted), "total": len(rows),
                "mean_scored": 100 * sum(r["normalized_score"] for r in accepted) / len(accepted) if accepted else None}
            write_rows(run / condition / "accepted_scores.jsonl", accepted)
        for branch in ("noevent", "event"):
            statuses = {p.stem: read(p) for p in (run / "pipelines" / branch).glob("*.json")}
            result["pipelines"][branch] = {"videos": statuses,
                "finished": sum(row["status"] != "running" for row in statuses.values()),
                "running": sum(row["status"] == "running" for row in statuses.values()),
                "pending": config["video_count"] - len(statuses)}
        enabled = execution_conditions(config)
        result["execution_conditions"] = list(enabled)
        result["retained_reference_conditions"] = [condition for condition in CONDITIONS if condition not in enabled]
        if config.get("_execution_scope"):
            result["execution_scope_sha256"] = config["_execution_scope"]["sha256"]
        result["complete"] = all(result["conditions"][condition]["scored"] == len(rows) for condition in enabled)
        result["semantics"] = "first valid official scores; missing excluded; parent successes preserved by question and condition"
        write(run / "status.json", result)
        return result


def stage_selection(run, stage, key, questions):
    path = run / "stage_questions" / stage / (key + ".json")
    freeze(path, questions)
    return path


def stage_run(args, run, config, stage, key, command, repo, env):
    verify_runtime_sources(config)
    if stage in ("answer", "score") and key.startswith("base-") and "base" not in execution_conditions(config):
        raise ValueError("new base answer/score work is disabled by the execution scope")
    state = run_command(run / "tasks" / stage / key, command, repo, env)
    if state["status"] in ("running", "needs_audit"):
        raise RuntimeError("previous stage outcome is ambiguous; no dependent requests are allowed")
    return state


def video_pipeline(args, run, config, index, branch, video):
    questions = records(run / "questions" / (video + ".json"))
    inherited = index["branches"][branch]["videos"].get(video, {"state": "unstarted", "plans": {}})
    repo = config["repos"][branch]
    key = branch + "-" + video
    state_path = run / "pipelines" / branch / (video + ".json")
    write(state_path, {"status": "running", "started_unix": time.time()})
    env = os.environ.copy()
    env["PATH"] = str(Path(args.runtime) / "bin") + os.pathsep + env.get("PATH", "")
    env["MODEL_API_KEY"] = read(args.credential_file)["MODEL_API_KEY"]
    module = "methods.longemo" if branch == "event" else "methods.longemo.noevent_runner"
    memory = memory_root(run, branch, video)
    try:
        if inherited["state"] == "blocked":
            write(state_path, {"status": "blocked_parent_build", "reason": inherited.get("reason", "parent build attempt failed or is unresolved")})
            return
        if inherited["state"] != "reusable":
            command = [args.python, "-u", "-m", module, "build", "--data-path", str(run / "questions" / (video + ".json")),
                "--videos-dir", config["videos_dir"], "--subtitles-dir", config["subtitles_dir"], "--output-dir", str(memory),
                "--credential-file", args.credential_file, *common_args(config, "build"), "--workers", "1", "--with-audio",
                "--audio-model", config["audio_model"], "--audio-base-url", config["audio_base_url"],
                "--window-seconds", str(config["window_seconds"]), "--padding", str(config["padding"]),
                "--fps", str(config["fps"]), "--max-frames", str(config["max_frames"]), "--max-pixels", str(config["max_pixels"])]
            built = stage_run(args, run, config, "build", key, command, repo, env)
            if built["status"] != "ok":
                write(state_path, {"status": "build_failed", "task": str(run / "tasks/build" / key / "task.json")})
                return
            if not read(memory / video / "memory.json").get("complete"):
                raise ValueError("build exited successfully with incomplete memory")
            freeze(run / branch / "frozen_memories" / (video + ".json"), {"memory_sha256": sha(memory / video / "memory.json")})
        verify_memory(run, branch, video)
        plan_states = inherited.get("plans", {})
        pending_plans = [q for q in questions if plan_states.get(q["question_id"], {}).get("state", "unstarted") == "unstarted"]
        plan_questions = stage_selection(run, "plan", key, pending_plans)
        if pending_plans:
            plan_coordinator = original_coordinator(config) if config.get("_execution_scope") else Path(__file__)
            command = [args.python, str(plan_coordinator.resolve()), "_plan", "--repo", repo, "--run", str(run),
                "--branch", branch, "--video", video, "--questions", str(plan_questions),
                "--results", str(run / branch / "plans/results" / (video + ".jsonl")),
                "--credential-file", args.credential_file, "--workers", str(args.question_workers)]
            stage_run(args, run, config, "plan", key, command, repo, env)
        available_plans = {}
        for q in questions:
            qid = q["question_id"]
            path = run / branch / "plans" / (qid + ".json")
            if plan_states.get(qid, {}).get("state") != "blocked" and path.exists():
                available_plans[qid] = sha(path)
        freeze(run / branch / "plans/frozen" / (video + ".json"), available_plans)
        conditions = tuple(condition for condition in (("noevent",) if branch == "noevent" else ("base", "method"))
                           if condition in execution_conditions(config))
        embedding_guard = inherited.get("embedding_guard", {"state": "clear"})
        blocked_answers = {}
        for condition in conditions:
            ckey = condition + "-" + video
            pending_answers = [q for q in questions if q["question_id"] in available_plans and
                               inherited_question(index, condition, q["question_id"]).get("answer_state", "unstarted") == "unstarted"]
            if embedding_guard["state"] == "needs_audit":
                blocked_answers[condition] = [q["question_id"] for q in pending_answers]
                pending_answers = []
            answer_questions = stage_selection(run, "answer", ckey, pending_answers)
            if pending_answers:
                command = [args.python, "-u", "-m", module, "answer", "--data-path", str(answer_questions),
                    "--memory-dir", str(memory), "--output-dir", str(run / "answers" / condition / video),
                    "--plans-dir", str(run / branch / "plans"), "--credential-file", args.credential_file,
                    *common_args(config, "answer"), "--workers", str(args.question_workers),
                    "--embedding-backend", "gemini", "--embedding-model", config["embedding_model"],
                    "--embedding-base-url", config["embedding_base_url"], "--embedding-cache-dir", str(run / branch / "embeddings"),
                    "--evidence-chars", str(config["evidence_chars"])]
                if branch == "event":
                    command += ["--retrieval", "graph" if condition == "base" else "progressive", "--progressive-routing", "none",
                        "--max-inspections", "0", "--progressive-rounds", str(config["progressive_rounds"]),
                        "--progressive-anchor-k", str(config["progressive_anchor_k"])]
                stage_run(args, run, config, "answer", ckey, command, repo, env)
            verify_memory(run, branch, video)
            if any(sha(run / branch / "plans" / (qid + ".json")) != expected for qid, expected in available_plans.items()):
                raise ValueError("shared plans changed during answering")
            predictions = prediction_inventory(run, index, condition, video)
            pending_scores = [q for q in questions if q["question_id"] in predictions and
                              inherited_question(index, condition, q["question_id"]).get("score_state", "unstarted") == "unstarted"]
            score_questions = stage_selection(run, "score", ckey, pending_scores)
            if pending_scores:
                score_input = run / "score_inputs" / condition / (video + ".jsonl")
                selected = [predictions[q["question_id"]] for q in pending_scores]
                if score_input.exists():
                    if records(score_input) != selected:
                        raise ValueError("first-judgment predictions changed")
                else:
                    write_rows(score_input, selected)
                command = [args.python, "-u", "-m", "evaluation.eval", "--data-path", str(score_questions),
                    "--predictions", str(score_input), "--granularity", "episode", "--output-dir", str(run / "scores" / condition / video),
                    *common_args(config, "score"), "--workers", str(args.question_workers)]
                stage_run(args, run, config, "score", ckey, command, repo, env)
                accept_video_scores(run, index, condition, video, questions)
            summarize(run, config)
        completed = {condition: sum((run / "accepted" / condition / (q["question_id"] + ".json")).exists() for q in questions)
                     for condition in conditions}
        write(state_path, {"status": "complete" if all(n == len(questions) for n in completed.values()) else "partial",
                           "scored": completed, "questions": len(questions), "finished_unix": time.time(),
                           "embedding_guard": embedding_guard, "embedding_blocked_answers": blocked_answers})
    except Exception as exc:
        write(state_path, {"status": "needs_audit", "error_type": type(exc).__name__, "error": str(exc)[:500], "finished_unix": time.time()})
    finally:
        summarize(run, config)


def media_status(config, video):
    expected = config["media"][video]
    for folder, suffix, prefix in (("videos_dir", ".mp4", "video"), ("subtitles_dir", ".json", "subtitles")):
        path = Path(config[folder]) / (video + suffix)
        if not path.exists():
            return "waiting"
        if not path.is_file() or path.stat().st_size != expected[prefix + "_bytes"] or sha(path) != expected[prefix + "_sha256"]:
            return "integrity_failed"
    return "ready"


def producer_finished(config):
    marker = config["execution"]["producer_done"]
    if not marker or not Path(marker).exists():
        return False
    value = read(marker)
    if value.get("media_manifest_sha256") != config["media_manifest_sha256"] or value.get("complete") is not True:
        raise ValueError("producer_done does not identify the frozen complete media manifest")
    return True


def run_phase(args, run, config, index, branch):
    phase_path = run / "phases" / (branch + ".json")
    if not phase_path.exists():
        freeze(phase_path, {"started_unix": time.time()})
    deadline = read(phase_path)["started_unix"] + config["execution"]["media_wait_timeout"]
    remaining = set(config["media"])
    active = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while remaining or active:
            for video in sorted(remaining):
                state = run / "pipelines" / branch / (video + ".json")
                if state.exists() and read(state)["status"] != "running":
                    remaining.remove(video)
                    continue
                inherited = index["branches"][branch]["videos"].get(video, {"state": "unstarted"})
                if inherited["state"] == "blocked":
                    write(state, {"status": "blocked_parent_build", "reason": inherited.get("reason", "parent failed/unresolved build attempt")})
                    remaining.remove(video)
                    continue
                if len(active) >= args.workers:
                    break
                readiness = media_status(config, video)
                if readiness == "ready":
                    active[pool.submit(video_pipeline, args, run, config, index, branch, video)] = video
                    remaining.remove(video)
                elif readiness == "integrity_failed":
                    write(state, {"status": "media_integrity_failed", "finished_unix": time.time()})
                    remaining.remove(video)
                elif producer_finished(config) or time.time() >= deadline:
                    write(state, {"status": "media_missing", "reason": "producer_done" if producer_finished(config) else "frozen_wait_timeout"})
                    remaining.remove(video)
            summarize(run, config)
            if active:
                done, _ = wait(active, timeout=config["execution"]["poll_seconds"], return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
                    del active[future]
            elif remaining:
                time.sleep(config["execution"]["poll_seconds"])
    write(run / "phases" / (branch + "_finished.json"), {"finished_unix": time.time()})


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "run", "status"))
    for name in ("parent-run", "questions", "media-manifest", "method-repo", "noevent-repo", "runtime", "run-name", "credential-file"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--execution-scope", help="immutable run/execution_scope.json for an authorized narrower continuation")
    p.add_argument("--videos-dir")
    p.add_argument("--subtitles-dir")
    p.add_argument("--producer-done")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--expected-questions", type=int, default=558)
    p.add_argument("--expected-videos", type=int, default=141)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--question-workers", type=int, default=2)
    p.add_argument("--poll-seconds", type=float, default=10)
    p.add_argument("--parent-wait-timeout", type=float, default=86400)
    p.add_argument("--media-wait-timeout", type=float, default=21600)
    return p


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_plan":
        p = argparse.ArgumentParser()
        for name in ("repo", "run", "branch", "video", "questions", "results", "credential-file"):
            p.add_argument("--" + name, required=True)
        p.add_argument("--workers", type=int, required=True)
        return child_plan(p.parse_args(argv[1:]))
    args = parser().parse_args(argv)
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        raise ValueError("run name must be one directory name")
    if min(args.workers, args.question_workers, args.poll_seconds, args.parent_wait_timeout, args.media_wait_timeout) <= 0:
        raise ValueError("concurrency and waits must be positive")
    runtime = Path(args.runtime).resolve()
    args.videos_dir = args.videos_dir or str(runtime / "data/episode/videos")
    args.subtitles_dir = args.subtitles_dir or str(runtime / "data/prepared_subtitles")
    args.credential_file = str(Path(args.credential_file).resolve())
    run = runtime / "runs" / args.run_name
    with run_lock(run):
        config, questions = prepare(args, run)
        if args.stage == "run":
            for path in (run / "tasks").glob("*/*/task.json"):
                state = read(path)
                if state.get("child"):
                    assert_identity_stopped(state["child"], str(path))
                elif state.get("status") in ("running", "needs_audit"):
                    raise ValueError("a previous task lacks an auditable child identity; inspect it before starting new requests")
            index = parent_wait(args, run, config, questions)
            verify_runtime_sources(config)
            materialize(run, config, index)
            recover_stopped_scores(run, config, index)
            for branch in ("noevent", "event"):
                run_phase(args, run, config, index, branch)
        result = summarize(run, config)
        print(json.dumps(result["conditions"], ensure_ascii=False))
    return 0 if args.stage != "run" or result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
