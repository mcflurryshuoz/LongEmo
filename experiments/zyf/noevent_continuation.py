"""A separately frozen, one-task continuation of unscored noevent videos.

The parent is never repaired or restarted.  Only validated window records are
inherited; the original noevent builder, retrieval and official judge run from
the frozen core checkout.  This protocol permits a new visual output-token
budget, not a change of representation, models, media sampling or judge.

Selection JSON: {"schema_version": 1, "condition": "noevent",
                 "question_ids": [...], "reason": "authorized continuation"}.
prepare/run need the same arguments.  report only needs --run (and --output).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import uuid

from experiments.zyf import matched_pilot as pilot
from experiments.zyf.matched_pilot import (
    assert_identity_stopped, digest, freeze, memory_root, read, records, run_command,
    run_lock, sha, source_hash, stopped_parent, validate_parent_checkpoint, write,
)

_REPORT_LOCK = threading.RLock()
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_REFUSALS = {"contentfilter", "contentpolicyviolation", "responsibleaipolicyviolation",
             "safety", "prohibitedcontent", "blockedprompt", "blockedinput",
             "insufficientquota", "insufficientcredits", "insufficientbalance",
             "authenticationerror", "invalidapikey", "permissiondenied", "unauthorized", "forbidden"}


def regular(path):
    """Reject symlinks, including an ancestor alias, before trusting a hash."""
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError(f"expected a regular file without symlink ancestors: {path}")
    return path


def inventory(directory):
    directory = Path(directory).absolute()
    if any(p.is_symlink() for p in (directory, *directory.parents)) or not directory.is_dir():
        raise ValueError(f"expected an ordinary directory: {directory}")
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("symlink in checkpoint inventory")
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = sha(path)
    return result


def verify_inventory(directory, expected):
    if inventory(directory) != expected:
        raise ValueError(f"immutable source inventory changed: {directory}")


def media_identity(path):
    path = regular(path)
    value = path.stat()
    return {"path": str(path), "size": value.st_size, "mtime_ns": value.st_mtime_ns,
            "ctime_ns": value.st_ctime_ns, "inode": value.st_ino, "device": value.st_dev}


def verified_media(old, videos):
    result = {}
    for video in videos:
        result[video] = {}
        for kind, directory, suffix in (("video", old["videos_dir"], ".mp4"),
                                         ("subtitles", old["subtitles_dir"], ".json")):
            path = Path(directory) / (video + suffix)
            before = media_identity(path)
            if sha(path) != old["media"][video][kind + "_sha256"] or media_identity(path) != before:
                raise ValueError("media changed since the original experiment")
            result[video][kind] = before
    return result


def selected_questions(selection, questions, protected):
    if (not isinstance(selection, dict) or set(selection) != {"schema_version", "condition", "question_ids", "reason"}
            or selection["schema_version"] != 1 or selection["condition"] != "noevent"):
        raise ValueError("selection must name only the noevent condition")
    if not isinstance(selection["reason"], str) or not selection["reason"].strip():
        raise ValueError("an explicit continuation reason is required")
    ids = selection["question_ids"]
    if (not isinstance(ids, list) or not ids or any(not isinstance(q, str) or not _SAFE_ID.fullmatch(q) for q in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("selection requires unique safe question IDs")
    qmap = {q["question_id"]: q for q in questions}
    if len(qmap) != len(questions) or not set(ids) <= set(qmap):
        raise ValueError("selection contains duplicate or unknown parent questions")
    if set(ids) & set(protected):
        raise ValueError("selection overlaps protected first scores")
    result = [qmap[q] for q in ids]
    if any(q.get("granularity") != "episode" or not _SAFE_ID.fullmatch(q["video_id"]) for q in result):
        raise ValueError("only safe episode video IDs are supported")
    return result


def protected_scores(parent, questions, expected_count):
    qmap = {q["question_id"]: q for q in questions}
    result = {}
    for path in sorted((parent / "accepted/noevent").glob("*.json")):
        regular(path)
        value = read(path)
        row, qid = value["score"], path.stem
        if qid not in qmap or row.get("question_id") != qid or row.get("status") != "ok":
            raise ValueError("invalid parent first-score envelope")
        maximum = max(int(k) for k in qmap[qid]["rubric"]["scores"])
        if (row.get("max_score") != maximum or type(row.get("score")) not in (int, float)
                or not 0 <= row["score"] <= maximum
                or not math.isclose(row.get("normalized_score", -1), row["score"] / maximum)):
            raise ValueError("parent score differs from its frozen rubric")
        original = regular(value["source"])
        if row not in records(original):
            raise ValueError("parent first score is missing from its official source")
        result[qid] = {"path": str(path), "sha256": sha(path), "envelope": value,
                       "official_source_sha256": sha(original)}
    if len(result) != expected_count:
        raise ValueError("protected first-score count changed; audit before selecting more work")
    return result


def resolve_checkpoint(parent, video):
    candidate = memory_root(parent, "noevent", video) / video
    if (candidate / "manifest.json").exists():
        return candidate
    index = read(regular(parent / "inheritance/parent_index.json"))
    item = index["branches"]["noevent"]["videos"].get(video, {})
    candidate = Path(item.get("memory_dir", ""))
    allowed = (parent / "inheritance/parent").resolve()
    if not candidate.is_absolute() or not candidate.resolve().is_relative_to(allowed):
        raise ValueError("inherited checkpoint must be inside the immutable parent snapshot")
    regular(candidate / "manifest.json")
    return candidate


def failure_evidence(folder):
    memory = read(folder / "memory.json") if (folder / "memory.json").exists() else {}
    if memory.get("complete"):
        raise ValueError("this continuation only selects incomplete perception memories")
    if memory and memory.get("representation") != "window_records":
        raise ValueError("event graphs cannot enter a noevent continuation")
    complete = set(memory.get("completed_windows", []))
    errors = [r for ledger in (folder / "calls.jsonl", folder / "audio/calls.jsonl") for r in records(ledger)
              if r.get("status") == "error" and r.get("purpose", "").split(":")[-1] not in complete]
    if not errors:
        raise ValueError("missing evidence for the old unfinished-window failure")
    for row in errors:
        code = re.sub(r"[^a-z0-9]", "", str(row.get("service_error_code", "")).lower())
        if code in _REFUSALS or row.get("http_status") in (401, 402, 403):
            raise ValueError("explicit policy/authentication/credit refusal is not eligible for this protocol")
    return errors


def ensure_no_backend_attempts(parent, questions):
    qids = {q["question_id"] for q in questions}
    videos = {q["video_id"] for q in questions}
    for stage in ("plan", "answer", "score"):
        for vid in videos:
            task = parent / "tasks" / stage / ("noevent-" + vid) / "task.json"
            if task.exists():
                raise ValueError("selected incomplete video already has a backend task")
    for qid in qids:
        if any((parent / "noevent/plans" / name).exists() for name in
               (qid + ".json", "calls/" + qid + ".jsonl", "failures/" + qid + ".json")):
            raise ValueError("old planning attempt cannot receive a new budget")
    for root in (parent, parent / "inheritance/parent"):
        for path in (root / "scores/noevent").glob("*/run_*/scores.jsonl"):
            if any(row.get("question_id") in qids for row in records(path)):
                raise ValueError("selected question already has a first judgment attempt")
        for path in (root / "answers/noevent").glob("*/predictions.jsonl"):
            if any(row.get("question_id") in qids for row in records(path)):
                raise ValueError("selected question already has an answer attempt")


def checkpoint_receipt(parent, repo, old, video):
    source = resolve_checkpoint(parent, video)
    files = inventory(source)
    manifest = read(source / "manifest.json")
    cfg = manifest["configuration"]
    if cfg.get("representation") != "window_records" or cfg.get("code_hash") != old["source_hashes"]["noevent"]:
        raise ValueError("checkpoint representation/core hash differs")
    clients = {k: cfg[k] for k in ("model", "audio_observer")}
    checked = validate_parent_checkpoint(source, repo, "noevent", video, old["media"][video], old, clients)
    errors = failure_evidence(source)
    regular(source / "manifest.json")
    # Pending audio is only eligible for deferred strict fingerprint validation
    # by unchanged bridge_audio. Its model must already match the old observer.
    next_window = f"W{checked['completed_windows'] + 1:05d}"
    pending = "audio/" + next_window + ".json"
    pending_files = {}
    if pending in files:
        audio = read(source / pending)
        signature = audio.get("input_fingerprint")
        if (audio.get("model") == clients["audio_observer"] and isinstance(signature, str)
                and re.fullmatch(r"[a-f0-9]{64}", signature)):
            pending_files[pending] = files[pending]
    return {"source": str(source), "files": files, "committed_files": checked["files"],
            "pending_audio_strict_validation": pending_files, "completed_windows": checked["completed_windows"],
            "parent_manifest_sha256": files["manifest.json"], "parent_model": cfg["model"],
            "parent_audio_observer": cfg["audio_observer"], "failure_evidence": errors,
            "old_attempts": "preserved in ancestry; not reset", "new_task_budget": "one build task; no outer retries"}


def clone_checkpoint(run, video, receipt):
    source, target = Path(receipt["source"]), memory_root(run, "noevent", video) / video
    verify_inventory(source, receipt["files"])
    marker = run / "inheritance" / (video + ".json")
    if marker.exists():
        if read(marker) != receipt:
            raise ValueError("checkpoint receipt changed")
        # Once a build task is claimed its memory may legitimately grow. Never
        # clone over it, even when the task later fails or is ambiguous.
        if not (run / "tasks/build" / ("noevent-" + video) / "task.json").exists():
            for name, expected in {**receipt["committed_files"], **receipt["pending_audio_strict_validation"]}.items():
                if sha(regular(target / name)) != expected:
                    raise ValueError("unstarted copied checkpoint changed")
        return
    if target.exists():
        raise ValueError("refuse to overwrite an unknown checkpoint destination")
    staging = target.with_name(target.name + ".import-" + uuid.uuid4().hex)
    staging.mkdir(parents=True)
    try:
        for name, expected in receipt["files"].items():
            destination = staging / "ancestry/parent" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, destination)
            if sha(destination) != expected:
                raise ValueError("source changed while archiving its old attempts")
        copied = {**receipt["committed_files"], **receipt["pending_audio_strict_validation"]}
        for name, expected in copied.items():
            destination = staging / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, destination)
            if sha(destination) != expected:
                raise ValueError("checkpoint changed during copying")
        verify_inventory(source, receipt["files"])
        staging.replace(target)
        freeze(marker, receipt)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def verify(run, config):
    marker = read(regular(run / "prepared.json"))
    if sha(regular(run / "configuration.json")) != marker["configuration_sha256"]:
        raise ValueError("frozen continuation configuration changed")
    if config.get("condition") != "noevent" or config.get("execution_conditions") != ["noevent"]:
        raise ValueError("base/method execution is forbidden")
    if sha(regular(Path(__file__))) != config["coordinator_sha256"] or sha(regular(pilot.__file__)) != config["support_sha256"]:
        raise ValueError("continuation coordinator/support changed")
    if sha(regular(config["diagnostic_wrapper"])) != config["diagnostic_wrapper_sha256"]:
        raise ValueError("private diagnostic wrapper changed")
    if source_hash(Path(config["core_repo"])) != config["core_sha256"]:
        raise ValueError("frozen noevent core changed")
    parent = Path(config["parent_run"])
    for name, expected in config["parent_root_hashes"].items():
        if sha(regular(parent / name)) != expected:
            raise ValueError("frozen parent metadata changed")
    actual = {p.stem: sha(regular(p)) for p in (parent / "accepted/noevent").glob("*.json")}
    expected = {qid: value["sha256"] for qid, value in config["protected_scores"].items()}
    if actual != expected:
        raise ValueError("protected parent first-score inventory changed")
    original_sources = {item["envelope"]["source"]: item["official_source_sha256"]
                        for item in config["protected_scores"].values()}
    if any(sha(regular(path)) != expected for path, expected in original_sources.items()):
        raise ValueError("protected official score source changed")
    questions = read(regular(run / "questions.json"))
    if digest(questions) != config["questions_sha256"]:
        raise ValueError("frozen continuation question selection changed")
    if {q["video_id"] for q in questions} != set(config["checkpoints"]):
        raise ValueError("selected question/video ownership changed")
    for video in config["checkpoints"]:
        if read(regular(run / "questions" / (video + ".json"))) != [q for q in questions if q["video_id"] == video]:
            raise ValueError("per-video question text, rubric or selection changed")
    for media in config["media_identity"].values():
        if any(media_identity(value["path"]) != value for value in media.values()):
            raise ValueError("previously SHA-verified media identity changed")
    for video, receipt in config["checkpoints"].items():
        verify_inventory(receipt["source"], receipt["files"])
        if read(regular(run / "inheritance" / (video + ".json"))) != receipt:
            raise ValueError("immutable checkpoint lineage changed")


def prepare(args):
    run, parent, repo = (Path(x).absolute() for x in (args.run, args.parent_run, args.core_repo))
    if (run == parent or run.is_relative_to(parent) or parent.is_relative_to(run)
            or any(p.is_symlink() for root in (run, parent, repo) for p in (root, *root.parents))):
        raise ValueError("parent, continuation and core must be ordinary, distinct paths")
    selection = read(regular(args.selection))
    old = read(regular(parent / "configuration.json"))
    if str(repo) != old["repos"]["noevent"] or source_hash(repo) != old["source_hashes"]["noevent"]:
        raise ValueError("continuation must use the parent's frozen noevent core")
    if args.visual_max_tokens < 1 or args.workers < 1 or args.question_workers < 1:
        raise ValueError("positive output token limit and worker counts are required")
    if old["stage_tries"]["score"] != 1 or old.get("max_tokens") != 8192:
        raise ValueError("unexpected original official judging profile")
    if old["stage_tries"]["plan"] != old["stage_tries"]["answer"]:
        raise ValueError("inline noevent planning requires matching original plan/answer budgets")
    with stopped_parent(parent):
        all_questions = read(regular(parent / "questions.json"))
        protected = protected_scores(parent, all_questions, args.protected_count)
        questions = selected_questions(selection, all_questions, protected)
        ensure_no_backend_attempts(parent, questions)
        checkpoints = {vid: checkpoint_receipt(parent, repo, old, vid)
                       for vid in sorted({q["video_id"] for q in questions})}
        media = verified_media(old, checkpoints)
        profile = {key: old[key] for key in (
            "model", "base_url", "timeout", "max_tokens", "audio_model", "audio_base_url", "embedding_model",
            "embedding_base_url", "window_seconds", "padding", "fps", "max_frames", "max_pixels", "evidence_chars", "stage_tries")}
        profile["visual"] = {**pilot.perception_config(old), "max_tokens": args.visual_max_tokens}
        root_names = ["configuration.json", "questions.json", "process.json"]
        root_names += [name for name in ("execution_scope.json", "inheritance/parent_index.json") if (parent / name).exists()]
        config = {"schema_version": 1, "condition": "noevent", "execution_conditions": ["noevent"],
            "parent_run": str(parent), "parent_root_hashes": {name: sha(parent / name) for name in root_names},
            "core_repo": str(repo), "core_sha256": old["source_hashes"]["noevent"],
            "coordinator_sha256": sha(__file__), "support_sha256": sha(pilot.__file__),
            "diagnostic_wrapper": str(Path(__file__).with_name("noevent_diagnostics_worker.py").absolute()),
            "diagnostic_wrapper_sha256": sha(Path(__file__).with_name("noevent_diagnostics_worker.py")),
            "selection": selection, "selection_file_sha256": sha(args.selection), "questions_sha256": digest(questions),
            "total_questions": len(all_questions), "protected_scores": protected,
            "profile": profile, "media": {vid: old["media"][vid] for vid in checkpoints}, "media_identity": media,
            "checkpoints": checkpoints,
            "videos_dir": old["videos_dir"], "subtitles_dir": old["subtitles_dir"],
            "python": args.python, "credential_file": str(Path(args.credential_file).absolute()),
            "workers": args.workers, "question_workers": args.question_workers,
            "budget_policy": "new user-authorized build task per selected video; old failures remain immutable; no outer retry; judge once",
            "comparison_scope": "Separate authorized continuation; inherited windows and the explicit visual output-token budget (possibly unchanged) are disclosed. The original attempt budgets and report are preserved."}
        freeze(run / "configuration.json", config)
        freeze(run / "questions.json", questions)
        for vid in checkpoints:
            freeze(run / "questions" / (vid + ".json"), [q for q in questions if q["video_id"] == vid])
            clone_checkpoint(run, vid, checkpoints[vid])
        freeze(run / "prepared.json", {"configuration_sha256": sha(run / "configuration.json"), "no_api_calls": True})
    verify(run, config)
    return run, config


def stage_command(run, config, stage, video, question_file=None):
    if config["condition"] != "noevent" or stage not in ("build", "answer", "score"):
        raise ValueError("only noevent build/answer/score are allowed")
    profile, memory = config["profile"], memory_root(run, "noevent", video)
    questions = question_file or run / "questions" / (video + ".json")
    cli = [config["python"], "-B", "-u", "-m"]
    model = profile["visual"] if stage == "build" else profile
    common = ["--model", model["model"], "--base-url", model["base_url"], "--max-tokens", str(model["max_tokens"]),
              "--timeout", str(model["timeout"]), "--tries", str(1 if stage == "score" else profile["stage_tries"][stage])]
    if stage == "build":
        return [config["python"], "-B", "-u", config["diagnostic_wrapper"], "--core-repo", config["core_repo"],
            "--core-sha256", config["core_sha256"], "--diagnostics-dir", str(run / "private_diagnostics/build" / video),
            "--", "build", "--data-path", str(questions),
            "--videos-dir", config["videos_dir"], "--subtitles-dir", config["subtitles_dir"],
            "--output-dir", str(memory), "--credential-file", config["credential_file"], *common,
            "--workers", "1", "--with-audio", "--audio-model", profile["audio_model"],
            "--audio-base-url", profile["audio_base_url"], "--window-seconds", str(profile["window_seconds"]),
            "--padding", str(profile["padding"]), "--fps", str(profile["fps"]), "--max-frames", str(profile["max_frames"]),
            "--max-pixels", str(profile["max_pixels"])]
    if stage == "answer":
        return cli + ["methods.longemo.noevent_runner", "answer", "--data-path", str(questions),
            "--memory-dir", str(memory), "--output-dir", str(run / "answers/noevent" / video),
            "--plans-dir", str(run / "noevent/plans"), "--credential-file", config["credential_file"], *common,
            "--workers", str(config["question_workers"]), "--embedding-backend", "gemini",
            "--embedding-model", profile["embedding_model"], "--embedding-base-url", profile["embedding_base_url"],
            "--embedding-cache-dir", str(run / "noevent/embeddings"), "--evidence-chars", str(profile["evidence_chars"])]
    return cli + ["evaluation.eval", "--data-path", str(questions), "--predictions", str(run / "score_inputs" / (video + ".json")),
                  "--granularity", "episode", "--output-dir", str(run / "scores/noevent" / video),
                  *common, "--workers", str(config["question_workers"])]


def run_stage(run, config, stage, video, env, question_file=None):
    verify(run, config)
    state = run_command(run / "tasks" / stage / ("noevent-" + video),
                        stage_command(run, config, stage, video, question_file), config["core_repo"], env)
    if state["status"] in ("running", "needs_audit"):
        raise ValueError("a prior task may have an in-flight request; do not repeat it")
    return state


def accept_scores(run, config, video):
    """Import written judgments after verifying the scorer is no longer alive."""
    task_path = run / "tasks/score" / ("noevent-" + video) / "task.json"
    sources = sorted((run / "scores/noevent" / video).glob("run_*/scores.jsonl"))
    if not sources:
        return
    task = read(regular(task_path))
    if not task.get("child"):
        raise ValueError("cannot audit the identity of the previous scoring child")
    assert_identity_stopped(task["child"], "continuation official scorer")
    selected = read(regular(run / "score_questions" / (video + ".json")))
    qmap = {q["question_id"]: q for q in selected}
    frozen_qmap = {q["question_id"]: q for q in read(regular(run / "questions" / (video + ".json")))}
    if len(qmap) != len(selected) or any(frozen_qmap.get(qid) != q for qid, q in qmap.items()):
        raise ValueError("first-judgment questions or rubric differ from the frozen video selection")
    predictions = {q["question_id"]: q for q in read(regular(run / "score_inputs" / (video + ".json")))}
    if set(qmap) != set(predictions) or set(qmap) & set(config["protected_scores"]):
        raise ValueError("first-judgment selection overlaps old scores or differs from inputs")
    seen = set()
    for source in sources:
        for row in records(regular(source)):
            qid = row.get("question_id")
            if qid not in qmap or qid in seen:
                raise ValueError("duplicate/out-of-selection first official judgment")
            seen.add(qid)
            if row.get("status") != "ok":
                continue
            maximum = max(int(k) for k in qmap[qid]["rubric"]["scores"])
            if (row.get("prediction") != predictions[qid]["prediction"] or row.get("max_score") != maximum
                    or type(row.get("score")) not in (int, float) or not 0 <= row["score"] <= maximum
                    or not math.isclose(row.get("normalized_score", -1), row["score"] / maximum)):
                raise ValueError("official score differs from frozen prediction/rubric")
            freeze(run / "accepted/noevent" / (qid + ".json"), {"score": row, "source": str(source),
                   "prediction_sha256": digest(row["prediction"]), "source_sha256": sha(source)})


def window_provenance(run, config, video):
    folder = memory_root(run, "noevent", video) / video
    memory = read(regular(folder / "memory.json"))
    if not memory.get("complete") or memory.get("representation") != "window_records":
        raise ValueError("only complete window-only memory can be answered")
    manifest = read(regular(folder / "manifest.json"))
    clients = {key: manifest["configuration"][key] for key in ("model", "audio_observer")}
    checked = validate_parent_checkpoint(folder, config["core_repo"], "noevent", video,
                                         config["media"][video], config["profile"], clients)
    if not checked["complete"]:
        raise ValueError("completed memory failed full window replay validation")
    receipt = config["checkpoints"][video]
    inherited = receipt["completed_windows"]
    result = {}
    for i, window in enumerate(memory["completed_windows"]):
        path = regular(folder / "windows" / (window + ".json"))
        key = "windows/" + window + ".json"
        actual = sha(path)
        if i < inherited and actual != receipt["committed_files"][key]:
            raise ValueError("inherited successful window changed")
        result[window] = {"window_sha256": actual, "source": "parent" if i < inherited else "continuation",
                          "parent_source": receipt["source"] if i < inherited else None,
                          "model": receipt["parent_model"] if i < inherited else manifest["configuration"]["model"]}
    value = {"memory_sha256": sha(folder / "memory.json"), "windows": result}
    freeze(run / "noevent/frozen_memories" / (video + ".json"), value)
    return value


def pipeline(run, config, video):
    state_path = run / "pipelines/noevent" / (video + ".json")
    try:
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env.pop("LONGEMO_ALLOW_AUDIO_CACHE_REUSE", None)
        env["MODEL_API_KEY"] = read(regular(config["credential_file"]))["MODEL_API_KEY"]
        # Use only the frozen core for child imports, even if the coordinator
        # was started with a staging PYTHONPATH ahead of that checkout.
        env["PYTHONPATH"] = config["core_repo"]
        runtime_bin = Path(config["parent_run"]).parents[1] / "bin"
        env["PATH"] = str(runtime_bin) + os.pathsep + env.get("PATH", "")
        write(state_path, {"status": "running", "video_id": video})
        built = run_stage(run, config, "build", video, env)
        if built["status"] != "ok":
            write(state_path, {"status": "build_failed"})
            return
        frozen = window_provenance(run, config, video)
        answer = run_stage(run, config, "answer", video, env)
        rows = records(run / "answers/noevent" / video / "predictions.jsonl")
        questions = read(run / "questions" / (video + ".json"))
        qmap, successes, seen = {q["question_id"]: q for q in questions}, [], set()
        for row in rows:
            qid = row.get("question_id")
            if qid not in qmap or qid in seen or qid in config["protected_scores"]:
                raise ValueError("answer selection acquired a duplicate/protected question")
            seen.add(qid)
            if row.get("status") == "ok":
                successes.append(row)
        if successes:
            pending = [qmap[row["question_id"]] for row in successes]
            question_file = run / "score_questions" / (video + ".json")
            freeze(question_file, pending)
            freeze(run / "score_inputs" / (video + ".json"), successes)
            # If the judge already wrote its score before a coordinator crash,
            # import it first; run_command will never start a claimed task twice.
            accept_scores(run, config, video)
            run_stage(run, config, "score", video, env, question_file)
            accept_scores(run, config, video)
        if sha(memory_root(run, "noevent", video) / video / "memory.json") != frozen["memory_sha256"]:
            raise ValueError("frozen noevent memory changed during answering")
        count = sum((run / "accepted/noevent" / (qid + ".json")).exists() for qid in qmap)
        write(state_path, {"status": "complete" if count == len(questions) else "partial", "scored": count,
                          "questions": len(questions), "answer_task": answer["status"]})
    except Exception as exc:
        # Avoid publishing raw transport exceptions or credential contents.
        write(state_path, {"status": "needs_audit", "error_type": type(exc).__name__})
        raise


def report(run, output=None):
    run = Path(run).absolute()
    config = read(regular(run / "configuration.json"))
    verify(run, config)
    protected = config["protected_scores"]
    old_rows = {qid: item["envelope"]["score"] for qid, item in protected.items()}
    qmap = {q["question_id"]: q for q in read(regular(run / "questions.json"))}
    new_rows, sources = {}, {}
    for path in sorted((run / "accepted/noevent").glob("*.json")):
        value, qid = read(regular(path)), path.stem
        if qid in old_rows or qid not in config["selection"]["question_ids"] or value["score"].get("question_id") != qid:
            raise ValueError("new first scores overlap old scores or escape the selection")
        source = regular(value["source"])
        if sha(source) != value["source_sha256"] or value["score"] not in records(source):
            raise ValueError("accepted first score differs from its official artifact")
        row, question = value["score"], qmap[qid]
        maximum = max(int(key) for key in question["rubric"]["scores"])
        inputs = read(regular(run / "score_inputs" / (question["video_id"] + ".json")))
        predictions = {prediction["question_id"]: prediction for prediction in inputs}
        if (len(predictions) != len(inputs) or row.get("status") != "ok" or row.get("video_id") != question["video_id"]
                or row.get("max_score") != maximum or type(row.get("score")) not in (int, float)
                or not 0 <= row["score"] <= maximum or type(row.get("normalized_score")) not in (int, float)
                or not math.isclose(row["normalized_score"], row["score"] / maximum)
                or row.get("prediction") != predictions.get(qid, {}).get("prediction")
                or value.get("prediction_sha256") != digest(row.get("prediction"))):
            raise ValueError("reported first score differs from its frozen question, prediction or rubric")
        new_rows[qid] = row
        sources[qid] = {"run": str(run), "accepted_sha256": sha(path), "score_source": str(source),
                        "score_source_sha256": value["source_sha256"]}
    combined = {**old_rows, **new_rows}
    mean = lambda rows: 100 * sum(r["normalized_score"] for r in rows.values()) / len(rows) if rows else None
    states = {vid: read(run / "pipelines/noevent" / (vid + ".json")) if (run / "pipelines/noevent" / (vid + ".json")).exists()
              else {"status": "pending"} for vid in config["checkpoints"]}
    result = {"schema_version": 1, "as_of": datetime.now(timezone.utc).isoformat(), "condition": "noevent",
              "parent_run": config["parent_run"], "run": str(run), "configuration_sha256": sha(run / "configuration.json"),
              "original": {"scored": len(old_rows), "mean": mean(old_rows)},
              "continuation": {"scored": len(new_rows), "selected": len(config["selection"]["question_ids"]), "mean": mean(new_rows)},
              "combined": {"scored": len(combined), "total": config["total_questions"],
                           "coverage": len(combined) / config["total_questions"], "mean": mean(combined)},
              "comparison_scope": config["comparison_scope"], "profile": config["profile"], "pipelines": states,
              "questions": [{"question_id": qid, "score": row, "origin": "parent" if qid in old_rows else "continuation",
                             "source": protected[qid] if qid in old_rows else sources[qid]} for qid, row in sorted(combined.items())]}
    destination = Path(output).absolute() if output else run / "report"
    if destination.is_relative_to(Path(config["parent_run"])):
        raise ValueError("the original same-configuration report must not be overwritten")
    with _REPORT_LOCK:
        write(destination / "report.json", result)
        combined_text = f"{mean(combined):.2f}" if combined else "—"
        destination.mkdir(parents=True, exist_ok=True)
        unchanged = config["profile"]["visual"]["max_tokens"] == config["profile"]["max_tokens"]
        budget_text = "保持原视觉输出上限" if unchanged else "本轮视觉输出上限为"
        (destination / "report.md").write_text(
            "# noevent 独立续跑结果\n\n"
            f"原始首次评分保留 **{len(old_rows)}** 条；本轮新增 **{len(new_rows)}** 条。"
            f"合并 **{len(combined)}/{config['total_questions']}** 题，已评分均分 **{combined_text}**。\n\n"
            "本轮继承已完成时间窗口，仅对固定未评分题目继续构建；模型、媒体采样与 noevent schema 保持一致，"
            f"{budget_text} {config['profile']['visual']['max_tokens']} tokens。"
            "本轮是另行授权的断点任务，保留原尝试和首分来源，不覆盖原实验报告，缺失题不计入均分。\n")
    return result


def execute(run, config):
    with run_lock(run):
        verify(run, config)
        failures = []
        with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
            pending = {pool.submit(pipeline, run, config, vid): vid for vid in config["checkpoints"]}
            for future in as_completed(pending):
                try:
                    future.result()
                except Exception as exc:
                    failures.append({"video_id": pending[future], "error_type": type(exc).__name__})
        verify(run, config)
        write(run / "finished.json", {"finished_at": datetime.now(timezone.utc).isoformat(), "failures_needing_audit": failures,
                                      "queue_finished": not failures, "not_all_questions_scored": True})
        return report(run)


def apply_probe_seed(*args, **kwargs):
    """Reserved hook: do not import an unverified probe as a successful window."""
    raise ValueError("probe seeding requires a separate exact-input receipt implementation; no API request was made")


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("stage", choices=("prepare", "run", "report"))
    p.add_argument("--run", required=True)
    for name in ("parent-run", "core-repo", "selection", "credential-file"):
        p.add_argument("--" + name)
    p.add_argument("--visual-max-tokens", type=int)
    p.add_argument("--protected-count", type=int, default=195)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--question-workers", type=int, default=2)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--output")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.stage == "report":
        result = report(args.run, args.output)
    else:
        if any(getattr(args, name) is None for name in ("parent_run", "core_repo", "selection", "credential_file", "visual_max_tokens")):
            raise SystemExit("prepare/run require --parent-run --core-repo --selection --credential-file --visual-max-tokens")
        run, config = prepare(args)
        result = execute(run, config) if args.stage == "run" else {"prepared": str(run), "selected": len(config["selection"]["question_ids"])}
    print(json.dumps(result.get("combined", result), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
