"""Offline, immutable import of a stopped matched pilot into a larger cohort.

This module never calls a model and never repairs or restarts a parent task.
The snapshot owns copies, not shared writable caches.  A parent task which was
claimed but left no per-question result remains blocked for its original
questions; expanding a video's question set does not reset an attempt budget.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import uuid

from experiments.zyf.matched_pilot import (
    CONDITIONS, digest, freeze, memory_root, read, records, sha, source_hash,
    stopped_parent, validate_parent_checkpoint,
)


_ROOT_FILES = ("configuration.json", "questions.json", "process.json", "gate.json")
_ROOT_DIRS = ("questions", "tasks", "event", "noevent", "answers", "scores", "accepted", "inheritance")
_COMPATIBLE = ("source_hashes", "model", "base_url", "audio_model", "audio_base_url",
               "embedding_model", "embedding_base_url", "evidence_chars", "window_seconds", "padding",
               "fps", "max_frames", "max_pixels", "stage_tries", "max_tokens", "timeout",
               "progressive_routing", "progressive_rounds", "progressive_anchor_k")


def _inventory(parent):
    result = {}
    candidates = [parent / name for name in _ROOT_FILES if (parent / name).exists()]
    for name in _ROOT_DIRS:
        directory = parent / name
        if directory.is_symlink():
            raise ValueError(f"parent artifact directory must not be a symlink: {directory}")
        if directory.exists():
            candidates.extend(directory.rglob("*"))
    for path in sorted(candidates):
        if path.is_symlink():
            raise ValueError(f"parent artifact must not be a symlink: {path}")
        if path.is_file():
            result[path.relative_to(parent).as_posix()] = {"sha256": sha(path), "size": path.stat().st_size,
                                                         "mtime_ns": path.stat().st_mtime_ns}
    return result


def _unique(rows, label):
    result = {}
    for row in rows:
        qid = row["question_id"]
        if qid in result:
            raise ValueError(f"duplicate question ID in {label}: {qid}")
        result[qid] = row
    return result


def _validate_config(parent, old, full_questions, config):
    parent_questions = records(parent / "questions.json")
    old_questions, full = _unique(parent_questions, "parent questions"), _unique(full_questions, "full questions")
    if old.get("questions_sha256") != digest(parent_questions):
        raise ValueError("parent question selection hash changed")
    if any(qid not in full or full[qid] != q for qid, q in old_questions.items()):
        raise ValueError("the full cohort must contain the exact parent questions")
    for branch in ("event", "noevent"):
        if source_hash(Path(old["repos"][branch])) != old["source_hashes"][branch]:
            raise ValueError(f"parent {branch} source no longer matches its frozen hash")
    for video in {q["video_id"] for q in parent_questions}:
        expected = [q for q in parent_questions if q["video_id"] == video]
        if records(parent / "questions" / (video + ".json")) != expected:
            raise ValueError("parent per-video question subset changed")
    if config is not None:
        for key in _COMPATIBLE:
            if old.get(key) != config.get(key):
                raise ValueError(f"full-run settings differ from the parent: {key}")
        from experiments.zyf.matched_pilot import perception_config
        if perception_config(old) != perception_config(config):
            raise ValueError("full-run perception settings differ from the parent")
        for video, media in old["media"].items():
            updated = config.get("media", {}).get(video, {})
            if any(updated.get(key) != media.get(key) for key in ("video_sha256", "subtitles_sha256")):
                raise ValueError(f"full-run media differs from the parent: {video}")
    return parent_questions, old_questions, full


def _task(parent, stage, key):
    path = parent / "tasks" / stage / key / "task.json"
    return read(path) if path.exists() else None


def _claimed(task):
    return task is not None and task.get("status") not in ("pending", "unstarted")


def _embedding_guard(parent, branch, parent_videos, artifact):
    """Keep unfinished document-index attempts from being reset by new QIDs.

    The old ledger has request hashes but normally no video ID. Its conservative
    fallback covers only parent videos whose answer/index process actually
    started, never the new cohort's untouched videos or their new graphs.
    """
    conditions = ("noevent",) if branch == "noevent" else ("base", "method")
    attempted_videos = sorted(video for video in parent_videos if any(
        _claimed(_task(parent, "answer", condition + "-" + video)) or
        (parent / "answers" / condition / video).exists() for condition in conditions))
    ledger = parent / branch / "embeddings/api/calls.jsonl"
    final = {}
    if ledger.exists():
        for number, row in enumerate(records(ledger), 1):
            if row.get("purpose") == "embedding_documents":
                request_hash = row.get("request_hash")
                key = request_hash if isinstance(request_hash, str) and request_hash else "unknown-row-" + str(number)
                final[key] = row
    unresolved, blocked = [], set()
    fallback = False
    for request_hash, row in final.items():
        if row.get("status") == "ok":
            continue
        explicit_video = row.get("video_id")
        affected = [explicit_video] if explicit_video in attempted_videos else attempted_videos
        fallback = fallback or explicit_video not in attempted_videos
        blocked.update(affected)
        unresolved.append({"request_hash": request_hash,
            "reason": "parent document-embedding request ended without a successful ledger result; no fresh attempt budget",
            "last_attempt": {key: row[key] for key in ("status", "error_type", "http_status", "attempt", "time_unix", "video_id") if key in row},
            "ledger": artifact(ledger), "blocked_videos": affected})
    return {"state": "needs_audit" if unresolved else "clear",
            "scope": "parent_attempted_videos" if fallback else "videos" if unresolved else "none",
            "applies_to": "new_answers_only", "parent_attempted_videos": attempted_videos,
            "blocked_videos": sorted(blocked), "unresolved_requests": unresolved}


def _validate_score(row, question, prediction):
    maximum = max(int(key) for key in question["rubric"]["scores"]) if question.get("rubric") else 1
    score, normalized = row.get("score"), row.get("normalized_score")
    if (row.get("status") != "ok" or row.get("prediction") != prediction or
            row.get("video_id") != question["video_id"] or row.get("type") != question["type"] or
            row.get("max_score") != maximum or type(score) not in (float, int) or
            type(normalized) not in (float, int) or not math.isfinite(score) or not math.isfinite(normalized) or
            not 0 <= score <= maximum or not math.isclose(normalized, score / maximum)):
        raise ValueError(f"parent official judgment differs from its question/prediction: {question['question_id']}")


def _validate_answer_manifest(parent, condition, branch, video, questions, old, memory_path):
    folder = parent / "answers" / condition / video
    value = read(folder / "manifest.json")
    config = value["configuration"]
    expected_questions = [{k: q[k] for k in ("question_id", "video_id", "question", "type")} for q in questions]
    if (value.get("fingerprint") != digest(config) or
            config.get("question_inputs", config.get("questions")) != expected_questions or
            config.get("code_hash") != old["source_hashes"][branch] or
            config.get("memories") != {video: sha(memory_path)} or
            config.get("evidence_chars") != old["evidence_chars"]):
        raise ValueError("parent answer manifest differs from frozen question/memory/source settings")
    if branch == "noevent":
        if config.get("representation") != "window_records":
            raise ValueError("parent noevent answer did not use window records")
    elif (config.get("retrieval") != ("graph" if condition == "base" else "progressive") or
          config.get("progressive_routing") != "none" or config.get("hybrid_graph_tasks")):
        raise ValueError("parent event retrieval condition is not the frozen pure ablation")
    if records(folder / "evaluation_questions.json") != questions:
        raise ValueError("parent answer evaluation question file changed")


def _index(parent, destination, full_questions, config, inventory):
    old = read(parent / "configuration.json")
    parent_questions, parent_by_id, full = _validate_config(parent, old, full_questions, config)
    parent_videos = {q["video_id"] for q in parent_questions}
    videos = sorted({q["video_id"] for q in full_questions})

    def artifact(path):
        relative = Path(path).relative_to(parent).as_posix()
        if relative not in inventory:
            raise ValueError(f"artifact absent from frozen parent inventory: {relative}")
        return {"path": str(destination / relative), "source": str(parent / relative),
                "relative_path": relative, **inventory[relative]}

    def files_below(path):
        prefix = Path(path).relative_to(parent).as_posix().rstrip("/") + "/"
        return [artifact(parent / relative) for relative in inventory if relative.startswith(prefix)]

    result = {"schema_version": 1, "parent_run": str(parent), "snapshot_dir": str(destination),
              "parent_configuration_sha256": sha(parent / "configuration.json"),
              "parent_questions_sha256": digest(parent_questions), "full_questions_sha256": digest(full_questions),
              "parent_question_ids": list(parent_by_id), "protocol": "stopped-parent-first-valid-scores-no-retry",
              "files": inventory, "branches": {}, "conditions": {}}
    for branch in ("event", "noevent"):
        plans_folder = parent / branch / "plans"
        branch_entry = {"videos": {}, "embeddings_dir": str(destination / branch / "embeddings"),
                        "embedding_files": files_below(parent / branch / "embeddings")}
        branch_entry["embedding_guard"] = _embedding_guard(parent, branch, parent_videos, artifact)
        result["branches"][branch] = branch_entry
        for video in videos:
            folder = memory_root(parent, branch, video) / video
            build_task = _task(parent, "build", branch + "-" + video)
            memory_path, frozen_path = folder / "memory.json", parent / branch / "frozen_memories" / (video + ".json")
            entry = {"state": "unstarted", "memory_dir": str(destination / folder.relative_to(parent)),
                     "memory": artifact(memory_path) if memory_path.exists() else None,
                     "frozen_memory": artifact(frozen_path) if frozen_path.exists() else None,
                     "files": files_below(folder), "plans": {}, "build_task": build_task,
                     "attempted_build": _claimed(build_task)}
            branch_entry["videos"][video] = entry
            entry["embedding_guard"] = {"state": "needs_audit" if video in branch_entry["embedding_guard"]["blocked_videos"] else "clear",
                                        "applies_to": "new_answers_only"}
            if entry["embedding_guard"]["state"] == "needs_audit":
                entry["embedding_guard"]["reason"] = "unresolved parent document-embedding attempt; existing predictions may still receive their first score"
            if video in parent_videos:
                inherited_block = parent / "inheritance" / "blocked" / (branch + "-" + video + ".json")
                if build_task and build_task.get("status") == "ok" and not build_task.get("validation_error"):
                    if not memory_path.exists() or not frozen_path.exists():
                        raise ValueError(f"successful parent build lacks frozen memory: {branch}/{video}")
                    manifest = read(folder / "manifest.json")
                    model_configs = {key: manifest["configuration"][key] for key in ("model", "audio_observer")}
                    checkpoint = validate_parent_checkpoint(folder, old["repos"][branch], branch, video,
                        old["media"][video], old, model_configs)
                    if not checkpoint["complete"] or read(frozen_path).get("memory_sha256") != sha(memory_path):
                        raise ValueError(f"parent complete-memory receipt differs: {branch}/{video}")
                    if manifest["configuration"].get("code_hash") != old["source_hashes"][branch]:
                        raise ValueError(f"parent builder source differs: {branch}/{video}")
                    entry.update(state="reusable", completed_windows=checkpoint["completed_windows"])
                elif _claimed(build_task) or entry["files"] or inherited_block.exists():
                    entry.update(state="blocked", attempted_build=True,
                                 reason="parent build failed, incomplete, or ambiguous; no new attempt budget")
                    if inherited_block.exists():
                        entry["inherited_block"] = {"row": read(inherited_block), "artifact": artifact(inherited_block)}
                plan_task = _task(parent, "plan", branch + "-" + video)
            else:
                plan_task = None
            frozen_plans = plans_folder / "frozen" / (video + ".json")
            old_plan_hashes = read(frozen_plans) if frozen_plans.exists() else {}
            expected_old_ids = {qid for qid, q in parent_by_id.items() if q["video_id"] == video}
            if frozen_plans.exists() and set(old_plan_hashes) != expected_old_ids:
                raise ValueError("parent frozen plan list changed")
            for q in (q for q in full_questions if q["video_id"] == video):
                qid = q["question_id"]
                plan = {"state": "unstarted", "attempted": False}
                entry["plans"][qid] = plan
                if qid not in parent_by_id:
                    continue
                plan_path = plans_folder / (qid + ".json")
                calls = plans_folder / "calls" / (qid + ".jsonl")
                failure = plans_folder / "failures" / (qid + ".json")
                plan.update(attempted=_claimed(plan_task) or plan_path.exists() or calls.exists() or failure.exists())
                if plan_path.exists():
                    if entry["state"] != "reusable":
                        raise ValueError("parent plan exists without validated complete memory")
                    value = read(plan_path)
                    from methods.longemo.retrieval import validate_plan
                    validate_plan(value["plan"])
                    if not isinstance(value.get("input_fingerprint"), str) or len(value["input_fingerprint"]) != 64:
                        raise ValueError("parent plan lacks an input fingerprint")
                    if qid in old_plan_hashes and sha(plan_path) != old_plan_hashes[qid]:
                        raise ValueError("parent frozen plan changed")
                    plan.update(state="success", artifact=artifact(plan_path), input_fingerprint=value["input_fingerprint"])
                elif plan["attempted"]:
                    plan.update(state="blocked", reason="parent plan attempted without a valid committed plan")
                if calls.exists():
                    plan["calls"] = artifact(calls)
                if failure.exists():
                    plan["failure"] = artifact(failure)

    for condition in CONDITIONS:
        branch = "noevent" if condition == "noevent" else "event"
        by_id = {}
        result["conditions"][condition] = {"questions": by_id}
        for video in videos:
            answer_task, score_task = (_task(parent, stage, condition + "-" + video) for stage in ("answer", "score"))
            prediction_path = parent / "answers" / condition / video / "predictions.jsonl"
            predictions = _unique(records(prediction_path), str(prediction_path))
            expected_ids = {qid for qid, q in parent_by_id.items() if q["video_id"] == video}
            if not set(predictions) <= expected_ids:
                raise ValueError("parent prediction is outside its original video subset")
            if any(row.get("status") == "ok" for row in predictions.values()):
                _validate_answer_manifest(parent, condition, branch, video,
                    [q for q in parent_questions if q["video_id"] == video], old,
                    memory_root(parent, branch, video) / video / "memory.json")
            score_rows = {}
            paths = sorted((parent / "scores" / condition / video).glob("run_*/scores.jsonl"),
                           key=lambda p: (p.stat().st_mtime_ns, str(p)))
            for path in paths:
                rows = _unique(records(path), str(path))
                if not set(rows) <= expected_ids:
                    raise ValueError("parent score is outside its original video subset")
                for qid, row in rows.items():
                    score_rows.setdefault(qid, []).append((row, path))
            for q in (q for q in full_questions if q["video_id"] == video):
                qid = q["question_id"]
                item = {"prediction": None, "accepted": None, "answer_state": "unstarted", "score_state": "unstarted",
                        "attempted_answer": False, "attempted_score": False, "blocked_reasons": []}
                by_id[qid] = item
                if qid not in parent_by_id:
                    continue
                prediction = predictions.get(qid)
                calls = parent / "answers" / condition / video / "calls" / (qid + ".jsonl")
                trace = parent / "answers" / condition / video / "traces" / (qid + ".json")
                item["attempted_answer"] = _claimed(answer_task) or prediction is not None or calls.exists() or trace.exists()
                if prediction is not None and prediction.get("status") == "ok":
                    if not isinstance(prediction.get("prediction"), str) or not prediction["prediction"].strip():
                        raise ValueError("parent successful answer is empty")
                    if any(prediction.get(key) != q[key] for key in ("question_id", "video_id", "granularity", "type", "question")):
                        raise ValueError("parent successful answer belongs to a changed question")
                    mem = result["branches"][branch]["videos"][video]
                    if mem["state"] != "reusable":
                        raise ValueError("parent successful answer lacks frozen complete memory")
                    actual_memory = read(Path(mem["memory"]["source"]))
                    if (prediction.get("memory_sha256") != digest(actual_memory) or
                            prediction.get("memory_fingerprint") != actual_memory["build_fingerprint"]):
                        raise ValueError("parent answer memory fingerprint changed")
                    item.update(answer_state="success", prediction={"row": prediction, "artifact": artifact(prediction_path)})
                elif item["attempted_answer"]:
                    item["answer_state"] = "blocked"
                    item["blocked_reasons"].append("parent answer attempted without a valid committed prediction")
                if calls.exists():
                    item["answer_calls"] = artifact(calls)
                if trace.exists():
                    item["answer_trace"] = artifact(trace)
                candidates = score_rows.get(qid, [])
                accepted_path = parent / "accepted" / condition / (qid + ".json")
                accepted = read(accepted_path) if accepted_path.exists() else None
                success = [(row, path) for row, path in candidates if row.get("status") == "ok"]
                if accepted is not None or success:
                    if item["answer_state"] != "success":
                        raise ValueError("parent official score has no valid frozen prediction")
                    answer = prediction["prediction"]
                    for row, _ in success:
                        _validate_score(row, q, answer)
                    if accepted is not None:
                        _validate_score(accepted["score"], q, answer)
                        if accepted.get("prediction_sha256") != digest(answer):
                            raise ValueError("parent accepted prediction hash changed")
                        original = next((row for row, path in success if str(path) == accepted.get("source")), None)
                        if original != accepted["score"]:
                            raise ValueError("parent accepted score differs from its original official row")
                        accepted_ref = artifact(accepted_path)
                    else:
                        first_row, first_path = success[0]
                        accepted = {"score": first_row, "source": str(first_path), "prediction_sha256": digest(answer)}
                        accepted_ref = None
                    item.update(score_state="success", attempted_score=True,
                                accepted={"envelope": accepted, "artifact": accepted_ref,
                                          "score_artifact": artifact(Path(accepted["source"]))})
                else:
                    # A committed missing_prediction row proves the judge was
                    # not called for that row. Any other claimed-but-missing
                    # result remains ambiguous even after the process stopped.
                    missing_only = bool(candidates) and all(row.get("status") == "missing_prediction" for row, _ in candidates)
                    item["attempted_score"] = bool(candidates) and not missing_only or _claimed(score_task) and not missing_only
                    if item["attempted_score"]:
                        item["score_state"] = "blocked"
                        item["blocked_reasons"].append("parent judge attempted or outcome unknown; preserve its first-attempt budget")
                item["score_outputs"] = [{"row": row, "artifact": artifact(path)} for row, path in candidates]
    return result


def snapshot_parent(parent_run, new_run, full_questions, config=None):
    """Lock and audit a stopped parent, then freeze a complete offline snapshot.

    The caller must not already hold the parent's flock. A live parent raises
    ValueError for the caller to wait on; a failed/ambiguous *stopped* task is
    recorded as blocked, never given a fresh attempt. Reentry validates every
    original file and refuses new parent outputs after the frozen cutoff.
    """
    parent, run = Path(parent_run).resolve(), Path(new_run).resolve()
    if parent == run or parent in run.parents or run in parent.parents:
        raise ValueError("parent and full run must be distinct, non-nested directories")
    receipt = run / "inheritance" / "parent_index.json"
    destination = run / "inheritance" / "parent"
    with stopped_parent(parent):
        inventory = _inventory(parent)
        if receipt.exists():
            prior = read(receipt)
            if (prior["parent_run"] != str(parent) or prior["files"] != inventory or
                    prior["full_questions_sha256"] != digest(full_questions)):
                raise ValueError("parent artifacts or full questions changed after inheritance")
            _validate_config(parent, read(parent / "configuration.json"), full_questions, config)
            for relative, expected in inventory.items():
                path = destination / relative
                if path.is_symlink() or not path.is_file() or sha(path) != expected["sha256"]:
                    raise ValueError("immutable parent snapshot changed")
            return prior
        index = _index(parent, destination, full_questions, config, inventory)
        if destination.exists():
            raise ValueError("refuse to overwrite an unknown parent snapshot")
        staging = destination.with_name("parent.import-" + uuid.uuid4().hex)
        staging.mkdir(parents=True)
        try:
            for relative, expected in inventory.items():
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(parent / relative, target)
                if sha(target) != expected["sha256"]:
                    raise ValueError("parent file changed while copying")
            if inventory != _inventory(parent):
                raise ValueError("parent files changed during inheritance")
            staging.replace(destination)
            freeze(receipt, index)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return index
