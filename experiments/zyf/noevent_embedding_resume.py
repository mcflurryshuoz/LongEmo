"""Resume the two unanswered V26 questions after a verified embedding probe.

Only new files are written. The complete parent memory, plans, and 195 first
scores remain immutable. Recorded answer/judge tasks are never repeated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time


def replacement(command, name, value):
    result = list(command)
    if result.count(name) != 1:
        raise ValueError("expected one command argument: " + name)
    result[result.index(name) + 1] = str(value)
    return result


def isolated(run, parent):
    if run == parent or run.is_relative_to(parent) or parent.is_relative_to(run):
        raise ValueError("new run must be isolated from the original run")


def verify_helper_sources(core, config):
    """The core hash excludes scheduler helpers; validate their frozen hashes too."""
    support = config["suite_support_hashes"]
    if set(support) != {"matched_pilot.py", "full_suite_inheritance.py"}:
        raise ValueError("unexpected frozen coordinator support files")
    expected = {"full_suite.py": config["suite_sha256"], **support}
    for name, value in expected.items():
        path = core / "experiments/zyf" / name
        if not path.is_file() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != value:
            raise ValueError("original coordinator helper changed: " + name)
    return expected


def verify_frozen_inputs(run, video, questions, imports, cache_values):
    """Fail before any worker can regenerate a missing batch or answer altered text."""
    def load(path):
        if not path.is_file() or path.is_symlink():
            raise ValueError("continuation input missing or not a regular file: " + str(path))
        return json.loads(path.read_text())

    for path in (run / "questions.json", run / "questions" / (video + ".json")):
        if load(path) != questions:
            raise ValueError("continuation question selection changed")
    if load(run / "embedding_imports.json") != imports:
        raise ValueError("continuation embedding import receipt changed")
    for item in imports:
        signature = item["signature"]
        if load(run / "noevent/embeddings/api" / (signature + ".json")) != cache_values[signature]:
            raise ValueError("continuation imported embedding batch changed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "run", "report"))
    p.add_argument("--parent-run", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--core-repo", type=Path, required=True)
    p.add_argument("--probe", type=Path, required=True)
    args = p.parse_args()
    sys.dont_write_bytecode = True
    parent, run, core, probe = [x.resolve() for x in (args.parent_run, args.run, args.core_repo, args.probe)]
    isolated(run, parent)
    config = json.loads((parent / "configuration.json").read_text())
    helper_hashes = verify_helper_sources(core, config)
    sys.path.insert(0, str(core))
    from experiments.zyf.matched_pilot import (read, records, sha, digest, freeze, write,
        source_hash, stopped_parent, run_lock, run_command, assert_identity_stopped)
    from experiments.zyf.full_suite import (accept_video_scores, prediction_inventory, write_rows, common_args)
    if core != Path(config["repos"]["noevent"]).resolve() or source_hash(core) != config["source_hashes"]["noevent"]:
        raise ValueError("original noevent core identity changed")
    from methods.longemo.noevent_runner import parser
    from methods.longemo.noevent_retrieval import _document
    from methods.longemo.embeddings import APIEncoder
    from methods.longemo.common import fingerprint

    video = "G2_V000026"
    parent_task_path = parent / "tasks/answer" / ("noevent-" + video) / "task.json"
    task = read(parent_task_path)
    if task["status"] != "error":
        raise ValueError("requires the original failed answer task")
    command = task["command"]
    old = parser().parse_args(command[command.index("methods.longemo.noevent_runner") + 1:])
    questions = records(old.data_path)
    full_questions = records(parent / "questions.json")
    if digest(full_questions) != config["questions_sha256"]:
        raise ValueError("original full question cohort changed")
    all_questions = {q["question_id"]: q for q in full_questions}
    if (len(questions) != 2 or len({q["question_id"] for q in questions}) != 2
            or any(q != all_questions.get(q["question_id"]) or q["video_id"] != video for q in questions)):
        raise ValueError("only the frozen two V26 questions may resume")
    qids = {q["question_id"] for q in questions}
    if any((parent / "accepted/noevent" / (qid + ".json")).exists() for qid in qids):
        raise ValueError("previously scored question cannot resume")
    if records(Path(old.output_dir) / "predictions.jsonl") or list((parent / "scores/noevent" / video).glob("run_*/scores.jsonl")):
        raise ValueError("old successful or uncertain answer/judge output must not be repeated")
    intent, result, saved = [read(probe / n) for n in ("intent.json", "result.json", "validated_vectors.json")]
    signature = intent["request_hash"]
    if (signature != "270812d0a362273c6c8ab8d28fd9a70055c9b7d5c982871f0ab01fa2f9da82ad"
            or result.get("http_status") != 200 or result.get("valid_vectors") is not True
            or result.get("request_hash") != signature or saved.get("signature") != signature
            or intent.get("parent_run") != str(parent) or intent.get("video_id") != video):
        raise ValueError("successful exact embedding probe required")
    source_memory = Path(old.memory_dir) / video / "memory.json"
    memory = read(source_memory)
    if (sha(source_memory) != intent["memory_sha256"] or not memory.get("complete")
            or memory.get("representation") != "window_records"):
        raise ValueError("frozen window memory differs from the verified probe")
    plans = {qid: Path(old.plans_dir) / (qid + ".json") for qid in qids}
    if any(not path.is_file() for path in plans.values()):
        raise ValueError("both previously successful plans are required")
    original_plans = read(parent / "noevent/plans/frozen" / (video + ".json"))
    original_memory = read(parent / "noevent/frozen_memories" / (video + ".json"))
    if (sha(source_memory) != original_memory["memory_sha256"]
            or any(sha(path) != original_plans.get(qid) for qid, path in plans.items())):
        raise ValueError("original frozen memory/plan receipts differ")
    protected = {p.name: sha(p) for p in sorted((parent / "accepted/noevent").glob("*.json"))}
    if len(protected) != 195:
        raise ValueError("original 195-score baseline changed")
    # Use the frozen encoder's exact chunk/config behavior without writing to an
    # old or resumed cache. All vectors are validated offline before preparation.
    with tempfile.TemporaryDirectory(prefix="longemo-embedding-verify-") as temporary:
        encoder = APIEncoder("unused-offline", Path(temporary), model=old.embedding_model,
                             base_url=old.embedding_base_url)
        pieces = [piece for w in memory["windows"] for piece in encoder.chunks(_document(memory, w))]
        imports, cache_values = [], {}
        for offset in range(0, len(pieces), 16):
            inputs = ["title: Emotional event | text: " + t for t in pieces[offset:offset + 16]]
            batch_signature = fingerprint({"config": encoder.config, "inputs": inputs})
            source = probe / "validated_vectors.json" if batch_signature == signature else Path(old.embedding_cache_dir) / "api" / (batch_signature + ".json")
            cache = read(source)
            if cache.get("signature") != batch_signature:
                raise ValueError("embedding batch signature changed")
            encoder._validate(cache["vectors"], len(inputs))
            cache_values[batch_signature] = cache
            imports.append({"signature": batch_signature, "source": str(source), "source_sha256": sha(source)})
        encoder_config = encoder.config
    if len(imports) != 3 or len(cache_values) != 3 or sum(x["signature"] == signature for x in imports) != 1:
        raise ValueError("requires the exact two original batches and one probe batch")
    manifest = {"schema_version": 1, "condition": "noevent", "parent_run": str(parent),
        "parent_configuration_sha256": sha(parent / "configuration.json"), "core_repo": str(core),
        "core_sha256": source_hash(core), "helper_hashes": helper_hashes,
        "driver_sha256": sha(__file__), "parent_answer_task_sha256": sha(parent_task_path), "question_ids": sorted(qids),
        "questions_sha256": digest(questions), "memory_source": str(source_memory),
        "memory_sha256": sha(source_memory), "plan_sources": {qid: {"path": str(path), "sha256": sha(path)} for qid, path in plans.items()},
        "protected_first_scores": protected, "probe": str(probe),
        "probe_hashes": {name: sha(probe / name) for name in ("intent.json", "result.json", "validated_vectors.json")},
        "embedding_imports": imports, "encoder_config": encoder_config,
        "embedding_profile": "unchanged; exact successful probe vectors imported into an independent cache",
        "new_answer_task_count": 1, "first_judge_tries": 1, "no_new_base_or_method": True}

    def verify():
        verify_helper_sources(core, config)
        if source_hash(core) != manifest["core_sha256"] or sha(parent_task_path) != manifest["parent_answer_task_sha256"]:
            raise ValueError("original core or answer command changed")
        if read(run / "configuration.json") != manifest:
            raise ValueError("immutable continuation configuration changed")
        if protected != {p.name: sha(p) for p in (parent / "accepted/noevent").glob("*.json")}:
            raise ValueError("old first scores changed")
        if sha(source_memory) != manifest["memory_sha256"] or any(sha(path) != manifest["plan_sources"][qid]["sha256"] for qid, path in plans.items()):
            raise ValueError("old memory/plans changed")
        copied = run / "noevent/videos" / video / "memory" / video / "memory.json"
        if sha(copied) != manifest["memory_sha256"] or any(
                sha(run / "noevent/plans" / (qid + ".json")) != manifest["plan_sources"][qid]["sha256"] for qid in qids):
            raise ValueError("continuation memory/plans changed")
        if any(sha(item["source"]) != item["source_sha256"] for item in imports):
            raise ValueError("original/probe embedding batch changed")
        verify_frozen_inputs(run, video, questions, imports, cache_values)

    if args.stage == "prepare":
        if run.exists():
            raise ValueError("prepare requires a new run directory")
        with stopped_parent(parent):
            run.mkdir(parents=True)
            freeze(run / "configuration.json", manifest)
            freeze(run / "questions.json", questions)
            freeze(run / "questions" / (video + ".json"), questions)
            destination = run / "noevent/videos" / video / "memory" / video / "memory.json"
            destination.parent.mkdir(parents=True)
            shutil.copy2(source_memory, destination)
            for qid, path in plans.items():
                target = run / "noevent/plans" / (qid + ".json")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
            for item in imports:
                batch_signature = item["signature"]
                freeze(run / "noevent/embeddings/api" / (batch_signature + ".json"), cache_values[batch_signature])
            freeze(run / "embedding_imports.json", imports)
            verify()
        print(json.dumps({"stage": "prepared", "run": str(run), "questions": sorted(qids)}))
        return

    verify()
    index = {"conditions": {"noevent": {"questions": {}}}}

    def report():
        verify()
        accept_video_scores(run, index, "noevent", video, questions)
        new = [read(p)["score"] for p in sorted((run / "accepted/noevent").glob("*.json"))]
        if any(r["question_id"] not in qids for r in new) or len({r["question_id"] for r in new}) != len(new):
            raise ValueError("new score overlap or unexpected question")
        previous = [read(p)["score"] for p in sorted((parent / "accepted/noevent").glob("*.json"))]
        combined = previous + new
        value = {"updated_unix": time.time(), "condition": "noevent", "parent_scored": len(previous),
                 "new_scored": len(new), "scored": len(combined), "total": 558,
                 "mean_scored": 100 * sum(r["normalized_score"] for r in combined) / len(combined),
                 "selected_questions": sorted(qids), "successful_answers_without_accepted_score": [qid for qid in prediction_inventory(run, index, "noevent", video)
                     if not (run / "accepted/noevent" / (qid + ".json")).exists()],
                 "judge_task_status": read(run / "tasks/score" / ("noevent-" + video) / "task.json").get("status")
                     if (run / "tasks/score" / ("noevent-" + video) / "task.json").exists() else "not_started",
                 "scope": "V26 backend recovery only; other missing noevent questions are separate"}
        write(run / "status.json", value)
        print(json.dumps(value))

    if args.stage == "report":
        report()
        return
    for path in run.glob("tasks/*/*/task.json"):
        state = read(path)
        if state.get("child"):
            assert_identity_stopped(state["child"], str(path))
        if state.get("status") in ("running", "needs_audit"):
            raise ValueError("previous outcome requires an audit; no retry")
    with run_lock(run):
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = str(core)
        env["MODEL_API_KEY"] = read(old.credential_file)["MODEL_API_KEY"]
        answer = list(command)
        for name, value in {"--data-path": run / "questions.json", "--memory-dir": run / "noevent/videos" / video / "memory",
                "--output-dir": run / "answers/noevent" / video, "--plans-dir": run / "noevent/plans",
                "--embedding-cache-dir": run / "noevent/embeddings"}.items():
            answer = replacement(answer, name, value)
        state = run_command(run / "tasks/answer" / ("noevent-" + video), answer, core, env)
        if state["status"] in ("running", "needs_audit"):
            raise ValueError("answer outcome uncertain")
        verify()
        predictions = prediction_inventory(run, index, "noevent", video)
        selected = [q for q in questions if q["question_id"] in predictions]
        freeze(run / "score_questions.json", selected)
        if selected:
            score_input = run / "score_inputs/noevent" / (video + ".jsonl")
            rows = [predictions[q["question_id"]] for q in selected]
            if score_input.exists():
                if records(score_input) != rows:
                    raise ValueError("first judge input changed")
            else:
                write_rows(score_input, rows)
            score = [config["python"], "-u", "-m", "evaluation.eval", "--data-path", str(run / "score_questions.json"),
                "--predictions", str(score_input), "--granularity", "episode", "--output-dir", str(run / "scores/noevent" / video),
                *common_args(config, "score"), "--workers", "2"]
            state = run_command(run / "tasks/score" / ("noevent-" + video), score, core, env)
            if state["status"] in ("running", "needs_audit"):
                raise ValueError("judge outcome uncertain")
        report()


if __name__ == "__main__":
    main()
