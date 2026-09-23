"""Fixed audio429 continuations importing successful one-shot audio probes.

Each process handles one fixed video, preserving the source output budget and
all first scores. Preparation imports audio without an API call or advancing a
window. One build task only, no automatic outer recovery.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import sys
import threading

from experiments.zyf import noevent_continuation as base
from experiments.zyf import noevent_probe_seed as seed
from experiments.zyf import noevent_retry_probe as probe
from experiments.zyf import noevent_audio429_probe as diagnostic
from experiments.zyf import noevent_length_continuation as length_helpers
from experiments.zyf.noevent_length_continuation import claim_video, score_snapshot, terminal_video


_ORIGINAL_VERIFY, _ORIGINAL_REPORT = base.verify, base.report
_SCORE_LOCK = threading.RLock()


def arguments(args):
    return {"video_id": args.video_id, "specs": str(Path(args.specs).resolve(strict=True)),
            "specs_sha256": base.sha(base.regular(args.specs)), "source_run": str(Path(args.source_run).resolve(strict=True)),
            "core_repo": str(Path(args.core_repo).resolve(strict=True)),
            "probe_dir": str(Path(args.probe_dir).resolve(strict=True)),
            "selection_sha256": base.sha(base.regular(args.selection)),
            "credential_file": str(Path(args.credential_file).resolve(strict=True)),
            "related_runs": sorted({str(Path(value).resolve(strict=True)) for value in [args.source_run, *args.related_run]}),
            "python": args.python, "workers": 1, "question_workers": 2}


def code_files():
    return {str(Path(module.__file__).resolve()): base.sha(base.regular(module.__file__))
            for module in (sys.modules[__name__], base, seed, probe, diagnostic, length_helpers)}


def original_probe_scores(probe_dir, expected_source, case):
    path = base.regular(Path(probe_dir) / "preflight.json")
    value = base.read(path)
    if (value.get("status") != "preflight_passed" or value.get("no_api_calls") is not True
            or value.get("video_id") != case["video"] or value.get("window_id") != case["window"]
            or value.get("request_hash") != case["ledger_rows"][-1]["request_hash"]
            or value.get("completed_windows") != case["completed_windows"]
            or value.get("protected_score_count") != diagnostic.FIRST_SCORES
            or value.get("artifact", {}).get("source_run") != str(expected_source)):
        raise ValueError("the successful probe lacks its original 243-score preflight receipt")
    protected = value.get("protected_files")
    if not isinstance(protected, dict):
        raise ValueError("probe protected-file inventory is missing")
    scores = {Path(name).stem: {"path": name, "sha256": expected} for name, expected in protected.items()
              if Path(name).parent.name == "noevent" and Path(name).parent.parent.name == "accepted"}
    score_paths = [name for name in protected if Path(name).parent.name == "noevent" and Path(name).parent.parent.name == "accepted"]
    if len(scores) != diagnostic.FIRST_SCORES or len(score_paths) != diagnostic.FIRST_SCORES:
        raise ValueError("probe first-score inventory is incomplete or duplicated")
    for name, expected in protected.items():
        if base.sha(base.regular(name)) != expected:
            raise ValueError("an original probe-protected input or first score changed")
    return {"path": str(path), "sha256": base.sha(path), "scores": scores, "protected_files": protected}


def observe_scores(run, config):
    recovery = config["audio_seed_recovery"]
    case = recovery["case"]
    folder = run / "observed_first_scores"
    folder.mkdir(parents=True, exist_ok=True)
    with _SCORE_LOCK, (folder / "guard.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = score_snapshot(recovery["arguments"]["related_runs"], base.read(Path(config["parent_run"]) / "questions.json"),
                                 set(config["selection"]["question_ids"]), config["protected_scores"])
        expected = dict(recovery["additional_protected_scores"])
        for path in folder.glob("*.json"):
            value = base.read(base.regular(path))
            if path.stem in expected and expected[path.stem] != value:
                raise ValueError("persistent observed first-score receipt changed")
            expected[path.stem] = value
        if any(current.get(qid) != value for qid, value in expected.items()):
            raise ValueError("a previously observed sibling first score changed or disappeared")
        for qid, value in current.items():
            base.freeze(folder / (qid + ".json"), value)
        return current


def verify(run, config):
    _ORIGINAL_VERIFY(run, config)
    recovery = config["audio_seed_recovery"]
    case = recovery["case"]
    if (set(config["checkpoints"]) != {case["video"]} or config["workers"] != 1 or config["question_workers"] != 2
            or config["profile"]["visual"]["max_tokens"] != case["visual_max_tokens"]
            or config["checkpoints"][case["video"]]["parent_audio_observer"]["max_tokens"] != 4096
            or os.environ.get("LONGEMO_ALLOW_AUDIO_CACHE_REUSE") == "1"):
        raise ValueError("audio429 must keep its original budgets, concurrency and strict audio fingerprint checks")
    for path, expected in recovery["implementation"].items():
        if base.sha(base.regular(path)) != expected:
            raise ValueError("the frozen audio429 audio seed implementation changed")
    source = Path(recovery["arguments"]["source_run"])
    terminal_video(source, case["video"])
    for path, expected in recovery["source_hashes"].items():
        if base.sha(base.regular(path)) != expected:
            raise ValueError("the stopped source configuration or video task changed")
    if base.read(base.regular(recovery["claim_path"])) != recovery["claim"]:
        raise ValueError("exclusive audio429 continuation claim changed")
    old = recovery["probe_first_scores"]
    if base.sha(base.regular(old["path"])) != old["sha256"]:
        raise ValueError("original successful-probe preflight receipt changed")
    for name, expected in old["protected_files"].items():
        if base.sha(base.regular(name)) != expected:
            raise ValueError("an original probe-protected input or first score changed")
    target = seed.target_folder(run, case["video"])
    receipt = seed.verify_seed_receipt(run, case["video"], config["checkpoints"][case["video"]], allow_memory_growth=
                                      (run / "tasks/build" / ("noevent-" + case["video"]) / "task.json").exists())
    if (not receipt or receipt["stage"] != "audio" or receipt["window_id"] != case["window"]
            or receipt["request_hash"] != case["ledger_rows"][-1]["request_hash"]
            or base.sha(base.regular(run / "probe_seeds" / case["video"] / "receipt.json")) != recovery["expected_seed_receipt_sha256"]):
        raise ValueError("the proven source-window audio seed is missing or changed")
    cache = base.read(base.regular(target / "audio" / (case["window"] + ".json")))
    if cache["input_fingerprint"] != recovery["seed_audio_input_fingerprint"]:
        raise ValueError("the audio seed no longer matches the original initial media input")
    return observe_scores(run, config)


def import_audio(run, checkpoint, artifact, source_hashes, case):
    target = seed.target_folder(run, case["video"])
    for relative, expected in {**checkpoint["committed_files"], **checkpoint["pending_audio_strict_validation"]}.items():
        if base.sha(base.regular(target / relative)) != expected:
            raise ValueError("unstarted clone differs from the stopped source")
    output = "audio/" + case["window"] + ".json"
    if (target / output).exists() or (target / "manifest.json").exists() or (target / "calls.jsonl").exists():
        raise ValueError("audio import cannot overwrite a cache or an initialized builder")
    memory = base.read(target / "memory.json")
    if len(memory["completed_windows"]) != case["completed_windows"] or case["window"] in memory["completed_windows"]:
        raise ValueError("audio import must not advance a completed window")
    root = run / "probe_seeds" / case["video"]
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    intent = {"schema_version": 1, "video_id": case["video"], "window_id": case["window"], "stage": "audio",
              "checkpoint_receipt_sha256": seed.digest(checkpoint), "probe_source_hashes": source_hashes, "no_api_calls": True}
    seed.exclusive_json(root / "intent.json", intent)
    seed.exclusive_json(target / output, {"input_fingerprint": artifact["audio_input_fingerprint"],
                                          "model": artifact["model_configuration"], "result": artifact["payload"]})
    receipt = {"schema_version": 1, "video_id": case["video"], "window_id": case["window"], "stage": "audio",
        "checkpoint_receipt_sha256": seed.digest(checkpoint), "intent_sha256": base.sha(root / "intent.json"),
        "probe_source_hashes": source_hashes,
        "target_files_before": {**checkpoint["committed_files"], **checkpoint["pending_audio_strict_validation"]},
        "target_files_after": {"memory.json": base.sha(target / "memory.json"), output: base.sha(target / output)},
        "source": "validated_one_request_probe", "request_hash": artifact["request_hash"],
        "model_configuration": artifact["model_configuration"], "official_result": False, "no_api_calls": True}
    seed.exclusive_json(root / "receipt.json", receipt)
    return receipt


def prepare(args):
    run = Path(args.run).resolve()
    frozen_args = arguments(args)
    case = diagnostic.load_case(frozen_args["specs"], frozen_args["video_id"])
    if case["source_run"] != frozen_args["source_run"]:
        raise ValueError("fixed case belongs to a different source run")
    source, repo = Path(frozen_args["source_run"]), Path(frozen_args["core_repo"])
    if any(run == root or root in run.parents or run in root.parents for root in (source, repo)):
        raise ValueError("audio429 continuation requires a separate ordinary directory")
    if (run / "configuration.json").exists():
        config = base.read(base.regular(run / "configuration.json"))
        if config["audio_seed_recovery"]["arguments"] != frozen_args:
            raise ValueError("frozen audio429 continuation arguments changed")
        verify(run, config)
        return run, config
    if os.environ.get("LONGEMO_ALLOW_AUDIO_CACHE_REUSE") == "1":
        raise ValueError("strict audio input fingerprint validation must remain enabled")
    with base.stopped_parent(source):
        prepared = diagnostic.prepare(frozen_args["specs"], case["video"], repo, frozen_args["related_runs"])
        source_config = base.read(base.regular(source / "configuration.json"))
        if frozen_args["credential_file"] != source_config["credential_file"]:
            raise ValueError("audio429 uses the original private credential source")
        baseline = Path(source_config["parent_run"])
        if run == baseline or baseline in run.parents or run in baseline.parents:
            raise ValueError("audio429 continuation may not overlap its original baseline")
        original_scores = original_probe_scores(frozen_args["probe_dir"], source, case)
        artifact, probe_hashes = seed.validate_probe(frozen_args["probe_dir"], prepared)
        probe_hashes[original_scores["path"]] = original_scores["sha256"]
        checkpoint = prepared["checkpoint"]
        if (artifact["stage"] != "audio" or artifact["window_id"] != case["window"]
                or checkpoint["completed_windows"] != case["completed_windows"]
                or checkpoint["parent_manifest_sha256"] != artifact["parent_manifest_sha256"]
                or checkpoint["committed_files"]["memory.json"] != artifact["parent_memory_sha256"]):
            raise ValueError("validated audio belongs to a different source checkpoint")
        all_questions = base.read(base.regular(baseline / "questions.json"))
        selection = base.read(base.regular(args.selection))
        questions = base.selected_questions(selection, all_questions, source_config["protected_scores"])
        actual = base.read(base.regular(source / "questions" / (case["video"] + ".json")))
        if (questions != actual or {q["video_id"] for q in questions} != {case["video"]}
                or selection["question_ids"] != case["question_ids"]):
            raise ValueError("selection must be the exact original unscored audio429 question list")
        additional = score_snapshot(frozen_args["related_runs"], all_questions, set(selection["question_ids"]), source_config["protected_scores"])
        source_hashes = terminal_video(source, case["video"])
        for name in ("configuration.json", "prepared.json", "questions.json", "process.json"):
            source_hashes[str(source / name)] = base.sha(base.regular(source / name))
        claim_path = baseline.parents[1] / "recovery_claims/noevent_audio429_seed" / (base.digest(str(source))[:20] + "-" + case["video"] + ".json")
        claim = {"source_run": str(source), "video_id": case["video"], "owner_run": str(run),
                 "selection_sha256": frozen_args["selection_sha256"], "source_configuration_sha256": source_hashes[str(source / "configuration.json")],
                 "probe_receipt_sha256": probe_hashes[str(Path(frozen_args["probe_dir"]) / "summary.json")], "driver_sha256": base.sha(__file__)}
        claim_video(claim_path, claim)
        run.mkdir(parents=True, exist_ok=True)
        with (run / "coordinator.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if any((run / "tasks").glob("*/*/task.json")):
                raise ValueError("audio429 cannot seed a claimed build/answer/score task")
            base.clone_checkpoint(run, case["video"], checkpoint)
            receipt = import_audio(run, checkpoint, artifact, probe_hashes, case)
            config = copy.deepcopy(source_config)
            source_lineage = inherited_lineage(source, source_config, case["video"], checkpoint)
            config.pop("length_recovery", None)
            config.update(selection=selection, selection_file_sha256=frozen_args["selection_sha256"],
                questions_sha256=base.digest(questions), checkpoints={case["video"]: checkpoint},
                media={case["video"]: source_config["media"][case["video"]]}, media_identity={case["video"]: prepared["media_identity"][case["video"]]},
                python=args.python, credential_file=frozen_args["credential_file"], workers=1, question_workers=2,
                comparison_scope="Separate fixed-case audio429 continuation; source visual budget and 4096 audio model/media/schema/retrieval/judge unchanged. The successful one-request source-window audio diagnostic is imported without reissuing it.")
            config["audio_seed_recovery"] = {"schema_version": 1, "case": case, "arguments": frozen_args, "implementation": code_files(),
                "source_hashes": source_hashes, "inherited_window_sources": source_lineage, "probe_first_scores": original_scores, "additional_protected_scores": additional,
                "claim_path": str(claim_path), "claim": claim, "seed_audio_input_fingerprint": artifact["audio_input_fingerprint"],
                "expected_seed_receipt_sha256": base.sha(run / "probe_seeds" / case["video"] / "receipt.json"),
                "policy": "The old three attempts remain exhausted and immutable; one separately authorized build task, no outer retry; official judge once."}
            base.freeze(run / "configuration.json", config)
            base.freeze(run / "questions.json", questions)
            base.freeze(run / "questions" / (case["video"] + ".json"), questions)
            base.freeze(run / "prepared.json", {"configuration_sha256": base.sha(run / "configuration.json"), "no_api_calls": True,
                                               "imported_audio": case["window"], "completed_windows": case["completed_windows"]})
        if any(base.sha(base.regular(path)) != expected for path, expected in prepared["protected"].items()):
            raise ValueError("source changed while importing successful audio")
    verify(run, config)
    return run, config



def inherited_lineage(source, source_config, video, checkpoint):
    """Resolve original/main/length windows before the new builder starts."""
    previous = source_config["checkpoints"][video]
    length = source_config.get("length_recovery", {}).get("videos", {}).get(video)
    memory = base.read(Path(checkpoint["source"]) / "memory.json")
    output = {}
    for i, window in enumerate(memory["completed_windows"]):
        if length and i < length["original_completed_windows"]:
            origin, directory, model = "original", length["original_source"], length["original_model"]
        elif length and i < length["source_completed_windows"]:
            origin, directory, model = "source_continuation", previous["source"], previous["parent_model"]
        elif i < previous["completed_windows"]:
            origin, directory, model = "original", previous["source"], previous["parent_model"]
        else:
            origin, directory, model = ("length_continuation" if length else "source_continuation"), checkpoint["source"], checkpoint["parent_model"]
        relative = "windows/" + window + ".json"
        expected = checkpoint["committed_files"][relative]
        if base.sha(base.regular(Path(directory) / relative)) != expected:
            raise ValueError("inherited window does not match its exact original source")
        output[window] = {"origin": origin, "source": directory, "model": model, "window_sha256": expected}
    return output


def window_provenance(run, config, video):
    folder = base.memory_root(run, "noevent", video) / video
    memory = base.read(base.regular(folder / "memory.json"))
    manifest = base.read(base.regular(folder / "manifest.json"))
    clients = {key: manifest["configuration"][key] for key in ("model", "audio_observer")}
    checked = base.validate_parent_checkpoint(folder, config["core_repo"], "noevent", video,
                                             config["media"][video], config["profile"], clients)
    if not memory.get("complete") or memory.get("representation") != "window_records" or not checked["complete"]:
        raise ValueError("only replay-validated complete window memory can be answered")
    inherited = config["audio_seed_recovery"]["inherited_window_sources"]
    result = {}
    for window in memory["completed_windows"]:
        actual = base.sha(base.regular(folder / "windows" / (window + ".json")))
        if window in inherited:
            row = inherited[window]
            if actual != row["window_sha256"] or base.sha(base.regular(Path(row["source"]) / "windows" / (window + ".json"))) != actual:
                raise ValueError("inherited window or its source changed")
            result[window] = copy.deepcopy(row)
        else:
            result[window] = {"origin": "audio429_continuation", "source": str(folder),
                              "model": manifest["configuration"]["model"], "window_sha256": actual}
    value = {"memory_sha256": base.sha(folder / "memory.json"), "windows": result}
    base.freeze(run / "noevent/frozen_memories" / (video + ".json"), value)
    base.freeze(run / "lineage/windows" / (video + ".json"), value)
    return value


def report(run, output=None):
    run = Path(run).resolve(strict=True)
    config = base.read(base.regular(run / "configuration.json"))
    case = config["audio_seed_recovery"]["case"]
    other = verify(run, config)
    value = _ORIGINAL_REPORT(run, output)
    rows = {row["question_id"]: row["score"] for row in value["questions"]}
    for qid, item in other.items():
        if qid in rows:
            raise ValueError("duplicate first score in audio429 combined report")
        rows[qid] = item["score"]
    value["combined_baseline_and_this_run"] = value["combined"]
    value["combined"] = {"scored": len(rows), "total": config["total_questions"],
        "coverage": len(rows) / config["total_questions"], "mean": 100 * sum(row["normalized_score"] for row in rows.values()) / len(rows)}
    value["other_continuation_first_scores"] = other
    value["audio_seed"] = {"window_id": case["window"], "request_hash": case["ledger_rows"][-1]["request_hash"],
        "visual_max_tokens": case["visual_max_tokens"], "audio_max_tokens": 4096, "no_successful_audio_request_repeated": True,
        "receipt_sha256": config["audio_seed_recovery"]["expected_seed_receipt_sha256"]}
    target = Path(output).resolve() if output else run / "report"
    base.write(target / "report.json", value)
    (target / "report.md").write_text("# noevent audio429 音频恢复评测\n\n"
        f"合并首次有效评分 **{len(rows)}/{config['total_questions']}**，均分 **{value['combined']['mean']:.2f}**；"
        f"本批新增 **{value['continuation']['scored']}/{value['continuation']['selected']}**。\n\n"
        f"继承{case['completed_windows']}个完整窗口并导入已验证音频；视觉{case['visual_max_tokens']}、音频4096及原模型/采样/检索/评分不变。"
        "成功音频不重发，旧尝试和首次评分保留，原实验报告不覆盖。\n")
    return value


def install_hooks():
    base.verify, base.report, base.window_provenance = verify, report, window_provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("stage", choices=("prepare", "run", "report"))
    parser.add_argument("--run", required=True)
    parser.add_argument("--video-id", choices=sorted(diagnostic.CASES))
    parser.add_argument("--specs")
    for name in ("source-run", "core-repo", "probe-dir", "selection", "credential-file"):
        parser.add_argument("--" + name)
    parser.add_argument("--related-run", action="append", default=[])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.stage == "report":
        install_hooks()
        result = report(args.run, args.output)
    else:
        if any(getattr(args, field) is None for field in ("source_run", "core_repo", "probe_dir", "selection", "credential_file", "video_id", "specs")):
            raise SystemExit("prepare/run require source, core, successful probe, selection and private credential paths")
        run, config = prepare(args)
        if args.stage == "run":
            install_hooks()
        result = base.execute(run, config) if args.stage == "run" else {"prepared": str(run), "selected": len(config["selection"]["question_ids"])}
    print(json.dumps(result.get("combined", result), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
