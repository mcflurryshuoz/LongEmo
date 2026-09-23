"""One V101 continuation using the already validated W64 audio response.

Preserve the stopped main24 source, its 63 completed windows, the original
195-score baseline and the 235 first scores frozen by the successful probe.
Import audio locally, then run one build task with the original 8192/4096
visual/audio budgets. No API request is made by prepare and there is no outer
retry. Other disjoint continuations may append their first scores normally.
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
from experiments.zyf import noevent_v101_audio500_probe as diagnostic
from experiments.zyf import noevent_length_continuation as length_helpers
from experiments.zyf.noevent_length_continuation import claim_video, score_snapshot, terminal_video


VIDEO, WINDOW = "G2_V000101", "W00064"
COMPLETED_WINDOWS = 63
PROBE_FIRST_SCORES = 235
EXPECTED_QIDS = ["G2_Q000368", "G2_Q000369", "G2_Q000370"]
_ORIGINAL_VERIFY, _ORIGINAL_REPORT = base.verify, base.report
_SCORE_LOCK = threading.RLock()


def arguments(args):
    return {"source_run": str(Path(args.source_run).resolve(strict=True)),
            "core_repo": str(Path(args.core_repo).resolve(strict=True)),
            "probe_dir": str(Path(args.probe_dir).resolve(strict=True)),
            "selection_sha256": base.sha(base.regular(args.selection)),
            "credential_file": str(Path(args.credential_file).resolve(strict=True)),
            "related_runs": sorted({str(Path(value).resolve(strict=True)) for value in [args.source_run, *args.related_run]}),
            "python": args.python, "workers": 1, "question_workers": 2}


def code_files():
    return {str(Path(module.__file__).resolve()): base.sha(base.regular(module.__file__))
            for module in (sys.modules[__name__], base, seed, probe, diagnostic, length_helpers)}


def original_probe_scores(probe_dir, expected_source):
    path = base.regular(Path(probe_dir) / "preflight.json")
    value = base.read(path)
    if (value.get("status") != "preflight_passed" or value.get("no_api_calls") is not True
            or value.get("video_id") != VIDEO or value.get("window_id") != WINDOW
            or value.get("request_hash") != diagnostic.FINAL_REQUEST
            or value.get("completed_windows") != COMPLETED_WINDOWS
            or value.get("protected_score_count") != PROBE_FIRST_SCORES
            or value.get("artifact", {}).get("source_run") != str(expected_source)):
        raise ValueError("the successful probe lacks its original 235-score preflight receipt")
    protected = value.get("protected_files")
    if not isinstance(protected, dict):
        raise ValueError("probe protected-file inventory is missing")
    scores = {Path(name).stem: {"path": name, "sha256": expected} for name, expected in protected.items()
              if Path(name).parent.name == "noevent" and Path(name).parent.parent.name == "accepted"}
    score_paths = [name for name in protected if Path(name).parent.name == "noevent" and Path(name).parent.parent.name == "accepted"]
    if len(scores) != PROBE_FIRST_SCORES or len(score_paths) != PROBE_FIRST_SCORES:
        raise ValueError("probe first-score inventory is incomplete or duplicated")
    for name, expected in protected.items():
        if base.sha(base.regular(name)) != expected:
            raise ValueError("an original probe-protected input or first score changed")
    return {"path": str(path), "sha256": base.sha(path), "scores": scores, "protected_files": protected}


def observe_scores(run, config):
    recovery = config["audio_seed_recovery"]
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
    if (set(config["checkpoints"]) != {VIDEO} or config["workers"] != 1 or config["question_workers"] != 2
            or config["profile"]["visual"]["max_tokens"] != 8192
            or config["checkpoints"][VIDEO]["parent_audio_observer"]["max_tokens"] != 4096
            or os.environ.get("LONGEMO_ALLOW_AUDIO_CACHE_REUSE") == "1"):
        raise ValueError("V101 must keep its original budgets, concurrency and strict audio fingerprint checks")
    for path, expected in recovery["implementation"].items():
        if base.sha(base.regular(path)) != expected:
            raise ValueError("the frozen V101 audio seed implementation changed")
    source = Path(recovery["arguments"]["source_run"])
    terminal_video(source, VIDEO)
    for path, expected in recovery["source_hashes"].items():
        if base.sha(base.regular(path)) != expected:
            raise ValueError("the stopped source configuration or video task changed")
    if base.read(base.regular(recovery["claim_path"])) != recovery["claim"]:
        raise ValueError("exclusive V101 continuation claim changed")
    old = recovery["probe_first_scores"]
    if base.sha(base.regular(old["path"])) != old["sha256"]:
        raise ValueError("original successful-probe preflight receipt changed")
    for name, expected in old["protected_files"].items():
        if base.sha(base.regular(name)) != expected:
            raise ValueError("an original probe-protected input or first score changed")
    target = seed.target_folder(run, VIDEO)
    receipt = seed.verify_seed_receipt(run, VIDEO, config["checkpoints"][VIDEO], allow_memory_growth=
                                      (run / "tasks/build" / ("noevent-" + VIDEO) / "task.json").exists())
    if (not receipt or receipt["stage"] != "audio" or receipt["window_id"] != WINDOW
            or receipt["request_hash"] != diagnostic.FINAL_REQUEST
            or base.sha(base.regular(run / "probe_seeds" / VIDEO / "receipt.json")) != recovery["expected_seed_receipt_sha256"]):
        raise ValueError("the proven W64 audio seed is missing or changed")
    cache = base.read(base.regular(target / "audio" / (WINDOW + ".json")))
    if cache["input_fingerprint"] != recovery["seed_audio_input_fingerprint"]:
        raise ValueError("the audio seed no longer matches the original initial media input")
    return observe_scores(run, config)


def import_audio(run, checkpoint, artifact, source_hashes):
    target = seed.target_folder(run, VIDEO)
    for relative, expected in {**checkpoint["committed_files"], **checkpoint["pending_audio_strict_validation"]}.items():
        if base.sha(base.regular(target / relative)) != expected:
            raise ValueError("unstarted clone differs from the stopped source")
    output = "audio/" + WINDOW + ".json"
    if (target / output).exists() or (target / "manifest.json").exists() or (target / "calls.jsonl").exists():
        raise ValueError("audio import cannot overwrite a cache or an initialized builder")
    memory = base.read(target / "memory.json")
    if len(memory["completed_windows"]) != COMPLETED_WINDOWS or WINDOW in memory["completed_windows"]:
        raise ValueError("audio import must not advance a completed window")
    root = run / "probe_seeds" / VIDEO
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    intent = {"schema_version": 1, "video_id": VIDEO, "window_id": WINDOW, "stage": "audio",
              "checkpoint_receipt_sha256": seed.digest(checkpoint), "probe_source_hashes": source_hashes, "no_api_calls": True}
    seed.exclusive_json(root / "intent.json", intent)
    seed.exclusive_json(target / output, {"input_fingerprint": artifact["audio_input_fingerprint"],
                                          "model": artifact["model_configuration"], "result": artifact["payload"]})
    receipt = {"schema_version": 1, "video_id": VIDEO, "window_id": WINDOW, "stage": "audio",
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
    source, repo = Path(frozen_args["source_run"]), Path(frozen_args["core_repo"])
    if any(run == root or root in run.parents or run in root.parents for root in (source, repo)):
        raise ValueError("V101 continuation requires a separate ordinary directory")
    if (run / "configuration.json").exists():
        config = base.read(base.regular(run / "configuration.json"))
        if config["audio_seed_recovery"]["arguments"] != frozen_args:
            raise ValueError("frozen V101 continuation arguments changed")
        verify(run, config)
        return run, config
    if os.environ.get("LONGEMO_ALLOW_AUDIO_CACHE_REUSE") == "1":
        raise ValueError("strict audio input fingerprint validation must remain enabled")
    with base.stopped_parent(source):
        prepared = diagnostic.prepare(source, repo, frozen_args["related_runs"])
        source_config = base.read(base.regular(source / "configuration.json"))
        if frozen_args["credential_file"] != source_config["credential_file"]:
            raise ValueError("V101 uses the original private credential source")
        baseline = Path(source_config["parent_run"])
        if run == baseline or baseline in run.parents or run in baseline.parents:
            raise ValueError("V101 continuation may not overlap its original baseline")
        original_scores = original_probe_scores(frozen_args["probe_dir"], source)
        artifact, probe_hashes = seed.validate_probe(frozen_args["probe_dir"], prepared)
        probe_hashes[original_scores["path"]] = original_scores["sha256"]
        checkpoint = prepared["checkpoint"]
        if (artifact["stage"] != "audio" or artifact["window_id"] != WINDOW
                or checkpoint["completed_windows"] != COMPLETED_WINDOWS
                or checkpoint["parent_manifest_sha256"] != artifact["parent_manifest_sha256"]
                or checkpoint["committed_files"]["memory.json"] != artifact["parent_memory_sha256"]):
            raise ValueError("validated audio belongs to a different source checkpoint")
        all_questions = base.read(base.regular(baseline / "questions.json"))
        selection = base.read(base.regular(args.selection))
        questions = base.selected_questions(selection, all_questions, source_config["protected_scores"])
        actual = base.read(base.regular(source / "questions" / (VIDEO + ".json")))
        if (questions != actual or {q["video_id"] for q in questions} != {VIDEO}
                or selection["question_ids"] != EXPECTED_QIDS):
            raise ValueError("selection must be the exact original unscored V101 question list")
        additional = score_snapshot(frozen_args["related_runs"], all_questions, set(selection["question_ids"]), source_config["protected_scores"])
        source_hashes = terminal_video(source, VIDEO)
        for name in ("configuration.json", "prepared.json", "questions.json", "process.json"):
            source_hashes[str(source / name)] = base.sha(base.regular(source / name))
        claim_path = baseline.parents[1] / "recovery_claims/noevent_audio_seed" / (base.digest(str(source))[:20] + "-" + VIDEO + ".json")
        claim = {"source_run": str(source), "video_id": VIDEO, "owner_run": str(run),
                 "selection_sha256": frozen_args["selection_sha256"], "source_configuration_sha256": source_hashes[str(source / "configuration.json")],
                 "probe_receipt_sha256": probe_hashes[str(Path(frozen_args["probe_dir"]) / "summary.json")], "driver_sha256": base.sha(__file__)}
        claim_video(claim_path, claim)
        run.mkdir(parents=True, exist_ok=True)
        with (run / "coordinator.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if any((run / "tasks").glob("*/*/task.json")):
                raise ValueError("V101 cannot seed a claimed build/answer/score task")
            base.clone_checkpoint(run, VIDEO, checkpoint)
            receipt = import_audio(run, checkpoint, artifact, probe_hashes)
            config = copy.deepcopy(source_config)
            config.update(selection=selection, selection_file_sha256=frozen_args["selection_sha256"],
                questions_sha256=base.digest(questions), checkpoints={VIDEO: checkpoint},
                media={VIDEO: source_config["media"][VIDEO]}, media_identity={VIDEO: prepared["media_identity"][VIDEO]},
                python=args.python, credential_file=frozen_args["credential_file"], workers=1, question_workers=2,
                comparison_scope="Separate one-task V101 continuation; same 8192 visual/4096 audio model/media/schema/retrieval/judge. The successful one-request W64 audio diagnostic is imported without reissuing it.")
            config["audio_seed_recovery"] = {"schema_version": 1, "arguments": frozen_args, "implementation": code_files(),
                "source_hashes": source_hashes, "probe_first_scores": original_scores, "additional_protected_scores": additional,
                "claim_path": str(claim_path), "claim": claim, "seed_audio_input_fingerprint": artifact["audio_input_fingerprint"],
                "expected_seed_receipt_sha256": base.sha(run / "probe_seeds" / VIDEO / "receipt.json"),
                "policy": "The old three attempts remain exhausted and immutable; one separately authorized build task, no outer retry; official judge once."}
            base.freeze(run / "configuration.json", config)
            base.freeze(run / "questions.json", questions)
            base.freeze(run / "questions" / (VIDEO + ".json"), questions)
            base.freeze(run / "prepared.json", {"configuration_sha256": base.sha(run / "configuration.json"), "no_api_calls": True,
                                               "imported_audio": WINDOW, "completed_windows": COMPLETED_WINDOWS})
        if any(base.sha(base.regular(path)) != expected for path, expected in prepared["protected"].items()):
            raise ValueError("source changed while importing successful audio")
    verify(run, config)
    return run, config


def report(run, output=None):
    run = Path(run).resolve(strict=True)
    config = base.read(base.regular(run / "configuration.json"))
    other = verify(run, config)
    value = _ORIGINAL_REPORT(run, output)
    rows = {row["question_id"]: row["score"] for row in value["questions"]}
    for qid, item in other.items():
        if qid in rows:
            raise ValueError("duplicate first score in V101 combined report")
        rows[qid] = item["score"]
    value["combined_baseline_and_this_run"] = value["combined"]
    value["combined"] = {"scored": len(rows), "total": config["total_questions"],
        "coverage": len(rows) / config["total_questions"], "mean": 100 * sum(row["normalized_score"] for row in rows.values()) / len(rows)}
    value["other_continuation_first_scores"] = other
    value["audio_seed"] = {"window_id": WINDOW, "request_hash": diagnostic.FINAL_REQUEST,
        "visual_max_tokens": 8192, "audio_max_tokens": 4096, "no_successful_audio_request_repeated": True,
        "receipt_sha256": config["audio_seed_recovery"]["expected_seed_receipt_sha256"]}
    target = Path(output).resolve() if output else run / "report"
    base.write(target / "report.json", value)
    (target / "report.md").write_text("# noevent V101 音频恢复评测\n\n"
        f"合并首次有效评分 **{len(rows)}/{config['total_questions']}**，均分 **{value['combined']['mean']:.2f}**；"
        f"本批新增 **{value['continuation']['scored']}/{value['continuation']['selected']}**。\n\n"
        "继承63个完整窗口并导入已验证的W64音频；视觉8192、音频4096及原模型/采样/检索/评分不变。"
        "成功音频不重发，旧尝试和首次评分保留，原实验报告不覆盖。\n")
    return value


def install_hooks():
    base.verify, base.report = verify, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("stage", choices=("prepare", "run", "report"))
    parser.add_argument("--run", required=True)
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
        if any(getattr(args, field) is None for field in ("source_run", "core_repo", "probe_dir", "selection", "credential_file")):
            raise SystemExit("prepare/run require source, core, successful probe, selection and private credential paths")
        run, config = prepare(args)
        if args.stage == "run":
            install_hooks()
        result = base.execute(run, config) if args.stage == "run" else {"prepared": str(run), "selected": len(config["selection"]["question_ids"])}
    print(json.dumps(result.get("combined", result), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
