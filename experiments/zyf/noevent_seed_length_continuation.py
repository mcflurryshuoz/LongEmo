"""A bounded 16K length continuation preserving prior audio-seed ancestry.

The source coordinator may still process unrelated videos. Only a source video
whose failed build child has stopped and whose pipeline is terminal is cloned.
The original 195-score parent remains the baseline; source and sibling scores
are protected as a monotone subset and are never scheduled by this driver.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import threading

from experiments.zyf import noevent_continuation as base
from experiments.zyf import noevent_probe_seed as seed
from experiments.zyf.matched_pilot import (
    assert_identity_stopped, digest, freeze, memory_root, read, records, sha,
    source_hash, stopped_parent, write,
)


_BASE_VERIFY = base.verify
_BASE_REPORT = base.report
_BASE_PROVENANCE = base.window_provenance
_REFUSAL = {"contentfilter", "contentpolicyviolation", "responsibleaipolicyviolation",
            "blockedprompt", "blockedinput", "safety", "prohibitedcontent"}
_SCORE_LOCK = threading.RLock()


def original_source_arguments(args):
    return {"source_run": str(Path(args.source_run).absolute()), "baseline_run": str(Path(args.baseline_run).absolute()),
            "core_repo": str(Path(args.core_repo).absolute()), "selection_sha256": sha(base.regular(args.selection)),
            "related_runs": sorted({str(Path(value).absolute()) for value in [args.source_run, *args.related_run]}),
            "credential_file": str(Path(args.credential_file).absolute()), "python": args.python,
            "workers": args.workers, "question_workers": args.question_workers}


def refuse_raw(raw):
    if not isinstance(raw, dict):
        return True
    feedback = raw.get("promptFeedback")
    if isinstance(feedback, dict) and feedback.get("blockReason"):
        return True
    error = raw.get("error")
    if error:
        return True
    for choice in raw.get("choices") or []:
        if not isinstance(choice, dict):
            return True
        message = choice.get("message")
        if choice.get("finish_reason") == "content_filter" or isinstance(message, dict) and message.get("refusal"):
            return True
    return False


def terminal_video(source_run, video):
    task_path = source_run / "tasks/build" / ("noevent-" + video) / "task.json"
    pipeline_path = source_run / "pipelines/noevent" / (video + ".json")
    task, state = read(base.regular(task_path)), read(base.regular(pipeline_path))
    if (task.get("status") != "error" or type(task.get("returncode")) is not int
            or task["returncode"] == 0 or state.get("status") != "build_failed"):
        raise ValueError("source video must have a terminal failed build and pipeline")
    if not task.get("child"):
        raise ValueError("source build child identity is missing")
    assert_identity_stopped(task["child"], "length continuation source build")
    for stage in ("plan", "answer", "score"):
        if (source_run / "tasks" / stage / ("noevent-" + video) / "task.json").exists():
            raise ValueError("source video already has backend work")
    return {str(task_path): sha(task_path), str(pipeline_path): sha(pipeline_path)}


def length_evidence(source_run, video, checkpoint):
    folder = Path(checkpoint["source"])
    window = f"W{checkpoint['completed_windows'] + 1:05d}"
    purpose = "window_perception:" + video + ":" + window
    rows = [row for row in records(folder / "calls.jsonl") if row.get("purpose") == purpose]
    if not rows:
        raise ValueError("missing final visual failure ledger")
    last = rows[-1]
    if (last.get("status") != "error" or last.get("failure_stage") != "generate"
            or last.get("error_type") != "RuntimeError" or last.get("will_retry") is not False):
        raise ValueError("last source request is not a stopped visual RuntimeError")
    request_hash = last.get("request_hash")
    if not re.fullmatch(r"[a-f0-9]{64}", str(request_hash)):
        raise ValueError("final visual request lacks a matching hash")
    code = re.sub(r"[^a-z0-9]", "", str(last.get("service_error_code", "")).lower())
    if code in _REFUSAL:
        raise ValueError("a policy refusal cannot be length-recovered")
    found = []
    for path in sorted((source_run / "private_diagnostics/build" / video).glob("*.json")):
        if path.name.endswith(".intent.json"):
            continue
        value = read(base.regular(path))
        if value.get("request_hash") != request_hash:
            continue
        if value.get("outcome") != "error" or value.get("error_type") != "RuntimeError":
            raise ValueError("matched diagnostic has an unexpected outcome")
        if (value.get("model_configuration") != checkpoint["parent_model"]
                or value.get("model_configuration", {}).get("max_tokens") != 8192):
            raise ValueError("length recovery is restricted to the original 8192-token visual request")
        responses = value.get("responses")
        if not isinstance(responses, list) or len(responses) != 1 or responses[0].get("http_status") != 200:
            raise ValueError("length recovery requires one successful HTTP response")
        response = responses[0]
        body = response.get("body_text")
        if not isinstance(body, str) or hashlib.sha256(body.encode()).hexdigest() != response.get("body_sha256"):
            raise ValueError("raw diagnostic response was altered or redacted; audit rather than guessing")
        raw = json.loads(body)
        choices = raw.get("choices") if isinstance(raw, dict) else None
        if (refuse_raw(raw) or not isinstance(choices, list) or len(choices) != 1
                or choices[0].get("finish_reason") != "length"):
            raise ValueError("raw response does not exclusively establish output truncation")
        usage = raw.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        # Providers differ in whether completion_tokens includes reasoning.
        # Store the observed numbers without inferring a universal sum formula.
        usage_fields = {key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens")
                        if type(usage.get(key)) in (int, float)}
        if type(details.get("reasoning_tokens")) in (int, float):
            usage_fields["completion_tokens_details.reasoning_tokens"] = details["reasoning_tokens"]
        found.append({"video_id": video, "window_id": window, "request_hash": request_hash,
                      "diagnostic_path": str(path), "diagnostic_sha256": sha(path), "response_sha256": response["body_sha256"],
                      "http_status": 200, "finish_reason": "length", "original_visual_max_tokens": 8192,
                      "usage": usage_fields, "ledger_path": str(folder / "calls.jsonl"), "ledger_sha256": sha(folder / "calls.jsonl")})
    if len(found) != 1:
        raise ValueError("expected exactly one matching final truncation diagnostic")
    return found[0]


def score_snapshot(runs, all_questions, forbidden, baseline):
    qmap = {q["question_id"]: q for q in all_questions}
    seen = set(baseline)
    result = {}
    for directory in runs:
        folder = Path(directory)
        if not folder.is_dir() or folder.is_symlink():
            raise ValueError("related noevent run is unavailable")
        for path in sorted((folder / "accepted/noevent").glob("*.json")):
            value, qid = read(base.regular(path)), path.stem
            if qid in forbidden or qid in seen or qid not in qmap:
                raise ValueError("cross-run first-score overlap or forbidden selected question")
            seen.add(qid)
            row = value["score"]
            maximum = max(int(k) for k in qmap[qid]["rubric"]["scores"])
            if (row.get("question_id") != qid or row.get("video_id") != qmap[qid]["video_id"] or row.get("status") != "ok"
                    or row.get("max_score") != maximum or type(row.get("score")) not in (int, float)
                    or not 0 <= row["score"] <= maximum or type(row.get("normalized_score")) not in (int, float)
                    or not math.isclose(row["normalized_score"], row["score"] / maximum)):
                raise ValueError("invalid related first score")
            source = base.regular(value["source"])
            if row not in records(source):
                raise ValueError("related first score is absent from its official source")
            result[qid] = {"path": str(path), "sha256": sha(path), "source": str(source),
                           "source_sha256": sha(source), "score": row, "run": str(folder)}
    return result


def claim_video(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError("recovery claim may not use a symlink")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if read(base.regular(path)) != value:
            raise ValueError("another recovery run already owns this source video") from None
    else:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.write("\n")


def protect_observed_scores(run, config):
    """Freeze each newly observed sibling first score; never forget later ones."""
    recovery = config["length_recovery"]
    directory = run / "observed_first_scores"
    directory.mkdir(parents=True, exist_ok=True)
    with _SCORE_LOCK, (directory / "guard.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # Take the snapshot under the lock: an earlier snapshot must not mistake
        # another verifier's newly observed score for a subsequent deletion.
        current = score_snapshot(recovery["related_runs"], read(Path(config["parent_run"]) / "questions.json"),
                                 set(config["selection"]["question_ids"]), config["protected_scores"])
        expected = dict(recovery["additional_protected_scores"])
        for path in directory.glob("*.json"):
            value = read(base.regular(path))
            if path.stem in expected and expected[path.stem] != value:
                raise ValueError("persistent first-score observation changed")
            expected[path.stem] = value
        for qid, original in expected.items():
            if current.get(qid) != original:
                raise ValueError("a previously observed related first score changed or disappeared")
        for qid, value in current.items():
            freeze(directory / (qid + ".json"), value)
        return current



def audio_seed_ancestry(source, source_config):
    """Verify the stopped audio seed source without writing its observations."""
    recovery = source_config.get("audio_seed_recovery")
    if not recovery:
        return {"protected_files": {}, "inherited_window_sources": {}}
    case = recovery["case"]
    video = case["video"]
    if set(source_config["checkpoints"]) != {video}:
        raise ValueError("audio seed source must remain its fixed single-video task")
    terminal_video(source, video)
    receipt_path = source / "probe_seeds" / video / "receipt.json"
    receipt = seed.verify_seed_receipt(source, video, source_config["checkpoints"][video], allow_memory_growth=True)
    if (not receipt or sha(base.regular(receipt_path)) != recovery["expected_seed_receipt_sha256"]
            or receipt["stage"] != "audio" or receipt["window_id"] != case["window"]):
        raise ValueError("source validated audio seed receipt changed")
    old = recovery["probe_first_scores"]
    protected = dict(old["protected_files"])
    protected[old["path"]] = old["sha256"]
    protected.update(recovery["implementation"])
    protected.update(recovery["source_hashes"])
    protected[str(receipt_path)] = recovery["expected_seed_receipt_sha256"]
    protected[str(Path(seed.__file__).resolve())] = sha(base.regular(seed.__file__))
    if read(base.regular(recovery["claim_path"])) != recovery["claim"]:
        raise ValueError("source audio seed exclusive ownership changed")
    protected[recovery["claim_path"]] = sha(base.regular(recovery["claim_path"]))
    folder = memory_root(source, "noevent", video) / video
    cache = read(base.regular(folder / "audio" / (case["window"] + ".json")))
    if cache["input_fingerprint"] != recovery["seed_audio_input_fingerprint"]:
        raise ValueError("source audio seed no longer matches its original media")
    inherited = recovery["inherited_window_sources"]
    windows = {}
    for window in read(base.regular(folder / "memory.json"))["completed_windows"]:
        actual = sha(base.regular(folder / "windows" / (window + ".json")))
        if window in inherited:
            item = inherited[window]
            if actual != item["window_sha256"]:
                raise ValueError("source changed an inherited window")
            original = str(Path(item["source"]) / "windows" / (window + ".json"))
            protected[original] = actual
            windows[window] = item
        else:
            windows[window] = {"origin": "audio429_continuation", "source": str(folder),
                              "window_sha256": actual, "model": read(base.regular(folder / "manifest.json"))["configuration"]["model"]}
    for path, expected in protected.items():
        if sha(base.regular(path)) != expected:
            raise ValueError("an audio-seed ancestor or original first score changed")
    return {"protected_files": protected, "inherited_window_sources": windows}


def verify(run, config):
    _BASE_VERIFY(run, config)
    recovery = config["length_recovery"]
    if sha(base.regular(__file__)) != recovery["driver_sha256"]:
        raise ValueError("length continuation driver changed")
    if config["profile"]["visual"]["max_tokens"] != 16384:
        raise ValueError("length continuation must use its fixed 16384-token profile")
    source = Path(recovery["source_run"])
    if sha(base.regular(source / "configuration.json")) != recovery["source_configuration_sha256"]:
        raise ValueError("source continuation configuration changed")
    if (sha(base.regular(source / "prepared.json")) != recovery["source_prepared_sha256"]
            or read(source / "prepared.json")["configuration_sha256"] != recovery["source_configuration_sha256"]):
        raise ValueError("source preparation receipt changed")
    for video, item in recovery["videos"].items():
        terminal_video(source, video)
        for path, expected in item["source_state_hashes"].items():
            if sha(base.regular(path)) != expected:
                raise ValueError("source terminal video evidence changed")
        if sha(base.regular(item["evidence"]["diagnostic_path"])) != item["evidence"]["diagnostic_sha256"]:
            raise ValueError("original length response evidence changed")
        if read(base.regular(item["claim_path"])) != item["claim"]:
            raise ValueError("exclusive source-video recovery claim changed")
    for path, expected in recovery["source_seed_ancestry"]["protected_files"].items():
        if sha(base.regular(path)) != expected:
            raise ValueError("frozen audio-seed ancestry changed")
    return protect_observed_scores(run, config)


def prepare(args):
    run, source, baseline = (Path(value).absolute() for value in (args.run, args.source_run, args.baseline_run))
    arguments = original_source_arguments(args)
    if (run == source or run == baseline or run.is_relative_to(source) or source.is_relative_to(run)
            or run.is_relative_to(baseline) or baseline.is_relative_to(run)):
        raise ValueError("recovery must own a distinct run directory")
    if (run / "configuration.json").exists():
        config = read(base.regular(run / "configuration.json"))
        if config["length_recovery"]["arguments"] != arguments:
            raise ValueError("frozen length continuation arguments changed")
        verify(run, config)
        return run, config
    source_config = read(base.regular(source / "configuration.json"))
    if read(base.regular(source / "prepared.json"))["configuration_sha256"] != sha(source / "configuration.json"):
        raise ValueError("source configuration differs from its preparation receipt")
    if source_config.get("condition") != "noevent" or source_config.get("parent_run") != str(baseline):
        raise ValueError("source must be the noevent continuation of the closed baseline")
    seed_ancestry = audio_seed_ancestry(source, source_config)
    if source_config["profile"]["visual"]["max_tokens"] != 8192:
        raise ValueError("source visual output budget must be exactly 8192")
    if args.workers < 1 or args.question_workers != 2:
        raise ValueError("positive video concurrency and the existing two-question concurrency are required")
    with stopped_parent(baseline):
        old = read(base.regular(baseline / "configuration.json"))
        repo = Path(args.core_repo).absolute()
        if str(repo) != old["repos"]["noevent"] or source_hash(repo) != old["source_hashes"]["noevent"]:
            raise ValueError("frozen noevent core differs")
        if source_config["core_repo"] != str(repo) or source_config["core_sha256"] != old["source_hashes"]["noevent"]:
            raise ValueError("source continuation core differs from the closed baseline")
        if (source_config["coordinator_sha256"] != sha(base.__file__)
                or source_config["support_sha256"] != sha(base.pilot.__file__)
                or source_config["diagnostic_wrapper_sha256"] != sha(base.regular(source_config["diagnostic_wrapper"]))):
            raise ValueError("source continuation coordinator/support/diagnostic wrapper changed")
        all_questions = read(base.regular(baseline / "questions.json"))
        protected = base.protected_scores(baseline, all_questions, 195)
        selection = read(base.regular(args.selection))
        questions = base.selected_questions(selection, all_questions, protected)
        source_qmap = {q["question_id"]: q for q in read(base.regular(source / "questions.json"))}
        if any(source_qmap.get(q["question_id"]) != q for q in questions):
            raise ValueError("selected question differs from its actual failed source task")
        videos = sorted({q["video_id"] for q in questions})
        base.ensure_no_backend_attempts(baseline, questions)
        base.ensure_no_backend_attempts(source, questions)
        related = arguments["related_runs"]
        if str(baseline) in related or str(run) in related:
            raise ValueError("baseline/this run cannot also appear as a related continuation")
        additional = score_snapshot(related, all_questions, set(selection["question_ids"]), protected)
        checkpoints, recovery_videos = {}, {}
        registry = baseline.parents[1] / "recovery_claims/noevent_length"
        source_config_sha = sha(source / "configuration.json")
        for video in videos:
            states = terminal_video(source, video)
            checkpoint = base.checkpoint_receipt(source, repo, old, video)
            evidence = length_evidence(source, video, checkpoint)
            prior = source_config["checkpoints"][video]
            if checkpoint["completed_windows"] < prior["completed_windows"]:
                raise ValueError("source lost a previously committed window")
            for name, expected in prior["committed_files"].items():
                if name.startswith("windows/") and checkpoint["files"].get(name) != expected:
                    raise ValueError("source changed an inherited original window")
            path = registry / (digest(str(source))[:20] + "-" + video + ".json")
            claim = {"schema_version": 1, "source_run": str(source), "video_id": video, "owner_run": str(run),
                     "source_configuration_sha256": source_config_sha, "selection_sha256": arguments["selection_sha256"],
                     "new_visual_max_tokens": 16384, "driver_sha256": sha(__file__)}
            claim_video(path, claim)
            checkpoints[video] = checkpoint
            recovery_videos[video] = {"source_state_hashes": states, "evidence": evidence,
                                     "claim_path": str(path), "claim": claim,
                                     "original_completed_windows": prior["completed_windows"],
                                     "source_completed_windows": checkpoint["completed_windows"],
                                     "original_model": prior["parent_model"], "original_source": prior["source"]}
        config = copy.deepcopy(source_config)
        config.pop("audio_seed_recovery", None)
        config.update(selection=selection, selection_file_sha256=arguments["selection_sha256"],
                      questions_sha256=digest(questions), protected_scores=protected, checkpoints=checkpoints,
                      media={v: old["media"][v] for v in videos}, media_identity=base.verified_media(old, videos),
                      python=args.python, credential_file=arguments["credential_file"], workers=args.workers,
                      question_workers=args.question_workers, coordinator_sha256=sha(base.__file__),
                      support_sha256=sha(base.pilot.__file__),
                      comparison_scope="Separate, explicitly authorized length recovery: same model/media/schema/retrieval/judge; only visual max_tokens 8192 to 16384. Preserve original, main-continuation, and length-continuation window/first-score provenance.")
        config["profile"]["visual"]["max_tokens"] = 16384
        config["length_recovery"] = {"schema_version": 1, "source_seed_ancestry": seed_ancestry, "arguments": arguments, "driver_sha256": sha(__file__),
            "source_run": str(source), "source_configuration_sha256": source_config_sha, "related_runs": related,
            "source_prepared_sha256": sha(source / "prepared.json"),
            "additional_protected_scores": additional, "videos": recovery_videos,
            "budget_policy": "One new build task per source video; no automatic outer retries; preserve old exhausted attempts and first judgments.",
            "reason": "Exact final visual responses returned HTTP 200 with finish_reason length; raw usage is retained per video. Increase the total output/reasoning budget without altering input or safety settings."}
        freeze(run / "configuration.json", config)
        freeze(run / "questions.json", questions)
        for video in videos:
            freeze(run / "questions" / (video + ".json"), [q for q in questions if q["video_id"] == video])
            base.clone_checkpoint(run, video, checkpoints[video])
        freeze(run / "prepared.json", {"configuration_sha256": sha(run / "configuration.json"), "no_api_calls": True})
    verify(run, config)
    return run, config


def window_provenance(run, config, video):
    value = _BASE_PROVENANCE(run, config, video)
    source = config["length_recovery"]["videos"][video]
    original_end, main_end = source["original_completed_windows"], source["source_completed_windows"]
    lineage = {}
    ancestral = config["length_recovery"]["source_seed_ancestry"]["inherited_window_sources"]
    for i, (window, row) in enumerate(value["windows"].items()):
        if window in ancestral:
            if row["window_sha256"] != ancestral[window]["window_sha256"]:
                raise ValueError("frozen inherited audio-seed window changed")
            lineage[window] = ancestral[window]
            continue
        if i < original_end:
            origin, directory, model = "original", source["original_source"], source["original_model"]
        elif i < main_end:
            origin, directory, model = "source_continuation", config["checkpoints"][video]["source"], row["model"]
        else:
            origin, directory, model = "length_continuation", str(run), row["model"]
        lineage[window] = {"origin": origin, "source": directory, "model": model, "window_sha256": row["window_sha256"]}
    freeze(run / "lineage/windows" / (video + ".json"), {"memory_sha256": value["memory_sha256"], "windows": lineage})
    return value


def report(run, output=None):
    run = Path(run).absolute()
    config = read(base.regular(run / "configuration.json"))
    current = verify(run, config)
    result = _BASE_REPORT(run, output)
    all_rows = {q["question_id"]: q["score"] for q in result["questions"]}
    for qid, item in current.items():
        if qid in all_rows:
            raise ValueError("combined noevent report contains overlapping first scores")
        all_rows[qid] = item["score"]
    result["combined_baseline_and_this_run"] = result["combined"]
    result["combined"] = {"scored": len(all_rows), "total": config["total_questions"],
                          "coverage": len(all_rows) / config["total_questions"],
                          "mean": 100 * sum(row["normalized_score"] for row in all_rows.values()) / len(all_rows)}
    result["other_continuation_first_scores"] = current
    result["length_evidence"] = {video: value["evidence"] for video, value in config["length_recovery"]["videos"].items()}
    destination = Path(output).absolute() if output else run / "report"
    write(destination / "report.json", result)
    (destination / "report.md").write_text(
        "# noevent 长度恢复评测\n\n"
        f"合并已知首次有效评分 **{len(all_rows)}/{config['total_questions']}**，均分 **{result['combined']['mean']:.2f}**。"
        f"本批新增 **{result['continuation']['scored']}/{result['continuation']['selected']}**。\n\n"
        "本批依据原始响应中明确的输出截断，仅将视觉总输出预算从8192提高到16384；模型、媒体采样、窗口schema、检索与官方评分不变。"
        "保留原窗口、主续跑窗口及本批窗口来源，以及各次首分，不覆盖原实验报告。\n")
    return result


def install_hooks():
    # Only this separate driver process changes Python function bindings. No
    # source file, active coordinator or frozen core is edited or reloaded.
    base.verify = verify
    base.window_provenance = window_provenance
    base.report = report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("stage", choices=("prepare", "run", "report"))
    p.add_argument("--run", required=True)
    for name in ("source-run", "baseline-run", "core-repo", "selection", "credential-file"):
        p.add_argument("--" + name)
    p.add_argument("--related-run", action="append", default=[])
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--question-workers", type=int, default=2)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--output")
    args = p.parse_args(argv)
    install_hooks()
    if args.stage == "report":
        result = report(args.run, args.output)
    else:
        if any(getattr(args, key) is None for key in ("source_run", "baseline_run", "core_repo", "selection", "credential_file")):
            raise SystemExit("prepare/run require source, baseline, frozen core, selection and private credential paths")
        run, config = prepare(args)
        result = base.execute(run, config) if args.stage == "run" else {"prepared": str(run), "selected": len(config["selection"]["question_ids"])}
    print(json.dumps(result.get("combined", result), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
