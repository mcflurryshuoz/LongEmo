"""Run one separately selected noevent video after importing its successful probe.

The frozen continuation file is never edited.  Its prepare function runs once,
then the successful probe is imported, and execute acquires the original run
lock.  In-process guards add seed provenance checks to every original stage.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time

from experiments.zyf import noevent_continuation as continuation
from experiments.zyf import noevent_probe_seed as seed
from experiments.zyf import noevent_retry_probe as probe


COORDINATOR_SHA256 = "1e9776267db1f38330a3bca2c66fb031ce6ec3b925abb11531a5c31748052192"


def frozen_code():
    files = {"launcher": Path(__file__), "coordinator": Path(continuation.__file__),
             "seed_helper": Path(seed.__file__), "probe": Path(probe.__file__)}
    result = {name: {"path": str(path.resolve()), "sha256": probe.sha(seed.regular(path))}
              for name, path in files.items()}
    if result["coordinator"]["sha256"] != COORDINATOR_SHA256:
        raise ValueError("this launcher requires the exact frozen continuation implementation")
    return result


def exclusion_receipts(excluded_runs, parent_run, selected, video_id):
    result = []
    if not excluded_runs:
        raise ValueError("the separately running cohort must be explicitly excluded")
    for value in excluded_runs:
        run = Path(value).absolute()
        config_path = seed.regular(run / "configuration.json")
        config = seed.read(config_path)
        if config.get("condition") != "noevent" or config.get("parent_run") != str(Path(parent_run).absolute()):
            raise ValueError("excluded cohort is not a noevent continuation of the same parent")
        ids = config["selection"]["question_ids"]
        videos = list(config["checkpoints"])
        if set(selected) & set(ids) or video_id in videos:
            raise ValueError("seeded cohort overlaps another continuation by question or video")
        result.append({"run": str(run), "configuration_sha256": probe.sha(config_path),
                       "question_ids": ids, "video_ids": videos})
    return result


def verify_execution(run, config, expected_manifest_sha256):
    run = Path(run)
    path = run / "seed_execution.json"
    if probe.sha(seed.regular(path)) != expected_manifest_sha256:
        raise ValueError("frozen seed execution manifest changed")
    value = seed.read(path)
    if (value.get("schema_version") != 1 or value.get("execution_conditions") != ["noevent"] or
            value.get("configuration_sha256") != probe.sha(run / "configuration.json") or
            value.get("selected_question_ids") != config["selection"]["question_ids"] or
            value.get("parent_run") != config["parent_run"] or value.get("frozen_code") != frozen_code()):
        raise ValueError("seed execution configuration or implementation changed")
    video = value["video_id"]
    if set(config["checkpoints"]) != {video} or set(value["selected_question_ids"]) & set(config["protected_scores"]):
        raise ValueError("seeded selection differs from its unscored isolated video")
    for item in value["excluded_runs"]:
        if probe.sha(seed.regular(Path(item["run"]) / "configuration.json")) != item["configuration_sha256"]:
            raise ValueError("another continuation changed its frozen question selection")
        if set(value["selected_question_ids"]) & set(item["question_ids"]) or video in item["video_ids"]:
            raise ValueError("another continuation overlaps the seeded selection")
    if probe.sha(seed.regular(run / "probe_seeds" / video / "receipt.json")) != value["seed_receipt_sha256"]:
        raise ValueError("seed import receipt changed")
    receipt = seed.verify_seed_receipt(run, video, config["checkpoints"][video], allow_memory_growth=True)
    if not receipt or receipt["stage"] != "visual":
        raise ValueError("this launcher requires an imported successful visual window")
    return value, receipt


def exact_seed_provenance(run, config, execution, receipt, *, complete_memory=None):
    value = {"schema_version": 1, "video_id": execution["video_id"], "window_id": receipt["window_id"],
             "source": "validated_one_request_probe", "request_hash": receipt["request_hash"],
             "model_configuration": receipt["model_configuration"], "no_probe_replayed": True,
             "seed_execution_sha256": probe.sha(Path(run) / "seed_execution.json"),
             "seed_receipt_sha256": execution["seed_receipt_sha256"],
             "probe_source_hashes": receipt["probe_source_hashes"],
             "window_sha256": receipt["target_files_after"]["windows/" + receipt["window_id"] + ".json"],
             "parent_window_count": config["checkpoints"][execution["video_id"]]["completed_windows"]}
    if complete_memory is not None:
        value["completed_memory_sha256"] = complete_memory["memory_sha256"]
        value["other_windows_provenance"] = {key: dict(item) for key, item in complete_memory["windows"].items()}
        # Supply an explicit correction alongside the unchanged frozen report.
        value["other_windows_provenance"][receipt["window_id"]] = {
            "window_sha256": value["window_sha256"], "source": value["source"],
            "model": receipt["model_configuration"], "request_hash": receipt["request_hash"]}
    return value


@contextmanager
def installed_guards(run, config, expected_sha):
    original_verify, original_provenance = continuation.verify, continuation.window_provenance

    def verify(target, actual):
        if Path(target).absolute() != Path(run).absolute():
            raise ValueError("the launcher cannot verify or dispatch another run")
        original_verify(target, actual)
        verify_execution(target, actual, expected_sha)

    def provenance(target, actual, video):
        result = original_provenance(target, actual, video)
        execution, receipt = verify_execution(target, actual, expected_sha)
        if video != execution["video_id"]:
            raise ValueError("unexpected seeded video")
        exact = exact_seed_provenance(target, actual, execution, receipt, complete_memory=result)
        continuation.freeze(Path(target) / "noevent/seed_provenance" / (video + ".json"), exact)
        return result

    continuation.verify, continuation.window_provenance = verify, provenance
    try:
        verify(run, config)
        yield
    finally:
        continuation.verify, continuation.window_provenance = original_verify, original_provenance


def launch(args):
    code = frozen_code()
    run = Path(args.run).absolute()
    parent, repo = Path(args.parent_run).absolute(), Path(args.core_repo).absolute()
    if (run == parent or parent in run.parents or run in parent.parents or
            run == repo or repo in run.parents or any(path.is_symlink() for path in (run, *run.parents))):
        raise ValueError("seed launcher requires a separate ordinary run directory")
    if (args.visual_max_tokens != 8192 or args.workers != 1 or args.question_workers != 2 or
            args.protected_count != 195):
        raise ValueError("seeded continuation keeps 8192 tokens, one video, two questions and the 195-score parent")
    selection = seed.read(args.selection)
    selected = selection.get("question_ids", [])
    if len(selected) != 3 or len(set(selected)) != 3 or selection.get("condition") != "noevent":
        raise ValueError("this protocol requires the fixed three-question noevent selection")
    exclusions = exclusion_receipts(args.exclude_run, args.parent_run, selected, args.video_id)
    if run.exists():
        raise FileExistsError("seeded launcher output already exists; audit instead of launching twice")
    run.mkdir(parents=True, mode=0o700)
    seed.exclusive_json(run / "seed_launcher_claim.json", {"schema_version": 1, "pid": os.getpid(),
        "started_unix": time.time(), "video_id": args.video_id, "selected_question_ids": selected,
        "frozen_code": code, "parameters": vars(args)})
    prepared_run, config = continuation.prepare(args)
    if (Path(prepared_run) != run or set(config["checkpoints"]) != {args.video_id} or
            config["selection"]["question_ids"] != selected or
            any(q["video_id"] != args.video_id for q in seed.read(run / "questions.json"))):
        raise ValueError("prepared cohort differs from the single selected video")
    receipt = seed.apply_seed(args.parent_run, run, args.video_id, args.probe_dir, args.core_repo)
    if receipt["stage"] != "visual":
        raise ValueError("seeded launcher requires a completed visual window")
    value = {"schema_version": 1, "execution_conditions": ["noevent"], "video_id": args.video_id,
             "selected_question_ids": selected, "parent_run": config["parent_run"],
             "configuration_sha256": probe.sha(run / "configuration.json"), "frozen_code": code,
             "seed_receipt_sha256": probe.sha(run / "probe_seeds" / args.video_id / "receipt.json"),
             "parameters": vars(args), "excluded_runs": exclusions,
             "protocol": "prepare once; import proven window; directly execute with per-stage seed guards"}
    seed.exclusive_json(run / "seed_execution.json", value)
    expected_sha = probe.sha(run / "seed_execution.json")
    seed.exclusive_json(run / "seed_execution.sha256.json", {"sha256": expected_sha})
    continuation.freeze(run / "seed_provenance.json", exact_seed_provenance(run, config, value, receipt))
    with installed_guards(run, config, expected_sha):
        # Never call continuation.main: it prepares again and would correctly
        # reject the now-grown clone.  execute owns the original run_lock.
        return continuation.execute(run, config)


def report(run, output=None):
    run = Path(run).absolute()
    config = seed.read(run / "configuration.json")
    expected = seed.read(run / "seed_execution.sha256.json")["sha256"]
    with installed_guards(run, config, expected):
        return continuation.report(run, output)


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("stage", choices=("run", "report"))
    p.add_argument("--run", required=True)
    for name in ("parent-run", "core-repo", "selection", "credential-file", "probe-dir"):
        p.add_argument("--" + name)
    p.add_argument("--exclude-run", action="append", default=[])
    p.add_argument("--video-id", default="G2_V000031")
    p.add_argument("--visual-max-tokens", type=int, default=8192)
    p.add_argument("--protected-count", type=int, default=195)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--question-workers", type=int, default=2)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--output")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.stage == "report":
            result = report(args.run, args.output)
        else:
            if any(getattr(args, key) is None for key in ("parent_run", "core_repo", "selection", "credential_file", "probe_dir")):
                raise ValueError("all continuation and probe inputs are required")
            result = launch(args)
    except Exception as exc:
        print(json.dumps({"status": "needs_audit", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps(result.get("combined", result), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
