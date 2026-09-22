"""Read-only full-suite audit and Chinese report; standalone standard library CLI.

Collect with --run DIR --output NEW_DIR, or render an exported snapshot with
--source report.json --output DIR. Never imports worker code or writes the run.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket

CONDITIONS = ("noevent", "base", "method")
TASKS = (("emotional intensity comparison", "情感强度比较"),
         ("emotion trajectory", "情感轨迹"), ("emotional reasoning", "情感推理"))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


class Reader:
    """Read each file once so provenance hashes identify exactly the parsed bytes."""
    def __init__(self):
        self.cache = {}

    def raw(self, path):
        path = Path(path)
        if path not in self.cache:
            self.cache[path] = path.read_bytes()
        return self.cache[path]

    def sha(self, path):
        return hashlib.sha256(self.raw(path)).hexdigest()

    def json(self, path, default=None):
        return json.loads(self.raw(path)) if Path(path).exists() else default

    def rows(self, path, active=False):
        if not Path(path).exists():
            return []
        raw = self.raw(path)
        if active and raw and not raw.endswith(b"\n"):
            raw = raw.rsplit(b"\n", 1)[0] if b"\n" in raw else b""
        return [json.loads(line) for line in raw.splitlines() if line.strip()]

    def artifact(self, ref):
        require(ref and self.sha(ref["path"]) == ref["sha256"], "inherited artifact hash mismatch")
        return Path(ref["path"])


def source_series(value):
    # Identical grouping rule to experiments/zyf/series_scores.py.
    match = re.fullmatch(r"(.+?)\s+[sS]\d+[eE]\d+", value.strip())
    return {"jiayouernv": "Home with Kids"}.get(match[1], match[1]) if match else "unclassified"


def source_hash(reader, repo):
    repo = Path(repo)
    files = list((repo / "methods/longemo").glob("*.py")) + list((repo / "evaluation").rglob("*.py"))
    require(bool(files), "method source files absent")
    return digest({str(p.relative_to(repo)): reader.sha(p) for p in sorted(files) if "emollm" not in p.parts})


def classify_failure(row):
    if row.get("service_error_code") in ("content_filter", "content_policy_violation"):
        return "content_filter"
    if row.get("http_status") == 428:
        return "http_428_unclassified"
    if row.get("error_type") == "RuntimeError" and row.get("failure_stage") == "generate":
        return "generate_runtime_error_unknown"
    if row.get("failure_stage") == "validate" and row.get("error_type") in ("ValueError", "JSONDecodeError"):
        return "output_validation_failed"
    return "other"


def failure_evidence(reader, paths, completed=(), active=False):
    candidates = []
    for path in paths:
        for row in reader.rows(path, active=active):
            if row.get("status") == "error" and row.get("purpose", "").split(":")[-1] not in completed:
                candidates.append((row, path))
    if not candidates:
        return {"category": "other", "evidence": "no structured terminal error record"}
    row, path = max(candidates, key=lambda x: x[0].get("time_unix", 0))
    fields = ("purpose", "attempt", "error_type", "http_status", "service_error_code", "failure_stage", "time_unix", "request_hash")
    return {**{k: row[k] for k in fields if k in row}, "category": classify_failure(row),
            "source": str(path), "source_sha256": reader.sha(path)}


def embedding_failure(reader, run, branch, task_path, task):
    """A stopped index-construction traceback is evidence of an embedding failure.

    Shared ledger timestamps are only corroboration, never asserted as an exact
    video/request match when the provider did not log a video identifier.
    """
    output = task_path.parent / "output.log"
    if task.get("status") not in ("error", "needs_audit") or not output.exists():
        return None
    text = reader.raw(output).decode(errors="replace")
    if not (re.search(r"(?:WindowIndex|EventIndex)", text) and
            re.search(r"encoder\.encode\(|embeddings\.py", text)):
        return None
    status = re.findall(r"HTTP\s+(\d{3})", text)
    result = {"category": "embedding_failed", "source": str(output), "source_sha256": reader.sha(output),
              "gpt_answer_started": False, "stage": "document_index_construction"}
    if status:
        result["http_status"] = int(status[-1])
    ledger = run / branch / "embeddings/api/calls.jsonl"
    candidates = [row for row in reader.rows(ledger, active=True)
                  if row.get("purpose") == "embedding_documents" and row.get("status") == "error"
                  and task.get("started_unix", float("inf")) <= row.get("time_unix", 0) <= task.get("finished_unix", 0)]
    result["concurrent_ledger_candidates"] = [{k: r[k] for k in ("request_hash", "http_status", "time_unix") if k in r}
                                             for r in candidates]
    if candidates:
        result.update(ledger=str(ledger), ledger_sha256=reader.sha(ledger))
    return result


def metrics(values, total):
    return {"scored": len(values), "total": total, "coverage": len(values) / total if total else None,
            "mean_scored": sum(values) / len(values) if values else None}


def grouped_metrics(accepted, questions):
    return {c: metrics([accepted[c][q["question_id"]]["normalized_score_100"] for q in questions
                        if q["question_id"] in accepted[c]], len(questions)) for c in CONDITIONS}


def paired_metrics(accepted, questions, conditions):
    ids = sorted(set.intersection(*(set(accepted[c]) for c in conditions)) & {q["question_id"] for q in questions})
    means = {c: sum(accepted[c][qid]["normalized_score_100"] for qid in ids) / len(ids) if ids else None for c in conditions}
    result = {"count": len(ids), "question_ids": ids, "means": means}
    for left, right in (("method", "base"), ("base", "noevent"), ("method", "noevent")):
        if left in means and right in means:
            result[left + "_minus_" + right] = means[left] - means[right] if ids else None
    return result


def validate_score(reader, run, index, condition, path, q):
    value = reader.json(path); row = value["score"]; qid, video = q["question_id"], q["video_id"]
    require(path.stem == qid and row.get("question_id") == qid and row.get("status") == "ok", "invalid accepted first score")
    maximum = max(int(x) for x in q["rubric"]["scores"])
    require(all(row.get(k) == q[k] for k in ("video_id", "type")), "accepted question metadata differs")
    score, normalized = row.get("score"), row.get("normalized_score")
    require(type(score) in (int, float) and type(normalized) in (int, float) and math.isfinite(score)
            and math.isfinite(normalized) and maximum > 0 and row.get("max_score") == maximum
            and 0 <= score <= maximum and math.isclose(normalized, score / maximum), "invalid rubric-normalized score")
    require(digest(row.get("prediction")) == value.get("prediction_sha256"), "accepted prediction hash mismatch")
    old = index.get("conditions", {}).get(condition, {}).get("questions", {}).get(qid, {})
    parent_score = old.get("accepted")
    source = Path(value["source"])
    if parent_score:
        require(value == parent_score["envelope"], "inherited first score changed")
        source_read = reader.artifact(parent_score["score_artifact"])
        require(str(source) == parent_score["score_artifact"]["source"], "parent score source changed")
        origin = "parent_pilot"
    else:
        require(source.is_relative_to(run / "scores" / condition / video), "unexpected first score source")
        source_read, origin = source, "full_suite"
    raw_matches = [r for r in reader.rows(source_read) if r.get("question_id") == qid]
    require(len(raw_matches) == 1 and raw_matches[0] == row, "official raw score does not uniquely match accepted score")
    parent_prediction = old.get("prediction")
    new_predictions = run / "answers" / condition / video / "predictions.jsonl"
    new_matches = [r for r in reader.rows(new_predictions) if r.get("question_id") == qid and r.get("status") == "ok"]
    require(len(new_matches) <= 1, "duplicate successful full-suite predictions")
    if parent_prediction:
        require(not new_matches, "successful parent prediction was repeated")
        pred_path = reader.artifact(parent_prediction["artifact"])
        prediction = parent_prediction["row"]
        require(sum(r == prediction for r in reader.rows(pred_path)) == 1, "parent prediction artifact differs")
        answer_run = Path(index["snapshot_dir"])
    else:
        require(len(new_matches) == 1, "first score lacks successful prediction")
        prediction, pred_path, answer_run = new_matches[0], new_predictions, run
    require(prediction.get("prediction") == row["prediction"] and all(prediction.get(k) == q[k]
            for k in ("question_id", "video_id", "type")), "score prediction differs from frozen question")
    branch = "noevent" if condition == "noevent" else "event"
    memory = answer_run / branch / "videos" / video / "memory" / video / "memory.json"
    frozen_memory = reader.json(answer_run / branch / "frozen_memories" / (video + ".json"))
    require(frozen_memory and reader.sha(memory) == frozen_memory["memory_sha256"], "answer memory differs from its frozen receipt")
    plan = answer_run / branch / "plans" / (qid + ".json")
    frozen_plans = reader.json(answer_run / branch / "plans/frozen" / (video + ".json"), {})
    require(qid in frozen_plans and reader.sha(plan) == frozen_plans[qid], "answer plan differs from its frozen receipt")
    answer_folder = answer_run / "answers" / condition / video
    embedding = reader.json(answer_folder / "embedding_indexes.json")[video]["signature"]
    trace = answer_folder / "traces" / (qid + ".json")
    routing = reader.json(trace).get("routing")
    if condition in ("base", "method"):
        require(routing and routing.get("effective_retrieval") == ("graph" if condition == "base" else "progressive")
                and routing.get("progressive_routing") == "none" and routing.get("hybrid_graph_tasks") == [], "non-pure event retrieval route")
    return {"status": "scored", "score": score, "max_score": maximum, "normalized_score_100": 100 * normalized,
            "origin": origin, "source": str(source), "source_snapshot": str(source_read), "source_sha256": reader.sha(source_read),
            "accepted_sha256": reader.sha(path), "prediction_sha256": value["prediction_sha256"],
            "prediction_artifact_sha256": reader.sha(pred_path), "memory_sha256": reader.sha(memory),
            "plan_sha256": reader.sha(plan), "embedding_signature": embedding, "trace_sha256": reader.sha(trace), "routing": routing}


def missing_status(reader, run, index, condition, q, video_info, tasks):
    qid, video = q["question_id"], q["video_id"]
    branch = "noevent" if condition == "noevent" else "event"
    old = index.get("conditions", {}).get(condition, {}).get("questions", {}).get(qid, {})
    inherited = index.get("branches", {}).get(branch, {}).get("videos", {}).get(video, {})
    result = {"status": "not_started", "normalized_score_100": None}
    if inherited.get("state") == "blocked":
        return {**result, "status": "parent_build_blocked", "reason": "parent failed/incomplete build; attempts preserved"}
    pipeline_status = video_info["status"]
    if pipeline_status in ("build_failed", "media_missing", "media_integrity_failed", "needs_audit"):
        return {**result, "status": pipeline_status, "failure": video_info.get("final_failure")}
    if not video_info["memory_complete"]:
        return {**result, "status": "perception_running" if pipeline_status == "running" else "waiting_for_media_or_phase"}
    if old.get("score_state") == "blocked":
        return {**result, "status": "parent_score_blocked", "reason": "parent first judgment failed or outcome unknown"}
    pred_path = run / "answers" / condition / video / "predictions.jsonl"
    predictions = [r for r in reader.rows(pred_path) if r.get("question_id") == qid]
    prediction = old.get("prediction") or next((r for r in predictions if r.get("status") == "ok"), None)
    if prediction:
        score_task = tasks.get("score/" + condition + "-" + video, {})
        state = score_task.get("status")
        return {**result, "status": {"running": "scoring", "error": "score_failed", "needs_audit": "score_needs_audit",
                "ok": "awaiting_score_acceptance"}.get(state, "waiting_for_first_score")}
    if old.get("answer_state") == "blocked":
        return {**result, "status": "parent_answer_blocked", "reason": "parent answer attempt failed or outcome unknown"}
    if inherited.get("plans", {}).get(qid, {}).get("state") == "blocked":
        return {**result, "status": "parent_plan_blocked"}
    if inherited.get("embedding_guard", {}).get("state") == "needs_audit":
        return {**result, "status": "parent_embedding_needs_audit"}
    plan_path = run / branch / "plans" / (qid + ".json")
    if not plan_path.exists():
        state = tasks.get("plan/" + branch + "-" + video, {}).get("status")
        return {**result, "status": {"running": "planning", "error": "plan_failed", "needs_audit": "plan_needs_audit",
                                    "ok": "plan_missing"}.get(state, "waiting_for_plan")}
    task_key = "answer/" + condition + "-" + video
    task = tasks.get(task_key, {})
    embedding = embedding_failure(reader, run, branch, run / "tasks" / task_key / "task.json", task)
    if embedding and not predictions:
        return {**result, "status": "embedding_failed", "failure": embedding}
    state = task.get("status")
    return {**result, "status": {"running": "answering", "error": "answer_failed", "needs_audit": "answer_needs_audit",
                                "ok": "answer_missing"}.get(state, "waiting_for_answer")}


def snapshot(run):
    run = Path(run).resolve(); reader = Reader(); started = datetime.now(timezone.utc).isoformat()
    config = reader.json(run / "configuration.json"); questions = reader.json(run / "questions.json")
    require(digest(questions) == config["questions_sha256"], "frozen full question hash differs")
    qmap = {q["question_id"]: q for q in questions}
    require(len(qmap) == len(questions) == config["question_count"] and len({q["video_id"] for q in questions}) == config["video_count"], "full cohort count differs")
    require(len(questions) == 558 and config["video_count"] == 141, "report requires the frozen 558-question/141-video full cohort")
    integrity = {"configuration_sha256": reader.sha(run / "configuration.json"), "questions_sha256": reader.sha(run / "questions.json"),
                 "frozen_source_hashes": config["source_hashes"], "source_hashes_verified": {}, "first_scores_verified": True}
    for branch, folder in config["repos"].items():
        actual = source_hash(reader, folder)
        require(actual == config["source_hashes"][branch], "deployed method/evaluator source differs: " + branch)
        integrity["source_hashes_verified"][branch] = actual
    suite_folder = Path(config["repos"]["noevent"]) / "experiments/zyf"
    require(reader.sha(suite_folder / "full_suite.py") == config["suite_sha256"], "frozen coordinator source differs")
    for name, expected in config["suite_support_hashes"].items():
        require(reader.sha(suite_folder / name) == expected, "frozen support source differs: " + name)
    integrity["suite_sha256"] = config["suite_sha256"]
    index_path = run / "inheritance/parent_index.json"
    index = reader.json(index_path, {})
    if index:
        require(index["full_questions_sha256"] == digest(questions) and index["parent_configuration_sha256"] == config["parent_configuration_sha256"], "parent index cohort/config differs")
        require(reader.sha(Path(index["snapshot_dir"]) / "configuration.json") == config["parent_configuration_sha256"], "parent configuration snapshot differs")
        marker = reader.json(run / "inheritance/materialized.json", {})
        require(marker.get("parent_index_sha256") == reader.sha(index_path), "materialized inheritance receipt differs")
        integrity["parent_index_sha256"] = reader.sha(index_path)
    # Take accepted-path inventories once; later arrivals belong to the next report.
    accepted_paths = {c: sorted((run / "accepted" / c).glob("*.json")) for c in CONDITIONS}
    tasks = {str(p.parent.relative_to(run / "tasks")): reader.json(p) for p in run.glob("tasks/*/*/task.json")}
    branches = {}
    for branch in ("noevent", "event"):
        entries = []
        for video in sorted(config["media"]):
            pipeline = reader.json(run / "pipelines" / branch / (video + ".json"), {})
            folder = run / branch / "videos" / video / "memory" / video
            memory = reader.json(folder / "memory.json", {})
            inherited = index.get("branches", {}).get(branch, {}).get("videos", {}).get(video, {})
            state = pipeline.get("status", "pending")
            item = {"video_id": video, "status": state, "memory_complete": bool(memory.get("complete")),
                    "completed_windows": len(memory.get("completed_windows", [])),
                    "total_windows": math.ceil(memory["duration"] / config["window_seconds"]) if memory else None,
                    "inherited_state": inherited.get("state", "unstarted"), "final_failure": None}
            if state == "build_failed":
                item["final_failure"] = failure_evidence(reader, [folder / "calls.jsonl", folder / "audio/calls.jsonl"], memory.get("completed_windows", []))
            entries.append(item)
        branches[branch] = {"counts": dict(Counter(v["status"] for v in entries)), "memory_complete": sum(v["memory_complete"] for v in entries),
                            "completed_windows": sum(v["completed_windows"] for v in entries), "videos": entries}
    accepted = {c: {} for c in CONDITIONS}
    for c in CONDITIONS:
        for path in accepted_paths[c]:
            require(path.stem in qmap, "accepted question is outside full cohort")
            accepted[c][path.stem] = validate_score(reader, run, index, c, path, qmap[path.stem])
    for qid in set(accepted["base"]) & set(accepted["method"]):
        require(all(accepted["base"][qid][key] == accepted["method"][qid][key] for key in ("memory_sha256", "plan_sha256", "embedding_signature")), "paired event resources differ")
    integrity["paired_event_resources_and_pure_routing_verified"] = True
    conditions = grouped_metrics(accepted, questions)
    series = sorted({source_series(q.get("source", {}).get("from", "")) for q in questions})
    groups = {"by_type": {t: [q for q in questions if q["type"] == t] for t, _ in TASKS},
              "by_series": {s: [q for q in questions if source_series(q.get("source", {}).get("from", "")) == s] for s in series}}
    paired = {}
    for name, cs in (("all_three", CONDITIONS), ("base_method", ("base", "method"))):
        paired[name] = paired_metrics(accepted, questions, cs)
        for group_name, entries in groups.items():
            paired[name][group_name] = {g: paired_metrics(accepted, qs, cs) for g, qs in entries.items()}
    for c in CONDITIONS:
        for group_name, entries in groups.items():
            conditions[c][group_name] = {g: grouped_metrics(accepted, qs)[c] for g, qs in entries.items()}
    video_lookup = {b: {v["video_id"]: v for v in branches[b]["videos"]} for b in branches}
    output_questions = []
    for q in questions:
        item = {k: q[k] for k in ("question_id", "video_id", "type")}
        item["series"] = source_series(q.get("source", {}).get("from", "")); item["conditions"] = {}
        for c in CONDITIONS:
            b = "noevent" if c == "noevent" else "event"
            item["conditions"][c] = accepted[c].get(q["question_id"]) or missing_status(reader, run, index, c, q, video_lookup[b][q["video_id"]], tasks)
        output_questions.append(item)
    process = reader.json(run / "process.json", {})
    stat = Path("/proc") / str(process.get("pid", -1)) / "stat"
    try:
        proc = stat.read_text().rsplit(")", 1)[1].split()
        alive = process.get("host") == socket.gethostname() and str(process.get("start_ticks")) == proc[19] and proc[0] != "Z"
    except FileNotFoundError:
        alive = False
    return {"schema_version": 1, "as_of": datetime.now(timezone.utc).isoformat(), "snapshot_started": started,
            "run": run.name, "coordinator": {"pid": process.get("pid"), "alive": alive},
            "configuration": {k: config[k] for k in ("question_count", "video_count", "perception_model", "model", "audio_model", "embedding_model", "stage_tries", "execution") if k in config},
            "parent_status": reader.json(run / "parent_status.json", {}), "conditions": conditions, "paired": paired,
            "branches": branches, "questions": output_questions, "integrity": integrity,
            "missing_counts": {c: dict(Counter(q["conditions"][c]["status"] for q in output_questions if q["conditions"][c]["status"] != "scored")) for c in CONDITIONS},
            "notes": ["每题首次有效官方评分归一化为0–100后等权平均；未评分不计零分。",
                      "各条件已评分题集不同，不直接比较总体均分；差值仅对相同题号交集计算。早期小样本不能代表全集效果。",
                      "分剧仅依据source.from的明确剧名和集号；URL来源保留unclassified，不猜测。",
                      "进度是持续运行期间的只读快照，采集起止时间分别记录；新增评分留待下一次快照。",
                      "HTTP428与生成RuntimeError的原因未明；仅明确service_error_code才归内容过滤。"]}


def number(value):
    return "—" if value is None else f"{value:.2f}"


def render(data):
    metric = lambda r: f"{number(r['mean_scored'])}（{r['scored']}/{r['total']}）"
    as_of = datetime.fromisoformat(data["as_of"]).astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M CST")
    lines = ["# 三层消融全集评测进度", "", f"快照：{as_of}；全集558题／141视频。按 noevent → event（base / method共图）执行。", "",
             "**均分仅覆盖已评分题，不同题集不能直接比较；当前同题结果仍是早期小样本。**", "",
             "| 条件 | 已评分／总题数 | 已评分均分／100 |", "|---|---:|---:|"]
    for c in CONDITIONS:
        row = data["conditions"][c]; lines.append(f"| {c} | {row['scored']}/{row['total']} | {number(row['mean_scored'])} |")
    origins = {c: Counter(q["conditions"][c].get("origin") for q in data["questions"] if q["conditions"][c]["status"] == "scored") for c in CONDITIONS}
    lines += ["", "首分来源（试跑继承＋全集新增）：" + "；".join(
        f"{c} {origins[c]['parent_pilot']}＋{origins[c]['full_suite']}" for c in CONDITIONS) + "。"]
    for name, title in (("base_method", "base与method同题"), ("all_three", "三路共同题")):
        p = data["paired"][name]
        means = "、".join(f"{c} {number(p['means'][c])}" for c in p["means"])
        lines += ["", f"{title} **{p['count']}题**：{means}；method − base **{number(p['method_minus_base'])}**。"]
    for group, title, names in (("by_type", "分题型", [t for t, _ in TASKS]),
                                ("by_series", "分剧", sorted(data["conditions"]["noevent"]["by_series"]))):
        labels = dict(TASKS) if group == "by_type" else {"unclassified": "来源未分类"}
        lines += ["", title + "（单元格为均分〔已评分／该组总题数〕）：", "", "| 类别 | noevent | base | method |", "|---|---:|---:|---:|"]
        for name in names:
            lines.append("| " + labels.get(name, name) + " | " + " | ".join(metric(data["conditions"][c][group][name]) for c in CONDITIONS) + " |")
    lines += ["", "| 流水阶段 | 完成 | 运行 | 本轮构建失败 | 父构建阻断 | 部分完成 | 其他终态 | 待启动 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for b in ("noevent", "event"):
        counts = data["branches"][b]["counts"]
        known = ("complete", "running", "build_failed", "blocked_parent_build", "partial", "pending")
        other = sum(v for k, v in counts.items() if k not in known)
        vals = [counts.get(k, 0) for k in known[:-1]] + [other, counts.get("pending", 0)]
        lines.append("| " + b + " | " + " | ".join(map(str, vals)) + " |")
    failures = Counter(v["final_failure"]["category"] for b in data["branches"].values() for v in b["videos"] if v.get("final_failure"))
    if failures:
        labels = {"content_filter": "明确内容过滤", "http_428_unclassified": "HTTP428原因未明", "generate_runtime_error_unknown": "生成RuntimeError原因未明", "output_validation_failed": "输出结构校验失败", "other": "其他已停止失败"}
        lines += ["", "本轮已停止构建失败：" + "；".join(f"{labels.get(k,k)} {v}视频" for k, v in sorted(failures.items())) + "。"]
    if any(q["conditions"][c]["status"] == "embedding_failed" for q in data["questions"] for c in CONDITIONS):
        lines += ["", "部分题在Embedding文档索引阶段失败，尚未进入GPT回答；不归为GPT内容过滤。"]
    lines += ["", "缺失不计零分。父pilot首分保留，运行中的格式尝试不计终态失败。分剧仅采用明确来源，URL保留未分类。",
              "", "[558题逐题状态、首分来源、分组同题比较与校验SHA](report.json)。", "", f"运行：`{data['run']}`。", ""]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", type=Path); group.add_argument("--source", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv); output = args.output.resolve()
    if args.run:
        run = args.run.resolve()
        require(not output.is_relative_to(run) and not run.is_relative_to(output), "output must be outside the evaluated run")
        data = snapshot(run)
    else:
        data = json.loads(args.source.read_text())
    output.mkdir(parents=True, exist_ok=True)
    if args.run:
        raw = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        temp = output / ("report.json." + str(os.getpid()) + ".tmp"); temp.write_text(raw); temp.replace(output / "report.json")
    elif args.source.resolve() != output / "report.json":
        (output / "report.json").write_bytes(args.source.read_bytes())
    (output / "report.md").write_text(render(data))
    print(json.dumps({"as_of": data["as_of"], "conditions": {c: {k: data['conditions'][c][k] for k in ('scored','total','mean_scored')} for c in CONDITIONS},
                      "paired": {k: {n:v for n,v in x.items() if n not in ('question_ids','by_type','by_series')} for k,x in data['paired'].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
