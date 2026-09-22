"""Render an audited three-condition pilot snapshot as concise Chinese Markdown.

Offline only: reads one report.json and writes its Markdown companion. Scores,
task states and failure classifications must come from the audited collector;
this formatter never reads workers, ledgers, credentials, or network services.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path


CONDITIONS = ("noevent", "base", "method")
TASKS = (("emotional intensity comparison", "情感强度比较"),
         ("emotion trajectory", "情感轨迹"),
         ("emotional reasoning", "情感推理"))
FAILURES = {
    "content_filter": "明确内容过滤",
    "http_428_unclassified": "HTTP 428，原因未明",
    "generate_runtime_error_unknown": "生成阶段 RuntimeError，原因未明",
    "other": "其他终态失败",
}


def number(value):
    if value is None:
        return "—"
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("score must be a finite number or null")
    return f"{value:.2f}"


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def metric(value):
    count, total = value["scored"], value["total"]
    if (type(count) is not int or type(total) is not int or not 0 <= count <= total or
            (count == 0 and value["mean_scored"] is not None)):
        raise ValueError("invalid scored coverage or zero-coverage mean")
    mean = number(value["mean_scored"])
    if count and (value["mean_scored"] is None or not 0 <= value["mean_scored"] <= 100):
        raise ValueError("nonempty score mean must be on the 0–100 scale")
    return mean, f"{count}/{total}"


def failure_rows(branches):
    """Only terminal failed videos can contribute to this table."""
    groups = defaultdict(list)
    for branch in ("event", "noevent"):
        for video in branches[branch]["videos"]:
            if video["status"] != "failed":
                continue
            failure = video.get("final_failure") or {}
            category = failure.get("category", "other")
            label = FAILURES.get(category, f"未分类终态失败（{category}）")
            stage = failure.get("failure_stage") or "阶段未标注"
            details = []
            if failure.get("http_status") is not None and category != "http_428_unclassified":
                details.append(f"HTTP {failure['http_status']}")
            if failure.get("service_error_code"):
                details.append(str(failure["service_error_code"]))
            if failure.get("error_type") and category in ("other", "generate_runtime_error_unknown"):
                details.append(str(failure["error_type"]))
            detail = "；".join(dict.fromkeys(details)) or "—"
            groups[(branch, label, stage, detail)].append(video["video_id"])
    return ["| " + " | ".join(cell(x) for x in (*key, "、".join(sorted(videos)))) + " |"
            for key, videos in sorted(groups.items())]


def render_report(data, *, json_link="report.json"):
    config = data["configuration"]
    question_count, video_count = config["question_count"], config["video_count"]
    gate = data.get("gate") or {}
    workers = data.get("execution", {}).get("build_workers", config.get("build_workers"))
    concurrency = f"构建并发上限 **{workers}**（两种表示合计）" if workers is not None else "后续任务并发执行"
    gate_text = (f"完整视频验收 **已通过**（{cell(gate['video_id'])}），已放行后续队列；{concurrency}。"
                 if gate.get("passed") else "完整视频验收 **尚未通过**，其余视频队列尚未放行。")
    lines = ["# 三层消融实验进度", "", gate_text, "",
             f"本次 pilot：**{question_count} 题、{video_count} 个视频**。以下均分仅覆盖已评分题，尚不能作为总体效果结论。", "",
             "| 条件 | 已评分 / 总题数 | 已评分均分 /100 |", "|---|---:|---:|"]
    for condition in CONDITIONS:
        row = data["conditions"][condition]
        if row["total"] != question_count:
            raise ValueError("condition total differs from the frozen pilot")
        mean, coverage = metric(row)
        lines.append(f"| {condition} | {coverage} | {mean} |")
    lines += ["", "| 题型 | noevent | base | method |", "|---|---:|---:|---:|"]
    for task, label in TASKS:
        values = [metric(data["conditions"][condition]["by_type"][task]) for condition in CONDITIONS]
        lines.append("| " + label + " | " + " | ".join(f"{mean}（{coverage}）" for mean, coverage in values) + " |")
    lines += ["", "题型单元格为“已评分均分（已评分 / 该类总题数）”；未评分显示 —，不计零分。", ""]
    paired = data["paired"]
    if paired["count"]:
        means = "、".join(f"{condition} {number(paired['means'][condition])}" for condition in CONDITIONS)
        lines += [f"三路共同已评分 **{paired['count']} 题**：{means}；"
                  f"method − base **{number(paired['method_minus_base'])}**，"
                  f"base − noevent **{number(paired['base_minus_noevent'])}**。同题比较也仅代表这一小样本。", ""]
    else:
        lines += ["三路尚无共同已评分题，暂不计算同题差值。", ""]
    lines += ["| 感知表示 | 完成视频 | 运行中 | 失败 | 待启动 | 成功窗口 |", "|---|---:|---:|---:|---:|---:|"]
    for branch, label in (("event", "事件图（base / method 共用）"), ("noevent", "时间窗口（noevent）")):
        row = data["branches"][branch]
        counts = [row[key] for key in ("complete", "running", "failed", "pending")]
        if any(type(x) is not int or x < 0 for x in counts) or sum(counts) != video_count:
            raise ValueError("build-state counts must partition the frozen videos")
        lines.append("| " + label + " | " + " | ".join(str(x) for x in (*counts, row["completed_windows"])) + " |")
    lines += ["", *[cell(note) for note in data.get("notes", []) if "阶段屏障" in note]]
    failures = failure_rows(data["branches"])
    if failures:
        lines += ["", "已停止的构建失败：", "",
                  "| 表示 | 证据分类 | 阶段 | 返回信息 | 视频 |", "|---|---|---|---|---|", *failures,
                  "", "运行中的格式修复不计为失败视频；HTTP 428 与未知 RuntimeError 不推断为内容过滤。generate 阶段包含响应解析，现有记录不足以区分缺少消息、输出截断或非文本返回等原因。"]
    lines += ["", "每题保留首次有效官方评分。base 与 method 使用同一冻结事件图和 plans；noevent 使用独立窗口记忆。",
              f"[逐题首分、同题集合与来源校验]({json_link})。", "",
              f"运行：`{cell(data['run'])}`；核验时间：{cell(data['as_of'])}。", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Defaults to report.md beside the input JSON")
    args = parser.parse_args(argv)
    output = args.output or args.source.with_name("report.md")
    if output.resolve() == args.source.resolve():
        raise ValueError("output must not replace the evidence JSON")
    data = json.loads(args.source.read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(data, json_link=Path(os.path.relpath(args.source.resolve(), output.parent.resolve())).as_posix()))
    print(str(output))


if __name__ == "__main__":
    main()
