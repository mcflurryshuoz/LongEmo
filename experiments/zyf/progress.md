# noevent 同源观察消融已启动

2026-09-24 05:19 CST，完整题单558题，当前92题可执行。独立来源、无新感知，详见[启动审计](results/noevent_shared_observations_20260924/README.md)。历史成绩不覆盖。

# LongEmo 当前进度

noevent-v2本批已结束：244/558题，64.45分；候选279题中35题失败。无新感知、无漏评。原50题pilot的80%门槛未通过，后续229题为审计后独立未尝试题阶段，未改变推理协议或重抽低分。

[最新结果](results/noevent_progressive_v2_20260924/report.md) · [逐题状态](results/noevent_progressive_v2_20260924/question_status.csv) · [结束审计](results/noevent_progressive_v2_20260924/final_audit.json)

---

# LongEmo 评测进度

2026-09-23 **18:28 CST**：noevent **279/558，25.42分，覆盖率50%**。全部可执行队列已结束，最后新增15条首分，原264条及来源不变。method保留97/558、59.45分；base保留本次试跑13/558、63.46分，均无新增请求。

最后五视频全部完成17/17评分。结束审计确认无活动评测进程、未决任务、成功答案漏评、有效评分漏归档或成功种子重复调用。

剩余279题已逐题归因：189题感知内容过滤、83题感知DLP拒绝、3题规划过滤、4题首次评分过滤；不重复提交明确拒绝请求，不能宣称558题全量成功。

成绩含247题8192且预处理不变、20题继承后16384、12题8192加固定单窗边界处理，保留逐题来源。缺失排除、真实零分保留。

[总体、三类任务与分剧结果](results/noevent_resume_20260923/report.md) · [558题状态CSV](results/noevent_resume_20260923/question_status.csv) · [逐题最终审计](results/noevent_resume_20260923/final_audit.json) · [最后五视频结束审计](results/noevent_resume_20260923/http428_seed5_end_audit.json)
