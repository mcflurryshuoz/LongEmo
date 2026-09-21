# 流式补评最终审计

核验时间：2026-09-21T10:14:00+00:00。本次流式49题运行已结束，新增评分0；全部9个包及两组producer_done已导入，前后端均退出。

累计 **506/558（90.68%），已评分题混合来源等题权均分57.76**。已有506条首分及六个评分文件SHA保持不变。缺失不计零分，不能视为同一配置的全量成绩。

流式续评共新增60个通过校验的窗口，9个视频仍未形成完整图谱，因此未进入答题／官方评分；续评范围内没有成功答案遗漏首次评分。

| 未评分原因 | 题数 | 证据 |
|---|---:|---|
| AICodeMirror网关HTTP520/522 | 27 | 视觉V112/V113/V116；音频V115/V117 |
| 结构、JSON或流完整性失败 | 16 | V104/V106/V129，未出现明确内容过滤证据 |
| 本地媒体提取超时 | 6 | V105第22窗口尚未发起模型调用 |
| 旧Matrix明确拒绝 | 3 | Q259/Q260回答、Q350评分，保留未完成 |

| 视频 | 完成窗口 | 新增有效窗口 | 最后失败阶段 |
|---|---:|---:|---|
| G2_V000104 | 59/63 | 2 | W00060: missing message_stop; event outside core interval; JSON parse failure |
| G2_V000105 | 21/65 | 11 | W00022: RuntimeError video media extraction timed out; no model call for that window |
| G2_V000106 | 36/64 | 0 | W00037: undeclared observation subject; JSON parse failure; missing message_stop |
| G2_V000112 | 20/65 | 7 | W00021: HTTP520 |
| G2_V000113 | 54/65 | 9 | W00055: core-interval validation failure, then HTTP520 |
| G2_V000115 | 46/65 | 9 | W00047: empty voice validation failure, then HTTP520 |
| G2_V000116 | 47/64 | 10 | W00048: HTTP520 |
| G2_V000117 | 63/64 | 9 | W00064: HTTP522 |
| G2_V000129 | 32/64 | 3 | W00033: core-interval validation failure; JSON parse failure; missing message_stop |

V117仅余第64窗口，音频HTTP522仍导致整集图谱未完成；未把部分图谱伪装为完成。每视频新协议只允许一次断点任务，当前次数已用完，没有重置预算或重复既有分数。

g450 SSH曾出现握手超时，原中转在三次失败后停止。最终V105包经单独传输、哈希校验和导入核验后补齐；没有重发已成功接收的7个包。所有源构建进程、两个后端和本地传输均已结束。

[三任务与分剧成绩](report.md) · [558题首分与来源](report.json) · [52题失败原因及哈希](final_stream_audit.json)。
