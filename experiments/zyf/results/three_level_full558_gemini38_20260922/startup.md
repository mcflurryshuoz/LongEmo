# 全集三层消融评测

已在 AIStudio 实验 **62910175** 启动全集协调器，固定 **558 题／141 视频**。2026-09-22 20:27 CST 正等待原 50 题试跑结束并审计继承；此时尚未产生新全集评分。历史题型路由版分数不并入本实验。

| 条件 | 记忆与检索 | 执行顺序 |
|---|---|---|
| noevent | 时间窗口记录上的 BM25＋Embedding 检索 | 先运行全集 |
| base | 冻结事件图召回与相邻关系扩展 | 第二阶段 |
| method | 与 base 共图、共索引和计划；纯渐进披露 | 第二阶段，同视频在 base 后运行 |

感知统一 Matrix Gemini 3.8 Flash，规划／回答／官方评分使用 Matrix GPT-6，向量使用 OpenRouter Gemini Embedding 2。每阶段最多12个视频流水线，每视频最多2题并发；逐视频完成记忆后立即答题与评分。原有成功记忆、答案及首次有效评分经校验后继承；失败预算保留，成功答案未评分时仅提交首次评分。

完整141个本地视频已校验，约68.38 GB；AIStudio原有21个视频，剩余约66.52 GB按视频补传。20:27快照已有23/141视频校验就绪，141份字幕与全集题单均就绪；视频使用唯一临时文件上传、SHA核验后原子发布，已到文件可先处理。

- 运行目录：`/root/longemo/runtime/runs/three_level_full558_gemini38_20260922`
- 全集协调器版本：`f8f9392`；部署：`/root/longemo/LongEmo-noevent-full-f8f9392`
- 启动脚本：[run_full_suite.sh](https://github.com/mcflurryshuoz/LongEmo/blob/noevent/experiments/zyf/run_full_suite.sh)
- 配置与恢复规则：[full_suite.md](https://github.com/mcflurryshuoz/LongEmo/blob/noevent/experiments/zyf/full_suite.md)
- 本地与 AIStudio 各53项检查通过；冻结来源见 [startup.json](startup.json)。

后续按每路已评分覆盖率、总体／题型／分剧、同题比较和逐题来源报告。缺失不计零分；队列结束不代表558题全部成功。

