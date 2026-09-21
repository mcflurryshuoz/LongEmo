# Claude 视觉与 Gemini 音频补评

用户同意尝试后，针对剩余 15 个图谱未完成的视频建立独立运行 `aicodemirror_claude5_gemini25_remaining_20260921`，涉及 79 道尚未评分的题目、507 个待补窗口。之前四批的 476 个首次有效评分保持不变；不包含另外 3 道 GPT-6 回答／评分过滤题。

- 视觉感知：AICodeMirror `claude-opus-5`，Anthropic Messages，多图输入，max_tokens=8192，不指定 temperature 或 thinking 扩展。
- 音频观察：AICodeMirror `gemini-2.5-pro`，原生 Gemini，max_tokens=4096，thinkingBudget=1024。
- 媒体：20 秒窗口、2 秒 padding、1 FPS、最多 16 帧、150528 像素。原提示词、图谱 schema、图检索与官方评分代码保持不变。
- 旧 BlackAI 窗口保留原来源。旧音频只在媒体／提示词／原模型配置指纹完全匹配时复用；不存在匹配缓存时调用新音频模型。新日志与旧日志分开保存。
- g450 构图，AIStudio 62910175 使用 Matrix GPT-6 规划／回答／官方评分与 OpenRouter Gemini Embedding 2。AIStudio 的 141 个视频已齐全，只传新图谱包和日志。

先以 3 并发验证 V137（余 4 窗口）、V48（余 7）、V104（余 9）的完整视频图谱。至少一个完整图谱成功，且三者均已返回后，再以最多 6 视频并发处理其余 12 个。若首批全部失败，保存原因并暂停后续前端，不把尚未尝试的视频当作 API 拒绝。

保留原每窗口最多 3 次 JSON 校验尝试。每视频只对明确的暂时 HTTP 错误额外做最多 3 次断点尝试；明确 refusal/content_filter 不重发，未知空响应／格式失败不自动归为暂时错误。中断且仍标记 in_flight 的作业先审计进程与请求，不自动重发。每个视频结束后才发布不可变包；不重复覆盖已导入结果。

实验配置保存方法源码哈希、感知适配器哈希、题号、媒体参数与来源文件哈希。完整图谱自动进入独立 GPT-6 评分队列；新评分与旧四批分开报告。累计成绩仍属于混合模型／提供方与部分覆盖结果，不能视作同一配置的全量分数。

运行位置：g450 `/mnt/data1/zyf/LongEmo-runtime/runs/aicodemirror_claude5_gemini25_remaining_20260921`；AIStudio `/root/longemo/runtime/runs/aicodemirror_claude5_gemini25_remaining_20260921`。状态以各端 frontend_status.json、backend_status.json、逐视频 attempts.json／worker_process.json、accepted_scores.jsonl 和 scores.jsonl 为准。


## 首批实际结果（2026-09-21）

首批 3 个视频完成后，V48 和 V137 的图谱完整并在 AIStudio 完成 10 道题评分；V104 以及其余 12 个视频在最多 3 次额外断点尝试后仍失败。当前新增评分均分、分剧明细和逐题状态由 AIStudio report 保存；本轮总覆盖从 476/558 变为 **486/558**。

失败原因按最后失败窗口归类：9 个 HTTP 524、2 个 HTTP 520、2 个音频观察 schema 错误（`voice` 为空）。这批请求没有新的 content_filter。由于视觉网关错误和音频结构错误仍未解决，剩余 13 个视频不能按当前配置估计为必然完成；若降低并发或媒体预算另开恢复，预计至少 2--4 小时，且需要重新验证图谱一致性。
