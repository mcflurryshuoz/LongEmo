# LongEmo 实验进度

[研究方案](plan.md) · [方法整体流程](../../README.md#zyf-分支当前方法整体流程) · [完整实验记录](README.md) · [初步结果核验](results/preliminary_review_20260917/review.md)

E09 v2 保留暂停检查点。用户于 2026-09-18 授权在 AIStudio 实验 62910175 配置环境并启动新实验 E10；正在迁移数据，服务检查与独立实验配置见 [E10 协议](aistudio_benchmark.md)。下表的旧成绩保持原样，新实验不复用旧预测或评分。

## 保留的全量实验：E09 v2

固定范围为 episode 的 141 个视频、558 道题、8342 个窗口。Gemini 3.8 Flash 负责音频观察和视频感知，原生 Gemini Embedding 2 用于事件／问题向量，Azure GPT-6 负责检索规划、答题和官方评分。详见[配置协议](blackai_benchmark.md)。

| 环节 | 已完成 | 状态 |
|---|---:|---|
| 通过校验并保存的窗口 | 995 / 8342 | 保留检查点，可在后续获准恢复时续跑 |
| 完整视频记忆 | 21 / 141 | 等待 Embedding 服务，尚未进入答题 |
| 本轮 Embedding 调用 | 0 | 前期接口验证返回 404，调度器已延后答题 |
| 已评分题目 | 0 / 558 | 尚未评分，不表示准确率为零 |

本轮保存了 1253 次音频尝试（1006 次成功）和 1113 次感知尝试（995 次成功）。失败包含校验／JSON 错误，以及音频 4 次 HTTP 502、感知 6 次 HTTP 524；尝试次数包含重试，不能等同于唯一窗口数。实际费用未返回，不能把 token 数当作已确认账单金额。

完整暂停快照：[progress.md](results/blackai_gemini38_gpt6_full_v2/paused/progress.md)／[progress.json](results/blackai_gemini38_gpt6_full_v2/paused/progress.json)。旧的 [450 窗口快照](results/blackai_gemini38_gpt6_full_v2/progress.md)作为历史记录保留，不能代表暂停后的进度。

## 已有得分与可比范围

| 实验 | 感知配置 | 评分覆盖 | 已评分题目的归一化均分 |
|---|---|---:|---:|
| E06 | Azure GPT-6 Astra | 185 / 558 | 48.56% |
| E08 | OpenRouter Gemini 3.8 Flash | 6 / 558 | 75.00% |
| E09 v2 | BlackAI 原生 Gemini 3.8 Flash | 0 / 558 | 尚无分数 |

E06 与 E08 使用的题目集合不同；对齐共同的 4 道题、2 个视频后，两者均为 **75.00%**，不能用上表均分之差宣称改进。

现有 direct 对照为 Friends S01E08 的 Q271／Q272：Gemini 2.5 Flash 直接视频方法与 GPT-6 图方法均为 **58.33%**。生成模型、媒体预算不同，且只有两题，尚不能证明图方法的收益。详见[同题对照记录](results/legacy_direct_friends_s01e08_v1/comparison.md)。

## 已定位问题与后续验证

- **Q42：检索上下文遗漏结尾。** E06 图中存在 359.4–360.8 秒的微笑事件，却未进入回答上下文；E08 返回了对应结尾事件，轨迹评分为 4/4，对比 E06 的 2/4。优先验证全时段时间线和证据压缩。
- **Q8：强度峰值选择错误。** E08 选择了不同的发怒时刻，评分从 E06 的 1/1 降为 0/1。需进一步核对感知、检索与回答，尚不能归因于某个模块。
- **Embedding 服务已找到可用路由。** BlackAI 原生服务仍未开通该模型；E10 明确使用 OpenRouter 的 Gemini Embedding 2，3072 维真实向量检查通过。旧 E09 不切换服务、不混用缓存。
- **仍缺严格的 direct 对照。** 后续应固定题目、生成模型、媒体预算与 judge，再评估图检索收益。

检索与感知质量改进仍是待验证事项。本次 E10 先验证新主机与服务组合，再处理完整 episode 题单；保留首次成功官方评分，不以低分触发重抽。
