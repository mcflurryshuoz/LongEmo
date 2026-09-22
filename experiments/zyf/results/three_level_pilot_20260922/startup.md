# 三层对照实验启动记录

2026-09-22 17:54 CST 快照；这是新协议的 50 题／21 视频 pilot，尚无新评分，不能作为全集结果。

AIStudio 62910175 直连 Hugging Face 在鉴权前返回 `Network is unreachable`。已复用本地缓存，将 21 个视频和 21 份字幕（1,856,623,225 字节）传入新容器，42 个文件全部通过 SHA256 校验。

运行：`/root/longemo/runtime/runs/three_level_pilot50_matched_20260922`。协调器 PID 183365，Linux start ticks 1083942。感知总并发 12，后续视频并发 6、每视频问题并发 2。Matrix GPT-6 视觉、Matrix Gemini 3.8 Flash 音频和 OpenRouter Gemini Embedding 2 均通过真实输入预检。

base 和 method 共用新 schema v2 事件图、冻结计划与向量索引，分别采用 `graph` 和纯 `progressive --progressive-routing none`；noevent 独立构建窗口记忆。该控制组的 base 不能直接等同于历史 base 分支成绩。每路 799 个感知窗口；证据预算参数为 48000，实际字符及累计 token 另行记录，不声称总推理预算相同。各阶段只尝试一次，成功首分保留，失败任务不自动重发。

| 感知表示 | 运行中视频 | 已失败视频 | 未开始视频 | 已完成窗口 |
|---|---:|---:|---:|---:|
| event（base/method 共用） | 10 | 11 | 0 | 6 |
| noevent | 2 | 18 | 1 | 0 |

首批失败主要来自音频 `voice` 为空和观察引用未满足 schema。event 另有一次 `content_policy_violation` 与一次 `RemoteDisconnected`，须分别归因。API 预检成功不能代表所有窗口可用；当前启动成功也不代表构建或评测完成。保留现有请求和成功窗口，后续修正需要新的冻结配置与明确继承记录，不能删除失败记录或重抽有效评分。

源码与计数详见 [startup.json](startup.json)。逐视频任务、窗口和调用账本以远端现场文件为准；此文件是带时间的启动快照。
