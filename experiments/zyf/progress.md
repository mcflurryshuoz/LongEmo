# LongEmo 三层消融进度

2026-09-23 04:09 CST。全集固定558题／141视频，02:48 已在 AIStudio 部署并核验 **noevent → method** 调度。base 仅保留已有评分参考，不新增回答或评分；未评分 base 题标为 `not_requested`。

| 条件 | 已评分／558题 | 已评分均分／100 |
|---|---:|---:|
| noevent | 154/558 | 28.41 |
| base（已有参考） | 13/558 | 63.46 |
| method | 13/558 | 69.23 |

base／method 当前均为试跑继承的同 **13 题**，method 暂高 **5.77 分**。noevent 的已评分题集不同，不能直接比较上表均分。三路共同9题：noevent **62.96**、base **58.33**、method **77.78**，仍属早期小样本。

该快照媒体已就绪 **135/141**，其余继续上传；noevent 当前8个视频运行、6个等待媒体，完成可执行队列后进入 method。已就绪媒体共59,955,689,664字节。上一快照164条首分逐字段不变，本次新增16条 noevent，共180条首次有效评分；冻结配置和原失败预算不变，不重抽评分。历史题型路由结果60.24分单独报告。

[全集报告与558题状态](results/three_level_full558_gemini38_20260922/report.md) · [范围切换核验](results/three_level_full558_gemini38_20260922/scope_change.md) · [50题试跑最终快照](results/three_level_pilot_gemini38_20260922/report.md)
