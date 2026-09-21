# BlackAI Gemini 3.6 补评

核验时间：2026-09-21T08:59:08+00:00；后端状态：finished。

新实验：62.92 (20/69)；累计混合来源：57.76 (506/558)。原 486 道首次评分文件哈希保持不变。

本轮共同覆盖原先缺失的 69 题、13 视频。BlackAI Gemini 3.6 与 AICodeMirror Claude Opus 5 视觉＋Gemini 2.5 Pro 音频按视频分工，保留父图谱观察来源；两家产生的首次有效评分不重叠。Matrix GPT-6 规划/答题/官方评分、OpenRouter Gemini Embedding 2 不变。缺失题不计零分，不能作为同一配置的全量成绩。

执行分配（题数）：`{"BlackAI Gemini 3.6": 36, "AICodeMirror Claude Opus 5 + Gemini 2.5 Pro": 33}`。

| 任务 | 分数 /100（已评分/总题数） |
|---|---:|
| emotion trajectory | 54.28 (216/235) |
| emotional intensity comparison | 51.18 (170/194) |
| emotional reasoning | 73.33 (120/129) |

| 剧名 | 强度比较 | 情感轨迹 | 情感推理（等题权） |
|---|---:|---:|---:|
| 欢乐一家亲 | 51.52 (33/33) | 52.70 (37/37) | 83.33 (18/18) |
| 老友记 | 53.85 (26/26) | 51.89 (53/54) | 83.33 (22/24) |
| 家有儿女 | 50.00 (2/2) | 45.00 (5/5) | 33.33 (4/4) |
| Malcolm in the Middle | 40.00 (20/41) | 51.47 (17/34) | 79.17 (8/15) |
| 摩登家庭 | 50.00 (34/37) | 53.95 (38/39) | 72.55 (17/17) |
| 剧名未标注 | 54.55 (55/55) | 58.71 (66/66) | 67.97 (51/51) |

未评分 52 题；已有成功答案待首次评分 0 题。

逐题状态：`{"scored": 506, "blocked_question_policy": 2, "blocked_judge_policy": 1, "frontend_failed": 45, "pending_memory_or_score": 4}`。
旧有 2 道回答过滤与 1 道评分过滤保留；转交 AICodeMirror 的题目读取新运行状态，保留先前 BlackAI 失败或转交记录。新感知拒绝、暂时故障和待执行状态按逐题来源区分。

[逐题分数、来源、哈希和状态](report.json)。
