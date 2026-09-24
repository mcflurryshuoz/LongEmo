# LongEmoBench

<!-- WINDOW_RAG_TOP8:START -->
## Window-RAG 单次窗口检索基线

2026-09-24 12:06 CST队列结束：**502/558题，51.64分，覆盖率89.96%**。Top-8单次检索＋一次回答，无事件结构、无规划或渐进检索。剩余25题缓存缺失、22题既定后端拒绝、8题Embedding429、1题回答HTTP400；无漏评。[结果与审计](experiments/zyf/results/window_rag_top8_20260924/report.md)。
<!-- WINDOW_RAG_TOP8:END -->


<!-- LONGEMO_SHARED_OBS:START -->
## noevent 全集评测（运行中）

2026-09-24 06:01:56 CST：**355/558题，61.41分**。123个视频缓存正在并发评测，剩余18视频正在补缺失窗口。仅原始窗口观察＋渐进检索，移除事件结构；多来源单独标记，未混入旧版成绩。见[结果与558题来源](experiments/zyf/results/noevent_full_bank_20260924/report.md)。
<!-- LONGEMO_SHARED_OBS:END -->


[中文](README_zh.md) · [Dataset on Hugging Face](https://huggingface.co/datasets/mcflurryshuoz/LongEmoBench/tree/main)

LongEmoBench evaluates emotion understanding at two video granularities: **clip** and **episode**. Questions cover emotion recognition, transitions, trajectories, causes, intensity comparisons, and reasoning across a longer video. All tasks use open-ended questions; the expected answer may be emotion labels, a short result, or a natural-language explanation.

This repository provides prediction generation and a shared evaluator. Predictions from your own model or agent can be evaluated directly. **Evaluation requires question annotations and predictions; it does not load videos or subtitles.**

## method：事件图与渐进式检索

先从视频、音频和字幕构建带时间、人物、情感对象及证据引用的事件图；回答时用 **BM25＋Gemini Embedding 2** 定位相关事件，再按需披露时间流和关系流，由 GPT-6 基于证据作答。构图与问题无关，参考答案仅用于官方评分。

<!-- LONGEMO_NOEVENT_V2:START -->
### noevent-v2：渐进式窗口检索

2026-09-24：**244/558题，均分64.45**；完整缓存候选279题，其中35题未完成。复用已有窗口，不重新感知。三类分数：轨迹62.50、强度比较55.81、情感推理80.00。

本批队列已结束，无成功答案漏评。剩余35题：15题内容过滤、5题参数错误、8题输出格式错误、7题Embedding HTTP429。缓存包含不同输出预算和边界处理来源，分层报告。原50题pilot成功率39/50，80%自动门槛未通过；另行审计启动未尝试的229题，原gate和所有失败记录均保留。

这是新的多轮检索协议，不与下面旧单次检索或旧method不同题集均分直接比较；共享观察上的method-v2尚未评测。

[总体、任务和分剧成绩](experiments/zyf/results/noevent_progressive_v2_20260924/report.md) · [558题状态](experiments/zyf/results/noevent_progressive_v2_20260924/question_status.csv) · [结束审计](experiments/zyf/results/noevent_progressive_v2_20260924/final_audit.json)
<!-- LONGEMO_NOEVENT_V2:END -->

<!-- LONGEMO_FULL558_RESULTS:START -->
### 历史单次检索评测结果

2026-09-23 **18:28 CST**：noevent已完成所有可执行队列，**279/558题、25.42分，覆盖率50%**。method与base保留已有成绩。

| 条件 | 已评分／558题 | 已评分均分／100 |
|---|---:|---:|
| noevent | 279/558 | 25.42 |
| method（保留成绩） | 97/558 | 59.45 |
| base（本次试跑参考） | 13/558 | 63.46 |

剩余279题：189题感知内容过滤、83题感知DLP拒绝、3题规划过滤、4题首次评分过滤。没有漏评、漏归档或活动评测进程；尚未实现558题全部成功。

汇总含**247题原8192预算＋20题继承后16384＋12题8192及固定窗口边界处理**，逐题保留差异。缺失排除、真零分保留，不直接比较不同成功题集总体均分。初轮共同68题为noevent32.60、method62.13（+29.53），仅作非随机交集参考；历史路由版60.24独立保留。

[三类任务与分剧结果](experiments/zyf/results/noevent_resume_20260923/report.md) · [558题状态](experiments/zyf/results/noevent_resume_20260923/question_status.csv) · [最终审计](experiments/zyf/results/noevent_resume_20260923/final_audit.json)
<!-- LONGEMO_FULL558_RESULTS:END -->

### 历史题型路由结果

**60.24/100，已评分 511/520 题（98.27%）。** 在相同 511 题上，base 为 57.76，提升 **2.48 个百分点**。本次采用题型路由：轨迹题渐进披露，强度比较与推理题使用完整图证据。

| 指标 | method /100 | base 同题 /100 | 差值 | 已评分 / 总题数 |
|---|---:|---:|---:|---:|
| **总体** | **60.24** | **57.76** | **+2.48** | **511/520** |
| 情感强度比较 | 49.13 | 51.45 | −2.31 | 173/174 |
| 情感轨迹 | 60.60 | 54.61 | +5.99 | 217/222 |
| 情感推理 | 75.48 | 72.45 | +3.03 | 121/124 |

分剧成绩如下，括号为已评分题数。

| 剧名 | 强度比较 | 情感轨迹 | 情感推理 |
|---|---:|---:|---:|
| 欢乐一家亲（Frasier） | 48.48 (33) | 63.89 (36) | 82.35 (17) |
| 老友记（Friends） | 50.00 (26) | 61.54 (52) | 87.88 (22) |
| 家有儿女 | 50.00 (2) | 50.00 (5) | 33.33 (4) |
| Malcolm in the Middle | 41.67 (24) | 65.91 (22) | 76.67 (10) |
| 摩登家庭 | 44.12 (34) | 63.51 (37) | 80.39 (17) |
| 剧名未标注 | 55.56 (54) | 55.38 (65) | 69.28 (51) |

每题归一化后等权平均；8 题缺少预测、1 题评分服务报错，缺失不计零分。这是 520 题子集的路由版结果；schema v2 三层消融独立评测，结果不与本表混合。详见[实验报告](experiments/zyf/results/method_hybrid_20260922/report.md)、[同题对比](experiments/zyf/results/method_hybrid_20260922/comparison.json)和[逐题评分](experiments/zyf/results/method_hybrid_20260922/question_scores.jsonl)。

### 方法概括

1. **感知与构图：** 20 秒核心窗口＋两侧 2 秒上下文，融合视觉、字幕和带时间戳的音频观察，抽取人物、事件、情感状态及证据。跨窗持续事件可归并；时间、引用和模态校验通过后保存。
2. **全图定位：** GPT-6 规划人物、对象、时间与查询词；BM25／结构化匹配和 Gemini Embedding 2 双路召回，通过 RRF 融合找到锚点。
3. **渐进披露：** 先提供少量锚点，模型按人物、情感对象、时间方向或显式关系请求下一页事件，最多扩展 3 轮；达到证据覆盖或预算上限后作答。图谱保持冻结，时间邻接不当作因果证据。
4. **结构改进：** schema v2 新增动作、对象、参与者和媒体中明确出现的数量／评分等信号，旨在改善强度比较与候选漏检；其效果由新的三路实验验证。

实现见[事件图](methods/longemo/memory.py)、[双路检索](methods/longemo/retrieval.py)、[渐进披露](methods/longemo/progressive_retrieval.py)。音视频感知来源逐窗记录；向量使用 OpenRouter Gemini Embedding 2，规划、答题和官方评分使用 Matrix GPT-6 Astra。

### 三层级消融与论文分析

| 分支 | 感知记忆 | 检索方式 | 要验证的作用 |
|---|---|---|---|
| `noevent` | 仅时间窗口记录，不生成事件、状态或关系 | 窗口上的语义＋Embedding 检索 | 窗口表示基线 |
| `base` | 完整事件图 | 全图召回后一次性扩展相邻事件与关系 | 事件表示、跨窗归并与导航 |
| `method` | 完整冻结事件图 | 全图定位锚点，逐步披露相关事件流 | 按需检索的证据覆盖与效率 |

本次 noevent 与 method 固定媒体、采样、模型、题单、评分器及 48000 字符证据预算参数；实际输入和累计 token 另行统计。已完成的13题 base／method 试跑共用冻结图、索引和规划。全集固定为 **558 题／141 视频**，本轮按 `noevent → method` 执行；已有 base 保留配置和逐题来源，不再运行。纯渐进式与历史题型路由版结果分开报告。

本轮 **Gemini 3.8 Flash 感知＋GPT-6 规划／回答／评分** 实验覆盖全集题单，初轮可执行队列已结束；noevent 按新指令继续。试跑首分核验后继承，运行时最多 **12 视频并发、每视频 2 题并发**，依次执行 noevent 和 method；媒体边传边处理，每视频完成记忆后答题评分。完整题单和媒体 SHA 固定，失败预算与首次评分保留；未评分范围见上方审计。启动方式见 [全集脚本](https://github.com/mcflurryshuoz/LongEmo/blob/noevent/experiments/zyf/full_suite.md)。

50题试跑已结束并完成首分审计；[试跑报告与逐题来源](experiments/zyf/results/three_level_pilot_gemini38_20260922/report.md)单独保留。

重点分析总体／题型／分剧成绩与覆盖率、同题差值及视频聚类置信区间、跨阶段证据覆盖、感知与检索错误，以及披露事件数、token、延迟和成本。论文突出两项可检验贡献：**事件图能否改善人物—对象—时间证据的一致性；渐进披露能否以更少上下文保留长时间依赖。** 强度比较当前下降，应作为待验证的改进点。

[完整消融与论文分析计划](experiments/zyf/three_level_ablation.md) · [noevent 方案](experiments/zyf/noevent_plan.md) · [方法文档](methods/longemo/README.md)

初轮可执行队列与媒体传输已结束；noevent 已按新指令继续，后续记录见上方链接。[队列结束审计](experiments/zyf/results/three_level_full558_gemini38_20260922/completion_audit.md) · [历史启动记录](experiments/zyf/results/three_level_full558_gemini38_20260922/startup.md)。

## Benchmark setup

Use Python 3.10 or later. Clone the repository and run the following commands from its root directory:

```bash
git clone https://github.com/mcflurryshuoz/LongEmo.git
cd LongEmo
python -m pip install -e .
```

## 1. Get the data

Questions and media are distributed through the [Hugging Face dataset](https://huggingface.co/datasets/mcflurryshuoz/LongEmoBench/tree/main). Access requires approval on the dataset page and authentication with an approved account.

```bash
python -m pip install -U huggingface_hub
hf auth login
hf download mcflurryshuoz/LongEmoBench \
  --repo-type dataset \
  --local-dir data/LongEmoBench
```

The published episode directory is organized as follows. Refer to the dataset page for the available files in each release.

```text
data/LongEmoBench/
  episode/
    G2_Q000001.json
    G2_Q000002.json
    ...
    videos/
      G2_V000001.mp4
      ...
    subtitles/
      G2_V000001.srt
      ...
```

Use `--data-path data/LongEmoBench/episode` with `-g episode`. For prepared clip data, select its question directory with `-g clip`. The two granularities have independent question and video IDs. Multiple questions can share one `video_id`.

`--data-path` accepts a single JSON object, a JSON array, a JSONL file, or a flat directory of JSON/JSONL files. It does not recursively load subdirectories. Video inference looks for `<video_id>.mp4` in the data directory's `videos/` folder unless `--videos-dir` is provided.

### Tasks

| Granularity | `type` | What the question asks | Answer and evaluation |
|---|---|---|---|
| clip | `contextual emotion` | Sustained individual emotions, shared emotions, or a group's overall emotional atmosphere. | EMOTIC labels; Precision, Recall, F1, EM. |
| clip | `emotion transition` | Emotions before and after a specified event, action, statement, or interaction. | Before/after label sets; score each set separately, then average. |
| clip | `emotion influence` | The emotional response to a specified event or interaction. | EMOTIC labels; Precision, Recall, F1, EM. |
| clip | `emotion trajectory` | A person's emotional development, including intensity changes or recurring patterns. | Natural language; holistic 0–4 rubric. |
| clip | `emotion cause` | The cause of an emotion, or an emotion-related motive or intention. | Natural language; holistic 0–3 rubric. |
| episode | `emotional intensity comparison` | An intensity comparison or extreme supported by the video. | A definite result, or one of the accepted alternatives; 0/1 rubric. |
| episode | `emotion trajectory` | Emotional development within the specified storyline, experience, scene, or relationship. | Natural language; holistic 0–4 rubric. |
| episode | `emotional reasoning` | Emotion-dependent inference integrating information across the video. | Definite results: 0/1; explanations: 0–3. |

For episode trajectories, the question identifies the relevant scope. The complete video can be the scope when it follows one coherent storyline. The evaluator always uses the score bands stored in each question's `rubric`.

### Question format

| Field | Meaning |
|---|---|
| `question_id` | Question ID, unique within its granularity. |
| `video_id` | ID of the corresponding prepared video. |
| `source` | Original source in `from`; source-video intervals in `segments`, or `null` when no interval list is needed. |
| `granularity` | `clip` or `episode`. |
| `type` | Task name from the table above. |
| `question` | Question text and any required answer format. |
| `answer` | Reference label list, before/after label sets, or natural-language answer. |
| `answer_details` | Additional reference information, or `null`. |
| `rubric` | Scoring criterion and score-band descriptions; `null` for locally scored label questions. |
| `subtitles` | Embedded subtitle rows for clip questions, or `null`. |

`answer_details` contains `description` and `items`. The description explains how to use the items: chronological stages, necessary causal factors, or alternative complete answers. Necessary components are considered together; when alternatives are provided and the question requests one answer, any one valid alternative suffices. **These items provide reference content; they do not assign separate points.**

This synthetic example demonstrates the format, not a released question or its gold answer:

```json
{
  "question_id": "G2_Q_EXAMPLE",
  "video_id": "G2_V_EXAMPLE",
  "source": {"from": "Example video", "segments": null},
  "granularity": "episode",
  "type": "emotional intensity comparison",
  "question": "During which match does A appear most nervous?",
  "answer": "The final match.",
  "answer_details": null,
  "rubric": {
    "criterion": "Evaluate whether the answer identifies the correct match.",
    "scores": {
      "0": "The answer is incorrect, incomplete, or contradictory.",
      "1": "The answer correctly identifies the final match. Equivalent wording is accepted."
    }
  },
  "subtitles": null
}
```

Label questions use the following vocabulary. Labels are case-insensitive during scoring.

```text
peace, affection, esteem, anticipation, engagement, confidence,
happiness, pleasure, excitement, surprise, sympathy, doubt/confusion,
disconnection, fatigue, embarrassment, yearning, disapproval, aversion,
annoyance, anger, sensitivity, sadness, disquietment, fear, pain, suffering
```

Reference answers, answer details, and rubrics are used for evaluation and must not be included in the tested model's input.

## 2. Evaluate predictions

The API inference runner and evaluator use the Python standard library. LLM-scored questions require access to a judge model API; label-only evaluation does not.

### Prepare a prediction file

Save one record per question in JSONL, or use a JSON array. Match the released `question_id` exactly and keep clip and episode predictions in separate files.

Examples below illustrate serialization only:

```jsonl
{"question_id":"G1_LABEL_EXAMPLE","prediction":["sadness","yearning"]}
{"question_id":"G1_TRANSITION_EXAMPLE","prediction":{"before":["fear"],"after":["peace"]}}
{"question_id":"G1_TRAJECTORY_EXAMPLE","prediction":"A is initially hopeful, becomes anxious after the setback, and regains confidence."}
```

For every LLM-scored answer, `prediction` must be text, including numbers and Yes/No answers:

```jsonl
{"question_id":"G2_RESULT_EXAMPLE","prediction":"3"}
{"question_id":"G2_EXPLANATION_EXAMPLE","prediction":"A is worried about disappointing the team."}
```

The evaluator also accepts `pred_answer` when `prediction` is absent or blank. Label predictions can alternatively be comma-separated text; transition text can use `Before: ...` and `After: ...` on two lines. The repository's inference entry points produce `predictions.jsonl` directly.

### Run evaluation

Configure your judge service. The placeholders below must be replaced with the service's model name, base URL, and key.

```bash
export MODEL_API_KEY="YOUR_API_KEY"
export MODEL_BASE_URL="https://your-service.example/v1"
export JUDGE_MODEL="YOUR_JUDGE_MODEL"

python -m evaluation.eval \
  --data-path data/LongEmoBench/episode \
  --predictions output/episode/predictions.jsonl \
  -g episode \
  --model "$JUDGE_MODEL" \
  --output-dir output/episode/scores
```

For clip evaluation, use the clip question and prediction paths with `-g clip`. If the selected data contains only label questions, omit the model and API settings. `longemobench-score` is equivalent to `python -m evaluation.eval` after installation.

Each invocation evaluates all questions of the selected granularity under `--data-path`. To evaluate a subset, provide a question file or directory containing that subset. A smaller prediction file alone does not reduce the evaluation scope. Optional `--workers` controls concurrency and `--tries` sets the maximum request attempts, including the first attempt.

### How scores are computed

**Label questions (`rubric: null`).** Compare predicted and reference label sets. Labels are normalized and deduplicated; extra incorrect labels reduce precision, and omissions reduce recall.

| Metric | Definition |
|---|---|
| Precision | Correct predicted labels / predicted labels. |
| Recall | Correct predicted labels / reference labels. |
| F1 | `2 × Precision × Recall / (Precision + Recall)`. |
| EM | 1 when the two sets match exactly; otherwise 0. |

For transition questions, compute each metric separately for `before` and `after`, then average the two values. Task metrics average questions equally. F1 is the label question's contribution to the overall normalized score.

**Questions with a rubric.** The LLM judge receives the question, reference answer, optional `answer_details`, model prediction, and the complete rubric. It returns one allowed integer `score` and a `reason`. This route also handles short results such as names, counts, and Yes/No answers. The judge does not receive the video. Prompts and the four embedded examples are in [judge_prompts.py](evaluation/judge_prompts.py).

Each score is normalized as `score / maximum allowed score`. Trajectories therefore report both their raw 0–4 mean and a percentage equal to `raw mean × 25`; 0–3 explanations use `raw mean / 3 × 100`. Binary-result averages are ACC. The rubric specifies an overall judgment, not a sum over reference items.

**Aggregation.** Reports include each task's results and the unweighted mean of normalized question scores within the selected granularity. Clip and episode scores are reported separately.

Episode emotional reasoning additionally reports:

- Result-question ACC and explanation-question raw/normalized means separately.
- An unweighted mean over all scored reasoning questions.
- A weighted score: `0.6 × result ACC + 0.4 × normalized explanation mean`, with a 0–100 version.

When only one reasoning group has scored answers, the weighted result uses that group's normalized mean. **Missing or blank predictions and failed evaluations are excluded from score means and reported in coverage/status counts.** Compare scores together with `n_scored / n_total`; an incomplete run is not a full benchmark result.

### Read the outputs

Every evaluation creates a new directory, including repeated runs on the same predictions:

```text
output/episode/scores/run_YYYYMMDD_HHMM/
  scores.jsonl    # Per-question score, reason, prediction, and judge details
  metrics.json    # Task metrics, overall mean, coverage, and reasoning aggregates
  summary.md     # Readable score tables
```

A numeric suffix distinguishes runs started in the same minute. `metrics.json` contains `tasks`, `overall_unweighted`, and, for episode runs, `emotional_reasoning`. The evaluator exits with a nonzero code if any selected question is unscored or fails.

## 3. Generate predictions with the API baseline

Use the shared API entry point when you also need model predictions. Video processing requires `ffmpeg` and `ffprobe` on `PATH`.

```bash
export MODEL_API_KEY="YOUR_INFERENCE_API_KEY"
export MODEL_BASE_URL="https://your-inference-service.example/v1"
export INFERENCE_MODEL="YOUR_VIDEO_MODEL"

python -m evaluation.inference.run \
  --data-path data/LongEmoBench/episode \
  --videos-dir data/LongEmoBench/episode/videos \
  -g episode \
  --modality video \
  --model "$INFERENCE_MODEL" \
  --output-dir output/episode
```

Set `MODEL_BASE_URL` and `MODEL_API_KEY` for the inference service, which may differ from the judge service. Use `-g clip` for clip questions. Episode video input uses frames sampled across the video; frame count and resolution can be controlled with `--fps`, `--max-frames`, and `--frame-max-pixels`.

Subtitles are not appended by default. `--with-subtitle` appends them, while `--modality text` runs a subtitle-only baseline. Clip subtitles are read from each question. The current episode subtitle reader expects `subtitles/<video_id>.json` at the repository root, containing rows with `id`, `t`, `speaker`, and `text`; convert the released SRT files to this format before running a subtitle baseline. This conversion is not needed for evaluation or video inference without subtitles.

Sampled frames do not include audio by default; `--with-audio` adds a separate track where the model/API supports it. Native video requests carry the source file. Record the actual media settings when comparing results. Inference reuses successful predictions by question ID in the same output directory. Use a new `--output-dir` or `--force` when changing the model or input settings.

## Code entry points

| File | Purpose |
|---|---|
| [evaluation/eval.py](evaluation/eval.py) | Evaluate local or externally generated predictions. |
| [evaluation/metrics.py](evaluation/metrics.py) | Label metrics, normalization, and score aggregation. |
| [evaluation/judge_prompts.py](evaluation/judge_prompts.py) | Shared judge prompts and output validation. |
| [evaluation/io_utils.py](evaluation/io_utils.py) | Question loading and subtitle format. |
| [evaluation/inference/run.py](evaluation/inference/run.py) | API prediction generation. |
| [evaluation/inference/transformers.py](evaluation/inference/transformers.py) | Local Transformers prediction generation. |
| [methods/agentic/runner.py](methods/agentic/runner.py) | Agentic frame-inspection method. |

Specialist emotion-model code is under `evaluation/inference/emollm/` and retains its upstream license files. The previous repository version is preserved on the [`v0` branch](https://github.com/mcflurryshuoz/LongEmo/tree/v0).
