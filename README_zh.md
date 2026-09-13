# LongEmoBench

[English](README.md) · [Hugging Face 数据集](https://huggingface.co/datasets/mcflurryshuoz/LongEmoBench/tree/main)

LongEmoBench 在 **clip（片段）** 和 **episode（长视频）** 两个粒度上评估情绪理解能力，覆盖情绪识别、前后变化、情绪轨迹、原因解释、强度比较和跨片段推理。所有任务采用开放问答，答案可以是情绪标签、明确的短答案或自然语言解释。

本仓库提供推理与统一评估代码，也支持直接评估其他模型或 Agent 生成的答案。**评估已有预测时，只需要问题标注和预测文件，不需要加载视频或字幕。**

使用 Python 3.10 或更高版本。克隆仓库后，在仓库根目录执行下方命令：

```bash
git clone https://github.com/mcflurryshuoz/LongEmo.git
cd LongEmo
python -m pip install -e .
```

## 1. 获取数据

问题和媒体文件发布在 [Hugging Face 数据集](https://huggingface.co/datasets/mcflurryshuoz/LongEmoBench/tree/main)。请先在数据集页面申请访问，获批后使用对应账号登录下载。

```bash
python -m pip install -U huggingface_hub
hf auth login
hf download mcflurryshuoz/LongEmoBench \
  --repo-type dataset \
  --local-dir data/LongEmoBench
```

已发布的 episode 目录采用以下结构。各版本可用的问题和媒体文件以数据集页面为准。

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

读取上述问题时，使用 `--data-path data/LongEmoBench/episode` 和 `-g episode`。读取准备好的 clip 数据时，指定其问题目录并使用 `-g clip`。两个粒度单独编号；同一视频的多道问题共用一个 `video_id`。

`--data-path` 支持单个 JSON 对象、JSON 数组、JSONL 文件，以及平铺的 JSON/JSONL 问题目录，不递归读取子目录。视频推理默认在数据目录的 `videos/` 下查找 `<video_id>.mp4`，也可以通过 `--videos-dir` 指定视频目录。

### 任务类别

| 粒度 | `type` | 考查内容 | 答案与评估 |
|---|---|---|---|
| clip | `contextual emotion` | 单人持续情绪、多人共同情绪或群体整体情绪氛围。 | EMOTIC 标签；Precision、Recall、F1、EM。 |
| clip | `emotion transition` | 指定事件、动作、话语或互动前后的情绪。 | 前后两组标签分别计分，再取平均。 |
| clip | `emotion influence` | 指定事件或互动引起的情绪反应。 | EMOTIC 标签；Precision、Recall、F1、EM。 |
| clip | `emotion trajectory` | 人物的情绪发展，包括强度变化及反复模式。 | 自然语言；0–4 分整体 rubric。 |
| clip | `emotion cause` | 情绪产生的原因，或与情绪相关的动机、意图。 | 自然语言；0–3 分整体 rubric。 |
| episode | `emotional intensity comparison` | 视频支持的情绪强度比较或极值判断。 | 明确结果，或可接受答案之一；0/1 rubric。 |
| episode | `emotion trajectory` | 指定故事线、经历、场景或人物关系内的情绪发展。 | 自然语言；0–4 分整体 rubric。 |
| episode | `emotional reasoning` | 结合视频多处信息、依赖情绪判断完成推理。 | 明确结果使用 0/1；解释使用 0–3 分。 |

Episode 轨迹题由题干限定所问范围；整段视频只有一条连贯故事线时，也可以覆盖整个视频。评估时以每题保存的 `rubric` 档位为准。

### 问题格式

| 字段 | 含义 |
|---|---|
| `question_id` | 问题编号，在所属粒度内唯一。 |
| `video_id` | 对应的已准备视频编号。 |
| `source` | `from` 保存原始来源；`segments` 保存原视频时间范围，无需区间列表时为 `null`。 |
| `granularity` | `clip` 或 `episode`。 |
| `type` | 上表中的任务名称。 |
| `question` | 题干及必要的回答格式要求。 |
| `answer` | 参考答案：情绪标签列表、前后标签集合或自然语言。 |
| `answer_details` | 更细化的参考依据，无需补充时为 `null`。 |
| `rubric` | 评价要求和各档分数的描述；由代码评分的标签题为 `null`。 |
| `subtitles` | Clip 题目内嵌的字幕列表，或 `null`。 |

`answer_details` 包含 `description` 和 `items`。前者说明如何使用条目，后者可以列出按时间排序的阶段、必要原因因素或其他完整有效答案。必要组成内容共同核对；题目只要求一个答案、且提供了多个可接受答案时，答出其中任意一个即可。**这些条目提供参考内容，不单独分配分值。**

以下为格式示例，不是数据集中某道题的真实标注：

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

标签题使用以下词表，评估时不区分大小写：

```text
peace, affection, esteem, anticipation, engagement, confidence,
happiness, pleasure, excitement, surprise, sympathy, doubt/confusion,
disconnection, fatigue, embarrassment, yearning, disapproval, aversion,
annoyance, anger, sensitivity, sadness, disquietment, fear, pain, suffering
```

参考答案、`answer_details` 和 `rubric` 仅用于评估，不应传给被测模型。

## 2. 评估预测结果

API 推理入口和评估程序使用 Python 标准库。LLM 评分需要可访问的裁判模型 API；仅评估标签题时不需要模型服务。

### 准备预测文件

JSONL 每行保存一道题，也支持 JSON 数组。`question_id` 必须与数据中的编号一致，同一题只保存一条预测，两个粒度分别保存文件。

以下内容仅展示序列化格式：

```jsonl
{"question_id":"G1_LABEL_EXAMPLE","prediction":["sadness","yearning"]}
{"question_id":"G1_TRANSITION_EXAMPLE","prediction":{"before":["fear"],"after":["peace"]}}
{"question_id":"G1_TRAJECTORY_EXAMPLE","prediction":"A is initially hopeful, becomes anxious after the setback, and regains confidence."}
```

所有需要 LLM 评分的答案均使用文本，包括数字和 Yes/No：

```jsonl
{"question_id":"G2_RESULT_EXAMPLE","prediction":"3"}
{"question_id":"G2_EXPLANATION_EXAMPLE","prediction":"A is worried about disappointing the team."}
```

当 `prediction` 缺失或为空时，程序也接受 `pred_answer`。标签题还支持逗号分隔的标签文本；前后变化题可以使用 `Before: ...` 和 `After: ...` 两行文本。本仓库的推理入口会直接生成 `predictions.jsonl`。

### 执行评估

配置裁判模型服务。下方的模型名称、服务地址和密钥占位符需替换为实际配置。

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

评估 clip 时，替换为对应的问题、预测和输出路径，并设置 `-g clip`。所选问题全部是标签题时，可以省略模型和 API 配置。安装后也可以使用 `longemobench-score`，其作用与 `python -m evaluation.eval` 相同。

每次评估都会读取 `--data-path` 下所选粒度的全部问题。若只评估一部分，应提供仅包含该子集的问题文件或目录；只缩小预测文件不会缩小评估范围。`--workers` 控制并发数，`--tries` 设置包含首次请求在内的最大尝试次数。

### 如何计算分数

**标签题（`rubric: null`）。** 比较预测情绪集合与参考集合，统一名称格式并去重；错加情绪降低精确率，漏答情绪降低召回率。

| 指标 | 定义 |
|---|---|
| Precision | 正确预测的标签数 / 预测标签总数。 |
| Recall | 正确预测的标签数 / 参考标签总数。 |
| F1 | `2 × Precision × Recall / (Precision + Recall)`。 |
| EM | 两个集合完全一致为 1，否则为 0。 |

Transition 对 `before` 和 `after` 分别计算上述指标，再取两者平均值。任务成绩按题目等权平均；标签题使用 F1 参与整体归一化成绩的汇总。

**带 rubric 的题目。** LLM 裁判接收问题、参考答案、可选的 `answer_details`、模型预测及完整 rubric，输出一个允许的整数 `score` 和判分理由 `reason`。人物、次数、Yes/No 等短答案也通过此流程评分。裁判不接收视频。评估提示词和四个示例位于 [judge_prompts.py](evaluation/judge_prompts.py)。

每题按 `score / 该题满分` 归一化。轨迹题同时报告 0–4 分原始平均值与 `原始均分 × 25` 的百分制成绩；0–3 分解释题的百分制为 `原始均分 / 3 × 100`；0/1 结果题的平均值就是 ACC。Rubric 给出的是整条回答的得分，不把参考条目逐项相加。

**成绩汇总。** 分别报告每个任务的成绩，以及所选粒度内所有已评分题目的归一化均分。Clip 与 episode 单独评估和汇报。

Episode 情绪推理额外报告：

- 结果题 ACC，以及解释题的原始均分与归一化均分。
- 所有已评分推理题的归一化分数直接平均，得到不加权成绩。
- 加权成绩：`0.6 × 结果题 ACC + 0.4 × 解释题归一化均分`，同时报告乘 100 后的百分制。

只有一组存在已评分答案时，加权成绩使用该组的归一化均分。**缺失、空白预测及评估失败不计入均分，单独计入覆盖率和状态统计。** 比较成绩时应同时查看 `n_scored / n_total`，未完成的评估不能视为完整 benchmark 成绩。

### 查看结果

每次评估都会新建一个目录，重复评估也不会覆盖前一次结果：

```text
output/episode/scores/run_YYYYMMDD_HHMM/
  scores.jsonl    # 逐题分数、理由、预测及裁判评估记录
  metrics.json    # 各任务、整体成绩、覆盖率和推理加权成绩
  summary.md     # 可直接阅读的成绩表
```

同一分钟内的多次运行用数字后缀区分。`metrics.json` 中包含 `tasks`、`overall_unweighted`；episode 还包含 `emotional_reasoning`。只要有题目未评分或发生错误，程序就返回非零退出码。

## 3. 使用 API 基线生成预测

需要先生成模型答案时，可以使用共用 API 推理入口。处理视频前需确保 `ffmpeg` 和 `ffprobe` 已安装并可从命令行调用。

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

将 `MODEL_BASE_URL` 和 `MODEL_API_KEY` 配置为被测模型的服务，其与裁判服务可以不同。Clip 使用 `-g clip`。Episode 视频输入采用覆盖整段视频的抽帧，可通过 `--fps`、`--max-frames` 和 `--frame-max-pixels` 控制采样。

默认不附加字幕；`--with-subtitle` 用于附加字幕，`--modality text` 用于纯字幕基线。Clip 从题目中读取字幕。当前 episode 字幕读取器要求仓库根目录下存在 `subtitles/<video_id>.json`，列表中每行包含 `id`、`t`、`speaker`、`text`；运行字幕基线前，需将发布的 SRT 文件转换为此格式。评估已有预测或不加字幕的视频推理均无需转换。

抽帧默认不包含音频；模型/API 支持时，`--with-audio` 会附加独立音轨。原生视频请求发送源视频文件。比较模型时应记录实际使用的输入模态和采样配置。推理按同一输出目录中的题号复用已成功的预测；更换模型或输入配置时，应使用新的 `--output-dir` 或添加 `--force`。

## 代码入口

| 文件 | 用途 |
|---|---|
| [evaluation/eval.py](evaluation/eval.py) | 评估本仓库或外部方法生成的预测。 |
| [evaluation/metrics.py](evaluation/metrics.py) | 标签指标、分数归一化与成绩汇总。 |
| [evaluation/judge_prompts.py](evaluation/judge_prompts.py) | 共用评估提示词与输出检查。 |
| [evaluation/io_utils.py](evaluation/io_utils.py) | 问题读取与字幕格式。 |
| [evaluation/inference/run.py](evaluation/inference/run.py) | API 推理入口。 |
| [evaluation/inference/transformers.py](evaluation/inference/transformers.py) | 本地 Transformers 推理入口。 |
| [methods/agentic/runner.py](methods/agentic/runner.py) | Agentic 视频抽帧检查方法。 |

专用情绪模型源码位于 `evaluation/inference/emollm/`，保留其原有许可证文件。仓库旧版本保存在 [`v0` 分支](https://github.com/mcflurryshuoz/LongEmo/tree/v0)。
