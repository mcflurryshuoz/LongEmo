# LongEmoBench

LongEmoBench 是一个长视频情感理解基准。本目录提供题目数据、源视频与处理后
视频的目录规范、视频预处理脚本，以及模型推理和评分代码。

```text
LongEmoBench/
├── data/                         # 题目数据
│   └── <series>/
│       ├── g1_clip/              # 片段级题目
│       └── g2_episode/           # 整集级题目
├── subtitles/                    # 仅含对白的逐集字幕
│   └── <series>/
│       └── <episode>.json
├── videos/
│   ├── sources/<series>/         # 原始整集与 valid_ranges.json
│   └── processed/
│       ├── clips/                # g1_clip 使用的视频
│       └── episodes/             # 去除片头片尾后的整集
├── preprocess/                   # 视频预处理脚本
└── evaluation/                   # 推理和评分代码
    └── inference/
```

## 视频粒度

`g1_clip` 是片段级输入。每道题提供一个与问题相关的短视频片段，用于评估
模型对局部场景中的情绪识别和情感推理能力。

`g2_episode` 是整集级输入。每道题提供去除片头和片尾后的完整一集，用于评估
模型分析跨场景情绪变化，并整合长时间范围内情感信息的能力。

## 字幕

逐集字幕位于 `subtitles/<series>/<episode>.json`。字幕中已去除嵌入文本的
说话人标签、舞台动作、音效说明和字幕制作信息，避免非语言描述向模型泄露
视觉或音频证据。

每条字幕包含 `series`、`ep`、`unit_id`、`t`、`speaker` 和 `text`。
`t` 是以秒为单位的 `[start, end]` 时间区间；`speaker` 是已标注的角色名，
源标注无法可靠识别角色时为 `unknown`。

从审核后的 `s1_perception` 文件重新生成全部字幕：

```bash
PYTHONPATH=. python3 preprocess/prepare_subtitles.py \
  --inputs-root /path/to/vebench/outputs \
  --out subtitles
```

可添加 `--series friends` 或 `--episode S01E01` 限制导出范围，两个参数均可
重复使用。

## 视频预处理

原始视频可从 Hugging Face 的
[`videos/sources`](https://huggingface.co/datasets/mcflurryshuoz/LongEmoBench/tree/main/videos/sources)
目录下载。下载后将以下命令中的 `/path/to/source/videos` 替换为本地视频目录。

Hugging Face 也会在
[`videos/processed`](https://huggingface.co/datasets/mcflurryshuoz/LongEmoBench/tree/main/videos/processed)
目录提供已经处理好的 `clips` 和 `episodes`，可直接下载用于推理。只有需要从
原始视频重新生成输入视频时，才需要运行下面的预处理命令。

源视频按以下结构存放：

```text
/path/to/source/videos/<series>/
├── S01E01.mp4
├── S01E02.mp4
└── valid_ranges.json
```

episode 预处理会自动读取源视频旁边的 `valid_ranges.json`。已经生成的视频
默认跳过，可以安全地重新运行批处理命令。

### 处理 clip

处理 `data/*/g1_clip` 中涉及的全部剧：

```bash
bash preprocess/get_all_clips.sh \
  all data videos/processed/clips /path/to/source/videos
```

只处理一个系列，例如 `friends`：

```bash
bash preprocess/get_all_clips.sh \
  friends data videos/processed/clips /path/to/source/videos
```

只处理一个题目文件中引用的 clip：

```bash
PYTHONPATH=. python3 preprocess/prepare_clips.py \
  --questions data/friends/g1_clip/s01e01.json \
  --video-root /path/to/source/videos \
  --out videos/processed/clips
```

只处理该文件中的一道题：

```bash
PYTHONPATH=. python3 preprocess/prepare_clips.py \
  --questions data/friends/g1_clip/s01e01.json \
  --video-root /path/to/source/videos \
  --out videos/processed/clips \
  --qid friends/g1_2_s01e01_q001
```

clip 文件名严格使用题目中的 `video_name`；引用同一规范时间段的题目会复用
同一个视频文件。

### 处理 episode

处理 `data/*/g2_episode` 中涉及的全部剧：

```bash
bash preprocess/get_all_episodes.sh \
  all data /path/to/source/videos videos/processed/episodes
```

只处理一个系列，例如 `friends`：

```bash
bash preprocess/get_all_episodes.sh \
  friends data /path/to/source/videos videos/processed/episodes
```

只处理一集：

```bash
PYTHONPATH=. python3 preprocess/prepare_episodes.py \
  --series friends \
  --video-root /path/to/source/videos \
  --out videos/processed/episodes \
  --eps S01E01
```

episode 采用平铺命名，例如：

```text
videos/processed/episodes/friends_s01e01.mp4
```

本地视频与 ranges 分开存放时，用第五个参数指定 ranges 根目录：

```bash
bash preprocess/get_all_episodes.sh \
  friends data /path/to/source/videos videos/processed/episodes /path/to/ranges/root
```

## 推理

### Gemini

只推理一道 clip 级问题：

```bash
PYTHONPATH=. python3 evaluation/inference/gemini.py \
  --questions data/friends/g1_clip/s01e01.json \
  --clips-dir videos/processed/clips \
  --model gemini-2.5-flash \
  --base-url https://generativelanguage.googleapis.com/v1beta/openai/ \
  --api-key <gemini-api-key> \
  --qid g1_2_s01e01_q001 \
  --out output/friends_s01e01_q001_pred_gemini-2.5-flash.json
```

推理一个系列的全部 clip 级问题：

```bash
PYTHONPATH=. python3 evaluation/inference/gemini.py \
  --questions data/friends/g1_clip/all.json \
  --clips-dir videos/processed/clips \
  --model gemini-2.5-flash \
  --base-url https://generativelanguage.googleapis.com/v1beta/openai/ \
  --api-key <gemini-api-key>
```

推理一个 episode 级问题文件时，使用 `--episodes-dir`：

```bash
PYTHONPATH=. python3 evaluation/inference/gemini.py \
  --questions data/friends/g2_episode/s01e01.json \
  --episodes-dir videos/processed/episodes \
  --model gemini-2.5-flash \
  --base-url https://generativelanguage.googleapis.com/v1beta/openai/ \
  --api-key <gemini-api-key>
```

批量推理某个粒度下的全部系列：

```bash
bash evaluation/inference/run_all.sh \
  gemini all g1_clip \
  --clips-dir videos/processed/clips \
  --model gemini-2.5-flash \
  --base-url https://generativelanguage.googleapis.com/v1beta/openai/ \
  --api-key <gemini-api-key>
```

如需评估 Qwen-Omni，将 `gemini.py` 换成 `qwen_omni.py`，并提供兼容 OpenAI
格式的 Qwen-Omni API 地址。两个后端都支持 `g1_clip` 和 `g2_episode`。

每个粒度目录中的 `all.json` 包含该系列的全部问题。不指定 `--out` 时，结果
会按照系列、粒度、题目文件和模型名称自动写入 `output/`。

每次推理只处理一种粒度。推理代码只读取 `preprocess` 已经生成的视频，不会
在推理过程中裁剪或修改视频。输出保留原始问题，并为每道题增加
`pred_answer` 和 `pred_info`；后者记录原始响应、解析错误和请求错误。

推理默认加入题目字幕，但字幕不包含角色名或说话人标签。只有进行纯视频与
音频消融实验时才使用 `--no-transcript`。常用参数包括：

- `--qid`：只运行指定 qid，可重复传入以选择多道题。
- `--out`：指定预测 JSON 的保存路径。
- `--tries`：每道题的最大请求次数，默认 `3`。
- `--timeout`：单次 API 请求超时秒数，默认 `180`。
- `--thinking on|off|default`：控制模型端推理模式。
- `--force`：即使已有缓存，也重新运行选中的题目。

脚本每完成一道题都会更新输出文件。使用相同命令和输出路径重新运行时，成功
结果会被复用，请求错误会重新尝试。只重跑某个解析错误时，同时传入该题的
`--qid`、`--force` 和原来的 `--out`。

## 评分

闭集答案均在本地确定性评分。只有预测文件中存在已经回答的非 Yes/No 开放题
时，才需要文本 LLM judge。下面使用 DeepSeek-V4-Flash 评估一个完整的
Friends clip 级预测文件：

```bash
PYTHONPATH=. python3 evaluation/run_scoring.py \
  --pred output/friends_g1_clip_all_pred_gemini-2.5-flash.json \
  --out runs/friends_g1_clip_all_gemini-2.5-flash_judged_by_deepseek-v4-flash \
  --model deepseek-v4-flash \
  --base-url https://api.deepseek.com \
  --api-key <deepseek-api-key> \
  --timeout 180
```

如果不需要语义 judge，可省略 `--model`、`--base-url` 和 `--api-key`。judge
只接收问题、金标答案和模型答案，并返回语义等价与否的二值判断。闭集情绪、
前后情绪、单选、排序和 Yes/No 始终使用确定性评分。

每道原始问题只产生一个归一化分数。单选、排序、Yes/No 和经 judge 评估的
开放题记为 0 或 1；情绪多选题采用集合 F1，即
`2TP / (2TP + FP + FN)`；前后情绪题先分别计算两个方向的集合 F1，再在题内
取平均。缺失预测和推理错误贡献 0 分。

累计总分计算方式：

```text
earned_points   = 所有 question_score 之和
possible_points = 问题总数
overall_score   = earned_points / possible_points
```

汇总多个系列时，应分别累加分子和分母，不能直接平均各系列百分比：

```text
all_series_score = sum(series earned_points) / sum(series possible_points)
```

也可以直接合并多个已完成运行的 `metrics.json`：

```bash
jq -s '
  (map(.earned_points) | add) as $earned |
  (map(.possible_points) | add) as $possible |
  {earned_points: $earned, possible_points: $possible,
   overall_score: ($earned / $possible)}
' runs/friends/metrics.json runs/thebigbang/metrics.json
```

评估结果还会提供按任务等权的 `macro_score`，它只用于分析，不参与累计总分。

输出包括 `scores.jsonl`、`scores_closed_emotion.jsonl`、
`scores_questions.jsonl`、`metrics.json`、`prediction_format_issues.json`
和 `summary.md`：

- `summary.md`：累计总分、覆盖率、Accuracy 和情绪 F1 摘要。
- `metrics.json`：完整指标，包括 `earned_points`、`possible_points`、
  `overall_score`、`series_scores`、各任务得分和 `macro_score`。
- `scores_questions.jsonl`：每道原始问题最终进入累计总分的归一化分数。
- `scores.jsonl`：非情绪题的确定性结果或 judge 判断。
- `scores_closed_emotion.jsonl`：情绪标签各 slot 的 precision、recall 和 F1。
- `prediction_format_issues.json`：不符合规定输出格式的预测。
