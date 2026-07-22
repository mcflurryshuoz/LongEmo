# LongEmoBench

> English version: [README.md](README.md)。

LongEmoBench 是一个基于电视情景剧的长视频情感理解基准。当前发布覆盖
clip/event 粒度：**7 部剧共 1,476 道题**（Frasier、Friends、How I Met Your
Mother、Malcolm in the Middle、Modern Family、The Big Bang Theory、Will &
Grace），每道题都锚定在具体的视频片段上，并配有时间对齐的字幕。

本仓库是完整的评测工具链：数据校验、视频预处理、推理脚本、正式评分器。

- **数据集**：https://huggingface.co/datasets/mcflurryshuoz/LongEmoBench

## 1. 数据

```
LongEmoBench（Hugging Face）
├── questions/<剧名>/grade_1/<集号>.json   # 1,476 道题（149 个文件）
├── g1_clips/                              # 1,476 个预切好的题目片段（原画质）
└── original_videos/<剧名>/<集号>.mp4       # 原始整集视频
```

下载：

```bash
pip install -U huggingface_hub
hf download mcflurryshuoz/LongEmoBench --repo-type dataset --local-dir data
```

通常只需要 `questions/` + `g1_clips/`：每道题的输入视频都已切好，文件名是确定性
的，推理按名字直接找到对应片段。`original_videos/` 仅在你想自己重切片段时需要。

### 题目 schema

```json
{
  "series": "friends",
  "qid": "g1_2_S01E02_q001",
  "question_type": "single_choice | multi_select | ranking | open_ended",
  "question": "...（以规范的答案格式指令收尾）...",
  "options": ["A. happy", "B. sad", "C. angry"],
  "answer": "A",
  "emotion_answer": true,
  "input_video": {"scope_kind": "event", "eps": ["S01E02"], "segments": [{"ep": "S01E02", "start": 54.9, "end": 82.6}]},
  "input_transcript": [{"ep": "S01E02", "t": [55.0, 57.2], "text": "..."}]
}
```

各题型金标：`single_choice` 单个选项字母；`multi_select` 字母列表；`ranking`
有序字母列表（选项可重复出现）；`open_ended` 且 `emotion_answer: true` 为开放
词表情感对象——`{"emotion": [...]}` 或转折题的
`{"from_emotion": [...], "to_emotion": [...]}`；其余 `open_ended` 为短文本。

**题目唯一标识是 `series/qid`**——裸 qid 会跨剧重复，所有索引（找片、预测缓存）
都用复合键。

校验任意题目文件（或整个 `questions/` 目录）是否符合格式契约：

```bash
python3 benchmark/preprocess/check_question_contract.py --questions data/questions
```

## 2. 环境

```bash
pip install -r requirements.txt     # openai + huggingface_hub
```

Python 3.10+。`ffmpeg` 仅在重切片段或使用流式模式时需要。

## 3. 推理

内置两个参考推理脚本，均对接任意 OpenAI 兼容端点，支持断点续跑（重跑自动跳过
已答的题）。

```bash
# Qwen-Omni（或任何 OpenAI 兼容的视频模型）
python3 benchmark/inference/qwen_omni.py \
  --questions data/questions/friends/grade_1/S01E01.json \
  --clips-dir data/g1_clips \
  --base-url "$BASE_URL" --api-key "$API_KEY" \
  --model qwen3-omni-30b-a3b-instruct \
  --with-transcript

# Gemini
export GEMINI_API_KEY=...
python3 benchmark/inference/gemini.py \
  --questions data/questions/friends/grade_1/S01E01.json \
  --clips-dir data/g1_clips \
  --model gemini-2.5-flash \
  --with-transcript
```

Prompt 由固定的响应契约加题目本身组成（选项随题发送；`--with-transcript` 附带
时间对齐字幕，不含说话人标签）。模型须回复 JSON `{reason, answer}`，答案格式由
每道题自己的收尾指令规定。

输出写到 `output/<剧名>_<集号>_pred_<模型>.json`：题目列表原样复制，每道已答的
题追加 `pred_answer` 和 `pred_info: {model, reason, raw, error}` 块。

### 评测你自己的模型

不必使用内置推理脚本。只要产出一个"题目列表 + 每题 `pred_answer`"的预测文件
（`pred_info` 可选），即可直接交给评分器：

- 单选：`"B"` ；多选：`"B D"` ；排序：`"A B C"`
- 情感开放题：`{"emotion": ["hurt"]}` 或 `{"from_emotion": ["anxious"], "to_emotion": ["hurt"]}`
- 其他开放题：一个短语字符串

## 4. 评分

```bash
python3 benchmark/evaluation/run_scoring.py \
  --pred output/friends_S01E01_pred_<模型>.json \
  --out output/scoring_<模型> \
  --judge-api-key "$JUDGE_API_KEY"
```

指标（三条轨道）：

| 轨道 | 题目 | 指标 |
|---|---|---|
| Accuracy | 单选、排序、Yes/No、裁判判定的开放题 | 准确率（排序须与金标序列完全一致；Yes/No 确定性判分；其余开放题由 LLM 裁判给二值判定） |
| Multi-select | 多选 | 选项集合的 precision / recall / F1 |
| Open emotion | `emotion_answer: true` | OV-MER 式 GPT 语义分组后按情感簇算 P/R/F1；转折题拆 from/to 两个槽位分别计分 |

裁判（分组 + 开放题判定）可用任意 OpenAI 兼容模型——默认
`deepseek-v4-flash` @ `https://api.deepseek.com`；用
`--judge-model/--judge-base-url/--judge-api-key` 或
`JUDGE_MODEL/JUDGE_BASE_URL/JUDGE_API_KEY` 环境变量覆盖。参考：
[OV-MER](https://arxiv.org/abs/2410.01495)。

`--out` 产物：`summary.md`（人读摘要）、`metrics.json`（总体 + 按题型/剧/情感
槽位细分）、`scores.jsonl`（逐题明细）、`gpt_grouping.json`（情感簇），以及金标
/预测格式问题报告。pred 文件本身不会被修改。未作答或出错的题记为"未评"，不计入
准确率分母。

**可复现规程**：LLM 情感分组有随机性——第一次评分生成分组后，人工检查
`gpt_grouping.json`，之后所有模型统一用 `--reuse-grouping` 指向这份文件。同一
`--out` 目录重跑会自动复用自己的分组，评分结果逐位一致。

## 5. 重切片段（可选）

数据集已附带预切片段。如需从原片重切（时间精确重编码，保持原分辨率/帧率/声道）：

```bash
python3 benchmark/preprocess/prepare_clips.py \
  --questions data/questions/friends/grade_1/S01E01.json \
  --video-root data/original_videos \
  --out data/g1_clips

# 全部剧一把跑
benchmark/get_all_clips.sh all data/questions data/g1_clips data/original_videos
```

`benchmark/preprocess/prepare_episodes.py` 另可生成去掉片头/片尾的有效整集视频，
供后续整集级粒度使用。

## 仓库结构

```
benchmark/
├── common/        # 共享：I/O 与题目键、格式契约、ffmpeg 工具
├── preprocess/    # 数据契约校验、切片、有效整集
├── inference/     # prompt 组装 + Qwen-Omni / Gemini 推理脚本
└── evaluation/    # 正式评分器、OV-MER 情感分组、LLM 裁判
```
