# LongEmoBench

> 中文文档见 [README_zh.md](README_zh.md)。

LongEmoBench is a long-video emotion understanding benchmark built on TV sitcoms.
The current release covers clip/event-level questions: **1,476 questions across 7
series** (Frasier, Friends, How I Met Your Mother, Malcolm in the Middle, Modern
Family, The Big Bang Theory, Will & Grace), each grounded in a specific video
segment with time-aligned subtitles.

This repository contains the full evaluation toolkit: dataset validation, video
preprocessing, inference runners, and the formal scorer.

- **Dataset**: https://huggingface.co/datasets/mcflurryshuoz/LongEmoBench

## 1. Data

```
LongEmoBench (Hugging Face)
├── questions/<series>/grade_1/<episode>.json   # 1,476 questions (149 files)
├── g1_clips/                                   # 1,476 pre-cut question clips (source quality)
└── original_videos/<series>/<episode>.mp4      # original episodes
```

Download:

```bash
pip install -U huggingface_hub
hf download mcflurryshuoz/LongEmoBench --repo-type dataset --local-dir data
```

You normally only need `questions/` + `g1_clips/`: every question's input video is
already cut and named deterministically, so inference finds each question's clip by
name. `original_videos/` is only needed if you want to re-cut clips yourself.

### Question schema

```json
{
  "series": "friends",
  "qid": "g1_2_S01E02_q001",
  "question_type": "single_choice | multi_select | ranking | open_ended",
  "question": "... ends with the canonical answer-format instruction ...",
  "options": ["A. happy", "B. sad", "C. angry"],
  "answer": "A",
  "emotion_answer": true,
  "input_video": {"scope_kind": "event", "eps": ["S01E02"], "segments": [{"ep": "S01E02", "start": 54.9, "end": 82.6}]},
  "input_transcript": [{"ep": "S01E02", "t": [55.0, 57.2], "text": "..."}]
}
```

Gold answers by type: `single_choice` one option letter; `multi_select` a list of
letters; `ranking` an ordered list of letters (an option may repeat); `open_ended`
with `emotion_answer: true` an open-vocabulary emotion object — `{"emotion": [...]}`
or `{"from_emotion": [...], "to_emotion": [...]}` for emotion transitions; other
`open_ended` a short free-text string.

**Question identity is `series/qid`** — bare qids repeat across series, so every
index (clip lookup, prediction cache) uses the combined key.

Validate any question file (or the whole `questions/` tree) against the format
contract:

```bash
python3 benchmark/preprocess/check_question_contract.py --questions data/questions
```

## 2. Setup

```bash
pip install -r requirements.txt     # openai + huggingface_hub
```

Python 3.10+. `ffmpeg` is required only if you re-cut clips or use streaming mode.

## 3. Inference

Two reference runners are included; both talk to any OpenAI-compatible endpoint and
support resume (re-running skips answered questions).

```bash
# Qwen-Omni (or any OpenAI-compatible video model)
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

The prompt contains a fixed response contract plus the question itself (options
included; `--with-transcript` adds the time-aligned subtitles without speaker
labels). The model must reply with JSON `{reason, answer}`; the answer format is
specified by each question's own final instruction.

Output goes to `output/<series>_<episode>_pred_<model>.json`: the question items
plus, per answered item, `pred_answer` and a `pred_info: {model, reason, raw,
error}` block.

### Evaluating your own model

You don't have to use the provided runners. Produce a prediction file that mirrors
the question list and adds `pred_answer` per item (a `pred_info` block is optional),
then run the scorer on it:

- single-choice: `"B"` &nbsp;·&nbsp; multi-select: `"B D"` &nbsp;·&nbsp; ranking: `"A B C"`
- open emotion: `{"emotion": ["hurt"]}` or `{"from_emotion": ["anxious"], "to_emotion": ["hurt"]}`
- other open questions: a short phrase string

## 4. Scoring

```bash
python3 benchmark/evaluation/run_scoring.py \
  --pred output/friends_S01E01_pred_<model>.json \
  --out output/scoring_<model> \
  --judge-api-key "$JUDGE_API_KEY"
```

Metrics (three tracks):

| Track | Questions | Metric |
|---|---|---|
| Accuracy | single_choice, ranking, Yes/No, judged open QA | accuracy (ranking requires the exact gold sequence; Yes/No is deterministic; remaining open QA gets a binary LLM-judge verdict) |
| Multi-select | multi_select | option-set precision / recall / F1 |
| Open emotion | `emotion_answer: true` | OV-MER-style GPT grouping, then emotion-cluster P/R/F1; a transition counts as two slots (from / to) |

The judge (grouping + open-answer verdicts) is any OpenAI-compatible model —
default `deepseek-v4-flash` @ `https://api.deepseek.com`; override with
`--judge-model/--judge-base-url/--judge-api-key` or `JUDGE_MODEL/JUDGE_BASE_URL/
JUDGE_API_KEY` env vars. Reference: [OV-MER](https://arxiv.org/abs/2410.01495).

Outputs in `--out`: `summary.md` (human-readable), `metrics.json` (overall + by
question type / series / emotion slot), `scores.jsonl` (per-question records),
`gpt_grouping.json` (emotion clusters), plus gold/prediction format issue reports.
The pred file itself is never modified. Unanswered or errored questions are
reported as unscored — they never count against accuracy.

**Reproducibility protocol**: LLM-based emotion grouping is not deterministic, so
generate the grouping once, review `gpt_grouping.json`, then pin it for every model
with `--reuse-grouping path/to/gpt_grouping.json`. Re-runs against the same `--out`
directory reuse its grouping automatically, so repeated scoring is bit-identical.

## 5. Re-cutting clips (optional)

Pre-cut clips ship with the dataset. To re-cut from the original episodes
(time-exact re-encode, source resolution/fps/audio kept):

```bash
python3 benchmark/preprocess/prepare_clips.py \
  --questions data/questions/friends/grade_1/S01E01.json \
  --video-root data/original_videos \
  --out data/g1_clips

# every series in one go
benchmark/get_all_clips.sh all data/questions data/g1_clips data/original_videos
```

`benchmark/preprocess/prepare_episodes.py` additionally builds effective full
episodes (opening/credits removed) for the upcoming episode-level granularities.

## Repository layout

```
benchmark/
├── common/        # shared: I/O + question keys, format contract, ffmpeg helpers
├── preprocess/    # dataset contract check, clip cutting, effective episodes
├── inference/     # prompt assembly + Qwen-Omni / Gemini runners
└── evaluation/    # formal scorer, OV-MER emotion grouping, LLM judge
```
