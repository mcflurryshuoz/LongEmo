# LongEmoBench

LongEmoBench is a benchmark for emotion understanding in long videos. This
directory provides question data, conventions for source and processed videos,
video preprocessing scripts, and code for model inference and scoring.

```text
LongEmoBench/
├── data/                         # benchmark questions
│   └── <series>/
│       ├── g1_clip/              # clip-level questions
│       └── g2_episode/           # episode-level questions
├── subtitles/                    # dialogue-only episode subtitles
│   └── <series>/
│       └── <episode>.json
├── videos/
│   ├── sources/<series>/         # source episodes and valid_ranges.json
│   └── processed/
│       ├── clips/                # media used by g1_clip questions
│       └── episodes/             # episodes with opening/credits removed
├── preprocess/                   # video preprocessing
└── evaluation/                   # inference and scoring
    └── inference/
```

## Video Granularities

Questions live under `LongEmoBench/data/<series>/g1_clip` and
`LongEmoBench/data/<series>/g2_episode`.

`g1_clip` is the clip-level setting. Each question provides one short video
clip relevant to the question and evaluates emotion recognition and emotional
reasoning within a local scene.

`g2_episode` is the episode-level setting. Each question provides a complete
episode with its opening and credits removed and evaluates the ability to
analyze emotional changes across scenes and integrate emotional information
over a long temporal span.

## Subtitles

Dialogue-only subtitles are provided under `subtitles/<series>`. Speaker labels
embedded in the subtitle text, stage directions, sound-effect captions, and
subtitle credits have been removed so that non-verbal descriptions do not leak
visual or audio evidence to the model.

Each `<series>/<episode>.json` stores `series`, `ep`, `unit_id`, `t`, `speaker`,
and `text`, where `t` is the `[start, end]` interval in seconds.
`speaker` identifies the character when available and is `unknown` when the
source annotation does not identify one reliably.

Regenerate all released subtitles from the reviewed `s1_perception` files:

```bash
PYTHONPATH=. python3 preprocess/prepare_subtitles.py \
  --inputs-root /path/to/vebench/outputs \
  --out subtitles
```

Use `--series friends` or `--episode S01E01` to restrict an export. Both
options may be repeated.

## Preprocessing

Source videos can be downloaded from the Hugging Face
[`videos/sources`](https://huggingface.co/datasets/mcflurryshuoz/LongEmoBench/tree/main/videos/sources)
directory. Replace `/path/to/source/videos` in the commands below with the
local download location.

Hugging Face also provides ready-to-use `clips` and `episodes` under
[`videos/processed`](https://huggingface.co/datasets/mcflurryshuoz/LongEmoBench/tree/main/videos/processed).
Download these files directly for inference. Run the preprocessing commands
below only when the inputs need to be regenerated from the source videos.

The expected source layout is:

```text
/path/to/source/videos/<series>/
├── S01E01.mp4
├── S01E02.mp4
└── valid_ranges.json
```

`valid_ranges.json` is used by episode preprocessing and is discovered beside
the source videos automatically. Existing processed videos are skipped.

### Prepare clips

Prepare clips for every series represented under `data/*/g1_clip`:

```bash
bash preprocess/get_all_clips.sh \
  all data videos/processed/clips /path/to/source/videos
```

Prepare clips for one series:

```bash
bash preprocess/get_all_clips.sh \
  friends data videos/processed/clips /path/to/source/videos
```

Prepare every clip referenced by one question file:

```bash
PYTHONPATH=. python3 preprocess/prepare_clips.py \
  --questions data/friends/g1_clip/s01e01.json \
  --video-root /path/to/source/videos \
  --out videos/processed/clips
```

Prepare only one question from that file by adding its qid:

```bash
PYTHONPATH=. python3 preprocess/prepare_clips.py \
  --questions data/friends/g1_clip/s01e01.json \
  --video-root /path/to/source/videos \
  --out videos/processed/clips \
  --qid friends/g1_2_s01e01_q001
```

Clip filenames come from each question's exact `video_name`. Questions that
share one canonical interval reuse the same file.

### Prepare episodes

Prepare effective episodes for every series represented under
`data/*/g2_episode`:

```bash
bash preprocess/get_all_episodes.sh \
  all data /path/to/source/videos videos/processed/episodes
```

Prepare every effective episode for one series:

```bash
bash preprocess/get_all_episodes.sh \
  friends data /path/to/source/videos videos/processed/episodes
```

Prepare one episode only:

```bash
PYTHONPATH=. python3 preprocess/prepare_episodes.py \
  --series friends \
  --video-root /path/to/source/videos \
  --out videos/processed/episodes \
  --eps S01E01
```

Outputs use flat `<out>/<series>_<episode>.mp4` names, such as
`friends_s01e01.mp4`.

For a local layout where videos and range metadata have different roots, pass
the ranges root as the fifth argument to `get_all_episodes.sh`:

```bash
bash preprocess/get_all_episodes.sh \
  friends data /path/to/source/videos videos/processed/episodes /path/to/ranges/root
```

## Inference

### Gemini

Run one clip-level question:

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

Run every clip-level question in one series:

```bash
PYTHONPATH=. python3 evaluation/inference/gemini.py \
  --questions data/friends/g1_clip/all.json \
  --clips-dir videos/processed/clips \
  --model gemini-2.5-flash \
  --base-url https://generativelanguage.googleapis.com/v1beta/openai/ \
  --api-key <gemini-api-key>
```

Run one episode-level question file by replacing `--clips-dir` with
`--episodes-dir`:

```bash
PYTHONPATH=. python3 evaluation/inference/gemini.py \
  --questions data/friends/g2_episode/s01e01.json \
  --episodes-dir videos/processed/episodes \
  --model gemini-2.5-flash \
  --base-url https://generativelanguage.googleapis.com/v1beta/openai/ \
  --api-key <gemini-api-key>
```

Run every available series for one granularity:

```bash
bash evaluation/inference/run_all.sh \
  gemini all g1_clip \
  --clips-dir videos/processed/clips \
  --model gemini-2.5-flash \
  --base-url https://generativelanguage.googleapis.com/v1beta/openai/ \
  --api-key <gemini-api-key>
```

Use `qwen_omni.py` instead of `gemini.py`, together with an OpenAI-compatible
Qwen-Omni endpoint, to evaluate Qwen-Omni. The two backends support both video
granularities.

Each granularity directory contains an `all.json` file with every question for
that series. If `--out` is omitted, inference writes to `output/` using a name
derived from the series, granularity, question file, and model.

Each inference run processes one granularity. Inference only consumes media
produced by `preprocess`; it never cuts or modifies videos. The inference output
preserves every question and adds `pred_answer` plus `pred_info`, including
granularity, active template, raw response, parse errors, and transport errors.

Subtitles are included by default without character names or speaker labels.
Use `--no-transcript` only for a video-and-audio-only ablation. Useful runtime
options are:

- `--qid`: run only the specified qid; repeat it to select multiple questions.
- `--out`: choose the prediction JSON path.
- `--tries`: maximum request attempts per question; default `3`.
- `--timeout`: timeout in seconds for each API request; default `180`.
- `--thinking on|off|default`: control provider-side reasoning when supported.
- `--force`: re-run selected questions even when a cached prediction exists.

The output is rewritten after each question. Re-running the same command resumes
from that file: successful predictions are reused and transport errors are
retried. To retry a parse error without re-running the full file, combine its
`--qid` with `--force` and keep the same `--out`.

## Scoring

Closed answers are scored locally. A text-only LLM judge is required only when
the prediction file contains answered open-ended questions other than Yes/No.
For example, score a complete Friends clip-level run with DeepSeek-V4-Flash:

```bash
PYTHONPATH=. python3 evaluation/run_scoring.py \
  --pred output/friends_g1_clip_all_pred_gemini-2.5-flash.json \
  --out runs/friends_g1_clip_all_gemini-2.5-flash_judged_by_deepseek-v4-flash \
  --model deepseek-v4-flash \
  --base-url https://api.deepseek.com \
  --api-key <deepseek-api-key> \
  --timeout 180
```

If no semantic judge is needed, omit `--model`, `--base-url`, and `--api-key`.
The judge receives only the question, gold answer, and model answer, and returns
a binary semantic-equivalence verdict. Closed emotion labels, before/after
labels, single choice, ranking, and Yes/No are always scored deterministically.

Every original question receives one normalized score. Single-choice, ranking,
Yes/No, and judged open answers receive 0 or 1. Multi-label emotion answers use
set F1, `2TP / (2TP + FP + FN)`. Before/after questions average the two
directional set-F1 scores. Missing predictions and inference errors receive 0.

The cumulative benchmark score is:

```text
earned_points   = sum(question_score)
possible_points = number of questions
overall_score   = earned_points / possible_points
```

Combine series by adding their points, not by averaging their percentages:

```text
all_series_score = sum(series earned_points) / sum(series possible_points)
```

The same calculation can be applied directly to completed run metrics:

```bash
jq -s '
  (map(.earned_points) | add) as $earned |
  (map(.possible_points) | add) as $possible |
  {earned_points: $earned, possible_points: $possible,
   overall_score: ($earned / $possible)}
' runs/friends/metrics.json runs/thebigbang/metrics.json
```

`macro_score` is also reported as a task-balanced diagnostic and is not used in
the cumulative score.

Outputs include `scores.jsonl`, `scores_closed_emotion.jsonl`,
`scores_questions.jsonl`, `metrics.json`, `prediction_format_issues.json`, and
`summary.md`:

- `summary.md`: headline cumulative score, coverage, accuracy, and emotion F1.
- `metrics.json`: all aggregates, including `earned_points`, `possible_points`,
  `overall_score`, `series_scores`, task scores, and `macro_score`.
- `scores_questions.jsonl`: one normalized score per original question.
- `scores.jsonl`: deterministic or judge verdicts for non-emotion questions.
- `scores_closed_emotion.jsonl`: per-slot label precision, recall, and F1.
- `prediction_format_issues.json`: outputs that violate the required format.
