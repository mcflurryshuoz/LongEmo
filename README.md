# LongEmoBench

[中文](README_zh.md) · [Dataset on Hugging Face](https://huggingface.co/datasets/mcflurryshuoz/LongEmoBench/tree/main)

LongEmoBench evaluates emotion understanding at two video granularities: **clip** and **episode**. Questions cover emotion recognition, transitions, trajectories, causes, intensity comparisons, and reasoning across a longer video. All tasks use open-ended questions; the expected answer may be emotion labels, a short result, or a natural-language explanation.

This repository provides prediction generation and a shared evaluator. Predictions from your own model or agent can be evaluated directly. **Evaluation requires question annotations and predictions; it does not load videos or subtitles.**

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
