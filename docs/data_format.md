# Unified question and prediction format

`clip` and `episode` use the same outer fields, with independent question/video numbering. Each question references one prepared video. A flat question directory may contain one JSON file per question; arrays and JSONL are also supported. Video files are named `<video_id>.mp4`.

## Question fields

| Field | Meaning |
|---|---|
| `question_id` | Stable question identifier, unique within the granularity. |
| `video_id` | Stable video identifier; multiple questions can share it. |
| `source` | `{ "from": "source URL or series/season/episode", "segments": [...] }`. |
| `granularity` | `clip` or `episode`. |
| `type` | A task name from the table below. |
| `question` | The question, including any question-specific output requirements. |
| `answer` | Reference answer: labels, before/after label sets, or natural-language text. |
| `answer_details` | Detailed reference content with a description of how to use it, or `null`. |
| `rubric` | Required criterion and score-band descriptions for LLM-scored questions; `null` for emotion-label questions. |
| `subtitles` | Subtitle list, or `null`. This is the last field in displayed examples. |

All ten fields are required. IDs use letters, digits, `_`, `-` and `.` and begin with a letter or digit. Dataset preparation performs legacy renaming; the runtime loader validates this format without changing the source files.

| Granularity | `type` | `answer` | `answer_details` | `rubric` |
|---|---|---|---|---|
| clip | `contextual emotion` | EMOTIC label list | `null` | `null` |
| clip | `emotion transition` | `before` / `after` label lists | `null` | `null` |
| clip | `emotion influence` | EMOTIC label list | `null` | `null` |
| clip, episode | `emotion trajectory` | Text | Required stage list | 0–4 |
| clip | `emotion cause` | Text | Cause factors when necessary, otherwise `null` | 0–3 |
| episode | `emotional intensity comparison` | Text | Alternative valid answers when necessary, otherwise `null` | 0/1 |
| episode | `emotional reasoning` | Text | Reference details when necessary, otherwise `null` | Explanations: 0–3; other results: 0/1 |

For G2 reasoning, every question requires a stored rubric: 0–3 for explanations and 0/1 for results. A provided rubric selects LLM evaluation and defines the allowed scores; `null` selects local label metrics. Reports use the original `type` field without inferring task categories from score ranges.

## Complete example

This synthetic record demonstrates the format; it is not an annotated benchmark sample.

```json
{
  "question_id": "Q000001",
  "video_id": "V000001",
  "source": {
    "from": "Example Series S01E01",
    "segments": [{"start": 100.0, "end": 120.0}]
  },
  "granularity": "clip",
  "type": "emotion transition",
  "question": "What emotions does A show before and after hearing the announcement?",
  "answer": {"before": ["fear"], "after": ["peace"]},
  "answer_details": null,
  "rubric": null,
  "subtitles": [
    {"id": "U1", "t": [108.0, 110.0], "speaker": "B", "text": "Your application has been accepted."}
  ]
}
```

`source.segments` lists intervals in seconds on the original source; their order follows the assembled input. It may be `null` for a whole-source video. Subtitle `t` values also retain source positions. The inference client uses subtitle list order rather than re-sorting by timestamps, since concatenated cuts may revisit earlier source positions. Subtitle `speaker` may be `null`. Neither source metadata nor subtitle timing/speaker labels are sent to tested models.

## Detailed reference content

`answer_details` supplies more specific reference information for judging correctness and completeness. It can unpack necessary content from `answer` or provide alternative valid answers. `description` specifies how the entries should be used; `items` contains the actual entries. It does not assign separate points.

Stage example:

```json
{
  "description": "These stages describe the required emotional process in chronological order. Equivalent wording is accepted.",
  "items": [
    {"id": "S1", "description": "Initially, A is excited.", "anchor": null, "emotion": "excitement", "intensity": null},
    {"id": "S2", "description": "A becomes worried after the announcement.", "anchor": "After the announcement", "emotion": "worry", "intensity": null}
  ]
}
```

Every trajectory stage contains `id`, `description`, `anchor`, `emotion` and `intensity`. Emotion fields contain emotion descriptions only, without the object or event. Null fields add no requirements. A recurring pattern can be described within a stage; the runtime does not impose a stage count. Trajectory descriptions are not restricted to the EMOTIC label vocabulary.

Cause-factor example:

```json
{
  "description": "Both factors are needed to explain the reference answer. Evaluate the explanation as a whole under the rubric.",
  "items": [
    {"id": "R1", "description": "A is upset by B's rejection."},
    {"id": "R2", "description": "A regrets having spoken impulsively."}
  ]
}
```

Alternative-answer example:

```json
{
  "description": "Each entry is an independently acceptable answer. The question asks for one moment; either entry is sufficient.",
  "items": [
    {"id": "C1", "description": "The moment before the final round begins."},
    {"id": "C2", "description": "The moment when the final result is announced."}
  ]
}
```

These are structural examples; actual factors, stages and accepted alternatives must come from verified video content. A single definite answer does not need to be repeated in `answer_details`.

## Rubric

Every LLM-scored question must store its own rubric, including definite-answer results such as names, numbers and Yes/No. The rubric contains `criterion` and a `scores` object whose keys are integer strings and values describe the corresponding bands. The runtime reads only this stored rubric; reserve `null` for locally scored label questions.

Store the approved task rubric in each trajectory/cause record. For result questions, write a 0/1 correctness rubric into the question during annotation, for example:

```json
{
  "criterion": "Determine whether the model answer correctly answers the question.",
  "scores": {
    "0": "The answer is incorrect, incomplete, contradictory, or does not answer the question.",
    "1": "The answer is correct and sufficient. When alternative valid answers are provided and only one is requested, any one is sufficient. Equivalent wording is accepted."
  }
}
```

Trajectory uses keys `0` through `4`; cause explanations use `0` through `3`. Reference entries are assessed jointly according to the rubric, not independently added together. A model response is never allowed to change the scoring instructions.

## Predictions from external methods

Prediction records require `question_id` and an answer in `prediction` or `pred_answer`:

```json
[
  {"question_id": "Q000001", "prediction": {"before": ["fear"], "after": ["peace"]}},
  {"question_id": "Q000002", "pred_answer": "A is first excited, then worried after the announcement."}
]
```

For ordinary label questions use a list such as `["affection", "yearning"]`. Text labels and `Before: ...` / `After: ...` responses are also parsed. Free answers, including numbers and Yes/No, must be strings.

Predictions are matched only by `question_id` to questions selected by `--granularity`. Prediction fields `granularity`, `video_id`, `status` and `error` are ignored. Extra question IDs are ignored; for duplicate IDs, the last record is used.

The evaluator uses `prediction` when non-null and non-blank, otherwise `pred_answer`. When both answer fields are absent, null or blank (including whitespace-only strings), the prediction is missing and does not participate in scoring. A missing record is handled the same way.

Results are grouped directly by `type`, without an additional answer-type field or mapping file. G2 emotional reasoning retains its defined 60% result / 40% explanation aggregate within that task; this calculation does not affect whether a question is sent to the LLM.
