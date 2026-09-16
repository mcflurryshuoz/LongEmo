# LongEmo emotional memory method

This API-based implementation builds question-independent audiovisual memories once per video, then answers each question from a fixed snapshot. It exports the benchmark's existing `question_id` / `prediction` JSONL interface. No model training or local GPU is required.

The first implementation combines localization and emotional perception in one model request per window. It is the joint-perception candidate from the plan, not a reproduction of the old three-call-per-window pipeline. Default windows are 20 seconds, with 2 seconds of context, 1 fps and at most 24 frames; audio is opt-in. Subtitles retain their timestamps and unknown speakers remain unknown.

## Memory and retrieval

- Entities have stable IDs and appearance descriptions. Observations record concrete cues, subjects, absolute time spans, modality and source windows.
- Events own their emotion states. A state describes its subject, emotional target, emotion, observed intensity, uncertainty, optional appraisal and observation references. There is no independently mutable duplicate state table.
- The model may explicitly extend the same ongoing occurrence. Similar repeated events are not automatically merged. Typed relations require valid event endpoints and evidence references.
- Optional `--allow-revisions` supports optimistic state versions, revision history and conservative invalidation of dependent event summaries/relations. A correction preserves the original state's time span and does not create a new transition.
- Retrieval uses BM25 over the same event/observation/relationship content in both conditions. `flat` takes lexical top-k; `graph` expands typed relations and provides temporal coverage plus a lightweight state timeline for global questions. This is an in-memory graph over JSON checkpoints, with no external graph service or embedding model.
- Answering never receives `answer`, `answer_details` or `rubric`. It can inspect a bounded original media interval if enabled. Inspected media and answer changes are question-local; the saved shared graph is immutable. Question-local inspection currently updates the answer evidence, not the persistent graph's state schema.

## Setup

Python 3.10+, FFmpeg 5+ (tested with 7.0.2) and ffprobe are required. API inference and retrieval use the Python standard library. On a host with an old system FFmpeg, an isolated option is:

```bash
python -m venv /ABS/runtime/venv
/ABS/runtime/venv/bin/pip install imageio-ffmpeg==0.6.0
```

Place the package's bundled FFmpeg executable on PATH, alongside the system ffprobe. Keep `TMPDIR`, downloads and API credentials outside the repository, on a volume with sufficient free space. No credentials are included in the examples, Git configuration, manifests or logs.

## Prepare a pinned dataset

```bash
python -m methods.longemo.prepare \
  --output /ABS/runtime/data \
  --token-file /ABS/private/hf_token \
  --pilot-videos 3 --workers 4
```

The first run resolves one HF revision; resumes reuse that revision. File sizes and SHA256 are checked; available LFS hashes are validated. All question metadata is downloaded. Pilot videos are selected by `sha256(seed + video_id)`, retaining every question of each selected video. No reference answer, score or perceived difficulty affects selection. A pilot is a development set, not a new official split or a full benchmark result.

Outputs include the pinned manifest, all questions, `pilot_questions.json`, selected videos and `prepared_subtitles/<video_id>.json`. The latter converts SRT into the benchmark subtitle schema without inventing speaker labels. `--all-videos` downloads the entire release; `--metadata-only` skips videos. This utility does not bypass dataset access controls.

## Run the two stages

Use existing `MODEL_API_KEY` and `MODEL_BASE_URL` environment variables for the API, with an explicit model ID. The shared adapters support provider-native Gemini requests when the base URL ends in `/v1beta`.

```bash
python -m methods.longemo build \
  --data-path /ABS/runtime/data/pilot_questions.json \
  --videos-dir /ABS/runtime/data/episode/videos \
  --subtitles-dir /ABS/runtime/data/prepared_subtitles \
  --output-dir /ABS/runtime/run/memory \
  --model gemini-2.5-flash --thinking off \
  --with-audio --workers 3 --max-tokens 8192

python -m methods.longemo answer \
  --data-path /ABS/runtime/data/pilot_questions.json \
  --memory-dir /ABS/runtime/run/memory \
  --plans-dir /ABS/runtime/run/shared_plans \
  --output-dir /ABS/runtime/run/graph \
  --model gemini-2.5-flash --thinking off \
  --retrieval graph --workers 2
```

Set `--retrieval flat` and a separate output directory for the comparison. A shared plans directory freezes the question planner across retrieval variants. Optional `--max-inspections 1 --inspection-seconds 60` also requires the matching `--videos-dir`; set `--with-audio` and `--subtitles-dir` to provide those inspection inputs. Exact source hashes are checked before reopening videos.

Each window commits atomically after validation. Successful windows/questions resume; failed questions retry. Configuration fingerprints cover model options, actual source hashes, prompts/code and relevant shared evaluator/adapter code. A mismatch requires a new output directory. Do not edit implementation files while a run is active.

## End-to-end pilot and scoring

```bash
python -m methods.longemo.experiment \
  --data-root /ABS/runtime/data \
  --output-dir /ABS/runtime/runs/pilot \
  --credential-file /ABS/private/api_config.json \
  --model gemini-2.5-flash --judge-model gemini-2.5-pro \
  --direct-baseline --with-inspection
```

The convenience experiment command currently uses the project's Gemini proxy resource JSON (`GEMINI_API_KEY` and `GOOGLE_GEMINI_BASE_URL`). The lower-level stage CLIs support the shared provider adapters/environment variables. Model names must be validated against the actual provider; these names document the initial g450 pilot, not a promise of future availability.

The experiment runs flat, graph, optionally graph-plus-inspection and direct 128-frame audiovisual/subtitle inference, then uses `evaluation.eval` with the same judge. A complete, unique, nonempty text prediction is required for every frozen question before scoring. The direct condition has different media sampling and repeated per-question media costs: it is a system comparison, not an equal-media-budget ablation. Flat/graph have the same character cap but may use different actual context lengths.

Official score formulas and judge prompts remain unchanged. The scorer now rejects duplicate prediction IDs and records its configuration/source hashes. Missing/failed results retain the benchmark's original unscored semantics; compare complete coverage only. The pilot orchestrator keeps completed judge outputs when resuming an unchanged experiment, avoiding unnecessary paid rescoring.

Artifacts:

```text
experiment_manifest.json, status.json
memory/<video_id>/manifest.json, memory.json, windows/, calls.jsonl, usage.json
shared_plans/<question_id>.json, calls/
flat|graph|graph_inspect|direct/predictions.jsonl
flat|graph|graph_inspect/traces/, calls/
<method>/scores/run_*/scores.jsonl, metrics.json, evaluation_config.json, summary.md
comparison.json, comparison.md
```

Call ledgers include every returned attempt's token usage and API latency, including JSON-repair retries. Missing usage is explicitly counted; no RMB price is inferred from a proxy alias. Construction and shared-planning costs should be amortized consistently across methods. Do not treat cached planning as a zero-cost capability of one condition.

## Validation and current limits

```bash
python -m unittest discover -s methods/longemo/tests -v
```

Tests cover atomic commits, evidence references, temporal/modality validation, continuation, correction history, stale cache rejection, subtitle conversion, retrieval budget and mutation safety, gold-field isolation, duplicate submissions, Agentic startup/cost totals, real frame PTS and delayed-audio alignment.

The method still depends on the perception model's identity matching and completeness; no face tracker, calibrated cross-scene intensity model or learned retrieval policy is included. Long windows and paraphrases can omit subtle expressions. Revision support is opt-in and not itself evidence of improved accuracy. A nine-question pilot cannot establish statistical superiority or generalization. Preserve all failed runs and report coverage, model versions, actual input budgets and costs alongside scores.
