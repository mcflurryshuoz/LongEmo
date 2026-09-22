# LongEmo event memory and hybrid graph retrieval

Build question-independent event memories once per video, then retrieve evidence from a fixed graph snapshot. The [method overview](../../README.md#zyf-分支当前方法整体流程) describes event graphs and semantic/embedding retrieval; the [current method results](../../experiments/zyf/results/current_method/report.md) report overall, task, and series scores with coverage. GPT-6 performs planning, answering, and official judging; Gemini Embedding 2 supplies dense retrieval. Perception models, providers, and inherited window configurations remain recorded in the [question-level provenance](../../experiments/zyf/results/current_method/report.json). Runtime changes and early protocols are archived in the [experiment history](../../experiments/zyf/progress_history.md).

The GPT-6 perception/OpenRouter commands below document the earlier pilot configuration, not E09. No local GPU or training is needed for the API pipeline; NumPy, Python 3.10+, FFmpeg 5+ and ffprobe are required. E5 remains an explicit optional local diagnostic backend, never a silent fallback.

## Evidence and retrieval

Events own emotion states, observations, absolute time spans and provenance. The current schema v2 also records `event_type`, concrete `actions`, `objects`, `participants`, explicit comparison `signals` (score/count/ordinal/superlative/quote/label) and a bounded confidence value. Signals are copied only when present in the media or subtitles; the perception model must not invent numeric intensity. States distinguish person, emotional target, emotion, observed intensity, appraisal and uncertainty. Explicit continuation joins the same occurrence; repeated similar actions are not automatically merged. Optional corrections retain historical versions and invalidate dependent summaries/relations. Schema v1 memories remain readable, but new fields require rebuilding perception.

The semantic route uses question terms and planner expansions, BM25, person identity/appearance, event actions/objects/signals and emotional target matches. The embedding route indexes the complete event's states and observations in overlapping chunks, using Gemini Embedding 2's asymmetric search prefixes and 3072-dimensional normalized vectors. Each event receives its highest chunk similarity. Top-12 from each route are combined by reciprocal rank fusion (constant 60), then expanded through event relations and adjacent events involving the same person. Global questions also receive temporal coverage and a lightweight timeline. Navigation links are never asserted as causes. A 48,000-character evidence cap applies, with omitted evidence and per-route scores recorded.

The planned `noevent` ablation removes the event layer entirely: perception stores only validated time-window records, observations and media provenance; no event/state/relationship objects are created. Its answer stage embeds and retrieves window records with the same planner, GPT-6, Embedding 2, question set, sampling budget and judge. It is a retrieval ablation, so it must use separate caches and manifests and report coverage and failures independently.

GPT-6 does not accept raw audio. The independent audio observer records timestamped speech/tone/pauses/laughter without seeing any benchmark question. GPT-6 receives those fallible observations alongside sampled video frames and subtitles. Audio identity is explicitly uncertain. The direct baseline receives 128 frames, subtitles and the identical audio-only observations; it never receives graph states or summaries. This is a system comparison, not a matched visual-budget ablation.

Neither perception nor retrieval/answering receives gold answers or rubric fields. Inspection, when enabled, is question-local and cannot modify the shared memory. Every window validates atomically. Failed requests and schema repairs remain in attempt/cost ledgers.

## Prepare and run (earlier GPT-6/OpenRouter pilot)

Use a private credential JSON outside the repository containing the already-provisioned `OPENROUTER_API_KEY`; the key is never placed in commands or result manifests. Dataset access uses a separate authorized HF token file.

```bash
python -m methods.longemo.prepare --output /ABS/runtime/data \
  --token-file /ABS/private/hf_token --pilot-videos 3 --workers 4

python -m methods.longemo.experiment \
  --data-root /ABS/runtime/data --output-dir /ABS/runtime/runs/gpt6_hybrid_v1 \
  --credential-file /ABS/private/api_config.json \
  --model openai/gpt-6-astra --judge-model openai/gpt-6-astra \
  --audio-model google/gemini-3.8-flash \
  --embedding-model google/gemini-embedding-2 \
  --embedding-cache-dir /ABS/runtime/cache/gemini-embedding-2 \
  --direct-baseline --with-inspection
```

The dataset preparer pins one HF revision and validates file size/SHA256 and available LFS hashes. The pilot deterministically hashes `seed + video_id` and keeps every question of the selected videos. Gold labels and scores do not affect selection. `--all-videos` downloads the full release.

The experiment uses 20-second windows plus 2-second context, 1 fps, at most 24 frames per window, 200704 pixels/frame, subtitles and source audio. GPT-6 uses medium reasoning with no unsupported sampling parameters; max output 8192. Audio uses low reasoning. The optional graph inspection has one request and 60 seconds per question. The same GPT-6 judge and original rubric/score formulas apply to all methods.

Individual stage commands are `python -m methods.longemo build ...` and `python -m methods.longemo answer ...`; see `--help`. For GPT-6 audio input, `--with-audio --audio-model google/gemini-3.8-flash` is required. `answer --retrieval graph` requires working dense embeddings and fails explicitly otherwise. `--retrieval direct` selects the shared-audio baseline. Legacy `flat` is only a semantic diagnostic. Shared `--plans-dir` freezes planning between graph and graph-plus-inspection. `--memory-dir` on the experiment command reuses a complete frozen graph.

Put FFmpeg on PATH and temporary files/cache/downloads on a roomy volume; do not use a full system disk. `--embedding-backend local --embedding-model intfloat/multilingual-e5-base` explicitly selects the pinned CPU E5 backend and requires Torch/Transformers. Primary API embedding needs only NumPy.

## Artifacts and reproducibility

Run artifacts include data/code/model manifests; per-video memories, window outputs and audio observations; all API ledgers; embedding vectors and content-addressed indexes; per-question plans, dual-recall rankings, graph evidence, inspection traces and predictions; official score details, coverage, metrics and comparisons.

Configuration changes require a fresh run directory. Source hashes identify code, while documentation-only Git commits may resume the same source configuration. Do not edit source files during a run. Stable remote model aliases do not expose immutable weight snapshots: manifests record the alias, returned model, provider, request settings and exact cached vectors; this limitation is explicit.

All unique predictions must be nonempty text with 100% coverage before scoring. Original scoring semantics are preserved: missing/error questions are unscored, so incomplete scores are not comparable. Judge and ordinary prompts/score formulas are unchanged. The scorer additionally rejects duplicate prediction IDs and records configuration hashes. Costs reported by OpenRouter are stored as provider-reported USD; missing usage/cost attempts are counted. Shared perception, audio, indexing and planning costs must be amortized consistently.

```bash
python -m unittest discover -s methods/longemo/tests -v
```

Tests cover graph consistency, provenance, time/modality validation, revisions, stale caches, dual recall and no fallback, chunk indexing and cache invalidation, label isolation, audio bridging, GPT-6 parameters, duplicate predictions, Agentic costs, and real codec timing/audio alignment.

Known limits: identity matching and event completeness depend on perception; no face tracker or calibrated cross-scene intensity model is included. The pilot has nine questions over three videos and cannot establish full-benchmark improvement. GPT-6 also judging GPT-6 answers may introduce judge bias; an independent judge should be included in expanded experiments.

API references: [GPT-6](https://developers.openai.com/api/docs/models/gpt-6-astra), [GPT-6 migration](https://developers.openai.com/api/docs/guides/latest-model), [Gemini embedding task prefixes](https://ai.google.dev/gemini-api/docs/embeddings).
