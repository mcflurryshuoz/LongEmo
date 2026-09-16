# Full episode evaluation protocol

The user requested evaluation of the complete episode release on 2026-09-16. This is **558 questions from 141 videos** (194 intensity comparisons, 235 trajectories, and 129 reasoning questions: 41 results + 88 explanations), at HF revision `bb1933541008571883fa1eec1e7bd44d94b6ad2b`; video files total 68,375,210,073 bytes. This does not include a separate clip release.

## Frozen first pass

Use the existing GPT-6 Astra perception and answer implementation with Gemini 3.8 Flash question-free audio observations and Gemini Embedding 2 dense recall. The method source hash remains `b7dee478e8dac8ee1de4de862ef37eaf1180401a7089d5d41b5b3a50dfdeb058`; the full-run orchestration lives outside the method/evaluator source tree, so existing perception checkpoints remain compatible. Each imported video still undergoes full source, model, and sampling configuration validation.

- 20-second perception windows with 2-second padding, 1 fps, maximum 24 frames and 200704 pixels/frame; medium GPT-6 reasoning, 8192 output-token limit.
- Structured semantic and embedding top-12 recall, RRF constant 60, event-relation/person-neighbor expansion, 48000-character evidence budget.
- Main full pass uses graph retrieval with zero question-specific inspections. No change to perception or retrieval based on full-set score feedback during this pass.
- The unchanged official episode judge prompts and scoring formulas are used with explicit GPT-6 Astra judging. Report self-judge bias. A complete baseline and independent judge remain separate follow-up comparisons.
- The nine development pilot questions remain in the official full score and are also reported separately from the other 549. Do not label the latter a new official split or claim a completely untouched holdout.

## Execution and resume

Run from `/mnt/data1/zyf/LongEmo-zyf`, with runtime FFmpeg first on PATH:

```bash
export PATH=/mnt/data1/zyf/LongEmo-runtime/venv/bin:$PATH
export TMPDIR=/mnt/data1/zyf/LongEmo-runtime/tmp
/mnt/data1/zyf/LongEmo-runtime/embedding-venv/bin/python -m experiments.zyf.full_benchmark \
  --data-root /mnt/data1/zyf/LongEmo-runtime/data \
  --output-dir /mnt/data1/zyf/LongEmo-runtime/runs/gpt6_full_episode_v1 \
  --credential-file /mnt/data1/zyf/LongEmo-runtime/api_config.json \
  --embedding-cache-dir /mnt/data1/zyf/LongEmo-runtime/cache/gemini-embedding-2-full \
  --import-memory /mnt/data1/zyf/LongEmo-runtime/runs/gpt6_hybrid_v1/memory \
  --video-workers 3 --workers 2 --minimum-credits 25 --execute
```

If the three-video pilot advances after full-run preparation, refresh its imported copies **before the first full-run execution**:

```bash
/mnt/data1/zyf/LongEmo-runtime/embedding-venv/bin/python -m experiments.zyf.refresh_imports \
  --run /mnt/data1/zyf/LongEmo-runtime/runs/gpt6_full_episode_v1
```

This checks that no full-run video stage exists, imported destinations remain unchanged, and already completed source windows are identical. It retains backups and updates the imported-cost snapshot. This prevents repeating pilot perception work after recharging. It refuses replacement once full-run work has started.

Omit `--execute` for data inventory and manifest preparation with **no model calls**. Preparation still records incomplete status until all videos and converted subtitles are available. The default credit floor is a stage admission threshold, **not** a total spending cap, a guaranteed per-video budget, or a provider reservation guarantee. An account balance below that floor stops execution before model subprocesses. HTTP/provider failures retain successful checkpoints and stop scheduling additional videos.

Reuse the same command and output directory for an identical configuration. A file lock rejects concurrent resumes. Perception checkpoints are copied into the new run, leaving the original pilot untouched. Parallel builds use separate build-result directories linked to their corresponding video memory, avoiding shared temporary-file writes.

Each video has independent questions, plans, predictions, call ledgers, and official scoring runs. Successful predictions are reused. The scorer sees only pending/failed judgments on resume; the first successful judgment is retained even if it has a low or zero score. Frozen input hashes reject changed predictions/questions or method configuration. The full aggregate always keeps all 558 questions in its denominator and marks partial results **INCOMPLETE**. Complete score coverage must be 558/558 before publishing a full score.

The seven full-run regression tests include duplicate-submission rejection, first-success judgment retention, changed-prediction rejection, full-denominator partial reporting, an empty run with no invented score, zero model subprocesses on insufficient credits, and a simulated interrupted official score pass that retries only its failed question and retains a valid zero score. Three additional checkpoint-refresh tests verify backup preservation, rejection of rewritten source windows, and rejection after full evaluation starts. All 18 original method/media tests also pass.

## Offline reporting

```bash
/mnt/data1/zyf/LongEmo-runtime/embedding-venv/bin/python -m experiments.zyf.analyze_full \
  --run /mnt/data1/zyf/LongEmo-runtime/runs/gpt6_full_episode_v1 \
  --embedding-cache /mnt/data1/zyf/LongEmo-runtime/cache/gemini-embedding-2-full \
  --output /mnt/data1/zyf/LongEmo-runtime/reports/gpt6_full_episode_v1
```

The sanitized report contains official overall/type/reasoning-subtype results, development overlap, video-duration groups, question-level scores, retrieval-route overlap and contribution, temporal coverage, and citations absent from the actual supplied context. It does not export raw gold annotations, transcripts, or video. Full coverage enables a 2000-resample video-cluster bootstrap interval; this does not quantify judge variance or correlations between related source videos.

Costs distinguish imported perception/audio work from new calls using `imported_costs.json`. Provider-returned costs are not a complete billing reconciliation when failed attempts return no usage. No GPU model is loaded by the primary API/NumPy path.

## Credit blocker and provisional forecast

At setup, the authorized OpenRouter account had approximately **USD 1.63** available and previously returned HTTP 402. The full inference pass has **not** started. The earlier USD 30–50 estimate covered the remaining nine-question pilot only.

All **141/141 videos and subtitles are now downloaded and prepared**. FFprobe and subtitle validation passed for every video: all have audio, no subtitle start-order violations, and no subtitle timestamps more than two seconds beyond the video duration. Total video duration is **45.9204 hours**, corresponding to **8342** 20-second perception windows. Downloaded video bytes match the release total; size and available LFS SHA256 checks were performed by the downloader.

The E03 returned cost was about USD 0.245555 per completed perception window including the audio observer. The two-question E04 incremental plan/embedding/answer/judge cost was about USD 0.264060 per question. Reusing the 37 existing perception windows yields an estimated remaining linear cost of **USD 2186.68**, with a suggested **USD 2800–3300** allowance for 25–50% headroom (rounded up to hundreds). This covers the full primary graph pass and GPT-6 judging, excluding a full direct baseline, inspection variant, and independent judge.

This is an extrapolation from only 37 windows and two questions, not a quote or confidence interval. Long-video cast/context growth, repairs, provider routing and price changes may increase spending. The first partial-inventory estimate was USD 2000–3000; the full-duration estimate above supersedes it. Exact inputs and assumptions are recorded in [cost_forecast.json](results/gpt6_full_episode_v1/cost_forecast.json). Data audit: [media_audit.json](results/gpt6_full_episode_v1/media_audit.json).

Data download, protocol preparation, and the initial offline diagnostics are complete without further model spending. No cheaper model is substituted silently and no payment is made by the agent.

## Improvement candidates from the development smoke

A [concrete trace audit](results/gpt6_smoke_v16/retrieval_audit.json) of `G2_Q000042` found a likely retrieval bottleneck relative to the stored memory:

- The memory contains event `E65`, at 274.64–281.68 seconds, recording the target person's focused scrutiny of the cassette-to-VHS sequence. Neither the full evidence nor the lightweight timeline supplied this event to the answer model.
- Full evidence contains only 9 of 83 events and consumes 47454 characters. Several selected records include 4–5 people and span 4–7 thousand characters each.
- The timeline contains the first 30 events and ends at 121.1 seconds in a 366-second video. The implementation appends chronologically until the timeline budget is exhausted, so its lightweight coverage is biased toward the beginning. Selected full records do include later moments; it would be wrong to say the model saw nothing after 121 seconds.
- The judge gave 3/4 and specifically found the serious logical-evaluation stage less explicit. This aligns with the missing stored event, but it does not independently verify the perception or prove a causal score improvement from retrieval changes.
- Every citation in this answer exists in its supplied context. More generally, existing inference validates IDs against the whole memory; the new offline diagnostic also checks visibility in the actual question context.

Prioritize a separately versioned retrieval comparison after the frozen first pass: (1) build a target-person timeline spanning the complete clip instead of taking a chronological prefix, (2) compress irrelevant people and duplicate observations while retaining evidence IDs, (3) reserve evidence for changes in target-person state and connected event chains, and (4) compare graph-only answers with bounded source reinspection for missing/contradictory stages. Evaluate on the same frozen memories and judge protocol, disclose development use, and report cost as well as score. None of these candidates is currently claimed to improve benchmark accuracy.
