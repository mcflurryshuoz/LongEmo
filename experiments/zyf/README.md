# zyf experiment log

Date: 2026-09-16. Host: `g450` (`t2vg-a100-G4-50`). Original repository base: `b29b70a361bf6d7b08d4907167720f288fdbc2fe`.

## Data and environment

- HF revision: `bb1933541008571883fa1eec1e7bd44d94b6ad2b`.
- Episode inventory: 558 questions, 141 videos, 68,375,210,073 video bytes. Types: intensity comparison 194, trajectory 235, emotional reasoning 129.
- Deterministic development pilot seed: `longemo-zyf-pilot-v1`, sorted by SHA256(seed + video_id), retaining all questions per selected video.
- Videos: `G2_V000016` (366.063 s), `G2_V000076` (1311.043 s), `G2_V000119` (1274.524 s).
- Nine questions: `G2_Q000041`, `G2_Q000042`, `G2_Q000271`, `G2_Q000272`, `G2_Q000465`, `G2_Q000466`, `G2_Q000467`, `G2_Q000468`, `G2_Q000469`. Types: 2 intensity, 5 trajectory, 2 reasoning. Both reasoning questions request explanations; this pilot does not cover the result subtype.
- Runtime/data/log root: `/mnt/data1/zyf/LongEmo-runtime`. Credentials are outside Git, access-restricted, and omitted from logs.
- Python 3.10.21 isolated venv. FFmpeg 7.0.2-static from imageio-ffmpeg 0.6.0; system ffprobe. System FFmpeg was too old for the repository's `fps_mode` option.
- Initial capacity: approximately 515 GiB available RAM and 571 GiB free storage on `/mnt/data1`; existing GPUs are shared and not allocated for the API pipeline.

## E00: Offline and provider validation

- Twelve tests passed: memory transactions/references/time/modality, continuation/revision, stale-cache rejection, subtitle conversion, retrieval budget/isolation, label leakage prevention, duplicate submissions, Agentic startup/all-round token totals, actual frame PTS and delayed-audio alignment.
- Native Gemini proxy `gemini-2.5-flash` text and real image+audio requests succeeded. Usage explicitly reported IMAGE and AUDIO tokens. A short no-subtitle excerpt produced quoted audible speech; this is a smoke test, not comprehensive modality validation.
- Initial judge connectivity test for `gemini-2.5-pro` returned HTTP 502; the judge must be validated before any result is accepted.

## E01: pilot_v1 — interrupted, not a score result

- Joint perception: 20 s core + 2 s padding, 1 fps, max 24 frames, max 200704 pixels/frame, source audio and timestamped subtitles, Gemini 2.5 Flash, temperature 0, thinking disabled, max output 8192, three parallel videos.
- Found a real source-frame boundary bug: an accurate seek at a non-frame-aligned interval end could return a later PTS and abort the whole window. Fixed by choosing an in-window final target and rejecting overshoot without relabeling timestamps.
- Interrupted for that fix. Artifacts and call ledgers retained under `runs/pilot_v1`; excluded from comparisons.

## E02: pilot_v2 — memory construction in progress

- Same perception configuration, with the frame-boundary fix. Three video builds run independently; windows within each video are sequential.
- Source memories are question-independent. Each window validates and commits atomically; schema-repair attempts are included in token ledgers.
- Direct audiovisual baseline precomputation: 128 uniform frames at max 200704 pixels, full source audio and subtitles; Gemini 2.5 Flash, temperature 0, thinking disabled, max output 4096, workers 2. This is a system baseline, not a media-budget-matched ablation.
- User steering: primary method is now **event-graph retrieval with structured semantic conditions plus embedding recall**, followed by graph expansion and temporal coverage. The earlier flat-retrieval comparison is no longer a required experiment.
- No benchmark scores reported yet. Do not interpret successful construction or unit tests as evidence of score improvement.

## Reporting protocol

Record code commit and source hashes, exact HF revision and question IDs, provider/model configurations, actual media budgets, all attempt/token ledgers, inference/score coverage and the unchanged official rubric-based metrics. Judge failures must be repaired before complete comparisons. Construction, indexing, question planning and answering costs are reported separately, with consistent amortization.

Every run directory retains its own manifests and logs. Changes to model, prompts, retrieval or graph content require a new run or explicitly versioned stage; no mixing cached predictions from different configurations. Nine development questions cannot establish statistical superiority or a full LongEmoBench result.

## E02 outcome and model upgrade

- First implementation commit `1c87407` was pushed to `origin/zyf`. Frozen source hash `01f6d4c952008d7109688cd66b1a5f6fa230676b1b758f80d614af27aedf619d` matches that code. Jobs started with uncommitted implementation on base `b29b70a`; their original manifests truthfully retain that base revision.
- E02 perception completed V16 (19/19 windows, 44 events), stopped V76 at 41/66 and V119 at 15/64 after exhausted JSON validation repairs. Original evidence/cost ledgers remain preserved; no incomplete memory is used for scoring.
- Original direct inference completed 4/9 predictions. V119 failed at the final frame. Cause: ffprobe's six-decimal PTS rounding could place the final select threshold just past the actual last frame. Fixed using a one-microsecond selection tolerance and a safe tail seek, with a real fractional-FPS regression test.
- E02 is retired following the user's explicit upgrade to GPT-6 and an API embedding model. It is not presented as a GPT-6 run or a scored comparison.

## E03: GPT-6 + Gemini Embedding 2

- User requested GPT-6 inference and Gemini 3.5 or a stronger embedding model. Official Google documentation identifies the dedicated embedding model as `gemini-embedding-2`; Gemini 3.5 Flash is a generative model, not an embedding endpoint.
- Existing authorized OpenRouter resource successfully tested: `openai/gpt-6-astra` returned OK (17 tokens, provider-reported USD 0.00037); `google/gemini-embedding-2` returned two 3072-dimensional vectors (27 tokens, USD 0.0000054). Successful connectivity is not an accuracy evaluation.
- Primary retrieval is structured semantic + dense event recall, RRF fusion, event-graph expansion and temporal coverage. No flat-retrieval experiment is queued. Per-route traces, cached vectors and all returned usage/cost records are retained.
- GPT-6 uses medium reasoning, no temperature/top-p. Since it has no audio input, independent question-free Gemini 3.8 Flash audio observations feed GPT-6 alongside frames and subtitles. Graph and direct baseline share the same audio observations. Raw audio is never silently dropped.
- E5 CPU smoke test succeeded but is retained only as an explicit optional diagnostic; it is not the primary embedding model.
- E03 uses a new graph from scratch, distinct from the retired Flash graph. Main graph and direct baseline share GPT-6 and the frozen input videos/questions. Optional one-inspection graph variant is recorded separately. All scores use the unchanged official evaluator and the same explicit GPT-6 judge.
- Eighteen targeted tests passed on g450, including real codecs, dual-route fusion, embedding cache invalidation, no fallback, GPT-6 payload constraints and audio bridge isolation. A broad compileall over vendored emollm encounters upstream Python 2 code; vendored code is not modified and is outside this method's runtime.
- E03 runtime directory: `/mnt/data1/zyf/LongEmo-runtime/runs/gpt6_hybrid_v1`. Status/results to be appended after the run; no improvement claim yet.

### E03 checkpoint: paused for provider credits

- Implementation `8136634` and documentation/report tooling `a5f64db` were pushed to `origin/zyf`.
- The real GPT-6 image + Gemini audio-observation path passed a 12-second no-subtitle smoke check. The actual V119 full-clip regression also returned all 128 frames successfully after the final-PTS fix.
- Completed windows: V16 **19/19** (83 events, 246 states); V76 **11/66**; V119 **7/64**. Total **37/149**. Only V16 is a complete memory. No full-pilot predictions or scores are claimed.
- Returned usage records report **USD 8.956630** for GPT-6 perception and **USD 0.128900** for the audio observer (USD 9.085530 combined). HTTP failures without returned usage/cost remain explicitly unaccounted for; this is not a complete billing reconciliation.
- Root cause of later API failures was independently confirmed: OpenRouter HTTP **402**, insufficient available credits after reserving in-flight requests. The original job and a diagnostic recovery attempt have exited; no background inference loop is left spending credits. Partial memories and successful audio observations remain resumable under the same source/model/data configuration.
- A diagnostic resume initially used default integer-valued sampling arguments instead of the experiment's explicit float-valued CLI, producing an audio input-hash mismatch. Repeating the exact original arguments matched the cache. Use the saved launch command for reproducible resume; do not hand-reconstruct defaults.
- Sampled summed RSS of the original process subtree peaked at approximately **215 MiB**, sampled every five seconds. This can miss short-lived peaks and excludes the separately launched failed diagnostic process. No local GPU model was allocated. It is not a measured upper bound for full concurrent baseline inference.
- Complete episode videos total **68.38 GB** of disk, not a RAM requirement. For this streaming API implementation, budget **16–32 GB RAM**, **zero local model VRAM**, and roughly **100 GB+ free disk** for the full video set plus caches/temp files. These are operational recommendations, not demonstrated minimum requirements.
- Estimated additional allowance to finish the remaining graph construction, 9-question conditions and judge: **USD 30–50**, based on observed window costs and allowing headroom. User was asked to refill the already-authorized OpenRouter account. No alternative model was silently substituted.
- Sanitized checkpoint report: [results JSON](results/gpt6_hybrid_v1/results.json), [readable snapshot](results/gpt6_hybrid_v1/results.md). No gold annotations or raw media are committed.

## E04: two-question engineering smoke — scored, no baseline comparison

- Reused the only complete E03 graph, V16, and retained both of that video's questions (`G2_Q000041`, `G2_Q000042`). Selection was based on checkpoint availability before scoring, not answer quality. This is an additional engineering smoke, not the full 9-question pilot.
- New run `/mnt/data1/zyf/LongEmo-runtime/runs/gpt6_smoke_v16`; separate plans and Gemini embedding cache; single worker. Graph inference and the unchanged official evaluator both completed **2/2**.
- Intensity comparison: **1/1**. Emotion trajectory: **3/4**. Official normalized mean: **87.5%** on these **two questions only**. No emotional reasoning questions are present. GPT-6 generated and judged these answers, with possible self-judge bias.
- The two retrieval contexts contained 8 and 9 full events, using 47,697 and 47,454 characters respectively; both had no invalid cited event/observation IDs. Dense and semantic route traces are retained.
- The trajectory judge noted weaker expression of the focused evaluation of the story's logic. This identifies a follow-up diagnostic, not proof of whether perception, retrieval or answer composition caused the omission.
- Direct 128-frame baseline did not produce valid predictions. A one-attempt diagnostic confirmed HTTP 402 with a credit-dependent prompt-token allowance below the requested 195,055 tokens. This is provider credit preflight metadata, not measured billed tokens. No baseline score, score gain or full-benchmark claim is supported.
- Detailed scored smoke and separate reused/incremental cost accounting: [results JSON](results/gpt6_smoke_v16/results.json), [readable report](results/gpt6_smoke_v16/results.md).

## E05: full episode pass — prepared, inference blocked by credits

- User requested all 558 episode questions after pipeline validation. Downloading the full 141-video release (68.38 GB) on `/mnt/data1/zyf`; the 558 questions were already available.
- Added [full evaluation protocol](full_benchmark.md), `full_benchmark.py`, `analyze_full.py`, and an empirical cost forecast. The method/evaluator source hash is unchanged, preserving compatible E03 checkpoints; copies leave the original pilot untouched.
- Seven new regression tests passed, including a simulated scorer interruption that resumes only failed judgments and retains a valid zero score. No synthetic test result is counted as benchmark performance.
- Full-run directory: `/mnt/data1/zyf/LongEmo-runtime/runs/gpt6_full_episode_v1`. Full inference remains unstarted because the authorized OpenRouter account had about USD 1.63 available. Download and offline preparation can proceed independently. The earlier USD 30–50 estimate applies only to completing the nine-question pilot.
- Preliminary full-pass allowance: about USD 2000–3000, extrapolated from partial data inventory and very small model-call samples; recalibrate after all durations are known. No payment or cheaper-model substitution is made.
- A development trace audit found stored event E65 (focused scrutiny at 274.64–281.68 seconds) absent from the Q42 context. Its prefix-truncated lightweight timeline ended at 121.1/366 seconds, while only 9/83 full events fit within 47454 characters. This supports testing full-span target-person timelines and evidence compression; it does not establish a score gain or independently verify perception truth.
- Full coverage, per-task and reasoning-subtype metrics, development overlap, duration groups, actual-context citation checks, and costs will be exported by the offline analyzer. No full result or improvement claim is available yet.
