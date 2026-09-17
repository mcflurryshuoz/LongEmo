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

- User requested all 558 episode questions after pipeline validation. Downloaded and verified the full 141-video release (68.38 GB), all subtitles and all 558 questions on `/mnt/data1/zyf`. Total duration is 45.9204 hours and the frozen sampler requires 8342 windows. All 141 media/subtitle audits passed.
- Added [full evaluation protocol](full_benchmark.md), `full_benchmark.py`, `analyze_full.py`, and an empirical cost forecast. The method/evaluator source hash is unchanged, preserving compatible E03 checkpoints; copies leave the original pilot untouched.
- Seven full-run regression tests and three checkpoint-refresh tests passed, alongside all 18 original method/media tests. Full-run tests include a simulated scorer interruption that resumes only failed judgments and retains a valid zero score. No synthetic test result is counted as benchmark performance.
- Full-run directory: `/mnt/data1/zyf/LongEmo-runtime/runs/gpt6_full_episode_v1`. Full inference remains unstarted because the authorized OpenRouter account had about USD 1.63 available. Download and offline preparation can proceed independently. The earlier USD 30–50 estimate applies only to completing the nine-question pilot.
- Full-duration calibration: estimated remaining linear cost USD 2186.68; suggested allowance USD 2800–3300 with headroom. This extrapolates from 37 perception windows and two questions, is not a quote, and excludes a full direct/inspection/independent-judge comparison. It supersedes the preliminary USD 2000–3000 estimate. Current credit remains USD 1.6328; the real execution preflight stops before any full-run model stage. No payment or cheaper-model substitution is made.
- A development trace audit found stored event E65 (focused scrutiny at 274.64–281.68 seconds) absent from the Q42 context. Its prefix-truncated lightweight timeline ended at 121.1/366 seconds, while only 9/83 full events fit within 47454 characters. This supports testing full-span target-person timelines and evidence compression; it does not establish a score gain or independently verify perception truth.
- Full coverage, per-task and reasoning-subtype metrics, development overlap, duration groups, actual-context citation checks, and costs will be exported by the offline analyzer. No full result or improvement claim is available yet.

### E05 prepared-data audit and artifacts

- All 141 videos have audio streams; converted subtitles validate, with no start-order issues or overruns above the two-second audit tolerance. All public question strings are nonempty.
- Full inference/scoring coverage remains **0/558**. The earlier E04 2-question score is not imported or misrepresented as a full result.
- The alternate authorized resource listed ten models but no GPT-6 or embedding endpoint; no silent provider/model substitution was attempted.
- Full-run preparation/report: [results JSON](results/gpt6_full_episode_v1/results.json), [readable report](results/gpt6_full_episode_v1/results.md), [cost forecast](results/gpt6_full_episode_v1/cost_forecast.json), [media audit](results/gpt6_full_episode_v1/media_audit.json). The exact executable protocol is in [full_benchmark.md](full_benchmark.md).


## E06: Azure full episode run — 16 concurrent videos

- User authorized higher-concurrency execution on g450. Verified the provided Azure resource using the existing default CLI context: both requested preview API versions and an image request succeeded. The transport smoke through the implemented client returned gpt-6-astra-2026-09-03, 15 tokens, approximately 2.97 seconds. No new credential, role or payment was created.
- Code commit c90ed18b8ce3b7842fbd8cd1f47e1f99598b6a99 was pushed to origin/zyf. Method/evaluator source SHA256 is 64183cb9e878459c0606a6e81e2ec07644f8452c08c6c76c3da176a1d3ea2dc4. Frozen execution checkout: /mnt/data1/zyf/LongEmo-azure-e06. Existing OpenRouter runs retain their original provider/source provenance.
- Full E06 was launched with supervisor PID 2736904 at Unix time 1789578602.6411803, output /mnt/data1/zyf/LongEmo-runtime/runs/azure_gpt6_full_v1. All 141 videos / 558 questions are queued. It starts 16 video workers, two answer/judge workers per video, with shared Azure limits of 250,000 reserved tokens/minute, 120 requests/minute and 24 in-flight requests. The first wave must finish at least one complete scored video before extending the queue. Checkpointed windows and the first successful judgments resume; exhausted failures are retained and can pause scheduling.
- Azure handles perception, planning, answering and official scoring. OpenRouter handles Gemini audio and Gemini Embedding 2 only; initial available OpenRouter credit was USD 31.6322. This replaces the earlier OpenRouter-only funding constraint. The earlier USD 2187 estimate is not an Azure billing estimate; Azure tokens are recorded with billing unknown.
- Only matching, question-free Gemini audio observations are imported from E03, with provenance hashes. Azure perception is rebuilt fresh. The retrieval settings and official prompts/score formulas stay frozen. No direct baseline or improvement claim is made in E06.
- All 34 applicable tests passed: 18 method/media, 10 existing orchestration/checkpoint tests and 6 new Azure tests. New checks cover routing of both generation and judge to Azure, cross-process quota accounting, live limiter configuration, 429 cooldown/attempt ledgers, fail-fast permissions and credential-free reporting.
- Available host resources before launch: approximately 447 GiB RAM and 501 GiB disk on /mnt/data1. The API pipeline allocates no local GPU model. Runtime progress and actual resource usage supersede sizing estimates.
- Protocol and operational notes: [Azure full benchmark](azure_full_benchmark.md). A launched queue is not a completed benchmark: report the actual scored coverage alongside any partial mean.


### E06 operational update: record policy refusals and increase throughput

- Initial Azure calls returned 58 successful, checkpointed perception windows before the planned operational restart. Two video inputs returned HTTP 400 content_policy_violation; zero 429 responses were observed. The blocked inputs are recorded as blocked_input_policy and excluded from automatic retries, while other videos continue. They remain in the 558-question denominator. A complete score cannot be claimed if these service refusals remain.
- Source eee9d3a5357e508752d3f3af760083b9543045fc was pushed to zyf; only the scheduler/tests/protocol changed, preserving the exact method/evaluator source hash. New immutable checkout /mnt/data1/zyf/LongEmo-azure-e06-ops2 runs the same experiment. Original manifests remain untouched and each scheduler revision receives a separate execution_revisions manifest. Added tests reject changes to semantic source/model settings during scheduler-only resume; all 36 relevant checks have passed.
- New admission was paused until the Azure reservation ledger showed zero active requests. The old process group was then stopped and all 58 saved windows resumed under supervisor PID 2800818 (Unix launch time 1789579449.3645391). launch_history and operational_changes.jsonl preserve the transition.
- Based on observed service token headroom above 900k and no 429 responses, the experiment limit increased from 250,000 to 500,000 tokens/minute, with 16 concurrent videos, 120 requests/minute and 24 maximum in-flight Azure requests unchanged. Actual observed throughput and provider refusals remain part of the result.


## E07: series breakdown and existing baseline comparison

- User requested a named-series comparison with existing baselines. Series groups use explicit source.from titles/episode codes; URL-only sources remain unclassified. Aliased jiayouernv to Home with Kids explicitly. Report both per-question mean and coverage; full-episode comparison subsets include every fully scored episode, selected by availability rather than score.
- Found four valid legacy Gemini 2.5 Flash direct predictions from E02/pilot_v2, but no completed direct-baseline scores. The only explicitly named episode in these successful predictions is Friends S01E08 (G2_V000076, Q271 and Q272). Both question IDs are also successfully scored in E06. Other GPT-6 direct attempts had failed for credits and cannot provide a score.
- Preserved the legacy answers exactly and performed their first official judgment with the current Azure GPT-6 Astra judge. Reused the graph method's existing first successful judgments; did not regenerate either method's predictions or rescore the graph answers. Comparison run: /mnt/data1/zyf/LongEmo-runtime/runs/legacy_direct_friends_s01e08_v1. Exact source hashes and model identities are retained in its manifest.
- Same-question result: legacy direct 2/3 on reasoning explanation and 2/4 on trajectory; graph method 2/3 and 2/4. Both normalized means are 58.33%, a 0.00 percentage-point difference, with 2/2 coverage. Both omit required causal/temporal details according to the official judge. Two successful new judge calls returned 6,146 tokens; actual Azure billing is unavailable.
- This is not a controlled graph ablation: generation models differ (Gemini 2.5 Flash versus GPT-6 Astra), direct uses 128 frames with raw audio/subtitles, and graph uses windowed visual perception plus Gemini audio observations. There are only two development questions and possible same-model judge bias. Do not generalize this tie to the whole show or benchmark.
- Sanitized artifacts: [series summary](results/azure_gpt6_full_v1/by_series/series_scores.md), [series JSON](results/azure_gpt6_full_v1/by_series/series_scores.json), [paired comparison](results/legacy_direct_friends_s01e08_v1/comparison.md), [paired JSON](results/legacy_direct_friends_s01e08_v1/comparison.json). Offline exporters assert complete question-ID alignment, full denominators and successful first-score records; they correctly retain empty series as unscored rather than zero.


## E08: latest Gemini perception with Azure GPT-6 answering and scoring

- User explicitly selected replacement of video perception only. Verified the Google model catalog and authorized OpenRouter inventory on 2026-09-17: current stable Flash is Gemini 3.8 (`google/gemini-3.8-flash`), newer than 3.5. A real text request returned that model ID from provider Google, 59 tokens, reported cost USD 0.00020925. Its strict exact-string `OK` check was false; model availability is established, while structured perception is validated by the first real video pipeline.
- Added an explicit perception-model option to the Azure orchestrator. Gemini 3.8 perception uses medium reasoning, temperature 1 and 8192 maximum output tokens; GPT-6 planning, answers and official scoring remain on Azure. Audio remains Gemini 3.8 low reasoning and embedding remains Gemini Embedding 2. Media sampling, prompts, retrieval and the official score formula are unchanged. Method/evaluator source SHA256 remains 64183cb9e878459c0606a6e81e2ec07644f8452c08c6c76c3da176a1d3ea2dc4.
- New output directory: /mnt/data1/zyf/LongEmo-runtime/runs/gemini38_perception_gpt6_full_v1. Fresh Gemini graphs for all 141 videos / 558 questions; no E06 perception, predictions or scores imported. Reuse only exact-compatible question-free audio observations, with source hashes and input/model validation. The E06 process and frozen results are preserved.
- Added startup admission of one video (predeclared development G2_V000016). After the first complete scored pipeline succeeds, admission automatically expands to 16 videos and two answer/judge workers per video. Admission depends on completion, never score. OpenRouter balance was about USD 15.63 before the new run; a USD 2 floor pauses new paid stages while preserving checkpoints. Full-run affordability is not established.
- Routing regression checks verify Gemini is used only for build, GPT-6 still handles graph answering and judging, incompatible perception models cannot reuse a run, and the old Azure manifest semantics stay intact. All 37 applicable checks passed (31 routing/orchestration/memory checks and 6 media/checkpoint checks). The initial test shell used the outdated system FFmpeg; the three media checks passed after using the existing runtime FFmpeg 7 path, as the launcher does.
- Protocol and launcher: [Gemini perception benchmark](gemini_perception_benchmark.md), `launch_gemini_perception.py`. Runtime launch and first-video validation are recorded below after execution. No new benchmark score or improvement is claimed at preparation.


### E08 launch

- Execution commit 5f6268bda9ad25425166b1eda371cc99edcf49d8 was pushed to origin/zyf. Frozen checkout /mnt/data1/zyf/LongEmo-gemini-e08; supervisor PID 1980515, launch Unix time 1789601125.0984094. All 141 media/subtitle inventories passed. Reused 4994 compatible question-free audio observations at the initial snapshot.
- First confirmed status: running, active G2_V000016, 140 videos queued, 0/558 scored. The first real Gemini visual perception request was in flight. This is a launch confirmation, not a passed end-to-end validation or a benchmark score. Initial-pipeline completion gates automatic expansion.

- First real visual validation: 5/19 windows of G2_V000016 completed successfully, all returned `google/gemini-3.8-flash` and passed the existing memory-schema/reference checks. Those five perception calls reported USD 0.163383; the text connectivity probe cost is separate. The initial video is still in progress and no E08 official score exists at this snapshot. [Sanitized initial progress](results/gemini38_perception_gpt6_full_v1/initial_progress.json) records the capture time, actual model IDs, coverage and usage.


### Resource validation: BlackAI Gemini 3.8

- The user supplied a Gemini gateway key on 2026-09-17. It exactly matches the existing private Gemini credential on g450; no secret was exported. Both model catalogs return HTTP 200 and advertise Gemini 3.8, although the native and compatible catalogs differ.
- Native `gemini-3.8-flash` text generation succeeded (HTTP 200, modelVersion gemini-3.8-flash, STOP, OK). A real four-second V16 image-plus-audio request without subtitles also returned valid JSON; usage explicitly includes IMAGE and AUDIO tokens. These checks validate a possible perception/audio provider, not benchmark quality or sustained capacity.
- The gateway /v1/embeddings endpoint explicitly returns HTTP 404, saying Embeddings API is not supported for this platform. The existing OpenRouter balance was approximately USD 0.2410 at this check. The gateway returned tokens but neither billed costs nor a balance.
- These diagnostics did not launch or mutate a benchmark run. E06 remains partial_input_policy with 185/558 scored; E08 is blocked_api_credits with 6/558 scored. A future provider change requires a separate configuration and validated audio/embedding routing; simply changing the example environment variables cannot resume the currently pinned OpenRouter experiment.
- [Validation report](results/blackaicoding_validation/validation.md), [sanitized response metadata](results/blackaicoding_validation/validation.json).
