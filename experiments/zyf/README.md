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
