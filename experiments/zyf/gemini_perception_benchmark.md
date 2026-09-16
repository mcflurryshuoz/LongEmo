# E08: Gemini 3.8 perception, GPT-6 retrieval and scoring

The user requested the latest Gemini for video perception while keeping GPT-6
for planning, answering and official scoring. On 2026-09-17, Google's model
catalog and the authorized OpenRouter model inventory both list
`google/gemini-3.8-flash`. Use that explicit ID, rather than a moving latest alias.

- Video perception: Gemini 3.8 Flash through OpenRouter, medium reasoning,
  temperature 1, maximum 8192 output tokens.
- Audio observation: existing question-independent Gemini 3.8 Flash, low reasoning,
  maximum 4096 tokens. Compatible observations from E06 are copied with source
  hashes and validated against the source audio and exact client configuration.
- Planning, answering and official judge: Azure `gpt-6-astra`, medium reasoning,
  maximum 8192 output tokens; existing shared Azure rate limiter.
- Embedding: Gemini Embedding 2; semantic BM25 plus dense retrieval and event-graph
  expansion are unchanged.
- Sampling: 20-second windows, 2-second padding, 1 fps, at most 24 frames per
  window, maximum 200704 pixels. Retrieval: top 12 per route, 48000 characters,
  no additional visual inspections.
- Dataset: all 141 videos / 558 questions at the same pinned release. Perception
  sees no benchmark question, reference answer or rubric. The complete 558-question
  denominator is retained for partial results.
- Every video receives newly built Gemini perception; no GPT-6 memory, prediction
  or judgment is imported. The new run is independent of E06 and does not select
  videos by E06 success/failure or score. Safety defaults are unchanged.
- Retain the earliest successful official judgment, including zero scores.
  Known policy refusals remain missing results; never count them as a zero score.

## Execution

Freeze the tested revision in a separate checkout on g450. From its root:

```bash
python3 -m experiments.zyf.launch_gemini_perception
python3 -m experiments.zyf.launch_gemini_perception --execute
```

Output: `/mnt/data1/zyf/LongEmo-runtime/runs/gemini38_perception_gpt6_full_v1`.
Start with the already declared development video G2_V000016; after its full
pipeline is successfully scored, automatically expand to 16 concurrent videos,
two answer/judge workers each. The first video's score does not govern admission.
If no initial video completes, stop for diagnosis. An OpenRouter balance below
USD 2 pauses new paid stages and preserves completed windows. This is a balance
floor, not a spend cap or a guarantee that the account can fund all 8342 windows.

Record requested and returned model IDs, token usage, reported OpenRouter cost,
actual coverage, refusal/failure counts, source revision and configuration hashes.
Compare E06/E08 only on explicitly aligned question IDs and report full-run
coverage separately. Changing the perception model does not establish a score gain.

Sources checked 2026-09-17:
[Google model catalog](https://ai.google.dev/gemini-api/docs/models),
[OpenRouter Gemini 3.8 Flash](https://openrouter.ai/google/gemini-3.8-flash).
