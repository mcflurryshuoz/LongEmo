# GPT-6 hybrid graph pilot

This run contains 1 video(s) and 2 questions. It is not a full benchmark result.

| Method | Successful predictions | Scored / total | Overall (%) |
|---|---:|---:|---:|
| graph | 2/2 | 2/2 | 87.50 |
| direct | 0/2 | –/2 | pending |

| Video | Complete | Windows | Events | States |
|---|---|---:|---:|---:|
| G2_V000016 | True | 19 | 83 | 246 |

| Stage | Attempts | Returned tokens | Provider-reported USD | Attempts without cost |
|---|---:|---:|---:|---:|
| reused_memory_perception | 21 | 224245 | 4.666370 | 2 |
| reused_memory_audio_observation | 21 | 28321 | 0.058033 | 0 |
| shared_planning | 2 | 2928 | 0.052710 | 0 |
| embedding_shared | 8 | 32182 | 0.006438 | 0 |
| graph_answer | 2 | 31313 | 0.413335 | 0 |
| graph_inspection_audio | 0 | 0 | 0.000000 | 0 |
| direct_answer | 10 | 0 | 0.000000 | 10 |
| direct_inspection_audio | 0 | 0 | 0.000000 | 0 |

Reused-memory costs were incurred in the parent run and are not new charges in this run. Shared costs must be amortized equally. Cached query embeddings/plans are not a zero-cost ability of the second condition. Judge costs are recorded separately in the JSON. Provider-reported cost can exclude failed requests without returned usage.

The evaluated conditions use GPT-6 Astra; graph retrieval uses Gemini Embedding 2. Direct uses 128 sampled frames and the same subtitles/audio-only frontend but no graph states. Graph perception uses 20-second windows, so visual budgets differ.

Score formulas and judge prompts are unchanged. The parent three-video pilot has two reasoning explanation questions and no result questions; this smoke subset may have no reasoning questions. GPT-6 judging GPT-6 outputs introduces possible judge bias. No statistical superiority claim is supported by nine development questions.
