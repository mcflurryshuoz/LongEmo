# GPT-6 hybrid graph pilot

This run contains 3 video(s) and 9 questions. It is not a full benchmark result.

| Method | Successful predictions | Scored / total | Overall (%) |
|---|---:|---:|---:|
| graph | 0/9 | –/9 | pending |
| graph_inspect | 0/9 | –/9 | pending |
| direct | 0/9 | –/9 | pending |

| Video | Complete | Windows | Events | States |
|---|---|---:|---:|---:|
| G2_V000016 | True | 19 | 83 | 246 |
| G2_V000076 | False | 11 | 60 | 108 |
| G2_V000119 | False | 7 | 33 | 59 |

| Stage | Attempts | Returned tokens | Provider-reported USD | Attempts without cost |
|---|---:|---:|---:|---:|
| perception | 54 | 427155 | 8.956635 | 17 |
| audio_observation | 46 | 63883 | 0.128897 | 0 |
| shared_planning | 0 | 0 | 0.000000 | 0 |
| embedding_shared | 0 | 0 | 0.000000 | 0 |
| graph_answer | 0 | 0 | 0.000000 | 0 |
| graph_inspection_audio | 0 | 0 | 0.000000 | 0 |
| graph_inspect_answer | 0 | 0 | 0.000000 | 0 |
| graph_inspect_inspection_audio | 0 | 0 | 0.000000 | 0 |
| direct_answer | 0 | 0 | 0.000000 | 0 |
| direct_inspection_audio | 0 | 0 | 0.000000 | 0 |

Reused-memory costs were incurred in the parent run and are not new charges in this run. Shared costs must be amortized equally. Cached query embeddings/plans are not a zero-cost ability of the second condition. Judge costs are recorded separately in the JSON. Provider-reported cost can exclude failed requests without returned usage.

The evaluated conditions use GPT-6 Astra; graph retrieval uses Gemini Embedding 2. Direct uses 128 sampled frames and the same subtitles/audio-only frontend but no graph states. Graph perception uses 20-second windows, so visual budgets differ.

Score formulas and judge prompts are unchanged. The parent three-video pilot has two reasoning explanation questions and no result questions; this smoke subset may have no reasoning questions. GPT-6 judging GPT-6 outputs introduces possible judge bias. No statistical superiority claim is supported by nine development questions.
