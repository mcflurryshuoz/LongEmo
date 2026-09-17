# E09: native Gemini frontend and embedding, Azure GPT-6 answer and judge

The user selected the supplied BlackAI Gemini API for embedding as well as the Gemini video frontend. E09 uses a new run and cache because provider changes are semantic experiment changes. E06 and E08 remain preserved.

| Stage | Provider and model |
|---|---|
| Question-free audio observations | BlackAI native Gemini 3.8 Flash, low thinking |
| Visual perception and event memory | BlackAI native Gemini 3.8 Flash, medium thinking, temperature 1, 8192 output tokens |
| Event/query embeddings | BlackAI native Gemini Embedding 2, 3072 dimensions, normalized vectors |
| Query planning and graph answers | Existing Azure GPT-6 Astra deployment, medium reasoning |
| Official rubric judgment | Same Azure GPT-6; original prompts and normalized scoring unchanged |

Native base URL: `https://www.blackaicoding.com/v1beta`. Embeddings use `models/gemini-embedding-2:batchEmbedContents`, independent request entries per document and role-specific prefixes. The cache fingerprint includes the native protocol, endpoint, model, dimension, task prefix and exact text. No OpenRouter credential or request is needed for E09. No keyword-only or local embedding fallback is permitted.

The full scope remains 141 videos / 558 questions, 8342 twenty-second windows, at HF revision `bb1933541008571883fa1eec1e7bd44d94b6ad2b`. Media sampling, graph schemas, structured semantic recall, dense recall, RRF, graph expansion, temporal coverage, top-12 per route and 48000-character evidence budget are unchanged. All audio and perception are rebuilt through the new provider; old predictions and judgments are not imported. Development overlap remains explicit.

## Current service constraint

Real image/audio generation succeeded. Native Embedding 2 single/batch calls returned HTTP 404 because the account group has no configured model. Embedding 001 and 2-preview returned the same condition. This differs from the compatible `/v1/embeddings` endpoint's unsupported-platform error. [Validation evidence](results/blackaicoding_validation/validation.md) preserves both results. The new adapter is tested with controlled responses but a real successful native embedding response remains unverified.

Start with `--defer-answers` while embedding access is unavailable. This completes each video's question-free memory and records `memory_complete_pending_embedding_service`. After the first complete memory, admission expands from one to 16 videos. It does not represent a successful answering/scoring pipeline. The queue's terminal status is `partial_embedding_service`, with missing answers and scores retained in the full denominator. No embedding calls are repeatedly sent to the known unavailable model.

The same pinned model/endpoint can resume without `--defer-answers` after access is enabled, retaining complete windows and first successful judgments. A different embedding endpoint requires explicit configuration/provenance and a separate vector cache; a different embedding credential must be scoped to its own service rather than overwriting the audio/perception key. Do not edit a running checkout.

## Execution

Use an immutable Git checkout under `/mnt/data1/zyf`, with the existing Python 3.11 embedding environment and FFmpeg 7 path supplied by the launcher. Runtime and secrets stay outside Git.

```sh
/mnt/data1/zyf/LongEmo-runtime/embedding-venv/bin/python -m experiments.zyf.launch_blackai --defer-answers
/mnt/data1/zyf/LongEmo-runtime/embedding-venv/bin/python -m experiments.zyf.launch_blackai --defer-answers --execute
```

Run: `/mnt/data1/zyf/LongEmo-runtime/runs/blackai_gemini38_gpt6_full_v1`.
Cache: `/mnt/data1/zyf/LongEmo-runtime/cache/blackai_gemini38_gpt6_full_v1`.

The launcher retains its exact command, commit, working directory and PID. An OS lock prevents duplicate supervisors. Failed startup attempts remain bounded; three exhausted non-policy video failures stop further admission. Known policy refusals are terminal and recorded without automatic retries. Native perception authentication/credit failures stop admission. Azure uses the existing shared 500000-token/minute, 120-request/minute, 24-in-flight limiter. No local GPU model is allocated.

Report actual model IDs and token usage from ledgers. BlackAI has not exposed billed cost/balance in these responses, so token counts must not be presented as confirmed dollar costs. Full benchmark completion requires 558/558 successful official judgments; construction progress and test results are not accuracy scores.
