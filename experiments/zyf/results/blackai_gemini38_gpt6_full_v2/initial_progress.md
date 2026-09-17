# E09 native Gemini progress

Status: running. Scored: 0/558. Complete memories: 0/141. Saved windows: 5/8342.

Execution commit: 3c0d8fb0ab4185810820c5cfed8c76360bc5734c. Supervisor PID: 3497634.

Audio and perception use native Gemini 3.8 Flash; embedding is configured for native Gemini Embedding 2. Azure GPT-6 performs planning, answering and unchanged official judging. No OpenRouter calls or fallback.

Deferred answering: True. Complete memories awaiting embedding access are not scored results.

Active videos: 1; queued: 140.

| Stage | Attempts | Successful | Returned tokens | Reported model IDs |
|---|---:|---:|---:|---|
| audio | 7 | 6 | 12530 | {'gemini-3.8-flash': 7} |
| perception | 5 | 5 | 196480 | {'gemini-3.8-flash': 5} |
| embedding | 0 | 0 | 0 | {} |

Token usage includes returned usage on failed validation attempts. Missing billing is unknown, not zero. No full benchmark score or improvement is established until coverage and matched comparisons support it.
