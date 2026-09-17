# E09 native Gemini progress

Status: paused_by_user. Scored: 0/558. Complete memories: 21/141. Saved windows: 995/8342.

Execution commit: 3c0d8fb0ab4185810820c5cfed8c76360bc5734c. Supervisor PID: 3497634.

Audio and perception use native Gemini 3.8 Flash; embedding is configured for native Gemini Embedding 2. Azure GPT-6 performs planning, answering and unchanged official judging. No OpenRouter calls or fallback.

Deferred answering: True. Complete memories awaiting embedding access are not scored results.

Active videos: 0; queued: 107.

| Stage | Attempts | Successful | Returned tokens | Reported model IDs |
|---|---:|---:|---:|---|
| audio | 1253 | 1006 | 1998377 | {'gemini-3.8-flash': 1249} |
| perception | 1113 | 995 | 42863405 | {'gemini-3.8-flash': 1107} |
| embedding | 0 | 0 | 0 | {} |

Token usage includes returned usage on failed validation attempts. Missing billing is unknown, not zero. No full benchmark score or improvement is established until coverage and matched comparisons support it.
