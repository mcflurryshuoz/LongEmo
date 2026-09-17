# E09 native Gemini progress

Status: retired_native_output_limit. Scored: 0/558. Complete memories: 0/141. Saved windows: 8/8342.

Execution commit: c0dab97a11f4c5f34a7eb670fae34df54c55a2f5. Supervisor PID: 3480015.

Audio and perception use native Gemini 3.8 Flash; embedding is configured for native Gemini Embedding 2. Azure GPT-6 performs planning, answering and unchanged official judging. No OpenRouter calls or fallback.

Deferred answering: True. Complete memories awaiting embedding access are not scored results.

Active videos: 0; queued: 140.

| Stage | Attempts | Successful | Returned tokens | Reported model IDs |
|---|---:|---:|---:|---|
| audio | 12 | 9 | 23730 | {'gemini-3.8-flash': 12} |
| perception | 15 | 8 | 308692 | {'gemini-3.8-flash': 8} |
| embedding | 0 | 0 | 0 | {} |

Token usage includes returned usage on failed validation attempts. Missing billing is unknown, not zero. No full benchmark score or improvement is established until coverage and matched comparisons support it.
