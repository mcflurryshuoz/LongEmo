# E09 native Gemini progress

Status: running. Scored: 0/558. Complete memories: 0/141. Saved windows: 3/8342.

Execution commit: b113383a1a286606a2feac769f6bcfeb6c642fd6. Supervisor PID: 3451207.

Audio and perception use native Gemini 3.8 Flash; embedding is configured for native Gemini Embedding 2. Azure GPT-6 performs planning, answering and unchanged official judging. No OpenRouter calls or fallback.

Deferred answering: True. Complete memories awaiting embedding access are not scored results.

Active videos: 1; queued: 140.

| Stage | Attempts | Successful | Returned tokens | Reported model IDs |
|---|---:|---:|---:|---|
| audio | 6 | 4 | 12141 | {'gemini-3.8-flash': 6} |
| perception | 5 | 3 | 112819 | {'gemini-3.8-flash': 3} |
| embedding | 0 | 0 | 0 | {} |

Token usage includes returned usage on failed validation attempts. Missing billing is unknown, not zero. No full benchmark score or improvement is established until coverage and matched comparisons support it.
