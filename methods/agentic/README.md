# Agentic video understanding

This method uses the same timestamped-frame input for Qwen Omni/VL, GPT, Claude,
and Gemini. It first samples the complete clip at low cost, then lets the model
request bounded time ranges with a higher frame rate or pixel budget.

Audio and subtitles are run-level options. `--with-audio` and
`--with-subtitle` are fixed before the first request; an `inspect` action cannot
enable or disable either input. A model action is one of:

```json
{"action":"inspect","ranges":[{"start":32,"end":45,"fps":4,"max_frames":48,"max_pixels":602112}]}
```

or:

```json
{"action":"answer","answer":"..."}
```

Example:

```bash
python -m methods.agentic.runner \
  --data-path data/open_qa/questions \
  --videos-dir data/open_qa/videos \
  --output-dir output/agentic/gpt \
  --model gpt-4.1 \
  --with-audio --with-subtitle
```

Use `-g episode` for the second-granularity question set. The same controller
is used, but the question and video selection come from the episode records.

The output is `predictions.jsonl`; each record includes the final prediction and
an evidence trace containing the initial and requested frame ranges.
