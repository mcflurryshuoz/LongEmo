# Subtitles

This directory contains dialogue-only subtitles organized by series and
episode, parallel to the benchmark question data.

```text
subtitles/<series>/
└── <episode>.json
```

Rows contain `series`, `ep`, `unit_id`, `t`, `speaker`, and `text`; `t` is a
two-element `[start, end]` interval in seconds. `speaker` is the annotated
character name, or `unknown` when no reliable character label is available.

The released text excludes speaker labels, bracketed stage directions,
sound-effect captions, and subtitle credits. `cleaning_report.json` records the
number of source, changed, retained, and removed units for every episode.
