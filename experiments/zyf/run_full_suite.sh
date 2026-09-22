#!/usr/bin/env bash
# Deploy in a new noevent-full checkout. No model request is made by prepare.
set -euo pipefail
stage="${1:-prepare}"
case "$stage" in prepare|run|status) ;; *) echo 'usage: run_full_suite.sh [prepare|run|status]' >&2; exit 2 ;; esac
suite_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
suite_runtime="${LONGEMO_RUNTIME_ROOT:-/root/longemo/runtime}"
suite_metadata="${LONGEMO_FULL_METADATA:-$suite_runtime/data/full558_20260922}"
cd "$suite_repo"
export PATH="$suite_runtime/bin:$PATH"
exec "${LONGEMO_PYTHON:-/opt/conda/bin/python}" -u -m experiments.zyf.full_suite "$stage" \
  --parent-run "${LONGEMO_PARENT_RUN:-$suite_runtime/runs/three_level_pilot50_gemini38_perception_20260922}" \
  --method-repo "${LONGEMO_METHOD_REPO:-/root/longemo/LongEmo-method-3a6c0af}" \
  --noevent-repo "$suite_repo" --runtime "$suite_runtime" \
  --run-name "${LONGEMO_FULL_RUN_NAME:-three_level_full558_gemini38_20260922}" \
  --questions "$suite_metadata/questions.json" --media-manifest "$suite_metadata/media_manifest.json" \
  --producer-done "${LONGEMO_PRODUCER_DONE:-$suite_runtime/transfers/full558_20260922/producer_done.json}" \
  --credential-file "${LONGEMO_CREDENTIAL_FILE:-$suite_runtime/private/answer.json}" \
  --python "${LONGEMO_PYTHON:-/opt/conda/bin/python}" \
  --workers "${LONGEMO_FULL_WORKERS:-12}" --question-workers "${LONGEMO_QUESTION_WORKERS:-2}" \
  --parent-wait-timeout "${LONGEMO_PARENT_WAIT_TIMEOUT:-86400}" \
  --media-wait-timeout "${LONGEMO_MEDIA_WAIT_TIMEOUT:-86400}"
