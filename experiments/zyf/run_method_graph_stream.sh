#!/usr/bin/env bash
set -euo pipefail

# Start/resume the independent full-graph event-stream answer pass.
# The run directory must already contain immutable questions.json and memory/*
# prepared by the method-run setup step. This script never touches base scores.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${LONGEMO_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RUN_ROOT="${LONGEMO_RUN_ROOT:-$REPO_ROOT/runtime/runs/method_graph_stream_20260922}"
PYTHON_BIN="${LONGEMO_PYTHON:-python3}"
CREDENTIAL_FILE="${LONGEMO_CREDENTIAL_FILE:-$REPO_ROOT/runtime/private/answer.json}"
WORKERS="${LONGEMO_WORKERS:-16}"
SCORE_WORKERS="${LONGEMO_SCORE_WORKERS:-$WORKERS}"
SKIP_SCORE="${LONGEMO_SKIP_SCORE:-0}"
QUESTIONS="$RUN_ROOT/questions.json"
MEMORY_DIR="$RUN_ROOT/memory"
OUTPUT_DIR="${LONGEMO_OUTPUT_DIR:-$RUN_ROOT/answers_full}"
LOG_FILE="$RUN_ROOT/answer_full.log"
PROCESS_FILE="$RUN_ROOT/answer_full_process.json"

[[ -d "$REPO_ROOT" ]] || { echo "missing repo: $REPO_ROOT" >&2; exit 2; }
[[ -f "$QUESTIONS" && -d "$MEMORY_DIR" ]] || {
  echo "prepare immutable questions.json and memory/ under $RUN_ROOT first" >&2
  exit 2
}
[[ -f "$CREDENTIAL_FILE" ]] || { echo "missing credential file" >&2; exit 2; }

# Refuse a duplicate worker, but allow the runner to resume a partial output.
if [[ -f "$PROCESS_FILE" ]]; then
  old_pid="$($PYTHON_BIN - "$PROCESS_FILE" <<'PY'
import json,sys
try: print(json.load(open(sys.argv[1])).get("pid", ""))
except Exception: print("")
PY
)"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null \
     && tr '\0' ' ' < "/proc/$old_pid/cmdline" 2>/dev/null | grep -q 'methods.longemo answer'; then
    echo "already running pid=$old_pid"
    exit 0
  fi
fi

mkdir -p "$RUN_ROOT" "$OUTPUT_DIR"
cd "$REPO_ROOT"

nohup bash -c '
  set -euo pipefail
  exec 9>"$1/answer.lock"
  flock -n 9 || { echo "another method answer worker holds the lock" >&2; exit 0; }
  "$2" -m methods.longemo answer \
    --data-path "$1/questions.json" \
    --output-dir "$1/answers_full" \
    --credential-file "$3" \
    --model gpt-6-astra \
    --base-url https://matrixllm.alipay.com/v1 \
    --memory-dir "$1/memory" \
    --retrieval graph_stream \
    --embedding-backend gemini \
    --embedding-model google/gemini-embedding-2 \
    --embedding-cache-dir "$1/embedding_cache" \
    --workers "$4" \
    --tries 3 \
    --max-inspections 0 \
    --evidence-chars 48000 \
    --top-k 12 \
    --stream-page-size 64
  if [[ "$5" != "1" ]]; then
    export MODEL_API_KEY="$($2 - "$3" <<'PY'
import json,sys
cfg=json.load(open(sys.argv[1]))
print(cfg.get("MODEL_API_KEY", ""), end="")
PY
)"
    "$2" -m evaluation.eval \
      --data-path "$1/questions.json" \
      --predictions "$1/answers_full/predictions.jsonl" \
      --granularity episode \
      --output-dir "$1/scores" \
      --model gpt-6-astra \
      --base-url https://matrixllm.alipay.com/v1 \
      --max-tokens 8192 \
      --timeout 1800 \
      --workers "$6" \
      --tries 3
  fi
' _ "$RUN_ROOT" "$PYTHON_BIN" "$CREDENTIAL_FILE" "$WORKERS" "$SKIP_SCORE" "$SCORE_WORKERS" \
  >"$LOG_FILE" 2>&1 &
pid=$!

$PYTHON_BIN - "$PROCESS_FILE" "$pid" "$WORKERS" "$SCORE_WORKERS" "$RUN_ROOT" <<'PY'
import json,sys,time
path,pid,workers,score_workers,run=sys.argv[1:]
json.dump({"pid":int(pid),"started_unix":time.time(),"workers":int(workers),"score_workers":int(score_workers),"run":run},open(path,"w"),indent=2)
open(path,"a").write("\n")
PY
echo "started method graph-stream answer worker pid=$pid workers=$WORKERS run=$RUN_ROOT"
