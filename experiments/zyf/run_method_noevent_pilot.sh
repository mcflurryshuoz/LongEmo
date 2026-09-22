#!/usr/bin/env bash
set -u

# Start matched method/noevent perception and answer stages.  The two source
# checkouts and run directories are intentionally separate; only questions,
# media sampling and service configuration are shared.
ROOT="${LONGEMO_RUNTIME_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../runtime" 2>/dev/null || echo runtime)}"
METHOD_REPO="${LONGEMO_METHOD_REPO:-$ROOT/../LongEmo-method-v2}"
NOEVENT_REPO="${LONGEMO_NOEVENT_REPO:-$ROOT/../LongEmo-noevent}"
QUESTIONS="${LONGEMO_QUESTIONS:-$ROOT/runs/method_hybrid_pilot_v6_20260922/questions.json}"
CREDENTIAL="${LONGEMO_CREDENTIAL_FILE:-$ROOT/private/answer.json}"
PYTHON_BIN="${LONGEMO_PYTHON:-python3}"
MODEL="${LONGEMO_MODEL:-gpt-6-astra}"
BASE_URL="${LONGEMO_BASE_URL:-https://matrixllm.alipay.com/v1}"
AUDIO_MODEL="${LONGEMO_AUDIO_MODEL:-gemini-3.8-flash}"
AUDIO_BASE_URL="${LONGEMO_AUDIO_BASE_URL:-https://www.blackaicoding.com/v1beta}"
WORKERS="${LONGEMO_BUILD_WORKERS:-6}"

for path in "$QUESTIONS" "$CREDENTIAL" "$METHOD_REPO" "$NOEVENT_REPO"; do
  [[ -e "$path" ]] || { echo "missing required path: $path" >&2; exit 2; }
done

METHOD_RUN="${LONGEMO_METHOD_RUN:-$ROOT/runs/method_event_v2_pilot_$(date +%Y%m%d_%H%M%S)}"
NOEVENT_RUN="${LONGEMO_NOEVENT_RUN:-$ROOT/runs/noevent_window_pilot_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$METHOD_RUN" "$NOEVENT_RUN"
cp -n "$QUESTIONS" "$METHOD_RUN/questions.json"
cp -n "$QUESTIONS" "$NOEVENT_RUN/questions.json"

common=(--data-path "QUESTIONS" --videos-dir "$ROOT/data/episode/videos"
  --subtitles-dir "$ROOT/data/prepared_subtitles" --credential-file "$CREDENTIAL"
  --model "$MODEL" --base-url "$BASE_URL" --max-tokens 8192 --timeout 1800
  --tries 3 --workers "$WORKERS" --with-audio --audio-model "$AUDIO_MODEL"
  --audio-base-url "$AUDIO_BASE_URL" --window-seconds 20 --padding 2 --fps 1
  --max-frames 24 --max-pixels 200704)

start_build() {
  local label="$1" repo="$2" run="$3" module="$4"
  local command_file="$run/start_build.sh"
  {
    echo '#!/usr/bin/env bash'; echo 'set -u';
    echo 'export PATH="'"${ROOT}"'/bin:$PATH"'; echo 'cd "'"$repo"'"';
    printf 'exec %q -u -m %q build' "$PYTHON_BIN" "$module"
    for arg in "${common[@]}"; do
      if [[ "$arg" == QUESTIONS ]]; then printf ' %q' "$run/questions.json";
      else printf ' %q' "$arg"; fi
    done
    printf ' --output-dir %q\n' "$run/memory"
  } > "$command_file"
  chmod +x "$command_file"
  nohup "$command_file" >"$run/build.log" 2>&1 &
  local pid=$!
  printf '{"stage":"build","pid":%s,"run":"%s","module":"%s"}\n' "$pid" "$run" "$module" > "$run/build_process.json"
  echo "$label build pid=$pid run=$run"
}

start_build method "$METHOD_REPO" "$METHOD_RUN" methods.longemo
start_build noevent "$NOEVENT_REPO" "$NOEVENT_RUN" methods.longemo.noevent_runner
