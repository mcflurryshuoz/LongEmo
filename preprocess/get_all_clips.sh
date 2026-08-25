#!/usr/bin/env bash
# Cut input clips for every question file under <questions_root>/*/g1_clip/*.json.
#
# Usage:
#   preprocess/get_all_clips.sh [series] [questions_root] [out_dir] [video_root]
#
# series: one show (e.g. friends) or "all" (default) for every show.
# Defaults: all  data  videos/processed/clips  videos/sources
#
#   preprocess/get_all_clips.sh                # every series
#   preprocess/get_all_clips.sh friends        # only friends
#
# Already-cut clips are skipped automatically, so the script is safe to re-run
# after an interruption. A file that fails (e.g. missing source video) does not
# stop the rest; failed files are listed at the end.
set -uo pipefail

# whole body in a function: bash parses the full file before executing,
# so editing this script while a run is in flight cannot corrupt it
main() {

  cd "$(dirname "$0")/.."

  SERIES="${1:-all}"
  QUESTIONS_ROOT="${2:-data}"
  OUT="${3:-videos/processed/clips}"
  VIDEO_ROOT="${4:-videos/sources}"

  if [ "$SERIES" = "all" ]; then
    files=("$QUESTIONS_ROOT"/*/g1_clip/*.json)
  else
    files=("$QUESTIONS_ROOT/$SERIES"/g1_clip/*.json)
  fi
  # Skip aggregate question files; preprocessing uses the episode files directly.
  kept=()
  for f in "${files[@]}"; do
    case "$(basename "$f")" in all.json|pred_*) ;; *) kept+=("$f");; esac
  done
  files=("${kept[@]:-}")
  total=${#files[@]}
  if [ "$total" -eq 0 ] || [ ! -e "${files[0]}" ]; then
    echo "no question files found for series '$SERIES' under $QUESTIONS_ROOT" >&2
    exit 1
  fi

  failed=()
  n=0
  for f in "${files[@]}"; do
    n=$((n + 1))
    echo "=== [$n/$total] $f"
    if ! python3 preprocess/prepare_clips.py \
        --questions "$f" \
        --video-root "$VIDEO_ROOT" \
        --out "$OUT" \
        --keep-going; then
      failed+=("$f")
    fi
  done

  echo
  if [ ${#failed[@]} -gt 0 ]; then
    echo "FAILED files (${#failed[@]}):"
    printf '  %s\n' "${failed[@]}"
    exit 1
  fi
  echo "all done: $total files -> $OUT"
}

main "$@"
