#!/usr/bin/env bash
# Prepare effective full episodes for every series represented in g2_episode data.
#
# Usage:
#   preprocess/get_all_episodes.sh \
#     [series] [questions_root] [video_root] [out_dir] [ranges_root]
#
# Defaults:
#   series=all
#   questions_root=data
#   video_root=videos/sources
#   out_dir=videos/processed/episodes
#   ranges_root=<video_root>
#
# Examples:
#   preprocess/get_all_episodes.sh
#   preprocess/get_all_episodes.sh friends
#
# Videos and valid_ranges.json share one series directory in the released layout.
set -uo pipefail

main() {
  cd "$(dirname "$0")/.."

  SERIES="${1:-all}"
  QUESTIONS_ROOT="${2:-data}"
  VIDEO_ROOT="${3:-videos/sources}"
  OUT="${4:-videos/processed/episodes}"
  RANGES_ROOT="${5:-$VIDEO_ROOT}"

  selected=()
  if [ "$SERIES" = "all" ]; then
    for episode_dir in "$QUESTIONS_ROOT"/*/g2_episode; do
      if [ -d "$episode_dir" ]; then
        selected+=("$(basename "$(dirname "$episode_dir")")")
      fi
    done
  elif [ -d "$QUESTIONS_ROOT/$SERIES/g2_episode" ]; then
    selected+=("$SERIES")
  else
    echo "no g2_episode data found for series '$SERIES' under $QUESTIONS_ROOT" >&2
    exit 1
  fi

  total=${#selected[@]}
  if [ "$total" -eq 0 ]; then
    echo "no series with g2_episode data found under $QUESTIONS_ROOT" >&2
    exit 1
  fi

  failed=()
  n=0
  for series in "${selected[@]}"; do
    n=$((n + 1))
    manual="$RANGES_ROOT/$series/valid_ranges.json"
    automatic="$RANGES_ROOT/$series/valid_input_ranges.json"
    if [ -f "$manual" ]; then
      ranges="$manual"
    elif [ -f "$automatic" ]; then
      ranges="$automatic"
    else
      echo "=== [$n/$total] $series: MISSING ranges file" >&2
      failed+=("$series (missing ranges)")
      continue
    fi

    echo "=== [$n/$total] $series"
    echo "ranges: $ranges"
    if ! python3 preprocess/prepare_episodes.py \
        --ranges "$ranges" \
        --video-root "$VIDEO_ROOT" \
        --out "$OUT" \
        --keep-going; then
      failed+=("$series")
    fi
  done

  echo
  if [ ${#failed[@]} -gt 0 ]; then
    echo "FAILED series (${#failed[@]}):" >&2
    printf '  %s\n' "${failed[@]}" >&2
    exit 1
  fi
  echo "all done: $total series -> $OUT"
}

main "$@"
