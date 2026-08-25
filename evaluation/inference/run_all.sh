#!/usr/bin/env bash
# Run one inference backend over a whole series or every series in one granularity.
#
# Usage:
#   evaluation/inference/run_all.sh <gemini|qwen_omni|infer_text> <series|all> <g1_clip|g2_episode> [backend arguments...]
set -uo pipefail

main() {
  if [ "$#" -lt 3 ]; then
    echo "usage: $0 <gemini|qwen_omni|infer_text> <series|all> <g1_clip|g2_episode> [backend arguments...]" >&2
    exit 2
  fi

  cd "$(dirname "$0")/../.."

  backend="$1"
  series="$2"
  granularity="$3"
  shift 3

  case "$backend" in
    gemini|qwen_omni|infer_text) ;;
    *) echo "unsupported backend: $backend" >&2; exit 2 ;;
  esac
  case "$granularity" in
    g1_clip|g2_episode) ;;
    *) echo "unsupported granularity: $granularity" >&2; exit 2 ;;
  esac
  if [ "$backend" = "infer_text" ] && [ "$granularity" != "g2_episode" ]; then
    echo "infer_text supports only g2_episode" >&2
    exit 2
  fi

  if [ "$series" = "all" ]; then
    question_files=(data/*/"$granularity"/all.json)
  else
    question_files=(data/"$series"/"$granularity"/all.json)
  fi

  if [ ! -e "${question_files[0]}" ]; then
    echo "no aggregate question files found for $series/$granularity" >&2
    exit 1
  fi

  if [ "${#question_files[@]}" -gt 1 ]; then
    for arg in "$@"; do
      if [ "$arg" = "--out" ]; then
        echo "--out cannot be shared when processing all series; let each run use its default output path" >&2
        exit 2
      fi
    done
  fi

  failed=()
  total=${#question_files[@]}
  for i in "${!question_files[@]}"; do
    file="${question_files[$i]}"
    show="$(basename "$(dirname "$(dirname "$file")")")"
    echo "=== [$((i + 1))/$total] $show/$granularity"
    if ! PYTHONPATH=. python3 "evaluation/inference/${backend}.py" --questions "$file" "$@"; then
      failed+=("$show")
    fi
  done

  if [ "${#failed[@]}" -gt 0 ]; then
    echo "failed series (${#failed[@]}): ${failed[*]}" >&2
    exit 1
  fi
  echo "all done: $total series"
}

main "$@"
