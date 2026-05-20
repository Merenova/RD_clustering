#!/usr/bin/env bash
# Run all (model x dataset) combos for the scale-up experiments.
# Each combo invokes scripts/run_pipeline.sh with the right config and an
# output dir of the form {Dataset}_{ModelTag}/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# tag                        config                                              cloze_dir
COMBOS_RAW='
AmbigQA_Gemma3-1B-it       configs/gemma3_1b_it_ambigqa_config.json     data/cloze_llm_improved_split_ratio_0.1
AmbigQA_Gemma3-4B-it       configs/gemma3_4b_it_ambigqa_config.json     data/cloze_llm_improved_split_ratio_0.1
MMLU_Gemma3-1B-it          configs/gemma3_1b_it_mmlu_config.json        data/mmlu_cais_validation_selected
MMLU_Gemma3-4B-it          configs/gemma3_4b_it_mmlu_config.json        data/mmlu_cais_validation_selected
HarmBench_Qwen3-8B         configs/harmbench_Qwen3_8B_config.json       data/harmbench_walledai_standard
HarmBench_Qwen3-4B         configs/harmbench_Qwen3_4B_config.json       data/harmbench_walledai_standard
HarmBench_Gemma3-1B-it     configs/gemma3_1b_it_harmbench_config.json   data/harmbench_walledai_standard
HarmBench_Gemma3-4B-it     configs/gemma3_4b_it_harmbench_config.json   data/harmbench_walledai_standard
'

ensure_harmbench_prepped() {
  local cloze_dir="$1"
  if [[ -f "$cloze_dir/test_clozes.json" ]]; then
    return 0
  fi
  echo ">>> Preparing HarmBench data at $cloze_dir"
  uv run python scripts/prepare_harmbench_questions.py \
    --model Qwen/Qwen3-8B \
    --output "$cloze_dir/test_clozes.json"
}

ONLY=""
LIST=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only) ONLY="$2"; shift 2 ;;
    --list) LIST=true; shift ;;
    -h|--help) echo "Usage: $0 [--only TAG1,TAG2,...] [--list]"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

# Parse combos
declare -a TAGS CONFIGS CLOZE_DIRS
while IFS= read -r line; do
  line="${line## }"
  line="${line%% }"
  [[ -z "$line" ]] && continue
  read -r tag config cloze_dir <<<"$line"
  TAGS+=("$tag")
  CONFIGS+=("$config")
  CLOZE_DIRS+=("$cloze_dir")
done <<<"$COMBOS_RAW"

if $LIST; then
  echo "Available combos:"
  for i in "${!TAGS[@]}"; do
    printf "  %s\n    config: %s\n    cloze_dir: %s\n" "${TAGS[$i]}" "${CONFIGS[$i]}" "${CLOZE_DIRS[$i]}"
  done
  exit 0
fi

# Build the set of tags to run
declare -A WANT
if [[ -n "$ONLY" ]]; then
  IFS=',' read -ra parts <<<"$ONLY"
  for t in "${parts[@]}"; do WANT["$t"]=1; done
else
  for t in "${TAGS[@]}"; do WANT["$t"]=1; done
fi

for i in "${!TAGS[@]}"; do
  tag="${TAGS[$i]}"
  config="${CONFIGS[$i]}"
  cloze_dir="${CLOZE_DIRS[$i]}"
  [[ -z "${WANT[$tag]:-}" ]] && continue

  echo "==> $tag  config=$config  cloze_dir=$cloze_dir"

  if [[ "$cloze_dir" == "data/harmbench_walledai_standard" ]]; then
    ensure_harmbench_prepped "$cloze_dir"
  fi

  out_dir="$ROOT_DIR/$tag"
  mkdir -p "$out_dir/results" "$out_dir/logs"

  CONFIG_FILE="$config" bash "$SCRIPT_DIR/run_pipeline.sh" --output_dir "$out_dir"
done

echo "All requested combos finished."
