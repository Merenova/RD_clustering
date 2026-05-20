#!/usr/bin/env bash
# Complete pipeline for Latent Planning experiments
# Run this script to execute all stages sequentially
#
# Note: In the latent_planning pipeline:
# - Stage 2 = Branch Sampling (natural sampling)
# - Stage 3 = Attribution (prefix-to-continuation using attribute_prefix_to_continuations)
#
# Usage:
#   bash run_pipeline.sh [--output_dir DIR] [--resume STAGE] [--only STAGE] [--stages STAGES] [--cross-prefix-batching] [--skip-existing] [--quiet]
#
# Options:
#   --output_dir DIR  Directory for logs and results (default: current directory)
#                     Creates DIR/results/ and DIR/logs/ subdirectories
#
#   --resume STAGE    Resume from stage STAGE, reusing previous results
#                     STAGE can be: 0, 1, 2, 3, 4, 4a, 5, 6, 7, 7a, 7c, 7c1, 7c2, 8
#                     Examples:
#                       --resume 3    will skip stages 1 and 2
#                       --resume 5    will skip stages 1, 2, 3, 4a
#                       --resume 7    will skip stages 1-6, start from validation
#                       --resume 7c2  will skip stages 1-7c1, start from steering phase 2
#
#   --only STAGE      Run only this specific stage (no other stages)
#                     Examples:
#                       --only 5      run only stage 5 (clustering)
#                       --only 7c1    run only stage 7c1 (steering baselines)
#                       --only 7c2    run only stage 7c2 (steering interventions)
#
#   --stages STAGES   Run only the specified stages (comma-separated list)
#                     Examples:
#                       --stages 3,5     run stages 3 and 5
#                       --stages 3,4a,5  run stages 3, 4a, and 5
#                       --stages 7a,7c   run stages 7a and 7c
#
#   --cross-prefix-batching Enable heterogeneous batching across clusters/prefixes for Stage 7c
#   --skip-existing  Skip Stage 7 outputs if they already exist (Stage 7a/7c1/7c2)

set -euo pipefail

# Change to repository root so all relative stage paths resolve correctly
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# Function to convert stage identifier to numeric value for comparison
# Returns: major_stage * 100 + substage (e.g., "4a" -> 401, "7c1" -> 703)
stage_to_number() {
  local stage=$1

  # Handle stage 0 (preprocessing)
  if [[ "$stage" == "0" ]]; then
    echo 0
    return
  fi

  # Handle 7c1, 7c2 special cases
  if [[ "$stage" == "7c1" ]]; then
    echo 703
    return
  elif [[ "$stage" == "7c2" ]]; then
    echo 704
    return
  elif [[ "$stage" == "7c" ]]; then
    # 7c means start from 7c1
    echo 703
    return
  fi

  if [[ "$stage" =~ ^([1-8])([abc]?)$ ]]; then
    local major="${BASH_REMATCH[1]}"
    local minor="${BASH_REMATCH[2]}"

    local minor_num=0
    case "$minor" in
      a) minor_num=1 ;;
      b) minor_num=2 ;;
      # c is handled above for stage 7
      c) minor_num=3 ;;
      *) minor_num=0 ;;
    esac

    echo $((major * 100 + minor_num))
  else
    echo -1
  fi
}

# Function to check if a stage is in the STAGES_LIST
stage_in_list() {
  local stage=$1
  local IFS=','
  for s in $STAGES_LIST; do
    if [[ "$s" == "$stage" ]]; then
      return 0
    fi
  done
  # Special case: allow selecting 7c1/7c2 by specifying "7c" in the stages list.
  if [[ "$stage" == "7c1" || "$stage" == "7c2" ]]; then
    for s in $STAGES_LIST; do
      if [[ "$s" == "7c" ]]; then
        return 0
      fi
    done
  fi
  return 1
}

# Function to check if we should run a stage
should_run_stage() {
  local stage=$1
  local stage_num=$(stage_to_number "$stage")

  # If --stages mode, only run stages in the list
  if [ -n "$STAGES_LIST" ]; then
    stage_in_list "$stage"
  # If --only mode, only run the specified stage
  elif [ -n "$ONLY_STAGE" ]; then
    # Special case: --only 7c should run both 7c1 and 7c2
    if [[ "$ONLY_STAGE" == "7c" && ( "$stage" == "7c1" || "$stage" == "7c2" ) ]]; then
      return 0
    fi
    local only_num=$(stage_to_number "$ONLY_STAGE")
    [[ $stage_num -eq $only_num ]]
  else
    # Normal --resume mode: run this stage and all subsequent
    local resume_num=$(stage_to_number "$RESUME_STAGE")
    [[ $stage_num -ge $resume_num ]]
  fi
}

# Parse command line arguments
RESUME_STAGE="1"
ONLY_STAGE=""
STAGES_LIST=""
OUTPUT_DIR=""
CROSS_PREFIX_BATCHING="false"
SKIP_EXISTING="false"
while [[ $# -gt 0 ]]; do
  case $1 in
    --output_dir|--output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --resume)
      RESUME_STAGE="$2"
      if [[ ! "$RESUME_STAGE" =~ ^(0|[1-8]([ac])?|7c([12])?)$ ]]; then
        echo "Error: --resume must be 0, 1, 2, 3, 4, 4a, 5, 6, 7, 7a, 7c, 7c1, 7c2, or 8"
        echo "Examples:"
        echo "  --resume 1    Start from beginning"
        echo "  --resume 3    Skip stages 1 and 2, start from stage 3"
        echo "  --resume 5    Skip stages 1, 2, 3, 4a, start from stage 5 (clustering)"
        echo "  --resume 7    Skip stages 1-6, start from validation (7a + 7c)"
        echo "  --resume 7c1  Skip stages 1-7a, start from steering baselines"
        echo "  --resume 7c2  Skip stages 1-7c1, start from steering interventions"
        exit 1
      fi
      shift 2
      ;;
    --only)
      ONLY_STAGE="$2"
      if [[ ! "$ONLY_STAGE" =~ ^(0|[1-8]([ac])?|7c([12])?)$ ]]; then
        echo "Error: --only must be 0, 1, 2, 3, 4, 4a, 5, 6, 7, 7a, 7c, 7c1, 7c2, or 8"
        echo "Examples:"
        echo "  --only 5      Run only stage 5 (clustering)"
        echo "  --only 7c1    Run only stage 7c1 (steering baselines)"
        echo "  --only 7c2    Run only stage 7c2 (steering interventions)"
        exit 1
      fi
      shift 2
      ;;
    --stages)
      STAGES_LIST="$2"
      # Validate each stage in the comma-separated list
      IFS=',' read -ra STAGES_ARRAY <<< "$STAGES_LIST"
      for stage in "${STAGES_ARRAY[@]}"; do
        if [[ ! "$stage" =~ ^(0|[1-8]([ac])?|7c([12])?)$ ]]; then
          echo "Error: Invalid stage '$stage' in --stages list"
          echo "Valid stages: 0, 1, 2, 3, 4, 4a, 5, 6, 7, 7a, 7c, 7c1, 7c2, 8"
          echo "Examples:"
          echo "  --stages 3,5      Run stages 3 and 5"
          echo "  --stages 3,4a,5   Run stages 3, 4a, and 5"
          echo "  --stages 7a,7c    Run stages 7a and 7c"
          exit 1
        fi
      done
      shift 2
      ;;
    --quiet)
      QUIET="true"
      shift
      ;;
    --cross-prefix-batching)
      CROSS_PREFIX_BATCHING="true"
      shift
      ;;
    --skip-existing|--skip_existing)
      SKIP_EXISTING="true"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: bash run_pipeline.sh [--output_dir DIR] [--resume STAGE] [--only STAGE] [--stages STAGES] [--cross-prefix-batching] [--skip-existing] [--quiet]"
      exit 1
      ;;
  esac
done

# Helper: return success if directory exists and is non-empty
dir_has_files() {
  local d="$1"
  [[ -d "$d" ]] && find "$d" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null | grep -q .
}

SKIP_EXISTING_ARG=""
if [ "${SKIP_EXISTING:-false}" = "true" ]; then
  SKIP_EXISTING_ARG="--skip-existing"
fi

# Check for conflicting options
if [ -n "$STAGES_LIST" ] && [ -n "$ONLY_STAGE" ]; then
  echo "Error: Cannot use both --stages and --only"
  exit 1
fi
if [ -n "$STAGES_LIST" ] && [ "$RESUME_STAGE" != "1" ]; then
  echo "Error: Cannot use both --stages and --resume"
  exit 1
fi
if [ -n "$ONLY_STAGE" ] && [ "$RESUME_STAGE" != "1" ]; then
  echo "Error: Cannot use both --resume and --only"
  exit 1
fi

# Setup quiet flag for python scripts
QUIET_ARG=""
if [ "${QUIET:-false}" = "true" ]; then
  QUIET_ARG="--quiet"
fi
CROSS_PREFIX_ARG=""
if [ "${CROSS_PREFIX_BATCHING:-false}" = "true" ]; then
  CROSS_PREFIX_ARG="--cross-prefix-batching"
fi

echo "========================================"
echo "LATENT PLANNING PIPELINE"
echo "========================================"
if [ -n "$OUTPUT_DIR" ]; then
  echo "OUTPUT DIRECTORY: $OUTPUT_DIR"
fi
if [ -n "$STAGES_LIST" ]; then
  echo "RUNNING STAGES: $STAGES_LIST"
  echo "(Assuming previous stages already completed)"
elif [ -n "$ONLY_STAGE" ]; then
  echo "RUNNING ONLY STAGE $ONLY_STAGE"
  echo "(Assuming previous stages already completed)"
elif [ "$RESUME_STAGE" != "1" ]; then
  echo "RESUMING FROM STAGE $RESUME_STAGE"
  echo "(Reusing results from earlier stages)"
fi
echo ""

# Default config file
CONFIG_FILE=${CONFIG_FILE:-"configs/default_config.json"}

if [ ! -f "$CONFIG_FILE" ]; then
  echo "Error: Config file not found: $CONFIG_FILE"
  echo "Usage: CONFIG_FILE=configs/my_config.json bash run_pipeline.sh"
  exit 1
fi

echo "Loading configuration from: $CONFIG_FILE"
echo ""

# Parse config with Python
CONFIG_VARS=$(uv run python -c "
import json
import sys

with open('$CONFIG_FILE') as f:
    config = json.load(f)

# Extract configuration values
print(f\"EXPERIMENT_NAME={config['experiment_name']}\")
print(f\"RANDOM_SEED={config['random_seed']}\")

# Global config (new)
global_cfg = config.get('global', {})
print(f\"GLOBAL_BATCH_SIZE={global_cfg.get('batch_size', 128)}\")
print(f\"GLOBAL_MAX_SEQ_LEN={global_cfg.get('max_seq_len', 2048)}\")

# Data config
data = config.get('data', {})
print(f\"CLOZE_DIR={data.get('cloze_dir', '')}\")
print(f\"DATA_SPLIT={data.get('split', 'train')}\")
print(f\"RAW_CLOZE_DIR={data.get('raw_cloze_dir', '')}\")

# Stage 0 (preprocess) config
stage_0 = config.get('stage_0_preprocess', {})
print(f\"PREPROCESS_ENABLED={str(stage_0.get('enabled', False)).lower()}\")
print(f\"PREPROCESS_SKIP_RULE_BASED={str(stage_0.get('skip_rule_based', False)).lower()}\")
print(f\"PREPROCESS_SKIP_LLM={str(stage_0.get('skip_llm', False)).lower()}\")
print(f\"PREPROCESS_SKIP_SPLIT={str(stage_0.get('skip_split', False)).lower()}\")
print(f\"PREPROCESS_LLM_MODEL={stage_0.get('llm_model', 'Qwen/Qwen3-30B-A3B-Instruct-2507-FP8')}\")
print(f\"PREPROCESS_LLM_BATCH_SIZE={stage_0.get('llm_batch_size', 16)}\")
print(f\"PREPROCESS_LLM_GPU_MEMORY={stage_0.get('llm_gpu_memory', 0.8)}\")
print(f\"PREPROCESS_SPLIT_RATIO={stage_0.get('split_ratio', 0.1)}\")
print(f\"PREPROCESS_SPLIT_SEED={stage_0.get('split_seed', 42)}\")

# Model config
model = config['model']
print(f\"MODEL={model['base_model']}\")
print(f\"TRANSCODER={model['transcoder']}\")
print(f\"DTYPE={model['dtype']}\")
print(f\"DEVICE={model['device']}\")

# Sampling config
sampling = config['sampling']
print(f\"MAX_TOTAL_CONTINUATIONS={sampling.get('max_total_continuations', 10000)}\")
print(f\"NUCLEUS_P={sampling['nucleus_p']}\")
print(f\"TEMPERATURE={sampling['temperature']}\")
print(f\"MAX_TOKENS={sampling['max_tokens']}\")
print(f\"SAMPLING_BATCH_SIZE={sampling['batch_size']}\")
print(f\"MAX_BATCHES={sampling['max_batches']}\")
print(f\"MAX_COMPLETE={sampling.get('max_complete', '')}\")


# Embedding config
embedding = config['embedding']
print(f\"EMBEDDING_MODEL={embedding['model_name']}\")
print(f\"EMBEDDING_METHOD={embedding.get('method', 'contextual_continuation')}\")
print(f\"EMBEDDING_BATCH_SIZE={embedding['batch_size']}\")

# Attribution config
attribution = config.get('attribution', {})
print(f\"USE_CIRCUIT_TRACER={str(attribution.get('use_circuit_tracer', True)).lower()}\")
print(f\"ATTRIBUTION_MAX_N_LOGITS={attribution.get('max_n_logits', 10)}\")
print(f\"ATTRIBUTION_DESIRED_LOGIT_PROB={attribution.get('desired_logit_prob', 0.95)}\")
print(f\"ATTRIBUTION_BATCH_SIZE={attribution.get('batch_size', 256)}\")
print(f\"ATTRIBUTION_MAX_FEATURE_NODES={attribution.get('max_feature_nodes', 8192)}\")
print(f\"ATTRIBUTION_OFFLOAD={attribution.get('offload', 'cpu')}\")
print(f\"ATTRIBUTION_STORE_ALL={str(attribution.get('store_all', False)).lower()}\")

# vLLM config (use global.max_seq_len as fallback)
vllm = config.get('vllm', {})
print(f\"GPU_MEMORY_UTILIZATION={vllm.get('gpu_memory_utilization', 0.9)}\")
print(f\"TENSOR_PARALLEL_SIZE={vllm.get('tensor_parallel_size', 1)}\")
max_model_len = vllm.get('max_model_len', global_cfg.get('max_seq_len', 2048))
print(f\"MAX_MODEL_LEN={max_model_len}\")
print(f\"TRUST_REMOTE_CODE={str(vllm.get('trust_remote_code', True)).lower()}\")

# Clustering config
clustering = config['clustering']
print(f\"MAX_ITERATIONS={clustering['max_iterations']}\")
print(f\"CONVERGENCE_THRESHOLD={clustering['convergence_threshold']}\")

# Rate-distortion parameters
print(f\"BETA={clustering.get('beta', 2.0)}\")
print(f\"GAMMA={clustering.get('gamma', 0.5)}\")
print(f\"K_MAX={clustering.get('K_max', 20)}\")
print(f\"K_CLAMP={clustering.get('K_clamp', '')}\")
print(f\"CLUSTERING_POOLING={clustering.get('pooling', 'mean')}\")
print(f\"CLUSTERING_N_PREFIX_WORKERS={clustering.get('n_prefix_workers', 1)}\")
print(f\"CLUSTERING_SAVE_INTERMEDIATE={str(clustering.get('save_intermediate', False)).lower()}\")
print(f\"CLUSTERING_SKIP_EXISTING={str(clustering.get('skip_existing', False)).lower()}\")
print(f\"CLUSTERING_INTERMEDIATE_DIR={clustering.get('intermediate_dir', '')}\")

# Paths config
paths = config.get('paths', {})
print(f\"OUTPUT_BASE={paths.get('output_base', 'outputs')}\")
print(f\"TEST_RESULTS={paths.get('test_results', 'test_results')}\")
print(f\"SEMANTIC_GRAPHS={paths.get('semantic_graphs', 'semantic_graphs')}\")

# Stage 1 config
stage_1 = config.get('stage_1_data_prep', {})
print(f\"STAGE1_MODE={stage_1.get('mode', 'cloze')}\")
print(f\"STAGE1_N_GROUPS={stage_1.get('n_groups', 100)}\")

# Stage clustering config (n_workers for sweep mode)
clustering_sweeps = config.get('clustering', {}).get('sweeps', {})
print(f\"CLUSTERING_N_WORKERS={clustering_sweeps.get('n_workers', 4)}\")

# Stage 7c config
stage_7c = config.get('stage_7c_steering', {})
print(f\"STEERING_ENABLED={str(stage_7c.get('enabled', True)).lower()}\")
print(f\"STEERING_MODE={stage_7c.get('mode', 'two_phase')}\")
# Use global batch_size if max_batch_size is 0 or -1
max_batch_size = stage_7c.get('max_batch_size', -1)
if max_batch_size <= 0:
    max_batch_size = global_cfg.get('batch_size', 128)
print(f\"STEERING_MAX_BATCH_SIZE={max_batch_size}\")
print(f\"STEERING_CROSS_PREFIX_BATCHING={str(stage_7c.get('cross_prefix_batching', False)).lower()}\")
print(f\"STEERING_PREFIX_BATCH_SIZE={stage_7c.get('prefix_batch_size', '')}\")
")

# Source the configuration variables
eval "$CONFIG_VARS"

# Setup results directory
if [ -n "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
  RESULTS_DIR="$OUTPUT_DIR/results"
  LOGS_DIR="$OUTPUT_DIR/logs"
else
  RESULTS_DIR="results"
  LOGS_DIR="logs"
fi
mkdir -p "$RESULTS_DIR"
mkdir -p "$LOGS_DIR"
mkdir -p "$RESULTS_DIR/configs"
mkdir -p "$RESULTS_DIR/manifests"

# Save pipeline configuration
cat > "$RESULTS_DIR/pipeline_config.json" <<EOF
{
  "experiment_name": "$EXPERIMENT_NAME",
  "random_seed": $RANDOM_SEED,
  "config_file": "$CONFIG_FILE",
  "resume_from_stage": "$RESUME_STAGE",
  "only_stage": "$ONLY_STAGE",
  "timestamp": "$(date -Iseconds)",
  "global": {
    "batch_size": $GLOBAL_BATCH_SIZE,
    "max_seq_len": $GLOBAL_MAX_SEQ_LEN
  },
  "model": {
    "base_model": "$MODEL",
    "transcoder": "$TRANSCODER",
    "dtype": "$DTYPE",
    "device": "$DEVICE"
  },
  "data": {
    "cloze_dir": "$CLOZE_DIR",
    "split": "$DATA_SPLIT"
  },
  "attribution": {
    "max_n_logits": $ATTRIBUTION_MAX_N_LOGITS,
    "desired_logit_prob": $ATTRIBUTION_DESIRED_LOGIT_PROB,
    "batch_size": $ATTRIBUTION_BATCH_SIZE,
    "max_feature_nodes": $ATTRIBUTION_MAX_FEATURE_NODES,
    "offload": "$ATTRIBUTION_OFFLOAD"
  },
  "sampling": {
    "nucleus_p": $NUCLEUS_P,
    "temperature": $TEMPERATURE,
    "max_tokens": $MAX_TOKENS,
    "batch_size": $SAMPLING_BATCH_SIZE,
    "max_batches": $MAX_BATCHES
  },
  "embedding": {
    "model_name": "$EMBEDDING_MODEL",
    "method": "$EMBEDDING_METHOD",
    "batch_size": $EMBEDDING_BATCH_SIZE
  },
  "clustering": {
    "beta": $BETA,
    "gamma": $GAMMA,
    "K_max": $K_MAX,
    "K_clamp": ${K_CLAMP:-null},
    "max_iterations": $MAX_ITERATIONS,
    "convergence_threshold": $CONVERGENCE_THRESHOLD
  },
  "vllm": {
    "gpu_memory_utilization": $GPU_MEMORY_UTILIZATION,
    "tensor_parallel_size": $TENSOR_PARALLEL_SIZE,
    "max_model_len": $MAX_MODEL_LEN,
    "trust_remote_code": "$TRUST_REMOTE_CODE"
  },
  "stage_7c_steering": {
    "max_batch_size": $STEERING_MAX_BATCH_SIZE,
    "cross_prefix_batching": $STEERING_CROSS_PREFIX_BATCHING,
    "prefix_batch_size": "$STEERING_PREFIX_BATCH_SIZE"
  }
}
EOF

echo "Saved pipeline configuration to: $RESULTS_DIR/pipeline_config.json"

# Display configuration
echo "Configuration:"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Model: $MODEL"
echo "  Results directory: $RESULTS_DIR/"
echo ""
echo "Global Settings:"
echo "  Batch size: $GLOBAL_BATCH_SIZE"
echo "  Max sequence length: $GLOBAL_MAX_SEQ_LEN"
echo ""
echo "Data Preparation (Stage 1):"
echo "  Mode: $STAGE1_MODE"
echo "  Number of groups: $STAGE1_N_GROUPS"
echo ""
echo "Branch Sampling (Stage 2):"
echo "  Max logits: $ATTRIBUTION_MAX_N_LOGITS"
echo "  Desired logit prob: $ATTRIBUTION_DESIRED_LOGIT_PROB"
echo ""
echo "Attribution (Stage 3):"
echo "  Span mode: full"
echo "  Batch size: $ATTRIBUTION_BATCH_SIZE"
echo "  Max feature nodes: $ATTRIBUTION_MAX_FEATURE_NODES"
echo ""
echo "Continuation Sampling:"
echo "  Max total continuations: $MAX_TOTAL_CONTINUATIONS"
echo "  Nucleus p: $NUCLEUS_P"
echo "  Temperature: $TEMPERATURE"
echo "  Batch size: $SAMPLING_BATCH_SIZE"
echo "  Max batches: $MAX_BATCHES"
echo ""
echo "Clustering (Stage 5):"
echo "  beta: $BETA, gamma: $GAMMA"
echo "  K_max: $K_MAX (deprecated - clustering converges naturally)"
echo "  K_clamp: ${K_CLAMP:-'(not set, defaults to K_max)'}"
echo ""
echo "vLLM Configuration:"
echo "  GPU memory utilization: $GPU_MEMORY_UTILIZATION"
echo "  Tensor parallel size: $TENSOR_PARALLEL_SIZE"
echo "  Max model length: $MAX_MODEL_LEN"
echo "  Trust remote code: $TRUST_REMOTE_CODE"
echo ""
echo "Steering Validation (Stage 7c):"
echo "  Max batch size: $STEERING_MAX_BATCH_SIZE"
echo "  Cross-prefix batching: $STEERING_CROSS_PREFIX_BATCHING"
echo ""

# Stage 0: Preprocessing (optional - creates processed cloze dataset)
if should_run_stage "0"; then
  if [ "$PREPROCESS_ENABLED" = "true" ]; then
    echo "========================================"
    echo "STAGE 0: DATA PREPROCESSING"
    echo "========================================"

    # Define intermediate paths
    PREPROCESS_DIR="$RESULTS_DIR/0_preprocess"
    IMPROVED_DIR="$PREPROCESS_DIR/cloze_improved"
    LLM_IMPROVED_DIR="$PREPROCESS_DIR/cloze_llm_improved"

    mkdir -p "$PREPROCESS_DIR"

    # Step 0a: Rule-based cloze improvement
    if [ "$PREPROCESS_SKIP_RULE_BASED" != "true" ]; then
      echo "Step 0a: Rule-based cloze improvement..."
      uv run python 0_preprocess/improve_clozes.py \
        --input "$RAW_CLOZE_DIR" \
        --output "$IMPROVED_DIR" \
        --log-dir "$LOGS_DIR" \
        $QUIET_ARG
      echo ""
    else
      echo "Skipping Step 0a (rule-based improvement)"
      IMPROVED_DIR="$RAW_CLOZE_DIR"
    fi

    # Step 0b: LLM-based cloze improvement
    if [ "$PREPROCESS_SKIP_LLM" != "true" ]; then
      echo "Step 0b: LLM-based cloze improvement..."
      uv run python 0_preprocess/llm_improve_clozes.py \
        --input "$IMPROVED_DIR" \
        --output "$LLM_IMPROVED_DIR" \
        --model "$PREPROCESS_LLM_MODEL" \
        --batch-size "$PREPROCESS_LLM_BATCH_SIZE" \
        --gpu-memory "$PREPROCESS_LLM_GPU_MEMORY" \
        --log-dir "$LOGS_DIR" \
        $QUIET_ARG
      echo ""
    else
      echo "Skipping Step 0b (LLM improvement)"
      LLM_IMPROVED_DIR="$IMPROVED_DIR"
    fi

    # Step 0c: Dataset splitting
    if [ "$PREPROCESS_SKIP_SPLIT" != "true" ]; then
      echo "Step 0c: Dataset splitting..."
      uv run python 0_preprocess/split_dataset.py \
        --input "$LLM_IMPROVED_DIR" \
        --output "$CLOZE_DIR" \
        --keep-ratio "$PREPROCESS_SPLIT_RATIO" \
        --seed "$PREPROCESS_SPLIT_SEED" \
        --log-dir "$LOGS_DIR" \
        $QUIET_ARG
      echo ""
    else
      echo "Skipping Step 0c (dataset splitting)"
    fi

    # Save stage config
    cat > "$RESULTS_DIR/configs/stage_0_config.json" <<EOF
{
  "stage": "0_preprocess",
  "timestamp": "$(date -Iseconds)",
  "parameters": {
    "raw_cloze_dir": "$RAW_CLOZE_DIR",
    "output_cloze_dir": "$CLOZE_DIR",
    "llm_model": "$PREPROCESS_LLM_MODEL",
    "split_ratio": $PREPROCESS_SPLIT_RATIO,
    "split_seed": $PREPROCESS_SPLIT_SEED
  }
}
EOF
    echo ""
  else
    echo "Stage 0 (Preprocessing) is disabled in config"
    echo ""
  fi
else
  echo "Skipping Stage 0 (Preprocessing) - using existing processed data"
  echo ""
fi

# Stage 1: Data Preparation
if should_run_stage "1"; then
  echo "========================================"
  echo "STAGE 1: DATA PREPARATION"
  echo "========================================"
  # Build stage 1 command
  STAGE1_CMD="uv run python 1_data_preparation/select_test_clozes.py \
    --cloze-dir \"$CLOZE_DIR\" \
    --split \"$DATA_SPLIT\" \
    --n-samples \"$STAGE1_N_GROUPS\" \
    --seed \"$RANDOM_SEED\" \
    --mode \"$STAGE1_MODE\" \
    --output \"$RESULTS_DIR/test_clozes.json\" \
    $QUIET_ARG"

  # Add model argument if question mode
  if [ "$STAGE1_MODE" = "question" ]; then
    STAGE1_CMD="$STAGE1_CMD --model \"$MODEL\""
  fi

  eval "$STAGE1_CMD"

  # Save stage config
  cat > "$RESULTS_DIR/configs/stage_1_config.json" <<EOF
{
  "stage": "1_data_preparation",
  "timestamp": "$(date -Iseconds)",
  "parameters": {
    "cloze_dir": "$CLOZE_DIR",
    "split": "$DATA_SPLIT",
    "n_groups": $STAGE1_N_GROUPS,
    "random_seed": $RANDOM_SEED,
    "mode": "$STAGE1_MODE",
    "model": "$MODEL"
  }
}
EOF
  echo ""
else
  echo "Skipping Stage 1 (Data Preparation) - using existing results"
  echo ""
fi

# Stage 2: Branch Sampling (natural)
# NOTE: In latent_planning, Stage 2 performs natural top-p sampling from the prefix
if should_run_stage "2"; then
  echo "========================================"
  echo "STAGE 2: BRANCH SAMPLING (NATURAL)"
  echo "========================================"
  echo "Sampling continuations (natural)..."
  uv run python 2_branch_sampling/sample_branches.py \
    --test-clozes "$RESULTS_DIR/test_clozes.json" \
    --model "$MODEL" \
    --max-total-continuations "$MAX_TOTAL_CONTINUATIONS" \
    --nucleus-p "$NUCLEUS_P" \
    --temperature "$TEMPERATURE" \
    --max-tokens "$MAX_TOKENS" \
    --batch-size "$SAMPLING_BATCH_SIZE" \
    --max-batches "$MAX_BATCHES" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    $([ "$TRUST_REMOTE_CODE" = "true" ] && echo "--trust-remote-code") \
    $([ -n "$MAX_COMPLETE" ] && echo "--max-complete $MAX_COMPLETE") \
    --output-dir "$RESULTS_DIR/2_branch_sampling/" \
    $QUIET_ARG

  # Save stage config
  cat > "$RESULTS_DIR/configs/stage_2_config.json" <<EOF
{
  "stage": "2_branch_sampling",
  "timestamp": "$(date -Iseconds)",
  "parameters": {
    "model": "$MODEL",
    "max_total_continuations": $MAX_TOTAL_CONTINUATIONS,
    "nucleus_p": $NUCLEUS_P,
    "temperature": $TEMPERATURE,
    "max_tokens": $MAX_TOKENS,
    "batch_size": $SAMPLING_BATCH_SIZE,
    "max_batches": $MAX_BATCHES,
    "gpu_memory_utilization": $GPU_MEMORY_UTILIZATION,
    "tensor_parallel_size": $TENSOR_PARALLEL_SIZE,
    "max_model_len": $MAX_MODEL_LEN,
    "trust_remote_code": "$TRUST_REMOTE_CODE"
  }
}
EOF
  echo ""
else
  echo "Skipping Stage 2 (Branch Sampling) - using existing results"
  echo ""
fi

# Stage 3: Attribution (prefix-to-continuation)
# NOTE: In latent_planning, Stage 3 computes attribution using attribute_prefix_to_continuations
if should_run_stage "3"; then
  echo "========================================"
  echo "STAGE 3: ATTRIBUTION GRAPHS"
  echo "========================================"
  echo "Computing prefix-to-continuation attribution..."

  # Build store-all flag
  STORE_ALL_FLAG=""
  if [ "$ATTRIBUTION_STORE_ALL" = "true" ]; then
    STORE_ALL_FLAG="--store-all"
  fi

  # Backend dispatch: read attribution.backend from config (default: auto)
  BACKEND="$(uv run python -c "import json,sys; c=json.load(open('${CONFIG_FILE}')); print(c.get('attribution',{}).get('backend','auto'))")"

  uv run python 3_attribution_graphs/compute_continuation_attribution.py \
    --branches-dir "$RESULTS_DIR/2_branch_sampling/" \
    --model "$MODEL" \
    --transcoder "$TRANSCODER" \
    --dtype "$DTYPE" \
    --max-feature-nodes "$ATTRIBUTION_MAX_FEATURE_NODES" \
    --batch-size "$ATTRIBUTION_BATCH_SIZE" \
    --output-dir "$RESULTS_DIR/3_attribution_graphs/" \
    --backend "$BACKEND" \
    $STORE_ALL_FLAG \
    $QUIET_ARG

  # Save stage config
  cat > "$RESULTS_DIR/configs/stage_3_config.json" <<EOF
{
  "stage": "3_attribution_graphs",
  "timestamp": "$(date -Iseconds)",
  "parameters": {
    "model": "$MODEL",
    "transcoder": "$TRANSCODER",
    "dtype": "$DTYPE",
    "span_mode": "full",
    "store_all": $ATTRIBUTION_STORE_ALL,
    "max_feature_nodes": $ATTRIBUTION_MAX_FEATURE_NODES,
    "batch_size": $ATTRIBUTION_BATCH_SIZE
  }
}
EOF
  echo ""
else
  echo "Skipping Stage 3 (Attribution) - using existing results"
  echo ""
fi

# Stage 4: Feature Extraction (4a)
# Stage 4a: Compute Embeddings
if should_run_stage "4a"; then
  echo "========================================"
  echo "STAGE 4A: COMPUTE EMBEDDINGS"
  echo "========================================"
  echo "NOTE: Using selected continuations from branch sampling"
  uv run python 4_feature_extraction/compute_embeddings.py \
    --samples-dir "$RESULTS_DIR/2_branch_sampling/" \
    --embedding-model "$EMBEDDING_MODEL" \
    --batch-size "$EMBEDDING_BATCH_SIZE" \
    --device "$DEVICE" \
    --output-dir "$RESULTS_DIR/4_feature_extraction/embeddings/" \
    $QUIET_ARG

  # Save stage config
  cat > "$RESULTS_DIR/configs/stage_4a_config.json" <<EOF
{
  "stage": "4a_compute_embeddings",
  "timestamp": "$(date -Iseconds)",
  "parameters": {
    "embedding_model": "$EMBEDDING_MODEL",
    "batch_size": $EMBEDDING_BATCH_SIZE,
    "device": "$DEVICE",
    "method": "$EMBEDDING_METHOD"
  }
}
EOF
  echo ""
else
  echo "Skipping Stage 4a (Compute Embeddings) - using existing results"
  echo ""
fi

# Stage 5: Gaussian Clustering
if should_run_stage "5"; then
  echo "========================================"
  echo "STAGE 5: GAUSSIAN CLUSTERING"
  echo "========================================"

  uv run python 5_gaussian_clustering/cluster.py \
    --embeddings-dir "$RESULTS_DIR/4_feature_extraction/embeddings/" \
    --attribution-graphs-dir "$RESULTS_DIR/3_attribution_graphs/" \
    --samples-dir "$RESULTS_DIR/2_branch_sampling/" \
    --beta "$BETA" \
    --gamma "$GAMMA" \
    --K-max "$K_MAX" \
    $([ -n "$K_CLAMP" ] && echo "--K-clamp $K_CLAMP") \
    --max-iterations "$MAX_ITERATIONS" \
    --convergence-threshold "$CONVERGENCE_THRESHOLD" \
    --pooling "$CLUSTERING_POOLING" \
    --n-workers "$CLUSTERING_N_PREFIX_WORKERS" \
    --config "$CONFIG_FILE" \
    --save-intermediate \
    --intermediate-dir "$RESULTS_DIR/5_clustering/intermediate" \
    --output-dir "$RESULTS_DIR/5_clustering/" \
    --log-dir "$LOGS_DIR" \
    $([ "$CLUSTERING_SKIP_EXISTING" = "true" ] && echo "--skip-existing") \
    $QUIET_ARG

  # Save stage config
  cat > "$RESULTS_DIR/configs/stage_5_config.json" <<EOF
{
  "stage": "5_gaussian_clustering",
  "timestamp": "$(date -Iseconds)",
  "parameters": {
    "beta": $BETA,
    "gamma": $GAMMA,
    "K_max": $K_MAX,
    "K_clamp": ${K_CLAMP:-null},
    "max_iterations": $MAX_ITERATIONS,
    "convergence_threshold": $CONVERGENCE_THRESHOLD,
    "pooling": "$CLUSTERING_POOLING",
    "save_intermediate": true,
    "intermediate_dir": "$RESULTS_DIR/5_clustering/intermediate"
  }
}
EOF
  echo ""
else
  echo "Skipping Stage 5 (Gaussian Clustering) - using existing results"
  echo ""
fi

# Stage 6: Semantic Graph Extraction
if should_run_stage "6"; then
  echo "========================================"
  echo "STAGE 6: SEMANTIC GRAPH EXTRACTION"
  echo "========================================"
  uv run python 6_semantic_graphs/extract_graphs.py \
    --clustering-dir "$RESULTS_DIR/5_clustering/" \
    --samples-dir "$RESULTS_DIR/2_branch_sampling/" \
    --attribution-graphs-dir "$RESULTS_DIR/3_attribution_graphs/" \
    --output-dir "$RESULTS_DIR/6_semantic_graphs/" \
    --log-dir "$LOGS_DIR" \
    $QUIET_ARG

  # Save stage config
  cat > "$RESULTS_DIR/configs/stage_6_config.json" <<EOF
{
  "stage": "6_semantic_graphs",
  "timestamp": "$(date -Iseconds)",
  "parameters": {}
}
EOF
  echo ""
else
  echo "Skipping Stage 6 (Semantic Graph Extraction) - using existing results"
  echo ""
fi

# Stage 7a: Graph Validation
if should_run_stage "7a"; then
  STAGE7A_OUTDIR="$RESULTS_DIR/7_validation/7a_graph_validation/"
  if [ "$SKIP_EXISTING" = "true" ] && dir_has_files "$STAGE7A_OUTDIR"; then
    echo "Skipping Stage 7a (Graph Validation) - output exists: $STAGE7A_OUTDIR"
    echo ""
  else
    echo "========================================"
    echo "STAGE 7a: GRAPH VALIDATION"
    echo "========================================"
    uv run python 7_validation/7a_graph_validation.py \
      --attribution-graphs-dir "$RESULTS_DIR/3_attribution_graphs/" \
      --output-dir "$STAGE7A_OUTDIR" \
      --log-dir "$LOGS_DIR" \
      $QUIET_ARG

    echo ""
  fi
else
  echo "Skipping Stage 7a (Graph Validation) - using existing results"
  echo ""
fi

# Stage 7c1: Steering Validation (RD clusters)
# Uses on-the-fly steering with column-wise encoder (computes activations dynamically)
if [ "$STEERING_ENABLED" = "true" ]; then
  if should_run_stage "7c1"; then
    STAGE7C1_OUTDIR="$RESULTS_DIR/7_validation/7c_steering/"
    echo "========================================"
    echo "STAGE 7c1: STEERING VALIDATION (RD, on-the-fly)"
    echo "========================================"
    
    # Determine cross-prefix batching flag
    CROSS_PREFIX_FLAG=""
    if [ "${CROSS_PREFIX_BATCHING:-false}" = "true" ] || [ "$STEERING_CROSS_PREFIX_BATCHING" = "true" ]; then
      CROSS_PREFIX_FLAG="--cross-prefix-batching"
    fi
    
    uv run python 7_validation/7c_hypotheses.py \
      --attribution-graphs-dir "$RESULTS_DIR/3_attribution_graphs/" \
      --samples-dir "$RESULTS_DIR/2_branch_sampling/" \
      --clustering-dir "$RESULTS_DIR/5_clustering/" \
      --output-dir "$STAGE7C1_OUTDIR" \
      --config "$CONFIG_FILE" \
      --log-dir "$LOGS_DIR" \
      $SKIP_EXISTING_ARG \
      $CROSS_PREFIX_FLAG $QUIET_ARG
    echo ""
  else
    echo "Skipping Stage 7c1 (Steering Validation) - using existing results"
    echo ""
  fi

  STEERING_METHOD_USED="on_the_fly"

  # Stage 7c2: K-means baseline steering (semantic-only clustering)
  if should_run_stage "7c2"; then
    STAGE7C2_OUTDIR="$RESULTS_DIR/7_validation/7c_baseline_kmeans"
    echo "========================================"
    echo "STAGE 7c2: K-MEANS BASELINE (semantic-only clustering)"
    echo "========================================"

    CROSS_PREFIX_FLAG=""
    if [ "${CROSS_PREFIX_BATCHING:-false}" = "true" ] || [ "$STEERING_CROSS_PREFIX_BATCHING" = "true" ]; then
      CROSS_PREFIX_FLAG="--cross-prefix-batching"
    fi

    uv run python 7_validation/7c_baseline_kmeans.py \
      --samples-dir "$RESULTS_DIR/2_branch_sampling" \
      --embeddings-dir "$RESULTS_DIR/4_feature_extraction/embeddings" \
      --attribution-graphs-dir "$RESULTS_DIR/3_attribution_graphs" \
      --clustering-dir "$RESULTS_DIR/5_clustering" \
      --output-dir "$STAGE7C2_OUTDIR" \
      --config "$CONFIG_FILE" \
      $SKIP_EXISTING_ARG \
      $CROSS_PREFIX_FLAG \
      --prefix-batch-size "$STEERING_PREFIX_BATCH_SIZE" \
      --max-batch-size "$STEERING_MAX_BATCH_SIZE" \
      --log-dir "$LOGS_DIR" \
      $QUIET_ARG
    echo ""
  else
    echo "Skipping Stage 7c2 (K-means baseline) - using existing results"
    echo ""
  fi

  # Save stage config (include whichever substages were requested)
  STAGE7_SUBSTAGES=("7a_graph_validation")
  if should_run_stage "7c1"; then
    STAGE7_SUBSTAGES+=("7c1_steering_rd")
  fi
  if should_run_stage "7c2"; then
    STAGE7_SUBSTAGES+=("7c2_baseline_kmeans")
  fi
  export STAGE7_SUBSTAGES="$(IFS=','; echo "${STAGE7_SUBSTAGES[*]}")"
  SUBSTAGES_JSON=$(uv run python - <<'PY'
import json, os
subs = os.environ.get("STAGE7_SUBSTAGES", "")
arr = [s for s in subs.split(",") if s]
print(json.dumps(arr))
PY
)
  cat > "$RESULTS_DIR/configs/stage_7_config.json" <<EOF
{
  "stage": "7_validation",
  "timestamp": "$(date -Iseconds)",
  "substages": $SUBSTAGES_JSON,
  "parameters": {
    "steering_enabled": true,
    "steering_mode": "on_the_fly",
    "max_batch_size": $STEERING_MAX_BATCH_SIZE,
    "cross_prefix_batching": ${STEERING_CROSS_PREFIX_BATCHING:-false},
    "prefix_batch_size": "$STEERING_PREFIX_BATCH_SIZE"
  }
}
EOF

else
  echo "Skipping Stage 7c (Steering) - disabled in config"
  echo ""

  # Save stage config without steering
  cat > "$RESULTS_DIR/configs/stage_7_config.json" <<EOF
{
  "stage": "7_validation",
  "timestamp": "$(date -Iseconds)",
  "substages": ["7a_graph_validation"],
  "parameters": {
    "steering_enabled": false,
    "cross_prefix_batching": false
  }
}
EOF
fi

# Stage 8: Visualization (consolidated, organized by source stage)
if should_run_stage "8"; then
  echo "========================================"
  echo "STAGE 8: VISUALIZATION"
  echo "========================================"
  echo "Generating visualizations organized by source stage:"
  echo "  - 5_clustering/: t-SNE, Sankey, clustering history"
  echo "  - 6_semantic_graphs/: heatmaps, token scores"
  echo "  - parameter_sweep/: sweep analysis (if enabled)"
  echo "  - interactive/: HTML explorers"
  echo ""

  uv run python 8_visualization/visualize.py \
    --config "$CONFIG_FILE" \
    --output-dir "$RESULTS_DIR/8_visualization/" \
    --sweep-results-dir "$RESULTS_DIR/5_clustering/" \
    --log-dir "$LOGS_DIR" \
    $QUIET_ARG

  # Save stage config
  cat > "$RESULTS_DIR/configs/stage_8_config.json" <<EOF
{
  "stage": "8_visualization",
  "timestamp": "$(date -Iseconds)",
  "parameters": {
    "output_structure": {
      "5_clustering": ["tsne_embedding", "tsne_attribution", "tsne_comparison", "clustering_history"],
      "6_semantic_graphs": ["semantic_graph_heatmap"],
      "parameter_sweep": ["harmonic_scores", "rd_curves", "heatmaps", "summary"],
      "interactive": ["cluster_explorer_html"]
    }
  }
}
EOF
  echo ""
else
  echo "Skipping Stage 8 (Visualization) - using existing results"
  echo ""
fi

echo ""
echo "========================================"
echo "PIPELINE COMPLETE!"
echo "========================================"
echo ""
echo "All results saved in: $RESULTS_DIR/"
echo "All logs saved in: $LOGS_DIR/"
echo ""
echo "Results structure:"
echo "  - Test clozes: $RESULTS_DIR/test_clozes.json"
echo "  - Branch samples: $RESULTS_DIR/2_branch_sampling/"
echo "  - Attribution graphs: $RESULTS_DIR/3_attribution_graphs/"
echo "  - Feature extraction: $RESULTS_DIR/4_feature_extraction/"
echo "  - Clustering: $RESULTS_DIR/5_clustering/"
echo "  - Semantic graphs: $RESULTS_DIR/6_semantic_graphs/"
echo "  - Validation: $RESULTS_DIR/7_validation/"
echo "    - Graph validation: $RESULTS_DIR/7_validation/7a_graph_validation/"
echo "    - Steering: $RESULTS_DIR/7_validation/7c_steering/"
echo "  - Visualizations: $RESULTS_DIR/8_visualization/"
echo "    - Clustering plots: $RESULTS_DIR/8_visualization/5_clustering/"
echo "    - Semantic graphs: $RESULTS_DIR/8_visualization/6_semantic_graphs/"
echo "    - Parameter sweep: $RESULTS_DIR/8_visualization/parameter_sweep/"
echo "    - Interactive HTML: $RESULTS_DIR/8_visualization/interactive/"
echo "  - Stage configs: $RESULTS_DIR/configs/"
echo ""
echo "To view summaries:"
echo "  cat $RESULTS_DIR/5_clustering/clustering_summary.json | jq ."
echo ""
