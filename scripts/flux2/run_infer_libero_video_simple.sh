#!/usr/bin/env bash
# Simplified LIBERO video rollout inference script.
# This script doesn't require common.sh and works with direct environment variables.

set -euo pipefail

# Set GPU
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

# Project paths
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Default paths (can be overridden by environment variables)
FLUX2_SRC="${FLUX2_SRC:-${REPO_ROOT}/third_party/flux2}"
FLUX2_MODEL_PATH="${FLUX2_MODEL_PATH:-${REPO_ROOT}/checkpoints/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors}"
FLUX2_AE_MODEL_PATH="${FLUX2_AE_MODEL_PATH:-${REPO_ROOT}/checkpoints/flux2/FLUX.2-dev/ae.safetensors}"
FLUX2_QWEN3_MODEL_SPEC="${FLUX2_QWEN3_MODEL_SPEC:-Qwen/Qwen3-4B}"

# Default checkpoint (use latest if not specified)
CKPT_PATH="./runs/libero_flux2_klein_4b_base_imagewam/2026-07-15_16-26-58/checkpoints/weights/step_062000.pt"
DATASET_STATS_PATH="./data/dataset_stats.json"

# Check if files exist
if [ ! -f "${CKPT_PATH}" ]; then
    echo "Error: Checkpoint not found: ${CKPT_PATH}"
    echo "Please set CKPT_PATH environment variable"
    exit 1
fi

if [ ! -f "${DATASET_STATS_PATH}" ]; then
    echo "Error: Dataset stats not found: ${DATASET_STATS_PATH}"
    echo "Please set DATASET_STATS_PATH environment variable"
    exit 1
fi

# Other defaults
TASK="${TASK:-libero_flux2_klein_4b_base_imagewam}"
OUTPUT_DIR="${OUTPUT_DIR:-./evaluate_results/libero_video_rollout/$(date +%Y%m%d_%H%M%S)}"
NUM_TRIALS="${NUM_TRIALS:-3}"
ACTION_HORIZON="${ACTION_HORIZON:-16}"
REPLAN_STEPS="${REPLAN_STEPS:-12}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-30}"
GPU_ID="${GPU_ID:-0}"

# Category filtering
CATEGORY="${CATEGORY:-}"

# Python setup
export PYTHONPATH="${REPO_ROOT}/src:${FLUX2_SRC}/src:${FLUX2_SRC}:${PYTHONPATH:-}"
export PYTHON_BIN="${PYTHON_BIN:-python}"

# Print configuration
echo "=========================================="
echo "ImageWAM LIBERO Video Rollout Inference"
echo "=========================================="
echo "Category Filter: ${CATEGORY:-<none>}"
echo "Checkpoint: ${CKPT_PATH}"
echo "Dataset Stats: ${DATASET_STATS_PATH}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Action Horizon: ${ACTION_HORIZON}"
echo "Replan Steps: ${REPLAN_STEPS}"
echo "Number of Trials: ${NUM_TRIALS}"
echo "=========================================="

# Build command
CMD=(
    "${PYTHON_BIN}" experiments/libero/eval_libero_video_rollout.py
    --config-name sim_libero
    task="${TASK}"
    ckpt="${CKPT_PATH}"
    EVALUATION.dataset_stats_path="${DATASET_STATS_PATH}"
    EVALUATION.output_dir="${OUTPUT_DIR}"
    model.flux2_src_path="${FLUX2_SRC}"
    model.flux2_model_path="${FLUX2_MODEL_PATH}"
    model.ae_model_path="${FLUX2_AE_MODEL_PATH}"
    model.variant="klein-base-4b"
    model.qwen3_model_spec="${FLUX2_QWEN3_MODEL_SPEC}"
    model.load_text_encoder=true
    model.pack_proprio_after_text=true
    EVALUATION.env_num=1
    EVALUATION.num_trials="${NUM_TRIALS}"
    EVALUATION.action_horizon="${ACTION_HORIZON}"
    EVALUATION.replan_steps="${REPLAN_STEPS}"
    EVALUATION.num_steps_wait="${NUM_STEPS_WAIT}"
    EVALUATION.binarize_gripper=true
    gpu_id="${GPU_ID}"
    model.proprio_dim=8
    data.train.qwen_context_len=128
    data.train.qwen_text_cache_format=qwen3_flux2
)

# Add category filter if specified
if [ -n "${CATEGORY}" ]; then
    CATEGORY_FILE="${REPO_ROOT}/third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json"
    if [ -f "${CATEGORY_FILE}" ]; then
        CMD+=(+EVALUATION.category="${CATEGORY}")
        CMD+=(+EVALUATION.category_file="${CATEGORY_FILE}")
        echo "Category filter enabled: ${CATEGORY}"
    else
        echo "Warning: Category file not found, filtering disabled"
    fi
else
    # Use default task when no category specified
    CMD+=(+EVALUATION.task_suite_name="${TASK_SUITE_NAME:-libero_spatial}")
    CMD+=(+EVALUATION.task_id="${TASK_ID:-0}")
fi

# Run inference
echo "Running inference..."
"${CMD[@]}" "$@"
