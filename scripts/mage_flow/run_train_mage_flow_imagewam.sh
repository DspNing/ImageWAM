#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

GPU_PER_NODE="${GPU_PER_NODE:-8}"
TASK_TYPE="${TASK_TYPE:-libero}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/checkpoints}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
MAGE_FLOW_MODEL_ID="${MAGE_FLOW_MODEL_ID:-microsoft/Mage-Flow-Edit}"
MAGE_FLOW_MODEL_PATH="${MAGE_FLOW_MODEL_PATH:-${MODEL_ROOT}/mage_flow/$(basename "${MAGE_FLOW_MODEL_ID}")}"
ACTION_INIT="${ACTION_INIT:-${MODEL_ROOT}/action_dit_mage_flow_${TASK_TYPE}_init.pt}"
ZERO_STAGE="${ZERO_STAGE:-1}"

case "${TASK_TYPE}" in
  libero)
    ACTION_DIM=7
    TASK_NAME="libero_mage_flow_imagewam"
    DATASET_OVERRIDES=(
      "data.train.dataset_dirs=[${DATA_ROOT}/libero_spatial_no_noops_lerobot,${DATA_ROOT}/libero_object_no_noops_lerobot,${DATA_ROOT}/libero_goal_no_noops_lerobot,${DATA_ROOT}/libero_10_no_noops_lerobot]"
    )
    ;;
  robotwin)
    ACTION_DIM=14
    TASK_NAME="robotwin_mage_flow_imagewam"
    ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${DATA_ROOT}/robotwin2.0}"
    DATASET_OVERRIDES=("data.train.dataset_dirs=[${ROBOTWIN_ROOT}]" "data.val.dataset_dirs=[${ROBOTWIN_ROOT}]")
    ;;
  *) echo "TASK_TYPE must be libero or robotwin" >&2; exit 2 ;;
esac

export MAGE_FLOW_MODEL_PATH ZERO_STAGE TASK_TYPE
imagewam_require_env DATA_ROOT

if [ ! -d "${MAGE_FLOW_MODEL_PATH}" ]; then
  MAGE_FLOW_MODEL_ID="${MAGE_FLOW_MODEL_ID}" MAGE_FLOW_ROOT="${MAGE_FLOW_MODEL_PATH}" \
    imagewam_run bash "${SCRIPT_DIR}/prepare_mage_flow_files.sh"
fi

if [ "${REBUILD_ACTION_INIT:-false}" = "true" ] || [ ! -f "${ACTION_INIT}" ]; then
  imagewam_run imagewam_python "${SCRIPT_DIR}/preprocess_action_dit_mage_flow.py" \
    --model-path "${MAGE_FLOW_MODEL_PATH}" \
    --action-dim "${ACTION_DIM}" \
    --output "${ACTION_INIT}"
fi

COMMON_OVERRIDES=(
  "model.mage_flow_model_path=${MAGE_FLOW_MODEL_PATH}"
  "model.action_dit_config.action_dim=${ACTION_DIM}"
  "model.action_dit_config.pretrained_path=${ACTION_INIT}"
)

TASK="${TASK_NAME}" imagewam_run bash scripts/train_zero1.sh "${GPU_PER_NODE}" \
  task="${TASK_NAME}" \
  "${DATASET_OVERRIDES[@]}" \
  "${COMMON_OVERRIDES[@]}" \
  "$@"
