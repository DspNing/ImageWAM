#!/usr/bin/env bash
# Mage-Flow ImageWAM 一键训练脚本。
# 用法：修改下方参数块后执行 bash run_train_mage_flow.sh

set -euo pipefail

# ==================== 参数块(按需修改)====================
GPUS="4,5"                         # 使用哪些 GPU，如 "0,1" / "0,1,2,3"
TASK_TYPE="libero"                     # libero | robotwin
ZERO_STAGE="1"                         # ZeRO stage: 1 | 2
MOT_CHECKPOINT_MIXED_ATTN="false"      # true=启用 MoT gradient checkpointing，false=关闭
MAGEFLOW_MIXED_ATTN="flex"             # MoT 混合注意力后端: sdpa(默认,dense mask) | flex(block-sparse,~1.6x@L=18748)
PREHEAT_VIDEO_MMAP="1"                 # 1(默认)=数据集初始化时预热 video mmap(首批不卡), 0=跳过

IMAGE_SIZE="112"                      # 训练图像分辨率: 224 | 112
                                      # 224 -> mage_text_cache=libero224/, context_len=150
                                      # 112 -> mage_text_cache=libero/,   context_len=120
MAGE_FLOW_VARIANT="base"             # turbo | base | edit | turbo_2b | base_2b
MAGE_FLOW_MODEL_ID=""                 # 留空则由 MAGE_FLOW_VARIANT 选择
MAGE_FLOW_MODEL_PATH=""                # 留空则下载到 MODEL_ROOT/mage_flow/
MODEL_ROOT="./checkpoints"
DATA_ROOT="./data"
ROBOTWIN_ROOT="./data/robotwin2.0"      # TASK_TYPE=robotwin 时使用
ACTION_INIT=""                          # 留空则自动生成
REBUILD_ACTION_INIT="false"             # true=强制重建 Action DiT 初始化权重

# 训练超参(留空表示使用 task config 默认值)
GPU_PER_NODE=""                         # 留空则根据 GPUS 自动计算
BATCH_SIZE="32"                           # 每张 GPU 的 micro-batch
GRAD_ACCUM=""                           # 梯度累积步数
NUM_WORKERS="8"                         # 每张 GPU 的 dataloader worker 数
LR=""                                   # 学习率
NUM_EPOCHS="10"                         # epoch 数
MAX_STEPS=""                            # 最大训练步数，留空按 epoch 计算
SAVE_EVERY="2000"                       # 保存间隔
RESUME="./runs/libero_mage_flow_imagewam/2026-08-10_01-45-38/checkpoints/state/step_010000"                               # state checkpoint 路径，留空从头训练

WANDB_MODE="offline"                    # offline | online
SESSION_NAME="imagewam_mageflow" # tmux session 名称
OUTPUT_ROOT="./runs"                    # 日志和训练输出根目录
# ==========================================================

# 允许通过环境变量覆盖参数块，便于批处理和集群提交。
GPUS="${ENV_GPUS:-${GPUS}}"
TASK_TYPE="${ENV_TASK_TYPE:-${TASK_TYPE}}"
ZERO_STAGE="${ENV_ZERO_STAGE:-${ZERO_STAGE}}"
MOT_CHECKPOINT_MIXED_ATTN="${ENV_MOT_CHECKPOINT_MIXED_ATTN:-${MOT_CHECKPOINT_MIXED_ATTN}}"
MAGEFLOW_MIXED_ATTN="${ENV_MAGEFLOW_MIXED_ATTN:-${MAGEFLOW_MIXED_ATTN}}"
PREHEAT_VIDEO_MMAP="${ENV_PREHEAT_VIDEO_MMAP:-${PREHEAT_VIDEO_MMAP}}"
IMAGE_SIZE="${ENV_IMAGE_SIZE:-${IMAGE_SIZE}}"
MAGE_FLOW_MODEL_ID="${ENV_MAGE_FLOW_MODEL_ID:-${MAGE_FLOW_MODEL_ID}}"
MAGE_FLOW_VARIANT="${ENV_MAGE_FLOW_VARIANT:-${MAGE_FLOW_VARIANT}}"
MAGE_FLOW_MODEL_PATH="${ENV_MAGE_FLOW_MODEL_PATH:-${MAGE_FLOW_MODEL_PATH}}"
MODEL_ROOT="${ENV_MODEL_ROOT:-${MODEL_ROOT}}"
DATA_ROOT="${ENV_DATA_ROOT:-${DATA_ROOT}}"
ROBOTWIN_ROOT="${ENV_ROBOTWIN_ROOT:-${ROBOTWIN_ROOT}}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

export MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/checkpoints}"
export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/runs}"
export MAGE_FLOW_MODEL_ID MAGE_FLOW_VARIANT MAGE_FLOW_MODEL_PATH MODEL_ROOT DATA_ROOT OUTPUT_ROOT
export CUDA_VISIBLE_DEVICES="${GPUS}"
export GPU_PER_NODE="${GPU_PER_NODE:-$(tr ',' '\n' <<< "${GPUS}" | wc -l)}"
export TASK_TYPE ZERO_STAGE REBUILD_ACTION_INIT WANDB_MODE IMAGE_SIZE
export MAGEFLOW_MIXED_ATTN
export IMAGEWAM_PREHEAT_VIDEO_MMAP="${PREHEAT_VIDEO_MMAP}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

OVERRIDES=()
[ -n "${BATCH_SIZE}" ] && OVERRIDES+=("batch_size=${BATCH_SIZE}")
[ -n "${GRAD_ACCUM}" ] && OVERRIDES+=("gradient_accumulation_steps=${GRAD_ACCUM}")
[ -n "${NUM_WORKERS}" ] && OVERRIDES+=("num_workers=${NUM_WORKERS}")
[ -n "${LR}" ] && OVERRIDES+=("learning_rate=${LR}")
[ -n "${NUM_EPOCHS}" ] && OVERRIDES+=("num_epochs=${NUM_EPOCHS}")
[ -n "${MAX_STEPS}" ] && OVERRIDES+=("max_steps=${MAX_STEPS}")
[ -n "${SAVE_EVERY}" ] && OVERRIDES+=("save_every=${SAVE_EVERY}")
[ -n "${RESUME}" ] && OVERRIDES+=("resume=${RESUME}")
[ -n "${MOT_CHECKPOINT_MIXED_ATTN}" ] && OVERRIDES+=("model.mot_checkpoint_mixed_attn=${MOT_CHECKPOINT_MIXED_ATTN}")
[ -n "${MAGE_FLOW_MODEL_PATH}" ] && OVERRIDES+=("model.mage_flow_model_path=${MAGE_FLOW_MODEL_PATH}")
OVERRIDES+=("data.train.qwen_text_cache_dir=null" "data.train.require_text_cache=false")

# 图像分辨率 -> MAGE text cache 路径与 context 长度（224/112）
case "${IMAGE_SIZE}" in
  224)
    MAGE_TEXT_CACHE_DIR="${DATA_ROOT}/mage_text_cache/libero224/"
    MAGE_CONTEXT_LEN="150"
    VIDEO_FRAME_CACHE_DIR="${DATA_ROOT}/video_frames_cache/libero_224_mmap"
    ;;
  112)
    MAGE_TEXT_CACHE_DIR="${DATA_ROOT}/mage_text_cache/libero/"
    MAGE_CONTEXT_LEN="120"
    VIDEO_FRAME_CACHE_DIR="${DATA_ROOT}/video_frames_cache/libero_flux2_mmap"
    ;;
  *)
    echo "IMAGE_SIZE must be 224 or 112; got: ${IMAGE_SIZE}" >&2
    exit 2
    ;;
esac
OVERRIDES+=(
  "data.train.mage_text_cache_dir=${MAGE_TEXT_CACHE_DIR}"
  "data.train.mage_context_len=${MAGE_CONTEXT_LEN}"
  "data.train.video_frame_cache_dir=${VIDEO_FRAME_CACHE_DIR}"
  "data.eval.mage_text_cache_dir=${MAGE_TEXT_CACHE_DIR}"
  "data.eval.mage_context_len=${MAGE_CONTEXT_LEN}"
)
if [ "${WANDB_MODE}" = "offline" ]; then
  OVERRIDES+=("wandb.enabled=false")
fi

if [ -n "${ROBOTWIN_ROOT}" ]; then
  export ROBOTWIN_ROOT
fi

if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION_NAME}" >&2
    exit 1
  fi
  RUN_ID="$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="${OUTPUT_ROOT}/logs/mage_flow_${RUN_ID}.log"
  mkdir -p "$(dirname "${LOG_FILE}")"
  tmux new-session -d -s "${SESSION_NAME}" \
    "cd '${REPO_ROOT}' && export CUDA_VISIBLE_DEVICES='${GPUS}' && export HF_ENDPOINT='${HF_ENDPOINT}' TASK_TYPE='${TASK_TYPE}' GPU_PER_NODE='${GPU_PER_NODE}' DATA_ROOT='${DATA_ROOT}' ROBOTWIN_ROOT='${ROBOTWIN_ROOT}' MODEL_ROOT='${MODEL_ROOT}' MAGE_FLOW_VARIANT='${MAGE_FLOW_VARIANT}' MAGE_FLOW_MODEL_ID='${MAGE_FLOW_MODEL_ID}' MAGE_FLOW_MODEL_PATH='${MAGE_FLOW_MODEL_PATH}' ACTION_INIT='${ACTION_INIT}' REBUILD_ACTION_INIT='${REBUILD_ACTION_INIT}' ZERO_STAGE='${ZERO_STAGE}' IMAGE_SIZE='${IMAGE_SIZE}' MAGEFLOW_MIXED_ATTN='${MAGEFLOW_MIXED_ATTN}' IMAGEWAM_PREHEAT_VIDEO_MMAP='${PREHEAT_VIDEO_MMAP}' && echo CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES MAGEFLOW_MIXED_ATTN=\$MAGEFLOW_MIXED_ATTN IMAGEWAM_PREHEAT_VIDEO_MMAP=\$IMAGEWAM_PREHEAT_VIDEO_MMAP && bash scripts/mage_flow/run_train_mage_flow_imagewam.sh ${OVERRIDES[*]} 2>&1 | tee '${LOG_FILE}'"
  echo "Mage-Flow training started in tmux session: ${SESSION_NAME}"
  echo "Log: ${LOG_FILE}"
else
  exec bash scripts/mage_flow/run_train_mage_flow_imagewam.sh "${OVERRIDES[@]}"
fi

# 
