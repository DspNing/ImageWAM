#!/usr/bin/env bash
# Mage-Flow ImageWAM 一键训练脚本。
# 用法：修改下方参数块后执行 bash run_train_mage_flow.sh

set -euo pipefail

# ==================== 参数块(按需修改)====================
GPUS="6,7"                         # 使用哪些 GPU，如 "0,1" / "0,1,2,3"
TASK_TYPE="libero"                     # libero | robotwin
ZERO_STAGE="1"                         # ZeRO stage: 1 | 2
MOT_CHECKPOINT_MIXED_ATTN="false"      # true=启用 MoT gradient checkpointing，false=关闭
MAGEFLOW_MIXED_ATTN="sdpa"             # 双卡 ZeRO 训练必须 sdpa:flex 编译反向内核+DS 双卡
                                       #   三合一触发 illegal memory access(2026-08-24
                                       #   三次复现;sdpa 下 700 步干净,速度差<5%)
PREHEAT_VIDEO_MMAP="1"                 # 1(默认)=数据集初始化时预热 video mmap(首批不卡), 0=跳过

IMAGE_SIZE="112"                      # 训练图像分辨率: 224 | 112
                                      # 224 -> mage_text_cache=libero224/, context_len=150
                                      # 112 -> mage_text_cache=libero/,   context_len=120
MAGE_FLOW_VARIANT="base"             # turbo | base | edit | turbo_2b | base_2b
MAGE_FLOW_MODEL_ID=""                 # 留空则由 MAGE_FLOW_VARIANT 选择
MAGE_FLOW_MODEL_PATH=""                # 留空则下载到 MODEL_ROOT/mage_flow/
DELTA_DYNAMICS="true"                  # true=开变化量预测分支(训练期外挂头,推理时
                                       #   不存在,eval 栈零改动;生死门 GO:
                                       #   perturb_probe/results/delta_gate/report.txt)。
                                       #   2026-09-03 复活:baseline+变化头从头重训。
                                       #   纯 baseline 对照则改回 false。
HEAD_KIND="delta_tol"                  # 变化头种类(DELTA_DYNAMICS=true 时生效):
                                       #   delta_tol = 邻域容差 dz 头(2026-09-07):
                                       #               v1 头结构/锚点不动,监督空间
                                       #               精确度从逐格放宽到 k×k 邻域
                                       #               (Camera −4.37 空间锁定病灶);
                                       #               几何由 TOL_MODE 定,pool 主推
                                       #   delta     = 纯 dz 回归头(v1 原头)
TOL_KERNEL="3"                         # 邻域核宽(仅 delta_tol 生效;奇数;1=严格退回
                                       #   v1 逐位匹配;留空则用 config 默认 3)
TOL_MODE="pool"                        # 容差几何(仅 delta_tol):pool=目标侧窗均值
                                       #   池化(主推,位移完全解耦、质量守恒);
                                       #   nbmean=邻域均匀平均 L1(对照模式)
MODEL_ROOT="./checkpoints"
DATA_ROOT="./data"
ROBOTWIN_ROOT="./data/robotwin2.0"      # TASK_TYPE=robotwin 时使用
ACTION_INIT=""                          # 留空则自动生成
REBUILD_ACTION_INIT="false"             # true=强制重建 Action DiT 初始化权重

# 训练超参(留空表示使用 task config 默认值)
GPU_PER_NODE=""                         # 留空则根据 GPUS 自动计算
BATCH_SIZE="24"                           # 每张 GPU 的 micro-batch
GRAD_ACCUM=""                           # 梯度累积步数
NUM_WORKERS="8"                         # 每张 GPU 的 dataloader worker 数
LR=""                                   # 学习率
NUM_EPOCHS="10"                         # epoch 数
MAX_STEPS=""                            # 最大训练步数，留空按 epoch 计算
SAVE_EVERY="2000"                       # 保存间隔
RESUME="/data/WuKefei/ImageWAM/runs/libero_mage_flow_imagewam/2026-09-09_08-45-50/checkpoints/state/step_048000"                               # state checkpoint 路径，留空从头训练。
                                       # delta_tol 必须从头训:现有 ckpt 主干已被
                                       # 严格逐位监督 imprint,容差监督需要完整
                                       # cosine 周期重新塑形,且从头训才与 81.05
                                       # 锚协议可比;千万别 resume 旧头 run 的 state:
                                       # 参数组对不上(优化器加载会炸)。

WANDB_MODE="offline"                    # offline | online
SESSION_NAME="imagewam_mageflow" # tmux session 名称
OUTPUT_ROOT="./runs"                    # 日志和训练输出根目录
# ==========================================================

# 允许通过环境变量覆盖参数块，便于批处理和集群提交。
GPUS="${ENV_GPUS:-${GPUS}}"
DELTA_DYNAMICS="${ENV_DELTA_DYNAMICS:-${DELTA_DYNAMICS}}"
HEAD_KIND="${ENV_HEAD_KIND:-${HEAD_KIND}}"
TOL_KERNEL="${ENV_TOL_KERNEL:-${TOL_KERNEL}}"
TOL_MODE="${ENV_TOL_MODE:-${TOL_MODE}}"
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
# 双卡 NCCL 下 expandable_segments:True 会静默 NaN(单卡无害)——显式钉死 False,
# 防止外部环境误开(2026-08-25 实测 max_split_size 128 也会 29s/步,别用)。
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

OVERRIDES=()
[ -n "${BATCH_SIZE}" ] && OVERRIDES+=("batch_size=${BATCH_SIZE}")
[ -n "${GRAD_ACCUM}" ] && OVERRIDES+=("gradient_accumulation_steps=${GRAD_ACCUM}")
[ -n "${NUM_WORKERS}" ] && OVERRIDES+=("num_workers=${NUM_WORKERS}")
[ -n "${LR}" ] && OVERRIDES+=("learning_rate=${LR}")
[ -n "${NUM_EPOCHS}" ] && OVERRIDES+=("num_epochs=${NUM_EPOCHS}")
[ -n "${MAX_STEPS}" ] && OVERRIDES+=("max_steps=${MAX_STEPS}")
[ -n "${RESUME}" ] && OVERRIDES+=("resume=${RESUME}")
[ -n "${MOT_CHECKPOINT_MIXED_ATTN}" ] && OVERRIDES+=("model.mot_checkpoint_mixed_attn=${MOT_CHECKPOINT_MIXED_ATTN}")
[ -n "${MAGE_FLOW_MODEL_PATH}" ] && OVERRIDES+=("model.mage_flow_model_path=${MAGE_FLOW_MODEL_PATH}")
# 变化量预测分支:训练期外挂头(主相机区 Δz / 邻域容差监督,推理不加载)。
if [ "${DELTA_DYNAMICS}" = "true" ]; then
    OVERRIDES+=("model.delta_dynamics.enabled=true")
    OVERRIDES+=("model.delta_dynamics.kind=${HEAD_KIND}")
    [ -n "${TOL_KERNEL}" ] && [ "${HEAD_KIND}" = "delta_tol" ] && \
        OVERRIDES+=("model.delta_dynamics.tol_kernel=${TOL_KERNEL}")
    [ -n "${TOL_MODE}" ] && [ "${HEAD_KIND}" = "delta_tol" ] && \
        OVERRIDES+=("model.delta_dynamics.tol_mode=${TOL_MODE}")
fi
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
  if tmux has-session -t "${SESSION_NAME}" 2>&1; then
    echo "tmux session already exists: ${SESSION_NAME}" >&2
    exit 1
  fi
  RUN_ID="$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="${OUTPUT_ROOT}/logs/mage_flow_${RUN_ID}.log"
  mkdir -p "$(dirname "${LOG_FILE}")"
  tmux new-session -d -s "${SESSION_NAME}" \
    "cd '${REPO_ROOT}' && unset RUN_ID OUTPUT_DIR && export CUDA_VISIBLE_DEVICES='${GPUS}' && export HF_ENDPOINT='${HF_ENDPOINT}' TASK_TYPE='${TASK_TYPE}' GPU_PER_NODE='${GPU_PER_NODE}' DATA_ROOT='${DATA_ROOT}' ROBOTWIN_ROOT='${ROBOTWIN_ROOT}' MODEL_ROOT='${MODEL_ROOT}' MAGE_FLOW_VARIANT='${MAGE_FLOW_VARIANT}' MAGE_FLOW_MODEL_ID='${MAGE_FLOW_MODEL_ID}' MAGE_FLOW_MODEL_PATH='${MAGE_FLOW_MODEL_PATH}' ACTION_INIT='${ACTION_INIT}' REBUILD_ACTION_INIT='${REBUILD_ACTION_INIT}' ZERO_STAGE='${ZERO_STAGE}' IMAGE_SIZE='${IMAGE_SIZE}' MAGEFLOW_MIXED_ATTN='${MAGEFLOW_MIXED_ATTN}' IMAGEWAM_PREHEAT_VIDEO_MMAP='${PREHEAT_VIDEO_MMAP}' && echo CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES MAGEFLOW_MIXED_ATTN=\$MAGEFLOW_MIXED_ATTN IMAGEWAM_PREHEAT_VIDEO_MMAP=\$IMAGEWAM_PREHEAT_VIDEO_MMAP && bash scripts/mage_flow/run_train_mage_flow_imagewam.sh ${OVERRIDES[*]} 2>&1 | tee '${LOG_FILE}'"
  echo "Mage-Flow training started in tmux session: ${SESSION_NAME}"
  echo "Log: ${LOG_FILE}"
else
  exec bash scripts/mage_flow/run_train_mage_flow_imagewam.sh "${OVERRIDES[@]}"
fi

#
