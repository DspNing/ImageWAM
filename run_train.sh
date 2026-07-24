#!/usr/bin/env bash
# ImageWAM (FLUX.2) 训练一键脚本 —— tmux 后台运行,防终端关闭
#
# 用法:
#   1) conda deactivate   # 确保 shell 不在 conda base,避免用到错误的 hf/python
#   2) cd /data/WuKefei/ImageWAM
#   3) bash run_train.sh
#
# 只改下面【参数块】即可。GPU 由 GPUS 决定(逗号分隔的卡号);
# epoch / steps / batch / lr 等通过 hydra override 透传给训练入口。
# 训练在 tmux session 内运行,关掉终端不影响。
#
# 常用操作:
#   查看输出  : tmux attach -t imagewam_train
#   退出 tmux : Ctrl+B 然后按 D(不杀训练)
#   杀掉训练  : tmux kill-session -t imagewam_train
#   列出会话  : tmux list-sessions
# 恢复训练  : RESUME="runs/libero_flux2_klein_4b_base_imagewam/<时间戳>/checkpoints/state/step_001000"
#            然后 bash run_train.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256

# ==================== 参数块(按需修改)====================
GPUS="4,5,6,7"                          # 用哪些卡,逗号分隔,如 "0,1" / "0,1,2,3"

TASK_TYPE="libero"                               # libero | robotwin
FLUX2_VARIANT="2b"                               # 2b | 4b | 9b
ZERO_STAGE="1"                                   # ZeRO stage: 1 | 2

# 训练超参(留空 = 用 task config 默认:bs=10 / lr=1e-4 / epochs=10 / grad_accum=1)
BATCH_SIZE="64"                                    # per-GPU micro-batch,如 "8"
GRAD_ACCUM=""                                    # 梯度累积步数,如 "2"
NUM_WORKERS="8"                                   # 每卡 dataloader worker,如 "16"
LR=""                                            # 学习率,如 "5e-5"
NUM_EPOCHS="10"                                    # epoch 数,如 "5"
MAX_STEPS=""                                     # 最大步数,留空=按 epoch 算;如 "1000" 做 smoke test
SAVE_EVERY=""                                    # 每多少步存 checkpoint,默认 1000
RESUME=""                                        # 恢复训练的 state 目录,留空=从头

PRECOMPUTE_QWEN3_CACHE="false"                    # 首次训练必须 true,会预算 Qwen3 文本特征缓存
REBUILD_ACTION_INIT="false"                      # true=强制重建 ActionDiT 初始化权重

WANDB_MODE="offline"                              # offline=离线不登录/不上传(本地存日志);online=需先 wandb login

GRAD_CHECKPOINT="false"                           # mixture-attention 的 gradient checkpointing;false=关掉省重算提速(显存涨),true=开省显存

SESSION_NAME="imagewam_train"                    # tmux session 名(换名可并行多个)
# ==========================================================

set -euo pipefail

# ---------- 路径与路径变量(兜底,不依赖 .env.local 是否填好)----------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/libero_mujoco3.3.2}"
export MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/checkpoints}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/runs}"
export FLUX2_SRC="${FLUX2_SRC:-${REPO_ROOT}/third_party/flux2}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

case "${FLUX2_VARIANT}" in
  2b)
    export FLUX2_MODEL_PATH="${FLUX2_MODEL_PATH:-${MODEL_ROOT}/flux2/FLUX.2-klein-base-2B/flux-2-klein-base-2b.safetensors}"
    export FLUX2_AE_MODEL_PATH="${FLUX2_AE_MODEL_PATH:-${MODEL_ROOT}/flux2/FLUX.2-dev/ae.safetensors}"
    export FLUX2_QWEN3_MODEL_SPEC="${FLUX2_QWEN3_MODEL_SPEC:-Qwen/Qwen3-4B}"
    ;;
  4b)
    export FLUX2_MODEL_PATH="${FLUX2_MODEL_PATH:-${MODEL_ROOT}/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors}"
    export FLUX2_AE_MODEL_PATH="${FLUX2_AE_MODEL_PATH:-${MODEL_ROOT}/flux2/FLUX.2-dev/ae.safetensors}"
    export FLUX2_QWEN3_MODEL_SPEC="${FLUX2_QWEN3_MODEL_SPEC:-Qwen/Qwen3-4B}"
    ;;
  9b)
    export FLUX2_MODEL_PATH="${FLUX2_MODEL_PATH:-${MODEL_ROOT}/flux2/FLUX.2-klein-base-9B/flux-2-klein-base-9b.safetensors}"
    export FLUX2_AE_MODEL_PATH="${FLUX2_AE_MODEL_PATH:-${MODEL_ROOT}/flux2/FLUX.2-dev/ae.safetensors}"
    export FLUX2_QWEN3_MODEL_SPEC="${FLUX2_QWEN3_MODEL_SPEC:-Qwen/Qwen3-8B}"
    ;;
  *) echo "Error: FLUX2_VARIANT=${FLUX2_VARIANT}; 预期 2b、4b 或 9b" >&2; exit 1 ;;
esac

# robotwin 需要额外路径
if [ "${TASK_TYPE}" = "robotwin" ]; then
  export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${DATA_ROOT}/robotwin2.0}"
fi

# ---------- 解析 GPU ----------
[[ -z "${GPUS}" ]] && { echo "Error: GPUS 不能为空" >&2; exit 1; }
NUM_GPUS=$(echo "${GPUS}" | tr ',' '\n' | grep -c .)
export CUDA_VISIBLE_DEVICES="${GPUS}"
export GPU_PER_NODE="${NUM_GPUS}"

export TASK_TYPE FLUX2_VARIANT ZERO_STAGE PRECOMPUTE_QWEN3_CACHE REBUILD_ACTION_INIT
export WANDB_MODE

# ---------- 拼 hydra overrides(透传给训练入口的 "$@")----------
OVERRIDES=()
[[ -n "${BATCH_SIZE}"  ]] && OVERRIDES+=("batch_size=${BATCH_SIZE}")
[[ -n "${GRAD_ACCUM}"  ]] && OVERRIDES+=("gradient_accumulation_steps=${GRAD_ACCUM}")
[[ -n "${NUM_WORKERS}" ]] && OVERRIDES+=("num_workers=${NUM_WORKERS}")
[[ -n "${LR}"          ]] && OVERRIDES+=("learning_rate=${LR}")
[[ -n "${NUM_EPOCHS}"  ]] && OVERRIDES+=("num_epochs=${NUM_EPOCHS}")
[[ -n "${MAX_STEPS}"   ]] && OVERRIDES+=("max_steps=${MAX_STEPS}")
[[ -n "${SAVE_EVERY}"  ]] && OVERRIDES+=("save_every=${SAVE_EVERY}")
[[ -n "${RESUME}"      ]] && OVERRIDES+=("resume=${RESUME}")
[[ -n "${WANDB_MODE}"  ]] && OVERRIDES+=("wandb.mode=${WANDB_MODE}")
[[ -n "${GRAD_CHECKPOINT}" ]] && OVERRIDES+=("model.mot_checkpoint_mixed_attn=${GRAD_CHECKPOINT}")

# ---------- 环境自检(提前失败,别等 tmux 里才报错)----------
echo "==================== ImageWAM 训练前自检 ===================="
echo "  GPUS            : ${GPUS} (NUM_GPUS=${NUM_GPUS})"
echo "  TASK_TYPE       : ${TASK_TYPE}"
echo "  FLUX2_VARIANT   : ${FLUX2_VARIANT}"
echo "  ZERO_STAGE      : ${ZERO_STAGE}"
echo "  BATCH_SIZE      : ${BATCH_SIZE:-<默认 10>}"
echo "  GRAD_ACCUM      : ${GRAD_ACCUM:-<默认 1>}"
echo "  NUM_EPOCHS      : ${NUM_EPOCHS:-<默认 10>}"
echo "  MAX_STEPS       : ${MAX_STEPS:-<按 epoch 算>}"
echo "  LR              : ${LR:-<默认 1e-4>}"
echo "  RESUME          : ${RESUME:-<从头>}"
echo "  QWEN3_CACHE     : ${PRECOMPUTE_QWEN3_CACHE}"
echo "  WANDB_MODE      : ${WANDB_MODE}"
echo "  GRAD_CHECKPOINT : ${GRAD_CHECKPOINT}"
echo "  OVERRIDES       : ${OVERRIDES[*]:-<无>}"
echo "-------------------------------------------------------------"
ok=1
check_file() { [ -f "$1" ] && echo "  ✅ $2: $1" || { echo "  ❌ $2 不存在: $1"; ok=0; }; }
check_dir()  { [ -d "$1" ] && echo "  ✅ $2: $1" || { echo "  ❌ $2 不存在: $1"; ok=0; } }
check_file "${FLUX2_MODEL_PATH}"    "FLUX2 主权重"
check_file "${FLUX2_AE_MODEL_PATH}" "FLUX2 AE"
check_dir  "${FLUX2_SRC}"           "FLUX2 源码"
check_dir  "${DATA_ROOT}"           "数据集 DATA_ROOT"
if [ "${TASK_TYPE}" = "robotwin" ]; then
  check_dir "${ROBOTWIN_ROOT}" "ROBOTWIN_ROOT"
fi
[ -x "$(command -v tmux)" ] && echo "  ✅ tmux: $(tmux -V)" || { echo "  ❌ tmux 未安装"; ok=0; }
echo "============================================================="
[ "${ok}" = "1" ] || { echo "自检失败,请先解决上面打 ❌ 的项" >&2; exit 1; }

# 若同名 session 已存在,提示而不覆盖
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "Warning: tmux session '${SESSION_NAME}' 已存在。"
  echo "  查看输出: tmux attach -t ${SESSION_NAME}"
  echo "  杀掉重跑: tmux kill-session -t ${SESSION_NAME} && bash run_train.sh"
  exit 1
fi

# ---------- 日志文件 ----------
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_${RUN_ID}.log"

# ---------- 在 tmux 里启动训练 ----------
# 注意:tmux 启动子 shell 时要重新 export 这些变量,否则会被 tmux server 缓存的旧环境覆盖
tmux new-session -d -s "${SESSION_NAME}" "
  cd ${REPO_ROOT}
  unset RUN_ID
  export CUDA_VISIBLE_DEVICES=${GPUS}
  export GPU_PER_NODE=${NUM_GPUS}
  export TASK_TYPE=${TASK_TYPE}
  export FLUX2_VARIANT=${FLUX2_VARIANT}
  export ZERO_STAGE=${ZERO_STAGE}
  export PRECOMPUTE_QWEN3_CACHE=${PRECOMPUTE_QWEN3_CACHE}
  export REBUILD_ACTION_INIT=${REBUILD_ACTION_INIT}
  export DATA_ROOT=${DATA_ROOT}
  export MODEL_ROOT=${MODEL_ROOT}
  export OUTPUT_ROOT=${OUTPUT_ROOT}
  export FLUX2_SRC=${FLUX2_SRC}
  export FLUX2_MODEL_PATH=${FLUX2_MODEL_PATH}
  export FLUX2_AE_MODEL_PATH=${FLUX2_AE_MODEL_PATH}
  export FLUX2_QWEN3_MODEL_SPEC=${FLUX2_QWEN3_MODEL_SPEC}
  export HF_ENDPOINT=${HF_ENDPOINT}
  export WANDB_MODE=${WANDB_MODE}
  $( [ "${TASK_TYPE}" = "robotwin" ] && echo "export ROBOTWIN_ROOT=${ROBOTWIN_ROOT}" )
  bash scripts/flux2/run_train_flux2_klein_imagewam.sh ${OVERRIDES[*]} 2>&1 | tee ${LOG_FILE}
  echo
  echo '[训练结束] exit=\$? 按任意键关闭'
  read -n 1
"

echo
echo "训练已在 tmux session '${SESSION_NAME}' 内启动(后台运行)。"
echo "  查看输出 : tmux attach -t ${SESSION_NAME}"
echo "  实时日志 : tail -f ${LOG_FILE}"
echo "  退出 tmux : Ctrl+B 然后按 D(不杀训练)"
echo "  杀掉训练 : tmux kill-session -t ${SESSION_NAME}"
