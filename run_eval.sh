#!/usr/bin/env bash
# ImageWAM LIBERO / LIBERO-Plus 推理一键脚本
#
# 用法：直接运行
#   bash run_eval.sh
#
# 只改下面【参数块】即可，无需每次手打一堆 export。
# MODE="plus"   → LIBERO-Plus 鲁棒性推理（NUM_TRIALS 默认 1，走批量调度）
# MODE="master" → 原版 LIBERO 推理    （NUM_TRIALS 默认 25/50，走原版单任务调度）
#
# 结果按 MODE 自动分流到子目录：
#   plus  → evaluate_results/libero_plus/$RUN_ID/
#   master→ evaluate_results/libero/$RUN_ID/

# ==================== 参数块（按需修改）====================
MODE="plus"                                       # "plus" | "master"
GPUS="6,7"                                    # 用哪些卡，逗号分隔
MAGE_FLOW_VARIANT="${MAGE_FLOW_VARIANT:-base}"

#   plus : 传入任务清单文件路径（或由调度脚本自动生成）
#   master: 传入任务清单文件路径
TASK_LIST="./task_lists/libero_plus_all.txt"                                      # 留空 = 自动生成任务列表

# checkpoint 路径（必填）：
# CKPT="./checkpoints/imagewam_release/libero/flux2_klein_4b/model.pt"
CKPT="/data/WuKefei/ImageWAM/runs/libero_mage_flow_imagewam/2026-09-09_08-45-50/checkpoints/weights/step_048000.pt"                                           # 例如 "./runs/xxx/checkpoints/weights/step_040000.pt"

# dataset_stats 路径（留空 = 自动从 ckpt 父目录查找）：
STATS="./data/dataset_stats.json"

NUM_TRIALS=""                                      # 留空：plus→1 / master→25；填数字则强制覆盖
NUM_INFERENCE_STEPS="10"                             # 每次动作预测的去噪步数

# 推理图像分辨率（必须与训练一致）：112 | 224
IMAGE_SIZE="112"

# —— 视频控制 ——
SAVE_VIDEO="false"                                #  是否保存 rollout 视频
MAX_VIDEOS_PER_WORKER="5"                         #  plus 模式：每个 worker 最多保存多少个 task 的视频
VISUALIZE_FUTURE_VIDEO="false"                    #  是否可视化未来动作视频

# —— 以下一般不用改 ——
CONFIG="libero_mage_flow_imagewam"      # configs/task/ 下的配置名（不带 .yaml）
MAX_TASKS_PER_GPU=7                               # 每卡并发任务数

# 用哪个 conda env 跑 worker。
CONDA_ENV="mageflow"

WORKERS_PER_GPU=""                                 # 留空：默认 = MAX_TASKS_PER_GPU

SHARED_ENCODER="${SHARED_ENCODER:-true}"           # true=每卡共享一个 text encoder(省显存, 多塞 worker) | false=每 worker 在线加载(默认)

case "${MAGE_FLOW_VARIANT}" in
    turbo) MAGE_FLOW_DEFAULT_MODEL_PATH="${MODEL_ROOT:-./checkpoints}/mage_flow/Mage-Flow-Edit-Turbo" ;;
    turbo_2b) MAGE_FLOW_DEFAULT_MODEL_PATH="${MODEL_ROOT:-./checkpoints}/mage_flow/Mage-Flow-Edit-Turbo-2B" ;;
    base) MAGE_FLOW_DEFAULT_MODEL_PATH="${MODEL_ROOT:-./checkpoints}/mage_flow/Mage-Flow-Edit-Base" ;;
    base_2b) MAGE_FLOW_DEFAULT_MODEL_PATH="${MODEL_ROOT:-./checkpoints}/mage_flow/Mage-Flow-Edit-Base-2B" ;;
    edit) MAGE_FLOW_DEFAULT_MODEL_PATH="${MODEL_ROOT:-./checkpoints}/mage_flow/Mage-Flow-Edit" ;;
    *) echo "Error: MAGE_FLOW_VARIANT must be turbo, turbo_2b, base, or edit" >&2; exit 1 ;;
esac
MAGE_FLOW_MODEL_PATH="${MAGE_FLOW_MODEL_PATH:-${MAGE_FLOW_DEFAULT_MODEL_PATH}}"


# ==========================================================

set -euo pipefail

# 切到项目根目录，确保 .env.local 里的 $(pwd) 展开正确
cd "$(dirname "$0")"

# 加载环境变量（FLUX2_SRC、MODEL_ROOT 等）
if [ -f ".env.local" ]; then
    set -a
    source ".env.local"
    set +a
    echo "[config] Loaded .env.local"
fi

# 清理 PYTHONPATH 中可能残留的旧路径（如 FastWAM/LIBERO），避免 robosuite/mujoco 版本冲突
export PYTHONPATH=$(echo "${PYTHONPATH:-}" | tr ':' '\n' | grep -v 'FastWAM' | grep -v 'Fastwambase' | paste -sd: -)

# 根据 MODE 决定环境变量、默认 NUM_TRIALS、结果子目录、调度脚本
if [[ "${MODE}" == "plus" ]]; then
    NUM_TRIALS="${NUM_TRIALS:-1}"
    RESULTS_SUBDIR="libero_plus"
elif [[ "${MODE}" == "master" ]]; then
    NUM_TRIALS="${NUM_TRIALS:-25}"
    RESULTS_SUBDIR="libero"
else
    echo "Error: MODE 必须是 'plus' 或 'master'，当前: '${MODE}'" >&2
    exit 1
fi

# Use the GPU EGL backend for both schedulers.  The master scheduler defaults
# to OSMesa, which is not usable in the Mage-Flow environment on this host.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"


# 切换 libero 的 bddl/init 路径配置(根据 MODE 指向不同目录)
export LIBERO_PKG_DIR="${LIBERO_PKG_DIR:-$(pwd)/third_party/LIBERO-${MODE}}"
LIBERO_CFG_DIR="${HOME}/.libero"
mkdir -p "${LIBERO_CFG_DIR}"
cat > "${LIBERO_CFG_DIR}/config.yaml" << EOF
assets: ${LIBERO_PKG_DIR}/libero/libero/assets
bddl_files: ${LIBERO_PKG_DIR}/libero/libero/bddl_files
benchmark_root: ${LIBERO_PKG_DIR}/libero/libero
datasets: ${DATA_ROOT:-$(pwd)/data/libero_mujoco3.3.2}
init_states: ${LIBERO_PKG_DIR}/libero/libero/init_files
EOF
echo "[config] libero config.yaml -> ${LIBERO_PKG_DIR}/libero/libero (MODE=${MODE})"

# 把 LIBERO pkg 加到 PYTHONPATH（让 libero.libero 能找到）
export PYTHONPATH="${LIBERO_PKG_DIR}:${PYTHONPATH:-}"

# 文件存在性校验
if [[ -n "${TASK_LIST}" ]]; then
    [[ -f "${TASK_LIST}" ]] || { echo "Error: TASK_LIST 不存在: ${TASK_LIST}" >&2; exit 1; }
fi
[[ -f "${CKPT}" ]] || { echo "Error: CKPT 不存在: ${CKPT}" >&2; exit 1; }
case "${IMAGE_SIZE}" in
    112|224) : ;;
    *) echo "Error: IMAGE_SIZE 必须是 112 或 224，当前: '${IMAGE_SIZE}'" >&2; exit 1 ;;
esac
if [[ -n "${STATS}" ]]; then
    [[ -f "${STATS}" ]] || { echo "Error: STATS 不存在: ${STATS}" >&2; exit 1; }
fi

export CONFIG
export IMAGE_SIZE
export CUDA_VISIBLE_DEVICES="${GPUS}"
export MAX_TASKS_PER_GPU
export NUM_TRIALS
export CKPT
export RESULTS_SUBDIR
export CONDA_ENV
export CONDA_ENV_PYTHON="${CONDA_ENV_PYTHON:-${HOME}/miniconda3/envs/${CONDA_ENV}/bin/python}"

# 生成 RUN_ID
export RUN_ID="${RUN_ID:-eval_$(date +%Y%m%d_%H%M%S)}"
export ROOT_DIR="${ROOT_DIR:-$(pwd)}"
export OUTPUT_DIR="${ROOT_DIR}/evaluate_results/${RESULTS_SUBDIR}/${RUN_ID}"

if [[ -n "${STATS}" ]]; then
    export EXTRA_ARGS="EVALUATION.dataset_stats_path=${STATS}"
else
    export EXTRA_ARGS=""
fi
EXTRA_ARGS="${EXTRA_ARGS} EVALUATION.num_inference_steps=${NUM_INFERENCE_STEPS}"
EXTRA_ARGS="${EXTRA_ARGS} EVALUATION.visualize_future_video=${VISUALIZE_FUTURE_VIDEO}"

if [[ "${CONFIG}" == *mage_flow* ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} model.mage_flow_model_path=${MAGE_FLOW_MODEL_PATH} model.load_text_encoder=true"
fi

# 视频控制旋钮
export SAVE_VIDEO
export MAX_VIDEOS_PER_WORKER

# backbone 相关路径参数（供 batch scheduler 传递）
export MAGE_FLOW_MODEL_PATH="${MAGE_FLOW_MODEL_PATH:-}"
export SHARED_ENCODER
export VF_HF_ATTN_IMPL="${MAGE_ATTN_IMPL:-sdpa}"   # mage encoder attn: sdpa(默认, 无需 flash-attn) | flash_attention_2(需装 flash-attn)

# 自动生成任务清单
if [[ -z "${TASK_LIST}" ]]; then
    TASK_LIST="$OUTPUT_DIR/generated_tasks.txt"
    mkdir -p "$(dirname "$TASK_LIST")"
    python -c "
import sys, random, math
sys.path.insert(0, '${ROOT_DIR}')
from libero.libero import benchmark
suites = ['libero_10', 'libero_goal', 'libero_spatial', 'libero_object']
ratio = ${TASK_SAMPLE_RATIO:-1.0}
seed = ${TASK_SAMPLE_SEED:-42}
benchmark_dict = benchmark.get_benchmark_dict()
with open('${TASK_LIST}', 'w') as f:
    for sn in suites:
        ts = benchmark_dict[sn]()
        n = int(ts.n_tasks)
        tids = list(range(n))
        if ratio < 1.0:
            ns = max(1, int(math.ceil(n * ratio)))
            r = random.Random(f'{seed}:{sn}')
            tids = sorted(r.sample(tids, ns))
        for tid in tids:
            f.write(f'{sn},{tid}\n')
"
    echo "[auto-generated task list: $(wc -l < "$TASK_LIST") tasks]"
fi


echo "==================== LIBERO 推理 ===================="
echo "  MODE          : ${MODE}"
echo "  GPUS          : ${GPUS}"
echo "  TASK_LIST     : ${TASK_LIST} ($(wc -l < "${TASK_LIST}") 任务)"
echo "  NUM_TRIALS    : ${NUM_TRIALS}"
echo "  MAX_TASKS_PER_GPU: ${MAX_TASKS_PER_GPU}"
echo "  CKPT          : ${CKPT}"
echo "  IMAGE_SIZE    : ${IMAGE_SIZE}"
echo "  STATS         : ${STATS:-<auto-find>}"
echo "  CONFIG        : ${CONFIG}"
echo "  CONDA_ENV     : ${CONDA_ENV:-<默认>}"
echo "  RESULTS_SUBDIR: ${RESULTS_SUBDIR}"
echo "  OUTPUT_DIR    : ${OUTPUT_DIR}"
echo "  VISUALIZE_FUTURE_VIDEO: ${VISUALIZE_FUTURE_VIDEO}"
if [[ "${MODE}" == "plus" ]]; then
    echo "  WORKERS_PER_GPU: ${WORKERS_PER_GPU:-<默认=MAX_TASKS_PER_GPU>}"
    echo "  SAVE_VIDEO    : ${SAVE_VIDEO}"
    echo "  MAX_VIDEOS_PER_WORKER: ${MAX_VIDEOS_PER_WORKER}"
fi
echo "====================================================="

if [[ "${MODE}" == "plus" ]]; then
    bash experiments/libero/run_libero_plus_batch.sh "${TASK_LIST}"
else
    bash experiments/libero/run_libero_parallel_test.sh "${TASK_LIST}"
fi

# tmux kill-window -t libero_plus_batch
# tmux kill-window -t libero_test_v3
