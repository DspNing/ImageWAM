#!/usr/bin/env bash
# Precompute Mage-Flow text (Qwen3-VL edit) context cache.
#
# IMPORTANT: the cache key is a hash of the frame *bytes*. The frame source
# here MUST match training exactly (same video_frame_cache_dir + same IMAGE_SIZE),
# otherwise the keys drift and training dies with "Missing Mage text cache".
# This script pins both, mirroring run_train_mage_flow.sh.
#
# 多卡策略：起 N 个独立单卡进程，各自分片(IMAGEWAM_SHARD_RANK/WORLD)、互不通信。
# 不用 torchrun / init_process_group，避免外部 mage_flow encoder 在多卡时自起 NCCL
# 并在冷 mmap I/O 上超时。每个进程只见 1 张 GPU。
set -euo pipefail

# ==================== 参数块(按需修改)====================
GPUS="6,7"                            # 用哪些 GPU，如 "2" / "2,3" / "0,1,2,3"
IMAGE_SIZE="224"                       # 必须与训练一致: 224 | 112
TASK="libero_mage_flow_imagewam"
MODEL_PATH="./checkpoints/mage_flow/Mage-Flow-Edit-Base"
BATCH_SIZE="256"                       # 每卡 batch，按显存调 (单卡 encoder bf16)
NUM_WORKERS="8"
DATA_ROOT="./data"
OVERWRITE="false"                      # true=强制全量重算; false=只补缺失项
PYTHON_BIN="${PYTHON_BIN:-python}"     # 默认用当前 env 的 python
# =========================================================

# 允许环境变量覆盖，便于集群/批处理提交。
GPUS="${ENV_GPUS:-${GPUS}}"
IMAGE_SIZE="${ENV_IMAGE_SIZE:-${IMAGE_SIZE}}"
TASK="${ENV_TASK:-${TASK}}"
MODEL_PATH="${ENV_MODEL_PATH:-${MODEL_PATH}}"
DATA_ROOT="${ENV_DATA_ROOT:-${DATA_ROOT}}"
OVERWRITE="${ENV_OVERWRITE:-${OVERWRITE}}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# IMAGE_SIZE -> 帧缓存目录 + mage cache 输出目录
# 必须与 run_train_mage_flow.sh 里的 case 完全一致。
case "${IMAGE_SIZE}" in
  224)
    VIDEO_FRAME_CACHE_DIR="${DATA_ROOT}/video_frames_cache/libero_224_mmap"
    MAGE_TEXT_CACHE_DIR="${DATA_ROOT}/mage_text_cache/libero224"
    ;;
  112)
    VIDEO_FRAME_CACHE_DIR="${DATA_ROOT}/video_frames_cache/libero_flux2_mmap"
    MAGE_TEXT_CACHE_DIR="${DATA_ROOT}/mage_text_cache/libero"
    ;;
  *)
    echo "IMAGE_SIZE must be 224 or 112; got: ${IMAGE_SIZE}" >&2
    exit 2
    ;;
esac

mkdir -p "${MAGE_TEXT_CACHE_DIR}"

# 解析 GPU 列表
IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NGPU="${#GPU_ARR[@]}"

# config 的 crop_transform / Resize 用 ${envint:IMAGE_SIZE,...}，必须导出与训练一致。
export IMAGE_SIZE
export DATA_ROOT

EXTRA=()
[ "${OVERWRITE}" = "true" ] && EXTRA+=(--overwrite)

echo "[precompute] IMAGE_SIZE=${IMAGE_SIZE}"
echo "[precompute] video_frame_cache_dir=${VIDEO_FRAME_CACHE_DIR}"
echo "[precompute] output=${MAGE_TEXT_CACHE_DIR}"
echo "[precompute] model=${MODEL_PATH}"
echo "[precompute] gpus=${GPUS} (${NGPU} 独立进程)  overwrite=${OVERWRITE}"

# 每个 GPU 起一个独立分片进程，各自写各自 shard 的文件。
PIDS=()
for i in "${!GPU_ARR[@]}"; do
  g="${GPU_ARR[$i]}"
  CUDA_VISIBLE_DEVICES="${g}" \
  IMAGE_SIZE="${IMAGE_SIZE}" DATA_ROOT="${DATA_ROOT}" \
  IMAGEWAM_SHARD_RANK="${i}" IMAGEWAM_SHARD_WORLD="${NGPU}" \
  "${PYTHON_BIN}" scripts/mage_flow/precompute_text_cache.py \
    --task "${TASK}" \
    --model-path "${MODEL_PATH}" \
    --output "${MAGE_TEXT_CACHE_DIR}" \
    --video-frame-cache-dir "${VIDEO_FRAME_CACHE_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    "${EXTRA[@]}" \
    > "${MAGE_TEXT_CACHE_DIR}/.shard${i}.log" 2>&1 &
  PIDS+=("$!")
  echo "[precompute] launched shard${i} on GPU ${g} (pid $!) -> ${MAGE_TEXT_CACHE_DIR}/.shard${i}.log"
done

# 等所有分片结束；任意一个失败则整体失败。
FAIL=0
for p in "${PIDS[@]}"; do
  if ! wait "$p"; then
    echo "[precompute] ERROR: shard pid $p exited non-zero" >&2
    FAIL=1
  fi
done

if [ "${FAIL}" -ne 0 ]; then
  echo "[precompute] FAILED — 见各 shard 日志: ${MAGE_TEXT_CACHE_DIR}/.shard*.log" >&2
  exit 1
fi

echo "[precompute] DONE — 缓存写入 ${MAGE_TEXT_CACHE_DIR}"
echo "[precompute] 各 shard 日志: ${MAGE_TEXT_CACHE_DIR}/.shard*.log"
