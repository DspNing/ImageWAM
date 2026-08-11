#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/checkpoints}"
SOURCE_MODEL="${SOURCE_MODEL:-${MODEL_ROOT}/mage_flow/Mage-Flow-Edit-Base}"
OUTPUT_MODEL="${OUTPUT_MODEL:-${MODEL_ROOT}/mage_flow/Mage-Flow-Edit-Base-2B}"
MODEL_ID="${MODEL_ID:-microsoft/Mage-Flow-Edit-Base}"

mkdir -p "${SOURCE_MODEL}"
if [ ! -f "${SOURCE_MODEL}/transformer/config.json" ]; then
  if command -v hf >/dev/null 2>&1; then
    hf download "${MODEL_ID}" --repo-type model --local-dir "${SOURCE_MODEL}"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "${MODEL_ID}" --repo-type model --local-dir "${SOURCE_MODEL}" --resume-download
  else
    echo "Missing Hugging Face CLI: install huggingface_hub to provide hf." >&2
    exit 1
  fi
fi

if [ -e "${OUTPUT_MODEL}" ]; then
  echo "Refusing to overwrite existing output: ${OUTPUT_MODEL}" >&2
  exit 1
fi

echo "prepare......"

PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}" python "${SCRIPT_DIR}/prune_mage_flow.py" \
  --source "${SOURCE_MODEL}" \
  --destination "${OUTPUT_MODEL}" \
  --keep-layers 6

echo "Use MAGE_FLOW_MODEL_PATH=${OUTPUT_MODEL} for the 2B Mage-Flow model."
