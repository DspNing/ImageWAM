#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/checkpoints}"
MAGE_FLOW_VARIANT="base"        # 可选值：turbo | base | edit
case "${MAGE_FLOW_VARIANT}" in
  turbo) MAGE_FLOW_DEFAULT_MODEL_ID="microsoft/Mage-Flow-Edit-Turbo" ;;
  base)  MAGE_FLOW_DEFAULT_MODEL_ID="microsoft/Mage-Flow-Edit-Base" ;;
  edit)  MAGE_FLOW_DEFAULT_MODEL_ID="microsoft/Mage-Flow-Edit" ;;
  *)
    echo "MAGE_FLOW_VARIANT must be turbo, base, or edit; got: ${MAGE_FLOW_VARIANT}" >&2
    exit 2
    ;;
esac
MAGE_FLOW_MODEL_ID="${MAGE_FLOW_MODEL_ID:-${MAGE_FLOW_DEFAULT_MODEL_ID}}"
MAGE_FLOW_ROOT="${MAGE_FLOW_ROOT:-${MODEL_ROOT}/mage_flow/$(basename "${MAGE_FLOW_MODEL_ID}")}"

mkdir -p "${MAGE_FLOW_ROOT}"
imagewam_print_config MAGE_FLOW_MODEL_ID MAGE_FLOW_ROOT
if command -v hf >/dev/null 2>&1; then
  imagewam_run hf download "${MAGE_FLOW_MODEL_ID}" \
    --repo-type model --local-dir "${MAGE_FLOW_ROOT}"
elif command -v huggingface-cli >/dev/null 2>&1; then
  imagewam_run huggingface-cli download "${MAGE_FLOW_MODEL_ID}" \
    --repo-type model --local-dir "${MAGE_FLOW_ROOT}" --resume-download
else
  echo "Missing Hugging Face CLI: install huggingface_hub to provide hf." >&2
  exit 1
fi

echo "Mage-Flow weights prepared at ${MAGE_FLOW_ROOT}"
