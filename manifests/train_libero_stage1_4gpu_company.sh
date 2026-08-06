#!/usr/bin/env bash
set -euo pipefail

# Run from the GuidedVLA repository root:
#   bash manifests/train_libero_stage1_4gpu_company.sh
#
# Company layout:
#   <base>/runtime/.venv
#   <base>/models and <base>/assets
#   <owner-root>/datasets/public/ybwowen/libero-477f7959

if [[ ! -f scripts/train_pytorch.py ]]; then
    echo "Please run this command from the GuidedVLA repository root."
    exit 1
fi

PROJECT_ROOT="$(pwd)"
BASE="$(dirname "${PROJECT_ROOT}")"
OWNER_ROOT="$(dirname "$(dirname "${BASE}")")"
RUNTIME="${BASE}/runtime/.venv"
DATA_ROOT="${OWNER_ROOT}/datasets/public/ybwowen/libero-477f7959"
PI0_BASE="${BASE}/models/pi0_base_pytorch_float32"
ASSETS_ROOT="${BASE}/assets"
TOKENIZER_PATH="${BASE}/models/paligemma_tokenizer.model"

RUN_ID="guidedvla_libero_stage1_4gpu_30k"
OUTPUT_ROOT="${BASE}/outputs/${RUN_ID}"
LOG_ROOT="${BASE}/logs/${RUN_ID}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/packages/openpi-client/src:${PROJECT_ROOT}/third_party/depth_anything/src"
export OPENPI_PALIGEMMA_TOKENIZER_PATH="${TOKENIZER_PATH}"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

"${RUNTIME}/bin/python" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=4 \
    scripts/train_pytorch.py \
    pi0_libero \
    --exp-name "${RUN_ID}" \
    --repo_id ybwowen/libero \
    --local_root_dir "${DATA_ROOT}" \
    --data.assets.asset-id ybwowen/libero \
    --assets-base-dir "${ASSETS_ROOT}" \
    --checkpoint-base-dir "${OUTPUT_ROOT}" \
    --batch-size 16 \
    --gradient-accumulation-steps 4 \
    --num-workers 2 \
    --pytorch-training-precision float32 \
    --no-wandb-enabled \
    --num-train-steps 30000 \
    --log-interval 10 \
    --save-interval 10000 \
    --val-interval 10000 \
    --val-max-batches 1 \
    --pytorch-weight-path "${PI0_BASE}" \
    2>&1 | tee "${LOG_ROOT}/train.log"
