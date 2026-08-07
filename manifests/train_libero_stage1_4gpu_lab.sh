#!/usr/bin/env bash
set -euo pipefail

# Run from the GuidedVLA repository root:
#   bash manifests/train_libero_stage1_4gpu_lab.sh
#
# Lab layout:
#   <base>/runtime/pi0-conversion-env
#   <base>/datasets/ybwowen-libero-477f7959
#   <base>/models and <base>/assets

if [[ ! -f scripts/train_pytorch.py ]]; then
    echo "Please run this command from the GuidedVLA repository root."
    exit 1
fi

PROJECT_ROOT="$(pwd)"
BASE="$(dirname "${PROJECT_ROOT}")"
RUNTIME="${BASE}/runtime/pi0-conversion-env"
DATA_ROOT="${BASE}/datasets/ybwowen-libero-477f7959"
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

TRAIN_CMD=(
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
)

if [[ "${GUIDEDVLA_CHECK_ONLY:-0}" == "1" ]]; then
    [[ -x "${RUNTIME}/bin/python" ]] || { echo "Missing Python runtime: ${RUNTIME}/bin/python"; exit 1; }
    [[ -d "${DATA_ROOT}" ]] || { echo "Missing dataset: ${DATA_ROOT}"; exit 1; }
    [[ -d "${PI0_BASE}" ]] || { echo "Missing PI0 checkpoint: ${PI0_BASE}"; exit 1; }
    [[ -d "${ASSETS_ROOT}" ]] || { echo "Missing assets: ${ASSETS_ROOT}"; exit 1; }
    [[ -f "${TOKENIZER_PATH}" ]] || { echo "Missing tokenizer: ${TOKENIZER_PATH}"; exit 1; }
    command -v nvidia-smi >/dev/null || { echo "nvidia-smi is unavailable"; exit 1; }

    IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
    [[ "${#gpu_ids[@]}" -eq 4 ]] || { echo "Expected exactly 4 GPU IDs, got: ${CUDA_VISIBLE_DEVICES}"; exit 1; }
    gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
    for gpu_id in "${gpu_ids[@]}"; do
        [[ "${gpu_id}" =~ ^[0-9]+$ ]] && (( gpu_id < gpu_count )) || {
            echo "GPU ${gpu_id} is not visible; nvidia-smi reports ${gpu_count} GPU(s)."
            exit 1
        }
    done

    echo "Startup check passed. Training was not started."
    printf 'CUDA_VISIBLE_DEVICES=%q ' "${CUDA_VISIBLE_DEVICES}"
    printf '%q ' "${TRAIN_CMD[@]}"
    printf '\n'
    exit 0
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
"${TRAIN_CMD[@]}" 2>&1 | tee "${LOG_ROOT}/train.log"
