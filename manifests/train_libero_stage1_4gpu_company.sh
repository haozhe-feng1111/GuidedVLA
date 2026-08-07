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
CACHE_ROOT="${BASE}/cache/${RUN_ID}"
WANDB_ROOT="${BASE}/wandb/${RUN_ID}"
# Keep multiprocessing forkserver socket paths short while staying on the
# personal disk. The run-specific cache path is too long for AF_UNIX sockets.
TMP_ROOT="${BASE}/tmp"

export HF_HOME="${CACHE_ROOT}/hf"
export HF_HUB_CACHE="${CACHE_ROOT}/hf-hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${CACHE_ROOT}/hf-datasets"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/transformers"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export CUDA_CACHE_PATH="${CACHE_ROOT}/cuda"
export TMPDIR="${TMP_ROOT}"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/packages/openpi-client/src:${PROJECT_ROOT}/third_party/depth_anything/src"
export OPENPI_PALIGEMMA_TOKENIZER_PATH="${TOKENIZER_PATH}"
export WANDB_MODE=offline
export WANDB_DIR="${WANDB_ROOT}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p \
    "${OUTPUT_ROOT}" "${LOG_ROOT}" "${WANDB_ROOT}" \
    "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" \
    "${TRANSFORMERS_CACHE}" "${XDG_CACHE_HOME}" \
    "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" \
    "${CUDA_CACHE_PATH}" "${TMP_ROOT}"

RUNTIME_LOG="${LOG_ROOT}/launcher_runtime.log"
{
    echo "timestamp=$(date -Is)"
    echo "hostname=$(hostname)"
    echo "project_root=${PROJECT_ROOT}"
    echo "run_id=${RUN_ID}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
    echo "cache_root=${CACHE_ROOT}"
    echo "wandb_root=${WANDB_ROOT}"
    echo "--- df -hT ---"
    df -hT || true
    echo "--- df -i ---"
    df -i || true
    echo "--- selected environment ---"
    env | sort | grep -E '^(HOME|TMPDIR|TMP|TEMP|XDG_CACHE_HOME|HF_HOME|HF_HUB_CACHE|HUGGINGFACE_HUB_CACHE|HF_DATASETS_CACHE|TRANSFORMERS_CACHE|TORCHINDUCTOR_CACHE_DIR|TRITON_CACHE_DIR|CUDA_CACHE_PATH)=' || true
    echo "--- selected local directories ---"
    for path in "${TMPDIR:-}" /tmp /var/tmp \
        "${HOME:-}/.cache" "${HOME:-}/.cache/huggingface" \
        "${HOME:-}/.cache/torch" "${HOME:-}/.triton"; do
        if [[ -n "${path}" && -e "${path}" ]]; then
            du -x -sh "${path}" 2>&1 || true
        fi
    done
} >"${RUNTIME_LOG}" 2>&1 || true

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
    --wandb-enabled \
    --num-train-steps 30000 \
    --log-interval 10 \
    --save-interval 10000 \
    --val-interval 10000 \
    --val-max-batches 1 \
    --pytorch-weight-path "${PI0_BASE}" \
    2>&1 | tee "${LOG_ROOT}/train.log"
