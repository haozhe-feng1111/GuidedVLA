#!/usr/bin/env bash
set -euo pipefail

# DINOv2-Base deep-guidance Stage-2 launcher for the existing fudan-lab recipe.
# Relative to the established formal launcher, the only training-hyperparameter
# changes are physical global batch 16 -> 24 (local batch 4 -> 6 on four ranks)
# and gradient accumulation 4 -> 1, so effective global batch is 24.

if [[ ! -f scripts/train_pytorch.py ]]; then
    echo "Please run this command from the GuidedVLA repository root."
    exit 1
fi

PROJECT_ROOT="$(pwd)"
BASE="$(dirname "${PROJECT_ROOT}")"
RUNTIME="${BASE}/runtime/pi0-conversion-env"
DATA_ROOT="${BASE}/datasets/ybwowen-libero-477f7959"
ASSETS_ROOT="${BASE}/assets"
TOKENIZER_PATH="${BASE}/models/paligemma_tokenizer.model"
DINO_MODEL="${GUIDEDVLA_DINOV2_BASE_MODEL_PATH:-${BASE}/models/dinov2-base}"

STAGE1_RUN_ID="guidedvla_libero_stage1_4gpu_30k"
STAGE1_STEP=30000
STAGE1_CHECKPOINT="${BASE}/outputs/${STAGE1_RUN_ID}/pi0_libero/${STAGE1_RUN_ID}/${STAGE1_STEP}"

RUN_ID="${GUIDEDVLA_RUN_ID:-guidedvla_libero_stage2_object_dinov2_base_skill_deep_b6_4gpu_30k_20260827_v1}"
OUTPUT_ROOT="${BASE}/outputs/${RUN_ID}"
LOG_ROOT="${BASE}/logs/${RUN_ID}"
WANDB_ROOT="${BASE}/wandb/${RUN_ID}"
CACHE_ROOT="${BASE}/cache/${RUN_ID}"
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
export TORCH_COMPILE=0
export COMPILE_WARMUP_STEPS=0

TRAIN_CMD=(
    "${RUNTIME}/bin/python" -m torch.distributed.run
    --standalone
    --nnodes=1
    --nproc-per-node=4
    scripts/train_pytorch.py
    pi0_libero_object_dinov2_base_skill
    --exp-name "${RUN_ID}"
    --repo_id ybwowen/libero
    --local_root_dir "${DATA_ROOT}"
    --data.assets.asset-id ybwowen/libero
    --assets-base-dir "${ASSETS_ROOT}"
    --checkpoint-base-dir "${OUTPUT_ROOT}"
    --model.depth-model-name "${DINO_MODEL}"
    --batch-size 24
    --gradient-accumulation-steps 1
    --num-workers 2
    --pytorch-training-precision float32
    --use-gradient-checkpointing
    --ddp-find-unused-parameters
    --wandb-enabled
    --num-train-steps 30000
    --log-interval 10
    --save-interval 10000
    --val-interval 10000
    --val-max-batches 1
    --pytorch-weight-path "${STAGE1_CHECKPOINT}"
)

[[ -x "${RUNTIME}/bin/python" ]] || { echo "Missing Python runtime: ${RUNTIME}/bin/python"; exit 1; }
[[ -d "${DATA_ROOT}" ]] || { echo "Missing dataset: ${DATA_ROOT}"; exit 1; }
[[ -f "${STAGE1_CHECKPOINT}/model.safetensors" ]] || { echo "Missing Stage 1 checkpoint"; exit 1; }
[[ -f "${DINO_MODEL}/config.json" ]] || { echo "Missing DINO config: ${DINO_MODEL}/config.json"; exit 1; }
[[ -f "${DINO_MODEL}/model.safetensors" ]] || { echo "Missing DINO weights: ${DINO_MODEL}/model.safetensors"; exit 1; }
[[ -f "${TOKENIZER_PATH}" ]] || { echo "Missing tokenizer: ${TOKENIZER_PATH}"; exit 1; }

grep -q 'name="pi0_libero_object_dinov2_base_skill"' "${PROJECT_ROOT}/src/openpi/training/config.py" || {
    echo "Missing DINOv2 deep Stage-2 config."
    exit 2
}

IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
[[ "${#gpu_ids[@]}" -eq 4 ]] || { echo "Expected exactly 4 GPU IDs: ${CUDA_VISIBLE_DEVICES}"; exit 1; }

if [[ "${GUIDEDVLA_CHECK_ONLY:-0}" == "1" ]]; then
    echo "Startup path/config checks passed. Training was not started."
    echo "physical_global_batch=24 local_batch_per_rank=6 gradient_accumulation_steps=1 effective_global_batch=24"
    printf 'CUDA_VISIBLE_DEVICES=%q ' "${CUDA_VISIBLE_DEVICES}"
    printf '%q ' "${TRAIN_CMD[@]}"
    printf '\n'
    exit 0
fi

if [[ -e "${OUTPUT_ROOT}" || -e "${LOG_ROOT}" ]]; then
    echo "Refusing to reuse existing output/log directory for ${RUN_ID}."
    exit 2
fi

mkdir -p \
    "${OUTPUT_ROOT}" "${LOG_ROOT}" "${WANDB_ROOT}" \
    "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" \
    "${TRANSFORMERS_CACHE}" "${XDG_CACHE_HOME}" \
    "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" \
    "${CUDA_CACHE_PATH}" "${TMP_ROOT}"

{
    echo "manifest=${PROJECT_ROOT}/manifests/$(basename "$0")"
    echo "run_id=${RUN_ID}"
    echo "git_head=$(git rev-parse HEAD)"
    echo "encoder=dinov2-base deep-guidance layers=9,10,11,12"
    echo "encoder_checkpoint=${DINO_MODEL}"
    echo "physical_global_batch=24 local_batch_per_rank=6 gradient_accumulation_steps=1 effective_global_batch=24"
    echo "command:"
    printf '%q ' "${TRAIN_CMD[@]}"
    printf '\n'
} | tee "${LOG_ROOT}/launch_metadata.txt"

"${TRAIN_CMD[@]}" 2>&1 | tee "${LOG_ROOT}/train.log"
