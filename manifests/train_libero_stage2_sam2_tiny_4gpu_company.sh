#!/usr/bin/env bash
# SAM2.1 Tiny external-encoder arm for the matched GuidedVLA LIBERO Stage-2 ablation.
# This launcher only defines the run; it never submits a scheduler job itself.
set -euo pipefail

BASE=/mnt/dataset/haozhe_feng/personal/GuidedVLA
PROJECT_ROOT="${PROJECT_ROOT:-${BASE}/repo-libero-sam2-tiny-encoder-20260816}"
RUNTIME="${RUNTIME:-${BASE}/runtime/.venv}"
DATA_ROOT="${DATA_ROOT:-/mnt/dataset/haozhe_feng/datasets/public/ybwowen/libero-477f7959}"
ASSETS_ROOT="${ASSETS_ROOT:-${BASE}/assets}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${BASE}/models/paligemma_tokenizer.model}"
SAM2_ROOT="${SAM2_ROOT:-${PROJECT_ROOT}/third_party/sam2}"
SAM2_DEPS_ROOT="${SAM2_DEPS_ROOT:-${PROJECT_ROOT}/third_party/sam2_deps}"
SAM2_CONFIG="${SAM2_CONFIG:-configs/sam2.1/sam2.1_hiera_t.yaml}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-${BASE}/models/sam2.1-hiera-tiny/sam2.1_hiera_tiny.pt}"
SAM2_SOURCE_COMMIT="${SAM2_SOURCE_COMMIT:-2b90b9f5ceec907a1c18123530e92e794ad901a4}"
SAM2_CHECKPOINT_SHA256="${SAM2_CHECKPOINT_SHA256:-7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-${BASE}/outputs/guidedvla_libero_stage1_4gpu_30k/pi0_libero/guidedvla_libero_stage1_4gpu_30k/30000}"

CONFIG_NAME="${CONFIG_NAME:-pi0_libero_object_sam2_tiny_skill}"
NORM_STATS="${NORM_STATS:-${ASSETS_ROOT}/${CONFIG_NAME}/ybwowen/libero/norm_stats.json}"
RUN_NAME="${RUN_NAME:-guidedvla_libero_stage2_object_sam2_tiny_skill_4gpu_30k_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BASE}/outputs/${RUN_NAME}}"
LOG_ROOT="${LOG_ROOT:-${BASE}/logs/${RUN_NAME}}"
CACHE_ROOT="${CACHE_ROOT:-${BASE}/cache/${RUN_NAME}}"
WANDB_ROOT="${WANDB_ROOT:-${BASE}/wandb/${RUN_NAME}}"
# Keep forkserver socket paths short while remaining on the personal disk.
TEMP_ROOT="${TEMP_ROOT:-${BASE}/tmp}"

# The benchmark-matched physical global batch is 16 (4 ranks x local 4);
# accumulation 4 gives effective global batch 64. Do not change independently.
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<<"${GPU_IDS}"
if [ "${#GPU_ARRAY[@]}" -ne 4 ]; then
    echo "Expected exactly 4 visible GPUs, got CUDA_VISIBLE_DEVICES=${GPU_IDS}" >&2
    exit 2
fi

for required in \
    "${RUNTIME}/bin/python" \
    "${PROJECT_ROOT}/scripts/train_pytorch.py" \
    "${SAM2_ROOT}/sam2/build_sam.py" \
    "${SAM2_DEPS_ROOT}/hydra/__init__.py" \
    "${SAM2_DEPS_ROOT}/iopath/__init__.py" \
    "${SAM2_CHECKPOINT}" \
    "${NORM_STATS}" \
    "${STAGE1_CHECKPOINT}" \
    "${TOKENIZER_PATH}"; do
    if [ ! -e "${required}" ]; then
        echo "Missing required path: ${required}" >&2
        exit 2
    fi
done

ACTUAL_SAM2_SOURCE_COMMIT="$(git -C "${SAM2_ROOT}" rev-parse HEAD)"
if [ "${ACTUAL_SAM2_SOURCE_COMMIT}" != "${SAM2_SOURCE_COMMIT}" ]; then
    echo "SAM2 source commit mismatch: expected ${SAM2_SOURCE_COMMIT}, got ${ACTUAL_SAM2_SOURCE_COMMIT}" >&2
    exit 2
fi
ACTUAL_SAM2_CHECKPOINT_SHA256="$(sha256sum "${SAM2_CHECKPOINT}" | awk '{print $1}')"
if [ "${ACTUAL_SAM2_CHECKPOINT_SHA256}" != "${SAM2_CHECKPOINT_SHA256}" ]; then
    echo "SAM2 checkpoint SHA-256 mismatch for ${SAM2_CHECKPOINT}" >&2
    exit 2
fi

# A new run name is required: never append to or overwrite another arm's state.
if [ -e "${OUTPUT_ROOT}" ] || [ -e "${LOG_ROOT}" ]; then
    echo "Refusing to reuse existing output/log directory for ${RUN_NAME}" >&2
    exit 2
fi

mkdir -p \
    "${OUTPUT_ROOT}" "${LOG_ROOT}" "${WANDB_ROOT}" "${TEMP_ROOT}" \
    "${CACHE_ROOT}/hf" "${CACHE_ROOT}/hf-hub" "${CACHE_ROOT}/hf-datasets" \
    "${CACHE_ROOT}/transformers" "${CACHE_ROOT}/xdg" \
    "${CACHE_ROOT}/torchinductor" "${CACHE_ROOT}/triton" "${CACHE_ROOT}/cuda"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/packages/openpi-client/src:${PROJECT_ROOT}/third_party/depth_anything/src:${SAM2_ROOT}:${SAM2_DEPS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${CACHE_ROOT}/hf"
export HF_HUB_CACHE="${CACHE_ROOT}/hf-hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${CACHE_ROOT}/hf-datasets"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/transformers"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export CUDA_CACHE_PATH="${CACHE_ROOT}/cuda"
export WANDB_DIR="${WANDB_ROOT}"
export TMPDIR="${TEMP_ROOT}"
export TEMP="${TEMP_ROOT}"
export TMP="${TEMP_ROOT}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export OPENPI_PALIGEMMA_TOKENIZER_PATH="${TOKENIZER_PATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export TORCH_COMPILE="${TORCH_COMPILE:-0}"
export COMPILE_WARMUP_STEPS="${COMPILE_WARMUP_STEPS:-0}"
export OPENPI_SAM2_MODEL_CONFIG="${SAM2_CONFIG}"
export OPENPI_SAM2_CHECKPOINT_PATH="${SAM2_CHECKPOINT}"

cd "${PROJECT_ROOT}"

{
    echo "run_name=${RUN_NAME}"
    echo "git_head=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
    echo "sam2_source_head=${ACTUAL_SAM2_SOURCE_COMMIT}"
    echo "sam2_checkpoint_sha256=${ACTUAL_SAM2_CHECKPOINT_SHA256}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
    echo "physical_global_batch=16"
    echo "gradient_accumulation_steps=4"
    echo "effective_global_batch=64"
    "${RUNTIME}/bin/python" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
        scripts/train_pytorch.py "${CONFIG_NAME}" \
        --exp-name "${RUN_NAME}" \
        --repo_id ybwowen/libero \
        --local_root_dir "${DATA_ROOT}" \
        --data.assets.asset-id ybwowen/libero \
        --assets-base-dir "${ASSETS_ROOT}" \
        --checkpoint-base-dir "${OUTPUT_ROOT}" \
        --model.sam2-model-config "${SAM2_CONFIG}" \
        --model.sam2-checkpoint-path "${SAM2_CHECKPOINT}" \
        --model.sam2-image-size 1024 \
        --model.sam2-token-grid-size 16 \
        --batch-size 16 \
        --gradient-accumulation-steps 4 \
        --num-workers 2 \
        --pytorch-training-precision float32 \
        --use-gradient-checkpointing \
        --ddp-find-unused-parameters \
        --wandb-enabled \
        --num-train-steps 30000 \
        --log-interval 10 \
        --save-interval 10000 \
        --val-interval 10000 \
        --val-max-batches 1 \
        --pytorch-weight-path "${STAGE1_CHECKPOINT}"
} 2>&1 | tee "${LOG_ROOT}/train.log"
