#!/usr/bin/env bash
# Matched MAE/DINOv3 ViT-B/16 Stage-2 encoder ablations. Defines a run only.
set -euo pipefail

BASE=/mnt/dataset/haozhe_feng/personal/GuidedVLA
PROJECT_ROOT="${PROJECT_ROOT:-${BASE}/repo-libero-mae-dinov3-base-encoders-20260817}"
RUNTIME="${RUNTIME:-${BASE}/runtime/.venv}"
DATA_ROOT="${DATA_ROOT:-/mnt/dataset/haozhe_feng/datasets/public/ybwowen/libero-477f7959}"
ASSETS_ROOT="${ASSETS_ROOT:-${BASE}/assets}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${BASE}/models/paligemma_tokenizer.model}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-${BASE}/outputs/guidedvla_libero_stage1_4gpu_30k/pi0_libero/guidedvla_libero_stage1_4gpu_30k/30000}"
MAE_DEPS_ROOT="${MAE_DEPS_ROOT:-${PROJECT_ROOT}/third_party/mae_deps}"
ENCODER_KIND="${ENCODER_KIND:?Set ENCODER_KIND to mae or dinov3}"

case "${ENCODER_KIND}" in
    mae)
        CONFIG_NAME=pi0_libero_object_mae_base_skill
        DEFAULT_RUN_NAME=guidedvla_libero_stage2_object_mae_base_skill_4gpu_30k_v1
        SOURCE_ROOT="${SOURCE_ROOT:-}"
        CHECKPOINT_PATH="${CHECKPOINT_PATH:-${BASE}/models/mae-vit-base-patch16/mae_pretrain_vit_base.pth}"
        ;;
    dinov3)
        CONFIG_NAME=pi0_libero_object_dinov3_base_skill
        DEFAULT_RUN_NAME=guidedvla_libero_stage2_object_dinov3_base_skill_4gpu_30k_v1
        SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}/third_party/dinov3}"
        CHECKPOINT_PATH="${CHECKPOINT_PATH:-${BASE}/models/dinov3-vitb16/dinov3_vitb16_pretrain_lvd1689m.pth}"
        ;;
    *) echo "ENCODER_KIND must be mae or dinov3" >&2; exit 2 ;;
esac

RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"
NORM_STATS="${NORM_STATS:-${ASSETS_ROOT}/${CONFIG_NAME}/ybwowen/libero/norm_stats.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BASE}/outputs/${RUN_NAME}}"
LOG_ROOT="${LOG_ROOT:-${BASE}/logs/${RUN_NAME}}"
CACHE_ROOT="${CACHE_ROOT:-${BASE}/cache/${RUN_NAME}}"
WANDB_ROOT="${WANDB_ROOT:-${BASE}/wandb/${RUN_NAME}}"
TEMP_ROOT="${TEMP_ROOT:-${BASE}/tmp}"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<<"${GPU_IDS}"
[ "${#GPU_ARRAY[@]}" -eq 4 ] || { echo "Expected exactly 4 visible GPUs" >&2; exit 2; }

for required in "${RUNTIME}/bin/python" "${PROJECT_ROOT}/scripts/train_pytorch.py" \
    "${CHECKPOINT_PATH}" "${NORM_STATS}" "${STAGE1_CHECKPOINT}" "${TOKENIZER_PATH}"; do
    [ -e "${required}" ] || { echo "Missing required path: ${required}" >&2; exit 2; }
done
if [ "${ENCODER_KIND}" = dinov3 ] && [ ! -f "${SOURCE_ROOT}/hubconf.py" ]; then
    echo "Missing official DINOv3 source: ${SOURCE_ROOT}/hubconf.py" >&2
    exit 2
fi
if [ "${ENCODER_KIND}" = mae ] && [ ! -f "${MAE_DEPS_ROOT}/timm/__init__.py" ]; then
    echo "Missing isolated timm dependency: ${MAE_DEPS_ROOT}/timm/__init__.py" >&2
    exit 2
fi
if [ -e "${OUTPUT_ROOT}" ] || [ -e "${LOG_ROOT}" ]; then
    echo "Refusing to reuse existing output/log directory for ${RUN_NAME}" >&2
    exit 2
fi
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${WANDB_ROOT}" "${TEMP_ROOT}" \
    "${CACHE_ROOT}/hf" "${CACHE_ROOT}/hf-hub" "${CACHE_ROOT}/hf-datasets" \
    "${CACHE_ROOT}/transformers" "${CACHE_ROOT}/xdg" \
    "${CACHE_ROOT}/torchinductor" "${CACHE_ROOT}/triton" "${CACHE_ROOT}/cuda"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/packages/openpi-client/src:${PROJECT_ROOT}/third_party/depth_anything/src:${MAE_DEPS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${CACHE_ROOT}/hf" HF_HUB_CACHE="${CACHE_ROOT}/hf-hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}" HF_DATASETS_CACHE="${CACHE_ROOT}/hf-datasets"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/transformers" XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor" TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export CUDA_CACHE_PATH="${CACHE_ROOT}/cuda" WANDB_DIR="${WANDB_ROOT}" WANDB_MODE="${WANDB_MODE:-offline}"
export TMPDIR="${TEMP_ROOT}" TEMP="${TEMP_ROOT}" TMP="${TEMP_ROOT}"
export OPENPI_PALIGEMMA_TOKENIZER_PATH="${TOKENIZER_PATH}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}" TORCH_COMPILE="${TORCH_COMPILE:-0}"
export COMPILE_WARMUP_STEPS="${COMPILE_WARMUP_STEPS:-0}"

cd "${PROJECT_ROOT}"
{
    echo "run_name=${RUN_NAME}"
    echo "git_head=$(git rev-parse HEAD)"
    echo "encoder_kind=${ENCODER_KIND} checkpoint_path=${CHECKPOINT_PATH}"
    echo "source_grid=14x14 target_grid=16x16 interpolation=bilinear"
    echo "physical_global_batch=16 gradient_accumulation_steps=4 effective_global_batch=64"
    "${RUNTIME}/bin/python" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
        scripts/train_pytorch.py "${CONFIG_NAME}" \
        --exp-name "${RUN_NAME}" --repo_id ybwowen/libero --local_root_dir "${DATA_ROOT}" \
        --data.assets.asset-id ybwowen/libero --assets-base-dir "${ASSETS_ROOT}" \
        --checkpoint-base-dir "${OUTPUT_ROOT}" \
        --model.patch16-encoder-kind "${ENCODER_KIND}" \
        --model.patch16-source-root "${SOURCE_ROOT}" \
        --model.patch16-checkpoint-path "${CHECKPOINT_PATH}" \
        --batch-size 16 --gradient-accumulation-steps 4 --num-workers 2 \
        --pytorch-training-precision float32 --use-gradient-checkpointing \
        --ddp-find-unused-parameters --wandb-enabled --num-train-steps 30000 \
        --log-interval 10 --save-interval 10000 --val-interval 10000 --val-max-batches 1 \
        --pytorch-weight-path "${STAGE1_CHECKPOINT}"
} 2>&1 | tee "${LOG_ROOT}/train.log"
