#!/usr/bin/env bash
# RAFT-Large fnet single-frame encoder arm for GuidedVLA LIBERO Stage-2.
# Defines a run only; it does not submit a scheduler job itself.
set -euo pipefail

BASE=/mnt/dataset/haozhe_feng/personal/GuidedVLA
PROJECT_ROOT="${PROJECT_ROOT:-${BASE}/repo-raft-fnet-encoder-20260828}"
RUNTIME="${RUNTIME:-${BASE}/runtime/.venv}"
DATA_ROOT="${DATA_ROOT:-/mnt/dataset/haozhe_feng/datasets/public/ybwowen/libero-477f7959}"
ASSETS_ROOT="${ASSETS_ROOT:-${BASE}/assets}"
NORM_STATS_SOURCE="${NORM_STATS_SOURCE:-${ASSETS_ROOT}/pi0_libero_object_depth_skill}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${BASE}/models/paligemma_tokenizer.model}"
STAGE1_RUN_ID=guidedvla_libero_stage1_4gpu_30k
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-${BASE}/outputs/${STAGE1_RUN_ID}/pi0_libero/${STAGE1_RUN_ID}/30000}"
RAFT_CHECKPOINT="${RAFT_CHECKPOINT:-${BASE}/models/raft/raft_large_C_T_SKHT_V2-ff5fadd5.pth}"

CONFIG_NAME=pi0_libero_object_raft_large_fnet_skill
RUN_NAME="${RUN_NAME:-guidedvla_libero_stage2_object_raft_large_fnet_skill_4gpu_30k_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BASE}/outputs/${RUN_NAME}}"
LOG_ROOT="${LOG_ROOT:-${BASE}/logs/${RUN_NAME}}"
CACHE_ROOT="${CACHE_ROOT:-${BASE}/cache/${RUN_NAME}}"
WANDB_ROOT="${WANDB_ROOT:-${BASE}/wandb/${RUN_NAME}}"
TEMP_ROOT="${TEMP_ROOT:-${BASE}/tmp}"

GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<<"${GPU_IDS}"
[ "${#GPU_ARRAY[@]}" -eq 4 ] || { echo "Expected exactly 4 visible GPUs" >&2; exit 2; }

for required in \
    "${RUNTIME}/bin/python" \
    "${PROJECT_ROOT}/scripts/train_pytorch.py" \
    "${RAFT_CHECKPOINT}" \
    "${NORM_STATS_SOURCE}/ybwowen/libero/norm_stats.json" \
    "${STAGE1_CHECKPOINT}/model.safetensors" \
    "${TOKENIZER_PATH}"; do
    [ -e "${required}" ] || { echo "Missing required path: ${required}" >&2; exit 2; }
done

if [ -e "${OUTPUT_ROOT}" ] || [ -e "${LOG_ROOT}" ]; then
    echo "Refusing to reuse existing output/log directory for ${RUN_NAME}" >&2
    exit 2
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${WANDB_ROOT}" "${TEMP_ROOT}" \
    "${CACHE_ROOT}/hf" "${CACHE_ROOT}/hf-hub" "${CACHE_ROOT}/hf-datasets" \
    "${CACHE_ROOT}/transformers" "${CACHE_ROOT}/xdg" \
    "${CACHE_ROOT}/torchinductor" "${CACHE_ROOT}/triton" "${CACHE_ROOT}/cuda"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/packages/openpi-client/src:${PROJECT_ROOT}/third_party/depth_anything/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${CACHE_ROOT}/hf" HF_HUB_CACHE="${CACHE_ROOT}/hf-hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}" HF_DATASETS_CACHE="${CACHE_ROOT}/hf-datasets"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/transformers" XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor" TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export CUDA_CACHE_PATH="${CACHE_ROOT}/cuda" WANDB_DIR="${WANDB_ROOT}" WANDB_MODE="${WANDB_MODE:-offline}"
export TMPDIR="${TEMP_ROOT}" TEMP="${TEMP_ROOT}" TMP="${TEMP_ROOT}"
export OPENPI_PALIGEMMA_TOKENIZER_PATH="${TOKENIZER_PATH}"
export OPENPI_RAFT_CHECKPOINT_PATH="${RAFT_CHECKPOINT}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}" TORCH_COMPILE="${TORCH_COMPILE:-0}"
export COMPILE_WARMUP_STEPS="${COMPILE_WARMUP_STEPS:-0}"

cd "${PROJECT_ROOT}"
{
    echo "run_name=${RUN_NAME}"
    echo "encoder=torchvision RAFT-Large C_T_SKHT_V2 fnet only, current base camera frame, 224x224, frozen/eval"
    echo "tokens=28x28x256 bilinear_to_16x16x256 shared_across_4_independent_headwise_xavier_kv_projectors"
    echo "interpretation=single-frame visual prior; no optical flow, image pair, correlation, context net, or update block"
    echo "physical_global_batch=16 gradient_accumulation_steps=4 effective_global_batch=64"
    "${RUNTIME}/bin/python" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
        scripts/train_pytorch.py "${CONFIG_NAME}" \
        --exp-name "${RUN_NAME}" --repo_id ybwowen/libero --local_root_dir "${DATA_ROOT}" \
        --data.assets.asset-id ybwowen/libero --data.assets.assets-dir "${NORM_STATS_SOURCE}" \
        --assets-base-dir "${ASSETS_ROOT}" --checkpoint-base-dir "${OUTPUT_ROOT}" \
        --model.raft-checkpoint-path "${RAFT_CHECKPOINT}" \
        --batch-size 16 --gradient-accumulation-steps 4 --num-workers 2 \
        --pytorch-training-precision float32 --use-gradient-checkpointing \
        --ddp-find-unused-parameters --wandb-enabled --num-train-steps 30000 \
        --log-interval 10 --save-interval 10000 --val-interval 10000 --val-max-batches 1 \
        --pytorch-weight-path "${STAGE1_CHECKPOINT}"
} 2>&1 | tee "${LOG_ROOT}/train.log"
