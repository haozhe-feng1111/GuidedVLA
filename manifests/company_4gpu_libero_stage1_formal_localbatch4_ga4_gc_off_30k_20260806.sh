#!/usr/bin/env bash
set -Eeuo pipefail

# Company 4-GPU Stage-1 run for the formal GuidedVLA LIBERO reproduction.
# Run with no external exports:
#   bash manifests/company_4gpu_libero_stage1_formal_localbatch4_ga4_gc_off_30k_20260806.sh
#
# 4 GPUs × micro-batch 4 = physical global batch 16; accumulation 4 gives
# effective global batch 64. Activation gradient checkpointing is disabled.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${GUIDEDVLA_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
BASE="${GUIDEDVLA_BASE:-$(cd -- "${PROJECT_ROOT}/.." && pwd)}"
OWNER_ROOT="$(cd -- "${BASE}/../.." && pwd)"
RUNTIME="${GUIDEDVLA_RUNTIME:-${BASE}/runtime/.venv}"
if [[ ! -x "${RUNTIME}/bin/python" ]]; then
    RUNTIME="${BASE}/runtime/pi0-conversion-env"
fi
DATA_ROOT="${GUIDEDVLA_DATA_ROOT:-${OWNER_ROOT}/datasets/public/ybwowen/libero-477f7959}"
PI0_BASE="${GUIDEDVLA_PI0_BASE:-${BASE}/models/pi0_base_pytorch_float32}"
TOKENIZER_PATH="${GUIDEDVLA_TOKENIZER_PATH:-${BASE}/models/paligemma_tokenizer.model}"
ASSETS_ROOT="${GUIDEDVLA_ASSETS_ROOT:-${BASE}/assets}"

RUN_ID="${GUIDEDVLA_RUN_ID:-company_4gpu_pi0_libero_stage1_localbatch4_ga4_gc_off_30k_20260806}"
OUTPUT_ROOT="${BASE}/outputs/${RUN_ID}"
LOG_ROOT="${BASE}/logs/${RUN_ID}"
CACHE_ROOT="${BASE}/cache/${RUN_ID}"
STATUS_PATH="${LOG_ROOT}/status.txt"
PREFLIGHT_ROOT="${LOG_ROOT}/preflight"
LAUNCHER_PATH="${PROJECT_ROOT}/manifests/$(basename -- "${BASH_SOURCE[0]}")"

AUTHOR_COMMIT="04be059e0d6bd448be5cb45fdbafc775f7eb5e38"
FORMAL_CODE_COMMIT="e2ed8d6a9a32e3110c6260dafb5db5c65107b403"
STAGE1_CONFIG="pi0_libero"
STAGE1_RUN="pi0_libero_company_4gpu_localbatch4_ga4_gc_off_30k_20260806"
STAGE1_DIR="${OUTPUT_ROOT}/${STAGE1_CONFIG}/${STAGE1_RUN}"
STATS1="${ASSETS_ROOT}/${STAGE1_CONFIG}/ybwowen/libero/norm_stats.json"

NPROC=4
LOCAL_BATCH=4
PHYSICAL_GLOBAL_BATCH=$((NPROC * LOCAL_BATCH))
GRADIENT_ACCUMULATION_STEPS=4
EFFECTIVE_GLOBAL_BATCH=$((PHYSICAL_GLOBAL_BATCH * GRADIENT_ACCUMULATION_STEPS))
NUM_TRAIN_STEPS=30000
SAVE_INTERVAL=10000
LOG_INTERVAL=10
VAL_INTERVAL=10000
VAL_MAX_BATCHES=1
MIN_GPU_MEMORY_BYTES="${GUIDEDVLA_MIN_GPU_MEMORY_BYTES:-79000000000}"

test ! -e "${OUTPUT_ROOT}"
test ! -e "${LOG_ROOT}"
test ! -e "${CACHE_ROOT}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${PREFLIGHT_ROOT}" \
    "${CACHE_ROOT}/hf" "${CACHE_ROOT}/triton" "${CACHE_ROOT}/torchinductor" "${CACHE_ROOT}/xdg"
exec > >(tee -a "${LOG_ROOT}/launcher.log") 2>&1
printf 'RUNNING\n' >"${STATUS_PATH}"

completed=0
monitor_pid=""
on_exit() {
    exit_code=$?
    if [[ -n "${monitor_pid}" ]]; then
        kill "${monitor_pid}" 2>/dev/null || true
        wait "${monitor_pid}" 2>/dev/null || true
    fi
    if [[ ${exit_code} -eq 0 && ${completed} -eq 1 ]]; then
        printf 'SUCCESS\n' >"${STATUS_PATH}"
    else
        printf 'FAILED exit_code=%s completed=%s\n' "${exit_code}" "${completed}" >"${STATUS_PATH}"
    fi
}
trap on_exit EXIT

test -x "${RUNTIME}/bin/python"
test -f "${DATA_ROOT}/meta/info.json"
test -f "${DATA_ROOT}/meta/stats.json"
test -f "${DATA_ROOT}/meta/tasks.parquet"
test -f "${PI0_BASE}/model.safetensors"
test -f "${TOKENIZER_PATH}"
test -f "${STATS1}"
test -f "${LAUNCHER_PATH}"

test "$(git -C "${PROJECT_ROOT}" merge-base "${AUTHOR_COMMIT}" HEAD)" = "${AUTHOR_COMMIT}"
test "$(git -C "${PROJECT_ROOT}" merge-base "${FORMAL_CODE_COMMIT}" HEAD)" = "${FORMAL_CODE_COMMIT}"
test "$(git -C "${PROJECT_ROOT}" hash-object scripts/train_pytorch.py)" = \
    "$(git -C "${PROJECT_ROOT}" rev-parse "${FORMAL_CODE_COMMIT}:scripts/train_pytorch.py")"
test "$(git -C "${PROJECT_ROOT}" hash-object src/openpi/training/config.py)" = \
    "$(git -C "${PROJECT_ROOT}" rev-parse "${FORMAL_CODE_COMMIT}:src/openpi/training/config.py")"
test -z "$(git -C "${PROJECT_ROOT}" diff --name-only --ignore-submodules=all)"

sha256sum "${DATA_ROOT}/meta/info.json" | awk '{print $1}' | grep -qx \
    "b94a2f0abe4a75b13935869f5d8aca7e746e7473c6f8f60e35fb245d74c067e6"
sha256sum "${DATA_ROOT}/meta/stats.json" | awk '{print $1}' | grep -qx \
    "7a188908edbf8b8465b6c663c66a863fbc85cc75482a012b91a5c957ac73e926"
sha256sum "${DATA_ROOT}/meta/tasks.parquet" | awk '{print $1}' | grep -qx \
    "3166b8b5a12b19b5ae39c3300ffef15c676c3f49a5e73bf4400b8baf7b316748"
sha256sum "${STATS1}" | awk '{print $1}' | grep -qx \
    "9393d032f6caf10b6433b50a0c23b09a211f773fd42ee14142cb691149a93855"
sha256sum "${PI0_BASE}/model.safetensors" | awk '{print $1}' | grep -qx \
    "c275e91bd2727e2f1651528f323548b441108e50d32d9aec679e11dc3705c362"
sha256sum "${TOKENIZER_PATH}" | awk '{print $1}' | grep -qx \
    "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"

export CUDA_VISIBLE_DEVICES="${GUIDEDVLA_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/packages/openpi-client/src:${PROJECT_ROOT}/third_party/depth_anything/src"
export OPENPI_PALIGEMMA_TOKENIZER_PATH="${TOKENIZER_PATH}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,expandable_segments:True"
export HF_HOME="${CACHE_ROOT}/hf"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/hf/hub"
export HF_DATASETS_CACHE="${CACHE_ROOT}/hf/datasets"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export WANDB_MODE=disabled
export COMPILE_WARMUP_STEPS=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

"${RUNTIME}/bin/python" - "${NPROC}" "${MIN_GPU_MEMORY_BYTES}" <<'PY'
import sys

import torch

expected_count, minimum_memory = map(int, sys.argv[1:])
assert torch.cuda.device_count() == expected_count, torch.cuda.device_count()
for index in range(expected_count):
    properties = torch.cuda.get_device_properties(index)
    assert properties.total_memory >= minimum_memory, properties.total_memory
    assert "A100" in properties.name or "A800" in properties.name, properties.name
    print(f"GPU[{index}]={properties.name} total_memory_bytes={properties.total_memory}")
print("per_gpu_batch=4 physical_global_batch=16 GA=4 effective_global_batch=64 GC=false")
PY

"${RUNTIME}/bin/python" - <<'PY'
from openpi.training import config as training_config

config = training_config._CONFIGS_DICT["pi0_libero"]  # noqa: SLF001
assert config.use_gradient_checkpointing is False
assert config.gradient_accumulation_steps == 1
assert config.pytorch_training_precision == "float32"
print("STAGE1_CONFIG_OK default_GC=false; launcher overrides physical_batch=16 and GA=4")
PY

printf '%s\n' \
    "project_root=${PROJECT_ROOT}" \
    "runtime=${RUNTIME}" \
    "data_root=${DATA_ROOT}" \
    "run_id=${RUN_ID}" \
    "world_size=${NPROC}" \
    "per_gpu_micro_batch=${LOCAL_BATCH}" \
    "physical_global_batch=${PHYSICAL_GLOBAL_BATCH}" \
    "gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}" \
    "effective_global_batch=${EFFECTIVE_GLOBAL_BATCH}" \
    "activation_gradient_checkpointing=false" \
    "num_train_steps=${NUM_TRAIN_STEPS}" \
    "save_interval=${SAVE_INTERVAL}" \
    "log_interval=${LOG_INTERVAL}" \
    "val_interval=${VAL_INTERVAL}" \
    "val_max_batches=${VAL_MAX_BATCHES}" \
    "wandb=disabled" \
    "stage1_init=${PI0_BASE}" >"${PREFLIGHT_ROOT}/settings.txt"
git -C "${PROJECT_ROOT}" rev-parse HEAD >"${PREFLIGHT_ROOT}/git_commit.txt"
sha256sum "${LAUNCHER_PATH}" "${PROJECT_ROOT}/scripts/train_pytorch.py" \
    "${PROJECT_ROOT}/src/openpi/training/config.py" >"${PREFLIGHT_ROOT}/code_sha256.txt"
nvidia-smi -L >"${PREFLIGHT_ROOT}/gpu.txt"

(
    while true; do
        date --iso-8601=seconds
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
            --format=csv,noheader,nounits
        sleep 5
    done
) >"${LOG_ROOT}/gpu_memory.log" 2>&1 &
monitor_pid=$!

cd "${PROJECT_ROOT}"
"${RUNTIME}/bin/python" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node="${NPROC}" \
    scripts/train_pytorch.py "${STAGE1_CONFIG}" \
    --exp-name "${STAGE1_RUN}" \
    --repo_id ybwowen/libero \
    --local_root_dir "${DATA_ROOT}" \
    --data.assets.asset-id ybwowen/libero \
    --assets-base-dir "${ASSETS_ROOT}" \
    --checkpoint-base-dir "${OUTPUT_ROOT}" \
    --batch-size "${PHYSICAL_GLOBAL_BATCH}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --num-workers 2 \
    --pytorch-training-precision float32 \
    --no-wandb-enabled \
    --num-train-steps "${NUM_TRAIN_STEPS}" \
    --log-interval "${LOG_INTERVAL}" \
    --save-interval "${SAVE_INTERVAL}" \
    --val-interval "${VAL_INTERVAL}" \
    --val-max-batches "${VAL_MAX_BATCHES}" \
    --pytorch-weight-path "${PI0_BASE}" \
    2>&1 | tee "${LOG_ROOT}/${STAGE1_RUN}.log"

for step in 10000 20000 30000; do
    for filename in model.safetensors optimizer.pt metadata.pt; do
        test -f "${STAGE1_DIR}/${step}/${filename}"
    done
done
grep -q "Disabled gradient checkpointing for PI0Pytorch model" "${LOG_ROOT}/${STAGE1_RUN}.log"
grep -q "gradient_checkpointing=False" "${LOG_ROOT}/${STAGE1_RUN}.log"
printf 'STAGE1_TRAINING_COMPLETE\n' >"${LOG_ROOT}/stage1_complete.txt"
completed=1
