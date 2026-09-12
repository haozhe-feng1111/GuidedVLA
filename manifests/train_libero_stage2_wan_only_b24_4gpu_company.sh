#!/usr/bin/env bash
# Platform startup command; one node with four GPUs. Does not create a platform job.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${GUIDEDVLA_BASE:-$(dirname "${PROJECT_ROOT}")}"
OWNER_ROOT="$(dirname "$(dirname "${BASE}")")"
RUNTIME="${RUNTIME:-${BASE}/runtime/.venv}"
DATA_ROOT="${DATA_ROOT:-${OWNER_ROOT}/datasets/public/ybwowen/libero-477f7959}"
ASSETS_ROOT="${ASSETS_ROOT:-${BASE}/assets}"
NORM_STATS_SOURCE="${NORM_STATS_SOURCE:-${ASSETS_ROOT}/pi0_libero_object_depth_skill}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${BASE}/models/paligemma_tokenizer.model}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-${BASE}/outputs/guidedvla_libero_stage1_4gpu_30k/pi0_libero/guidedvla_libero_stage1_4gpu_30k/30000}"
WAN22_SOURCE_ROOT="${WAN22_SOURCE_ROOT:-${BASE}/external/Wan2.2}"
WAN22_CHECKPOINT="${WAN22_CHECKPOINT:-${BASE}/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
WAN22_SOURCE_COMMIT=42bf4cfaa384bc21833865abc2f9e6c0e67233dc
WAN22_CHECKPOINT_SHA256=20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36
RUN_ID="${GUIDEDVLA_RUN_ID:-guidedvla_libero_stage2_wan_only_b24_ga1_30k_company_v1}"
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid RUN_ID' >&2; exit 2; }
OUTPUT_ROOT="${BASE}/outputs/${RUN_ID}"
LOG_ROOT="${BASE}/logs/${RUN_ID}"
CACHE_ROOT="${BASE}/cache/${RUN_ID}"
WANDB_ROOT="${BASE}/wandb/${RUN_ID}"
TMP_ROOT="/tmp/gvla-${UID}-$$"
cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/packages/openpi-client/src:${PROJECT_ROOT}/third_party/depth_anything/src"
export OPENPI_PALIGEMMA_TOKENIZER_PATH="${TOKENIZER_PATH}"
export OPENPI_WAN22_SOURCE_ROOT="${WAN22_SOURCE_ROOT}" OPENPI_WAN22_CHECKPOINT_PATH="${WAN22_CHECKPOINT}"
export HF_HOME="${CACHE_ROOT}/hf" HF_HUB_CACHE="${CACHE_ROOT}/hf-hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE" HF_DATASETS_CACHE="${CACHE_ROOT}/hf-datasets"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/transformers" XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor" TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export CUDA_CACHE_PATH="${CACHE_ROOT}/cuda" WANDB_DIR="${WANDB_ROOT}" WANDB_MODE=offline
export TMPDIR="$TMP_ROOT" TMP="$TMP_ROOT" TEMP="$TMP_ROOT"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TORCH_COMPILE=0 COMPILE_WARMUP_STEPS=0

ARGS=(pi0_libero_wan22_vae_only_b24
    --exp-name "$RUN_ID" --repo_id ybwowen/libero --local_root_dir "$DATA_ROOT"
    --data.assets.asset-id ybwowen/libero --data.assets.assets-dir "$NORM_STATS_SOURCE"
    --assets-base-dir "$ASSETS_ROOT" --checkpoint-base-dir "$OUTPUT_ROOT"
    --model.wan22-source-root "$WAN22_SOURCE_ROOT" --model.wan22-checkpoint-path "$WAN22_CHECKPOINT"
    --model.wan22-dtype bfloat16 --model.disable-image-augmentation
    --object-loss-weight 0 --skill-loss-weight 0
    --batch-size 24 --gradient-accumulation-steps 1 --num-workers 2
    --pytorch-training-precision float32 --use-gradient-checkpointing --ddp-find-unused-parameters
    --wandb-enabled --num-train-steps 30000 --log-interval 10
    --save-interval 10000 --val-interval 10000 --val-max-batches 1
    --pytorch-weight-path "$STAGE1_CHECKPOINT")
CMD=("${RUNTIME}/bin/python" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4
    scripts/train_pytorch.py "${ARGS[@]}")
if [[ "${GUIDEDVLA_PRINT_ONLY:-0}" == 1 ]]; then
    printf '%q ' "${CMD[@]}"; printf '\n'; exit 0
fi
[[ "${NNODES:-1}" == 1 ]] || { echo 'This launcher requires one node with four GPUs.' >&2; exit 2; }
for p in "$OUTPUT_ROOT" "$LOG_ROOT" "$CACHE_ROOT" "$WANDB_ROOT"; do
    [[ ! -e "$p" ]] || { echo "Refusing existing run directory: $p" >&2; exit 2; }
done
for p in "${RUNTIME}/bin/python" "${DATA_ROOT}/meta/info.json" \
    "${STAGE1_CHECKPOINT}/model.safetensors" "$TOKENIZER_PATH" \
    "${NORM_STATS_SOURCE}/ybwowen/libero/norm_stats.json" \
    "${WAN22_SOURCE_ROOT}/wan/modules/vae2_2.py" "$WAN22_CHECKPOINT"; do
    [[ -f "$p" ]] || { echo "Missing required asset: $p" >&2; exit 2; }
done
[[ "$(git -C "$WAN22_SOURCE_ROOT" rev-parse HEAD)" == "$WAN22_SOURCE_COMMIT" ]] || { echo 'Wan source commit mismatch' >&2; exit 2; }
[[ -z "$(git -C "$WAN22_SOURCE_ROOT" status --porcelain --untracked-files=no)" ]] || { echo 'Wan source has tracked modifications' >&2; exit 2; }
[[ "$(sha256sum "$WAN22_CHECKPOINT" | awk '{print $1}')" == "$WAN22_CHECKPOINT_SHA256" ]] || { echo 'Wan checkpoint hash mismatch' >&2; exit 2; }
mkdir -m 700 "$TMP_ROOT"
trap 'rm -rf -- "$TMP_ROOT"' EXIT
# Parse the production CLI on CPU before touching CUDA or starting workers.
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu HF_HOME="$TMP_ROOT/check/hf" HF_HUB_CACHE="$TMP_ROOT/check/hub" HUGGINGFACE_HUB_CACHE="$TMP_ROOT/check/hub" HF_DATASETS_CACHE="$TMP_ROOT/check/data" TRANSFORMERS_CACHE="$TMP_ROOT/check/transformers" XDG_CACHE_HOME="$TMP_ROOT/check/xdg" "${RUNTIME}/bin/python" scripts/check_wan_only_b24_config.py "${ARGS[@]}" >"$TMP_ROOT/config-check.log" 2>&1 || {
    cat "$TMP_ROOT/config-check.log"; exit 2;
}
cat "$TMP_ROOT/config-check.log"
if [[ "${GUIDEDVLA_CHECK_ONLY:-0}" == 1 ]]; then
    echo 'Asset/hash/resolved-config checks passed. No training or GPU work started.'; exit 0
fi
"${RUNTIME}/bin/python" - <<'PY'
import torch
assert torch.cuda.device_count() == 4, 'Exactly four visible GPUs are required'
for i in range(4):
    free, total = torch.cuda.mem_get_info(i)
    assert total >= 75 * 2**30, 'Use four 80GB-class GPUs; smaller cards are not validated'
    assert total - free < 2 * 2**30, f'GPU {i} is already occupied'
PY
# Atomic claim prevents two submissions with the default RUN_ID from racing.
mkdir -p "${BASE}/logs"
mkdir "$LOG_ROOT"
mkdir -p "$OUTPUT_ROOT" "$CACHE_ROOT" "$WANDB_ROOT"
cp "$TMP_ROOT/config-check.log" "$LOG_ROOT/resolved-config.log"
{
    echo "run_id=$RUN_ID"
    echo "source_revision=$(git rev-parse HEAD 2>/dev/null || cat .source_revision)"
    echo "wan22_source_commit=$WAN22_SOURCE_COMMIT wan22_checkpoint_sha256=$WAN22_CHECKPOINT_SHA256"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo 'world_size=4 local_batch=6 global_batch=24 accumulation=1'
    sha256sum scripts/train_pytorch.py scripts/check_wan_only_b24_config.py src/openpi/training/config.py \
        "${BASH_SOURCE[0]}" "${NORM_STATS_SOURCE}/ybwowen/libero/norm_stats.json" "${STAGE1_CHECKPOINT}/model.safetensors"
    printf '%q ' "${CMD[@]}"; printf '\n'
} >"$LOG_ROOT/launch_metadata.txt"
set +e
"${CMD[@]}" 2>&1 | tee "$LOG_ROOT/train.log"
status=${PIPESTATUS[0]}
echo "$status" >"$LOG_ROOT/exit_code"
exit "$status"
