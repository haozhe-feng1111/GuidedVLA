#!/usr/bin/env bash
set -euo pipefail

BASE=/data/junke_dont_remove/haozhe/guidedvla_repro_20260730
REPO="$BASE/repo-libero-da3-shallow-fast-train-20260821"
RUNTIME="$BASE/runtime/pi0-conversion-env"
RUN=guidedvla_libero_stage2_object_da3_skill_shallow_guidance_fast_4gpu_30k_20260821_v1
OUT="$BASE/outputs/$RUN"
LOG="$BASE/logs/$RUN"
CACHE="$BASE/cache/$RUN"
TMP=/tmp/gv-da3-shallow-fast-formal-v1

test ! -e "$OUT"
test ! -e "$LOG"
test -f "$BASE/models/da3-small-e08cab65/model.safetensors"
test -f "$BASE/outputs/guidedvla_libero_stage1_4gpu_30k/pi0_libero/guidedvla_libero_stage1_4gpu_30k/30000/model.safetensors"
test "$(git -C "$REPO" rev-parse HEAD)" = f794ac2522c136c31572f72dbddb364a3c2c9fcf
git -C "$REPO" diff --check
test "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n '5,8p' | awk '{sum += $1} END {print sum + 0}')" -eq 0
available_bytes=$(df --output=avail -B1 "$BASE" | tail -1)
test "$available_bytes" -ge 429496729600
mkdir -p "$LOG/preflight" "$CACHE" "$TMP"

git -C "$REPO" diff > "$LOG/preflight/worktree.diff"
sha256sum "$LOG/preflight/worktree.diff" > "$LOG/preflight/worktree.diff.sha256"
sha256sum "$BASE/models/da3-small-e08cab65/config.json" \
  "$BASE/models/paligemma_tokenizer.model" > "$LOG/preflight/input_small_files.sha256"
git -C "$REPO" rev-parse HEAD > "$LOG/preflight/git_head.txt"
nvidia-smi -q > "$LOG/preflight/nvidia_smi_q.txt"
df -h /data > "$LOG/preflight/df_h_data.txt"

export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONPATH="$REPO/src:$REPO/packages/openpi-client/src:$BASE/repo-libero-author-ga2-20260731/third_party/depth_anything/src"
export TORCH_COMPILE=0 COMPILE_WARMUP_STEPS=0
export GUIDEDVLA_FREEZE_DEAD_PALI_TAIL=1 GUIDEDVLA_REPORT_UNUSED_PARAMS=1
export GUIDEDVLA_DIAGNOSTIC_SKIP_CHECKPOINT=0
export OPENPI_PALIGEMMA_TOKENIZER_PATH="$BASE/models/paligemma_tokenizer.model"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export TMPDIR="$TMP" TEMP="$TMP" TMP="$TMP"
export HF_HOME="$CACHE/hf" HF_HUB_CACHE="$CACHE/hf-hub" TRANSFORMERS_CACHE="$CACHE/transformers" XDG_CACHE_HOME="$CACHE/xdg"
export HF_DATASETS_CACHE="$BASE/cache/dino_frozen_fp32_4gpu_eager_ab_20260820/hf/datasets"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/torchinductor" TRITON_CACHE_DIR="$CACHE/triton" CUDA_CACHE_PATH="$CACHE/cuda"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
"$RUNTIME/bin/python" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
  scripts/train_pytorch.py pi0_libero_object_depth_skill_shallow_guidance \
  --exp-name "$RUN" --checkpoint-base-dir "$OUT" \
  --repo_id ybwowen/libero --local_root_dir "$BASE/datasets/ybwowen-libero-477f7959" \
  --data.assets.asset-id ybwowen/libero \
  --data.assets.assets-dir "$BASE/assets/pi0_libero_object_depth_skill" \
  --assets-base-dir "$BASE/assets" \
  --model.depth-model-name "$BASE/models/da3-small-e08cab65" \
  --batch-size 16 --gradient-accumulation-steps 4 --num-workers 2 \
  --pytorch-training-precision float32 --use-gradient-checkpointing \
  --wandb-enabled --num-train-steps 30000 --log-interval 10 --save-interval 10000 \
  --val-interval 10000 --val-max-batches 1 \
  --pytorch-weight-path "$BASE/outputs/guidedvla_libero_stage1_4gpu_30k/pi0_libero/guidedvla_libero_stage1_4gpu_30k/30000" \
  > "$LOG/train.log" 2>&1
