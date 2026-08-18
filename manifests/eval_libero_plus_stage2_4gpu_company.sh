#!/usr/bin/env bash
set -euo pipefail

# Run from the GuidedVLA repository root in a job with exactly four visible GPUs:
#   bash manifests/eval_libero_plus_stage2_4gpu_company.sh
#
# The scheduler owns CUDA_VISIBLE_DEVICES. This launcher never overwrites it or
# assumes that allocated devices use host indices 0,1,2,3.

if [[ ! -f examples/libero_plus/eval_libero_plus.py || ! -f scripts/serve_policy.py ]]; then
    echo "Please run this command from the GuidedVLA repository root."
    exit 1
fi

PROJECT_ROOT="$(pwd)"
BASE="$(dirname "${PROJECT_ROOT}")"

SERVER_PYTHON="${GUIDEDVLA_SERVER_PYTHON:-${BASE}/runtime/.venv/bin/python}"
CLIENT_PYTHON="${GUIDEDVLA_CLIENT_PYTHON:-${BASE}/runtime/libero-plus-conda-py310/bin/python}"
LIBERO_PLUS_PATH="${GUIDEDVLA_LIBERO_PLUS_PATH:-${BASE}/LIBERO-plus-4976dc3}"
COMPANY_LIBERO_CONFIG_PATH="${GUIDEDVLA_COMPANY_LIBERO_CONFIG_PATH:-${BASE}/runtime/libero-plus-config}"
COMPANY_OSMESA_LIB_PATH="${GUIDEDVLA_COMPANY_OSMESA_LIB_PATH:-${BASE}/runtime/company-osmesa/lib}"
COMPANY_MAGICK_HOME="${GUIDEDVLA_COMPANY_MAGICK_HOME:-${BASE}/runtime/company-osmesa}"
DEPTH_MODEL="${GUIDEDVLA_DEPTH_MODEL_PATH:-${BASE}/models/da3-small-e08cab65}"
TOKENIZER_PATH="${GUIDEDVLA_TOKENIZER_PATH:-${BASE}/models/paligemma_tokenizer.model}"
REQUIRE_DEPTH_ASSETS="${GUIDEDVLA_REQUIRE_DEPTH_ASSETS:-1}"
EXPECTED_GPU_COUNT="${GUIDEDVLA_EXPECTED_GPU_COUNT:-4}"

case "${REQUIRE_DEPTH_ASSETS}" in
    0|1) ;;
    *)
        echo "GUIDEDVLA_REQUIRE_DEPTH_ASSETS must be 0 or 1, got: ${REQUIRE_DEPTH_ASSETS}"
        exit 1
        ;;
esac
[[ "${EXPECTED_GPU_COUNT}" =~ ^[1-9][0-9]*$ ]] || {
    echo "GUIDEDVLA_EXPECTED_GPU_COUNT must be a positive integer, got: ${EXPECTED_GPU_COUNT}"
    exit 1
}

RUN_ID="${GUIDEDVLA_EVAL_RUN_ID:-libero_plus_stage2_4gpu}"
CHECKPOINT="${GUIDEDVLA_CHECKPOINT:-${BASE}/outputs/guidedvla_libero_stage2_object_depth_skill_4gpu_30k/pi0_libero_object_depth_skill/guidedvla_libero_stage2_object_depth_skill_4gpu_30k/30000}"
POLICY_CONFIG="${GUIDEDVLA_POLICY_CONFIG:-pi0_libero_object_depth_skill}"
RESULTS_ROOT="${BASE}/eval_results/${RUN_ID}"
LOG_ROOT="${BASE}/logs/${RUN_ID}"
CACHE_ROOT="${BASE}/cache/${RUN_ID}"
TMP_ROOT="${GUIDEDVLA_TMP_ROOT:-${BASE}/tmp}"

export HF_HOME="${CACHE_ROOT}/hf"
export HF_HUB_CACHE="${CACHE_ROOT}/hf-hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/transformers"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export CUDA_CACHE_PATH="${CACHE_ROOT}/cuda"
export TMPDIR="${TMP_ROOT}"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/packages/openpi-client/src:${PROJECT_ROOT}/third_party/depth_anything/src:${LIBERO_PLUS_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
export OPENPI_PALIGEMMA_TOKENIZER_PATH="${TOKENIZER_PATH}"
if [[ "${REQUIRE_DEPTH_ASSETS}" == "1" ]]; then
    export OPENPI_DEPTH_MODEL_PATH="${DEPTH_MODEL}"
else
    unset OPENPI_DEPTH_MODEL_PATH
fi
export TORCH_COMPILE_DISABLE=1
export TORCH_COMPILE=0
export COMPILE_WARMUP_STEPS=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export LIBERO_CONFIG_PATH="${COMPANY_LIBERO_CONFIG_PATH}"
export LD_LIBRARY_PATH="${COMPANY_OSMESA_LIB_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MAGICK_HOME="${COMPANY_MAGICK_HOME}"
export MAGICK_CONFIGURE_PATH="${COMPANY_MAGICK_HOME}/imagemagick/ImageMagick-6:${COMPANY_MAGICK_HOME}/imagemagick/ImageMagick-6.9.11/config-Q16"
export MAGICK_CODER_MODULE_PATH="${COMPANY_MAGICK_HOME}/imagemagick/ImageMagick-6.9.11/modules-Q16/coders"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-${MUJOCO_GL}}"

[[ -x "${SERVER_PYTHON}" ]] || { echo "Missing policy server Python: ${SERVER_PYTHON}"; exit 1; }
[[ -x "${CLIENT_PYTHON}" ]] || { echo "Missing LIBERO-plus client Python: ${CLIENT_PYTHON}"; exit 1; }
[[ -f "${CHECKPOINT}/model.safetensors" ]] || { echo "Missing Stage 2 checkpoint: ${CHECKPOINT}"; exit 1; }
if [[ "${REQUIRE_DEPTH_ASSETS}" == "1" ]]; then
    [[ -f "${DEPTH_MODEL}/config.json" && -f "${DEPTH_MODEL}/model.safetensors" ]] || {
        echo "Missing DA3-SMALL model files in: ${DEPTH_MODEL}"
        exit 1
    }
fi
[[ -f "${TOKENIZER_PATH}" ]] || { echo "Missing tokenizer: ${TOKENIZER_PATH}"; exit 1; }
[[ -f "${LIBERO_PLUS_PATH}/libero/libero/benchmark/task_classification.json" ]] || {
    echo "Missing LIBERO-plus checkout or assets: ${LIBERO_PLUS_PATH}"
    exit 1
}
[[ -f "${COMPANY_OSMESA_LIB_PATH}/libOSMesa.so.8" ]] || {
    echo "Missing company OSMesa runtime: ${COMPANY_OSMESA_LIB_PATH}"
    exit 1
}
[[ -f "${COMPANY_OSMESA_LIB_PATH}/libMagickWand-6.Q16.so.6" ]] || {
    echo "Missing company MagickWand runtime: ${COMPANY_OSMESA_LIB_PATH}"
    exit 1
}
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is unavailable"; exit 1; }

mapfile -t visible_gpu_ids < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
[[ "${#visible_gpu_ids[@]}" -eq "${EXPECTED_GPU_COUNT}" ]] || {
    echo "Expected exactly ${EXPECTED_GPU_COUNT} scheduler-visible GPUs; nvidia-smi reports ${#visible_gpu_ids[@]}."
    exit 1
}

EVAL_CMD=(
    "${SERVER_PYTHON}" examples/libero_plus/eval_libero_plus.py
    --checkpoint-dir "${CHECKPOINT}"
    --policy-config "${POLICY_CONFIG}"
    --server-python "${SERVER_PYTHON}"
    --client-python "${CLIENT_PYTHON}"
    --libero-plus-path "${LIBERO_PLUS_PATH}"
    # A100-80GB validation showed substantially better throughput with four
    # independent server/client pairs per GPU; VRAM gates below remain active.
    --max-workers-per-gpu 4
    --estimated-worker-vram-gb 12
    --vram-safe-threshold 0.90
    --client-mujoco-gl "${MUJOCO_GL}"
    --results-base-dir "${RESULTS_ROOT}"
    --video-base-dir "${RESULTS_ROOT}/videos"
    --log-dir "${LOG_ROOT}"
    --start-port "${GUIDEDVLA_START_PORT:-18080}"
    --num-trials-per-task "${GUIDEDVLA_NUM_TRIALS_PER_TASK:-1}"
)

# Preserve the full four-suite default.  Set GUIDEDVLA_TASK_SUITES to pass an
# explicit comma-separated subset accepted by the evaluator, e.g. libero_object.
if [[ -n "${GUIDEDVLA_TASK_SUITES:-}" ]]; then
    EVAL_CMD+=(--task-suites "${GUIDEDVLA_TASK_SUITES}")
fi
if [[ -n "${GUIDEDVLA_CATEGORIES:-}" ]]; then
    EVAL_CMD+=(--categories "${GUIDEDVLA_CATEGORIES}")
fi
if [[ -n "${GUIDEDVLA_TASK_IDS:-}" ]]; then
    EVAL_CMD+=(--task-ids "${GUIDEDVLA_TASK_IDS}")
fi

mkdir -p \
    "${RESULTS_ROOT}" "${LOG_ROOT}" \
    "${COMPANY_LIBERO_CONFIG_PATH}" \
    "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" \
    "${XDG_CACHE_HOME}" "${TORCHINDUCTOR_CACHE_DIR}" \
    "${TRITON_CACHE_DIR}" "${CUDA_CACHE_PATH}" "${TMP_ROOT}"

COMPANY_LIBERO_CONFIG_FILE="${COMPANY_LIBERO_CONFIG_PATH}/config.yaml"
COMPANY_LIBERO_CONFIG_TMP="${COMPANY_LIBERO_CONFIG_PATH}/.config.yaml.tmp.$$"
printf '%s\n' \
    "benchmark_root: ${LIBERO_PLUS_PATH}/libero/libero" \
    "bddl_files: ${LIBERO_PLUS_PATH}/libero/libero/bddl_files" \
    "init_states: ${LIBERO_PLUS_PATH}/libero/libero/init_files" \
    "datasets: ${LIBERO_PLUS_PATH}/libero/datasets" \
    "assets: ${LIBERO_PLUS_PATH}/libero/libero/assets" \
    > "${COMPANY_LIBERO_CONFIG_TMP}"
mv "${COMPANY_LIBERO_CONFIG_TMP}" "${COMPANY_LIBERO_CONFIG_FILE}"

"${CLIENT_PYTHON}" -c \
    'import mujoco; from OpenGL import GL; from wand.api import library; assert GL.glGetError() == 0' \
    || { echo "Company client native runtime preflight failed."; exit 1; }

echo "Scheduler-visible GPUs: ${visible_gpu_ids[*]}"
echo "Expected scheduler-visible GPU count: ${EXPECTED_GPU_COUNT}"
echo "Policy server Python: ${SERVER_PYTHON}"
echo "LIBERO-plus client Python: ${CLIENT_PYTHON}"
echo "Company LIBERO config: ${COMPANY_LIBERO_CONFIG_FILE}"
echo "Company OSMesa runtime: ${COMPANY_OSMESA_LIB_PATH}"
echo "Company Magick runtime: ${COMPANY_MAGICK_HOME}"
echo "Policy config: ${POLICY_CONFIG}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Task suites: ${GUIDEDVLA_TASK_SUITES:-ALL}"
echo "Categories: ${GUIDEDVLA_CATEGORIES:-ALL}"
echo "Task IDs: ${GUIDEDVLA_TASK_IDS:-ALL}"
if [[ "${REQUIRE_DEPTH_ASSETS}" == "1" ]]; then
    echo "External encoder: ${DEPTH_MODEL}"
else
    echo "External encoder: intentionally not required for this policy config"
fi
echo "MuJoCo backend: ${MUJOCO_GL}"

if [[ "${GUIDEDVLA_CHECK_ONLY:-0}" == "1" ]]; then
    printf '%q ' "${EVAL_CMD[@]}"
    printf '\n'
    exit 0
fi

exec "${EVAL_CMD[@]}"
