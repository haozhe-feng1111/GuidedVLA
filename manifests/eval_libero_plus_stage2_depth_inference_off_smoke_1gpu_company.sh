#!/usr/bin/env bash
set -euo pipefail

# One-GPU, one-rollout integration smoke for the inference-only depth-off ablation.
# It is engineering evidence only, not a benchmark score or full ablation result.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BASE="$(dirname "${PROJECT_ROOT}")"
RUN_ID="${GUIDEDVLA_EVAL_RUN_ID:-libero_plus_stage2_30k_depth_inference_off_smoke_1gpu_spatial_objlayout_task1725_v1}"
LOG_ROOT="${BASE}/logs/${RUN_ID}"
RESULTS_ROOT="${BASE}/eval_results/${RUN_ID}"
CACHE_ROOT="${BASE}/cache/${RUN_ID}"
TMP_ROOT="${BASE}/tmp/${RUN_ID}"
LIBERO_CONFIG_ROOT="${BASE}/runtime/libero-plus-config/${RUN_ID}"
CHECKPOINT="${BASE}/outputs/guidedvla_libero_stage2_object_depth_skill_4gpu_30k/pi0_libero_object_depth_skill/guidedvla_libero_stage2_object_depth_skill_4gpu_30k/30000"

for path in "${RESULTS_ROOT}" "${CACHE_ROOT}" "${TMP_ROOT}" "${LIBERO_CONFIG_ROOT}"; do
    [[ ! -e "${path}" ]] || {
        echo "Refusing to reuse existing smoke path: ${path}"
        exit 1
    }
done
[[ ! -e "${LOG_ROOT}" ]] || {
    echo "Refusing to reuse existing smoke log path: ${LOG_ROOT}"
    exit 1
}
mkdir -p "${LOG_ROOT}"

STATUS_FILE="${LOG_ROOT}/status.txt"
on_exit() {
    local exit_code=$?
    trap - EXIT
    printf 'EXIT_CODE=%s\nFINISHED_AT=%s\n' "${exit_code}" "$(date -Is)" >"${STATUS_FILE}"
    exit "${exit_code}"
}
trap on_exit EXIT
printf 'RUNNING\nSTARTED_AT=%s\n' "$(date -Is)" >"${STATUS_FILE}"

git -C "${PROJECT_ROOT}" diff --check
git -C "${PROJECT_ROOT}" diff --binary >"${LOG_ROOT}/code.diff"
{
    echo "timestamp=$(date -Is)"
    echo "hostname=$(hostname)"
    echo "project_root=${PROJECT_ROOT}"
    echo "run_id=${RUN_ID}"
    echo "checkpoint=${CHECKPOINT}"
    echo "policy_config=pi0_libero_object_depth_skill_depth_inference_off"
    echo "task_suites=libero_spatial"
    echo "categories=Objects Layout"
    echo "task_ids=1725"
    echo "num_trials_per_task=1"
    echo "expected_gpu_count=1"
    echo "depth_assets_required=0"
    echo "--- git head ---"
    git -C "${PROJECT_ROOT}" rev-parse HEAD
    echo "--- git status ---"
    git -C "${PROJECT_ROOT}" status --short
    echo "--- tracked diff hash ---"
    sha256sum "${LOG_ROOT}/code.diff"
    echo "--- launcher hashes ---"
    sha256sum         "${SCRIPT_DIR}/eval_libero_plus_stage2_4gpu_company.sh"         "${SCRIPT_DIR}/eval_libero_plus_stage2_depth_inference_off_4gpu_company.sh"         "${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
    echo "--- checkpoint files ---"
    sha256sum "${CHECKPOINT}/model.safetensors" "${CHECKPOINT}/metadata.pt"
    echo "--- command ---"
    printf '%q ' env         GUIDEDVLA_EVAL_RUN_ID="${RUN_ID}"         GUIDEDVLA_TASK_SUITES=libero_spatial         GUIDEDVLA_CATEGORIES="Objects Layout"         GUIDEDVLA_TASK_IDS=1725         GUIDEDVLA_NUM_TRIALS_PER_TASK=1         GUIDEDVLA_EXPECTED_GPU_COUNT=1         GUIDEDVLA_START_PORT=18420         GUIDEDVLA_TMP_ROOT="${TMP_ROOT}"         GUIDEDVLA_COMPANY_LIBERO_CONFIG_PATH="${LIBERO_CONFIG_ROOT}"         bash "${SCRIPT_DIR}/eval_libero_plus_stage2_depth_inference_off_4gpu_company.sh"
    printf '\n'
} >"${LOG_ROOT}/SMOKE_MANIFEST.txt" 2>&1

export GUIDEDVLA_EVAL_RUN_ID="${RUN_ID}"
export GUIDEDVLA_TASK_SUITES=libero_spatial
export GUIDEDVLA_CATEGORIES="Objects Layout"
export GUIDEDVLA_TASK_IDS=1725
export GUIDEDVLA_NUM_TRIALS_PER_TASK=1
export GUIDEDVLA_EXPECTED_GPU_COUNT=1
export GUIDEDVLA_START_PORT=18420
export GUIDEDVLA_TMP_ROOT="${TMP_ROOT}"
export GUIDEDVLA_COMPANY_LIBERO_CONFIG_PATH="${LIBERO_CONFIG_ROOT}"

cd "${PROJECT_ROOT}"
bash "${SCRIPT_DIR}/eval_libero_plus_stage2_depth_inference_off_4gpu_company.sh" 2>&1 | tee "${LOG_ROOT}/launcher.log"
