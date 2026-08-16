#!/usr/bin/env bash
set -euo pipefail

# Full four-GPU evaluation wrapper for the explicit RGB-only inference ablation.
# It intentionally reuses the standard launcher so checkpoint, protocol, and
# runtime settings remain identical apart from the named policy configuration.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export GUIDEDVLA_EVAL_RUN_ID="${GUIDEDVLA_EVAL_RUN_ID:-libero_plus_stage2_depth_inference_off_4gpu}"
export GUIDEDVLA_POLICY_CONFIG="pi0_libero_object_depth_skill_depth_inference_off"
# The policy config explicitly omits the depth encoder and depth KV adapter.
# Do not require or export a DA3 checkpoint for this inference-only ablation.
export GUIDEDVLA_REQUIRE_DEPTH_ASSETS=0
exec bash "${SCRIPT_DIR}/eval_libero_plus_stage2_4gpu_company.sh" "$@"
