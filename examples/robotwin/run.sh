#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
ROBOWIN_ROOT="${REPO_ROOT}/third_party/RoboTwin"
ENTRYPOINT="${SCRIPT_DIR}/main.py"

export CUROBO_USE_LRU_CACHE=1

if [[ ! -d "${ROBOWIN_ROOT}" ]]; then
  echo "RoboTwin root not found: ${ROBOWIN_ROOT}" >&2
  exit 1
fi

if [[ ! -f "${ENTRYPOINT}" ]]; then
  echo "RobotWin entrypoint not found: ${ENTRYPOINT}" >&2
  exit 1
fi

PYTHON="python"

cd "${ROBOWIN_ROOT}"
exec "${PYTHON}" "${ENTRYPOINT}" "$@"
