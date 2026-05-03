#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Virtualenv python not found: ${VENV_DIR}/bin/python" >&2
  exit 1
fi

source "${VENV_DIR}/bin/activate"
python "${SCRIPT_DIR}/run_regression_parameter_scan.py" "$@"
