#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT_DIR}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python"
fi

cd "${ROOT_DIR}"

"${PYTHON}" try_pic_RGB/scan_feature_curve_matching.py "$@"
"${PYTHON}" try_pic_RGB/apply_best_and_rerun.py

echo "Done. Summary: ${ROOT_DIR}/try_pic_RGB/final_summary.txt"
