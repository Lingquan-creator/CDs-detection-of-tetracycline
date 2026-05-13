#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
OUT_DIR="$ROOT_DIR/OUTPUT/predict_exp"

"$PYTHON" "$OUT_DIR/evaluate_all_exp_params.py" \
  --params "$ROOT_DIR/dataset/train/model_fit/concentration_EXP/exp_fit_params.csv" \
  --roi-csv "$ROOT_DIR/dataset/test/roi_results/roi_detailed_statistics.csv" \
  --out-dir "$OUT_DIR"
