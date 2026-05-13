#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
OUT_DIR="$ROOT_DIR/OUTPUT/predict_exp"

PARAMS="$ROOT_DIR/dataset/train/model_fit/concentration_EXP/exp_fit_params.csv"
ROI_CSV="$ROOT_DIR/dataset/test/roi_results/roi_detailed_statistics.csv"
PRED_CSV="$OUT_DIR/test_exp_predictions.csv"
PRED_PNG="$OUT_DIR/test_exp_builtin_plot.png"

"$PYTHON" "$ROOT_DIR/concentration_EXP/predict.py" \
  --params "$PARAMS" \
  --roi-csv "$ROI_CSV" \
  --output "$PRED_CSV" \
  --plot-out "$PRED_PNG"

"$PYTHON" "$OUT_DIR/visualize_exp_predictions.py" \
  --pred-csv "$PRED_CSV" \
  --out-dir "$OUT_DIR"
