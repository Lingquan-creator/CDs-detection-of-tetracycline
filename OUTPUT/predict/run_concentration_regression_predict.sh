#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
OUT_DIR="$ROOT_DIR/OUTPUT/predict"

MODEL="$ROOT_DIR/dataset/train/model_fit/concentration_regression/ridge_model.pkl"
ROI_CSV="$ROOT_DIR/dataset/test/roi_results/roi_detailed_statistics.csv"
PRED_CSV="$OUT_DIR/test_ridge_predictions.csv"
PRED_PNG="$OUT_DIR/test_ridge_builtin_plot.png"

"$PYTHON" "$ROOT_DIR/concentration_regression/predict.py" \
  --model "$MODEL" \
  --roi-csv "$ROI_CSV" \
  --out "$PRED_CSV" \
  --plot-out "$PRED_PNG"

"$PYTHON" "$OUT_DIR/visualize_predictions.py" \
  --pred-csv "$PRED_CSV" \
  --out-dir "$OUT_DIR"
