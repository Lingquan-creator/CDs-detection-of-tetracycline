#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"

TEST_DATASET_NAME="${TEST_DATASET_NAME:-food}"
MODEL_DATASET_NAME="${MODEL_DATASET_NAME:-mate20x}"
ROI_CSV="${ROI_CSV:-${ROOT_DIR}/dataset/${TEST_DATASET_NAME}/roi_detailed_statistics.csv}"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/OUTPUT/${MODEL_DATASET_NAME}}"
PRED_OUTPUT_DIR="${PRED_OUTPUT_DIR:-${ROOT_DIR}/OUTPUT/${TEST_DATASET_NAME}/${MODEL_DATASET_NAME}_model_predictions}"

EXP_PARAMS="${EXP_PARAMS:-${MODEL_DIR}/exp_fit_params.csv}"
RGB_MODEL="${RGB_MODEL:-${MODEL_DIR}/rgb_concentration_model.pkl}"
GS_MODEL="${GS_MODEL:-${MODEL_DIR}/gs_amortized_posterior_model.pt}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Virtualenv python not found: ${VENV_DIR}/bin/python" >&2
  exit 1
fi

if [[ ! -f "${ROI_CSV}" ]]; then
  echo "ROI CSV not found: ${ROI_CSV}" >&2
  exit 1
fi

if [[ ! -f "${EXP_PARAMS}" ]]; then
  echo "EXP params not found: ${EXP_PARAMS}" >&2
  exit 1
fi

if [[ ! -f "${RGB_MODEL}" ]]; then
  echo "RGB model not found: ${RGB_MODEL}" >&2
  exit 1
fi

if [[ ! -f "${GS_MODEL}" ]]; then
  echo "GS model not found: ${GS_MODEL}" >&2
  exit 1
fi

source "${VENV_DIR}/bin/activate"
mkdir -p "${PRED_OUTPUT_DIR}"

EXP_PRED_CSV="${PRED_OUTPUT_DIR}/exp_predictions.csv"
EXP_PLOT="${PRED_OUTPUT_DIR}/exp_prediction_distribution.png"
RGB_PRED_CSV="${PRED_OUTPUT_DIR}/rgb_predictions.csv"
RGB_PLOT="${PRED_OUTPUT_DIR}/rgb_prediction_distribution.png"
GS_SUMMARY_CSV="${PRED_OUTPUT_DIR}/gs_posterior_summary.csv"
GS_GRID_CSV="${PRED_OUTPUT_DIR}/gs_posterior_probabilities.csv"
GS_PLOT="${PRED_OUTPUT_DIR}/gs_prediction_distribution.png"
METRICS_CSV="${PRED_OUTPUT_DIR}/prediction_metrics.csv"
METRICS_TXT="${PRED_OUTPUT_DIR}/prediction_metrics.txt"
EVAL_PLOT="${PRED_OUTPUT_DIR}/prediction_evaluation.png"
GROUP_PLOT_DIR="${PRED_OUTPUT_DIR}/group_evaluation_plots"

echo "============================================================"
echo "ROI CSV:   ${ROI_CSV}"
echo "Model dir: ${MODEL_DIR}"
echo "Output:    ${PRED_OUTPUT_DIR}"

echo "[EXP] Predicting with exponential model..."
python "${ROOT_DIR}/concentration_EXP/predict.py" \
  --params "${EXP_PARAMS}" \
  --roi-csv "${ROI_CSV}" \
  --channel light_best \
  --output "${EXP_PRED_CSV}" \
  --plot-out "${EXP_PLOT}" \
  2>&1 | tee "${PRED_OUTPUT_DIR}/exp_predict.log"

echo "[RGB] Predicting with Ridge model..."
python "${ROOT_DIR}/concentration_regression/predict.py" \
  --model "${RGB_MODEL}" \
  --roi-csv "${ROI_CSV}" \
  --out "${RGB_PRED_CSV}" \
  --plot-out "${RGB_PLOT}" \
  2>&1 | tee "${PRED_OUTPUT_DIR}/rgb_predict.log"

echo "[GS] Predicting with amortized posterior model..."
python "${ROOT_DIR}/concentration_GS/predict.py" \
  --model "${GS_MODEL}" \
  --roi-csv "${ROI_CSV}" \
  --out "${GS_SUMMARY_CSV}" \
  --grid-out "${GS_GRID_CSV}" \
  --plot-out "${GS_PLOT}" \
  2>&1 | tee "${PRED_OUTPUT_DIR}/gs_predict.log"

echo "[EVAL] Extracting ground truth from filenames and computing grouped metrics..."
python - "${EXP_PRED_CSV}" "${RGB_PRED_CSV}" "${GS_SUMMARY_CSV}" "${METRICS_CSV}" "${METRICS_TXT}" "${EVAL_PLOT}" "${GROUP_PLOT_DIR}" <<'PY'
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def concentration_from_filename(filename: object) -> float:
    stem = Path(str(filename)).stem
    match = NUMBER_RE.search(stem)
    return float(match.group(0)) if match else math.nan


def group_from_filename(filename: object) -> str:
    stem = Path(str(filename)).stem.lower()
    if stem.startswith("chicken"):
        return "chicken"
    if stem.startswith("egg"):
        return "egg"
    return "other"


def metric_row(model_name: str, group: str, df: pd.DataFrame, pred_col: str) -> dict[str, float | str | int]:
    source = df if group == "all" else df[df["sample_group"] == group]
    y_true = source["conc_true"].to_numpy(dtype=float)
    y_pred = source[pred_col].to_numpy(dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        return {
            "model": model_name,
            "sample_group": group,
            "prediction_column": pred_col,
            "n": 0,
            "r2": math.nan,
            "rmse": math.nan,
            "mae": math.nan,
            "mean_error": math.nan,
            "median_abs_error": math.nan,
            "max_abs_error": math.nan,
        }

    y_true = y_true[valid]
    y_pred = y_pred[valid]
    residual = y_pred - y_true
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = math.nan if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    return {
        "model": model_name,
        "sample_group": group,
        "prediction_column": pred_col,
        "n": int(valid.sum()),
        "r2": r2,
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
        "mean_error": float(np.mean(residual)),
        "median_abs_error": float(np.median(np.abs(residual))),
        "max_abs_error": float(np.max(np.abs(residual))),
    }


def add_truth_group_and_errors(df: pd.DataFrame, pred_col: str, residual_col: str) -> pd.DataFrame:
    df = df.copy()
    df["conc_true"] = df["filename"].apply(concentration_from_filename)
    df["sample_group"] = df["filename"].apply(group_from_filename)
    df[residual_col] = df[pred_col].astype(float) - df["conc_true"].astype(float)
    df[f"abs_error_{residual_col.removeprefix('residual_')}"] = df[residual_col].abs()
    return df


def draw_panel(ax: plt.Axes, df: pd.DataFrame, pred_col: str, title: str, metrics: dict[str, object]) -> None:
    y_true = df["conc_true"].to_numpy(dtype=float)
    y_pred = df[pred_col].to_numpy(dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        ax.text(0.5, 0.5, "No valid labels", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    x = y_true[valid]
    y = y_pred[valid]
    ax.scatter(x, y, s=36, alpha=0.85, color="#1f77b4", edgecolor="white", linewidth=0.4)
    low = float(min(np.min(x), np.min(y)))
    high = float(max(np.max(x), np.max(y)))
    pad = max((high - low) * 0.04, 1.0)
    low -= pad
    high += pad
    ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1.0)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Ground truth from filename (uM)")
    ax.set_ylabel("Predicted concentration (uM)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.text(
        0.04,
        0.96,
        f"R2={float(metrics['r2']):.4g}\nRMSE={float(metrics['rmse']):.4g}\nMAE={float(metrics['mae']):.4g}\nn={int(metrics['n'])}",
        ha="left",
        va="top",
        transform=ax.transAxes,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.9},
    )


def draw_eval_plot(
    exp: pd.DataFrame,
    rgb: pd.DataFrame,
    gs: pd.DataFrame,
    rows: list[dict[str, object]],
    group: str,
    out_path: Path,
) -> None:
    group_exp = exp if group == "all" else exp[exp["sample_group"] == group]
    group_rgb = rgb if group == "all" else rgb[rgb["sample_group"] == group]
    group_gs = gs if group == "all" else gs[gs["sample_group"] == group]
    row_map = {(row["model"], row["sample_group"]): row for row in rows}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=150)
    draw_panel(axes[0], group_exp, "conc_pred", f"EXP - {group}", row_map[("exp", group)])
    draw_panel(axes[1], group_rgb, "conc_pred", f"RGB Ridge - {group}", row_map[("rgb_ridge", group)])
    draw_panel(axes[2], group_gs, "posterior_median", f"GS median - {group}", row_map[("gs_posterior_median", group)])
    fig.suptitle(f"Food dataset prediction ({group})", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


exp_csv = Path(sys.argv[1])
rgb_csv = Path(sys.argv[2])
gs_csv = Path(sys.argv[3])
metrics_csv = Path(sys.argv[4])
metrics_txt = Path(sys.argv[5])
eval_plot = Path(sys.argv[6])
group_plot_dir = Path(sys.argv[7])

exp = add_truth_group_and_errors(pd.read_csv(exp_csv), "conc_pred", "residual")
rgb = add_truth_group_and_errors(pd.read_csv(rgb_csv), "conc_pred", "residual")
gs = add_truth_group_and_errors(pd.read_csv(gs_csv), "posterior_median", "residual_median")
gs["residual_mean"] = gs["posterior_mean"].astype(float) - gs["conc_true"].astype(float)
gs["abs_error_mean"] = gs["residual_mean"].abs()

exp.to_csv(exp_csv, index=False, float_format="%.6g")
rgb.to_csv(rgb_csv, index=False, float_format="%.6g")
gs.to_csv(gs_csv, index=False, float_format="%.6g")

groups = ["all"] + sorted(set(exp["sample_group"]) | set(rgb["sample_group"]) | set(gs["sample_group"]))
rows = []
for group in groups:
    rows.extend(
        [
            metric_row("exp", group, exp, "conc_pred"),
            metric_row("rgb_ridge", group, rgb, "conc_pred"),
            metric_row("gs_posterior_median", group, gs, "posterior_median"),
            metric_row("gs_posterior_mean", group, gs, "posterior_mean"),
        ]
    )

metrics = pd.DataFrame(rows)
metrics_csv.parent.mkdir(parents=True, exist_ok=True)
metrics.to_csv(metrics_csv, index=False, float_format="%.6g")

with metrics_txt.open("w", encoding="utf-8") as f:
    f.write(f"EXP predictions: {exp_csv}\n")
    f.write(f"RGB predictions: {rgb_csv}\n")
    f.write(f"GS summary:      {gs_csv}\n")
    f.write(f"Evaluation plot: {eval_plot}\n")
    f.write(f"Group plots:     {group_plot_dir}\n\n")
    for group in groups:
        f.write(f"[{group}]\n")
        for row in [r for r in rows if r["sample_group"] == group]:
            f.write(
                "  {model} ({prediction_column}): n={n}, R2={r2:.6g}, RMSE={rmse:.6g}, "
                "MAE={mae:.6g}, mean_error={mean_error:.6g}, "
                "median_abs_error={median_abs_error:.6g}, max_abs_error={max_abs_error:.6g}\n".format(**row)
            )
        f.write("\n")

draw_eval_plot(exp, rgb, gs, rows, "all", eval_plot)
for group in groups:
    if group == "all":
        continue
    draw_eval_plot(exp, rgb, gs, rows, group, group_plot_dir / f"prediction_evaluation_{group}.png")

print(f"Updated EXP CSV   -> {exp_csv}")
print(f"Updated RGB CSV   -> {rgb_csv}")
print(f"Updated GS CSV    -> {gs_csv}")
print(f"Saved metrics CSV -> {metrics_csv}")
print(f"Saved metrics TXT -> {metrics_txt}")
print(f"Saved eval plot   -> {eval_plot}")
print(f"Saved group plots -> {group_plot_dir}")
print(metrics.to_string(index=False))
PY

echo "Prediction job completed."
