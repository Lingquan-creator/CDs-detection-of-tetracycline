#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATASET_DIR="${DATASET_DIR:-${ROOT_DIR}/dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/OUTPUT}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
TRAIN_DATASET_NAME="${TRAIN_DATASET_NAME:-}"
ROI_CSV_NAME="${ROI_CSV_NAME:-roi_detailed_statistics.csv}"
ROI_CSV="${ROI_CSV:-}"
EXCLUDE_DATASET_NAMES="${EXCLUDE_DATASET_NAMES:-food}"

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "Dataset directory not found: ${DATASET_DIR}" >&2
  exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Virtualenv python not found: ${VENV_DIR}/bin/python" >&2
  exit 1
fi

source "${VENV_DIR}/bin/activate"
mkdir -p "${OUTPUT_DIR}"

is_excluded_dataset() {
  local rel_path="$1"
  local top_dir="${rel_path%%/*}"
  local name

  IFS=',' read -ra excluded_names <<< "${EXCLUDE_DATASET_NAMES}"
  for name in "${excluded_names[@]}"; do
    name="${name//[[:space:]]/}"
    if [[ -n "${name}" && "${top_dir}" == "${name}" ]]; then
      return 0
    fi
  done
  return 1
}

roi_csvs=()

if [[ -n "${ROI_CSV}" ]]; then
  roi_csvs+=("${ROI_CSV}")
elif [[ -n "${TRAIN_DATASET_NAME}" ]]; then
  roi_csvs+=("${DATASET_DIR}/${TRAIN_DATASET_NAME}/${ROI_CSV_NAME}")
else
  while IFS= read -r csv_path; do
    rel_path="${csv_path#${DATASET_DIR}/}"
    if is_excluded_dataset "${rel_path}"; then
      continue
    fi
    roi_csvs+=("${csv_path}")
  done < <(find "${DATASET_DIR}" -type f -name "${ROI_CSV_NAME}" | sort)
fi

if [[ "${#roi_csvs[@]}" -eq 0 ]]; then
  echo "No training ROI CSV files found under: ${DATASET_DIR}" >&2
  exit 1
fi

echo "Found ${#roi_csvs[@]} training ROI CSV file(s). Excluded top-level dataset(s): ${EXCLUDE_DATASET_NAMES}"

for ROI_CSV in "${roi_csvs[@]}"; do
  if [[ ! -f "${ROI_CSV}" ]]; then
    echo "Training ROI CSV not found: ${ROI_CSV}" >&2
    exit 1
  fi

  rel_path="${ROI_CSV#${DATASET_DIR}/}"
  rel_dir="$(dirname "${rel_path}")"
  out_dir="${OUTPUT_DIR}/${rel_dir}"
  mkdir -p "${out_dir}"

  echo "============================================================"
  echo "Train input: ${ROI_CSV}"
  echo "Output:      ${out_dir}"

  echo "[EXP] Training exponential model..."
  python "${ROOT_DIR}/concentration_EXP/train.py" \
    --roi-csv "${ROI_CSV}" \
    --channel light_best \
    --params-out "${out_dir}/exp_fit_params.csv" \
    --pred-out "${out_dir}/exp_fit_predictions.csv" \
    --conc-out "${out_dir}/exp_concentration_predictions.csv" \
    --plot-out "${out_dir}/exp_fit_evaluation.png" \
    2>&1 | tee "${out_dir}/exp_train.log"

  echo "[GS] Training amortized posterior model and LOO test..."
  python "${ROOT_DIR}/concentration_GS/train.py" \
    --roi-csv "${ROI_CSV}" \
    --use-bg true \
    --feature-set compact \
    --model-out "${out_dir}/gs_amortized_posterior_model.pt" \
    --train-out "${out_dir}/gs_train_posterior_summary.csv" \
    --loo-out "${out_dir}/gs_loo_posterior_summary.csv" \
    --train-grid-out "${out_dir}/gs_train_posterior_probabilities.csv" \
    --loo-grid-out "${out_dir}/gs_loo_posterior_probabilities.csv" \
    --plot-out "${out_dir}/gs_posterior_evaluation.png" \
    2>&1 | tee "${out_dir}/gs_train.log"

  echo "[RGB] Training Ridge regression model and LOO test..."
  python "${ROOT_DIR}/concentration_regression/train.py" \
    --roi-csv "${ROI_CSV}" \
    --use-bg true \
    --feature-set compact \
    --bg-correction subtract \
    --model-out "${out_dir}/rgb_concentration_model.pkl" \
    --pred-out "${out_dir}/rgb_ridge_predictions.csv" \
    --loo-out "${out_dir}/rgb_ridge_predictions_loo.csv" \
    --coef-out "${out_dir}/rgb_ridge_coefficients.csv" \
    --plot-out "${out_dir}/rgb_ridge_evaluation.png" \
    2>&1 | tee "${out_dir}/rgb_train.log"

  echo "Training job completed: ${rel_path}"
done

echo "All training jobs completed."
