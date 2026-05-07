#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FIT_DATA_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DATASET_DIR="${FIT_DATA_DIR}/dataset"

export PYTHONPATH="${FIT_DATA_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

run_fit() {
  local sample_name="$1"
  local input_csv="${DATASET_DIR}/${sample_name}.csv"
  local output_csv="${SCRIPT_DIR}/${sample_name}_results_spec.csv"
  local output_txt="${SCRIPT_DIR}/${sample_name}_results_spec.txt"
  local output_png="${SCRIPT_DIR}/${sample_name}_results_spec.png"
  local output_log="${SCRIPT_DIR}/${sample_name}_fit.log"

  if [[ ! -f "${input_csv}" ]]; then
    echo "Missing input CSV: ${input_csv}" >&2
    return 1
  fi

  echo "Fitting ${input_csv}"
  echo "  log: ${output_log}"
  python - "${FIT_DATA_DIR}" "${input_csv}" "${output_csv}" "${output_txt}" "${output_png}" >"${output_log}" 2>&1 <<'PY'
import sys
from pathlib import Path

fit_data_dir = Path(sys.argv[1]).resolve()
input_csv = Path(sys.argv[2]).resolve()
output_csv = Path(sys.argv[3]).resolve()
output_txt = Path(sys.argv[4]).resolve()
output_png = Path(sys.argv[5]).resolve()

sys.path.insert(0, str(fit_data_dir))

import fit_config as cfg
import fit_spectrum

cfg.DATA_FILE = input_csv
cfg.SAVE_CSV = True
cfg.CSV_OUTPUT_FILE = output_csv
cfg.SAVE_RESULTS = True
cfg.RESULTS_FILE = output_txt
cfg.SAVE_FIGURE = True
cfg.FIGURE_FILE = output_png

fit_spectrum.main()
PY
  echo "  results: ${output_csv}"
}

run_fit egg
run_fit milk

echo "Done. Results written to ${SCRIPT_DIR}"
