#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

pyinstaller \
  --noconfirm \
  --windowed \
  --name ConcentrationPredictor \
  --add-data "models:models" \
  --add-data "model_registry.json:." \
  app.py

echo "Built: ${SCRIPT_DIR}/dist/ConcentrationPredictor"
