#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
python -m pip install -r "${SCRIPT_DIR}/requirements_pic_rgb_gui.txt"

python -m PyInstaller \
  --noconfirm \
  --windowed \
  --name PicRGBRoiTool \
  --paths "${REPO_ROOT}/pic_RGB" \
  --hidden-import rawpy._rawpy \
  --collect-submodules rawpy \
  --distpath "${SCRIPT_DIR}/dist" \
  --workpath "${SCRIPT_DIR}/build" \
  --specpath "${SCRIPT_DIR}" \
  "${SCRIPT_DIR}/pic_rgb_gui.py"

echo "Built: ${SCRIPT_DIR}/dist/PicRGBRoiTool"
