#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
python -m pip install -r "${SCRIPT_DIR}/requirements_pic_rgb_gui.txt"
python "${SCRIPT_DIR}/pic_rgb_gui.py"
