#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"

"$PYTHON" "$ROOT_DIR/optimization/test/train_test_hyperopt/run_train_test_hyperopt.py" "$@"
