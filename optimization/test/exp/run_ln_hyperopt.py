from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optimization.ln_hyperopt import main


if __name__ == "__main__":
    main(default_model="exp", default_eval_mode="test", default_out_dir=Path(__file__).resolve().parent / "OUTPUT")
