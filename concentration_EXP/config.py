"""
Configuration for response-function fitting concentration pipeline.

The default training script compares exponential, linear, and quadratic RGB responses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

# Base paths
BASE_DIR = Path(__file__).resolve().parent
ROI_STATS_PATH = BASE_DIR.parent / "dataset" / "train" / "roi_results" / "roi_detailed_statistics.csv"
TEST_ROI_STATS_PATH = BASE_DIR.parent / "dataset" / "test" / "roi_results" / "roi_detailed_statistics.csv"
OUTPUT_DIR = BASE_DIR / "exp_fit_output"

# Response fitting defaults
DEFAULT_CHANNEL = "ln_G_sig"  # current train->test optimized EXP selected channel
DEFAULT_FEATURE_LIST: Tuple[str, ...] = ("ln_R_sig", "ln_G_sig", "ln_B_sig")
DEFAULT_WEIGHT_EPS = 0.0  # current train->test optimized EXP weighting parameter

# Output filenames
PARAMS_FILENAME = "exp_fit_params.csv"
PREDICTIONS_FILENAME = "exp_fit_predictions.csv"
CONCENTRATION_SUFFIX = "_pred_concentration_exp.csv"
FIT_PLOT_FILENAME = "exp_fit_evaluation.png"
CONCENTRATION_PLOT_SUFFIX = "_pred_concentration_exp.png"


@dataclass
class ExpConfig:
    """Configuration for exponential fitting."""

    roi_csv: Path = ROI_STATS_PATH
    output_dir: Path = OUTPUT_DIR
    channel: str = DEFAULT_CHANNEL
    feature_list: Tuple[str, ...] = DEFAULT_FEATURE_LIST
    weight_eps: float = DEFAULT_WEIGHT_EPS

    @property
    def params_path(self) -> Path:
        return self.output_dir / PARAMS_FILENAME

    @property
    def predictions_path(self) -> Path:
        return self.output_dir / PREDICTIONS_FILENAME

    @property
    def fit_plot_path(self) -> Path:
        return self.output_dir / FIT_PLOT_FILENAME

    def concentration_path_for(self, roi_csv: Path) -> Path:
        """Build output path for concentration predictions."""
        return self.output_dir / f"{roi_csv.stem}{CONCENTRATION_SUFFIX}"

    def concentration_plot_path_for(self, roi_csv: Path) -> Path:
        """Build output path for concentration prediction visualization."""
        return self.output_dir / f"{roi_csv.stem}{CONCENTRATION_PLOT_SUFFIX}"
