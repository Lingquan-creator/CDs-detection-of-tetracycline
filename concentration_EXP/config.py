"""
Configuration for exponential fitting concentration pipeline.

Simplified config for exponential model: I = a * exp(b * C) + c
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
ROI_STATS_PATH = BASE_DIR.parent / "pic_RGB" / "roi_results2" / "roi_detailed_statistics.csv"
OUTPUT_DIR = BASE_DIR / "exp_fit_output"

# Exponential fitting defaults
DEFAULT_CHANNEL = "G1_corr"  # Which corrected channel to use for final concentration inversion
DEFAULT_WEIGHT_EPS = 5.0  # Weighting parameter: weights ~ 1/(conc + eps)

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
