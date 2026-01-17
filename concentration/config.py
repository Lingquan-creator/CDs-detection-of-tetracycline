"""
Central configuration for the concentration pipeline (physics-informed Ridge regression).

- Paths: keep training/inference inputs and outputs in one place.
- Defaults: share alpha / g-mode across scripts while allowing CLI overrides.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from physics_features import DEFAULT_G_MODE

# Base paths
BASE_DIR = Path(__file__).resolve().parent
ROI_STATS_PATH = BASE_DIR.parent / "pic_RGB" / "roi_results" / "roi_detailed_statistics.csv"
OUTPUT_DIR = BASE_DIR / "rgb_fit_output"

# Filenames
MODEL_FILENAME = "concentration_model_v1.pkl"
PREDICTIONS_FILENAME = "ridge_predictions.csv"
COEFFICIENTS_FILENAME = "ridge_coefficients.csv"
PRED_SUFFIX = "_pred_concentration.csv"


@dataclass
class TrainConfig:
    """Defaults for model training."""

    roi_csv: Path = ROI_STATS_PATH
    output_dir: Path = OUTPUT_DIR
    alpha: float = 0.5
    g_mode: Literal["g1", "avg"] = DEFAULT_G_MODE

    @property
    def model_path(self) -> Path:
        return self.output_dir / MODEL_FILENAME

    @property
    def predictions_path(self) -> Path:
        return self.output_dir / PREDICTIONS_FILENAME

    @property
    def coefficients_path(self) -> Path:
        return self.output_dir / COEFFICIENTS_FILENAME


@dataclass
class PredictConfig:
    """Defaults for model inference."""

    model_path: Path = OUTPUT_DIR / MODEL_FILENAME
    default_roi_csv: Path = ROI_STATS_PATH
    output_suffix: str = PRED_SUFFIX

    def output_path_for(self, roi_csv: Path) -> Path:
        """Build default output path for a ROI CSV."""
        return roi_csv.with_name(roi_csv.stem + self.output_suffix)

