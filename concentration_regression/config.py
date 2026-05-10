"""
Central configuration for the concentration pipeline (physics-informed Ridge regression).

- Paths: keep training/inference inputs and outputs in one place.
- Defaults: share alpha / g-mode across scripts while allowing CLI overrides.
- Target transform: log-target regression to emphasize low concentration accuracy and enforce non-negative outputs (used when no background subtraction).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

from physics_features import DEFAULT_G_MODE

# Training/inference defaults
TARGET_TRANSFORM = "log"  # log-target regression (see improve2.md)
TARGET_EPS = 1e-6         # small constant to avoid log(0) and for inverse transform
DEFAULT_USE_BG = True     # detailed ROI statistics include bg_* columns; use them by default
DEFAULT_BG_CORRECTION: Literal["subtract", "divide", "relative"] = "subtract"
DEFAULT_FEATURE_SET: Literal[
    "full",
    "compact",
    "compact_exposure",
    "ln_rgb",
] = "compact"
# Expanded alpha grid to include large regularization for smoothing low-concentration noise
# REMOVED small alphas to force smoother models and avoid overfitting noise.
# FORCED high alphas (starting at 50.0) to combat multicollinearity-induced oscillation.
ALPHAS: Tuple[float, ...] = (100.0,)
# Increased epsilon to reduce sensitivity to noise in low concentrations (e.g. <10uM)
# Higher eps (20 -> 50) means weights are flatter, preventing low-conc noise from dominating.
WEIGHT_EPS = 50.0         # sample weight denominator: 1/(conc + WEIGHT_EPS)
WEIGHT_BETA = 1.0         # sample weight denominator: 1/(WEIGHT_BETA * conc + WEIGHT_EPS)
CLIP_MIN = 0.1            # minimum signal after background subtraction to avoid log of negative/zero

# Base paths
BASE_DIR = Path(__file__).resolve().parent
ROI_STATS_PATH = BASE_DIR.parent / "dataset" / "food" / "roi_detailed_statistics.csv"
OUTPUT_DIR = BASE_DIR / "rgb_fit_output"

# Filenames
MODEL_FILENAME = "concentration_model_v1.pkl"
PREDICTIONS_FILENAME = "ridge_predictions.csv"
COEFFICIENTS_FILENAME = "ridge_coefficients.csv"
PRED_SUFFIX = "_pred_concentration.csv"
PRED_PLOT_SUFFIX = "_pred_concentration.png"


@dataclass
class TrainConfig:
    """Defaults for model training."""

    roi_csv: Path = ROI_STATS_PATH
    output_dir: Path = OUTPUT_DIR
    alpha: float = 0.5
    g_mode: Literal["g1", "avg"] = DEFAULT_G_MODE
    use_bg: bool = DEFAULT_USE_BG
    feature_set: Literal[
        "full",
        "compact",
        "compact_exposure",
        "ln_rgb",
    ] = DEFAULT_FEATURE_SET
    bg_correction: Literal["subtract", "divide", "relative"] = DEFAULT_BG_CORRECTION

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
    output_dir: Path = OUTPUT_DIR
    output_suffix: str = PRED_SUFFIX
    plot_suffix: str = PRED_PLOT_SUFFIX

    def output_path_for(self, roi_csv: Path) -> Path:
        """Build default output CSV path for a ROI CSV."""
        return self.output_dir / f"{roi_csv.stem}{self.output_suffix}"

    def plot_path_for(self, roi_csv: Path) -> Path:
        """Build default output plot path for a ROI CSV."""
        return self.output_dir / f"{roi_csv.stem}{self.plot_suffix}"
