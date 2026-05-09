"""
Configuration for the concentration_GS amortized-inference pipeline.

The model learns a conditional posterior over log-concentration:
    q_phi(log c | x) = Normal(mu_phi(x), sigma_phi(x)^2)

Training and inference share the same paths and hyperparameters here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


BASE_DIR = Path(__file__).resolve().parent
ROI_STATS_PATH = BASE_DIR.parent / "dataset" / "food" / "roi_detailed_statistics.csv"
OUTPUT_DIR = BASE_DIR / "amortized_output"

MODEL_FILENAME = "amortized_posterior_model.pt"
TRAIN_SUMMARY_FILENAME = "train_posterior_summary.csv"
LOO_SUMMARY_FILENAME = "loo_posterior_summary.csv"
TRAIN_GRID_FILENAME = "train_posterior_probabilities.csv"
LOO_GRID_FILENAME = "loo_posterior_probabilities.csv"
PLOT_FILENAME = "posterior_evaluation.png"

PREDICT_SUMMARY_SUFFIX = "_pred_concentration_gs.csv"
PREDICT_GRID_SUFFIX = "_pred_concentration_gs_prob.csv"
PREDICT_PLOT_SUFFIX = "_pred_concentration_gs.png"

DEFAULT_G_MODE: Literal["g1", "avg"] = "avg"
DEFAULT_USE_BG = "true"
OPTIMIZED_FEATURE_SET = "ratio_rg_ratio_rb_delta_rg_delta_rb_delta_gb"
DEFAULT_FEATURE_SET: Literal["full", "compact", "compact_exposure", "ratio_rg_ratio_rb_delta_rg_delta_rb_delta_gb"] = OPTIMIZED_FEATURE_SET
TARGET_EPS = 1e-6
DEFAULT_CONFIDENCE = 0.90

# Optimization defaults for the amortized posterior network.
HIDDEN_DIM = 8
DROPOUT = 0.05
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 1e-3
EPOCHS = 2500
PATIENCE = 250
MIN_STD = 0.08          # posterior std floor in standardized log-space
STD_REG_LAMBDA = 0.01   # penalty on predicted posterior std to avoid variance blow-up
MAX_STD_LOG = 2.0       # posterior std ceiling after converting back to log-concentration space
WEIGHT_EPS = 30.0       # sample weight = 1 / (WEIGHT_BETA * concentration + WEIGHT_EPS)
WEIGHT_BETA = 1.0
GRAD_CLIP_NORM = 5.0
SEED = 42


@dataclass
class TrainConfig:
    roi_csv: Path = ROI_STATS_PATH
    output_dir: Path = OUTPUT_DIR
    g_mode: Literal["g1", "avg"] = DEFAULT_G_MODE
    use_bg: str = DEFAULT_USE_BG
    feature_set: Literal["full", "compact", "compact_exposure", "ratio_rg_ratio_rb_delta_rg_delta_rb_delta_gb"] = DEFAULT_FEATURE_SET
    weight_beta: float = WEIGHT_BETA
    hidden_dim: int = HIDDEN_DIM
    dropout: float = DROPOUT
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    epochs: int = EPOCHS
    patience: int = PATIENCE
    min_std: float = MIN_STD
    std_reg_lambda: float = STD_REG_LAMBDA
    max_std_log: float = MAX_STD_LOG
    weight_eps: float = WEIGHT_EPS
    grad_clip_norm: float = GRAD_CLIP_NORM
    seed: int = SEED
    confidence: float = DEFAULT_CONFIDENCE

    @property
    def model_path(self) -> Path:
        return self.output_dir / MODEL_FILENAME

    @property
    def train_summary_path(self) -> Path:
        return self.output_dir / TRAIN_SUMMARY_FILENAME

    @property
    def loo_summary_path(self) -> Path:
        return self.output_dir / LOO_SUMMARY_FILENAME

    @property
    def train_grid_path(self) -> Path:
        return self.output_dir / TRAIN_GRID_FILENAME

    @property
    def loo_grid_path(self) -> Path:
        return self.output_dir / LOO_GRID_FILENAME

    @property
    def plot_path(self) -> Path:
        return self.output_dir / PLOT_FILENAME


@dataclass
class PredictConfig:
    model_path: Path = OUTPUT_DIR / MODEL_FILENAME
    default_roi_csv: Path = ROI_STATS_PATH
    output_dir: Path = OUTPUT_DIR
    confidence: float = DEFAULT_CONFIDENCE

    def summary_output_path_for(self, roi_csv: Path) -> Path:
        return self.output_dir / f"{roi_csv.stem}{PREDICT_SUMMARY_SUFFIX}"

    def probability_output_path_for(self, roi_csv: Path) -> Path:
        return self.output_dir / f"{roi_csv.stem}{PREDICT_GRID_SUFFIX}"

    def plot_output_path_for(self, roi_csv: Path) -> Path:
        return self.output_dir / f"{roi_csv.stem}{PREDICT_PLOT_SUFFIX}"
