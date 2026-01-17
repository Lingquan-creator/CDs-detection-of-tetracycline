"""
Shared utilities for physics-informed ridge regression on ROI statistics.

The feature set follows the design in improve.md:
- ln_R, ln_G, ln_B: linearize the exponential decay.
- Ratio_RG, Ratio_BG: cancel absolute brightness drift.
- Var_G: approximate shot-noise level.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

FEATURE_NAMES: Tuple[str, ...] = ("ln_R", "ln_G", "ln_B", "Ratio_RG", "Ratio_BG", "Var_G")
DEFAULT_G_MODE = "g1"  # use G1 only. Alternatives: "avg" (G1/G2 average).
_EPS = 1e-6


def extract_concentration_from_name(name: str) -> Optional[float]:
    """Return numeric concentration parsed from filename stem; None if parsing fails."""
    stem = Path(name).stem
    try:
        return float(stem)
    except ValueError:
        return None


def _select_green_mean(df: pd.DataFrame, g_mode: str) -> pd.Series:
    g1 = df["roi_G1_mean"]
    if g_mode == "avg" and "roi_G2_mean" in df.columns:
        g2 = df["roi_G2_mean"]
        return (g1 + g2) / 2.0
    return g1


def _select_green_var(df: pd.DataFrame, g_mode: str) -> pd.Series:
    g1_var = df["roi_G1_var"]
    if g_mode == "avg" and "roi_G2_var" in df.columns:
        g2_var = df["roi_G2_var"]
        return (g1_var + g2_var) / 2.0
    return g1_var


def build_feature_frame(df: pd.DataFrame, g_mode: str = DEFAULT_G_MODE) -> pd.DataFrame:
    """Compute physics-informed features from a ROI dataframe."""
    missing = [col for col in ["roi_R_mean", "roi_G1_mean", "roi_B_mean", "roi_G1_var"] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for feature build: {missing}")

    g_mean = _select_green_mean(df, g_mode)
    g_var = _select_green_var(df, g_mode)
    r_mean = df["roi_R_mean"]
    b_mean = df["roi_B_mean"]

    features = pd.DataFrame(
        {
            "ln_R": np.log(r_mean.to_numpy() + _EPS),
            "ln_G": np.log(g_mean.to_numpy() + _EPS),
            "ln_B": np.log(b_mean.to_numpy() + _EPS),
            "Ratio_RG": r_mean.to_numpy() / (g_mean.to_numpy() + _EPS),
            "Ratio_BG": b_mean.to_numpy() / (g_mean.to_numpy() + _EPS),
            "Var_G": g_var.to_numpy(),
        }
    )
    return features


def build_feature_vector(
    r: float,
    g1: float,
    b: float,
    *,
    var_g1: float,
    g2: Optional[float] = None,
    var_g2: Optional[float] = None,
    g_mode: str = DEFAULT_G_MODE,
) -> np.ndarray:
    """Create a single feature row for inference."""
    if g_mode == "avg" and g2 is not None:
        g_mean = (g1 + g2) / 2.0
        if var_g2 is not None:
            var_g = (var_g1 + var_g2) / 2.0
        else:
            var_g = var_g1
    else:
        g_mean = g1
        var_g = var_g1

    ln_r = np.log(r + _EPS)
    ln_g = np.log(g_mean + _EPS)
    ln_b = np.log(b + _EPS)
    ratio_rg = r / (g_mean + _EPS)
    ratio_bg = b / (g_mean + _EPS)

    return np.array([[ln_r, ln_g, ln_b, ratio_rg, ratio_bg, var_g]], dtype=float)
