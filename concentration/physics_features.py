"""
Shared utilities for physics-informed feature extraction from ROI statistics.

This module provides unified feature engineering tools for all concentration
prediction methods (Ridge, Exponential, Gaussian amortized posterior).

Supported feature sets:
- Raw features (no background): ln_R/ln_G/ln_B, Ratio_RG/Delta_RG/Ratio_BG, Var_G
- Background-subtracted features: ln_R_sig/ln_G_sig/ln_B_sig using signal=ROI-BG, plus ratios/Var_G

Based on Beer-Lambert law: I ∝ exp(-εlC), so ln(I) should be linear with concentration.

Note: This module is duplicated in concentration_EXP and concentration_GS for independence.
All three versions are kept synchronized.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

FEATURE_NAMES_RAW: Tuple[str, ...] = ("ln_R", "ln_G", "ln_B", "Ratio_RG", "Delta_RG", "Ratio_BG", "Var_G")
FEATURE_NAMES_BG: Tuple[str, ...] = ("ln_R_sig", "ln_G_sig", "ln_B_sig", "Ratio_RG", "Delta_RG", "Ratio_BG", "Var_G")
FEATURE_NAMES: Tuple[str, ...] = FEATURE_NAMES_RAW  # backward-compatibility alias
DEFAULT_G_MODE = "avg"  # default to averaged green response, per art.md.
_EPS = 1e-6
_CLIP_MIN = 0.1  # minimum signal after bg subtraction to avoid log of non-positive


def extract_concentration_from_name(name: str) -> Optional[float]:
    """Return numeric concentration parsed from filename stem; None if parsing fails."""
    stem = Path(name).stem
    try:
        return float(stem)
    except ValueError:
        return None


def feature_names(use_bg: bool) -> Tuple[str, ...]:
    """Return the active feature names based on background usage."""
    return FEATURE_NAMES_BG if use_bg else FEATURE_NAMES_RAW


def has_background_columns(df: pd.DataFrame, g_mode: str = DEFAULT_G_MODE) -> bool:
    required = {"roi_R_mean", "bg_R_mean", "roi_B_mean", "bg_B_mean", "roi_G1_mean", "bg_G1_mean", "roi_G1_var"}
    if g_mode == "avg":
        required |= {"roi_G2_mean", "bg_G2_mean"}
    return required.issubset(set(df.columns))


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


def _signal(roi: pd.Series, bg: pd.Series, clip_min: float) -> np.ndarray:
    sig = roi.to_numpy() - bg.to_numpy()
    return np.maximum(sig, clip_min)


def build_feature_frame(df: pd.DataFrame, g_mode: str = DEFAULT_G_MODE, *, use_bg: bool = False, clip_min: float = _CLIP_MIN) -> pd.DataFrame:
    """Compute physics-informed features from a ROI dataframe (with optional background subtraction)."""
    base_missing = [col for col in ["roi_R_mean", "roi_G1_mean", "roi_B_mean", "roi_G1_var"] if col not in df.columns]
    if base_missing:
        raise ValueError(f"Missing required columns for feature build: {base_missing}")

    if use_bg:
        bg_required = ["bg_R_mean", "bg_B_mean", "bg_G1_mean"]
        if g_mode == "avg":
            bg_required += ["bg_G2_mean"]
        bg_missing = [col for col in bg_required if col not in df.columns]
        if bg_missing:
            raise ValueError(f"Background mode requested but CSV missing columns: {bg_missing}")

        r_sig = _signal(df["roi_R_mean"], df["bg_R_mean"], clip_min)
        b_sig = _signal(df["roi_B_mean"], df["bg_B_mean"], clip_min)
        if g_mode == "avg" and "roi_G2_mean" in df.columns:
            g_sig = ( _signal(df["roi_G1_mean"], df["bg_G1_mean"], clip_min) + _signal(df["roi_G2_mean"], df["bg_G2_mean"], clip_min) ) / 2.0
        else:
            g_sig = _signal(df["roi_G1_mean"], df["bg_G1_mean"], clip_min)
    else:
        r_sig = df["roi_R_mean"].to_numpy()
        b_sig = df["roi_B_mean"].to_numpy()
        g_sig = _select_green_mean(df, g_mode).to_numpy()

    g_var = _select_green_var(df, g_mode).to_numpy()

    features = pd.DataFrame(
        {
            "ln_R_sig" if use_bg else "ln_R": np.log(r_sig + _EPS),
            "ln_G_sig" if use_bg else "ln_G": np.log(g_sig + _EPS),
            "ln_B_sig" if use_bg else "ln_B": np.log(b_sig + _EPS),
            "Ratio_RG": r_sig / (g_sig + _EPS),
            "Delta_RG": (g_sig - r_sig) / (g_sig + r_sig + _EPS),
            "Ratio_BG": b_sig / (g_sig + _EPS),
            "Var_G": g_var,
        }
    )
    return features


def build_feature_vector(
    r: float,
    g1: float,
    b: float,
    *,
    var_g1: float,
    bg_r: float = 0.0,
    bg_g1: float = 0.0,
    bg_b: float = 0.0,
    g2: Optional[float] = None,
    bg_g2: float = 0.0,
    var_g2: Optional[float] = None,
    g_mode: str = DEFAULT_G_MODE,
    use_bg: bool = False,
    clip_min: float = _CLIP_MIN,
) -> np.ndarray:
    """Create a single feature row for inference (supports optional background subtraction)."""
    if use_bg:
        r_sig = max(r - bg_r, clip_min)
        b_sig = max(b - bg_b, clip_min)
        g1_sig = max(g1 - bg_g1, clip_min)
        if g_mode == "avg" and g2 is not None:
            g2_sig = max(g2 - bg_g2, clip_min)
            g_mean = (g1_sig + g2_sig) / 2.0
            var_g = (var_g1 + (var_g2 or var_g1)) / 2.0
        else:
            g_mean = g1_sig
            var_g = var_g1
    else:
        r_sig = r
        b_sig = b
        if g_mode == "avg" and g2 is not None:
            g_mean = (g1 + g2) / 2.0
            var_g = (var_g1 + (var_g2 or var_g1)) / 2.0
        else:
            g_mean = g1
            var_g = var_g1

    ln_r = np.log(r_sig + _EPS)
    ln_g = np.log(g_mean + _EPS)
    ln_b = np.log(b_sig + _EPS)
    ratio_rg = r_sig / (g_mean + _EPS)
    delta_rg = (g_mean - r_sig) / (g_mean + r_sig + _EPS)
    ratio_bg = b_sig / (g_mean + _EPS)

    return np.array([[ln_r, ln_g, ln_b, ratio_rg, delta_rg, ratio_bg, var_g]], dtype=float)
