"""
Shared utilities for physics-informed feature extraction from ROI statistics.

This module provides unified feature engineering tools for all concentration
prediction methods (Ridge, Exponential, Gaussian amortized posterior).

Supported feature sets:
- Raw features (no background): ln_R/ln_G/ln_B, Ratio_RG/Delta_RG/Ratio_BG, Var_G
- Background-subtracted features: ln_R_sig/ln_G_sig/ln_B_sig using signal=ROI-BG, plus ratios/Var_G

Based on Beer-Lambert law: I is proportional to exp(-epsilon*l*C), so ln(I) should be linear with concentration.

Note: This module is duplicated in concentration_EXP and concentration_GS for independence.
All three versions are kept synchronized.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

FEATURE_NAMES_RAW: Tuple[str, ...] = (
    "ln_R",
    "ln_G",
    "ln_B",
    "ln_R_per_exp",
    "ln_G_per_exp",
    "ln_B_per_exp",
    "Ratio_RG",
    "Delta_RG",
    "Ratio_BG",
    "Var_G",
    "exposure_time",
    "ln_exposure",
)
FEATURE_NAMES_BG: Tuple[str, ...] = (
    "ln_R_sig",
    "ln_G_sig",
    "ln_B_sig",
    "ln_R_sig_per_exp",
    "ln_G_sig_per_exp",
    "ln_B_sig_per_exp",
    "Ratio_RG",
    "Delta_RG",
    "Ratio_BG",
    "Var_G",
    "exposure_time",
    "ln_exposure",
)
FEATURE_NAMES: Tuple[str, ...] = FEATURE_NAMES_RAW  # backward-compatibility alias
DEFAULT_G_MODE = "avg"  # default to averaged green response, per art.md.
_EPS = 1e-6
_CLIP_MIN = 0.1  # minimum signal after bg subtraction to avoid log of non-positive
_EXPOSURE_MIN = 1e-6
_DEFAULT_EXPOSURE_TIME = 1.0


def extract_sample_metadata_from_name(name: str) -> Dict[str, Optional[float]]:
    """Parse concentration/exposure metadata from `concentration_exposure.ext` filenames."""
    stem = Path(name).stem

    if "_" in stem:
        concentration_text, exposure_text = stem.rsplit("_", maxsplit=1)
        try:
            return {
                "concentration": float(concentration_text),
                "exposure_time": float(exposure_text),
            }
        except ValueError:
            pass

    if "." in stem:
        concentration_text, exposure_text = stem.rsplit(".", maxsplit=1)
        try:
            exposure_value = float(exposure_text)
            if exposure_value < 100:
                raise ValueError
            return {
                "concentration": float(concentration_text),
                "exposure_time": exposure_value,
            }
        except ValueError:
            pass

    try:
        return {"concentration": float(stem), "exposure_time": None}
    except ValueError:
        return {"concentration": None, "exposure_time": None}


def extract_concentration_from_name(name: str) -> Optional[float]:
    """Return numeric concentration parsed from filename stem; None if parsing fails."""
    return extract_sample_metadata_from_name(name)["concentration"]


def extract_exposure_time_from_name(name: str) -> Optional[float]:
    """Return exposure time parsed from filename stem; None if parsing fails."""
    return extract_sample_metadata_from_name(name)["exposure_time"]


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


def resolve_exposure_time_series(df: pd.DataFrame) -> np.ndarray:
    """Resolve exposure time, assuming a constant exposure when metadata is missing."""
    exposure = pd.Series(np.nan, index=df.index, dtype=float)

    if "exposure_time" in df.columns:
        exposure = pd.to_numeric(df["exposure_time"], errors="coerce").astype(float)

    if "filename" in df.columns:
        parsed = df["filename"].apply(extract_exposure_time_from_name)
        parsed_series = pd.to_numeric(parsed, errors="coerce").astype(float)
        exposure = exposure.fillna(parsed_series)

    exposure_values = exposure.to_numpy(dtype=float, copy=True)
    valid_mask = np.isfinite(exposure_values) & (exposure_values > 0.0)
    missing_mask = ~valid_mask
    if np.any(missing_mask):
        fallback = exposure_values[valid_mask][0] if np.any(valid_mask) else _DEFAULT_EXPOSURE_TIME
        exposure_values[missing_mask] = fallback

    return np.maximum(exposure_values, _EXPOSURE_MIN)


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
    exposure = resolve_exposure_time_series(df)
    ln_exposure = np.log(exposure + _EPS)
    r_sig_per_exp = r_sig / exposure
    g_sig_per_exp = g_sig / exposure
    b_sig_per_exp = b_sig / exposure

    features = pd.DataFrame(
        {
            "ln_R_sig" if use_bg else "ln_R": np.log(r_sig + _EPS),
            "ln_G_sig" if use_bg else "ln_G": np.log(g_sig + _EPS),
            "ln_B_sig" if use_bg else "ln_B": np.log(b_sig + _EPS),
            "ln_R_sig_per_exp" if use_bg else "ln_R_per_exp": np.log(r_sig_per_exp + _EPS),
            "ln_G_sig_per_exp" if use_bg else "ln_G_per_exp": np.log(g_sig_per_exp + _EPS),
            "ln_B_sig_per_exp" if use_bg else "ln_B_per_exp": np.log(b_sig_per_exp + _EPS),
            "Ratio_RG": r_sig / (g_sig + _EPS),
            "Delta_RG": (g_sig - r_sig) / (g_sig + r_sig + _EPS),
            "Ratio_BG": b_sig / (g_sig + _EPS),
            "Var_G": g_var,
            "exposure_time": exposure,
            "ln_exposure": ln_exposure,
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
    exposure_time: float = 1.0,
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
    exposure = max(float(exposure_time), _EXPOSURE_MIN)
    ln_r_per_exp = np.log(r_sig / exposure + _EPS)
    ln_g_per_exp = np.log(g_mean / exposure + _EPS)
    ln_b_per_exp = np.log(b_sig / exposure + _EPS)
    ratio_rg = r_sig / (g_mean + _EPS)
    delta_rg = (g_mean - r_sig) / (g_mean + r_sig + _EPS)
    ratio_bg = b_sig / (g_mean + _EPS)

    return np.array(
        [[ln_r, ln_g, ln_b, ln_r_per_exp, ln_g_per_exp, ln_b_per_exp, ratio_rg, delta_rg, ratio_bg, var_g, exposure, np.log(exposure + _EPS)]],
        dtype=float,
    )
