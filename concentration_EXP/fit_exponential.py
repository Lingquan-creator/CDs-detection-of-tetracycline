"""
Exponential Model Concentration Fitting and Inversion.

Fits intensity = a * exp(b * conc) + c for each RGB channel, then inverts to predict concentration.

Usage:
    python fit_exponential.py                    # Use default ROI CSV
    python fit_exponential.py --roi-csv <path>   # Custom ROI CSV
    python fit_exponential.py --channel G_corr   # Use averaged green channel
    python fit_exponential.py --weight-eps 10.0  # Adjust low-concentration emphasis
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import ExpConfig
from physics_features import extract_concentration_from_name


def model_exponential(x: np.ndarray, a: float, b: float, c0: float) -> np.ndarray:
    """
    Exponential model: y = a * exp(b * x) + c0
    
    With numerical stability clipping.
    """
    return a * np.exp(np.clip(b * x, -700, 50)) + c0


def _model_exponential_i0(x: np.ndarray, i0: float, b: float, c0: float) -> np.ndarray:
    """
    Re-parameterized exponential: y = (i0 - c0) * exp(b*x) + c0
    
    This parameterization enforces i0 (intensity at conc=0) >= max(observed intensity),
    preventing negative concentration predictions during inversion.
    """
    return (i0 - c0) * np.exp(np.clip(b * x, -700, 50)) + c0


def fit_exponential(
    conc: np.ndarray,
    intensity: np.ndarray,
    *,
    weight_eps: float = 0.0,
) -> Tuple[Tuple[float, float, float], float, float]:
    """
    Fit exponential model: intensity = a * exp(b * conc) + c
    
    Args:
        conc: Concentration values (x-axis)
        intensity: Measured intensity values (y-axis)
        weight_eps: Weight emphasis parameter. weights = 1/sqrt(conc + weight_eps)
                   Lower values emphasize low concentration more.
    
    Returns:
        ((a, b, c), r2, rmse): Fitted parameters, R² score, and RMSE
    
    Raises:
        RuntimeError: If scipy is not installed
        ValueError: If data is insufficient or invalid
    """
    try:
        from scipy.optimize import curve_fit  # type: ignore[import-not-found]
    except Exception as e:
        raise RuntimeError(
            "scipy is required for exponential fitting. Install it via `pip install scipy`."
        ) from e

    x = np.asarray(conc, dtype=float)
    y = np.asarray(intensity, dtype=float)

    if x.size < 3:
        raise ValueError("Need at least 3 points to fit exponential model.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite values found in data for exponential fit.")

    y_min = float(np.min(y))
    y_max = float(np.max(y))

    # Re-parameterization: i0 (intensity at conc=0) must be >= max observed intensity
    # This ensures the inverse function doesn't produce negative concentrations
    i0_min = y_max
    p0 = (y_max * 1.05, -0.01, max(0.0, y_min))
    bounds = ([i0_min, -np.inf, 0.0], [np.inf, 0.0, y_max])

    # Optional weighting to emphasize low-concentration fitting
    sigma: Optional[np.ndarray] = None
    if weight_eps is not None and float(weight_eps) >= 0.0:
        sigma = np.sqrt(x + float(weight_eps))

    # Perform curve fitting
    popt, _ = curve_fit(
        _model_exponential_i0,
        x,
        y,
        p0=p0,
        bounds=bounds,
        maxfev=20000,
        sigma=sigma,
        absolute_sigma=False,
    )

    i0, b, c0 = float(popt[0]), float(popt[1]), float(popt[2])
    a = i0 - c0

    # Calculate fit quality metrics
    y_pred = model_exponential(x, a, b, c0)
    residuals = y - y_pred
    rss = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - rss / ss_tot if ss_tot > 0 else 1.0
    rmse = float(math.sqrt(rss / len(y)))

    return (a, b, c0), float(r2), rmse


def invert_exponential_intensity(
    intensity: np.ndarray,
    *,
    a: float,
    b: float,
    c0: float,
) -> np.ndarray:
    """
    Inverse of I = a*exp(b*C) + c0 to get C = ln((I-c0)/a) / b
    
    Returns NaN for invalid parameters.
    Clips ratio to avoid log of non-positive values.
    """
    i = np.asarray(intensity, dtype=float)
    
    if a <= 0.0 or b == 0.0:
        return np.full_like(i, np.nan, dtype=float)

    ratio = (i - c0) / a
    ratio = np.clip(ratio, 1e-12, 1.0)
    return np.log(ratio) / b


def build_corrected_channels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build background-corrected mean intensities per channel.
    
    Computes: Signal = max(ROI - Background, 0)
    
    Required columns:
        - roi_R_mean, roi_G1_mean, roi_B_mean
        - bg_R_mean, bg_G1_mean, bg_B_mean
    
    Optional:
        - roi_G2_mean, bg_G2_mean (for dual-green sensors)
    
    Returns:
        DataFrame with columns: R_corr, G1_corr, G2_corr, G_corr, B_corr
    """
    required = {
        "roi_R_mean",
        "roi_G1_mean",
        "roi_B_mean",
        "bg_R_mean",
        "bg_G1_mean",
        "bg_B_mean",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"ROI CSV missing required columns for exponential fit: {missing}"
        )

    r = (df["roi_R_mean"] - df["bg_R_mean"]).clip(lower=0.0)
    g1 = (df["roi_G1_mean"] - df["bg_G1_mean"]).clip(lower=0.0)
    b = (df["roi_B_mean"] - df["bg_B_mean"]).clip(lower=0.0)

    has_g2 = "roi_G2_mean" in df.columns and "bg_G2_mean" in df.columns
    if has_g2:
        g2 = (df["roi_G2_mean"] - df["bg_G2_mean"]).clip(lower=0.0)
        g = ((g1 + g2) / 2.0).clip(lower=0.0)
    else:
        g2 = pd.Series([np.nan] * len(df), index=df.index, dtype=float)
        g = g1

    return pd.DataFrame(
        {
            "R_corr": r.astype(float),
            "G1_corr": g1.astype(float),
            "G2_corr": g2.astype(float),
            "G_corr": g.astype(float),
            "B_corr": b.astype(float),
        }
    )


def parse_args(default_cfg: ExpConfig) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fit concentration from ROI statistics using exponential model."
    )
    
    parser.add_argument(
        "--roi-csv",
        type=Path,
        default=default_cfg.roi_csv,
        help=f"Path to ROI statistics CSV (default: {default_cfg.roi_csv})",
    )
    
    parser.add_argument(
        "--channel",
        choices=["R_corr", "G1_corr", "G2_corr", "G_corr", "B_corr"],
        default=default_cfg.channel,
        help=f"Which corrected channel to use for concentration inversion (default: {default_cfg.channel})",
    )
    
    parser.add_argument(
        "--weight-eps",
        type=float,
        default=default_cfg.weight_eps,
        help=f"Weighting parameter for low-concentration emphasis (default: {default_cfg.weight_eps})",
    )
    
    parser.add_argument(
        "--params-out",
        type=Path,
        default=default_cfg.params_path,
        help="Output CSV for fitted parameters per channel",
    )
    
    parser.add_argument(
        "--pred-out",
        type=Path,
        default=default_cfg.predictions_path,
        help="Output CSV for fitted intensity predictions",
    )
    
    parser.add_argument(
        "--conc-out",
        type=Path,
        help="Output CSV for concentration predictions (default: <roi-csv>_pred_concentration_exp.csv)",
    )
    
    return parser.parse_args()


def run_exponential_fitting(args: argparse.Namespace, cfg: ExpConfig) -> None:
    """Main fitting workflow."""
    
    # Load and validate data
    df_raw = pd.read_csv(args.roi_csv)
    if df_raw.empty:
        raise ValueError("ROI CSV is empty.")

    df = df_raw.copy()
    df["conc_true"] = df["filename"].apply(extract_concentration_from_name)
    df = df.dropna(subset=["conc_true"]).sort_values("conc_true").reset_index(drop=True)
    
    if df.empty:
        raise ValueError(
            "No valid samples found after parsing concentration numbers from filenames."
        )

    conc = df["conc_true"].to_numpy(dtype=float)
    corr = build_corrected_channels(df)

    # Fit all channels
    channels = ["R_corr", "G1_corr", "G2_corr", "G_corr", "B_corr"]
    results: List[Dict[str, object]] = []
    params_by_channel: Dict[str, Tuple[float, float, float]] = {}

    print(f"\nFitting exponential model to {len(df)} samples...")
    print(f"Concentration range: {conc.min():.2f} - {conc.max():.2f} µM")
    print(f"Weight epsilon: {args.weight_eps}\n")

    for ch in channels:
        y = corr[ch].to_numpy(dtype=float)
        if np.all(np.isnan(y)):
            print(f"[SKIP] {ch}: all NaN values")
            continue

        try:
            (a, b, c0), r2, rmse = fit_exponential(
                conc, y, weight_eps=float(args.weight_eps)
            )
            params_by_channel[ch] = (a, b, c0)
            results.append({
                "channel": ch,
                "a": a,
                "b": b,
                "c": c0,
                "r2": r2,
                "rmse": rmse,
            })
            print(
                f"[FIT] {ch:8s}: a={a:9.3f}, b={b:9.6f}, c={c0:9.3f}  "
                f"R²={r2:.4f}  RMSE={rmse:.3f}"
            )
        except Exception as e:
            print(f"[FAIL] {ch}: {e}")
            continue

    if not results:
        raise RuntimeError("Exponential fit failed for all channels.")

    # Ensure output directories exist
    args.params_out.parent.mkdir(parents=True, exist_ok=True)
    args.pred_out.parent.mkdir(parents=True, exist_ok=True)

    # Save fit parameters
    pd.DataFrame(results).to_csv(args.params_out, index=False)
    print(f"\n✓ Saved fit parameters → {args.params_out}")

    # Save fitted intensity predictions per channel
    pred_df = pd.DataFrame({
        "filename": df["filename"],
        "concentration_uM": conc,
    })

    for ch in channels:
        pred_df[ch] = corr[ch]
        if ch in params_by_channel:
            a, b, c0 = params_by_channel[ch]
            pred_df[f"{ch}_pred"] = model_exponential(conc, a, b, c0)
        else:
            pred_df[f"{ch}_pred"] = np.nan

    pred_df.to_csv(args.pred_out, index=False)
    print(f"✓ Saved intensity predictions → {args.pred_out}")

    # Invert selected channel to get concentration predictions
    if args.channel not in params_by_channel:
        raise ValueError(
            f"Selected channel '{args.channel}' was not fitted (missing or invalid data)."
        )

    a, b, c0 = params_by_channel[args.channel]
    conc_pred = invert_exponential_intensity(
        corr[args.channel].to_numpy(dtype=float),
        a=a,
        b=b,
        c0=c0,
    )
    conc_pred = np.maximum(np.asarray(conc_pred, dtype=float), 0.0)

    # Save concentration predictions
    conc_out = args.conc_out or cfg.concentration_path_for(args.roi_csv)
    conc_out.parent.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame({
        "filename": df["filename"],
        "conc_pred": conc_pred,
        "conc_true": conc,
        "residual": conc_pred - conc,
    })
    out_df.to_csv(conc_out, index=False)

    # Calculate and display final metrics
    valid_mask = np.isfinite(conc_pred)
    if valid_mask.sum() > 0:
        r2_final = 1 - np.sum((conc[valid_mask] - conc_pred[valid_mask])**2) / np.sum((conc[valid_mask] - conc[valid_mask].mean())**2)
        rmse_final = np.sqrt(np.mean((conc[valid_mask] - conc_pred[valid_mask])**2))
        print(f"\n✓ Saved concentration predictions → {conc_out}")
        print(f"\nFinal Evaluation (channel={args.channel}):")
        print(f"  R² = {r2_final:.4f}")
        print(f"  RMSE = {rmse_final:.3f} µM")
    else:
        print(f"\n✓ Saved concentration predictions → {conc_out}")
        print("\nWARNING: All predictions are invalid (NaN)")


def main() -> None:
    """Entry point."""
    cfg = ExpConfig()
    args = parse_args(cfg)
    run_exponential_fitting(args, cfg)


if __name__ == "__main__":
    main()
