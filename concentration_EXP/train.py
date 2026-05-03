"""
Exponential Model Concentration Fitting and Inversion.

Fits intensity = a * exp(b * conc) + c for each RGB channel, then inverts to predict concentration.

Usage:
    python train.py                    # Use default ROI CSV
    python train.py --roi-csv <path>   # Custom ROI CSV
    python train.py --channel G_corr   # Use averaged green channel
    python train.py --weight-eps 10.0  # Adjust low-concentration emphasis
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd

from config import ExpConfig
from physics_features import build_feature_frame, extract_concentration_from_name

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

_EXPOSURE_MIN = 1e-6
CSV_FLOAT_FORMAT = "%.6g"


def format_plot_number(value: float, _pos: object = None) -> str:
    if not np.isfinite(value):
        return ""
    text = f"{value:.6g}"
    return "0" if text in ("", "-0") else text


def format_figure_numbers(fig: plt.Figure) -> None:
    formatter = FuncFormatter(format_plot_number)
    for ax in fig.axes:
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)


plt.rcParams.update({"axes.unicode_minus": False})


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
        ((a, b, c), r2, rmse): Fitted parameters, R2 score, and RMSE
    
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
    
    Computes:
    - Signal = max(ROI - Background, 0)
    - Signal_per_exposure = Signal / exposure_time
    
    Required columns:
        - roi_R_mean, roi_G1_mean, roi_B_mean
        - bg_R_mean, bg_G1_mean, bg_B_mean
    
    Optional:
        - roi_G2_mean, bg_G2_mean (for dual-green sensors)
    
    Returns:
        DataFrame with columns:
        - corrected intensities: R_corr, G1_corr, G2_corr, G_corr, B_corr
        - exposure metadata: exposure_time
        - exposure-normalized intensities: <channel>_per_exp
    """
    from physics_features import resolve_exposure_time_series

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

    corr = pd.DataFrame(
        {
            "R_corr": r.astype(float),
            "G1_corr": g1.astype(float),
            "G2_corr": g2.astype(float),
            "G_corr": g.astype(float),
            "B_corr": b.astype(float),
        }
    )
    exposure = resolve_exposure_time_series(df)
    corr["exposure_time"] = exposure
    for ch in ("R_corr", "G1_corr", "G2_corr", "G_corr", "B_corr"):
        corr[f"{ch}_per_exp"] = corr[ch].to_numpy(dtype=float) / np.maximum(exposure, _EXPOSURE_MIN)
    try:
        physics_features = build_feature_frame(df, use_bg=True)
    except ValueError:
        physics_features = pd.DataFrame(index=df.index)

    for ch in [
        col
        for col in physics_features.columns
        if col.startswith(("exp_roi_", "log_roi_", "sqrt_roi_"))
    ]:
        corr[ch] = physics_features[ch].to_numpy(dtype=float)
        corr[f"{ch}_per_exp"] = corr[ch].to_numpy(dtype=float)
    return corr


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
        default=default_cfg.channel,
        help=f"Which corrected channel to use for concentration inversion (fit uses exposure-normalized {default_cfg.channel})",
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
    parser.add_argument(
        "--plot-out",
        type=Path,
        help="Output PNG for fitting/prediction visualization (default: exp_fit_output/exp_fit_evaluation.png)",
    )
    
    return parser.parse_args()


def resolve_output_path(path: Optional[Path], default_path: Path, output_dir: Path) -> Path:
    chosen = (path or default_path).expanduser()
    if not chosen.is_absolute():
        chosen = output_dir / chosen
    return chosen.resolve()


def run_exponential_fitting(args: argparse.Namespace, cfg: ExpConfig) -> None:
    """Main fitting workflow."""
    params_out = resolve_output_path(args.params_out, cfg.params_path, cfg.output_dir)
    pred_out = resolve_output_path(args.pred_out, cfg.predictions_path, cfg.output_dir)
    
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

    # Fit all corrected intensity channels plus optional transformed ROI distribution features.
    channels = [
        col
        for col in corr.columns
        if col != "exposure_time" and not col.endswith("_per_exp")
    ]
    results: List[Dict[str, object]] = []
    params_by_channel: Dict[str, Tuple[float, float, float]] = {}

    print(f"\nFitting exponential model to {len(df)} samples...")
    print(f"Concentration range: {conc.min():.6g} - {conc.max():.6g} uM")
    print(f"Weight epsilon: {float(args.weight_eps):.6g}\n")

    for ch in channels:
        y = corr[f"{ch}_per_exp"].to_numpy(dtype=float)
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
                "fit_signal": f"{ch}_per_exp",
                "a": a,
                "b": b,
                "c": c0,
                "r2": r2,
                "rmse": rmse,
            })
            print(
                f"[FIT] {ch:8s} (per_exp): a={a:9.6g}, b={b:9.6g}, c={c0:9.6g}  "
                f"R2={r2:.6g}  RMSE={rmse:.6g}"
            )
        except Exception as e:
            print(f"[FAIL] {ch}: {e}")
            continue

    if not results:
        raise RuntimeError("Exponential fit failed for all channels.")

    # Ensure output directories exist
    params_out.parent.mkdir(parents=True, exist_ok=True)
    pred_out.parent.mkdir(parents=True, exist_ok=True)

    # Save fit parameters
    pd.DataFrame(results).to_csv(params_out, index=False, float_format=CSV_FLOAT_FORMAT)
    print(f"\nSaved fit parameters -> {params_out}")

    # Save fitted intensity predictions per channel
    pred_df = pd.DataFrame({
        "filename": df["filename"],
        "concentration_uM": conc,
        "exposure_time": corr["exposure_time"],
    })

    for ch in channels:
        pred_df[ch] = corr[ch]
        pred_df[f"{ch}_per_exp"] = corr[f"{ch}_per_exp"]
        if ch in params_by_channel:
            a, b, c0 = params_by_channel[ch]
            pred_df[f"{ch}_per_exp_pred"] = model_exponential(conc, a, b, c0)
        else:
            pred_df[f"{ch}_per_exp_pred"] = np.nan

    pred_df.to_csv(pred_out, index=False, float_format=CSV_FLOAT_FORMAT)
    print(f"Saved intensity predictions -> {pred_out}")

    # Invert selected channel to get concentration predictions
    if args.channel not in params_by_channel:
        raise ValueError(
            f"Selected channel '{args.channel}' was not fitted (missing or invalid data)."
        )

    a, b, c0 = params_by_channel[args.channel]
    conc_pred = invert_exponential_intensity(
        corr[f"{args.channel}_per_exp"].to_numpy(dtype=float),
        a=a,
        b=b,
        c0=c0,
    )
    conc_pred = np.maximum(np.asarray(conc_pred, dtype=float), 0.0)

    # Save concentration predictions
    conc_out = resolve_output_path(args.conc_out, cfg.concentration_path_for(args.roi_csv), cfg.output_dir)
    conc_out.parent.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame({
        "filename": df["filename"],
        "exposure_time": corr["exposure_time"],
        "conc_pred": conc_pred,
        "conc_true": conc,
        f"{args.channel}_intensity": corr[args.channel],
        f"{args.channel}_per_exp": corr[f"{args.channel}_per_exp"],
        "residual": conc_pred - conc,
    })
    out_df.to_csv(conc_out, index=False, float_format=CSV_FLOAT_FORMAT)

    # Calculate and display final metrics
    valid_mask = np.isfinite(conc_pred)
    if valid_mask.sum() > 0:
        r2_final = 1 - np.sum((conc[valid_mask] - conc_pred[valid_mask])**2) / np.sum((conc[valid_mask] - conc[valid_mask].mean())**2)
        rmse_final = np.sqrt(np.mean((conc[valid_mask] - conc_pred[valid_mask])**2))
        print(f"\nSaved concentration predictions -> {conc_out}")
        print(f"\nFinal Evaluation (channel={args.channel}, signal={args.channel}_per_exp):")
        print(f"  R2 = {r2_final:.6g}")
        print(f"  RMSE = {rmse_final:.6g} uM")
    else:
        print(f"\nSaved concentration predictions -> {conc_out}")
        print("\nWARNING: All predictions are invalid (NaN)")

    plot_out = resolve_output_path(args.plot_out, cfg.fit_plot_path, cfg.output_dir)
    draw_fit_visualization(
        conc=conc,
        corrected_channels=corr,
        params_by_channel=params_by_channel,
        selected_channel=args.channel,
        conc_pred=conc_pred,
        out_path=plot_out,
        channels=channels,
    )


def draw_fit_visualization(
    *,
    conc: np.ndarray,
    corrected_channels: pd.DataFrame,
    params_by_channel: Dict[str, Tuple[float, float, float]],
    selected_channel: str,
    conc_pred: np.ndarray,
    out_path: Path,
    channels: List[str],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_axes = len(channels) + 1
    ncols = 3
    nrows = int(math.ceil(total_axes / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.5 * nrows), dpi=150)
    axes = axes.ravel()
    x_curve = np.linspace(float(np.min(conc)), float(np.max(conc)), 200)

    for idx, ch in enumerate(channels):
        ax = axes[idx]
        y = corrected_channels[f"{ch}_per_exp"].to_numpy(dtype=float)
        ax.scatter(conc, y, s=20, alpha=0.7, label="Measured")
        if ch in params_by_channel:
            a, b, c0 = params_by_channel[ch]
            y_curve = model_exponential(x_curve, a, b, c0)
            ax.plot(x_curve, y_curve, "r-", linewidth=1.5, label="Fit")
            ax.set_title(f"{ch}: fit (per_exp)")
        else:
            ax.set_title(f"{ch}: skipped")
        ax.set_xlabel("Concentration (uM)")
        ax.set_ylabel("Corrected intensity / exposure")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    ax_eval = axes[-1]
    valid = np.isfinite(conc_pred)
    if np.any(valid):
        x = conc[valid]
        y = conc_pred[valid]
        ax_eval.scatter(x, y, s=22, alpha=0.8, color="#1f77b4")
        low = float(min(np.min(x), np.min(y)))
        high = float(max(np.max(x), np.max(y)))
        ax_eval.plot([low, high], [low, high], "k--", linewidth=1.0)
        ax_eval.set_title(f"Concentration inversion ({selected_channel})")
        ax_eval.set_xlabel("True concentration (uM)")
        ax_eval.set_ylabel("Pred concentration (uM)")
        ax_eval.grid(alpha=0.25)
    else:
        ax_eval.text(0.5, 0.5, "No valid prediction", ha="center", va="center")
        ax_eval.set_title(f"Concentration inversion ({selected_channel})")

    for ax in axes[total_axes:]:
        ax.axis("off")

    format_figure_numbers(fig)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved fit plot -> {out_path}")


def main() -> None:
    """Entry point."""
    cfg = ExpConfig()
    args = parse_args(cfg)
    run_exponential_fitting(args, cfg)


if __name__ == "__main__":
    main()
