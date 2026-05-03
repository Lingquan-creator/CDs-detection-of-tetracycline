"""
Predict concentration from ROI statistics using saved exponential fit parameters.

This script loads previously fitted parameters and applies them to new ROI data.

Usage:
    python predict.py --params exp_fit_output/exp_fit_params.csv --roi-csv <path>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd

from config import ExpConfig
from train import AUTO_CHANNELS, BEST_CHANNEL_ALIASES, build_corrected_channels, predict_from_fitted_features
from physics_features import extract_concentration_from_name

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


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


def load_fit_parameters(params_csv: Path, channel: str) -> Tuple[float, float, float]:
    """
    Load fitted parameters (a, b, c) for a specific channel.
    
    Args:
        params_csv: Path to exp_fit_params.csv
        channel: Channel name (e.g., 'G1_corr')
    
    Returns:
        (a, b, c): Fitted parameters
    
    Raises:
        FileNotFoundError: If params file doesn't exist
        ValueError: If channel not found in params
    """
    if not params_csv.exists():
        raise FileNotFoundError(f"Parameters file not found: {params_csv}")
    
    df = pd.read_csv(params_csv)
    
    row = df[df["channel"] == channel]
    if row.empty:
        available = df["channel"].tolist()
        raise ValueError(
            f"Channel '{channel}' not found in parameters file. "
            f"Available channels: {available}"
        )
    
    row = row.iloc[0]
    return float(row["a"]), float(row["b"]), float(row["c"])


def load_all_fit_parameters(params_csv: Path) -> dict[str, Tuple[float, float, float]]:
    """Load all fitted feature parameters from exp_fit_params.csv."""
    if not params_csv.exists():
        raise FileNotFoundError(f"Parameters file not found: {params_csv}")

    df = pd.read_csv(params_csv)
    required = {"channel", "a", "b", "c"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Parameters CSV missing required columns: {missing}")

    params: dict[str, Tuple[float, float, float]] = {}
    for row in df.itertuples(index=False):
        channel = str(getattr(row, "channel"))
        params[channel] = (
            float(getattr(row, "a")),
            float(getattr(row, "b")),
            float(getattr(row, "c")),
        )
    return params


def resolve_selected_feature_from_params(params_csv: Path, selected_channel: str) -> str:
    if selected_channel not in BEST_CHANNEL_ALIASES:
        return selected_channel

    df = pd.read_csv(params_csv)
    score_col = "inv_r2" if "inv_r2" in df.columns else "r2"
    if score_col not in df.columns:
        raise ValueError("Cannot use light_best because parameter CSV has no r2/inv_r2 column.")
    ranked = df[np.isfinite(pd.to_numeric(df[score_col], errors="coerce"))].copy()
    if ranked.empty:
        raise ValueError(f"Cannot use light_best because no fitted light channel has finite {score_col}.")
    ranked[score_col] = pd.to_numeric(ranked[score_col], errors="coerce")
    return str(ranked.sort_values(score_col, ascending=False).iloc[0]["channel"])


def parse_args(default_cfg: ExpConfig) -> argparse.Namespace:
    """Parse command-line arguments for prediction."""
    parser = argparse.ArgumentParser(
        description="Predict concentration using saved exponential fit parameters."
    )
    
    parser.add_argument(
        "--params",
        type=Path,
        default=default_cfg.params_path,
        help=f"Path to fit parameters CSV (default: {default_cfg.params_path})",
    )
    
    parser.add_argument(
        "--roi-csv",
        type=Path,
        required=True,
        help="Path to ROI statistics CSV for prediction",
    )
    
    parser.add_argument(
        "--channel",
        default=default_cfg.channel,
        help=f"Channel to use for prediction (default: {default_cfg.channel})",
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path (default: <roi-csv>_pred_concentration_exp.csv)",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        help="Output PNG path (default: <output_dir>/<roi_stem>_pred_concentration_exp.png)",
    )
    
    return parser.parse_args()


def resolve_output_path(path: Optional[Path], default_path: Path, output_dir: Path) -> Path:
    chosen = (path or default_path).expanduser()
    if not chosen.is_absolute():
        chosen = output_dir / chosen
    return chosen.resolve()


def predict_concentration(args: argparse.Namespace, cfg: ExpConfig) -> None:
    """Run concentration prediction."""
    
    # Load fitted parameters
    params_path = args.params.expanduser()
    if not params_path.is_absolute():
        params_path = cfg.output_dir / params_path
    params_path = params_path.resolve()

    print(f"Loading parameters from {params_path}...")
    params_by_channel = load_all_fit_parameters(params_path)
    selected_channel = resolve_selected_feature_from_params(params_path, args.channel)
    if selected_channel in params_by_channel:
        a, b, c = params_by_channel[selected_channel]
        print(f"Using light channel: {selected_channel} (requested: {args.channel})")
        print(f"Parameters: a={a:.6g}, b={b:.6g}, c={c:.6g}\n")
    elif selected_channel in AUTO_CHANNELS:
        print(f"Using aggregate light-channel mode: {args.channel}\n")
    else:
        available = sorted(params_by_channel) + sorted(AUTO_CHANNELS)
        raise ValueError(f"Selected channel '{args.channel}' is unavailable. Available: {available}")
    
    # Load ROI data
    df_raw = pd.read_csv(args.roi_csv)
    if df_raw.empty:
        raise ValueError("ROI CSV is empty.")
    
    # Build corrected channels
    corr = build_corrected_channels(df_raw)
    
    channels = [col for col in corr.columns if col != "exposure_time" and not col.endswith("_per_exp")]
    conc_pred, feature_conc_pred = predict_from_fitted_features(corr, params_by_channel, channels, selected_channel)
    
    # Prepare output
    output_df = pd.DataFrame({
        "filename": df_raw["filename"],
        "exposure_time": corr["exposure_time"],
        "conc_pred": conc_pred,
        "selected_channel": selected_channel,
    })
    if selected_channel in corr.columns:
        output_df[f"{selected_channel}_value"] = corr[selected_channel]
    for ch in feature_conc_pred.columns:
        output_df[f"{ch}_conc_pred"] = feature_conc_pred[ch].to_numpy(dtype=float)
    conc_true = df_raw["filename"].apply(extract_concentration_from_name)
    if conc_true.notna().any():
        output_df["conc_true"] = conc_true
        output_df["residual"] = output_df["conc_pred"] - output_df["conc_true"]
    
    # Save predictions
    output_path = resolve_output_path(args.output, cfg.concentration_path_for(args.roi_csv), cfg.output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False, float_format=CSV_FLOAT_FORMAT)

    plot_path = resolve_output_path(args.plot_out, cfg.concentration_plot_path_for(args.roi_csv), cfg.output_dir)
    _draw_prediction_plot(output_df, plot_path, selected_channel)
    
    # Display statistics
    valid_mask = np.isfinite(conc_pred)
    n_valid = valid_mask.sum()
    n_total = len(conc_pred)
    
    print(f"Predicted {n_valid}/{n_total} samples")
    if n_valid > 0:
        print(f"  Range: {conc_pred[valid_mask].min():.6g} - {conc_pred[valid_mask].max():.6g} uM")
        print(f"  Mean: {conc_pred[valid_mask].mean():.6g} uM")
    
    if n_valid < n_total:
        print(f"\nWARNING: {n_total - n_valid} samples have invalid (NaN) predictions")
    
    print(f"\nSaved predictions -> {output_path}")


def _draw_prediction_plot(output_df: pd.DataFrame, plot_path: Path, channel: str) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    conc_pred = output_df["conc_pred"].to_numpy(dtype=float)

    if "conc_true" in output_df.columns:
        conc_true = output_df["conc_true"].to_numpy(dtype=float)
        valid = np.isfinite(conc_true) & np.isfinite(conc_pred)
        if np.any(valid):
            x = conc_true[valid]
            y = conc_pred[valid]
            axes[0].scatter(x, y, s=24, alpha=0.8, color="#2ca02c")
            low = float(min(np.min(x), np.min(y)))
            high = float(max(np.max(x), np.max(y)))
            axes[0].plot([low, high], [low, high], "k--", linewidth=1.0)
            axes[0].set_xlabel("True concentration (uM)")
            axes[0].set_ylabel("Pred concentration (uM)")
            axes[0].set_title(f"Exp Predict ({channel})")
            axes[0].grid(alpha=0.25)

            residual = y - x
            axes[1].hist(residual, bins=20, color="#ff7f0e", alpha=0.8)
            axes[1].axvline(0.0, color="k", linestyle="--", linewidth=1.0)
            axes[1].set_xlabel("Residual (pred - true)")
            axes[1].set_ylabel("Count")
            axes[1].set_title("Residual Distribution")
            axes[1].grid(alpha=0.25)
        else:
            axes[0].text(0.5, 0.5, "No finite labels", ha="center", va="center")
            axes[1].text(0.5, 0.5, "No finite labels", ha="center", va="center")
    else:
        axes[0].hist(conc_pred[np.isfinite(conc_pred)], bins=20, color="#2ca02c", alpha=0.8)
        axes[0].set_xlabel("Pred concentration (uM)")
        axes[0].set_ylabel("Count")
        axes[0].set_title(f"Exp Predict ({channel})")
        axes[0].grid(alpha=0.25)

        axes[1].plot(np.arange(len(conc_pred)), conc_pred, ".", markersize=4, color="#d62728")
        axes[1].set_xlabel("Sample index")
        axes[1].set_ylabel("Pred concentration (uM)")
        axes[1].set_title("Prediction by Sample")
        axes[1].grid(alpha=0.25)

    format_figure_numbers(fig)
    plt.tight_layout()
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot -> {plot_path}")


def main() -> None:
    """Entry point."""
    cfg = ExpConfig()
    args = parse_args(cfg)
    predict_concentration(args, cfg)


if __name__ == "__main__":
    main()
