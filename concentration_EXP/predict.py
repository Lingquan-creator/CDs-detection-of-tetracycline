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

import numpy as np
import pandas as pd

from config import ExpConfig
from fit_exponential import build_corrected_channels, invert_exponential_intensity


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
        choices=["R_corr", "G1_corr", "G2_corr", "G_corr", "B_corr"],
        default=default_cfg.channel,
        help=f"Channel to use for prediction (default: {default_cfg.channel})",
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path (default: <roi-csv>_pred_concentration_exp.csv)",
    )
    
    return parser.parse_args()


def predict_concentration(args: argparse.Namespace, cfg: ExpConfig) -> None:
    """Run concentration prediction."""
    
    # Load fitted parameters
    print(f"Loading parameters from {args.params}...")
    a, b, c = load_fit_parameters(args.params, args.channel)
    print(f"Using channel: {args.channel}")
    print(f"Parameters: a={a:.3f}, b={b:.6f}, c={c:.3f}\n")
    
    # Load ROI data
    df_raw = pd.read_csv(args.roi_csv)
    if df_raw.empty:
        raise ValueError("ROI CSV is empty.")
    
    # Build corrected channels
    corr = build_corrected_channels(df_raw)
    
    # Get intensity for selected channel
    intensity = corr[args.channel].to_numpy(dtype=float)
    
    # Predict concentration
    conc_pred = invert_exponential_intensity(intensity, a=a, b=b, c0=c)
    conc_pred = np.maximum(np.asarray(conc_pred, dtype=float), 0.0)
    
    # Prepare output
    output_df = pd.DataFrame({
        "filename": df_raw["filename"],
        "conc_pred": conc_pred,
        f"{args.channel}_intensity": intensity,
    })
    
    # Save predictions
    output_path = args.output or cfg.concentration_path_for(args.roi_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    
    # Display statistics
    valid_mask = np.isfinite(conc_pred)
    n_valid = valid_mask.sum()
    n_total = len(conc_pred)
    
    print(f"✓ Predicted {n_valid}/{n_total} samples")
    if n_valid > 0:
        print(f"  Range: {conc_pred[valid_mask].min():.2f} - {conc_pred[valid_mask].max():.2f} µM")
        print(f"  Mean: {conc_pred[valid_mask].mean():.2f} µM")
    
    if n_valid < n_total:
        print(f"\n⚠ Warning: {n_total - n_valid} samples have invalid (NaN) predictions")
    
    print(f"\n✓ Saved predictions → {output_path}")


def main() -> None:
    """Entry point."""
    cfg = ExpConfig()
    args = parse_args(cfg)
    predict_concentration(args, cfg)


if __name__ == "__main__":
    main()
