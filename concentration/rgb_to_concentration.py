"""
Invert RGB intensity to concentration using fitted exponential parameters.

Inputs:
- concentration/rgb_fit_output/rgb_fit_params.csv  (per-channel a, b, c, R², RMSE)

Usage examples:
    python rgb_to_concentration.py --channel G_corr --intensity 75
    python rgb_to_concentration.py --channel G_corr --roi-csv ../pic_RGB/roi_results/roi_detailed_statistics.csv --out out_conc.csv

Notes:
- Model per channel: intensity = a * exp(b * conc) + c  (b < 0 for decay)
- Inversion: conc = ln((intensity - c) / a) / b
- If (intensity - c) <= 0, returns inf.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FIT_DIR = ROOT / "rgb_fit_output"
PARAM_PATH = FIT_DIR / "rgb_fit_params.csv"


def load_params(path: Path, channel: str) -> Tuple[float, float, float]:
    df = pd.read_csv(path)
    row = df.loc[df["channel"] == channel]
    if row.empty:
        raise ValueError(f"Channel {channel} not found in {path}")
    a, b, c = row.iloc[0][["a", "b", "c"]]
    return float(a), float(b), float(c)


def invert_intensity(intensity: float, params: Tuple[float, float, float]) -> float:
    a, b, c = params
    if b == 0:
        raise ValueError("Parameter b cannot be zero.")
    if intensity <= c or a == 0:
        return math.inf
    ratio = (intensity - c) / a
    if ratio <= 0:
        return math.inf
    return float(math.log(ratio) / b)


def invert_csv(
    roi_csv: Path,
    out_csv: Path,
    channel: str,
    params: Tuple[float, float, float],
) -> None:
    df = pd.read_csv(roi_csv)
    if channel not in df.columns:
        raise ValueError(f"Channel {channel} not found in {roi_csv}")
    df = df.copy()
    df[f"{channel}_pred_concentration"] = df[channel].apply(lambda x: invert_intensity(x, params))
    df.to_csv(out_csv, index=False)
    print(f"Saved inverted concentrations -> {out_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Map RGB intensity to concentration using fitted exponential params.")
    p.add_argument("--channel", default="G_corr", help="Channel column name from rgb_fit_params.csv (e.g., R_corr, G1_corr, G2_corr, G_corr, B_corr)")
    p.add_argument("--intensity", type=float, help="Single intensity value to invert.")
    p.add_argument("--roi-csv", type=Path, help="Optional ROI CSV to batch invert (must contain the chosen channel).")
    p.add_argument("--out", type=Path, help="Output CSV path for batch inversion.")
    p.add_argument("--params", type=Path, default=PARAM_PATH, help="Path to rgb_fit_params.csv")
    return p.parse_args()


def main():
    args = parse_args()
    params = load_params(args.params, args.channel)
    print(f"Using params for {args.channel}: a={params[0]:.4f}, b={params[1]:.6f}, c={params[2]:.4f}")

    if args.intensity is not None:
        conc = invert_intensity(args.intensity, params)
        print(f"Intensity {args.intensity} -> concentration {conc}")

    if args.roi_csv:
        out_path = args.out or args.roi_csv.with_name(args.roi_csv.stem + f"_{args.channel}_conc.csv")
        invert_csv(args.roi_csv, out_path, args.channel, params)


if __name__ == "__main__":
    main()
