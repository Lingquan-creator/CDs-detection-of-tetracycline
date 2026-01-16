"""
Solve exponential parameters A, B (and C) from 3 images: 1 background + 2 known concentrations.

Model: y = A * exp(B * x) + C
Given:
  - Background image -> intensity = C (using background-corrected channel or raw channel)
  - Sample1 (x1, y1), Sample2 (x2, y2)
Solve:
  B = ln((y2 - C) / (y1 - C)) / (x2 - x1)
  A = (y1 - C) / exp(B * x1)

Usage example:
    python solve_exp_params.py ^
        --roi-csv ../pic_RGB/roi_results/roi_detailed_statistics.csv ^
        --bg-file blank.dng ^
        --sample1 0.2.dng --conc1 0.2 ^
        --sample2 2.0.dng --conc2 2.0 ^
        --channel G_corr

Notes:
- If you choose channel G_corr, the script computes it as (roi_G1_mean+roi_G2_mean)/2 - (bg_G1_mean+bg_G2_mean)/2.
- For raw channels, you can use roi_G1_mean etc.; the script will just pull that column.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_ROI_CSV = ROOT.parent / "pic_RGB" / "roi_results" / "roi_detailed_statistics.csv"


def compute_channel(df_row: pd.Series, channel: str) -> float:
    if channel == "G_corr":
        g_roi = (df_row["roi_G1_mean"] + df_row["roi_G2_mean"]) / 2.0
        g_bg = (df_row["bg_G1_mean"] + df_row["bg_G2_mean"]) / 2.0
        return float(g_roi - g_bg)
    if channel.endswith("_corr"):
        base = channel[:-5].upper()  # e.g., R_corr -> R
        return float(df_row[f"roi_{base}_mean"] - df_row[f"bg_{base}_mean"])
    return float(df_row[channel])


def solve_params(y_bg: float, x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float]:
    C = y_bg
    if y1 <= C or y2 <= C:
        raise ValueError("Sample intensity must be greater than background (C).")
    B = math.log((y2 - C) / (y1 - C)) / (x2 - x1)
    A = (y1 - C) / math.exp(B * x1)
    return A, B, C


def main():
    ap = argparse.ArgumentParser(description="Solve exponential A,B,C from background + two samples.")
    ap.add_argument("--roi-csv", type=Path, default=DEFAULT_ROI_CSV, help="Path to roi_detailed_statistics.csv")
    ap.add_argument("--bg-file", required=True, help="Filename for background image (match in ROI CSV)")
    ap.add_argument("--sample1", required=True, help="Filename for first sample")
    ap.add_argument("--conc1", type=float, required=True, help="Concentration for first sample")
    ap.add_argument("--sample2", required=True, help="Filename for second sample")
    ap.add_argument("--conc2", type=float, required=True, help="Concentration for second sample")
    ap.add_argument("--channel", default="G_corr", help="Channel to use (G_corr, R_corr, G1_corr, G2_corr, B_corr or raw column name)")
    args = ap.parse_args()

    df = pd.read_csv(args.roi_csv)
    def row_for(name: str) -> pd.Series:
        row = df.loc[df["filename"] == name]
        if row.empty:
            raise ValueError(f"Filename {name} not found in {args.roi_csv}")
        return row.iloc[0]

    bg_val = compute_channel(row_for(args.bg_file), args.channel)
    y1 = compute_channel(row_for(args.sample1), args.channel)
    y2 = compute_channel(row_for(args.sample2), args.channel)

    A, B, C = solve_params(bg_val, args.conc1, y1, args.conc2, y2)

    print(f"Solved params for {args.channel}:")
    print(f"  A = {A:.6f}")
    print(f"  B = {B:.6f}")
    print(f"  C = {C:.6f}  (from background)")


if __name__ == "__main__":
    main()
