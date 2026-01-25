"""
Inspect roi_detailed_statistics.csv and print summary snippets.
"""
from pathlib import Path

import pandas as pd

from ..config.roi_config import OUTPUT_DIR


def analyze_detailed_statistics():
    """Load roi_detailed_statistics.csv and print key stats."""
    csv_path = Path(OUTPUT_DIR) / "roi_detailed_statistics.csv"

    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        print("Generate it by running ROI processing first: python -m pic_RGB.extraction.roi_processor")
        return

    df = pd.read_csv(csv_path)

    print("=" * 70)
    print("ROI detailed statistics")
    print("=" * 70)
    print(f"\nTotal images: {len(df)}\n")

    print("Available columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")

    mean_cols = ["filename", "roi_R_mean", "roi_G1_mean", "roi_G2_mean", "roi_B_mean"]
    if all(col in df.columns for col in mean_cols):
        print("\n" + "=" * 70)
        print("1) ROI channel means")
        print("=" * 70)
        print(df[mean_cols].to_string(index=False))

    ratio_cols = ["filename", "roi_R_G_ratio", "roi_B_G_ratio", "roi_R_B_ratio", "roi_G1_G2_ratio"]
    if all(col in df.columns for col in ratio_cols):
        print("\n" + "=" * 70)
        print("2) Channel ratios (background corrected)")
        print("=" * 70)
        print(df[ratio_cols].to_string(index=False))

    sat_cols = [
        "filename",
        "roi_R_saturation_ratio",
        "roi_G1_saturation_ratio",
        "roi_G2_saturation_ratio",
        "roi_B_saturation_ratio",
    ]
    if all(col in df.columns for col in sat_cols):
        print("\n" + "=" * 70)
        print("3) Saturation ratio checks")
        print("=" * 70)
        print(df[sat_cols].to_string(index=False))
        max_sat = df[[c for c in sat_cols if c != "filename"]].max(axis=1)
        over_saturated = df[max_sat > 0.05]
        print(f"\nWarning: {len(over_saturated)} images exceed 5% saturation in at least one channel")
        if len(over_saturated) > 0:
            print("  Images:", over_saturated["filename"].to_list())

    percentile_cols = ["filename", "roi_R_p10", "roi_R_p50", "roi_R_p90"]
    if all(col in df.columns for col in percentile_cols):
        print("\n" + "=" * 70)
        print("4) R-channel percentiles")
        print("=" * 70)
        print(df[percentile_cols].to_string(index=False))
        df["R_dynamic_range"] = df["roi_R_p90"] - df["roi_R_p10"]
        print("\nR channel dynamic range (p90 - p10):")
        print(df[["filename", "R_dynamic_range"]].to_string(index=False))

    var_cols = ["filename", "roi_R_var", "roi_G1_var", "roi_G2_var", "roi_B_var"]
    if all(col in df.columns for col in var_cols):
        print("\n" + "=" * 70)
        print("5) Variance per channel")
        print("=" * 70)
        print(df[var_cols].to_string(index=False))

    bg_cols = ["filename", "bg_R_mean", "bg_G1_mean", "bg_G2_mean", "bg_B_mean"]
    if all(col in df.columns for col in bg_cols):
        print("\n" + "=" * 70)
        print("6) Background means")
        print("=" * 70)
        print(df[bg_cols].to_string(index=False))

    if "roi_R_mean" in df.columns:
        print("\nSummary of ROI mean ranges:")
        for ch in ["R", "G1", "G2", "B"]:
            col = f"roi_{ch}_mean"
            if col in df.columns:
                print(
                    f"  {ch:3s}: [{df[col].min():8.2f}, {df[col].max():8.2f}], "
                    f"mean={df[col].mean():8.2f}, std={df[col].std():8.2f}"
                )

    if "roi_R_G_ratio" in df.columns:
        print("\nSummary of channel ratios (ROI):")
        for ratio in ["R_G_ratio", "B_G_ratio", "R_B_ratio"]:
            col = f"roi_{ratio}"
            if col in df.columns:
                print(
                    f"  {ratio:12s}: [{df[col].min():6.3f}, {df[col].max():6.3f}], "
                    f"mean={df[col].mean():6.3f}"
                )

    if "roi_G1_G2_ratio" in df.columns:
        g_ratio = df["roi_G1_G2_ratio"]
        print("\nG1/G2 consistency check:")
        print(f"  Range: [{g_ratio.min():.3f}, {g_ratio.max():.3f}]")
        print(f"  Mean: {g_ratio.mean():.3f} (target ~1.0)")
        outliers = df[(g_ratio < 0.8) | (g_ratio > 1.2)]
        if len(outliers) > 0:
            print(f"  Warning: {len(outliers)} images outside [0.8, 1.2] for G1/G2 ratio")


if __name__ == "__main__":
    analyze_detailed_statistics()
