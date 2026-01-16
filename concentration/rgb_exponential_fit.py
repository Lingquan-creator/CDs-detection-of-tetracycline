"""
Fit exponential decay for camera RGB intensities vs concentration and export params/plots.

Reads:
- pic_RGB/roi_results/roi_detailed_statistics.csv
  * Uses filename to parse concentration (numeric part)
  * Background-corrected channels: roi_mean - bg_mean

Outputs (under concentration/rgb_fit_output/):
- rgb_fit_params.csv         : per-channel fitted a, b, c, r2, rmse
- rgb_fit_predictions.csv    : per-sample observed/predicted intensities per channel
- rgb_fit_<channel>.png      : scatter + fitted curve for each channel (R, G1, G2, G_avg, B)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent  # .../concentration
ROI_STATS_PATH = ROOT.parent / "pic_RGB" / "roi_results" / "roi_detailed_statistics.csv"
OUTPUT_DIR = ROOT / "rgb_fit_output"


def _exp_model(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * np.exp(b * x) + c


def parse_concentration_from_name(name: str) -> float:
    stem = Path(name).stem
    match = re.search(r"([0-9]*\.?[0-9]+)", stem)
    if not match:
        raise ValueError(f"Cannot parse concentration from {name}")
    return float(match.group(1))


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["concentration_uM"] = df["filename"].apply(parse_concentration_from_name)

    df["R_corr"] = df["roi_R_mean"] - df["bg_R_mean"]
    df["G1_corr"] = df["roi_G1_mean"] - df["bg_G1_mean"]
    df["G2_corr"] = df["roi_G2_mean"] - df["bg_G2_mean"]
    df["B_corr"] = df["roi_B_mean"] - df["bg_B_mean"]
    df["G_corr"] = (df["G1_corr"] + df["G2_corr"]) / 2.0

    keep_cols = ["filename", "concentration_uM", "R_corr", "G1_corr", "G2_corr", "G_corr", "B_corr"]
    df = df[keep_cols].sort_values("concentration_uM").reset_index(drop=True)
    return df


@dataclass
class FitResult:
    a: float
    b: float
    c: float
    r2: float
    rmse: float

    def as_dict(self, channel: str) -> Dict:
        return {"channel": channel, "a": self.a, "b": self.b, "c": self.c, "r2": self.r2, "rmse": self.rmse}


def fit_channel(x: np.ndarray, y: np.ndarray) -> FitResult:
    # initial guess
    a0 = float(np.max(y) - np.min(y))
    c0 = float(np.min(y))
    b0 = -0.01
    bounds = ([0, -np.inf, -np.inf], [np.inf, 0, np.inf])  # decay => b <= 0
    popt, _ = curve_fit(_exp_model, x, y, p0=[a0, b0, c0], bounds=bounds, maxfev=20000)
    a, b, c = popt
    preds = _exp_model(x, a, b, c)
    residuals = y - preds
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    rmse = float(np.sqrt(np.mean(residuals**2)))
    return FitResult(a=float(a), b=float(b), c=float(c), r2=float(r2), rmse=rmse)


def plot_channel(df: pd.DataFrame, channel: str, fit_res: FitResult, out_dir: Path) -> None:
    x = df["concentration_uM"].to_numpy()
    xs = np.linspace(np.min(x), np.max(x), 200)
    plt.figure(figsize=(7, 5))
    plt.scatter(x, df[channel], color="tab:blue", label="Observed")
    plt.plot(xs, _exp_model(xs, fit_res.a, fit_res.b, fit_res.c), color="tab:red", label="Fit")
    plt.xlabel("Concentration (uM)")
    plt.ylabel(f"{channel} (background-corrected)")
    plt.title(f"{channel} vs concentration (exp decay)\nR²={fit_res.r2:.4f}, RMSE={fit_res.rmse:.2f}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"rgb_fit_{channel}.png", dpi=150)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset(ROI_STATS_PATH)

    channels = ["R_corr", "G1_corr", "G2_corr", "G_corr", "B_corr"]
    fit_rows = []

    for ch in channels:
        fit_res = fit_channel(df["concentration_uM"].to_numpy(), df[ch].to_numpy())
        fit_rows.append(fit_res.as_dict(ch))
        df[f"{ch}_pred"] = _exp_model(df["concentration_uM"], fit_res.a, fit_res.b, fit_res.c)
        plot_channel(df, ch, fit_res, OUTPUT_DIR)

    pd.DataFrame(fit_rows).to_csv(OUTPUT_DIR / "rgb_fit_params.csv", index=False)
    df.to_csv(OUTPUT_DIR / "rgb_fit_predictions.csv", index=False)
    print("Saved params and plots to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
