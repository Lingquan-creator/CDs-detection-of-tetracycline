from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent
CSV_PATH = DATASET_DIR / "roi_results" / "roi_detailed_statistics.csv"

RAW_NAME_RE = re.compile(r"^(?P<sample>.+)-(?P<batch>\d+)-(?P<concentration>[-+]?\d+(?:\.\d+)?)$")
TRAILING_CONCENTRATION_RE = re.compile(
    r"^(?P<sample>.*?)(?P<concentration>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def concentration_from_filename(filename: object) -> float:
    stem = Path(str(filename)).stem
    raw_match = RAW_NAME_RE.fullmatch(stem)
    if raw_match:
        return float(raw_match.group("concentration"))
    trailing_match = TRAILING_CONCENTRATION_RE.fullmatch(stem)
    if trailing_match and trailing_match.group("sample"):
        return float(trailing_match.group("concentration"))
    match = NUMBER_RE.search(stem)
    return float(match.group(0)) if match else math.nan


def load_data() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"ROI statistics CSV not found: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    if "filename" not in df.columns:
        raise ValueError(f"CSV must include a filename column: {CSV_PATH}")

    if "concentration" in df.columns and df["concentration"].notna().any():
        conc = pd.to_numeric(df["concentration"], errors="coerce")
    else:
        conc = df["filename"].apply(concentration_from_filename)

    out = pd.DataFrame({"filename": df["filename"], "concentration": conc})
    out["R_roi"] = pd.to_numeric(df["roi_R_mean"], errors="coerce")
    out["G_roi"] = (
        pd.to_numeric(df["roi_G1_mean"], errors="coerce")
        + pd.to_numeric(df["roi_G2_mean"], errors="coerce")
    ) / 2.0
    out["B_roi"] = pd.to_numeric(df["roi_B_mean"], errors="coerce")

    for column in ("R_sig", "G_sig", "B_sig"):
        if column in df.columns:
            out[column] = pd.to_numeric(df[column], errors="coerce")

    out = out.dropna(subset=["concentration"]).sort_values(["concentration", "filename"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("No valid concentration values were parsed.")
    return out


def save_plot(df: pd.DataFrame, columns: tuple[str, str, str], out_path: Path, title: str, ylabel: str) -> None:
    colors = {"R": "#d62728", "G": "#2ca02c", "B": "#1f77b4"}
    fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=160)
    for label, column in zip(("R", "G", "B"), columns):
        valid = np.isfinite(df["concentration"]) & np.isfinite(df[column])
        ax.scatter(
            df.loc[valid, "concentration"],
            df.loc[valid, column],
            s=34,
            alpha=0.82,
            color=colors[label],
            edgecolor="white",
            linewidth=0.35,
            label=label,
        )
    ax.set_xlabel("Concentration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(title="Channel")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_data()
    df.to_csv(SCRIPT_DIR / "rgb_vs_concentration_data.csv", index=False, float_format="%.6g")
    save_plot(
        df,
        ("R_roi", "G_roi", "B_roi"),
        SCRIPT_DIR / "roi_rgb_vs_concentration.png",
        f"{DATASET_DIR.name}: ROI RGB vs concentration",
        "ROI mean RGB",
    )
    if {"R_sig", "G_sig", "B_sig"}.issubset(df.columns):
        save_plot(
            df,
            ("R_sig", "G_sig", "B_sig"),
            SCRIPT_DIR / "signal_rgb_vs_concentration.png",
            f"{DATASET_DIR.name}: background-corrected RGB vs concentration",
            "ROI - BG signal",
        )
    print(f"Saved outputs to {SCRIPT_DIR}")


if __name__ == "__main__":
    main()
