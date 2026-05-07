from __future__ import annotations

import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
DATASET_DIR = ROOT / "dataset"
OUT_DIR = SCRIPT_DIR

DATASETS = {
    "mate20x": {
        "path": DATASET_DIR / "mate20x",
        "phase": "train",
        "line_style": "-",
    },
    "food": {
        "path": DATASET_DIR / "food",
        "phase": "test",
        "line_style": "--",
    },
}

CHANNELS = {
    "R": {"column": "R_sig", "color": "#d62728"},
    "G": {"column": "G_sig", "color": "#2ca02c"},
    "B": {"column": "B_sig", "color": "#1f77b4"},
}

RAW_RGB_COLUMNS = {
    "R": ("roi_R_mean", "R_mean"),
    "G": ("roi_G_mean", "G_mean", "roi_G1_mean", "G1_mean"),
    "B": ("roi_B_mean", "B_mean"),
}

SIGNAL_COLUMNS = {
    "R": ("R_sig", "R_display", "R_corrected"),
    "G": ("G_sig", "G_display", "G_corrected"),
    "B": ("B_sig", "B_display", "B_corrected"),
}

OUTPUT_PNG = OUT_DIR / "roi_rgb_intensity_mate20x_food.png"
OUTPUT_RAW_CSV = OUT_DIR / "roi_rgb_intensity_points.csv"
OUTPUT_MEAN_CSV = OUT_DIR / "roi_rgb_intensity_mean.csv"
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def parse_concentration(filename: object) -> float:
    stem = Path(str(filename)).stem
    matches = NUMBER_RE.findall(stem)
    if not matches:
        return np.nan
    return float(matches[-1])


def parse_sample_name(filename: object, dataset_name: str) -> str:
    stem = Path(str(filename)).stem
    if dataset_name != "food":
        return dataset_name
    return re.sub(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$", "", stem) or "food"


def first_existing_column(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def load_roi_table(dataset_path: Path) -> pd.DataFrame:
    csv_path = dataset_path / "roi_detailed_statistics.csv"
    json_path = dataset_path / "roi_results.json"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if json_path.exists():
        return pd.read_json(json_path)
    raise FileNotFoundError(f"No ROI csv/json found under {dataset_path}")


def normalize_rgb_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for channel in CHANNELS:
        signal_col = first_existing_column(out, SIGNAL_COLUMNS[channel])
        if signal_col is not None:
            out[f"{channel}_intensity"] = pd.to_numeric(out[signal_col], errors="coerce")
            continue

        raw_col = first_existing_column(out, RAW_RGB_COLUMNS[channel])
        if raw_col is None:
            raise ValueError(f"Cannot find {channel} ROI intensity column.")
        out[f"{channel}_intensity"] = pd.to_numeric(out[raw_col], errors="coerce")
    return out


def load_all_points() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for dataset_name, cfg in DATASETS.items():
        df = normalize_rgb_columns(load_roi_table(cfg["path"]))
        if "filename" not in df.columns:
            raise ValueError(f"ROI table for {dataset_name} must include filename column.")

        rows = pd.DataFrame(
            {
                "dataset": dataset_name,
                "phase": cfg["phase"],
                "sample": df["filename"].apply(lambda name: parse_sample_name(name, dataset_name)),
                "filename": df["filename"],
                "concentration_uM": df["filename"].apply(parse_concentration),
                "R_intensity": df["R_intensity"],
                "G_intensity": df["G_intensity"],
                "B_intensity": df["B_intensity"],
            }
        )
        frames.append(rows)

    points = pd.concat(frames, ignore_index=True)
    points = points.dropna(subset=["concentration_uM", "R_intensity", "G_intensity", "B_intensity"])
    return points.sort_values(["phase", "dataset", "sample", "concentration_uM"]).reset_index(drop=True)


def build_mean_curves(points: pd.DataFrame) -> pd.DataFrame:
    value_columns = [f"{channel}_intensity" for channel in CHANNELS]
    return (
        points.groupby(["dataset", "phase", "concentration_uM"], as_index=False)[value_columns]
        .mean()
        .sort_values(["phase", "dataset", "concentration_uM"])
        .reset_index(drop=True)
    )


def plot_rgb_curves(points: pd.DataFrame, means: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=180)

    for dataset_name, cfg in DATASETS.items():
        part = means[means["dataset"] == dataset_name]
        for channel, style in CHANNELS.items():
            ax.plot(
                part["concentration_uM"],
                part[f"{channel}_intensity"],
                linestyle=cfg["line_style"],
                linewidth=2.1,
                color=style["color"],
                marker="o",
                markersize=4.2,
                label=f"{dataset_name} {channel} ({cfg['phase']})",
            )

    food_points = points[points["dataset"] == "food"]
    for channel, style in CHANNELS.items():
        ax.scatter(
            food_points["concentration_uM"],
            food_points[f"{channel}_intensity"],
            s=18,
            color=style["color"],
            alpha=0.22,
            edgecolors="none",
        )

    ax.set_xlabel("Concentration (uM)")
    ax.set_ylabel("ROI RGB intensity")
    ax.set_title("ROI RGB intensity vs concentration")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    points = load_all_points()
    means = build_mean_curves(points)

    points.to_csv(OUTPUT_RAW_CSV, index=False, float_format="%.6g")
    means.to_csv(OUTPUT_MEAN_CSV, index=False, float_format="%.6g")
    plot_rgb_curves(points, means, OUTPUT_PNG)

    print(f"Saved plot: {OUTPUT_PNG}")
    print(f"Saved raw points: {OUTPUT_RAW_CSV}")
    print(f"Saved mean curves: {OUTPUT_MEAN_CSV}")


if __name__ == "__main__":
    main()
