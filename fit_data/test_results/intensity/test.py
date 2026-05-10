from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
TEST_RESULTS_DIR = SCRIPT_DIR.parent
FIT_DATA_DIR = TEST_RESULTS_DIR.parent
REPO_ROOT = FIT_DATA_DIR.parent
DATASET_DIR = FIT_DATA_DIR / "dataset"
RESULTS_DIR = FIT_DATA_DIR / "results"
OUTPUT_DIR = SCRIPT_DIR

sys.path.insert(0, str(FIT_DATA_DIR))

from integrate_and_fit import INTEGRATION_RANGE  # noqa: E402


MODEL_CSV = RESULTS_DIR / "results_intensity_models.csv"
SAMPLE_FILES = {
    "egg": DATASET_DIR / "egg.csv",
    "milk": DATASET_DIR / "milk.csv",
}

SAMPLE_RESULT_CSV = OUTPUT_DIR / "sample_predictions.csv"
R2_RESULT_CSV = OUTPUT_DIR / "r2_summary.csv"
PREDICTION_PLOT = OUTPUT_DIR / "prediction_r2_plot.png"


def parse_model_params(path: Path = MODEL_CSV) -> tuple[float, float]:
    model_df = pd.read_csv(path)
    if model_df.empty:
        raise ValueError(f"No fitted model rows found in {path}")

    best = model_df.iloc[0]
    if str(best["model"]).strip().lower() != "exponential":
        raise ValueError(f"Only exponential intensity model is supported, got {best['model']!r}")

    params = [float(part.strip()) for part in str(best["params"]).split(";")]
    if len(params) == 2:
        return params[0], params[1]
    if len(params) == 3 and abs(params[2]) <= 1e-12:
        return params[0], params[1]
    raise ValueError(f"Expected c=0 exponential params in {path}, got {params}")


def parse_sample_column(label: str) -> tuple[str, int, float]:
    """Parse sample columns named as food-test_batch-true_concentration."""
    match = re.match(
        r"^(?P<sample>.+)-(?P<trial>\d+)-(?P<added>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$",
        str(label).strip(),
    )
    if not match:
        raise ValueError(f"Cannot parse sample column header: {label!r}")
    return (
        match.group("sample"),
        int(match.group("trial")),
        float(match.group("added")),
    )


def invert_exponential_intensity(area: float, params: tuple[float, float]) -> float:
    a, b = params
    if a == 0 or b == 0:
        return math.nan
    ratio = area / a
    if ratio <= 0 or not math.isfinite(ratio):
        return math.nan
    return math.log(ratio) / b


def integrate_column(wavelengths: np.ndarray, intensity: np.ndarray) -> float:
    low, high = INTEGRATION_RANGE
    mask = (wavelengths >= low) & (wavelengths <= high)
    if not np.any(mask):
        raise ValueError(f"No wavelengths in integration range {INTEGRATION_RANGE}")
    return float(np.trapezoid(intensity[mask], wavelengths[mask]))


def load_sample_file(path: Path, model_params: tuple[float, float]) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.shape[1] < 2:
        raise ValueError(f"Expected wavelength + sample columns in {path}")

    wavelengths = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    rows: list[dict] = []
    for column in df.columns[1:]:
        sample_name, trial, added_uM = parse_sample_column(column)
        intensity = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(wavelengths) & np.isfinite(intensity)
        area = integrate_column(wavelengths[valid], intensity[valid])
        predicted_uM = invert_exponential_intensity(area, model_params)
        rows.append(
            {
                "column": str(column),
                "file": path.name,
                "sample": sample_name,
                "trial": trial,
                "true_uM": added_uM,
                "integrated_intensity": area,
                "predicted_uM": predicted_uM,
                "residual": predicted_uM - added_uM if math.isfinite(predicted_uM) else math.nan,
                "abs_error": abs(predicted_uM - added_uM) if math.isfinite(predicted_uM) else math.nan,
            }
        )

    return pd.DataFrame(rows)


def metric_row(group_name: str, samples: pd.DataFrame) -> dict:
    valid = np.isfinite(samples["true_uM"].to_numpy(dtype=float)) & np.isfinite(
        samples["predicted_uM"].to_numpy(dtype=float)
    )
    y_true = samples.loc[valid, "true_uM"].to_numpy(dtype=float)
    y_pred = samples.loc[valid, "predicted_uM"].to_numpy(dtype=float)
    if y_true.size == 0:
        return {
            "group": group_name,
            "n": 0,
            "r2": math.nan,
            "rmse": math.nan,
            "mae": math.nan,
            "mean_error": math.nan,
            "median_abs_error": math.nan,
            "max_abs_error": math.nan,
        }

    residual = y_pred - y_true
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "group": group_name,
        "n": int(y_true.size),
        "r2": math.nan if ss_tot <= 0 else 1.0 - ss_res / ss_tot,
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
        "mean_error": float(np.mean(residual)),
        "median_abs_error": float(np.median(np.abs(residual))),
        "max_abs_error": float(np.max(np.abs(residual))),
    }


def build_r2_summary(samples: pd.DataFrame) -> pd.DataFrame:
    rows = [metric_row("all", samples)]
    for sample_name, group in samples.groupby("sample", sort=True):
        rows.append(metric_row(str(sample_name), group))
    return pd.DataFrame(rows)


def draw_prediction_plot(samples: pd.DataFrame, summary: pd.DataFrame, output_path: Path) -> None:
    valid = np.isfinite(samples["true_uM"].to_numpy(dtype=float)) & np.isfinite(
        samples["predicted_uM"].to_numpy(dtype=float)
    )
    plot_df = samples.loc[valid].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=160)
    if plot_df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No finite predictions", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return

    colors = plt.cm.tab10(np.linspace(0, 1, max(plot_df["sample"].nunique(), 1)))
    for color, (sample_name, group) in zip(colors, plot_df.groupby("sample", sort=True)):
        axes[0].scatter(
            group["true_uM"],
            group["predicted_uM"],
            s=42,
            alpha=0.85,
            label=str(sample_name),
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        axes[1].scatter(
            group["true_uM"],
            group["residual"],
            s=42,
            alpha=0.85,
            label=str(sample_name),
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )

    low = float(min(plot_df["true_uM"].min(), plot_df["predicted_uM"].min()))
    high = float(max(plot_df["true_uM"].max(), plot_df["predicted_uM"].max()))
    pad = max((high - low) * 0.05, 1.0)
    axes[0].plot([low - pad, high + pad], [low - pad, high + pad], color="black", linestyle="--", linewidth=1.0)
    axes[0].set_xlim(low - pad, high + pad)
    axes[0].set_ylim(low - pad, high + pad)
    axes[0].set_xlabel("True concentration from column name (uM)")
    axes[0].set_ylabel("Predicted concentration (uM)")
    axes[0].set_title("Prediction vs truth")
    axes[0].grid(alpha=0.25)
    axes[0].legend(title="Food")

    all_row = summary[summary["group"] == "all"].iloc[0]
    axes[0].text(
        0.04,
        0.96,
        f"R2={float(all_row['r2']):.4g}\nRMSE={float(all_row['rmse']):.4g}\nMAE={float(all_row['mae']):.4g}\nn={int(all_row['n'])}",
        ha="left",
        va="top",
        transform=axes[0].transAxes,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.9},
    )

    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("True concentration from column name (uM)")
    axes[1].set_ylabel("Residual (predicted - true)")
    axes[1].set_title("Residuals")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_params = parse_model_params()

    sample_frames = [
        load_sample_file(path, model_params)
        for path in SAMPLE_FILES.values()
        if path.exists()
    ]
    samples = pd.concat(sample_frames, ignore_index=True)
    r2_summary = build_r2_summary(samples)

    samples.to_csv(SAMPLE_RESULT_CSV, index=False)
    r2_summary.to_csv(R2_RESULT_CSV, index=False)
    draw_prediction_plot(samples, r2_summary, PREDICTION_PLOT)

    print(f"Model params (a, b), c fixed at 0: {model_params}")
    print(f"Saved sample predictions: {SAMPLE_RESULT_CSV}")
    print(f"Saved R2 summary: {R2_RESULT_CSV}")
    print(f"Saved prediction plot: {PREDICTION_PLOT}")
    print(r2_summary.to_string(index=False))


if __name__ == "__main__":
    main()
