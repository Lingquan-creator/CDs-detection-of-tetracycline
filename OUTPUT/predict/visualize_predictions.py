from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CSV_FLOAT_FORMAT = "%.6g"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize concentration_regression predictions.")
    parser.add_argument("--pred-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float, int]:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y = y_true[valid]
    pred = y_pred[valid]
    if len(y) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    residual = pred - y
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    rmse = float(np.sqrt(ss_res / len(y)))
    mae = float(np.mean(np.abs(residual)))
    return r2, rmse, mae, int(len(y))


def food_label(filename: str) -> str:
    return str(filename).split("-", maxsplit=1)[0]


def concentration_from_filename(filename: str) -> float:
    stem = Path(str(filename)).stem
    matches = re.findall(r"\d+(?:\.\d+)?", stem)
    if not matches:
        return float("nan")
    return float(matches[-1])


def save_scatter(df: pd.DataFrame, out_path: Path, r2: float, rmse: float) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=160)
    foods = sorted(df["food"].dropna().unique())
    colors = {"chicken": "#1f77b4", "egg": "#ff7f0e"}
    for food in foods:
        sub = df[df["food"] == food]
        ax.scatter(
            sub["conc_true"],
            sub["conc_pred"],
            s=42,
            alpha=0.85,
            label=food,
            color=colors.get(food, None),
            edgecolor="white",
            linewidth=0.5,
        )
    low = float(np.nanmin([df["conc_true"].min(), df["conc_pred"].min()]))
    high = float(np.nanmax([df["conc_true"].max(), df["conc_pred"].max()]))
    pad = max((high - low) * 0.05, 1.0)
    ax.plot([low - pad, high + pad], [low - pad, high + pad], "k--", linewidth=1.0)
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)
    ax.set_xlabel("True concentration")
    ax.set_ylabel("Predicted concentration")
    ax.set_title(f"dataset/test prediction: R2={r2:.4g}, RMSE={rmse:.4g}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_residual_plot(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=160)
    foods = sorted(df["food"].dropna().unique())
    colors = {"chicken": "#1f77b4", "egg": "#ff7f0e"}
    for food in foods:
        sub = df[df["food"] == food]
        ax.scatter(
            sub["conc_true"],
            sub["residual"],
            s=38,
            alpha=0.85,
            label=food,
            color=colors.get(food, None),
            edgecolor="white",
            linewidth=0.5,
        )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("True concentration")
    ax.set_ylabel("Residual (pred - true)")
    ax.set_title("Residuals by concentration")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_sample_plot(df: pd.DataFrame, out_path: Path) -> None:
    ordered = df.sort_values(["food", "conc_true", "filename"]).reset_index(drop=True)
    x = np.arange(len(ordered))

    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=160)
    ax.plot(x, ordered["conc_true"], "o-", markersize=4, linewidth=1.0, label="true")
    ax.plot(x, ordered["conc_pred"], "s-", markersize=4, linewidth=1.0, label="pred")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["filename"], rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("Concentration")
    ax.set_title("Prediction by sample")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.pred_csv)
    if "conc_pred" not in df.columns:
        raise ValueError("Prediction CSV must contain a conc_pred column.")

    df = df.copy()
    df["food"] = df["filename"].apply(food_label)
    if "conc_true" in df.columns:
        df["conc_true_from_predict"] = df["conc_true"]
    df["conc_true"] = df["filename"].apply(concentration_from_filename)
    if "residual" not in df.columns:
        df["residual"] = df["conc_pred"] - df["conc_true"]
    else:
        df["residual_from_predict"] = df["residual"]
        df["residual"] = df["conc_pred"] - df["conc_true"]

    r2, rmse, mae, n = metrics(
        df["conc_true"].to_numpy(dtype=float),
        df["conc_pred"].to_numpy(dtype=float),
    )
    by_food = []
    for food, sub in df.groupby("food"):
        food_r2, food_rmse, food_mae, food_n = metrics(
            sub["conc_true"].to_numpy(dtype=float),
            sub["conc_pred"].to_numpy(dtype=float),
        )
        by_food.append(
            {
                "group": food,
                "r2": food_r2,
                "rmse": food_rmse,
                "mae": food_mae,
                "n": food_n,
            }
        )

    metrics_df = pd.DataFrame(
        [{"group": "all", "r2": r2, "rmse": rmse, "mae": mae, "n": n}, *by_food]
    )
    metrics_path = args.out_dir / "test_prediction_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, float_format=CSV_FLOAT_FORMAT)

    save_scatter(df, args.out_dir / "test_prediction_scatter.png", r2, rmse)
    save_residual_plot(df, args.out_dir / "test_prediction_residuals.png")
    save_sample_plot(df, args.out_dir / "test_prediction_by_sample.png")
    labeled_path = args.out_dir / "test_ridge_predictions_labeled.csv"
    df.to_csv(labeled_path, index=False, float_format=CSV_FLOAT_FORMAT)

    print(metrics_df.to_string(index=False))
    print(f"Saved metrics -> {metrics_path}")
    print(f"Saved relabeled predictions -> {labeled_path}")


if __name__ == "__main__":
    main()
