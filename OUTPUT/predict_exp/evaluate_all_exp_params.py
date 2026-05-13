from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "concentration_EXP"))

from predict import load_all_fit_parameters  # noqa: E402
from train import build_response_feature_table, invert_response_by_grid  # noqa: E402


CSV_FLOAT_FORMAT = "%.6g"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate every concentration_EXP fit parameter row on dataset/test.")
    parser.add_argument(
        "--params",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "train" / "model_fit" / "concentration_EXP" / "exp_fit_params.csv",
    )
    parser.add_argument(
        "--roi-csv",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "test" / "roi_results" / "roi_detailed_statistics.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "OUTPUT" / "predict_exp")
    return parser.parse_args()


def concentration_from_filename(filename: str) -> float:
    stem = Path(str(filename)).stem
    matches = re.findall(r"\d+(?:\.\d+)?", stem)
    return float(matches[-1]) if matches else float("nan")


def food_label(filename: str) -> str:
    stem = Path(str(filename)).stem
    return stem.split("-", maxsplit=1)[0] if "-" in stem else "all"


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y) & np.isfinite(pred)
    y = y[valid]
    pred = pred[valid]
    if y.size == 0:
        return {"n": 0, "r2": np.nan, "rmse": np.nan, "mae": np.nan, "mean_error": np.nan}
    residual = pred - y
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    return {
        "n": int(y.size),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0,
        "rmse": float(np.sqrt(ss_res / y.size)),
        "mae": float(np.mean(np.abs(residual))),
        "mean_error": float(np.mean(residual)),
    }


def spec_param_text(spec: dict[str, Any]) -> str:
    return ";".join(f"{float(value):.12g}" for value in spec["params"])


def build_predictions(
    *,
    df_raw: pd.DataFrame,
    fitted_models: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    channels = tuple(dict.fromkeys(str(spec["channel"]) for spec in fitted_models.values()))
    features = build_response_feature_table(df_raw, channels)

    base = pd.DataFrame(
        {
            "filename": df_raw["filename"],
            "food": df_raw["filename"].apply(food_label),
            "conc_true": df_raw["filename"].apply(concentration_from_filename),
            "exposure_time": features["exposure_time"],
        }
    )
    y_true = base["conc_true"].to_numpy(dtype=float)

    long_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    wide = base.copy()

    for fit_id, spec in fitted_models.items():
        channel = str(spec["channel"])
        if channel not in features.columns:
            continue
        pred = np.maximum(
            invert_response_by_grid(
                features[channel].to_numpy(dtype=float),
                model=str(spec["model"]),
                params=tuple(float(v) for v in spec["params"]),
                conc_min=float(spec["conc_min"]),
                conc_max=float(spec["conc_max"]),
            ),
            0.0,
        )

        wide[f"{fit_id}_conc_pred"] = pred
        one = base.copy()
        one["fit_id"] = fit_id
        one["channel"] = channel
        one["model"] = str(spec["model"])
        one["feature_value"] = features[channel].to_numpy(dtype=float)
        one["conc_pred"] = pred
        one["residual"] = one["conc_pred"] - one["conc_true"]
        long_rows.append(one)

        row = {
            "fit_id": fit_id,
            "channel": channel,
            "model": str(spec["model"]),
            "params": spec_param_text(spec),
            "train_fit_r2": spec.get("r2", np.nan),
            "train_inv_r2": spec.get("inv_r2", np.nan),
            **metrics(y_true, pred),
        }
        metric_rows.append(row)
        for food, sub in one.groupby("food"):
            food_metrics = metrics(
                sub["conc_true"].to_numpy(dtype=float),
                sub["conc_pred"].to_numpy(dtype=float),
            )
            metric_rows.append(
                {
                    "fit_id": fit_id,
                    "channel": channel,
                    "model": str(spec["model"]),
                    "params": spec_param_text(spec),
                    "train_fit_r2": spec.get("r2", np.nan),
                    "train_inv_r2": spec.get("inv_r2", np.nan),
                    "group": food,
                    **food_metrics,
                }
            )
        metric_rows[-(1 + len(one["food"].unique()))]["group"] = "all"

    long_df = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame()
    metrics_df = pd.DataFrame(metric_rows)
    return long_df, wide, metrics_df


def draw_top_metrics(metrics_df: pd.DataFrame, out_path: Path) -> None:
    all_df = metrics_df[metrics_df["group"] == "all"].copy()
    all_df = all_df.sort_values(["r2", "rmse"], ascending=[False, True]).head(25)
    labels = [f"{row.channel}\n{row.model}" for row in all_df.itertuples(index=False)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    for ax, col, title in zip(axes, ("r2", "rmse", "mae"), ("R2", "RMSE", "MAE")):
        ax.bar(range(len(all_df)), all_df[col], color=["#2ca02c" if i == 0 else "#1f77b4" for i in range(len(all_df))])
        ax.set_title(title)
        ax.set_xticks(range(len(all_df)))
        ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def draw_best_prediction(long_df: pd.DataFrame, metrics_df: pd.DataFrame, out_path: Path) -> None:
    best = metrics_df[metrics_df["group"] == "all"].sort_values(["r2", "rmse"], ascending=[False, True]).iloc[0]
    sub = long_df[long_df["fit_id"] == best["fit_id"]].copy()

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=160)
    colors = {"chicken": "#1f77b4", "egg": "#ff7f0e"}
    for food, food_df in sub.groupby("food"):
        ax.scatter(
            food_df["conc_true"],
            food_df["conc_pred"],
            s=42,
            alpha=0.85,
            label=str(food),
            color=colors.get(str(food), None),
            edgecolor="white",
            linewidth=0.5,
        )
    low = float(min(sub["conc_true"].min(), sub["conc_pred"].min()))
    high = float(max(sub["conc_true"].max(), sub["conc_pred"].max()))
    pad = max((high - low) * 0.05, 1.0)
    ax.plot([low - pad, high + pad], [low - pad, high + pad], "k--", linewidth=1.0)
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)
    ax.set_xlabel("True concentration")
    ax.set_ylabel("Predicted concentration")
    ax.set_title(f"Best EXP fit: {best['fit_id']}\nR2={float(best['r2']):.4g}, RMSE={float(best['rmse']):.4g}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fitted_models = load_all_fit_parameters(args.params.resolve())
    df_raw = pd.read_csv(args.roi_csv.resolve())
    long_df, wide_df, metrics_df = build_predictions(df_raw=df_raw, fitted_models=fitted_models)
    metrics_df = metrics_df.sort_values(["group", "r2", "rmse"], ascending=[True, False, True]).reset_index(drop=True)

    long_path = args.out_dir / "test_exp_all_fit_predictions_long.csv"
    wide_path = args.out_dir / "test_exp_all_fit_predictions_wide.csv"
    metrics_path = args.out_dir / "test_exp_all_fit_metrics.csv"
    long_df.to_csv(long_path, index=False, float_format=CSV_FLOAT_FORMAT)
    wide_df.to_csv(wide_path, index=False, float_format=CSV_FLOAT_FORMAT)
    metrics_df.to_csv(metrics_path, index=False, float_format=CSV_FLOAT_FORMAT)

    draw_top_metrics(metrics_df, args.out_dir / "test_exp_all_fit_top25_metrics.png")
    draw_best_prediction(long_df, metrics_df, args.out_dir / "test_exp_all_fit_best_prediction.png")

    best_rows = metrics_df[metrics_df["group"] == "all"].sort_values(["r2", "rmse"], ascending=[False, True]).head(20)
    print(best_rows[["fit_id", "channel", "model", "r2", "rmse", "mae", "n", "train_inv_r2"]].to_string(index=False))
    print(f"Saved long predictions -> {long_path}")
    print(f"Saved wide predictions -> {wide_path}")
    print(f"Saved metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
