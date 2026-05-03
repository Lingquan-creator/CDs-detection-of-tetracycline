from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REG_DIR = ROOT / "concentration_regression"
if str(REG_DIR) not in sys.path:
    sys.path.insert(0, str(REG_DIR))

from config import CLIP_MIN, TARGET_EPS, TARGET_TRANSFORM, WEIGHT_EPS  # noqa: E402
from physics_features import build_feature_frame, extract_concentration_from_name  # noqa: E402
from train import apply_calibrator, fit_calibrator, inverse_target, select_feature_set, transform_target  # noqa: E402


CSV_FLOAT_FORMAT = "%.6g"
ALPHA_VALUES = (0.0, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8)
BETA_VALUES = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
WEIGHT_EPS_VALUES = (1.0, 10.0, 30.0, 50.0, 80.0, 120.0)


@dataclass(frozen=True)
class Variant:
    scan: str
    variant_id: str
    alpha: float
    beta: float


def make_sample_weights(y: np.ndarray, *, weight_eps: float, beta: float) -> np.ndarray:
    denom = beta * np.asarray(y, dtype=float) + float(weight_eps)
    if np.any(denom <= 0):
        raise ValueError("beta * concentration + weight_eps must stay positive.")
    weights = 1.0 / denom
    return weights / np.mean(weights)


def train_model_fixed_alpha(X: np.ndarray, y_target: np.ndarray, *, alpha: float, sample_weight: np.ndarray) -> Pipeline:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=float(alpha), fit_intercept=True)),
        ]
    )
    model.fit(X, y_target, regressor__sample_weight=sample_weight)
    return model


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        return {
            "n": 0,
            "r2": math.nan,
            "rmse": math.nan,
            "mae": math.nan,
            "mean_error": math.nan,
            "median_abs_error": math.nan,
            "max_abs_error": math.nan,
        }
    y = y_true[valid]
    pred = y_pred[valid]
    residual = pred - y
    return {
        "n": int(valid.sum()),
        "r2": float(r2_score(y, pred)) if y.size >= 2 else math.nan,
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "mean_error": float(np.mean(residual)),
        "median_abs_error": float(np.median(np.abs(residual))),
        "max_abs_error": float(np.max(np.abs(residual))),
    }


def group_from_filename(filename: object) -> str:
    stem = Path(str(filename)).stem.lower()
    if stem.startswith("chicken"):
        return "chicken"
    if stem.startswith("egg"):
        return "egg"
    return "other"


def load_labeled_roi(csv_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"ROI CSV is empty: {csv_path}")
    if "filename" not in df.columns:
        raise ValueError(f"ROI CSV must include filename column: {csv_path}")
    df = df.copy()
    df["conc_true"] = df["filename"].apply(extract_concentration_from_name)
    df = df.dropna(subset=["conc_true"]).sort_values("conc_true").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No valid concentration labels parsed from: {csv_path}")
    return df, df["conc_true"].to_numpy(dtype=float)


def build_compact_features(df: pd.DataFrame, *, feature_names: Optional[Iterable[str]] = None) -> pd.DataFrame:
    features = build_feature_frame(
        df,
        g_mode="avg",
        use_bg=True,
        clip_min=CLIP_MIN,
        bg_correction="subtract",
    )
    selected = select_feature_set(features, use_bg=True, feature_set="compact")
    if feature_names is not None:
        selected = selected.loc[:, list(feature_names)]
    return selected


def fit_predict_variant(
    variant: Variant,
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    *,
    weight_eps: float,
    out_dir: Path,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    X_train_df = build_compact_features(train_df)
    feature_names = tuple(X_train_df.columns)
    X_train = X_train_df.to_numpy(dtype=float)
    X_test = build_compact_features(test_df, feature_names=feature_names).to_numpy(dtype=float)

    y_target = transform_target(y_train, TARGET_TRANSFORM, TARGET_EPS)
    sample_weight = make_sample_weights(y_train, weight_eps=weight_eps, beta=variant.beta)
    model = train_model_fixed_alpha(X_train, y_target, alpha=variant.alpha, sample_weight=sample_weight)

    train_raw = inverse_target(model.predict(X_train), TARGET_TRANSFORM, TARGET_EPS)
    calibrator = fit_calibrator(train_raw, y_train, sample_weight)
    train_pred = apply_calibrator(train_raw, calibrator)

    loo_pred = np.zeros_like(y_train, dtype=float)
    loo = LeaveOneOut()
    for fold_train_idx, fold_test_idx in loo.split(X_train):
        X_fold_train = X_train[fold_train_idx]
        X_fold_test = X_train[fold_test_idx]
        y_fold_train = y_train[fold_train_idx]
        y_fold_target = transform_target(y_fold_train, TARGET_TRANSFORM, TARGET_EPS)
        w_fold = make_sample_weights(y_fold_train, weight_eps=weight_eps, beta=variant.beta)
        fold_model = train_model_fixed_alpha(X_fold_train, y_fold_target, alpha=variant.alpha, sample_weight=w_fold)
        fold_train_raw = inverse_target(fold_model.predict(X_fold_train), TARGET_TRANSFORM, TARGET_EPS)
        fold_calibrator = fit_calibrator(fold_train_raw, y_fold_train, w_fold)
        fold_test_raw = inverse_target(fold_model.predict(X_fold_test), TARGET_TRANSFORM, TARGET_EPS)
        loo_pred[fold_test_idx] = apply_calibrator(fold_test_raw, fold_calibrator)

    test_raw = inverse_target(model.predict(X_test), TARGET_TRANSFORM, TARGET_EPS)
    test_pred = apply_calibrator(test_raw, calibrator)

    pred_rows = []
    for phase, df, y_true, y_pred in (
        ("train", train_df, y_train, train_pred),
        ("loo", train_df, y_train, loo_pred),
        ("food", test_df, y_test, test_pred),
    ):
        phase_df = pd.DataFrame(
            {
                "scan": variant.scan,
                "variant_id": variant.variant_id,
                "alpha": variant.alpha,
                "beta": variant.beta,
                "phase": phase,
                "filename": df["filename"].to_numpy(),
                "sample_group": df["filename"].apply(group_from_filename).to_numpy(),
                "conc_true": y_true,
                "conc_pred": y_pred,
                "residual": y_pred - y_true,
            }
        )
        pred_rows.append(phase_df)
    predictions = pd.concat(pred_rows, ignore_index=True)

    metric_rows: list[dict[str, object]] = []
    for phase, df in predictions.groupby("phase", sort=False):
        groups = ("all", "chicken", "egg") if phase == "food" else ("all",)
        for group in groups:
            part = df if group == "all" else df[df["sample_group"] == group]
            row = {
                "scan": variant.scan,
                "variant_id": variant.variant_id,
                "alpha": variant.alpha,
                "beta": variant.beta,
                "weight_eps": weight_eps,
                "effective_weight_eps": math.inf if variant.beta == 0 else weight_eps / variant.beta,
                "phase": phase,
                "sample_group": group,
                "n_features": len(feature_names),
                "feature_names": ";".join(feature_names),
            }
            row.update(metrics(part["conc_true"].to_numpy(float), part["conc_pred"].to_numpy(float)))
            metric_rows.append(row)

    variant_dir = out_dir / variant.variant_id
    variant_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(variant_dir / "predictions.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    return metric_rows, predictions


def plot_scan(metrics_df: pd.DataFrame, scan_name: str, x_col: str, out_path: Path) -> None:
    food_all = metrics_df[
        (metrics_df["scan"] == scan_name)
        & (metrics_df["phase"] == "food")
        & (metrics_df["sample_group"] == "all")
    ].copy()
    loo_all = metrics_df[
        (metrics_df["scan"] == scan_name)
        & (metrics_df["phase"] == "loo")
        & (metrics_df["sample_group"] == "all")
    ].copy()
    if food_all.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=150)
    for ax, metric_name, title in zip(axes, ("r2", "rmse", "mae"), ("R2", "RMSE", "MAE")):
        for source, label, color in ((food_all, "food all", "#1f77b4"), (loo_all, "mate20x LOO", "#d62728")):
            if source.empty:
                continue
            source = source.sort_values(x_col)
            x = source[x_col].to_numpy(float)
            y = source[metric_name].to_numpy(float)
            if np.any(x == 0):
                ax.scatter(x[x == 0], y[x == 0], s=42, label=f"{label} (0)", color=color, marker="x")
            mask = x > 0
            ax.plot(x[mask], y[mask], marker="o", linewidth=1.5, label=label, color=color)
        ax.set_xscale("symlog", linthresh=1e-4 if x_col == "beta" else 1.0)
        ax.set_xlabel(x_col)
        ax.set_ylabel(metric_name)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"{scan_name} parameter scan")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_best_food_prediction(predictions: pd.DataFrame, best_variant: str, out_path: Path) -> None:
    df = predictions[(predictions["variant_id"] == best_variant) & (predictions["phase"] == "food")].copy()
    if df.empty:
        return
    colors = {"chicken": "#d62728", "egg": "#ff7f0e", "other": "#777777"}
    fig, ax = plt.subplots(figsize=(6.2, 5.6), dpi=150)
    for group, part in df.groupby("sample_group"):
        ax.scatter(
            part["conc_true"],
            part["conc_pred"],
            s=42,
            alpha=0.85,
            color=colors.get(group, "#777777"),
            label=group,
            edgecolor="white",
            linewidth=0.4,
        )
    low = float(min(df["conc_true"].min(), df["conc_pred"].min()))
    high = float(max(df["conc_true"].max(), df["conc_pred"].max()))
    pad = max((high - low) * 0.04, 1.0)
    ax.plot([low - pad, high + pad], [low - pad, high + pad], "k--", linewidth=1.0)
    row = metrics(df["conc_true"].to_numpy(float), df["conc_pred"].to_numpy(float))
    ax.set_title(f"{best_variant}: food all\nR2={row['r2']:.4g}, RMSE={row['rmse']:.4g}, MAE={row['mae']:.4g}")
    ax.set_xlabel("Ground truth (uM)")
    ax.set_ylabel("Predicted concentration (uM)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan Ridge alpha and beta in sample-weight formula.")
    parser.add_argument("--train-csv", type=Path, default=ROOT / "dataset" / "mate20x" / "roi_detailed_statistics.csv")
    parser.add_argument("--test-csv", type=Path, default=ROOT / "dataset" / "food" / "roi_detailed_statistics.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "parameter_scan" / "OUTPUT")
    parser.add_argument("--weight-eps", type=float, default=WEIGHT_EPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, y_train = load_labeled_roi(args.train_csv)
    test_df, y_test = load_labeled_roi(args.test_csv)

    all_metric_rows: list[dict[str, object]] = []
    all_predictions: list[pd.DataFrame] = []

    alpha_variants = [
        Variant(scan="alpha", variant_id=f"alpha_{alpha:.0e}_beta_1".replace("+", ""), alpha=alpha, beta=1.0)
        for alpha in ALPHA_VALUES
    ]
    for variant in alpha_variants:
        print(f"[alpha scan] {variant.variant_id}: alpha={variant.alpha:g}, beta={variant.beta:g}")
        rows, preds = fit_predict_variant(
            variant,
            train_df,
            y_train,
            test_df,
            y_test,
            weight_eps=float(args.weight_eps),
            out_dir=out_dir,
        )
        all_metric_rows.extend(rows)
        all_predictions.append(preds)

    metrics_df = pd.DataFrame(all_metric_rows)
    alpha_food = metrics_df[
        (metrics_df["scan"] == "alpha")
        & (metrics_df["phase"] == "food")
        & (metrics_df["sample_group"] == "all")
    ].copy()
    best_alpha_row = alpha_food.sort_values(["r2", "rmse"], ascending=[False, True]).iloc[0]
    best_alpha = float(best_alpha_row["alpha"])

    beta_variants = [
        Variant(scan="beta", variant_id=f"beta_{beta:.0e}_alpha_{best_alpha:.0e}".replace("+", ""), alpha=best_alpha, beta=beta)
        for beta in BETA_VALUES
    ]
    for variant in beta_variants:
        print(f"[beta scan] {variant.variant_id}: alpha={variant.alpha:g}, beta={variant.beta:g}")
        rows, preds = fit_predict_variant(
            variant,
            train_df,
            y_train,
            test_df,
            y_test,
            weight_eps=float(args.weight_eps),
            out_dir=out_dir,
        )
        all_metric_rows.extend(rows)
        all_predictions.append(preds)

    weight_eps_variants = [
        Variant(
            scan="weight_eps",
            variant_id=f"weight_eps_{weight_eps:g}_alpha_{best_alpha:.0e}_beta_1".replace("+", ""),
            alpha=best_alpha,
            beta=1.0,
        )
        for weight_eps in WEIGHT_EPS_VALUES
    ]
    for variant, weight_eps in zip(weight_eps_variants, WEIGHT_EPS_VALUES):
        print(f"[weight_eps scan] {variant.variant_id}: alpha={variant.alpha:g}, beta={variant.beta:g}, weight_eps={weight_eps:g}")
        rows, preds = fit_predict_variant(
            variant,
            train_df,
            y_train,
            test_df,
            y_test,
            weight_eps=float(weight_eps),
            out_dir=out_dir,
        )
        all_metric_rows.extend(rows)
        all_predictions.append(preds)

    metrics_df = pd.DataFrame(all_metric_rows)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    metrics_path = out_dir / "regression_parameter_scan_metrics.csv"
    predictions_path = out_dir / "regression_parameter_scan_predictions.csv"
    alpha_rank_path = out_dir / "alpha_food_all_ranked.csv"
    beta_rank_path = out_dir / "beta_food_all_ranked.csv"
    summary_path = out_dir / "summary.txt"

    metrics_df.to_csv(metrics_path, index=False, float_format=CSV_FLOAT_FORMAT)
    predictions_df.to_csv(predictions_path, index=False, float_format=CSV_FLOAT_FORMAT)

    alpha_rank = metrics_df[
        (metrics_df["scan"] == "alpha")
        & (metrics_df["phase"] == "food")
        & (metrics_df["sample_group"] == "all")
    ].sort_values(["r2", "rmse"], ascending=[False, True])
    beta_rank = metrics_df[
        (metrics_df["scan"] == "beta")
        & (metrics_df["phase"] == "food")
        & (metrics_df["sample_group"] == "all")
    ].sort_values(["r2", "rmse"], ascending=[False, True])
    weight_eps_rank = metrics_df[
        (metrics_df["scan"] == "weight_eps")
        & (metrics_df["phase"] == "food")
        & (metrics_df["sample_group"] == "all")
    ].sort_values(["r2", "rmse"], ascending=[False, True])
    alpha_rank.to_csv(alpha_rank_path, index=False, float_format=CSV_FLOAT_FORMAT)
    beta_rank.to_csv(beta_rank_path, index=False, float_format=CSV_FLOAT_FORMAT)
    weight_eps_rank.to_csv(out_dir / "weight_eps_food_all_ranked.csv", index=False, float_format=CSV_FLOAT_FORMAT)

    best_overall = metrics_df[
        (metrics_df["phase"] == "food")
        & (metrics_df["sample_group"] == "all")
    ].sort_values(["r2", "rmse"], ascending=[False, True]).iloc[0]

    plot_scan(metrics_df, "alpha", "alpha", out_dir / "alpha_scan_metrics.png")
    plot_scan(metrics_df, "beta", "beta", out_dir / "beta_scan_metrics.png")
    plot_scan(metrics_df, "weight_eps", "weight_eps", out_dir / "weight_eps_scan_metrics.png")
    plot_best_food_prediction(predictions_df, str(best_overall["variant_id"]), out_dir / "best_food_prediction.png")

    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"Train CSV: {args.train_csv.resolve()}\n")
        f.write(f"Test CSV:  {args.test_csv.resolve()}\n")
        f.write(f"weight formula: w_i = 1 / (beta * C_i + weight_eps), weight_eps={float(args.weight_eps):.6g}\n")
        f.write(f"alpha scan beta fixed at 1; beta scan alpha fixed at best alpha from alpha scan: {best_alpha:.6g}\n")
        f.write(f"weight_eps scan alpha fixed at {best_alpha:.6g}, beta fixed at 1\n\n")
        f.write("[best alpha scan rows: food all]\n")
        f.write(alpha_rank[["variant_id", "alpha", "beta", "r2", "rmse", "mae", "mean_error"]].head(10).to_string(index=False))
        f.write("\n\n[best beta scan rows: food all]\n")
        f.write(beta_rank[["variant_id", "alpha", "beta", "effective_weight_eps", "r2", "rmse", "mae", "mean_error"]].head(10).to_string(index=False))
        f.write("\n\n[best weight_eps scan rows: food all]\n")
        f.write(weight_eps_rank[["variant_id", "alpha", "beta", "weight_eps", "effective_weight_eps", "r2", "rmse", "mae", "mean_error"]].head(10).to_string(index=False))
        f.write("\n\n[best overall: food all]\n")
        f.write(best_overall[["scan", "variant_id", "alpha", "beta", "effective_weight_eps", "r2", "rmse", "mae", "mean_error"]].to_string())
        f.write("\n")

    print(f"Saved metrics -> {metrics_path}")
    print(f"Saved predictions -> {predictions_path}")
    print(f"Saved summary -> {summary_path}")
    print("\nBest alpha scan rows:")
    print(alpha_rank[["variant_id", "alpha", "beta", "r2", "rmse", "mae", "mean_error"]].head(10).to_string(index=False))
    print("\nBest beta scan rows:")
    print(beta_rank[["variant_id", "alpha", "beta", "effective_weight_eps", "r2", "rmse", "mae", "mean_error"]].head(10).to_string(index=False))
    print("\nBest weight_eps scan rows:")
    print(weight_eps_rank[["variant_id", "alpha", "beta", "weight_eps", "effective_weight_eps", "r2", "rmse", "mae", "mean_error"]].head(10).to_string(index=False))
    print("\nBest overall:")
    print(best_overall[["scan", "variant_id", "alpha", "beta", "effective_weight_eps", "r2", "rmse", "mae", "mean_error"]].to_string())


if __name__ == "__main__":
    main()
