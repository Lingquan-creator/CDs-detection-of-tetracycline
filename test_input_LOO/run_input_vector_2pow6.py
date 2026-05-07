from __future__ import annotations

import itertools
import math
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "test_input_LOO" / "OUTPUT"
TRAIN_CSV = ROOT / "dataset" / "mate20x" / "roi_detailed_statistics.csv"
TEST_CSV = ROOT / "dataset" / "food" / "roi_detailed_statistics.csv"
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")

sys.path.insert(0, str(ROOT / "concentration_regression"))
from config import ALPHAS, CLIP_MIN, TARGET_EPS, WEIGHT_EPS  # noqa: E402
from physics_features import build_feature_frame  # noqa: E402


BASE_FEATURES = ("ln_R_sig", "ln_G_sig", "ln_B_sig")
OPTIONAL_FEATURES = (
    "Ratio_RG",
    "Ratio_BG",
    "Ratio_RB",
    "Delta_RG",
    "Delta_RB",
    "Delta_GB",
)
BG_CORRECTION = "subtract"
TARGET_TRANSFORM = "log"
EPS = 1e-6


def concentration_from_filename(filename: object) -> float:
    stem = Path(str(filename)).stem
    match = NUMBER_RE.search(stem)
    return float(match.group(0)) if match else math.nan


def group_from_filename(filename: object) -> str:
    stem = Path(str(filename)).stem.lower()
    if stem.startswith("chicken"):
        return "chicken"
    if stem.startswith("egg"):
        return "egg"
    return "other"


def load_dataframe(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(path)
    df = df.copy()
    df["conc_true"] = df["filename"].apply(concentration_from_filename)
    df["sample_group"] = df["filename"].apply(group_from_filename)
    df = df.dropna(subset=["conc_true"]).sort_values(["conc_true", "filename"]).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No valid labels parsed from {path}")
    return df, df["conc_true"].to_numpy(dtype=float)


def transform_target(y: np.ndarray) -> np.ndarray:
    return np.log(y + TARGET_EPS)


def inverse_target(y_target: np.ndarray) -> np.ndarray:
    return np.maximum(np.exp(y_target) - TARGET_EPS, 0.0)


def make_sample_weights(y: np.ndarray) -> np.ndarray:
    weights = 1.0 / (y + WEIGHT_EPS)
    return weights / np.mean(weights)


def apply_calibrator(pred_raw: np.ndarray, calibrator: IsotonicRegression | None) -> np.ndarray:
    if calibrator is None:
        return np.maximum(np.asarray(pred_raw, dtype=float), 0.0)
    return np.maximum(calibrator.predict(np.asarray(pred_raw, dtype=float)), 0.0)


def train_model(X: np.ndarray, y_true: np.ndarray) -> tuple[Pipeline, np.ndarray, np.ndarray, IsotonicRegression | None]:
    y_target = transform_target(y_true)
    sample_weight = make_sample_weights(y_true)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", RidgeCV(alphas=tuple(ALPHAS), scoring="neg_mean_squared_error")),
        ]
    )
    model.fit(X, y_target, regressor__sample_weight=sample_weight)

    pred_raw = inverse_target(model.predict(X))
    calibrator: IsotonicRegression | None = None
    if pred_raw.size >= 2 and float(np.max(pred_raw) - np.min(pred_raw)) > 1e-12:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(pred_raw, y_true, sample_weight=sample_weight)
    pred = apply_calibrator(pred_raw, calibrator)
    return model, pred_raw, pred, calibrator


def run_loo(X: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    y_pred = np.zeros_like(y_true, dtype=float)
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X):
        model, _, _, calibrator = train_model(X[train_idx], y_true[train_idx])
        raw = inverse_target(model.predict(X[test_idx]))
        y_pred[test_idx] = apply_calibrator(raw, calibrator)
    return y_pred


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    residual = y_pred - y_true
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "n": int(valid.sum()),
        "r2": math.nan if ss_tot <= 0 else 1.0 - ss_res / ss_tot,
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mean_error": float(np.mean(residual)),
        "median_abs_error": float(np.median(np.abs(residual))),
        "max_abs_error": float(np.max(np.abs(residual))),
    }


def add_optional_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    feature_df = feature_df.copy()
    ratio_rg = feature_df["Ratio_RG"].to_numpy(dtype=float)
    ratio_bg = feature_df["Ratio_BG"].to_numpy(dtype=float)
    ratio_rb = ratio_rg / (ratio_bg + EPS)
    feature_df["Ratio_RB"] = ratio_rb
    feature_df["Delta_RB"] = (ratio_bg - ratio_rg) / (ratio_bg + ratio_rg + EPS)
    feature_df["Delta_GB"] = (ratio_bg - 1.0) / (ratio_bg + 1.0 + EPS)
    return feature_df


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    feature_df = build_feature_frame(
        df,
        g_mode="avg",
        use_bg=True,
        clip_min=CLIP_MIN,
        bg_correction=BG_CORRECTION,
    )
    return add_optional_features(feature_df)


def variant_name(features: tuple[str, ...]) -> str:
    if not features:
        return "ln_rgb_only"
    return "_".join(feature.lower() for feature in features)


def save_top_plot(metrics_df: pd.DataFrame) -> None:
    loo_all = metrics_df[(metrics_df["phase"] == "loo") & (metrics_df["sample_group"] == "all")]
    top = loo_all.sort_values(["r2", "rmse"], ascending=[False, True]).head(20)
    labels = top["variant_id"].str.replace("_", "\n")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    colors = ["#2ca02c" if idx == 0 else "#1f77b4" for idx in range(len(top))]
    for ax, metric_name, title in zip(axes, ("r2", "rmse", "mae"), ("R2", "RMSE", "MAE")):
        ax.bar(labels, top[metric_name], color=colors)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelsize=7)
    fig.suptitle("Top 20 input-vector combinations by leave-one-out validation")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "top20_metric_comparison.png", bbox_inches="tight")
    plt.close(fig)


def save_best_prediction_plot(best_df: pd.DataFrame, variant_id: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.4), dpi=150)
    colors = {"chicken": "#d62728", "egg": "#ff7f0e", "other": "#777777"}
    for group, part in best_df.groupby("sample_group"):
        ax.scatter(
            part["conc_true"],
            part["conc_pred"],
            s=38,
            alpha=0.85,
            color=colors.get(group, "#777777"),
            edgecolor="white",
            linewidth=0.4,
            label=group,
        )
    low = float(min(best_df["conc_true"].min(), best_df["conc_pred"].min()))
    high = float(max(best_df["conc_true"].max(), best_df["conc_pred"].max()))
    pad = max((high - low) * 0.05, 1.0)
    low -= pad
    high += pad
    ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1.0)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Ground truth (uM)")
    ax.set_ylabel("Predicted concentration (uM)")
    row = metrics(best_df["conc_true"].to_numpy(float), best_df["conc_pred"].to_numpy(float))
    ax.set_title(f"{variant_id}\nR2={row['r2']:.4g}, RMSE={row['rmse']:.4g}, MAE={row['mae']:.4g}")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "best_loo_prediction.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df, y_train = load_dataframe(TRAIN_CSV)
    test_df, y_test = load_dataframe(TEST_CSV)
    train_features = build_feature_table(train_df)
    test_features = build_feature_table(test_df)

    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []

    for mask in itertools.product((False, True), repeat=len(OPTIONAL_FEATURES)):
        optional = tuple(feature for feature, enabled in zip(OPTIONAL_FEATURES, mask) if enabled)
        feature_names = (*BASE_FEATURES, *optional)
        variant_id = variant_name(optional)

        X_train = train_features.loc[:, list(feature_names)].to_numpy(dtype=float)
        X_test = test_features.loc[:, list(feature_names)].to_numpy(dtype=float)
        model, train_pred_raw, train_pred, calibrator = train_model(X_train, y_train)
        loo_pred = run_loo(X_train, y_train)
        test_pred_raw = inverse_target(model.predict(X_test))
        test_pred = apply_calibrator(test_pred_raw, calibrator)

        for phase, source_df, pred, pred_raw in (
            ("train", train_df, train_pred, train_pred_raw),
            ("loo", train_df, loo_pred, np.full_like(loo_pred, np.nan, dtype=float)),
            ("food", test_df, test_pred, test_pred_raw),
        ):
            out = source_df[["filename", "conc_true", "sample_group"]].copy()
            out.insert(0, "variant_id", variant_id)
            out["phase"] = phase
            out["feature_names"] = ";".join(feature_names)
            out["n_features"] = len(feature_names)
            out["conc_pred_raw"] = pred_raw
            out["conc_pred"] = pred
            out["residual"] = out["conc_pred"] - out["conc_true"]
            prediction_rows.append(out)

            groups = ["all"] if phase != "food" else ["all", "chicken", "egg"]
            for group in groups:
                part = out if group == "all" else out[out["sample_group"] == group]
                row: dict[str, object] = {
                    "variant_id": variant_id,
                    "phase": phase,
                    "sample_group": group,
                    "feature_names": ";".join(feature_names),
                    "n_features": len(feature_names),
                }
                for feature in OPTIONAL_FEATURES:
                    row[f"use_{feature.lower()}"] = feature in optional
                row.update(metrics(part["conc_true"].to_numpy(float), part["conc_pred"].to_numpy(float)))
                metric_rows.append(row)

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.concat(prediction_rows, ignore_index=True)
    metrics_df.to_csv(OUT_DIR / "input_vector_2pow6_metrics.csv", index=False, float_format="%.6g")
    predictions_df.to_csv(OUT_DIR / "input_vector_2pow6_predictions.csv", index=False, float_format="%.6g")

    loo_all = metrics_df[(metrics_df["phase"] == "loo") & (metrics_df["sample_group"] == "all")]
    ranked = loo_all.sort_values(["r2", "rmse"], ascending=[False, True]).reset_index(drop=True)
    ranked.to_csv(OUT_DIR / "loo_all_ranked.csv", index=False, float_format="%.6g")

    best_variant = str(ranked.iloc[0]["variant_id"])
    best_loo = predictions_df[(predictions_df["phase"] == "loo") & (predictions_df["variant_id"] == best_variant)]
    best_loo.to_csv(OUT_DIR / "best_loo_predictions.csv", index=False, float_format="%.6g")
    save_top_plot(metrics_df)
    save_best_prediction_plot(best_loo, best_variant)

    with (OUT_DIR / "input_vector_2pow6_summary.txt").open("w", encoding="utf-8") as f:
        f.write("Base features: ln_R_sig, ln_G_sig, ln_B_sig\n")
        f.write("Optional features: Ratio_RG, Ratio_BG, Ratio_RB, Delta_RG, Delta_RB, Delta_GB\n")
        f.write("Correction: ROI-BG; model: RidgeCV + log target + isotonic calibration\n\n")
        f.write("Model selection criterion: leave-one-out validation on the training set (phase=loo, sample_group=all)\n\n")
        for phase in ("train", "loo", "food"):
            f.write(f"[{phase}]\n")
            phase_df = metrics_df[(metrics_df["phase"] == phase) & (metrics_df["sample_group"] == "all")]
            for _, row in phase_df.sort_values("r2", ascending=False).head(20).iterrows():
                f.write(
                    f"  {row['variant_id']}: n_features={int(row['n_features'])}, "
                    f"R2={row['r2']:.6g}, RMSE={row['rmse']:.6g}, MAE={row['mae']:.6g}, "
                    f"mean_error={row['mean_error']:.6g}, features={row['feature_names']}\n"
                )
            f.write("\n")

    print(f"Output directory: {OUT_DIR}")
    print(ranked[["variant_id", "n_features", "r2", "rmse", "mae", "mean_error", "feature_names"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
