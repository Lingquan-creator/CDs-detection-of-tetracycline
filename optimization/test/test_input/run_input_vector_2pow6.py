from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OPT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.environ.get("OUT_DIR", OPT_ROOT / "test_input" / "OUTPUT" / "rgb_ln")).resolve()
TRAIN_CSV = Path(
    os.environ.get("TRAIN_CSV", PROJECT_ROOT / "dataset" / "mate20x" / "roi_detailed_statistics.csv")
).resolve()
TEST_CSV = Path(
    os.environ.get("TEST_CSV", PROJECT_ROOT / "dataset" / "food_1" / "roi_detailed_statistics.csv")
).resolve()

RAW_NAME_RE = re.compile(r"^(?P<sample>.+)-(?P<batch>\d+)-(?P<concentration>[-+]?\d+(?:\.\d+)?)$")
TRAILING_CONCENTRATION_RE = re.compile(
    r"^(?P<sample>.*?)(?P<concentration>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")

sys.path.insert(0, str(PROJECT_ROOT / "concentration_regression"))
from config import ALPHAS, CLIP_MIN, TARGET_EPS, WEIGHT_EPS  # noqa: E402
from physics_features import build_feature_frame  # noqa: E402


BG_CORRECTION = "subtract"
EPS = 1e-6

RGB_VECTOR_CANDIDATES: dict[str, tuple[str, ...]] = {
    "rgb_raw": ("R_sig", "G_sig", "B_sig"),
    "rgb_sum_ratio": ("RGB_sum", "R_chroma", "B_chroma"),
}

LN_VECTOR_CANDIDATES: dict[str, tuple[str, ...]] = {
    "ln_rgb": ("ln_R_sig", "ln_G_sig", "ln_B_sig"),
    "ln_total": ("ln_RGB_sum",),
    "ln_total_contrast": ("ln_RGB_sum", "ln_R_over_G", "ln_B_over_G"),
    "ln_pair_sum": ("ln_RG_sum", "ln_RB_sum", "ln_GB_sum"),
    "ln_total_pair_contrast": ("ln_RGB_sum", "ln_RG_over_GB", "ln_RB_over_GB"),
}

DIRECT_VECTOR_CANDIDATES: dict[str, tuple[str, ...]] = {
    "ln_rgb_total_delta_ratio": (
        "ln_R_sig",
        "ln_G_sig",
        "ln_B_sig",
        "ln_RGB_sum",
        "Delta_RG",
        "Delta_RB",
        "Ratio_RG",
    ),
}


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


def group_from_filename(filename: object) -> str:
    stem = Path(str(filename)).stem.lower()
    raw_match = RAW_NAME_RE.fullmatch(stem)
    if raw_match:
        return raw_match.group("sample").lower()
    trailing_match = TRAILING_CONCENTRATION_RE.fullmatch(stem)
    if trailing_match and trailing_match.group("sample"):
        return trailing_match.group("sample").lower()
    if stem.startswith("chicken"):
        return "chicken"
    if stem.startswith("egg"):
        return "egg"
    return "other"


def load_dataframe(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"ROI CSV not found: {path}")

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
            ("regressor", RidgeCV(alphas=list(ALPHAS), scoring="neg_mean_squared_error")),
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


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if y_true.size == 0:
        return {
            "n": 0,
            "r2": math.nan,
            "rmse": math.nan,
            "mae": math.nan,
            "mean_error": math.nan,
            "median_abs_error": math.nan,
            "max_abs_error": math.nan,
        }

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


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    feature_df = build_feature_frame(
        df,
        g_mode="avg",
        use_bg=True,
        clip_min=CLIP_MIN,
        bg_correction=BG_CORRECTION,
    ).copy()

    r = np.maximum(np.exp(feature_df["ln_R_sig"].to_numpy(dtype=float)) - EPS, EPS)
    g = np.maximum(np.exp(feature_df["ln_G_sig"].to_numpy(dtype=float)) - EPS, EPS)
    b = np.maximum(np.exp(feature_df["ln_B_sig"].to_numpy(dtype=float)) - EPS, EPS)
    rgb_sum = np.maximum(r + g + b, EPS)
    rg_sum = np.maximum(r + g, EPS)
    rb_sum = np.maximum(r + b, EPS)
    gb_sum = np.maximum(g + b, EPS)

    feature_df["R_sig"] = r
    feature_df["G_sig"] = g
    feature_df["B_sig"] = b
    feature_df["RGB_sum"] = rgb_sum
    feature_df["R_chroma"] = r / rgb_sum
    feature_df["B_chroma"] = b / rgb_sum
    feature_df["Ratio_RG"] = r / (g + EPS)
    feature_df["Delta_RG"] = (g - r) / (g + r + EPS)
    feature_df["Delta_RB"] = (b - r) / (b + r + EPS)
    feature_df["ln_RGB_sum"] = np.log(rgb_sum + EPS)
    feature_df["ln_R_over_G"] = feature_df["ln_R_sig"] - feature_df["ln_G_sig"]
    feature_df["ln_B_over_G"] = feature_df["ln_B_sig"] - feature_df["ln_G_sig"]
    feature_df["ln_RG_sum"] = np.log(rg_sum + EPS)
    feature_df["ln_RB_sum"] = np.log(rb_sum + EPS)
    feature_df["ln_GB_sum"] = np.log(gb_sum + EPS)
    feature_df["ln_RG_over_GB"] = feature_df["ln_RG_sum"] - feature_df["ln_GB_sum"]
    feature_df["ln_RB_over_GB"] = feature_df["ln_RB_sum"] - feature_df["ln_GB_sum"]
    return feature_df


def candidate_variants() -> list[dict[str, object]]:
    variants = []
    for rgb_name, rgb_features in RGB_VECTOR_CANDIDATES.items():
        for ln_name, ln_features in LN_VECTOR_CANDIDATES.items():
            feature_names = (*rgb_features, *ln_features)
            variants.append(
                {
                    "variant_id": f"{rgb_name}__{ln_name}",
                    "rgb_vector": rgb_name,
                    "ln_vector": ln_name,
                    "feature_names": feature_names,
                }
            )
    for vector_name, feature_names in DIRECT_VECTOR_CANDIDATES.items():
        variants.append(
            {
                "variant_id": vector_name,
                "rgb_vector": "none",
                "ln_vector": vector_name,
                "feature_names": feature_names,
            }
        )
    return variants


def save_top_plot(metrics_df: pd.DataFrame, phase: str, out_path: Path) -> None:
    phase_all = metrics_df[(metrics_df["phase"] == phase) & (metrics_df["sample_group"] == "all")]
    top = phase_all.sort_values("r2", ascending=False).head(20)
    labels = top["variant_id"].str.replace("__", "\n+\n").str.replace("_", " ")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    colors = ["#2ca02c" if idx == 0 else "#1f77b4" for idx in range(len(top))]
    for ax, metric_name, title in zip(axes, ("r2", "rmse", "mae"), ("R2", "RMSE", "MAE")):
        ax.bar(labels, top[metric_name], color=colors)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelsize=7, rotation=45)
    fig.suptitle(f"Top input-vector combinations by {phase} validation")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_prediction_plot(best_df: pd.DataFrame, variant_id: str, phase: str, out_path: Path) -> None:
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
    ax.set_title(f"{phase}: {variant_id}\nR2={row['r2']:.4g}, RMSE={row['rmse']:.4g}, MAE={row['mae']:.4g}")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def append_phase_rows(
    metric_rows: list[dict[str, object]],
    prediction_rows: list[pd.DataFrame],
    *,
    variant: dict[str, object],
    phase: str,
    source_df: pd.DataFrame,
    pred: np.ndarray,
    pred_raw: np.ndarray,
    grouped: bool,
) -> None:
    feature_names = tuple(variant["feature_names"])
    out = source_df[["filename", "conc_true", "sample_group"]].copy()
    out.insert(0, "variant_id", str(variant["variant_id"]))
    out["rgb_vector"] = str(variant["rgb_vector"])
    out["ln_vector"] = str(variant["ln_vector"])
    out["phase"] = phase
    out["feature_names"] = ";".join(feature_names)
    out["n_features"] = len(feature_names)
    out["conc_pred_raw"] = pred_raw
    out["conc_pred"] = pred
    out["residual"] = out["conc_pred"] - out["conc_true"]
    prediction_rows.append(out)

    groups = ["all", *sorted(out["sample_group"].unique())] if grouped else ["all"]
    for group in groups:
        part = out if group == "all" else out[out["sample_group"] == group]
        row: dict[str, object] = {
            "variant_id": str(variant["variant_id"]),
            "rgb_vector": str(variant["rgb_vector"]),
            "ln_vector": str(variant["ln_vector"]),
            "phase": phase,
            "sample_group": group,
            "feature_names": ";".join(feature_names),
            "n_features": len(feature_names),
        }
        row.update(metrics(part["conc_true"].to_numpy(float), part["conc_pred"].to_numpy(float)))
        metric_rows.append(row)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df, y_train = load_dataframe(TRAIN_CSV)
    test_df, y_test = load_dataframe(TEST_CSV)
    train_features = build_feature_table(train_df)
    test_features = build_feature_table(test_df)

    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []

    print(f"Train CSV: {TRAIN_CSV} (n={len(train_df)})")
    print(f"Test CSV:  {TEST_CSV} (n={len(test_df)})")
    print(f"Output:    {OUT_DIR}")

    for variant in candidate_variants():
        feature_names = tuple(variant["feature_names"])
        missing_train = [name for name in feature_names if name not in train_features.columns]
        missing_test = [name for name in feature_names if name not in test_features.columns]
        if missing_train or missing_test:
            raise ValueError(
                f"{variant['variant_id']} missing features: train={missing_train}, test={missing_test}"
            )

        X_train = train_features.loc[:, list(feature_names)].to_numpy(dtype=float)
        X_test = test_features.loc[:, list(feature_names)].to_numpy(dtype=float)
        model, train_pred_raw, train_pred, calibrator = train_model(X_train, y_train)
        test_pred_raw = inverse_target(model.predict(X_test))
        test_pred = apply_calibrator(test_pred_raw, calibrator)

        append_phase_rows(
            metric_rows,
            prediction_rows,
            variant=variant,
            phase="train",
            source_df=train_df,
            pred=train_pred,
            pred_raw=train_pred_raw,
            grouped=False,
        )
        append_phase_rows(
            metric_rows,
            prediction_rows,
            variant=variant,
            phase="test",
            source_df=test_df,
            pred=test_pred,
            pred_raw=test_pred_raw,
            grouped=True,
        )

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.concat(prediction_rows, ignore_index=True)
    metrics_df.to_csv(OUT_DIR / "rgb_ln_input_vector_metrics.csv", index=False, float_format="%.6g")
    predictions_df.to_csv(OUT_DIR / "rgb_ln_input_vector_predictions.csv", index=False, float_format="%.6g")

    test_ranked = (
        metrics_df[(metrics_df["phase"] == "test") & (metrics_df["sample_group"] == "all")]
        .sort_values(["r2", "rmse"], ascending=[False, True])
        .reset_index(drop=True)
    )
    test_ranked.to_csv(OUT_DIR / "test_all_ranked.csv", index=False, float_format="%.6g")

    best_test_variant = str(test_ranked.iloc[0]["variant_id"])
    best_test = predictions_df[(predictions_df["phase"] == "test") & (predictions_df["variant_id"] == best_test_variant)]
    best_test.to_csv(OUT_DIR / "best_test_predictions.csv", index=False, float_format="%.6g")

    save_top_plot(metrics_df, "test", OUT_DIR / "top20_test_metric_comparison.png")
    save_prediction_plot(best_test, best_test_variant, "test", OUT_DIR / "best_test_prediction.png")

    foodwise_dir = OUT_DIR / "foodwise"
    for food in sorted(metrics_df.loc[metrics_df["phase"] == "test", "sample_group"].unique()):
        if food == "all":
            continue
        food_dir = foodwise_dir / food
        food_dir.mkdir(parents=True, exist_ok=True)
        food_ranked = (
            metrics_df[(metrics_df["phase"] == "test") & (metrics_df["sample_group"] == food)]
            .sort_values(["r2", "rmse"], ascending=[False, True])
            .reset_index(drop=True)
        )
        food_ranked.to_csv(food_dir / "test_ranked.csv", index=False, float_format="%.6g")
        food_best_variant = str(food_ranked.iloc[0]["variant_id"])
        food_best = predictions_df[
            (predictions_df["phase"] == "test")
            & (predictions_df["sample_group"] == food)
            & (predictions_df["variant_id"] == food_best_variant)
        ]
        food_best.to_csv(food_dir / "best_test_predictions.csv", index=False, float_format="%.6g")
        save_prediction_plot(food_best, food_best_variant, f"test/{food}", food_dir / "best_test_prediction.png")

    with (OUT_DIR / "rgb_ln_input_vector_summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"Train CSV: {TRAIN_CSV}\n")
        f.write(f"Test CSV: {TEST_CSV}\n")
        f.write("Correction: ROI-BG; model: RidgeCV + log target + isotonic calibration\n\n")
        f.write("RGB vector candidates:\n")
        for name, features in RGB_VECTOR_CANDIDATES.items():
            f.write(f"  {name}: {', '.join(features)}\n")
        f.write("\nln vector candidates:\n")
        for name, features in LN_VECTOR_CANDIDATES.items():
            f.write(f"  {name}: {', '.join(features)}\n")
        f.write("\ndirect vector candidates:\n")
        for name, features in DIRECT_VECTOR_CANDIDATES.items():
            f.write(f"  {name}: {', '.join(features)}\n")

        f.write("\n[test all]\n")
        for _, row in test_ranked.head(20).iterrows():
            f.write(
                f"  {row['variant_id']}: n_features={int(row['n_features'])}, "
                f"R2={row['r2']:.6g}, RMSE={row['rmse']:.6g}, MAE={row['mae']:.6g}, "
                f"mean_error={row['mean_error']:.6g}, features={row['feature_names']}\n"
            )

    print("\nBest by test set:")
    print(test_ranked[["variant_id", "n_features", "r2", "rmse", "mae", "mean_error", "feature_names"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
