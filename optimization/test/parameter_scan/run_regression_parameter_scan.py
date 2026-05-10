from __future__ import annotations

import argparse
import itertools
import math
import re
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPT_ROOT = Path(__file__).resolve().parents[1]
REG_DIR = PROJECT_ROOT / "concentration_regression"
if str(REG_DIR) not in sys.path:
    sys.path.insert(0, str(REG_DIR))

from config import CLIP_MIN, TARGET_EPS, TARGET_TRANSFORM, WEIGHT_EPS  # noqa: E402
from physics_features import build_feature_frame, extract_concentration_from_name  # noqa: E402
from train import apply_calibrator, fit_calibrator, inverse_target, select_feature_set, transform_target  # noqa: E402


CSV_FLOAT_FORMAT = "%.6g"
EPS = 1e-6
BEST_INPUT_VECTOR_CSV = OPT_ROOT / "test_input" / "OUTPUT" / "raw_all_ranked.csv"
RAW_NAME_RE = re.compile(r"^(?P<sample>.+)-(?P<batch>\d+)-(?P<concentration>[-+]?\d+(?:\.\d+)?)$")
BASE_FEATURES = ("ln_R_sig", "ln_G_sig", "ln_B_sig")
OPTIONAL_FEATURES = (
    "Ratio_RG",
    "Ratio_BG",
    "Ratio_RB",
    "Delta_RG",
    "Delta_RB",
    "Delta_GB",
)
ALPHA_VALUES = (0.0, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8)
BETA_VALUES = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
WEIGHT_EPS_VALUES = (1.0, 10.0, 30.0, 50.0, 80.0, 120.0)


@dataclass(frozen=True)
class Variant:
    scan: str
    variant_id: str
    alpha: float
    beta: float


@dataclass(frozen=True)
class HyperParams:
    alpha: float
    beta: float
    weight_eps: float


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
    raw_match = RAW_NAME_RE.fullmatch(stem)
    if raw_match:
        return raw_match.group("sample").lower()
    if stem.startswith("chicken"):
        return "chicken"
    if stem.startswith("egg"):
        return "egg"
    return "other"


def concentration_from_filename(filename: object) -> float | None:
    stem = Path(str(filename)).stem
    raw_match = RAW_NAME_RE.fullmatch(stem)
    if raw_match:
        return float(raw_match.group("concentration"))
    return extract_concentration_from_name(filename)


def load_labeled_roi(csv_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"ROI CSV is empty: {csv_path}")
    if "filename" not in df.columns:
        raise ValueError(f"ROI CSV must include filename column: {csv_path}")
    df = df.copy()
    df["conc_true"] = df["filename"].apply(concentration_from_filename)
    df = df.dropna(subset=["conc_true"]).sort_values("conc_true").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No valid concentration labels parsed from: {csv_path}")
    return df, df["conc_true"].to_numpy(dtype=float)


def load_feature_names(feature_source: Path) -> tuple[str, ...]:
    if not feature_source.exists():
        raise FileNotFoundError(f"Best input-vector ranking not found: {feature_source}")
    ranked = pd.read_csv(feature_source)
    if ranked.empty or "feature_names" not in ranked.columns:
        raise ValueError(f"Input-vector ranking must include feature_names: {feature_source}")
    feature_names = tuple(str(ranked.iloc[0]["feature_names"]).split(";"))
    if not feature_names or any(not feature for feature in feature_names):
        raise ValueError(f"No valid feature names found in: {feature_source}")
    return feature_names


def add_optional_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    feature_df = feature_df.copy()
    ratio_rg = feature_df["Ratio_RG"].to_numpy(dtype=float)
    ratio_bg = feature_df["Ratio_BG"].to_numpy(dtype=float)
    feature_df["Ratio_RB"] = ratio_rg / (ratio_bg + EPS)
    feature_df["Delta_RB"] = (ratio_bg - ratio_rg) / (ratio_bg + ratio_rg + EPS)
    feature_df["Delta_GB"] = (ratio_bg - 1.0) / (ratio_bg + 1.0 + EPS)
    return feature_df


def build_scan_features(df: pd.DataFrame, *, feature_names: Optional[Iterable[str]] = None) -> pd.DataFrame:
    features = build_feature_frame(
        df,
        g_mode="avg",
        use_bg=True,
        clip_min=CLIP_MIN,
        bg_correction="subtract",
    )
    selected = add_optional_features(features)
    if feature_names is None:
        selected = select_feature_set(selected, use_bg=True, feature_set="compact")
    if feature_names is not None:
        names = list(feature_names)
        missing = [name for name in names if name not in selected.columns]
        if missing:
            raise KeyError(f"Requested feature(s) not available: {', '.join(missing)}")
        selected = selected.loc[:, names]
    return selected


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    features = build_feature_frame(
        df,
        g_mode="avg",
        use_bg=True,
        clip_min=CLIP_MIN,
        bg_correction="subtract",
    )
    return add_optional_features(features)


def variant_name(optional_features: tuple[str, ...]) -> str:
    if not optional_features:
        return "ln_rgb_only"
    return "_".join(feature.lower() for feature in optional_features)


def rank_metric_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows).sort_values(["r2", "rmse"], ascending=[False, True]).reset_index(drop=True)


def evaluate_raw_prediction(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    feature_names: tuple[str, ...],
    params: HyperParams,
    *,
    food: str,
    step: str,
    scan: str,
    variant_id: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    X_train = train_features.loc[:, list(feature_names)].to_numpy(dtype=float)
    X_test = test_features.loc[:, list(feature_names)].to_numpy(dtype=float)
    y_target = transform_target(y_train, TARGET_TRANSFORM, TARGET_EPS)
    sample_weight = make_sample_weights(y_train, weight_eps=params.weight_eps, beta=params.beta)
    model = train_model_fixed_alpha(X_train, y_target, alpha=params.alpha, sample_weight=sample_weight)
    train_raw = inverse_target(model.predict(X_train), TARGET_TRANSFORM, TARGET_EPS)
    calibrator = fit_calibrator(train_raw, y_train, sample_weight)
    test_raw = inverse_target(model.predict(X_test), TARGET_TRANSFORM, TARGET_EPS)
    test_pred = apply_calibrator(test_raw, calibrator)

    pred_df = test_df[["filename", "conc_true", "sample_group"]].copy()
    pred_df.insert(0, "food", food)
    pred_df.insert(1, "step", step)
    pred_df.insert(2, "scan", scan)
    pred_df.insert(3, "variant_id", variant_id)
    pred_df["phase"] = "raw"
    pred_df["alpha"] = params.alpha
    pred_df["beta"] = params.beta
    pred_df["weight_eps"] = params.weight_eps
    pred_df["effective_weight_eps"] = math.inf if params.beta == 0 else params.weight_eps / params.beta
    pred_df["feature_names"] = ";".join(feature_names)
    pred_df["n_features"] = len(feature_names)
    pred_df["conc_pred_raw"] = test_raw
    pred_df["conc_pred"] = test_pred
    pred_df["residual"] = pred_df["conc_pred"] - pred_df["conc_true"]

    metric_row: dict[str, object] = {
        "food": food,
        "step": step,
        "scan": scan,
        "variant_id": variant_id,
        "phase": "raw",
        "sample_group": food,
        "alpha": params.alpha,
        "beta": params.beta,
        "weight_eps": params.weight_eps,
        "effective_weight_eps": math.inf if params.beta == 0 else params.weight_eps / params.beta,
        "n_features": len(feature_names),
        "feature_names": ";".join(feature_names),
    }
    for feature in OPTIONAL_FEATURES:
        metric_row[f"use_{feature.lower()}"] = feature in feature_names
    metric_row.update(metrics(y_test, test_pred))
    return metric_row, pred_df


def search_input_vectors_by_prediction(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    params: HyperParams,
    *,
    food: str,
    step: str,
    out_dir: Path,
) -> tuple[tuple[str, ...], pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for mask in itertools.product((False, True), repeat=len(OPTIONAL_FEATURES)):
        optional = tuple(feature for feature, enabled in zip(OPTIONAL_FEATURES, mask) if enabled)
        feature_names = (*BASE_FEATURES, *optional)
        name = variant_name(optional)
        row, pred = evaluate_raw_prediction(
            train_df,
            y_train,
            test_df,
            y_test,
            train_features,
            test_features,
            feature_names,
            params,
            food=food,
            step=step,
            scan="input",
            variant_id=name,
        )
        metric_rows.append(row)
        prediction_rows.append(pred)

    ranked = rank_metric_rows(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    ranked.to_csv(out_dir / f"{step}_input_ranked.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    predictions.to_csv(out_dir / f"{step}_input_predictions.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    return tuple(str(ranked.iloc[0]["feature_names"]).split(";")), ranked.iloc[0].to_frame().T


def scan_hyperparameters_by_prediction(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    feature_names: tuple[str, ...],
    start_params: HyperParams,
    *,
    food: str,
    step: str,
    out_dir: Path,
) -> tuple[HyperParams, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []

    def add_eval(scan: str, variant_id: str, params: HyperParams) -> None:
        row, pred = evaluate_raw_prediction(
            train_df,
            y_train,
            test_df,
            y_test,
            train_features,
            test_features,
            feature_names,
            params,
            food=food,
            step=step,
            scan=scan,
            variant_id=variant_id,
        )
        metric_rows.append(row)
        prediction_rows.append(pred)

    for alpha in ALPHA_VALUES:
        add_eval(
            "alpha",
            f"alpha_{alpha:.0e}_beta_{start_params.beta:g}_weight_eps_{start_params.weight_eps:g}".replace("+", ""),
            HyperParams(alpha=float(alpha), beta=start_params.beta, weight_eps=start_params.weight_eps),
        )
    best_alpha = float(rank_metric_rows([row for row in metric_rows if row["scan"] == "alpha"]).iloc[0]["alpha"])

    for beta in BETA_VALUES:
        add_eval(
            "beta",
            f"beta_{beta:.0e}_alpha_{best_alpha:.0e}_weight_eps_{start_params.weight_eps:g}".replace("+", ""),
            HyperParams(alpha=best_alpha, beta=float(beta), weight_eps=start_params.weight_eps),
        )
    best_beta = float(rank_metric_rows([row for row in metric_rows if row["scan"] == "beta"]).iloc[0]["beta"])

    for weight_eps in WEIGHT_EPS_VALUES:
        add_eval(
            "weight_eps",
            f"weight_eps_{weight_eps:g}_alpha_{best_alpha:.0e}_beta_{best_beta:g}".replace("+", ""),
            HyperParams(alpha=best_alpha, beta=best_beta, weight_eps=float(weight_eps)),
        )
    best_weight_eps = float(rank_metric_rows([row for row in metric_rows if row["scan"] == "weight_eps"]).iloc[0]["weight_eps"])

    final_params = HyperParams(alpha=best_alpha, beta=best_beta, weight_eps=best_weight_eps)
    add_eval(
        "final",
        f"final_alpha_{best_alpha:.0e}_beta_{best_beta:g}_weight_eps_{best_weight_eps:g}".replace("+", ""),
        final_params,
    )

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.concat(prediction_rows, ignore_index=True)
    metrics_df.to_csv(out_dir / f"{step}_hyperparameter_metrics.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    predictions_df.to_csv(out_dir / f"{step}_hyperparameter_predictions.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    for scan in ("alpha", "beta", "weight_eps", "final"):
        rank_metric_rows(metrics_df[metrics_df["scan"] == scan].to_dict("records")).to_csv(
            out_dir / f"{step}_{scan}_ranked.csv",
            index=False,
            float_format=CSV_FLOAT_FORMAT,
        )
    final_rank = rank_metric_rows(metrics_df[metrics_df["scan"] == "final"].to_dict("records"))
    return final_params, final_rank.iloc[0].to_frame().T


def save_food_plot(best_predictions: pd.DataFrame, out_path: Path, title: str) -> None:
    if best_predictions.empty:
        return
    fig, ax = plt.subplots(figsize=(5.8, 5.2), dpi=150)
    ax.scatter(
        best_predictions["conc_true"],
        best_predictions["conc_pred"],
        s=42,
        alpha=0.85,
        color="#1f77b4",
        edgecolor="white",
        linewidth=0.4,
    )
    low = float(min(best_predictions["conc_true"].min(), best_predictions["conc_pred"].min()))
    high = float(max(best_predictions["conc_true"].max(), best_predictions["conc_pred"].max()))
    pad = max((high - low) * 0.05, 1.0)
    ax.plot([low - pad, high + pad], [low - pad, high + pad], "k--", linewidth=1.0)
    row = metrics(best_predictions["conc_true"].to_numpy(float), best_predictions["conc_pred"].to_numpy(float))
    ax.set_title(f"{title}\nR2={row['r2']:.4g}, RMSE={row['rmse']:.4g}, MAE={row['mae']:.4g}")
    ax.set_xlabel("Ground truth (uM)")
    ax.set_ylabel("Predicted concentration (uM)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_foodwise_iterative_prediction(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    test_df: pd.DataFrame,
    *,
    out_dir: Path,
    weight_eps: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_features = build_all_features(train_df)
    test_df = test_df.copy()
    test_df["sample_group"] = test_df["filename"].apply(group_from_filename)
    initial_params = HyperParams(alpha=100.0, beta=1.0, weight_eps=float(weight_eps))
    summary_rows: list[pd.DataFrame] = []

    for food in sorted(test_df["sample_group"].unique()):
        food_dir = out_dir / food
        food_dir.mkdir(parents=True, exist_ok=True)
        food_df = test_df[test_df["sample_group"] == food].copy().sort_values(["conc_true", "filename"]).reset_index(drop=True)
        y_food = food_df["conc_true"].to_numpy(dtype=float)
        food_features = build_all_features(food_df)

        print(f"[{food}] prediction step1 input vector search")
        step1_features, step1_best = search_input_vectors_by_prediction(
            train_df,
            y_train,
            food_df,
            y_food,
            train_features,
            food_features,
            initial_params,
            food=food,
            step="step1",
            out_dir=food_dir,
        )

        print(f"[{food}] prediction step2 hyperparameter search")
        step2_params, step2_best = scan_hyperparameters_by_prediction(
            train_df,
            y_train,
            food_df,
            y_food,
            train_features,
            food_features,
            step1_features,
            initial_params,
            food=food,
            step="step2",
            out_dir=food_dir,
        )

        print(f"[{food}] prediction step3 input vector search")
        step3_features, step3_best = search_input_vectors_by_prediction(
            train_df,
            y_train,
            food_df,
            y_food,
            train_features,
            food_features,
            step2_params,
            food=food,
            step="step3",
            out_dir=food_dir,
        )

        print(f"[{food}] prediction step4 hyperparameter search")
        step4_params, step4_best = scan_hyperparameters_by_prediction(
            train_df,
            y_train,
            food_df,
            y_food,
            train_features,
            food_features,
            step3_features,
            step2_params,
            food=food,
            step="step4",
            out_dir=food_dir,
        )

        final_row, final_predictions = evaluate_raw_prediction(
            train_df,
            y_train,
            food_df,
            y_food,
            train_features,
            food_features,
            step3_features,
            step4_params,
            food=food,
            step="best",
            scan="final",
            variant_id=f"{food}_best",
        )
        final_predictions.to_csv(food_dir / "best_raw_predictions.csv", index=False, float_format=CSV_FLOAT_FORMAT)
        save_food_plot(final_predictions, food_dir / "best_raw_prediction.png", f"{food} best RAW prediction")

        food_summary = pd.concat(
            [
                step1_best.assign(stage="step1_input"),
                step2_best.assign(stage="step2_hyperparameter"),
                step3_best.assign(stage="step3_input"),
                step4_best.assign(stage="step4_hyperparameter"),
                pd.DataFrame([final_row]).assign(stage="best"),
            ],
            ignore_index=True,
        )
        food_summary.to_csv(food_dir / "summary.csv", index=False, float_format=CSV_FLOAT_FORMAT)
        summary_rows.append(pd.DataFrame([final_row]).assign(stage="best"))

        with (food_dir / "summary.txt").open("w", encoding="utf-8") as f:
            f.write(f"Food: {food}\n")
            f.write("Optimization order: input -> hyperparameter -> input -> hyperparameter\n")
            f.write("Selection criterion: RAW prediction R2, tie-broken by RMSE\n")
            f.write(f"Final features: {';'.join(step3_features)}\n")
            f.write(f"Final hyperparameters: alpha={step4_params.alpha:g}, beta={step4_params.beta:g}, weight_eps={step4_params.weight_eps:g}\n\n")
            f.write(food_summary[["stage", "variant_id", "alpha", "beta", "weight_eps", "r2", "rmse", "mae", "feature_names"]].to_string(index=False))
            f.write("\n")

    best_summary = pd.concat(summary_rows, ignore_index=True).sort_values("food").reset_index(drop=True)
    best_summary.to_csv(out_dir / "foodwise_best_summary.csv", index=False, float_format=CSV_FLOAT_FORMAT)


def run_overall_iterative_prediction(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    *,
    out_dir: Path,
    weight_eps: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["sample_group"] = train_df["filename"].apply(group_from_filename)
    test_df["sample_group"] = test_df["filename"].apply(group_from_filename)
    train_features = build_all_features(train_df)
    test_features = build_all_features(test_df)
    initial_params = HyperParams(alpha=100.0, beta=1.0, weight_eps=float(weight_eps))
    label = "all"

    print("[all] prediction step1 input vector search")
    step1_features, step1_best = search_input_vectors_by_prediction(
        train_df,
        y_train,
        test_df,
        y_test,
        train_features,
        test_features,
        initial_params,
        food=label,
        step="step1",
        out_dir=out_dir,
    )

    print("[all] prediction step2 hyperparameter search")
    step2_params, step2_best = scan_hyperparameters_by_prediction(
        train_df,
        y_train,
        test_df,
        y_test,
        train_features,
        test_features,
        step1_features,
        initial_params,
        food=label,
        step="step2",
        out_dir=out_dir,
    )

    print("[all] prediction step3 input vector search")
    step3_features, step3_best = search_input_vectors_by_prediction(
        train_df,
        y_train,
        test_df,
        y_test,
        train_features,
        test_features,
        step2_params,
        food=label,
        step="step3",
        out_dir=out_dir,
    )

    print("[all] prediction step4 hyperparameter search")
    step4_params, step4_best = scan_hyperparameters_by_prediction(
        train_df,
        y_train,
        test_df,
        y_test,
        train_features,
        test_features,
        step3_features,
        step2_params,
        food=label,
        step="step4",
        out_dir=out_dir,
    )

    final_row, final_predictions = evaluate_raw_prediction(
        train_df,
        y_train,
        test_df,
        y_test,
        train_features,
        test_features,
        step3_features,
        step4_params,
        food=label,
        step="best",
        scan="final",
        variant_id="all_best",
    )
    final_predictions.to_csv(out_dir / "best_raw_predictions.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    save_food_plot(final_predictions, out_dir / "best_raw_prediction.png", "all best RAW prediction")

    summary = pd.concat(
        [
            step1_best.assign(stage="step1_input"),
            step2_best.assign(stage="step2_hyperparameter"),
            step3_best.assign(stage="step3_input"),
            step4_best.assign(stage="step4_hyperparameter"),
            pd.DataFrame([final_row]).assign(stage="best"),
        ],
        ignore_index=True,
    )
    summary.to_csv(out_dir / "summary.csv", index=False, float_format=CSV_FLOAT_FORMAT)

    with (out_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("Group: all RAW rows\n")
        f.write("Optimization order: input -> hyperparameter -> input -> hyperparameter\n")
        f.write("Selection criterion: RAW prediction R2, tie-broken by RMSE\n")
        f.write(f"Final features: {';'.join(step3_features)}\n")
        f.write(f"Final hyperparameters: alpha={step4_params.alpha:g}, beta={step4_params.beta:g}, weight_eps={step4_params.weight_eps:g}\n\n")
        f.write(summary[["stage", "variant_id", "alpha", "beta", "weight_eps", "r2", "rmse", "mae", "feature_names"]].to_string(index=False))
        f.write("\n")


def fit_predict_variant(
    variant: Variant,
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    *,
    weight_eps: float,
    feature_names: tuple[str, ...],
    out_dir: Path,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    X_train_df = build_scan_features(train_df, feature_names=feature_names)
    X_train = X_train_df.to_numpy(dtype=float)
    X_test = build_scan_features(test_df, feature_names=feature_names).to_numpy(dtype=float)

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
        ("raw", test_df, y_test, test_pred),
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
        groups = ("all", *sorted(df["sample_group"].unique())) if phase == "raw" else ("all",)
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
    raw_all = metrics_df[
        (metrics_df["scan"] == scan_name)
        & (metrics_df["phase"] == "raw")
        & (metrics_df["sample_group"] == "all")
    ].copy()
    loo_all = metrics_df[
        (metrics_df["scan"] == scan_name)
        & (metrics_df["phase"] == "loo")
        & (metrics_df["sample_group"] == "all")
    ].copy()
    if raw_all.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=150)
    for ax, metric_name, title in zip(axes, ("r2", "rmse", "mae"), ("R2", "RMSE", "MAE")):
        for source, label, color in ((raw_all, "RAW all", "#1f77b4"), (loo_all, "train LOO", "#d62728")):
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


def plot_best_raw_prediction(predictions: pd.DataFrame, best_variant: str, out_path: Path) -> None:
    df = predictions[(predictions["variant_id"] == best_variant) & (predictions["phase"] == "raw")].copy()
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
    ax.set_title(f"{best_variant}: RAW all\nR2={row['r2']:.4g}, RMSE={row['rmse']:.4g}, MAE={row['mae']:.4g}")
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
    parser.add_argument("--train-csv", type=Path, default=PROJECT_ROOT / "dataset" / "RAW" / "roi_results" / "roi_detailed_statistics.csv")
    parser.add_argument("--test-csv", type=Path, default=PROJECT_ROOT / "dataset" / "RAW" / "roi_results" / "roi_detailed_statistics.csv")
    parser.add_argument("--out-dir", type=Path, default=OPT_ROOT / "parameter_scan" / "OUTPUT")
    parser.add_argument("--weight-eps", type=float, default=WEIGHT_EPS)
    parser.add_argument(
        "--feature-source",
        type=Path,
        default=BEST_INPUT_VECTOR_CSV,
        help="CSV ranked by test_input; first row feature_names is used for parameter scan.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, y_train = load_labeled_roi(args.train_csv)
    test_df, y_test = load_labeled_roi(args.test_csv)
    feature_names = load_feature_names(args.feature_source)

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
            feature_names=feature_names,
            out_dir=out_dir,
        )
        all_metric_rows.extend(rows)
        all_predictions.append(preds)

    metrics_df = pd.DataFrame(all_metric_rows)
    alpha_raw = metrics_df[
        (metrics_df["scan"] == "alpha")
        & (metrics_df["phase"] == "raw")
        & (metrics_df["sample_group"] == "all")
    ].copy()
    best_alpha_row = alpha_raw.sort_values(["r2", "rmse"], ascending=[False, True]).iloc[0]
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
            feature_names=feature_names,
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
            feature_names=feature_names,
            out_dir=out_dir,
        )
        all_metric_rows.extend(rows)
        all_predictions.append(preds)

    metrics_df = pd.DataFrame(all_metric_rows)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    metrics_path = out_dir / "regression_parameter_scan_metrics.csv"
    predictions_path = out_dir / "regression_parameter_scan_predictions.csv"
    alpha_rank_path = out_dir / "alpha_raw_all_ranked.csv"
    beta_rank_path = out_dir / "beta_raw_all_ranked.csv"
    summary_path = out_dir / "summary.txt"

    metrics_df.to_csv(metrics_path, index=False, float_format=CSV_FLOAT_FORMAT)
    predictions_df.to_csv(predictions_path, index=False, float_format=CSV_FLOAT_FORMAT)

    alpha_rank = metrics_df[
        (metrics_df["scan"] == "alpha")
        & (metrics_df["phase"] == "raw")
        & (metrics_df["sample_group"] == "all")
    ].sort_values(["r2", "rmse"], ascending=[False, True])
    beta_rank = metrics_df[
        (metrics_df["scan"] == "beta")
        & (metrics_df["phase"] == "raw")
        & (metrics_df["sample_group"] == "all")
    ].sort_values(["r2", "rmse"], ascending=[False, True])
    weight_eps_rank = metrics_df[
        (metrics_df["scan"] == "weight_eps")
        & (metrics_df["phase"] == "raw")
        & (metrics_df["sample_group"] == "all")
    ].sort_values(["r2", "rmse"], ascending=[False, True])
    alpha_rank.to_csv(alpha_rank_path, index=False, float_format=CSV_FLOAT_FORMAT)
    beta_rank.to_csv(beta_rank_path, index=False, float_format=CSV_FLOAT_FORMAT)
    weight_eps_rank.to_csv(out_dir / "weight_eps_raw_all_ranked.csv", index=False, float_format=CSV_FLOAT_FORMAT)

    best_overall = metrics_df[
        (metrics_df["phase"] == "raw")
        & (metrics_df["sample_group"] == "all")
    ].sort_values(["r2", "rmse"], ascending=[False, True]).iloc[0]

    plot_scan(metrics_df, "alpha", "alpha", out_dir / "alpha_scan_metrics.png")
    plot_scan(metrics_df, "beta", "beta", out_dir / "beta_scan_metrics.png")
    plot_scan(metrics_df, "weight_eps", "weight_eps", out_dir / "weight_eps_scan_metrics.png")
    plot_best_raw_prediction(predictions_df, str(best_overall["variant_id"]), out_dir / "best_raw_prediction.png")

    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"Train CSV: {args.train_csv.resolve()}\n")
        f.write(f"Test CSV:  {args.test_csv.resolve()}\n")
        f.write(f"Feature source: {args.feature_source.resolve()}\n")
        f.write(f"Feature names: {';'.join(feature_names)}\n")
        f.write(f"weight formula: w_i = 1 / (beta * C_i + weight_eps), weight_eps={float(args.weight_eps):.6g}\n")
        f.write(f"alpha scan beta fixed at 1; beta scan alpha fixed at best alpha from alpha scan: {best_alpha:.6g}\n")
        f.write(f"weight_eps scan alpha fixed at {best_alpha:.6g}, beta fixed at 1\n\n")
        f.write("[best alpha scan rows: RAW all]\n")
        f.write(alpha_rank[["variant_id", "alpha", "beta", "r2", "rmse", "mae", "mean_error"]].head(10).to_string(index=False))
        f.write("\n\n[best beta scan rows: RAW all]\n")
        f.write(beta_rank[["variant_id", "alpha", "beta", "effective_weight_eps", "r2", "rmse", "mae", "mean_error"]].head(10).to_string(index=False))
        f.write("\n\n[best weight_eps scan rows: RAW all]\n")
        f.write(weight_eps_rank[["variant_id", "alpha", "beta", "weight_eps", "effective_weight_eps", "r2", "rmse", "mae", "mean_error"]].head(10).to_string(index=False))
        f.write("\n\n[best overall: RAW all]\n")
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

    run_overall_iterative_prediction(
        train_df,
        y_train,
        test_df,
        y_test,
        out_dir=out_dir / "iterative_all",
        weight_eps=float(args.weight_eps),
    )


if __name__ == "__main__":
    main()
