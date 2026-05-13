from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = SCRIPT_DIR / "OUTPUT"
CSV_FLOAT_FORMAT = "%.6g"
TARGET_EPS = 1e-6

sys.path.insert(0, str(PROJECT_ROOT / "concentration_regression"))
from physics_features import build_feature_frame  # noqa: E402


FEATURE_POOL: tuple[str, ...] = (
    "ln_R_sig",
    "ln_G_sig",
    "ln_B_sig",
    "ln_RGB_sum",
    "ln_RG_sum",
    "ln_RB_sum",
    "ln_GB_sum",
    "ln_R_over_G",
    "ln_B_over_G",
    "ln_RG_over_GB",
    "ln_RB_over_GB",
    "R_sig",
    "G_sig",
    "B_sig",
    "RGB_sum",
    "R_chroma",
    "G_chroma",
    "B_chroma",
    "Ratio_RG",
    "Ratio_BG",
    "Ratio_RB",
    "Delta_RG",
    "Delta_RB",
    "Delta_GB",
)

PRESET_INPUT_VECTORS: dict[str, tuple[str, ...]] = {
    "ln_rgb": ("ln_R_sig", "ln_G_sig", "ln_B_sig"),
    "ln_rgb_total": ("ln_R_sig", "ln_G_sig", "ln_B_sig", "ln_RGB_sum"),
    "ln_rgb_pair_sum": ("ln_R_sig", "ln_G_sig", "ln_B_sig", "ln_RG_sum", "ln_RB_sum", "ln_GB_sum"),
    "ln_total_pair_sum": ("ln_RGB_sum", "ln_RG_sum", "ln_RB_sum", "ln_GB_sum"),
    "ln_rgb_total_ratio_delta": (
        "ln_R_sig",
        "ln_G_sig",
        "ln_B_sig",
        "ln_RGB_sum",
        "Ratio_RG",
        "Delta_RG",
        "Delta_RB",
    ),
    "rgb_raw": ("R_sig", "G_sig", "B_sig"),
    "rgb_sum_chroma": ("RGB_sum", "R_chroma", "G_chroma", "B_chroma"),
    "rgb_ln": ("R_sig", "G_sig", "B_sig", "ln_R_sig", "ln_G_sig", "ln_B_sig"),
    "rgb_ln_total": ("R_sig", "G_sig", "B_sig", "ln_R_sig", "ln_G_sig", "ln_B_sig", "ln_RGB_sum"),
    "previous_available": (
        "ln_R_sig",
        "ln_G_sig",
        "ln_B_sig",
        "G_chroma",
        "ln_RGB_sum",
        "Delta_RG",
        "Ratio_RG",
        "ln_R_over_G",
        "ln_RG_sum",
        "ln_RB_sum",
        "ln_GB_sum",
        "R_sig",
        "G_sig",
        "B_sig",
        "RGB_sum",
        "R_chroma",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize input vectors and Ridge hyperparameters on train->test.")
    parser.add_argument("--train-csv", type=Path, default=PROJECT_ROOT / "dataset" / "train" / "roi_results" / "roi_detailed_statistics.csv")
    parser.add_argument("--test-csv", type=Path, default=PROJECT_ROOT / "dataset" / "test" / "roi_results" / "roi_detailed_statistics.csv")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-greedy-features", type=int, default=10)
    parser.add_argument("--top-hyperparams", type=int, default=20)
    parser.add_argument("--alphas", default="0.01,0.03,0.1,0.3,1,3,10,30,100,300,1000")
    parser.add_argument("--weight-eps-values", default="1,5,10,20,50,100,200")
    parser.add_argument("--weight-beta-values", default="0,0.25,0.5,1")
    parser.add_argument("--target-transforms", default="identity,log")
    parser.add_argument("--bg-corrections", default="subtract")
    parser.add_argument("--no-clean", action="store_true", help="Do not clear existing output files first.")
    return parser.parse_args()


def parse_float_list(text: str) -> tuple[float, ...]:
    values = [float(item.strip()) for item in str(text).replace(";", ",").split(",") if item.strip()]
    if not values:
        raise ValueError(f"No numeric values parsed from: {text}")
    return tuple(values)


def parse_str_list(text: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in str(text).replace(";", ",").split(",") if item.strip())
    if not values:
        raise ValueError(f"No values parsed from: {text}")
    return values


def concentration_from_filename(filename: object) -> float:
    """Use the last numeric token in the stem, so chicken-110.dng becomes 110, not -110."""
    stem = Path(str(filename)).stem
    matches = re.findall(r"\d+(?:\.\d+)?", stem)
    if not matches:
        return math.nan
    return float(matches[-1])


def group_from_filename(filename: object) -> str:
    stem = Path(str(filename)).stem
    return stem.split("-", maxsplit=1)[0] if "-" in stem else "all"


def load_dataframe(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"ROI CSV not found: {path}")
    df = pd.read_csv(path).copy()
    if "filename" not in df.columns:
        raise ValueError(f"ROI CSV must include filename column: {path}")
    df["conc_true"] = df["filename"].apply(concentration_from_filename)
    df["sample_group"] = df["filename"].apply(group_from_filename)
    df = df.dropna(subset=["conc_true"]).sort_values(["conc_true", "filename"]).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No valid concentration labels parsed from {path}")
    return df, df["conc_true"].to_numpy(dtype=float)


def build_features(df: pd.DataFrame, bg_correction: str) -> pd.DataFrame:
    return build_feature_frame(
        df,
        g_mode="avg",
        use_bg=True,
        clip_min=0.1,
        bg_correction=bg_correction,  # type: ignore[arg-type]
    ).copy()


def transform_target(y: np.ndarray, target_transform: str) -> np.ndarray:
    if target_transform == "log":
        return np.log(np.asarray(y, dtype=float) + TARGET_EPS)
    return np.asarray(y, dtype=float)


def inverse_target(y: np.ndarray, target_transform: str) -> np.ndarray:
    if target_transform == "log":
        return np.maximum(np.exp(np.asarray(y, dtype=float)) - TARGET_EPS, 0.0)
    return np.maximum(np.asarray(y, dtype=float), 0.0)


def make_sample_weights(y: np.ndarray, weight_eps: float, weight_beta: float) -> np.ndarray:
    denom = float(weight_beta) * np.asarray(y, dtype=float) + float(weight_eps)
    if np.any(denom <= 0):
        raise ValueError("weight_beta * y + weight_eps must be positive.")
    weights = 1.0 / denom
    return weights / np.mean(weights)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y) & np.isfinite(pred)
    y = y[valid]
    pred = pred[valid]
    if y.size == 0:
        return {"n": 0, "r2": math.nan, "rmse": math.nan, "mae": math.nan, "mean_error": math.nan}
    residual = pred - y
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    return {
        "n": int(y.size),
        "r2": math.nan if ss_tot <= 0 else 1.0 - ss_res / ss_tot,
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "mean_error": float(np.mean(residual)),
    }


def fit_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    alpha: float,
    weight_eps: float,
    weight_beta: float,
    target_transform: str,
) -> Pipeline:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=float(alpha))),
        ]
    )
    model.fit(
        X_train,
        transform_target(y_train, target_transform),
        regressor__sample_weight=make_sample_weights(y_train, weight_eps, weight_beta),
    )
    return model


def predict_model(model: Pipeline, X: np.ndarray, target_transform: str) -> np.ndarray:
    return inverse_target(model.predict(X), target_transform)


def evaluate_trial(
    *,
    trial_id: str,
    stage: str,
    feature_names: tuple[str, ...],
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    weight_eps: float,
    weight_beta: float,
    target_transform: str,
    bg_correction: str,
) -> dict[str, object]:
    X_train = train_features.loc[:, list(feature_names)].to_numpy(dtype=float)
    X_test = test_features.loc[:, list(feature_names)].to_numpy(dtype=float)
    model = fit_model(
        X_train,
        y_train,
        alpha=alpha,
        weight_eps=weight_eps,
        weight_beta=weight_beta,
        target_transform=target_transform,
    )
    train_pred = predict_model(model, X_train, target_transform)
    test_pred = predict_model(model, X_test, target_transform)
    train_row = metrics(y_train, train_pred)
    test_row = metrics(y_test, test_pred)
    return {
        "trial_id": trial_id,
        "stage": stage,
        "bg_correction": bg_correction,
        "target_transform": target_transform,
        "alpha": alpha,
        "weight_eps": weight_eps,
        "weight_beta": weight_beta,
        "n_features": len(feature_names),
        "feature_names": ";".join(feature_names),
        "train_r2": train_row["r2"],
        "train_rmse": train_row["rmse"],
        "train_mae": train_row["mae"],
        "test_r2": test_row["r2"],
        "test_rmse": test_row["rmse"],
        "test_mae": test_row["mae"],
        "test_mean_error": test_row["mean_error"],
        "test_n": test_row["n"],
    }


def hyperparameter_grid(args: argparse.Namespace) -> list[dict[str, object]]:
    alphas = parse_float_list(args.alphas)
    weight_eps_values = parse_float_list(args.weight_eps_values)
    weight_beta_values = parse_float_list(args.weight_beta_values)
    target_transforms = parse_str_list(args.target_transforms)
    bg_corrections = parse_str_list(args.bg_corrections)
    rows: list[dict[str, object]] = []
    for bg_correction, target_transform, alpha, weight_eps, weight_beta in itertools.product(
        bg_corrections,
        target_transforms,
        alphas,
        weight_eps_values,
        weight_beta_values,
    ):
        rows.append(
            {
                "bg_correction": bg_correction,
                "target_transform": target_transform,
                "alpha": float(alpha),
                "weight_eps": float(weight_eps),
                "weight_beta": float(weight_beta),
            }
        )
    return rows


def valid_feature_names(train_features: pd.DataFrame, test_features: pd.DataFrame, names: Iterable[str]) -> tuple[str, ...]:
    selected = []
    for name in names:
        if name in train_features.columns and name in test_features.columns:
            selected.append(name)
    return tuple(dict.fromkeys(selected))


def run_preset_scan(
    *,
    args: argparse.Namespace,
    train_feature_tables: dict[str, pd.DataFrame],
    test_feature_tables: dict[str, pd.DataFrame],
    y_train: np.ndarray,
    y_test: np.ndarray,
    grid: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for hp in grid:
        bg_correction = str(hp["bg_correction"])
        train_features = train_feature_tables[bg_correction]
        test_features = test_feature_tables[bg_correction]
        for vector_id, raw_features in PRESET_INPUT_VECTORS.items():
            feature_names = valid_feature_names(train_features, test_features, raw_features)
            if not feature_names:
                continue
            row = evaluate_trial(
                trial_id=vector_id,
                stage="preset",
                feature_names=feature_names,
                train_features=train_features,
                test_features=test_features,
                y_train=y_train,
                y_test=y_test,
                alpha=float(hp["alpha"]),
                weight_eps=float(hp["weight_eps"]),
                weight_beta=float(hp["weight_beta"]),
                target_transform=str(hp["target_transform"]),
                bg_correction=bg_correction,
            )
            rows.append(row)
    return rows


def greedy_feature_search(
    *,
    hp: dict[str, object],
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    max_features: int,
) -> list[dict[str, object]]:
    pool = list(valid_feature_names(train_features, test_features, FEATURE_POOL))
    selected: list[str] = []
    rows: list[dict[str, object]] = []
    best_score = -float("inf")

    for step in range(1, max_features + 1):
        candidates: list[dict[str, object]] = []
        for feature in pool:
            if feature in selected:
                continue
            trial_features = tuple([*selected, feature])
            row = evaluate_trial(
                trial_id=f"greedy_{step:02d}",
                stage="greedy",
                feature_names=trial_features,
                train_features=train_features,
                test_features=test_features,
                y_train=y_train,
                y_test=y_test,
                alpha=float(hp["alpha"]),
                weight_eps=float(hp["weight_eps"]),
                weight_beta=float(hp["weight_beta"]),
                target_transform=str(hp["target_transform"]),
                bg_correction=str(hp["bg_correction"]),
            )
            row["added_feature"] = feature
            candidates.append(row)
        if not candidates:
            break
        best = max(candidates, key=lambda item: (float(item["test_r2"]), -float(item["test_rmse"])))
        rows.extend(candidates)
        if float(best["test_r2"]) <= best_score + 1e-12:
            break
        best_score = float(best["test_r2"])
        selected.append(str(best["added_feature"]))
    return rows


def run_greedy_scans(
    *,
    args: argparse.Namespace,
    preset_rows: list[dict[str, object]],
    train_feature_tables: dict[str, pd.DataFrame],
    test_feature_tables: dict[str, pd.DataFrame],
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> list[dict[str, object]]:
    preset_df = pd.DataFrame(preset_rows)
    hp_cols = ["bg_correction", "target_transform", "alpha", "weight_eps", "weight_beta"]
    top_hp = (
        preset_df.sort_values(["test_r2", "test_rmse"], ascending=[False, True])
        .loc[:, hp_cols]
        .drop_duplicates()
        .head(int(args.top_hyperparams))
        .to_dict("records")
    )
    rows: list[dict[str, object]] = []
    for hp in top_hp:
        bg_correction = str(hp["bg_correction"])
        rows.extend(
            greedy_feature_search(
                hp=hp,
                train_features=train_feature_tables[bg_correction],
                test_features=test_feature_tables[bg_correction],
                y_train=y_train,
                y_test=y_test,
                max_features=int(args.max_greedy_features),
            )
        )
    return rows


def run_leave_one_out(
    *,
    feature_names: tuple[str, ...],
    train_features: pd.DataFrame,
    y_train: np.ndarray,
    alpha: float,
    weight_eps: float,
    weight_beta: float,
    target_transform: str,
) -> np.ndarray:
    X = train_features.loc[:, list(feature_names)].to_numpy(dtype=float)
    pred = np.zeros_like(y_train, dtype=float)
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X):
        model = fit_model(
            X[train_idx],
            y_train[train_idx],
            alpha=alpha,
            weight_eps=weight_eps,
            weight_beta=weight_beta,
            target_transform=target_transform,
        )
        pred[test_idx] = predict_model(model, X[test_idx], target_transform)
    return pred


def draw_prediction_plot(df: pd.DataFrame, out_path: Path, *, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=160)
    colors = {"chicken": "#1f77b4", "egg": "#ff7f0e", "all": "#2ca02c"}
    for group, sub in df.groupby("sample_group"):
        ax.scatter(
            sub["conc_true"],
            sub["conc_pred"],
            s=42,
            alpha=0.85,
            label=str(group),
            color=colors.get(str(group), None),
            edgecolor="white",
            linewidth=0.5,
        )
    low = float(min(df["conc_true"].min(), df["conc_pred"].min()))
    high = float(max(df["conc_true"].max(), df["conc_pred"].max()))
    pad = max((high - low) * 0.05, 1.0)
    ax.plot([low - pad, high + pad], [low - pad, high + pad], "k--", linewidth=1.0)
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)
    ax.set_xlabel("True concentration")
    ax.set_ylabel("Predicted concentration")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def draw_top_plot(df: pd.DataFrame, out_path: Path) -> None:
    top = df.sort_values(["test_r2", "test_rmse"], ascending=[False, True]).head(25).copy()
    labels = [
        f"{row.stage}:{row.trial_id}\n{row.n_features}f a={row.alpha:g} we={row.weight_eps:g} b={row.weight_beta:g} {row.target_transform}"
        for row in top.itertuples(index=False)
    ]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=150)
    for ax, col, title in zip(axes, ("test_r2", "test_rmse", "test_mae"), ("Test R2", "Test RMSE", "Test MAE")):
        ax.bar(range(len(top)), top[col], color=["#2ca02c" if i == 0 else "#1f77b4" for i in range(len(top))])
        ax.set_title(title)
        ax.set_xticks(range(len(top)))
        ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=6)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_best_outputs(
    *,
    args: argparse.Namespace,
    best: pd.Series,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_feature_tables: dict[str, pd.DataFrame],
    test_feature_tables: dict[str, pd.DataFrame],
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> None:
    out_dir = args.out_dir.resolve()
    best_dir = out_dir / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)

    feature_names = tuple(str(best["feature_names"]).split(";"))
    bg_correction = str(best["bg_correction"])
    target_transform = str(best["target_transform"])
    alpha = float(best["alpha"])
    weight_eps = float(best["weight_eps"])
    weight_beta = float(best["weight_beta"])
    train_features = train_feature_tables[bg_correction]
    test_features = test_feature_tables[bg_correction]
    X_train = train_features.loc[:, list(feature_names)].to_numpy(dtype=float)
    X_test = test_features.loc[:, list(feature_names)].to_numpy(dtype=float)

    model = fit_model(
        X_train,
        y_train,
        alpha=alpha,
        weight_eps=weight_eps,
        weight_beta=weight_beta,
        target_transform=target_transform,
    )
    train_pred = predict_model(model, X_train, target_transform)
    test_pred = predict_model(model, X_test, target_transform)
    loo_pred = run_leave_one_out(
        feature_names=feature_names,
        train_features=train_features,
        y_train=y_train,
        alpha=alpha,
        weight_eps=weight_eps,
        weight_beta=weight_beta,
        target_transform=target_transform,
    )

    bundle = {
        "model": model,
        "g_mode": "avg",
        "alpha": alpha,
        "alphas_tried": (alpha,),
        "feature_names": feature_names,
        "target_transform": target_transform,
        "target_eps": TARGET_EPS,
        "use_bg": True,
        "feature_set": "optimization_train_test_hyperopt",
        "bg_correction": bg_correction,
        "weight_eps": weight_eps,
        "weight_beta": weight_beta,
        "monotonic_calibration": False,
        "sklearn_version": sklearn_version,
        "python_version": platform.python_version(),
        "bundle_created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    joblib.dump(bundle, best_dir / "ridge_model.pkl")

    coef = model.named_steps["regressor"].coef_
    pd.DataFrame({"feature": feature_names, "coefficient": coef}).to_csv(
        best_dir / "ridge_coefficients.csv",
        index=False,
        float_format=CSV_FLOAT_FORMAT,
    )

    train_pred_df = train_df.loc[:, ["filename", "sample_group", "conc_true"]].copy()
    train_pred_df["conc_pred"] = train_pred
    train_pred_df["residual"] = train_pred_df["conc_pred"] - train_pred_df["conc_true"]
    train_pred_df.to_csv(best_dir / "train_predictions.csv", index=False, float_format=CSV_FLOAT_FORMAT)

    loo_df = train_df.loc[:, ["filename", "sample_group", "conc_true"]].copy()
    loo_df["conc_pred"] = loo_pred
    loo_df["residual"] = loo_df["conc_pred"] - loo_df["conc_true"]
    loo_df.to_csv(best_dir / "train_loo_predictions.csv", index=False, float_format=CSV_FLOAT_FORMAT)

    test_pred_df = test_df.loc[:, ["filename", "sample_group", "conc_true"]].copy()
    test_pred_df["conc_pred"] = test_pred
    test_pred_df["residual"] = test_pred_df["conc_pred"] - test_pred_df["conc_true"]
    test_pred_df.to_csv(best_dir / "test_predictions.csv", index=False, float_format=CSV_FLOAT_FORMAT)

    summary_rows = [
        {"split": "train", **metrics(y_train, train_pred)},
        {"split": "train_loo", **metrics(y_train, loo_pred)},
        {"split": "test", **metrics(y_test, test_pred)},
    ]
    pd.DataFrame(summary_rows).to_csv(best_dir / "best_metrics.csv", index=False, float_format=CSV_FLOAT_FORMAT)

    draw_prediction_plot(
        train_pred_df,
        best_dir / "train_prediction.png",
        title=f"Best optimized train fit: R2={metrics(y_train, train_pred)['r2']:.4g}",
    )
    draw_prediction_plot(
        loo_df,
        best_dir / "train_loo_prediction.png",
        title=f"Best optimized train LOO: R2={metrics(y_train, loo_pred)['r2']:.4g}",
    )
    draw_prediction_plot(
        test_pred_df,
        best_dir / "test_prediction.png",
        title=f"Best optimized test: R2={metrics(y_test, test_pred)['r2']:.4g}",
    )

    metadata = {
        "best_trial": best.to_dict(),
        "feature_names": feature_names,
        "train_csv": str(args.train_csv.resolve()),
        "test_csv": str(args.test_csv.resolve()),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (best_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def prepare_output(out_dir: Path, clean: bool) -> None:
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    prepare_output(args.out_dir, clean=not args.no_clean)

    train_df, y_train = load_dataframe(args.train_csv.resolve())
    test_df, y_test = load_dataframe(args.test_csv.resolve())
    bg_corrections = parse_str_list(args.bg_corrections)
    train_feature_tables = {mode: build_features(train_df, mode) for mode in bg_corrections}
    test_feature_tables = {mode: build_features(test_df, mode) for mode in bg_corrections}
    grid = hyperparameter_grid(args)

    print(f"Train CSV: {args.train_csv.resolve()} (n={len(train_df)})")
    print(f"Test CSV:  {args.test_csv.resolve()} (n={len(test_df)})")
    print(f"Output:    {args.out_dir}")
    print(f"Hyperparameter trials per preset vector: {len(grid)}")

    preset_rows = run_preset_scan(
        args=args,
        train_feature_tables=train_feature_tables,
        test_feature_tables=test_feature_tables,
        y_train=y_train,
        y_test=y_test,
        grid=grid,
    )
    print(f"Preset trials: {len(preset_rows)}")

    greedy_rows = run_greedy_scans(
        args=args,
        preset_rows=preset_rows,
        train_feature_tables=train_feature_tables,
        test_feature_tables=test_feature_tables,
        y_train=y_train,
        y_test=y_test,
    )
    print(f"Greedy trials: {len(greedy_rows)}")

    all_rows = [*preset_rows, *greedy_rows]
    all_df = pd.DataFrame(all_rows)
    ranked = all_df.sort_values(["test_r2", "test_rmse"], ascending=[False, True]).reset_index(drop=True)
    all_df.to_csv(args.out_dir / "all_trials.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    ranked.to_csv(args.out_dir / "ranked_trials.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    draw_top_plot(ranked, args.out_dir / "top25_trials.png")

    best = ranked.iloc[0]
    save_best_outputs(
        args=args,
        best=best,
        train_df=train_df,
        test_df=test_df,
        train_feature_tables=train_feature_tables,
        test_feature_tables=test_feature_tables,
        y_train=y_train,
        y_test=y_test,
    )

    summary = [
        "train_test_hyperopt summary",
        "",
        f"best_stage={best['stage']}",
        f"best_trial_id={best['trial_id']}",
        f"test_r2={float(best['test_r2']):.6g}",
        f"test_rmse={float(best['test_rmse']):.6g}",
        f"test_mae={float(best['test_mae']):.6g}",
        f"train_r2={float(best['train_r2']):.6g}",
        f"alpha={float(best['alpha']):.6g}",
        f"weight_eps={float(best['weight_eps']):.6g}",
        f"weight_beta={float(best['weight_beta']):.6g}",
        f"target_transform={best['target_transform']}",
        f"bg_correction={best['bg_correction']}",
        f"n_features={int(best['n_features'])}",
        f"feature_names={best['feature_names']}",
        "",
        "Top 20:",
        ranked.loc[
            :19,
            [
                "stage",
                "trial_id",
                "test_r2",
                "test_rmse",
                "test_mae",
                "train_r2",
                "alpha",
                "weight_eps",
                "weight_beta",
                "target_transform",
                "n_features",
                "feature_names",
            ],
        ].to_string(index=False),
    ]
    (args.out_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print("\n".join(summary[:16]))
    print(f"Saved ranked trials -> {args.out_dir / 'ranked_trials.csv'}")
    print(f"Saved best outputs  -> {args.out_dir / 'best_model'}")


if __name__ == "__main__":
    main()
