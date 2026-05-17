from __future__ import annotations

import argparse
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn import __version__ as sklearn_version
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_FLOAT_FORMAT = "%.6g"

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optimization.ln_hyperopt import (  # noqa: E402
    DatasetBundle,
    evaluate_exp,
    evaluate_gs,
    evaluate_ridge,
    fit_exp_bundle,
    fit_ridge_model,
    load_datasets,
    load_exp_train_module,
    load_gs_train_module,
    metrics,
    ridge_predict_loo,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/apply best optimized models and calculate recovery/RSD.")
    parser.add_argument("--train-csv", type=Path, default=PROJECT_ROOT / "dataset/train/roi_results/roi_detailed_statistics.csv")
    parser.add_argument("--test-csv", type=Path, default=PROJECT_ROOT / "dataset/test/roi_results/roi_detailed_statistics.csv")
    return parser.parse_args()


def read_best(mode: str, model: str) -> tuple[str, tuple[str, ...], dict[str, Any]]:
    ranked_path = PROJECT_ROOT / "optimization" / mode / model / "OUTPUT" / "all_trials_ranked.csv"
    if not ranked_path.exists():
        raise FileNotFoundError(f"Missing optimization result: {ranked_path}")
    ranked = pd.read_csv(ranked_path)
    if ranked.empty:
        raise ValueError(f"Optimization result is empty: {ranked_path}")
    best = ranked.iloc[0]
    feature_text = str(best["feature_names"])
    features = tuple(name for name in feature_text.split(";") if name)
    params = json.loads(str(best["hyperparams"]))
    return str(best["vector_id"]), features, params


def prediction_df(source_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, *, pred_col: str = "conc_pred") -> pd.DataFrame:
    out = source_df[["filename", "sample_group", "conc_true"]].copy()
    out[pred_col] = np.asarray(y_pred, dtype=float)
    out["residual"] = out[pred_col] - np.asarray(y_true, dtype=float)
    return out


def save_metrics(pred_df: pd.DataFrame, pred_col: str, out_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for group_name, sub in [("all", pred_df), *pred_df.groupby("sample_group")]:
        if isinstance(group_name, tuple):
            group_name = str(group_name[0])
        metric = metrics(sub["conc_true"].to_numpy(dtype=float), sub[pred_col].to_numpy(dtype=float))
        rows.append(
            {
                "group": group_name,
                "r2": metric["r2"],
                "rmse": metric["rmse"],
                "mae": metric["mae"],
                "mean_error": metric["mean_error"],
                "n": metric["n"],
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False, float_format=CSV_FLOAT_FORMAT)


def save_ridge_model(
    data: DatasetBundle,
    features: tuple[str, ...],
    params: dict[str, Any],
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    X = data.train_features.loc[:, list(features)].to_numpy(dtype=float)
    model = fit_ridge_model(X, data.y_train, params)
    bundle = {
        "model": model,
        "g_mode": "avg",
        "alpha": float(params["alpha"]),
        "alphas_tried": (float(params["alpha"]),),
        "feature_names": features,
        "target_transform": str(params["target_transform"]),
        "target_eps": 1e-6,
        "use_bg": True,
        "feature_set": "optimized_ln",
        "bg_correction": "subtract",
        "weight_eps": float(params["weight_eps"]),
        "weight_beta": float(params["weight_beta"]),
        "monotonic_calibration": False,
        "sklearn_version": sklearn_version,
        "python_version": platform.python_version(),
        "bundle_created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    model_path = out_dir / "concentration_model_v1.pkl"
    joblib.dump(bundle, model_path)
    return model_path


def save_exp_params(
    exp_train: Any,
    data: DatasetBundle,
    features: tuple[str, ...],
    params: dict[str, Any],
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    corr = exp_train.build_response_feature_table(data.train_df, features)
    fitted, best_id = fit_exp_bundle(exp_train, data.y_train, corr, features, params)
    rows: list[dict[str, Any]] = []
    for fit_id, spec in fitted.items():
        feature = str(spec["channel"])
        train_pred = exp_train.invert_response_by_grid(
            corr[feature].to_numpy(dtype=float),
            model=str(spec["model"]),
            params=tuple(float(v) for v in spec["params"]),
            conc_min=float(spec["conc_min"]),
            conc_max=float(spec["conc_max"]),
        )
        metric = metrics(data.y_train, train_pred)
        rows.append(
            {
                "fit_id": fit_id,
                "channel": feature,
                "model": spec["model"],
                "params": ";".join(f"{float(v):.12g}" for v in spec["params"]),
                "conc_min": spec["conc_min"],
                "conc_max": spec["conc_max"],
                "inv_r2": metric["r2"],
                "inv_rmse": metric["rmse"],
                "selected_for_mode": fit_id == best_id,
            }
        )
    params_path = out_dir / "exp_fit_params.csv"
    pd.DataFrame(rows).to_csv(params_path, index=False, float_format=CSV_FLOAT_FORMAT)
    return params_path


def save_gs_model(
    gs_train: Any,
    data: DatasetBundle,
    features: tuple[str, ...],
    params: dict[str, Any],
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    X = data.train_features.loc[:, list(features)].to_numpy(dtype=np.float64)
    bundle = gs_train.fit_model(
        X,
        data.y_train,
        hidden_dim=int(params["hidden_dim"]),
        dropout=float(params["dropout"]),
        learning_rate=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
        epochs=int(params["epochs"]),
        patience=int(params["patience"]),
        min_std=float(params["min_std"]),
        std_reg_lambda=float(params["std_reg_lambda"]),
        max_std_log=float(params["max_std_log"]),
        weight_eps=float(params["weight_eps"]),
        weight_beta=float(params["weight_beta"]),
        grad_clip_norm=float(params["grad_clip_norm"]),
        seed=int(params["seed"]),
        g_mode="avg",
        use_bg=True,
        active_feature_names=features,
        reference_concentrations=np.sort(np.unique(data.y_train)),
    )
    bundle["confidence"] = 0.90
    bundle["feature_set"] = "optimized_ln"
    model_path = out_dir / "amortized_posterior_model.pt"
    torch.save(bundle, model_path)
    return model_path


def save_prediction_outputs(mode: str, model: str, pred_df: pd.DataFrame, pred_col: str) -> None:
    if model == "ridge":
        out_dir = PROJECT_ROOT / "OUTPUT" / mode / "predict"
        csv_name = "rgb_predictions.csv"
        metrics_name = "prediction_metrics.csv"
    elif model == "exp":
        out_dir = PROJECT_ROOT / "OUTPUT" / mode / "predict_exp"
        csv_name = "exp_predictions.csv"
        metrics_name = "prediction_metrics.csv"
    elif model == "gs":
        out_dir = PROJECT_ROOT / "OUTPUT" / mode / "predict_gs"
        csv_name = "gs_posterior_summary.csv"
        metrics_name = "prediction_metrics.csv"
    else:
        raise ValueError(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_dir / csv_name, index=False, float_format=CSV_FLOAT_FORMAT)
    save_metrics(pred_df, pred_col, out_dir / metrics_name)
    legacy_metrics = "test_prediction_metrics.csv" if mode == "test" else "loo_prediction_metrics.csv"
    save_metrics(pred_df, pred_col, out_dir / legacy_metrics)


def normalize_food(name: object) -> str:
    text = str(name).lower()
    if text in {"ch", "chicken"}:
        return "chicken"
    if text in {"e", "egg"}:
        return "egg"
    if text in {"", "nan"}:
        return "all"
    return text


def summarize_recovery(mode: str) -> None:
    output_dir = PROJECT_ROOT / "OUTPUT" / mode / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        ("exp", PROJECT_ROOT / "OUTPUT" / mode / "predict_exp" / "exp_predictions.csv", "conc_pred"),
        ("rgb_ridge", PROJECT_ROOT / "OUTPUT" / mode / "predict" / "rgb_predictions.csv", "conc_pred"),
        ("gs_posterior_median", PROJECT_ROOT / "OUTPUT" / mode / "predict_gs" / "gs_posterior_summary.csv", "posterior_median"),
    ]
    sample_frames: list[pd.DataFrame] = []
    for model_name, path, pred_col in sources:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if pred_col not in df.columns:
            continue
        sample = df[["filename", "sample_group", "conc_true", pred_col]].copy()
        sample["matrix"] = sample["sample_group"].apply(normalize_food)
        sample["model"] = model_name
        sample["prediction_column"] = pred_col
        sample["added_concentration_c3"] = pd.to_numeric(sample["conc_true"], errors="coerce")
        sample["detected_concentration_c1"] = pd.to_numeric(sample[pred_col], errors="coerce")
        denom = sample["added_concentration_c3"].to_numpy(dtype=float)
        detected = sample["detected_concentration_c1"].to_numpy(dtype=float)
        recovery = np.full_like(denom, np.nan, dtype=float)
        valid = denom > 0
        recovery[valid] = detected[valid] / denom[valid] * 100.0
        sample["recovery_percent"] = recovery
        sample_frames.append(
            sample[
                [
                    "model",
                    "prediction_column",
                    "matrix",
                    "filename",
                    "added_concentration_c3",
                    "detected_concentration_c1",
                    "recovery_percent",
                ]
            ]
        )

    if not sample_frames:
        pd.DataFrame().to_csv(output_dir / "model_recovery_samples.csv", index=False)
        pd.DataFrame().to_csv(output_dir / "model_recovery_summary.csv", index=False)
        return

    samples = pd.concat(sample_frames, ignore_index=True)
    summary_rows: list[dict[str, Any]] = []
    keys = ["model", "prediction_column", "matrix", "added_concentration_c3"]
    for key_values, group_df in samples.groupby(keys, sort=True):
        values = group_df["recovery_percent"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        detected = group_df["detected_concentration_c1"].to_numpy(dtype=float)
        detected = detected[np.isfinite(detected)]
        rec_mean = float(np.mean(values)) if values.size else math.nan
        rec_sd = float(np.std(values, ddof=1)) if values.size > 1 else math.nan
        rec_rsd = float(rec_sd / abs(rec_mean) * 100.0) if values.size > 1 and not math.isclose(rec_mean, 0.0) else math.nan
        summary_rows.append(
            {
                "model": key_values[0],
                "prediction_column": key_values[1],
                "matrix": key_values[2],
                "added_concentration_c3": key_values[3],
                "n": int(values.size),
                "detected_c1_mean": float(np.mean(detected)) if detected.size else math.nan,
                "detected_c1_sd": float(np.std(detected, ddof=1)) if detected.size > 1 else math.nan,
                "recovery_mean_percent": rec_mean,
                "recovery_sd_percent": rec_sd,
                "recovery_rsd_percent": rec_rsd,
            }
        )
    summary = pd.DataFrame(summary_rows)
    samples.to_csv(output_dir / "model_recovery_samples.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    summary.to_csv(output_dir / "model_recovery_summary.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    write_recovery_table_and_heatmap(summary, output_dir)


def recovery_label(recovery: float, rsd: float, n: float) -> str:
    rec_text = "" if not np.isfinite(recovery) else f"{recovery:.1f}%"
    rsd_text = "RSD n/a" if not np.isfinite(rsd) else f"RSD {rsd:.2f}%"
    n_text = "" if not np.isfinite(n) else f"n={int(n)}"
    return "\n".join(part for part in (rec_text, rsd_text, n_text) if part)


def write_recovery_table_and_heatmap(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return
    view = summary.copy()
    view["row_label"] = view["matrix"].astype(str) + " / " + view["model"].astype(str)
    recovery = view.pivot_table(
        index="row_label",
        columns="added_concentration_c3",
        values="recovery_mean_percent",
        aggfunc="first",
        dropna=False,
    )
    rsd = view.pivot_table(
        index="row_label",
        columns="added_concentration_c3",
        values="recovery_rsd_percent",
        aggfunc="first",
        dropna=False,
    ).reindex_like(recovery)
    n_table = view.pivot_table(
        index="row_label",
        columns="added_concentration_c3",
        values="n",
        aggfunc="first",
        dropna=False,
    ).reindex_like(recovery)
    recovery = recovery.dropna(axis=1, how="all")
    rsd = rsd.reindex_like(recovery)
    n_table = n_table.reindex_like(recovery)
    recovery.to_csv(output_dir / "model_recovery_heatmap_table.csv", float_format=CSV_FLOAT_FORMAT)

    values = recovery.to_numpy(dtype=float)
    fig_w = max(7.0, 1.35 * max(1, recovery.shape[1]) + 2.8)
    fig_h = max(4.5, 0.55 * max(1, recovery.shape[0]) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=160, constrained_layout=True)
    image = ax.imshow(values, cmap="RdYlGn", vmin=0, vmax=150, aspect="auto")
    ax.set_xticks(np.arange(recovery.shape[1]))
    ax.set_yticks(np.arange(recovery.shape[0]))
    ax.set_xticklabels([f"{float(col):.6g}" for col in recovery.columns], rotation=35, ha="right")
    ax.set_yticklabels([str(idx) for idx in recovery.index])
    ax.set_title("Recovery mean with RSD")
    ax.set_xlabel("Concentration")
    ax.set_ylabel("Food / model")
    for i in range(recovery.shape[0]):
        for j in range(recovery.shape[1]):
            rec = float(recovery.iloc[i, j]) if pd.notna(recovery.iloc[i, j]) else math.nan
            rec_rsd = float(rsd.iloc[i, j]) if pd.notna(rsd.iloc[i, j]) else math.nan
            n_value = float(n_table.iloc[i, j]) if pd.notna(n_table.iloc[i, j]) else math.nan
            if np.isfinite(rec):
                color = "white" if rec < 45 or rec > 125 else "black"
                ax.text(j, i, recovery_label(rec, rec_rsd, n_value), ha="center", va="center", fontsize=7, color=color)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Recovery mean (%)")
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "model_recovery_heatmap_rsd.png", bbox_inches="tight")
    plt.close(fig)


def run_mode(mode: str, train_csv: Path, test_csv: Path) -> None:
    eval_mode = "loo" if mode == "LOO" else "test"
    data = load_datasets(train_csv, test_csv, eval_mode)
    full_data = load_datasets(train_csv, test_csv, "test")
    exp_train = load_exp_train_module()
    gs_train = load_gs_train_module()

    ridge_vector_id, ridge_features, ridge_params = read_best(mode, "ridge")
    exp_vector_id, exp_features, exp_params = read_best(mode, "exp")
    gs_vector_id, gs_features, gs_params = read_best(mode, "gs")

    save_ridge_model(full_data, ridge_features, ridge_params, PROJECT_ROOT / "concentration_regression/rgb_fit_output" / f"best_{mode.lower()}")
    save_exp_params(exp_train, full_data, exp_features, exp_params, PROJECT_ROOT / "concentration_EXP/exp_fit_output" / f"best_{mode.lower()}")
    save_gs_model(gs_train, full_data, gs_features, gs_params, PROJECT_ROOT / "concentration_GS/amortized_output" / f"best_{mode.lower()}")

    y_true, y_pred, pred_source = evaluate_ridge(data, ridge_features, ridge_params, eval_mode)
    ridge_pred = prediction_df(pred_source, y_true, y_pred, pred_col="conc_pred")
    ridge_pred["vector_id"] = ridge_vector_id
    ridge_pred["feature_names"] = ";".join(ridge_features)
    ridge_pred["hyperparams"] = json.dumps(ridge_params, sort_keys=True)
    save_prediction_outputs(mode, "ridge", ridge_pred, "conc_pred")

    y_true, y_pred, pred_source = evaluate_exp(exp_train, data, exp_features, exp_params, eval_mode)
    exp_pred = prediction_df(pred_source, y_true, y_pred, pred_col="conc_pred")
    exp_pred["vector_id"] = exp_vector_id
    exp_pred["feature_names"] = ";".join(exp_features)
    exp_pred["hyperparams"] = json.dumps(exp_params, sort_keys=True)
    save_prediction_outputs(mode, "exp", exp_pred, "conc_pred")

    y_true, y_pred, pred_source = evaluate_gs(gs_train, data, gs_features, gs_params, eval_mode)
    gs_pred = prediction_df(pred_source, y_true, y_pred, pred_col="posterior_median")
    gs_pred["posterior_mean"] = gs_pred["posterior_median"]
    gs_pred["vector_id"] = gs_vector_id
    gs_pred["feature_names"] = ";".join(gs_features)
    gs_pred["hyperparams"] = json.dumps(gs_params, sort_keys=True)
    save_prediction_outputs(mode, "gs", gs_pred, "posterior_median")

    summarize_recovery(mode)


def main() -> None:
    args = parse_args()
    train_csv = args.train_csv.resolve()
    test_csv = args.test_csv.resolve()
    run_mode("test", train_csv, test_csv)
    run_mode("LOO", train_csv, test_csv)
    print("Saved optimized predictions and recovery summaries under OUTPUT/test and OUTPUT/LOO.")


if __name__ == "__main__":
    main()
