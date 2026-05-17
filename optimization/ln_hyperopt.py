from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REG_DIR = PROJECT_ROOT / "concentration_regression"
if str(REG_DIR) not in sys.path:
    sys.path.insert(0, str(REG_DIR))

from physics_features import build_feature_frame  # noqa: E402


CSV_FLOAT_FORMAT = "%.6g"
CLIP_MIN = 0.1
TARGET_EPS = 1e-6
RAW_NAME_RE = re.compile(r"^(?P<sample>.+)-(?P<batch>\d+)-(?P<concentration>[-+]?\d+(?:\.\d+)?)$")
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")

ALLOWED_LN_FEATURES: tuple[str, ...] = (
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
)

INPUT_VECTOR_CANDIDATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ln_rgb", ("ln_R_sig", "ln_G_sig", "ln_B_sig")),
    ("ln_rgb_sum", ("ln_RGB_sum",)),
    ("ln_pair_sums", ("ln_RG_sum", "ln_RB_sum", "ln_GB_sum")),
    ("ln_total_rgb_contrast", ("ln_RGB_sum", "ln_R_over_G", "ln_B_over_G")),
    ("ln_total_pair_contrast", ("ln_RGB_sum", "ln_RG_over_GB", "ln_RB_over_GB")),
    ("ln_rgb_total", ("ln_R_sig", "ln_G_sig", "ln_B_sig", "ln_RGB_sum")),
    (
        "ln_rgb_total_contrast",
        ("ln_R_sig", "ln_G_sig", "ln_B_sig", "ln_RGB_sum", "ln_R_over_G", "ln_B_over_G"),
    ),
    (
        "ln_rgb_pair_sums",
        ("ln_R_sig", "ln_G_sig", "ln_B_sig", "ln_RG_sum", "ln_RB_sum", "ln_GB_sum"),
    ),
    ("current_best_ln_g_rbovergb", ("ln_G_sig", "ln_RB_over_GB")),
    ("ln_r", ("ln_R_sig",)),
    ("ln_g", ("ln_G_sig",)),
    ("ln_b", ("ln_B_sig",)),
    ("ln_r_g", ("ln_R_sig", "ln_G_sig")),
    ("ln_g_b", ("ln_G_sig", "ln_B_sig")),
    ("ln_total_rbovergb", ("ln_RGB_sum", "ln_RB_over_GB")),
)


@dataclass(frozen=True)
class DatasetBundle:
    train_df: pd.DataFrame
    y_train: np.ndarray
    train_features: pd.DataFrame
    test_df: pd.DataFrame | None = None
    y_test: np.ndarray | None = None
    test_features: pd.DataFrame | None = None


def _import_module_from_file(module_name: str, module_path: Path, package_dir: Path) -> Any:
    old_path = list(sys.path)
    saved = {name: sys.modules.get(name) for name in ("config", "physics_features", "train")}
    for name in saved:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(package_dir))
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def load_exp_train_module() -> Any:
    return _import_module_from_file(
        "_optimization_concentration_exp_train",
        PROJECT_ROOT / "concentration_EXP" / "train.py",
        PROJECT_ROOT / "concentration_EXP",
    )


def load_gs_train_module() -> Any:
    return _import_module_from_file(
        "_optimization_concentration_gs_train",
        PROJECT_ROOT / "concentration_GS" / "train.py",
        PROJECT_ROOT / "concentration_GS",
    )


def concentration_from_filename(filename: object) -> float:
    stem = Path(str(filename)).stem
    raw_match = RAW_NAME_RE.fullmatch(stem)
    if raw_match:
        return float(raw_match.group("concentration"))
    matches = NUMBER_RE.findall(stem)
    return float(matches[-1]) if matches else math.nan


def group_from_filename(filename: object) -> str:
    stem = Path(str(filename)).stem.lower()
    raw_match = RAW_NAME_RE.fullmatch(stem)
    if raw_match:
        group = raw_match.group("sample").lower()
    elif "-" in stem:
        group = stem.split("-", maxsplit=1)[0]
    else:
        group = "all"
    if group in {"ch", "chicken"}:
        return "chicken"
    if group in {"e", "egg"}:
        return "egg"
    return group


def load_labeled_roi(csv_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    if not csv_path.exists():
        raise FileNotFoundError(f"ROI CSV not found: {csv_path}")
    df = pd.read_csv(csv_path).copy()
    if df.empty:
        raise ValueError(f"ROI CSV is empty: {csv_path}")
    if "filename" not in df.columns:
        raise ValueError(f"ROI CSV must include filename: {csv_path}")

    if "concentration" in df.columns:
        conc = pd.to_numeric(df["concentration"], errors="coerce")
    else:
        conc = pd.Series(np.nan, index=df.index, dtype=float)
    parsed = df["filename"].apply(concentration_from_filename)
    df["conc_true"] = conc.fillna(parsed).astype(float)
    df["sample_group"] = df["filename"].apply(group_from_filename)
    df = df.dropna(subset=["conc_true"]).sort_values(["conc_true", "filename"]).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No valid concentration labels parsed from: {csv_path}")
    return df, df["conc_true"].to_numpy(dtype=float)


def build_ln_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    features = build_feature_frame(
        df,
        g_mode="avg",
        use_bg=True,
        clip_min=CLIP_MIN,
        bg_correction="subtract",
    )
    missing = [name for name in ALLOWED_LN_FEATURES if name not in features.columns]
    if missing:
        raise ValueError(f"Feature builder is missing required ln features: {missing}")
    return features.loc[:, list(ALLOWED_LN_FEATURES)].copy()


def load_datasets(train_csv: Path, test_csv: Path | None, eval_mode: str) -> DatasetBundle:
    train_df, y_train = load_labeled_roi(train_csv)
    train_features = build_ln_feature_table(train_df)
    if eval_mode == "loo":
        return DatasetBundle(train_df=train_df, y_train=y_train, train_features=train_features)
    if test_csv is None:
        raise ValueError("test mode requires --test-csv")
    test_df, y_test = load_labeled_roi(test_csv)
    test_features = build_ln_feature_table(test_df)
    return DatasetBundle(
        train_df=train_df,
        y_train=y_train,
        train_features=train_features,
        test_df=test_df,
        y_test=y_test,
        test_features=test_features,
    )


def json_dumps(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def clean_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return math.nan
    return out if math.isfinite(out) else math.nan


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y) & np.isfinite(pred)
    if not np.any(valid):
        return {"n": 0, "r2": math.nan, "rmse": math.nan, "mae": math.nan, "mean_error": math.nan}
    y = y[valid]
    pred = pred[valid]
    residual = pred - y
    return {
        "n": int(y.size),
        "r2": float(r2_score(y, pred)) if y.size >= 2 else math.nan,
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "mean_error": float(np.mean(residual)),
    }


def transform_target(y: np.ndarray, transform: str) -> np.ndarray:
    if transform == "log":
        return np.log(np.asarray(y, dtype=float) + TARGET_EPS)
    return np.asarray(y, dtype=float)


def inverse_target(y: np.ndarray, transform: str) -> np.ndarray:
    if transform == "log":
        return np.maximum(np.exp(np.asarray(y, dtype=float)) - TARGET_EPS, 0.0)
    return np.maximum(np.asarray(y, dtype=float), 0.0)


def make_sample_weights(y: np.ndarray, weight_eps: float, weight_beta: float) -> np.ndarray:
    denom = float(weight_beta) * np.asarray(y, dtype=float) + float(weight_eps)
    if np.any(denom <= 0):
        raise ValueError("weight_beta * concentration + weight_eps must be positive.")
    weights = 1.0 / denom
    return weights / np.mean(weights)


def fit_ridge_model(X: np.ndarray, y: np.ndarray, params: dict[str, Any]) -> Pipeline:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=float(params["alpha"]), fit_intercept=True)),
        ]
    )
    sample_weight = make_sample_weights(y, float(params["weight_eps"]), float(params["weight_beta"]))
    model.fit(X, transform_target(y, str(params["target_transform"])), regressor__sample_weight=sample_weight)
    return model


def ridge_predict_loo(X: np.ndarray, y: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    pred = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in LeaveOneOut().split(X):
        model = fit_ridge_model(X[train_idx], y[train_idx], params)
        pred[test_idx] = inverse_target(model.predict(X[test_idx]), str(params["target_transform"]))
    return pred


def evaluate_ridge(
    data: DatasetBundle,
    feature_names: tuple[str, ...],
    params: dict[str, Any],
    eval_mode: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    X_train = data.train_features.loc[:, list(feature_names)].to_numpy(dtype=float)
    if eval_mode == "loo":
        y_pred = ridge_predict_loo(X_train, data.y_train, params)
        return data.y_train, y_pred, data.train_df
    if data.test_features is None or data.y_test is None or data.test_df is None:
        raise ValueError("test data is required for test evaluation")
    model = fit_ridge_model(X_train, data.y_train, params)
    X_test = data.test_features.loc[:, list(feature_names)].to_numpy(dtype=float)
    y_pred = inverse_target(model.predict(X_test), str(params["target_transform"]))
    return data.y_test, y_pred, data.test_df


def fit_exp_bundle(
    exp_train: Any,
    y_train: np.ndarray,
    corr_train: pd.DataFrame,
    feature_names: tuple[str, ...],
    params: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], str | None]:
    response_model = str(params["response_model"])
    weight_eps = float(params["weight_eps"])
    fitted: dict[str, dict[str, Any]] = {}
    inverse_scores: dict[str, float] = {}
    conc_min = float(np.min(y_train))
    conc_max = float(np.max(y_train))
    for feature in feature_names:
        try:
            fit_params, _fit_r2, _fit_rmse, _eq = exp_train.fit_response_model(
                response_model,
                y_train,
                corr_train[feature].to_numpy(dtype=float),
                weight_eps=weight_eps,
            )
            fit_id = f"{feature}__{response_model}"
            fitted[fit_id] = {
                "channel": feature,
                "model": response_model,
                "params": tuple(float(v) for v in fit_params),
                "conc_min": conc_min,
                "conc_max": conc_max,
            }
            train_pred = exp_train.invert_response_by_grid(
                corr_train[feature].to_numpy(dtype=float),
                model=response_model,
                params=tuple(float(v) for v in fit_params),
                conc_min=conc_min,
                conc_max=conc_max,
            )
            inverse_scores[fit_id] = clean_float(metrics(y_train, train_pred)["r2"])
        except Exception:
            continue
    if not fitted:
        raise RuntimeError("No EXP response model could be fitted.")
    best_id = max(inverse_scores, key=lambda key: inverse_scores[key]) if inverse_scores else None
    return fitted, best_id


def predict_exp_bundle(
    exp_train: Any,
    corr_eval: pd.DataFrame,
    feature_names: tuple[str, ...],
    fitted: dict[str, dict[str, Any]],
    selected_channel: str,
    best_id: str | None,
) -> np.ndarray:
    pred_by_fit: dict[str, np.ndarray] = {}
    for fit_id, spec in fitted.items():
        feature = str(spec["channel"])
        if feature not in feature_names:
            continue
        pred_by_fit[fit_id] = exp_train.invert_response_by_grid(
            corr_eval[feature].to_numpy(dtype=float),
            model=str(spec["model"]),
            params=tuple(float(v) for v in spec["params"]),
            conc_min=float(spec["conc_min"]),
            conc_max=float(spec["conc_max"]),
        )
    if not pred_by_fit:
        raise RuntimeError("No EXP predictions available.")

    response_model = next(iter(fitted.values()))["model"]
    direct_fit_id = f"{selected_channel}__{response_model}"
    if direct_fit_id in pred_by_fit:
        pred = pred_by_fit[direct_fit_id]
    elif selected_channel == "feature_best" and best_id in pred_by_fit:
        pred = pred_by_fit[str(best_id)]
    elif selected_channel == "feature_mean":
        pred = np.nanmean(np.vstack(list(pred_by_fit.values())), axis=0)
    elif selected_channel == "feature_median":
        pred = np.nanmedian(np.vstack(list(pred_by_fit.values())), axis=0)
    else:
        raise ValueError(f"Unsupported EXP selected_channel: {selected_channel}")
    return np.maximum(np.asarray(pred, dtype=float), 0.0)


def evaluate_exp(
    exp_train: Any,
    data: DatasetBundle,
    feature_names: tuple[str, ...],
    params: dict[str, Any],
    eval_mode: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    train_corr = exp_train.build_response_feature_table(data.train_df, feature_names)
    if eval_mode == "loo":
        pred = np.zeros_like(data.y_train, dtype=float)
        for train_idx, test_idx in LeaveOneOut().split(train_corr):
            fitted, best_id = fit_exp_bundle(
                exp_train,
                data.y_train[train_idx],
                train_corr.iloc[train_idx].reset_index(drop=True),
                feature_names,
                params,
            )
            pred[test_idx] = predict_exp_bundle(
                exp_train,
                train_corr.iloc[test_idx].reset_index(drop=True),
                feature_names,
                fitted,
                str(params["selected_channel"]),
                best_id,
            )
        return data.y_train, pred, data.train_df

    if data.test_df is None or data.y_test is None:
        raise ValueError("test data is required for test evaluation")
    fitted, best_id = fit_exp_bundle(exp_train, data.y_train, train_corr, feature_names, params)
    test_corr = exp_train.build_response_feature_table(data.test_df, feature_names)
    pred = predict_exp_bundle(exp_train, test_corr, feature_names, fitted, str(params["selected_channel"]), best_id)
    return data.y_test, pred, data.test_df


def gs_point_prediction(gs_train: Any, mu_log: np.ndarray) -> np.ndarray:
    return np.maximum(np.exp(np.asarray(mu_log, dtype=np.float64)) - float(gs_train.TARGET_EPS), 0.0)


def evaluate_gs(
    gs_train: Any,
    data: DatasetBundle,
    feature_names: tuple[str, ...],
    params: dict[str, Any],
    eval_mode: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    X_train = data.train_features.loc[:, list(feature_names)].to_numpy(dtype=np.float64)
    kwargs = {
        "hidden_dim": int(params["hidden_dim"]),
        "dropout": float(params["dropout"]),
        "learning_rate": float(params["learning_rate"]),
        "weight_decay": float(params["weight_decay"]),
        "epochs": int(params["epochs"]),
        "patience": int(params["patience"]),
        "min_std": float(params["min_std"]),
        "std_reg_lambda": float(params["std_reg_lambda"]),
        "max_std_log": float(params["max_std_log"]),
        "weight_eps": float(params["weight_eps"]),
        "weight_beta": float(params["weight_beta"]),
        "grad_clip_norm": float(params["grad_clip_norm"]),
        "seed": int(params["seed"]),
        "g_mode": "avg",
        "use_bg": True,
    }
    if eval_mode == "loo":
        mu_log, _std_log = gs_train.run_leave_one_out(
            X_train,
            data.y_train,
            active_feature_names=feature_names,
            **kwargs,
        )
        return data.y_train, gs_point_prediction(gs_train, mu_log), data.train_df

    if data.test_features is None or data.y_test is None or data.test_df is None:
        raise ValueError("test data is required for test evaluation")
    bundle = gs_train.fit_model(
        X_train,
        data.y_train,
        active_feature_names=feature_names,
        reference_concentrations=np.sort(np.unique(data.y_train)),
        **kwargs,
    )
    X_test = data.test_features.loc[:, list(feature_names)].to_numpy(dtype=np.float64)
    mu_log, _std_log = gs_train.posterior_from_bundle(bundle, X_test)
    return data.y_test, gs_point_prediction(gs_train, mu_log), data.test_df


def ridge_default_params() -> dict[str, Any]:
    return {
        "alpha": 0.03,
        "weight_eps": 50.0,
        "weight_beta": 0.0,
        "target_transform": "identity",
    }


def exp_default_params() -> dict[str, Any]:
    return {
        "response_model": "linear",
        "weight_eps": 5.0,
        "selected_channel": "feature_best",
    }


def gs_default_params(screen: bool = True) -> dict[str, Any]:
    return {
        "hidden_dim": 8,
        "dropout": 0.05,
        "learning_rate": 0.01,
        "weight_decay": 0.001,
        "weight_eps": 50.0,
        "weight_beta": 1.0,
        "min_std": 0.08,
        "std_reg_lambda": 0.01,
        "max_std_log": 2.0,
        "grad_clip_norm": 5.0,
        "seed": 42,
        "epochs": 800 if screen else 2500,
        "patience": 100 if screen else 250,
    }


def dedupe_params(params: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for item in params:
        key = json_dumps(item)
        if key not in seen:
            seen.add(key)
            out.append(dict(item))
    return out


def local_numeric_values(value: float, allowed: Sequence[float], *, include_zero: bool = False) -> list[float]:
    values = {float(value)}
    allowed_sorted = sorted(float(v) for v in allowed)
    if include_zero:
        values.add(0.0)
    if value > 0:
        values.update({value / 3.0, value * 3.0})
    closest = sorted(allowed_sorted, key=lambda v: abs(v - value))[:4]
    values.update(closest)
    return sorted(v for v in values if v >= 0)


def ridge_param_candidates(best: dict[str, Any], iteration: int) -> list[dict[str, Any]]:
    alphas = (0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
    eps_values = (1.0, 5.0, 10.0, 20.0, 30.0, 50.0, 80.0, 120.0, 200.0)
    betas = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)
    transforms = ("identity", "log")
    if iteration == 1:
        alpha_values = alphas
        weight_eps_values = eps_values
        beta_values = betas
        transform_values = transforms
    else:
        alpha_values = local_numeric_values(float(best["alpha"]), alphas, include_zero=True)
        weight_eps_values = local_numeric_values(float(best["weight_eps"]), eps_values)
        beta_values = local_numeric_values(float(best["weight_beta"]), betas, include_zero=True)
        transform_values = (str(best["target_transform"]),)
    return dedupe_params(
        {
            "alpha": float(alpha),
            "weight_eps": float(weight_eps),
            "weight_beta": float(beta),
            "target_transform": str(transform),
        }
        for alpha, weight_eps, beta, transform in itertools.product(
            alpha_values, weight_eps_values, beta_values, transform_values
        )
    )


def exp_param_candidates(best: dict[str, Any], feature_names: tuple[str, ...], iteration: int) -> list[dict[str, Any]]:
    response_models = ("exponential_c0", "exponential_offset", "linear", "quadratic")
    eps_values = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
    selected = tuple(feature_names) + ("feature_mean", "feature_median", "feature_best")
    if iteration == 1:
        model_values = response_models
        weight_eps_values = eps_values
        selected_values = selected
    else:
        model_values = (str(best["response_model"]),)
        weight_eps_values = local_numeric_values(float(best["weight_eps"]), eps_values, include_zero=True)
        selected_values = selected
    return dedupe_params(
        {
            "response_model": str(model),
            "weight_eps": float(weight_eps),
            "selected_channel": str(channel),
        }
        for model, weight_eps, channel in itertools.product(model_values, weight_eps_values, selected_values)
    )


def gs_param_candidates(best: dict[str, Any], iteration: int) -> list[dict[str, Any]]:
    ranges: dict[str, tuple[Any, ...]] = {
        "hidden_dim": (4, 8, 16),
        "dropout": (0.0, 0.05, 0.1),
        "learning_rate": (0.003, 0.01, 0.03),
        "weight_decay": (0.0, 0.0001, 0.001, 0.01),
        "weight_eps": (10.0, 30.0, 50.0, 80.0, 120.0),
        "weight_beta": (0.25, 0.5, 1.0),
        "min_std": (0.05, 0.08, 0.12),
        "std_reg_lambda": (0.0, 0.005, 0.01, 0.03),
    }
    base = dict(best)
    candidates = [base]
    for key, values in ranges.items():
        for value in values:
            item = dict(base)
            item[key] = value
            candidates.append(item)
    if iteration > 1:
        candidates.append(dict(best))
    return dedupe_params(candidates)


def evaluator_for(model_family: str) -> tuple[Callable[..., tuple[np.ndarray, np.ndarray, pd.DataFrame]], Any | None]:
    if model_family == "ridge":
        return evaluate_ridge, None
    if model_family == "exp":
        exp_train = load_exp_train_module()

        def _eval(data: DatasetBundle, feature_names: tuple[str, ...], params: dict[str, Any], eval_mode: str):
            return evaluate_exp(exp_train, data, feature_names, params, eval_mode)

        return _eval, exp_train
    if model_family == "gs":
        gs_train = load_gs_train_module()

        def _eval(data: DatasetBundle, feature_names: tuple[str, ...], params: dict[str, Any], eval_mode: str):
            return evaluate_gs(gs_train, data, feature_names, params, eval_mode)

        return _eval, gs_train
    raise ValueError(f"Unsupported model family: {model_family}")


def default_params_for(model_family: str) -> dict[str, Any]:
    if model_family == "ridge":
        return ridge_default_params()
    if model_family == "exp":
        return exp_default_params()
    if model_family == "gs":
        return gs_default_params(screen=True)
    raise ValueError(f"Unsupported model family: {model_family}")


def param_candidates_for(
    model_family: str,
    best_params: dict[str, Any],
    feature_names: tuple[str, ...],
    iteration: int,
) -> list[dict[str, Any]]:
    if model_family == "ridge":
        return ridge_param_candidates(best_params, iteration)
    if model_family == "exp":
        return exp_param_candidates(best_params, feature_names, iteration)
    if model_family == "gs":
        return gs_param_candidates(best_params, iteration)
    raise ValueError(f"Unsupported model family: {model_family}")


def rank_trials(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    ranked = df.copy()
    ranked["_r2_sort"] = ranked["r2"].fillna(-np.inf)
    ranked["_rmse_sort"] = ranked["rmse"].fillna(np.inf)
    ranked["_mae_sort"] = ranked["mae"].fillna(np.inf)
    ranked = ranked.sort_values(["_r2_sort", "_rmse_sort", "_mae_sort"], ascending=[False, True, True])
    return ranked.drop(columns=["_r2_sort", "_rmse_sort", "_mae_sort"]).reset_index(drop=True)


def build_trial_row(
    *,
    model_family: str,
    eval_mode: str,
    stage: str,
    iteration: int,
    vector_id: str,
    feature_names: tuple[str, ...],
    params: dict[str, Any],
    data: DatasetBundle,
    y_true: np.ndarray | None = None,
    y_pred: np.ndarray | None = None,
    error: str = "",
) -> dict[str, Any]:
    metric = metrics(y_true, y_pred) if y_true is not None and y_pred is not None else metrics(np.array([]), np.array([]))
    fallback_eval = len(data.y_train) if eval_mode == "loo" else (0 if data.y_test is None else len(data.y_test))
    n_eval = int(metric["n"]) if metric["n"] else fallback_eval
    return {
        "model_family": model_family,
        "eval_mode": eval_mode,
        "stage": stage,
        "iteration": iteration,
        "vector_id": vector_id,
        "feature_names": ";".join(feature_names),
        "n_features": len(feature_names),
        "hyperparams": json_dumps(params),
        "r2": metric["r2"],
        "rmse": metric["rmse"],
        "mae": metric["mae"],
        "mean_error": metric["mean_error"],
        "n_train": len(data.y_train),
        "n_eval": n_eval,
        "error": error,
    }


def evaluate_trial(
    evaluator: Callable[..., tuple[np.ndarray, np.ndarray, pd.DataFrame]],
    *,
    data: DatasetBundle,
    model_family: str,
    eval_mode: str,
    stage: str,
    iteration: int,
    vector_id: str,
    feature_names: tuple[str, ...],
    params: dict[str, Any],
) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray, pd.DataFrame] | None]:
    try:
        y_true, y_pred, pred_df = evaluator(data, feature_names, params, eval_mode)
        row = build_trial_row(
            model_family=model_family,
            eval_mode=eval_mode,
            stage=stage,
            iteration=iteration,
            vector_id=vector_id,
            feature_names=feature_names,
            params=params,
            data=data,
            y_true=y_true,
            y_pred=y_pred,
        )
        return row, (y_true, y_pred, pred_df)
    except Exception as exc:
        row = build_trial_row(
            model_family=model_family,
            eval_mode=eval_mode,
            stage=stage,
            iteration=iteration,
            vector_id=vector_id,
            feature_names=feature_names,
            params=params,
            data=data,
            error=f"{type(exc).__name__}: {exc}",
        )
        return row, None


def trial_cache_key(
    model_family: str,
    eval_mode: str,
    feature_names: tuple[str, ...],
    params: dict[str, Any],
) -> str:
    return json_dumps(
        {
            "model_family": model_family,
            "eval_mode": eval_mode,
            "feature_names": tuple(feature_names),
            "params": params,
        }
    )


def evaluate_trial_cached(
    evaluator: Callable[..., tuple[np.ndarray, np.ndarray, pd.DataFrame]],
    *,
    cache: dict[str, tuple[tuple[np.ndarray, np.ndarray, pd.DataFrame] | None, str]],
    data: DatasetBundle,
    model_family: str,
    eval_mode: str,
    stage: str,
    iteration: int,
    vector_id: str,
    feature_names: tuple[str, ...],
    params: dict[str, Any],
) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray, pd.DataFrame] | None]:
    key = trial_cache_key(model_family, eval_mode, feature_names, params)
    if key in cache:
        pred, error = cache[key]
        if pred is None:
            row = build_trial_row(
                model_family=model_family,
                eval_mode=eval_mode,
                stage=stage,
                iteration=iteration,
                vector_id=vector_id,
                feature_names=feature_names,
                params=params,
                data=data,
                error=error,
            )
            return row, None
        y_true, y_pred, _pred_df = pred
        row = build_trial_row(
            model_family=model_family,
            eval_mode=eval_mode,
            stage=stage,
            iteration=iteration,
            vector_id=vector_id,
            feature_names=feature_names,
            params=params,
            data=data,
            y_true=y_true,
            y_pred=y_pred,
        )
        return row, pred

    row, pred = evaluate_trial(
        evaluator,
        data=data,
        model_family=model_family,
        eval_mode=eval_mode,
        stage=stage,
        iteration=iteration,
        vector_id=vector_id,
        feature_names=feature_names,
        params=params,
    )
    cache[key] = (pred, str(row.get("error") or ""))
    return row, pred


def is_better(candidate: dict[str, Any], current: dict[str, Any] | None) -> bool:
    if current is None:
        return math.isfinite(clean_float(candidate.get("r2")))
    key = (
        clean_float(candidate.get("r2")),
        -clean_float(candidate.get("rmse")),
        -clean_float(candidate.get("mae")),
    )
    current_key = (
        clean_float(current.get("r2")),
        -clean_float(current.get("rmse")),
        -clean_float(current.get("mae")),
    )
    return key > current_key


def prediction_frame(source_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, best_row: dict[str, Any]) -> pd.DataFrame:
    out = source_df[["filename", "sample_group", "conc_true"]].copy()
    out["conc_pred"] = np.asarray(y_pred, dtype=float)
    out["residual"] = out["conc_pred"] - np.asarray(y_true, dtype=float)
    for key in ("model_family", "eval_mode", "vector_id", "feature_names", "hyperparams", "r2", "rmse", "mae"):
        out[key] = best_row.get(key)
    return out


def write_outputs(
    out_dir: Path,
    input_rows: list[dict[str, Any]],
    param_rows: list[dict[str, Any]],
    best_row: dict[str, Any],
    best_prediction: tuple[np.ndarray, np.ndarray, pd.DataFrame] | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    input_df = pd.DataFrame(input_rows)
    param_df = pd.DataFrame(param_rows)
    all_df = pd.concat([input_df, param_df], ignore_index=True)
    ranked_df = rank_trials(all_df)
    input_df.to_csv(out_dir / "input_vector_trials.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    param_df.to_csv(out_dir / "parameter_trials.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    all_df.to_csv(out_dir / "all_trials.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    ranked_df.to_csv(out_dir / "all_trials_ranked.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    pd.DataFrame([best_row]).to_csv(out_dir / "best_summary.csv", index=False, float_format=CSV_FLOAT_FORMAT)

    with (out_dir / "best_summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"model_family: {best_row.get('model_family')}\n")
        f.write(f"eval_mode: {best_row.get('eval_mode')}\n")
        f.write(f"r2: {best_row.get('r2')}\n")
        f.write(f"rmse: {best_row.get('rmse')}\n")
        f.write(f"mae: {best_row.get('mae')}\n")
        f.write(f"vector_id: {best_row.get('vector_id')}\n")
        f.write(f"feature_names: {best_row.get('feature_names')}\n")
        f.write(f"hyperparams: {best_row.get('hyperparams')}\n")

    if best_prediction is not None:
        y_true, y_pred, pred_df = best_prediction
        prediction_frame(pred_df, y_true, y_pred, best_row).to_csv(
            out_dir / "best_predictions.csv",
            index=False,
            float_format=CSV_FLOAT_FORMAT,
        )


def run_search(args: argparse.Namespace) -> dict[str, Any]:
    data = load_datasets(args.train_csv, args.test_csv, args.eval_mode)
    vectors = list(INPUT_VECTOR_CANDIDATES)
    if args.max_input_vectors and args.max_input_vectors > 0:
        vectors = vectors[: int(args.max_input_vectors)]
    try:
        evaluator, _module = evaluator_for(args.model_family)
    except ModuleNotFoundError as exc:
        vector_id, feature_names = vectors[0]
        params = default_params_for(args.model_family)
        row = build_trial_row(
            model_family=args.model_family,
            eval_mode=args.eval_mode,
            stage="dependency_check",
            iteration=0,
            vector_id=vector_id,
            feature_names=feature_names,
            params=params,
            data=data,
            error=f"{type(exc).__name__}: {exc}",
        )
        write_outputs(args.out_dir, [row], [], row, None)
        return row

    best_params = default_params_for(args.model_family)
    if args.model_family == "gs":
        best_params["epochs"] = int(args.gs_screen_epochs)
        best_params["patience"] = int(args.gs_screen_patience)
    best_vector_id, best_features = vectors[0]
    best_row: dict[str, Any] | None = None
    best_prediction: tuple[np.ndarray, np.ndarray, pd.DataFrame] | None = None
    input_rows: list[dict[str, Any]] = []
    param_rows: list[dict[str, Any]] = []
    trial_cache: dict[str, tuple[tuple[np.ndarray, np.ndarray, pd.DataFrame] | None, str]] = {}

    for iteration in range(1, int(args.iterations) + 1):
        for vector_id, feature_names in vectors:
            row, pred = evaluate_trial_cached(
                evaluator,
                cache=trial_cache,
                data=data,
                model_family=args.model_family,
                eval_mode=args.eval_mode,
                stage="input_vector",
                iteration=iteration,
                vector_id=vector_id,
                feature_names=feature_names,
                params=best_params,
            )
            input_rows.append(row)
            if is_better(row, best_row):
                best_row = row
                best_prediction = pred
                best_vector_id, best_features = vector_id, feature_names
            if best_row is not None:
                write_outputs(args.out_dir, input_rows, param_rows, best_row, best_prediction)

        param_candidates = param_candidates_for(args.model_family, best_params, best_features, iteration)
        if args.max_param_trials and args.max_param_trials > 0:
            param_candidates = param_candidates[: int(args.max_param_trials)]
        for params in param_candidates:
            row, pred = evaluate_trial_cached(
                evaluator,
                cache=trial_cache,
                data=data,
                model_family=args.model_family,
                eval_mode=args.eval_mode,
                stage="parameter_scan",
                iteration=iteration,
                vector_id=best_vector_id,
                feature_names=best_features,
                params=params,
            )
            param_rows.append(row)
            if is_better(row, best_row):
                best_row = row
                best_prediction = pred
                best_params = dict(params)
            if best_row is not None:
                write_outputs(args.out_dir, input_rows, param_rows, best_row, best_prediction)

    if best_row is None:
        raise RuntimeError("No finite trial completed.")
    write_outputs(args.out_dir, input_rows, param_rows, best_row, best_prediction)
    return best_row


def parse_args(
    default_model: str | None = None,
    default_eval_mode: str | None = None,
    default_out_dir: Path | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize ln-only input vectors and hyperparameters.")
    parser.add_argument("--model-family", choices=["exp", "ridge", "gs"], default=default_model)
    parser.add_argument("--eval-mode", choices=["loo", "test"], default=default_eval_mode)
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "train" / "roi_results" / "roi_detailed_statistics.csv",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "test" / "roi_results" / "roi_detailed_statistics.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=default_out_dir)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--max-input-vectors", type=int, default=0)
    parser.add_argument("--max-param-trials", type=int, default=0)
    parser.add_argument("--gs-screen-epochs", type=int, default=800)
    parser.add_argument("--gs-screen-patience", type=int, default=100)
    args = parser.parse_args()
    if args.model_family is None:
        parser.error("--model-family is required")
    if args.eval_mode is None:
        parser.error("--eval-mode is required")
    if args.out_dir is None:
        parser.error("--out-dir is required")
    args.train_csv = args.train_csv.resolve()
    args.test_csv = args.test_csv.resolve()
    args.out_dir = args.out_dir.resolve()
    return args


def main(
    default_model: str | None = None,
    default_eval_mode: str | None = None,
    default_out_dir: Path | None = None,
) -> None:
    args = parse_args(default_model, default_eval_mode, default_out_dir)
    best = run_search(args)
    print(
        "Best "
        f"{args.model_family}/{args.eval_mode}: "
        f"R2={best.get('r2'):.6g}, RMSE={best.get('rmse'):.6g}, MAE={best.get('mae'):.6g}, "
        f"features={best.get('feature_names')}, params={best.get('hyperparams')}"
    )


if __name__ == "__main__":
    main()
