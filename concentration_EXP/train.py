"""
Exponential Model Concentration Fitting and Inversion.

Fits multiple response models for each RGB channel, then inverts the selected fit to predict concentration.

Usage:
    python train.py                    # Use default ROI CSV
    python train.py --roi-csv <path>   # Custom ROI CSV
    python train.py --channel G_sig    # Use best fitted model for averaged green ROI-BG light channel
    python train.py --weight-eps 10.0  # Adjust low-concentration emphasis
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd

from config import ExpConfig
from physics_features import extract_concentration_from_name, resolve_exposure_time_series

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

_EXPOSURE_MIN = 1e-6
CSV_FLOAT_FORMAT = "%.6g"
EXP_LIGHT_CHANNELS = ("R_sig", "G1_sig", "G2_sig", "G_sig", "B_sig")
AGGREGATE_CHANNELS = {"light_median", "light_mean", "feature_median", "feature_mean"}
BEST_CHANNEL_ALIASES = {"light_best", "feature_best"}
AUTO_CHANNELS = AGGREGATE_CHANNELS | BEST_CHANNEL_ALIASES
FIT_MODEL_NAMES = ("exponential_c0", "exponential_offset", "linear", "quadratic")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def format_plot_number(value: float, _pos: object = None) -> str:
    if not np.isfinite(value):
        return ""
    text = f"{value:.6g}"
    return "0" if text in ("", "-0") else text


def format_figure_numbers(fig: plt.Figure) -> None:
    formatter = FuncFormatter(format_plot_number)
    for ax in fig.axes:
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)


plt.rcParams.update({"axes.unicode_minus": False})


def model_exponential(x: np.ndarray, a: float, b: float, c0: float = 0.0) -> np.ndarray:
    """
    Exponential model: y = a * exp(b * x) + c0.

    Fitting fixes c0=0. The argument is kept for saved-parameter compatibility.
    
    With numerical stability clipping.
    """
    return a * np.exp(np.clip(b * x, -700, 50)) + c0


def model_linear(x: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    return slope * x + intercept


def model_quadratic(x: np.ndarray, quad: float, slope: float, intercept: float) -> np.ndarray:
    return quad * x * x + slope * x + intercept


def _model_exponential_c0(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Pure exponential: y = a * exp(b*x), with baseline fixed at c=0.
    """
    return a * np.exp(np.clip(b * x, -700, 50))


def _model_exponential_offset(x: np.ndarray, a: float, b: float, c0: float) -> np.ndarray:
    return model_exponential(x, a, b, c0)


def predict_response_model(model: str, conc: np.ndarray, params: Tuple[float, ...]) -> np.ndarray:
    x = np.asarray(conc, dtype=float)
    if model == "exponential_c0":
        return model_exponential(x, float(params[0]), float(params[1]), 0.0)
    if model == "exponential_offset":
        return model_exponential(x, float(params[0]), float(params[1]), float(params[2]))
    if model == "linear":
        return model_linear(x, float(params[0]), float(params[1]))
    if model == "quadratic":
        return model_quadratic(x, float(params[0]), float(params[1]), float(params[2]))
    raise ValueError(f"Unsupported response model: {model}")


def fit_quality(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y) & np.isfinite(pred)
    if not np.any(valid):
        return float("nan"), float("nan")
    y = y[valid]
    pred = pred[valid]
    residuals = y - pred
    rss = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - rss / ss_tot if ss_tot > 0 else 1.0
    rmse = float(math.sqrt(rss / len(y)))
    return r2, rmse


def _polyfit_weights(x: np.ndarray, weight_eps: float) -> np.ndarray:
    sigma = np.sqrt(np.maximum(x + float(weight_eps), 1e-12))
    return 1.0 / sigma


def fit_exponential(
    conc: np.ndarray,
    intensity: np.ndarray,
    *,
    weight_eps: float = 0.0,
) -> Tuple[Tuple[float, float, float], float, float]:
    """
    Fit pure exponential model: intensity = a * exp(b * conc), with c fixed to 0.
    
    Args:
        conc: Concentration values (x-axis)
        intensity: Measured intensity values (y-axis)
        weight_eps: Weight emphasis parameter. weights = 1/sqrt(conc + weight_eps)
                   Lower values emphasize low concentration more.
    
    Returns:
        ((a, b, c), r2, rmse): Fitted parameters, R2 score, and RMSE. c is always 0.
    
    Raises:
        RuntimeError: If scipy is not installed
        ValueError: If data is insufficient or invalid
    """
    try:
        from scipy.optimize import curve_fit  # type: ignore[import-not-found]
    except Exception as e:
        raise RuntimeError(
            "scipy is required for exponential fitting. Install it via `pip install scipy`."
        ) from e

    x = np.asarray(conc, dtype=float)
    y = np.asarray(intensity, dtype=float)

    if x.size < 3:
        raise ValueError("Need at least 3 points to fit exponential model.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite values found in data for exponential fit.")

    y_min = float(np.min(y))
    y_max = float(np.max(y))
    if math.isclose(y_min, y_max, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Feature is nearly constant; cannot fit exponential model.")

    # Pure decaying exponential with fixed baseline c=0.
    p0 = (y_max, -0.01)
    bounds = ([0.0, -np.inf], [np.inf, 0.0])

    # Optional weighting to emphasize low-concentration fitting
    sigma: Optional[np.ndarray] = None
    if weight_eps is not None and float(weight_eps) >= 0.0:
        sigma = np.sqrt(x + float(weight_eps))

    # Perform curve fitting
    popt, _ = curve_fit(
        _model_exponential_c0,
        x,
        y,
        p0=p0,
        bounds=bounds,
        maxfev=20000,
        sigma=sigma,
        absolute_sigma=False,
    )

    a, b = float(popt[0]), float(popt[1])
    c0 = 0.0

    # Calculate fit quality metrics
    y_pred = model_exponential(x, a, b, c0)
    residuals = y - y_pred
    rss = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - rss / ss_tot if ss_tot > 0 else 1.0
    rmse = float(math.sqrt(rss / len(y)))

    return (a, b, c0), float(r2), rmse


def fit_response_model(
    model: str,
    conc: np.ndarray,
    intensity: np.ndarray,
    *,
    weight_eps: float = 0.0,
) -> Tuple[Tuple[float, ...], float, float, str]:
    """
    Fit one response model I=f(C) and return params, R2, RMSE, and equation text.
    """
    x = np.asarray(conc, dtype=float)
    y = np.asarray(intensity, dtype=float)
    if x.size < 3:
        raise ValueError("Need at least 3 points to fit response model.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite values found in data for response fit.")
    if math.isclose(float(np.min(y)), float(np.max(y)), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Feature is nearly constant; cannot fit response model.")

    if model == "exponential_c0":
        params3, r2, rmse = fit_exponential(x, y, weight_eps=weight_eps)
        a, b, _c0 = params3
        return (a, b), r2, rmse, "I = a * exp(b*C), c fixed at 0"

    if model == "exponential_offset":
        try:
            from scipy.optimize import curve_fit  # type: ignore[import-not-found]
        except Exception as e:
            raise RuntimeError(
                "scipy is required for exponential fitting. Install it via `pip install scipy`."
            ) from e

        y_min = float(np.min(y))
        y_max = float(np.max(y))
        p0 = (max(y_max - y_min, 1e-6), -0.01, y_min)
        bounds = ([0.0, -np.inf, -np.inf], [np.inf, 0.0, np.inf])
        sigma = np.sqrt(np.maximum(x + float(weight_eps), 1e-12))
        popt, _ = curve_fit(
            _model_exponential_offset,
            x,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=30000,
            sigma=sigma,
            absolute_sigma=False,
        )
        params = tuple(float(v) for v in popt)
        y_pred = predict_response_model(model, x, params)
        r2, rmse = fit_quality(y, y_pred)
        return params, r2, rmse, "I = a * exp(b*C) + c"

    if model == "linear":
        slope, intercept = np.polyfit(x, y, deg=1, w=_polyfit_weights(x, weight_eps))
        params = (float(slope), float(intercept))
        y_pred = predict_response_model(model, x, params)
        r2, rmse = fit_quality(y, y_pred)
        return params, r2, rmse, "I = m*C + k"

    if model == "quadratic":
        quad, slope, intercept = np.polyfit(x, y, deg=2, w=_polyfit_weights(x, weight_eps))
        params = (float(quad), float(slope), float(intercept))
        y_pred = predict_response_model(model, x, params)
        r2, rmse = fit_quality(y, y_pred)
        return params, r2, rmse, "I = q*C^2 + m*C + k"

    raise ValueError(f"Unsupported response model: {model}")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        return float("nan"), float("nan")
    y = y_true[valid]
    pred = y_pred[valid]
    residual = y - pred
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    rmse = float(math.sqrt(ss_res / len(y)))
    return r2, rmse


def invert_exponential_intensity(
    intensity: np.ndarray,
    *,
    a: float,
    b: float,
    c0: float,
) -> np.ndarray:
    """
    Inverse of I = a*exp(b*C) + c0 to get C = ln((I-c0)/a) / b
    
    Returns NaN for invalid parameters.
    Clips ratio to avoid log of non-positive values.
    """
    i = np.asarray(intensity, dtype=float)
    
    if a <= 0.0 or b == 0.0:
        return np.full_like(i, np.nan, dtype=float)

    ratio = (i - c0) / a
    ratio = np.clip(ratio, 1e-12, 1.0)
    return np.log(ratio) / b


def build_corrected_channels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build ROI-BG light intensity channels used by the exponential inversion.

    The model fits intensity directly as I(C)=a*exp(b*C)+c for R/G/B signals.
    ``*_per_exp`` columns are retained as diagnostics but are not fitted by default.
    """
    required = {
        "roi_R_mean",
        "roi_G1_mean",
        "roi_B_mean",
        "bg_R_mean",
        "bg_G1_mean",
        "bg_B_mean",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"ROI CSV missing required columns for exponential fit: {missing}")

    corr = pd.DataFrame(index=df.index)
    corr["exposure_time"] = resolve_exposure_time_series(df)
    corr["R_sig"] = np.maximum(df["roi_R_mean"].to_numpy(dtype=float) - df["bg_R_mean"].to_numpy(dtype=float), 0.1)
    corr["G1_sig"] = np.maximum(df["roi_G1_mean"].to_numpy(dtype=float) - df["bg_G1_mean"].to_numpy(dtype=float), 0.1)
    corr["B_sig"] = np.maximum(df["roi_B_mean"].to_numpy(dtype=float) - df["bg_B_mean"].to_numpy(dtype=float), 0.1)
    if "roi_G2_mean" in df.columns and "bg_G2_mean" in df.columns:
        corr["G2_sig"] = np.maximum(df["roi_G2_mean"].to_numpy(dtype=float) - df["bg_G2_mean"].to_numpy(dtype=float), 0.1)
        corr["G_sig"] = (corr["G1_sig"].to_numpy(dtype=float) + corr["G2_sig"].to_numpy(dtype=float)) / 2.0
    else:
        corr["G2_sig"] = np.nan
        corr["G_sig"] = corr["G1_sig"].to_numpy(dtype=float)

    exposure = np.maximum(corr["exposure_time"].to_numpy(dtype=float), _EXPOSURE_MIN)
    for channel in EXP_LIGHT_CHANNELS:
        corr[f"{channel}_per_exp"] = corr[channel].to_numpy(dtype=float) / exposure
    return corr


def resolve_feature_list(feature_list_arg: Optional[str]) -> Optional[Tuple[str, ...]]:
    if not feature_list_arg:
        return None
    value = str(feature_list_arg).strip()
    maybe_path = Path(value).expanduser()
    if maybe_path.exists():
        value = maybe_path.read_text(encoding="utf-8")
    for sep in (";", ",", "\n", "\t"):
        value = value.replace(sep, " ")
    features: list[str] = []
    seen: set[str] = set()
    for item in value.split():
        name = item.strip()
        if name and name not in seen:
            features.append(name)
            seen.add(name)
    if not features:
        raise ValueError("--feature-list did not contain any feature names.")
    return tuple(features)


def _load_regression_feature_builder() -> Any:
    module_path = PROJECT_ROOT / "concentration_regression" / "physics_features.py"
    spec = importlib.util.spec_from_file_location("concentration_regression_physics_features", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load feature builder from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_feature_frame


def build_response_feature_table(
    df: pd.DataFrame,
    feature_names: Optional[Tuple[str, ...] | List[str]] = None,
) -> pd.DataFrame:
    """
    Build the response values fitted by concentration_EXP.

    Without ``feature_names`` this preserves the original RGB signal channels.
    With custom names it reuses concentration_regression's feature builder so
    EXP and Ridge consume the same input-vector terms.
    """
    corr = build_corrected_channels(df)
    if feature_names is None:
        return corr

    requested = tuple(str(name) for name in feature_names)
    missing_from_corr = [name for name in requested if name not in corr.columns]
    if missing_from_corr:
        build_feature_frame = _load_regression_feature_builder()
        feature_df = build_feature_frame(
            df,
            g_mode="avg",
            use_bg=True,
            clip_min=0.1,
            bg_correction="subtract",
        )
        missing = [name for name in missing_from_corr if name not in feature_df.columns]
        if missing:
            raise ValueError(f"Custom feature list contains unavailable columns: {missing}")
        for name in missing_from_corr:
            corr[name] = feature_df[name].to_numpy(dtype=float)

    columns = ["exposure_time", *requested]
    per_exp_columns = [f"{name}_per_exp" for name in requested if f"{name}_per_exp" in corr.columns]
    return corr.loc[:, columns + per_exp_columns]


def invert_response_by_grid(
    intensity: np.ndarray,
    *,
    model: str,
    params: Tuple[float, ...],
    conc_min: float,
    conc_max: float,
    grid_size: int = 5000,
) -> np.ndarray:
    """Numerically invert I=f(C) by nearest fitted intensity on the training concentration range."""
    y = np.asarray(intensity, dtype=float)
    if not np.isfinite(conc_min) or not np.isfinite(conc_max) or conc_max <= conc_min:
        return np.full_like(y, np.nan, dtype=float)

    x_grid = np.linspace(float(conc_min), float(conc_max), int(grid_size))
    y_grid = predict_response_model(model, x_grid, params)
    valid_grid = np.isfinite(y_grid)
    if not np.any(valid_grid):
        return np.full_like(y, np.nan, dtype=float)
    x_grid = x_grid[valid_grid]
    y_grid = y_grid[valid_grid]

    out = np.full_like(y, np.nan, dtype=float)
    for idx, value in enumerate(y):
        if not np.isfinite(value):
            continue
        nearest = int(np.argmin(np.abs(y_grid - value)))
        out[idx] = x_grid[nearest]
    return out


def fit_id_for(channel: str, model: str) -> str:
    return f"{channel}__{model}"


def predict_from_fitted_features(
    corrected_channels: pd.DataFrame,
    fitted_models: Dict[str, Dict[str, Any]],
    channels: List[str],
    selected_channel: str,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Invert one fitted response model or aggregate fitted model inversions."""
    pred_by_feature: Dict[str, np.ndarray] = {}
    for fit_id, spec in fitted_models.items():
        ch = str(spec["channel"])
        if ch not in channels or ch not in corrected_channels.columns:
            continue
        pred_by_feature[fit_id] = invert_response_by_grid(
            corrected_channels[ch].to_numpy(dtype=float),
            model=str(spec["model"]),
            params=tuple(float(v) for v in spec["params"]),
            conc_min=float(spec["conc_min"]),
            conc_max=float(spec["conc_max"]),
        )

    if selected_channel in fitted_models:
        conc_pred = pred_by_feature[selected_channel]
    elif selected_channel in AGGREGATE_CHANNELS:
        if not pred_by_feature:
            raise RuntimeError("No fitted response models are available for aggregate prediction.")
        stacked = np.vstack([pred_by_feature[key] for key in pred_by_feature])
        if selected_channel in {"feature_mean", "light_mean"}:
            conc_pred = np.nanmean(stacked, axis=0)
        else:
            conc_pred = np.nanmedian(stacked, axis=0)
    else:
        available = sorted(fitted_models) + sorted(channels) + sorted(AUTO_CHANNELS)
        raise ValueError(f"Selected channel '{selected_channel}' is unavailable. Available: {available}")

    return np.maximum(np.asarray(conc_pred, dtype=float), 0.0), pd.DataFrame(pred_by_feature)


def resolve_selected_feature(
    selected_channel: str,
    fit_results: List[Dict[str, object]],
    fitted_models: Dict[str, Dict[str, Any]],
) -> str:
    if selected_channel in fitted_models or selected_channel in AGGREGATE_CHANNELS:
        return selected_channel
    valid_results = [
        row
        for row in fit_results
        if str(row["fit_id"]) in fitted_models and np.isfinite(float(row["inv_r2"]))
    ]
    if selected_channel not in BEST_CHANNEL_ALIASES:
        channel_results = [row for row in valid_results if str(row["channel"]) == selected_channel]
        if not channel_results:
            return selected_channel
        return str(max(channel_results, key=lambda row: float(row["inv_r2"]))["fit_id"])
    if not valid_results:
        raise RuntimeError("Cannot select best light model because no fitted model has finite inverse R2.")
    return str(max(valid_results, key=lambda row: float(row["inv_r2"]))["fit_id"])


def parse_args(default_cfg: ExpConfig) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fit concentration from ROI statistics using exponential model."
    )
    
    parser.add_argument(
        "--roi-csv",
        type=Path,
        default=default_cfg.roi_csv,
        help=f"Path to ROI statistics CSV (default: {default_cfg.roi_csv})",
    )
    
    parser.add_argument(
        "--channel",
        default=default_cfg.channel,
        help=(
            "Which light channel to use for concentration inversion, light_best, or light_median/light_mean "
            f"to aggregate all fitted channels (default: {default_cfg.channel})"
        ),
    )
    
    parser.add_argument(
        "--weight-eps",
        type=float,
        default=default_cfg.weight_eps,
        help=f"Weighting parameter for low-concentration emphasis (default: {default_cfg.weight_eps})",
    )
    
    parser.add_argument(
        "--params-out",
        type=Path,
        default=default_cfg.params_path,
        help="Output CSV for fitted parameters per channel",
    )
    
    parser.add_argument(
        "--pred-out",
        type=Path,
        default=default_cfg.predictions_path,
        help="Output CSV for fitted intensity predictions",
    )
    
    parser.add_argument(
        "--conc-out",
        type=Path,
        help="Output CSV for concentration predictions (default: <roi-csv>_pred_concentration_exp.csv)",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        help="Output PNG for fitting/prediction visualization (default: exp_fit_output/exp_fit_evaluation.png)",
    )
    parser.add_argument(
        "--feature-list",
        default=" ".join(str(name) for name in getattr(default_cfg, "feature_list", ()) or ()),
        help=(
            "Comma/whitespace separated custom feature columns, or a text file containing them. "
            "When set, EXP fits those feature values instead of only RGB signal channels. "
            "Default comes from ExpConfig.feature_list when configured."
        ),
    )
    
    return parser.parse_args()


def resolve_output_path(path: Optional[Path], default_path: Path, output_dir: Path) -> Path:
    chosen = (path or default_path).expanduser()
    if not chosen.is_absolute():
        chosen = output_dir / chosen
    return chosen.resolve()


def run_exponential_fitting(args: argparse.Namespace, cfg: ExpConfig) -> None:
    """Main fitting workflow."""
    params_out = resolve_output_path(args.params_out, cfg.params_path, cfg.output_dir)
    pred_out = resolve_output_path(args.pred_out, cfg.predictions_path, cfg.output_dir)
    
    # Load and validate data
    df_raw = pd.read_csv(args.roi_csv)
    if df_raw.empty:
        raise ValueError("ROI CSV is empty.")

    df = df_raw.copy()
    df["conc_true"] = df["filename"].apply(extract_concentration_from_name)
    df = df.dropna(subset=["conc_true"]).sort_values("conc_true").reset_index(drop=True)
    
    if df.empty:
        raise ValueError(
            "No valid samples found after parsing concentration numbers from filenames."
        )

    conc = df["conc_true"].to_numpy(dtype=float)
    feature_list = resolve_feature_list(args.feature_list)
    corr = build_response_feature_table(df, feature_list)

    # Fit ROI-BG light intensity channels directly.
    channels = list(feature_list) if feature_list is not None else [
        col
        for col in corr.columns
        if col != "exposure_time" and not col.endswith("_per_exp")
    ]
    results: List[Dict[str, object]] = []
    fitted_models: Dict[str, Dict[str, Any]] = {}

    print(f"\nFitting response models to {len(df)} samples...")
    print(f"Concentration range: {conc.min():.6g} - {conc.max():.6g} uM")
    print(f"Weight epsilon: {float(args.weight_eps):.6g}\n")
    if feature_list is not None:
        print(f"Custom feature list ({len(feature_list)}): {', '.join(feature_list)}\n")

    for ch in channels:
        y = corr[ch].to_numpy(dtype=float)
        if np.all(np.isnan(y)):
            print(f"[SKIP] {ch}: all NaN values")
            continue

        for model_name in FIT_MODEL_NAMES:
            try:
                params, r2, rmse, equation = fit_response_model(
                    model_name,
                    conc,
                    y,
                    weight_eps=float(args.weight_eps),
                )
                fit_id = fit_id_for(ch, model_name)
                spec = {
                    "fit_id": fit_id,
                    "channel": ch,
                    "model": model_name,
                    "params": params,
                    "conc_min": float(np.min(conc)),
                    "conc_max": float(np.max(conc)),
                }
                conc_fit_pred = np.maximum(
                    invert_response_by_grid(
                        y,
                        model=model_name,
                        params=params,
                        conc_min=float(np.min(conc)),
                        conc_max=float(np.max(conc)),
                    ),
                    0.0,
                )
                inv_r2, inv_rmse = regression_metrics(conc, conc_fit_pred)
                fitted_models[fit_id] = spec
                row: Dict[str, object] = {
                    "fit_id": fit_id,
                    "channel": ch,
                    "fit_signal": ch,
                    "model": model_name,
                    "equation": equation,
                    "params": ";".join(f"{value:.12g}" for value in params),
                    "r2": r2,
                    "rmse": rmse,
                    "inv_r2": inv_r2,
                    "inv_rmse": inv_rmse,
                    "conc_min": float(np.min(conc)),
                    "conc_max": float(np.max(conc)),
                }
                for idx, value in enumerate(params):
                    row[f"p{idx}"] = float(value)
                # Legacy columns keep old exponential scripts readable where possible.
                row["a"] = float(params[0]) if len(params) > 0 else np.nan
                row["b"] = float(params[1]) if len(params) > 1 else np.nan
                row["c"] = float(params[2]) if len(params) > 2 else 0.0
                results.append(row)
                print(
                    f"[FIT] {fit_id:28s}: R2={r2:.6g}  RMSE={rmse:.6g}  "
                    f"inv_R2={inv_r2:.6g}  inv_RMSE={inv_rmse:.6g}"
                )
            except Exception as e:
                print(f"[FAIL] {ch} / {model_name}: {e}")
                continue

    if not results:
        raise RuntimeError("Response model fitting failed for all channels.")

    # Ensure output directories exist
    params_out.parent.mkdir(parents=True, exist_ok=True)
    pred_out.parent.mkdir(parents=True, exist_ok=True)

    # Save fit parameters
    pd.DataFrame(results).to_csv(params_out, index=False, float_format=CSV_FLOAT_FORMAT)
    print(f"\nSaved fit parameters -> {params_out}")

    # Save fitted intensity predictions per channel
    pred_df = pd.DataFrame({
        "filename": df["filename"],
        "concentration_uM": conc,
        "exposure_time": corr["exposure_time"],
    })

    for ch in channels:
        pred_df[ch] = corr[ch]
        per_exp_col = f"{ch}_per_exp"
        if per_exp_col in corr.columns:
            pred_df[per_exp_col] = corr[per_exp_col]
        for model_name in FIT_MODEL_NAMES:
            fit_id = fit_id_for(ch, model_name)
            if fit_id in fitted_models:
                spec = fitted_models[fit_id]
                pred_df[f"{fit_id}_pred"] = predict_response_model(
                    model_name,
                    conc,
                    tuple(float(v) for v in spec["params"]),
                )
            else:
                pred_df[f"{fit_id}_pred"] = np.nan

    pred_df.to_csv(pred_out, index=False, float_format=CSV_FLOAT_FORMAT)
    print(f"Saved intensity predictions -> {pred_out}")

    # Invert selected feature, or aggregate all fitted feature inversions.
    selected_channel = resolve_selected_feature(args.channel, results, fitted_models)
    if selected_channel != args.channel:
        print(f"\nResolved channel '{args.channel}' -> '{selected_channel}'")
    conc_pred, feature_conc_pred = predict_from_fitted_features(corr, fitted_models, channels, selected_channel)

    # Save concentration predictions
    conc_out = resolve_output_path(args.conc_out, cfg.concentration_path_for(args.roi_csv), cfg.output_dir)
    conc_out.parent.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame({
        "filename": df["filename"],
        "exposure_time": corr["exposure_time"],
        "conc_pred": conc_pred,
        "conc_true": conc,
        "selected_fit": selected_channel,
        "residual": conc_pred - conc,
    })
    if selected_channel in fitted_models:
        selected_spec = fitted_models[selected_channel]
        out_df["selected_channel"] = str(selected_spec["channel"])
        out_df["selected_model"] = str(selected_spec["model"])
        out_df[f"{selected_spec['channel']}_value"] = corr[str(selected_spec["channel"])]
    for ch in feature_conc_pred.columns:
        out_df[f"{ch}_conc_pred"] = feature_conc_pred[ch].to_numpy(dtype=float)
    out_df.to_csv(conc_out, index=False, float_format=CSV_FLOAT_FORMAT)

    # Calculate and display final metrics
    valid_mask = np.isfinite(conc_pred)
    if valid_mask.sum() > 0:
        r2_final = 1 - np.sum((conc[valid_mask] - conc_pred[valid_mask])**2) / np.sum((conc[valid_mask] - conc[valid_mask].mean())**2)
        rmse_final = np.sqrt(np.mean((conc[valid_mask] - conc_pred[valid_mask])**2))
        print(f"\nSaved concentration predictions -> {conc_out}")
        print(f"\nFinal Evaluation (selected={selected_channel}, requested={args.channel}):")
        print(f"  R2 = {r2_final:.6g}")
        print(f"  RMSE = {rmse_final:.6g} uM")
    else:
        print(f"\nSaved concentration predictions -> {conc_out}")
        print("\nWARNING: All predictions are invalid (NaN)")

    plot_out = resolve_output_path(args.plot_out, cfg.fit_plot_path, cfg.output_dir)
    draw_fit_visualization(
        conc=conc,
        corrected_channels=corr,
        fitted_models=fitted_models,
        selected_channel=selected_channel,
        conc_pred=conc_pred,
        out_path=plot_out,
        channels=channels,
    )


def draw_fit_visualization(
    *,
    conc: np.ndarray,
    corrected_channels: pd.DataFrame,
    fitted_models: Dict[str, Dict[str, Any]],
    selected_channel: str,
    conc_pred: np.ndarray,
    out_path: Path,
    channels: List[str],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_axes = len(channels) + 1
    ncols = 3
    nrows = int(math.ceil(total_axes / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.5 * nrows), dpi=150)
    axes = axes.ravel()
    x_curve = np.linspace(float(np.min(conc)), float(np.max(conc)), 200)

    for idx, ch in enumerate(channels):
        ax = axes[idx]
        y = corrected_channels[ch].to_numpy(dtype=float)
        ax.scatter(conc, y, s=20, alpha=0.7, label="Measured")
        channel_specs = [spec for spec in fitted_models.values() if str(spec["channel"]) == ch]
        if channel_specs:
            for spec in channel_specs:
                model_name = str(spec["model"])
                y_curve = predict_response_model(
                    model_name,
                    x_curve,
                    tuple(float(v) for v in spec["params"]),
                )
                linewidth = 2.0 if str(spec["fit_id"]) == selected_channel else 1.2
                ax.plot(x_curve, y_curve, linewidth=linewidth, label=model_name)
            ax.set_title(f"{ch}: response fits")
        else:
            ax.set_title(f"{ch}: skipped")
        ax.set_xlabel("Concentration (uM)")
        ax.set_ylabel("Corrected intensity")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    ax_eval = axes[-1]
    valid = np.isfinite(conc_pred)
    if np.any(valid):
        x = conc[valid]
        y = conc_pred[valid]
        ax_eval.scatter(x, y, s=22, alpha=0.8, color="#1f77b4")
        low = float(min(np.min(x), np.min(y)))
        high = float(max(np.max(x), np.max(y)))
        ax_eval.plot([low, high], [low, high], "k--", linewidth=1.0)
        ax_eval.set_title(f"Concentration inversion ({selected_channel})")
        ax_eval.set_xlabel("True concentration (uM)")
        ax_eval.set_ylabel("Pred concentration (uM)")
        ax_eval.grid(alpha=0.25)
    else:
        ax_eval.text(0.5, 0.5, "No valid prediction", ha="center", va="center")
        ax_eval.set_title(f"Concentration inversion ({selected_channel})")

    for ax in axes[total_axes:]:
        ax.axis("off")

    format_figure_numbers(fig)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved fit plot -> {out_path}")


def main() -> None:
    """Entry point."""
    cfg = ExpConfig()
    args = parse_args(cfg)
    run_exponential_fitting(args, cfg)


if __name__ == "__main__":
    main()
