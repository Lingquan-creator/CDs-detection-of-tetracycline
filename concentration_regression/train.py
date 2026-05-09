"""
Train a Ridge regression model from ROI statistics and evaluate with Leave-One-Out.

Default data source:
`pic_RGB/roi_results/roi_detailed_statistics.csv` (from config.TrainConfig).
"""

from __future__ import annotations

import argparse
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import ALPHAS, CLIP_MIN, DEFAULT_BG_CORRECTION, TARGET_EPS, TARGET_TRANSFORM, WEIGHT_BETA, WEIGHT_EPS, TrainConfig
from physics_features import build_feature_frame, extract_concentration_from_name, has_background_columns

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


CSV_FLOAT_FORMAT = "%.6g"


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


def transform_target(y: np.ndarray, transform: str, eps: float) -> np.ndarray:
    if transform == "log":
        return np.log(y + eps)
    return y


def inverse_target(pred: np.ndarray, transform: str, eps: float) -> np.ndarray:
    if transform == "log":
        pred = np.exp(pred) - eps
    return np.maximum(np.asarray(pred, dtype=float), 0.0)


def apply_calibrator(pred: np.ndarray, calibrator: Optional[IsotonicRegression]) -> np.ndarray:
    if calibrator is None:
        return np.maximum(np.asarray(pred, dtype=float), 0.0)
    calibrated = calibrator.predict(np.asarray(pred, dtype=float))
    return np.maximum(np.asarray(calibrated, dtype=float), 0.0)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return r2, rmse, mae


def make_sample_weights(y: np.ndarray, weight_eps: float, weight_beta: float = 1.0) -> np.ndarray:
    if weight_eps < 0:
        raise ValueError("--weight-eps must be >= 0.")
    denom = weight_beta * np.asarray(y, dtype=float) + float(weight_eps)
    if np.any(denom <= 0):
        raise ValueError("weight_beta * concentration + weight_eps must stay positive.")
    weights = 1.0 / denom
    return weights / np.mean(weights)


def train_model(
    X: np.ndarray,
    y_target: np.ndarray,
    *,
    alphas: Iterable[float],
    sample_weight: np.ndarray,
) -> Pipeline:
    model = Pipeline(
        [("scaler", StandardScaler()), ("regressor", RidgeCV(alphas=list(alphas), scoring="neg_mean_squared_error"))]
    )
    model.fit(X, y_target, regressor__sample_weight=sample_weight)
    return model


def fit_calibrator(
    y_pred_raw: np.ndarray,
    y_true: np.ndarray,
    sample_weight: np.ndarray,
) -> Optional[IsotonicRegression]:
    if y_pred_raw.size < 2:
        return None
    if float(np.max(y_pred_raw) - np.min(y_pred_raw)) <= 1e-12:
        return None
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(y_pred_raw, y_true, sample_weight=sample_weight)
    return calibrator


def parse_args(default_cfg: TrainConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Ridge regression and run Leave-One-Out validation.")
    parser.add_argument("--roi-csv", type=Path, default=default_cfg.roi_csv, help="Input ROI statistics CSV.")
    parser.add_argument("--model-out", type=Path, default=default_cfg.model_path, help="Output model bundle (.pkl).")
    parser.add_argument("--pred-out", type=Path, default=default_cfg.predictions_path, help="Training prediction CSV.")
    parser.add_argument("--loo-out", type=Path, help="LOO prediction CSV. Default: <pred-out stem>_loo.csv")
    parser.add_argument("--coef-out", type=Path, default=default_cfg.coefficients_path, help="Coefficient CSV path.")
    parser.add_argument("--plot-out", type=Path, help="PNG plot output. Default: <model-out stem>_evaluation.png")
    parser.add_argument(
        "--g-mode",
        choices=["g1", "avg", "auto"],
        default=default_cfg.g_mode,
        help="Green channel mode. 'auto' selects the mode with better internal CV score.",
    )
    parser.add_argument(
        "--use-bg",
        choices=["auto", "true", "false"],
        default="true" if default_cfg.use_bg else "false",
        help="Background subtraction setting. 'auto' uses bg columns when available.",
    )
    parser.add_argument(
        "--target-transform",
        choices=["identity", "log"],
        default=TARGET_TRANSFORM,
        help="Regression target transform.",
    )
    parser.add_argument("--alpha", type=float, help="Single alpha value. If set, --alpha-grid is ignored.")
    parser.add_argument("--alpha-grid", nargs="+", type=float, help="Alpha grid for RidgeCV.")
    parser.add_argument(
        "--feature-set",
        choices=["full", "compact", "compact_exposure", "ln_rgb", "ratio_rg_ratio_rb_delta_rg_delta_rb_delta_gb"],
        default=default_cfg.feature_set,
        help="Feature set: compact=ln RGB + ratios; compact_exposure adds exposure_time.",
    )
    parser.add_argument(
        "--bg-correction",
        choices=["subtract", "divide", "relative"],
        default=getattr(default_cfg, "bg_correction", DEFAULT_BG_CORRECTION),
        help="Background correction mode for --use-bg true: subtract=ROI-BG, divide=ROI/BG, relative=(ROI-BG)/BG.",
    )
    parser.add_argument("--weight-eps", type=float, default=WEIGHT_EPS, help="Sample weight: 1 / (weight_beta * conc + weight_eps).")
    parser.add_argument("--weight-beta", "--beta", dest="weight_beta", type=float, default=WEIGHT_BETA, help="Sample weight beta in 1 / (weight_beta * conc + weight_eps).")
    parser.add_argument("--clip-min", type=float, default=CLIP_MIN, help="Min signal after bg subtraction.")
    return parser.parse_args()


def resolve_output_path(path: Path, output_dir: Path) -> Path:
    chosen = path.expanduser()
    if not chosen.is_absolute():
        chosen = output_dir / chosen
    return chosen.resolve()


def load_training_dataframe(roi_csv: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    if not roi_csv.exists():
        raise FileNotFoundError(f"ROI CSV not found: {roi_csv}")

    df = pd.read_csv(roi_csv)
    if df.empty:
        raise ValueError("ROI CSV is empty.")
    if "filename" not in df.columns:
        raise ValueError("ROI CSV must include a 'filename' column.")

    df = df.copy()
    df["conc_true"] = df["filename"].apply(extract_concentration_from_name)
    df = df.dropna(subset=["conc_true"]).sort_values("conc_true").reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid concentration labels parsed from filename.")

    return df, df["conc_true"].to_numpy(dtype=float)


def resolve_alphas(args: argparse.Namespace) -> Tuple[float, ...]:
    if args.alpha is not None and args.alpha_grid:
        raise ValueError("Use either --alpha or --alpha-grid, not both.")
    if args.alpha is not None:
        return (float(args.alpha),)
    if args.alpha_grid:
        return tuple(float(x) for x in args.alpha_grid)
    return tuple(float(x) for x in ALPHAS)


def format_float_sequence(values: Iterable[float]) -> str:
    return "(" + ", ".join(f"{float(value):.6g}" for value in values) + ")"


def resolve_use_bg(df: pd.DataFrame, g_mode: str, use_bg_arg: str) -> bool:
    if use_bg_arg == "true":
        if not has_background_columns(df, g_mode):
            raise ValueError(f"--use-bg true but required bg_* columns are missing for g_mode={g_mode}.")
        return True
    if use_bg_arg == "false":
        return False
    return has_background_columns(df, g_mode)


def select_feature_set(feature_df: pd.DataFrame, *, use_bg: bool, feature_set: str) -> pd.DataFrame:
    if feature_set == "full":
        return feature_df
    if feature_set not in {"compact", "compact_exposure", "ln_rgb", "ratio_rg_ratio_rb_delta_rg_delta_rb_delta_gb"}:
        raise ValueError(f"Unsupported feature set: {feature_set}")

    ln_rgb_columns = (
        ("ln_R_sig", "ln_G_sig", "ln_B_sig")
        if use_bg
        else ("ln_R", "ln_G", "ln_B")
    )
    no_exposure_columns = (
        ("ln_R_sig", "ln_G_sig", "ln_B_sig", "Ratio_RG", "Delta_RG", "Delta_GB")
        if use_bg
        else ("ln_R", "ln_G", "ln_B", "Ratio_RG", "Delta_RG", "Delta_GB")
    )
    optimized_columns = (
        ("ln_R_sig", "ln_G_sig", "ln_B_sig", "Ratio_RG", "Ratio_RB", "Delta_RG", "Delta_RB", "Delta_GB")
        if use_bg
        else ("ln_R", "ln_G", "ln_B", "Ratio_RG", "Ratio_RB", "Delta_RG", "Delta_RB", "Delta_GB")
    )
    exposure_columns = (
        (
            "ln_R_sig",
            "ln_G_sig",
            "ln_B_sig",
            "ln_R_sig_per_exp",
            "ln_G_sig_per_exp",
            "ln_B_sig_per_exp",
            "Ratio_RG",
            "Delta_RG",
            "Ratio_BG",
            "Var_G",
            "exposure_time",
            "ln_exposure",
        )
        if use_bg
        else (
            "ln_R",
            "ln_G",
            "ln_B",
            "ln_R_per_exp",
            "ln_G_per_exp",
            "ln_B_per_exp",
            "Ratio_RG",
            "Delta_RG",
            "Ratio_BG",
            "Var_G",
            "exposure_time",
            "ln_exposure",
        )
    )
    if feature_set == "ln_rgb":
        selected_columns = ln_rgb_columns
    elif feature_set == "ratio_rg_ratio_rb_delta_rg_delta_rb_delta_gb":
        selected_columns = optimized_columns
    else:
        selected_columns = exposure_columns if feature_set == "compact_exposure" else no_exposure_columns
    missing = [column for column in selected_columns if column not in feature_df.columns]
    if missing:
        raise ValueError(f"Cannot build {feature_set} feature set; missing columns: {missing}")
    return feature_df.loc[:, list(selected_columns)]


def model_selection_score(model: Pipeline, X: np.ndarray, y_target: np.ndarray) -> float:
    regressor = model.named_steps["regressor"]
    best_score = getattr(regressor, "best_score_", None)
    if best_score is not None:
        return float(best_score)
    y_hat = model.predict(X)
    return -float(mean_squared_error(y_target, y_hat))


def train_candidate(
    df: pd.DataFrame,
    y_true: np.ndarray,
    *,
    g_mode: str,
    use_bg: bool,
    target_transform: str,
    target_eps: float,
    alphas: Tuple[float, ...],
    weight_eps: float,
    weight_beta: float,
    clip_min: float,
    feature_set: str,
    bg_correction: str,
) -> Dict[str, object]:
    feature_df = build_feature_frame(df, g_mode=g_mode, use_bg=use_bg, clip_min=clip_min, bg_correction=bg_correction)
    feature_df = select_feature_set(feature_df, use_bg=use_bg, feature_set=feature_set)
    X = feature_df.to_numpy(dtype=float)
    y_target = transform_target(y_true, target_transform, target_eps)
    sample_weight = make_sample_weights(y_true, weight_eps, weight_beta)

    model = train_model(X, y_target, alphas=alphas, sample_weight=sample_weight)
    y_pred_target = model.predict(X)
    y_pred_raw = inverse_target(y_pred_target, target_transform, target_eps)
    calibrator = fit_calibrator(y_pred_raw, y_true, sample_weight)
    y_pred = apply_calibrator(y_pred_raw, calibrator)
    score = model_selection_score(model, X, y_target)

    return {
        "model": model,
        "g_mode": g_mode,
        "use_bg": use_bg,
        "feature_set": feature_set,
        "bg_correction": bg_correction,
        "feature_names": tuple(feature_df.columns),
        "X": X,
        "sample_weight": sample_weight,
        "y_pred_raw": y_pred_raw,
        "y_pred": y_pred,
        "calibrator": calibrator,
        "score": score,
    }


def run_leave_one_out(
    X: np.ndarray,
    y_true: np.ndarray,
    *,
    alphas: Tuple[float, ...],
    target_transform: str,
    target_eps: float,
    weight_eps: float,
    weight_beta: float,
) -> np.ndarray:
    loo = LeaveOneOut()
    y_pred_loo = np.zeros_like(y_true, dtype=float)

    for train_idx, test_idx in loo.split(X):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y_true[train_idx]

        y_train_target = transform_target(y_train, target_transform, target_eps)
        w_train = make_sample_weights(y_train, weight_eps, weight_beta)
        model_fold = train_model(X_train, y_train_target, alphas=alphas, sample_weight=w_train)

        pred_test_target = model_fold.predict(X_test)
        pred_test_raw = inverse_target(pred_test_target, target_transform, target_eps)

        pred_train_target = model_fold.predict(X_train)
        pred_train_raw = inverse_target(pred_train_target, target_transform, target_eps)
        calibrator_fold = fit_calibrator(pred_train_raw, y_train, w_train)
        pred_test = apply_calibrator(pred_test_raw, calibrator_fold)

        y_pred_loo[test_idx] = pred_test

    return y_pred_loo


def draw_result_plot(
    y_true: np.ndarray,
    y_train_pred: np.ndarray,
    y_loo_pred: np.ndarray,
    *,
    train_metrics: Tuple[float, float, float],
    loo_metrics: Tuple[float, float, float],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    v_min = float(min(np.min(y_true), np.min(y_train_pred), np.min(y_loo_pred)))
    v_max = float(max(np.max(y_true), np.max(y_train_pred), np.max(y_loo_pred)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    axes[0].scatter(y_true, y_train_pred, s=24, alpha=0.8)
    axes[0].plot([v_min, v_max], [v_min, v_max], "k--", linewidth=1.0)
    axes[0].set_xlabel("True concentration (uM)")
    axes[0].set_ylabel("Predicted concentration (uM)")
    axes[0].set_title(
        f"Training Fit\\nR2={train_metrics[0]:.6g}, RMSE={train_metrics[1]:.6g}, MAE={train_metrics[2]:.6g}"
    )
    axes[0].grid(alpha=0.25)

    axes[1].scatter(y_true, y_loo_pred, s=24, alpha=0.8, color="#d62728")
    axes[1].plot([v_min, v_max], [v_min, v_max], "k--", linewidth=1.0)
    axes[1].set_xlabel("True concentration (uM)")
    axes[1].set_ylabel("Predicted concentration (uM)")
    axes[1].set_title(f"Leave-One-Out\\nR2={loo_metrics[0]:.6g}, RMSE={loo_metrics[1]:.6g}, MAE={loo_metrics[2]:.6g}")
    axes[1].grid(alpha=0.25)

    format_figure_numbers(fig)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_outputs(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred_raw: np.ndarray,
    y_pred: np.ndarray,
    y_pred_loo: np.ndarray,
    *,
    model: Pipeline,
    calibrator: Optional[IsotonicRegression],
    model_out: Path,
    pred_out: Path,
    loo_out: Path,
    coef_out: Path,
    g_mode: str,
    use_bg: bool,
    target_transform: str,
    target_eps: float,
    alphas: Tuple[float, ...],
    weight_eps: float,
    weight_beta: float,
    feature_set: str,
    bg_correction: str,
    selected_feature_names: Tuple[str, ...],
) -> None:
    for path in (model_out, pred_out, loo_out, coef_out):
        path.parent.mkdir(parents=True, exist_ok=True)

    regressor = model.named_steps["regressor"]
    selected_alpha = float(getattr(regressor, "alpha_", alphas[0]))

    bundle = {
        "model": model,
        "g_mode": g_mode,
        "alpha": selected_alpha,
        "alphas_tried": tuple(alphas),
        "feature_names": selected_feature_names,
        "target_transform": target_transform,
        "target_eps": target_eps,
        "use_bg": use_bg,
        "feature_set": feature_set,
        "bg_correction": bg_correction,
        "weight_eps": weight_eps,
        "weight_beta": weight_beta,
        "isotonic": calibrator,
        "sklearn_version": sklearn_version,
        "python_version": platform.python_version(),
        "bundle_created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    joblib.dump(bundle, model_out)

    pred_df = pd.DataFrame(
        {
            "filename": df["filename"],
            "conc_true": y_true,
            "conc_pred_raw": y_pred_raw,
            "conc_pred": y_pred,
            "residual": y_pred - y_true,
        }
    )
    pred_df.to_csv(pred_out, index=False, float_format=CSV_FLOAT_FORMAT)

    loo_df = pd.DataFrame(
        {
            "filename": df["filename"],
            "conc_true": y_true,
            "conc_pred_loo": y_pred_loo,
            "residual_loo": y_pred_loo - y_true,
        }
    )
    loo_df.to_csv(loo_out, index=False, float_format=CSV_FLOAT_FORMAT)

    coef_df = pd.DataFrame({"feature": selected_feature_names, "coefficient": regressor.coef_})
    coef_df.to_csv(coef_out, index=False, float_format=CSV_FLOAT_FORMAT)


def load_model_bundle(path: Path, g_mode_override: Optional[str] = None) -> Tuple[object, str, str, float, bool, Optional[object]]:
    """
    Backward-compatible helper for callers that imported inference helpers
    from this module before `predict.py` became the dedicated inference entry.
    """
    from predict import load_model_bundle as _load_bundle

    loaded = _load_bundle(path, g_mode_override)
    _, model, g_mode, target_transform, target_eps, use_bg, *rest = loaded
    calibrator = rest[-1] if rest else None
    return model, g_mode, target_transform, target_eps, use_bg, calibrator


def main() -> None:
    cfg = TrainConfig()
    args = parse_args(cfg)
    output_dir = cfg.output_dir.resolve()
    args.model_out = resolve_output_path(args.model_out, output_dir)
    args.pred_out = resolve_output_path(args.pred_out, output_dir)
    args.coef_out = resolve_output_path(args.coef_out, output_dir)

    df, y_true = load_training_dataframe(args.roi_csv)
    alphas = resolve_alphas(args)
    target_transform = str(args.target_transform)

    g_modes = ["g1", "avg"] if args.g_mode == "auto" else [args.g_mode]
    best_result: Optional[Dict[str, object]] = None
    best_score = -float("inf")

    print(f"Training data: {args.roi_csv} (n={len(df)})")
    print(f"Alpha candidates: {format_float_sequence(alphas)}")
    bg_correction = str(args.bg_correction)

    print(
        f"Evaluating g_modes={g_modes}, use_bg={args.use_bg}, bg_correction={bg_correction}, "
        f"feature_set={args.feature_set}, target_transform={target_transform}"
    )

    for g_mode in g_modes:
        use_bg = resolve_use_bg(df, g_mode, args.use_bg)
        result = train_candidate(
            df,
            y_true,
            g_mode=g_mode,
            use_bg=use_bg,
            target_transform=target_transform,
            target_eps=TARGET_EPS,
            alphas=alphas,
            weight_eps=float(args.weight_eps),
            weight_beta=float(args.weight_beta),
            clip_min=float(args.clip_min),
            feature_set=str(args.feature_set),
            bg_correction=bg_correction,
        )
        score = float(result["score"])
        print(
            f"  candidate g_mode={g_mode}, use_bg={use_bg}, bg_correction={bg_correction}, "
            f"feature_set={args.feature_set}, cv_score={score:.6g}"
        )
        if score > best_score:
            best_score = score
            best_result = result

    if best_result is None:
        raise RuntimeError("No valid model candidate found.")

    model = best_result["model"]
    g_mode = str(best_result["g_mode"])
    use_bg = bool(best_result["use_bg"])
    feature_set = str(best_result["feature_set"])
    bg_correction = str(best_result["bg_correction"])
    selected_feature_names = tuple(str(name) for name in best_result["feature_names"])
    X = np.asarray(best_result["X"], dtype=float)
    y_pred_raw = np.asarray(best_result["y_pred_raw"], dtype=float)
    y_pred = np.asarray(best_result["y_pred"], dtype=float)
    calibrator = best_result["calibrator"]

    y_pred_loo = run_leave_one_out(
        X,
        y_true,
        alphas=alphas,
        target_transform=target_transform,
        target_eps=TARGET_EPS,
        weight_eps=float(args.weight_eps),
        weight_beta=float(args.weight_beta),
    )

    train_metrics = evaluate_predictions(y_true, y_pred)
    loo_metrics = evaluate_predictions(y_true, y_pred_loo)

    loo_default = args.pred_out.with_name(args.pred_out.stem + "_loo.csv")
    plot_default = args.model_out.with_name(args.model_out.stem + "_evaluation.png")
    loo_out = resolve_output_path(args.loo_out or loo_default, output_dir)
    plot_out = resolve_output_path(args.plot_out or plot_default, output_dir)

    save_outputs(
        df,
        y_true,
        y_pred_raw,
        y_pred,
        y_pred_loo,
        model=model,
        calibrator=calibrator if isinstance(calibrator, IsotonicRegression) else None,
        model_out=args.model_out,
        pred_out=args.pred_out,
        loo_out=loo_out,
        coef_out=args.coef_out,
        g_mode=g_mode,
        use_bg=use_bg,
        target_transform=target_transform,
        target_eps=TARGET_EPS,
        alphas=alphas,
        weight_eps=float(args.weight_eps),
        weight_beta=float(args.weight_beta),
        feature_set=feature_set,
        bg_correction=bg_correction,
        selected_feature_names=selected_feature_names,
    )
    draw_result_plot(
        y_true,
        y_pred,
        y_pred_loo,
        train_metrics=train_metrics,
        loo_metrics=loo_metrics,
        out_path=plot_out,
    )

    reg = model.named_steps["regressor"]
    selected_alpha = float(getattr(reg, "alpha_", alphas[0]))
    print(
        f"Selected model: g_mode={g_mode}, use_bg={use_bg}, bg_correction={bg_correction}, "
        f"feature_set={feature_set}, alpha={selected_alpha:.6g}"
    )
    print(f"Train   -> R2={train_metrics[0]:.6g}, RMSE={train_metrics[1]:.6g}, MAE={train_metrics[2]:.6g}")
    print(f"LOO-CV  -> R2={loo_metrics[0]:.6g}, RMSE={loo_metrics[1]:.6g}, MAE={loo_metrics[2]:.6g}")
    print(f"Saved model       -> {args.model_out}")
    print(f"Saved train preds -> {args.pred_out}")
    print(f"Saved LOO preds   -> {loo_out}")
    print(f"Saved coefficients-> {args.coef_out}")
    print(f"Saved plot        -> {plot_out}")


if __name__ == "__main__":
    main()
