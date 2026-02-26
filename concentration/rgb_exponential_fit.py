"""
Fit concentration from ROI RGB statistics using either:
- Exponential model (default): fit intensity = a * exp(b * conc) + c, then invert to predict concentration.
- Ridge regression: physics-informed features + RidgeCV (+ optional isotonic calibration).

Default behavior (requested):
- Use `pic_RGB/roi_results/roi_detailed_statistics.csv` as data.
- Use exponential fitting to predict concentration.
"""

from __future__ import annotations

import argparse
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import ALPHAS, CLIP_MIN, TARGET_EPS, TARGET_TRANSFORM, WEIGHT_EPS, TrainConfig
from physics_features import build_feature_frame, extract_concentration_from_name, feature_names, has_background_columns


def transform_target(y: np.ndarray, transform: str, eps: float) -> np.ndarray:
    """Apply target transform for training."""
    if transform == "log":
        return np.log(y + eps)
    return y


def inverse_target(pred: np.ndarray, transform: str, eps: float) -> np.ndarray:
    """Reverse the target transform and enforce non-negative concentration."""
    if transform == "log":
        conc = np.exp(pred) - eps
    else:
        conc = pred
    return np.maximum(conc, 0.0)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return float(r2), rmse


def train_model(X: np.ndarray, y: np.ndarray, *, alphas: Iterable[float], sample_weight: np.ndarray) -> Pipeline:
    model = Pipeline([("scaler", StandardScaler()), ("regressor", RidgeCV(alphas=tuple(alphas), scoring="neg_mean_squared_error"))])
    model.fit(X, y, regressor__sample_weight=sample_weight)
    return model


def model_exponential(x: np.ndarray, a: float, b: float, c0: float) -> np.ndarray:
    """y = a * exp(b * x) + c0 (with clipping for numerical stability)."""
    return a * np.exp(np.clip(b * x, -700, 50)) + c0


def _model_exponential_i0(x: np.ndarray, i0: float, b: float, c0: float) -> np.ndarray:
    """Re-parameterized exponential: y = (i0 - c0) * exp(b*x) + c0, with i0 >= max(y)."""
    return (i0 - c0) * np.exp(np.clip(b * x, -700, 50)) + c0


def fit_exponential(conc: np.ndarray, intensity: np.ndarray, *, weight_eps: float = 0.0) -> Tuple[Tuple[float, float, float], float, float]:
    """Fit intensity = a * exp(b*conc) + c; return (a,b,c), r2, rmse."""
    try:
        from scipy.optimize import curve_fit  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError("scipy is required for exponential fitting. Install it via `pip install scipy`.") from e

    x = np.asarray(conc, dtype=float)
    y = np.asarray(intensity, dtype=float)
    if x.size < 3:
        raise ValueError("Need at least 3 points to fit exponential model.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite values found in data for exponential fit.")

    y_min = float(np.min(y))
    y_max = float(np.max(y))
    y_range = y_max - y_min
    y_range = float(y_range if y_range > 0 else 1.0)

    # Parameterization enforces i0 (= intensity at conc=0) >= max observed intensity,
    # so inverse prediction will not produce negative concentrations.
    i0_min = y_max
    p0 = (y_max * 1.05, -0.01, max(0.0, y_min))
    bounds = ([i0_min, -np.inf, 0.0], [np.inf, 0.0, y_max])

    # Optional low-concentration emphasis: weight ~ 1/(conc + weight_eps)
    sigma: Optional[np.ndarray] = None
    if weight_eps is not None and float(weight_eps) >= 0.0:
        sigma = np.sqrt(x + float(weight_eps))

    popt, _ = curve_fit(_model_exponential_i0, x, y, p0=p0, bounds=bounds, maxfev=20000, sigma=sigma, absolute_sigma=False)
    i0, b, c0 = float(popt[0]), float(popt[1]), float(popt[2])
    a = i0 - c0

    y_pred = model_exponential(x, a, b, c0)
    residuals = y - y_pred
    rss = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - rss / ss_tot if ss_tot > 0 else 1.0
    rmse = float(math.sqrt(rss / len(y)))
    return (a, b, c0), float(r2), rmse


def invert_exponential_intensity(intensity: np.ndarray, *, a: float, b: float, c0: float) -> np.ndarray:
    """Inverse of I = a*exp(b*C)+c0 => C = ln((I-c0)/a)/b. Returns NaN for invalid params."""
    i = np.asarray(intensity, dtype=float)
    if a <= 0.0 or b == 0.0:
        return np.full_like(i, np.nan, dtype=float)

    ratio = (i - c0) / a
    ratio = np.clip(ratio, 1e-12, 1.0)
    return np.log(ratio) / b


def build_corrected_channels(df: pd.DataFrame) -> pd.DataFrame:
    """Build background-corrected mean intensities per channel (ROI - BG, clipped to >=0)."""
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

    r = (df["roi_R_mean"] - df["bg_R_mean"]).clip(lower=0.0)
    g1 = (df["roi_G1_mean"] - df["bg_G1_mean"]).clip(lower=0.0)
    b = (df["roi_B_mean"] - df["bg_B_mean"]).clip(lower=0.0)

    has_g2 = "roi_G2_mean" in df.columns and "bg_G2_mean" in df.columns
    if has_g2:
        g2 = (df["roi_G2_mean"] - df["bg_G2_mean"]).clip(lower=0.0)
        g = ((g1 + g2) / 2.0).clip(lower=0.0)
    else:
        g2 = pd.Series([np.nan] * len(df), index=df.index, dtype=float)
        g = g1

    return pd.DataFrame(
        {
            "R_corr": r.astype(float),
            "G1_corr": g1.astype(float),
            "G2_corr": g2.astype(float),
            "G_corr": g.astype(float),
            "B_corr": b.astype(float),
        }
    )


def save_artifacts(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred_raw: np.ndarray,
    y_pred_calibrated: np.ndarray,
    model: Pipeline,
    calibrator: Optional[IsotonicRegression],
    *,
    model_out: Path,
    pred_out: Path,
    coef_out: Path,
    g_mode: str,
    alpha: float,
    target_transform: str,
    target_eps: float,
    use_bg: bool,
    feature_names_active: Tuple[str, ...],
    alphas_tried: Iterable[float],
    weight_eps: float,
) -> None:
    bundle: Dict[str, object] = {
        "model": model,
        "g_mode": g_mode,
        "alpha": alpha,
        "alphas_tried": tuple(alphas_tried),
        "feature_names": feature_names_active,
        "target_transform": target_transform,
        "target_eps": target_eps,
        "use_bg": use_bg,
        "weight_eps": weight_eps,
        "sklearn_version": sklearn_version,
        "python_version": platform.python_version(),
        "bundle_created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "isotonic": calibrator,
    }
    joblib.dump(bundle, model_out)

    pred_df = pd.DataFrame(
        {
            "filename": df["filename"],
            "conc_true": y_true,
            "conc_pred_raw": y_pred_raw,
            "conc_pred": y_pred_calibrated,
            "residual": y_pred_calibrated - y_true,
        }
    )
    pred_df.to_csv(pred_out, index=False)

    coeffs = model.named_steps["regressor"].coef_
    coef_df = pd.DataFrame({"feature": feature_names_active, "coefficient": coeffs})
    coef_df.to_csv(coef_out, index=False)


def parse_args(default_cfg: TrainConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit concentration from ROI statistics (exponential or ridge).")
    parser.add_argument(
        "--method",
        choices=["exponential", "ridge"],
        default="exponential",
        help="Which model to run. Default: exponential (fit intensity=a*exp(b*conc)+c and invert).",
    )
    parser.add_argument("--roi-csv", type=Path, default=default_cfg.roi_csv, help="Path to roi_detailed_statistics.csv")

    # --- Exponential options ---
    parser.add_argument(
        "--exp-channel",
        choices=["R_corr", "G1_corr", "G2_corr", "G_corr", "B_corr"],
        default="G1_corr",
        help="Which corrected channel to invert for concentration output (exponential method).",
    )
    parser.add_argument(
        "--exp-out",
        type=Path,
        help="Output CSV for exponential concentration predictions. Defaults to <roi-csv>_pred_concentration_exp.csv",
    )
    parser.add_argument(
        "--exp-weight-eps",
        type=float,
        default=5.0,
        help="Exponential fit weighting: weights ~ 1/(conc + eps). Smaller eps emphasizes low concentration more.",
    )
    parser.add_argument(
        "--exp-params-out",
        type=Path,
        default=default_cfg.output_dir / "rgb_fit_params.csv",
        help="Output CSV for exponential fit parameters (per channel).",
    )
    parser.add_argument(
        "--exp-pred-out",
        type=Path,
        default=default_cfg.output_dir / "rgb_fit_predictions.csv",
        help="Output CSV for exponential fitted intensity predictions (per channel).",
    )

    # --- Ridge options ---
    parser.add_argument("--alpha", type=float, help="Use a single alpha (skips CV); default: RidgeCV over alpha-grid.")
    parser.add_argument(
        "--alpha-grid",
        nargs="+",
        type=float,
        help="Override alpha grid for RidgeCV (e.g., --alpha-grid 0.1 0.5 1.0). Defaults to config.ALPHAS.",
    )
    parser.add_argument(
        "--g-mode",
        choices=["g1", "avg", "auto"],
        default="auto",
        help="Use G1 only (g1), average G1/G2 (avg), or auto-select best (auto).",
    )
    parser.add_argument("--model-out", type=Path, default=default_cfg.model_path, help="Path to save joblib model bundle.")
    parser.add_argument("--pred-out", type=Path, default=default_cfg.predictions_path, help="Path to save predictions CSV.")
    parser.add_argument("--coef-out", type=Path, default=default_cfg.coefficients_path, help="Path to save coefficients CSV.")
    parser.add_argument(
        "--use-bg",
        choices=["auto", "true"],
        default="auto",
        help="Use background-subtracted features (bg_* columns must exist).",
    )
    parser.add_argument(
        "--target-transform",
        choices=["identity", "log"],
        help="Override target transform. Defaults to 'identity' (Strategy B removed).",
    )
    return parser.parse_args()


def decide_use_bg(df: pd.DataFrame, g_mode: str, requested: Optional[bool]) -> bool:
    if requested is False:
        raise ValueError("Strategy B (No BG mode) has been removed. Background subtraction is required.")

    if not has_background_columns(df, g_mode):
        raise ValueError(f"Background columns (bg_*) are missing for g_mode='{g_mode}'. Background subtraction is required.")

    return True


def run_exponential(args: argparse.Namespace, cfg: TrainConfig) -> None:
    df_raw = pd.read_csv(args.roi_csv)
    if df_raw.empty:
        raise ValueError("ROI CSV is empty.")

    df = df_raw.copy()
    df["conc_true"] = df["filename"].apply(extract_concentration_from_name)
    df = df.dropna(subset=["conc_true"]).sort_values("conc_true").reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid samples found after parsing concentration numbers from filenames.")

    conc = df["conc_true"].to_numpy(dtype=float)
    corr = build_corrected_channels(df)

    channels = ["R_corr", "G1_corr", "G2_corr", "G_corr", "B_corr"]
    results: List[Dict[str, object]] = []
    params_by_channel: Dict[str, Tuple[float, float, float]] = {}

    for ch in channels:
        y = corr[ch].to_numpy(dtype=float)
        if np.all(np.isnan(y)):
            continue
        (a, b, c0), r2, rmse = fit_exponential(conc, y, weight_eps=float(args.exp_weight_eps))
        params_by_channel[ch] = (a, b, c0)
        results.append({"channel": ch, "a": a, "b": b, "c": c0, "r2": r2, "rmse": rmse})
        print(f"[EXP] {ch}: a={a:.6g}, b={b:.6g}, c={c0:.6g}  r2={r2:.4f}  rmse={rmse:.4f}")

    if not results:
        raise RuntimeError("Exponential fit failed for all channels.")

    # Ensure output dirs exist
    for path in (args.exp_params_out, args.exp_pred_out):
        path.parent.mkdir(parents=True, exist_ok=True)

    # Save fit parameters
    pd.DataFrame(results).to_csv(args.exp_params_out, index=False)
    print(f"Saved exponential fit params -> {args.exp_params_out}")

    # Save fitted intensity predictions per channel (given true concentration)
    pred_df = pd.DataFrame({"filename": df["filename"], "concentration_uM": conc})
    for ch in channels:
        pred_df[ch] = corr[ch]
        if ch in params_by_channel:
            a, b, c0 = params_by_channel[ch]
            pred_df[f"{ch}_pred"] = model_exponential(conc, a, b, c0)
        else:
            pred_df[f"{ch}_pred"] = np.nan
    pred_df.to_csv(args.exp_pred_out, index=False)
    print(f"Saved exponential fit predictions -> {args.exp_pred_out}")

    # Save concentration predictions for selected channel
    if args.exp_channel not in params_by_channel:
        raise ValueError(f"Selected exp channel '{args.exp_channel}' was not fitted (missing data).")

    a, b, c0 = params_by_channel[args.exp_channel]
    conc_pred = invert_exponential_intensity(corr[args.exp_channel].to_numpy(dtype=float), a=a, b=b, c0=c0)
    conc_pred = np.maximum(np.asarray(conc_pred, dtype=float), 0.0)

    exp_out = args.exp_out or args.roi_csv.with_name(args.roi_csv.stem + "_pred_concentration_exp.csv")
    exp_out.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({"filename": df["filename"], "conc_pred": conc_pred, "conc_true": conc, "residual": conc_pred - conc})
    out_df.to_csv(exp_out, index=False)
    print(f"Saved exponential concentration predictions -> {exp_out} (channel={args.exp_channel})")


def run_ridge(args: argparse.Namespace, cfg: TrainConfig) -> None:
    # Ensure output directories exist
    for path in (args.model_out, args.pred_out, args.coef_out):
        path.parent.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(args.roi_csv)
    if df_raw.empty:
        raise ValueError("ROI CSV is empty.")

    # Prepare data
    df = df_raw.copy()
    df["conc"] = df["filename"].apply(extract_concentration_from_name)
    df = df.dropna(subset=["conc"]).sort_values("conc").reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid samples found after parsing concentration numbers from filenames.")

    y = df["conc"].to_numpy()

    # Sample weights for low-concentration emphasis
    sample_weights = 1.0 / (y + WEIGHT_EPS)
    sample_weights = sample_weights / np.mean(sample_weights)

    use_bg_requested: Optional[bool]
    if args.use_bg == "true":
        use_bg_requested = True
    else:  # auto
        use_bg_requested = None

    # Alpha grid selection
    if args.alpha is not None and args.alpha_grid:
        raise ValueError("Use either --alpha or --alpha-grid, not both.")
    if args.alpha_grid:
        alpha_grid = tuple(args.alpha_grid)
    elif args.alpha is not None:
        alpha_grid = (args.alpha,)
    else:
        alpha_grid = ALPHAS

    # g-mode Search
    g_modes = ["g1", "avg"] if args.g_mode == "auto" else [args.g_mode]

    best_result = None
    best_score = -float("inf")

    print(f"Starting training sweep. g_modes={g_modes}, use_bg_req={use_bg_requested}, alphas={alpha_grid}")

    for g_mode in g_modes:
        use_bg = decide_use_bg(df, g_mode, use_bg_requested)
        features_df = build_feature_frame(df, g_mode=g_mode, use_bg=use_bg, clip_min=CLIP_MIN)

        X = features_df.to_numpy()

        # Decide target transform
        target_transform = args.target_transform or "identity"
        y_target = transform_target(y, target_transform, TARGET_EPS)

        model = train_model(X, y_target, alphas=alpha_grid, sample_weight=sample_weights)

        # Score: best mean cross-validated score (negative MSE) from RidgeCV
        score = model.named_steps["regressor"].best_score_
        
        print(f"  [g_mode={g_mode}, use_bg={use_bg}] CV Score (neg MSE): {score:.4f}")

        if score > best_score:
            best_score = score
            best_result = {
                "model": model,
                "g_mode": g_mode,
                "use_bg": use_bg,
                "target_transform": target_transform,
                "X": X,
            }

    if best_result is None:
        raise RuntimeError("Model training failed.")

    # Unpack best result
    model = best_result["model"]
    g_mode = best_result["g_mode"]
    use_bg = best_result["use_bg"]
    target_transform = best_result["target_transform"]
    X = best_result["X"]

    print(f"Selected Best: g_mode={g_mode}, use_bg={use_bg}, target={target_transform}")

    # Predict and inverse-transform
    y_pred_target = model.predict(X)
    y_pred_raw = inverse_target(y_pred_target, target_transform, TARGET_EPS)
    y_pred_raw = np.maximum(y_pred_raw, 0.0)

    # Monotonic calibration (isotonic regression)
    calibrator: Optional[IsotonicRegression] = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(y_pred_raw, y, sample_weight=sample_weights)
    y_pred_calibrated = calibrator.predict(y_pred_raw)
    y_pred_calibrated = np.maximum(y_pred_calibrated, 0.0)

    r2, rmse = evaluate_predictions(y, y_pred_calibrated)
    print(f"R2 on train (calibrated): {r2:.4f}  <-- (Fitting Accuracy, biased high)")
    print(f"RMSE on train (calibrated): {rmse:.4f} uM")

    # --- Leave-One-Out Cross-Validation for Unbiased Evaluation ---
    print("\nRunning Leave-One-Out Cross-Validation (Unbiased Evaluation)...")
    from sklearn.model_selection import LeaveOneOut
    
    cv = LeaveOneOut()
    y_cv_preds = np.zeros_like(y)
    
    # Pre-calculate active features for CV to save time (using best settings)
    X_cv = X
    
    for train_ix, test_ix in cv.split(X_cv):
        X_train, X_test = X_cv[train_ix], X_cv[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        
        # 1. Transform Target
        y_train_target = transform_target(y_train, target_transform, TARGET_EPS)
        
        # 2. Weights
        w_train = 1.0 / (y_train + WEIGHT_EPS)
        w_train = w_train / np.mean(w_train)
        
        # 3. Train Ridge (using passed alphas)
        # Note: We use the same alpha_grid, RidgeCV will pick best alpha for this fold internally
        model_fold = train_model(X_train, y_train_target, alphas=alpha_grid, sample_weight=w_train)
        
        # DEBUG: Print selected alpha for the first few folds to verify regularizer behavior
        if test_ix[0] < 5:
           best_alpha = model_fold.named_steps["regressor"].alpha_
           print(f"    [LOO-CV via {g_mode}] Fold test_ix={test_ix[0]}: Selected alpha={best_alpha}")

        # 4. Predict
        pred_target = model_fold.predict(X_test)
        pred_raw = inverse_target(pred_target, target_transform, TARGET_EPS)
        pred_raw = np.maximum(pred_raw, 0.0)
        
        # 5. Calibrate (Isotonic) - key step: fit only on training fold!
        # Predict on training set to feed calibrator
        train_preds_target = model_fold.predict(X_train)
        train_preds_raw = inverse_target(train_preds_target, target_transform, TARGET_EPS)
        train_preds_raw = np.maximum(train_preds_raw, 0.0)
        
        calib_fold = IsotonicRegression(out_of_bounds="clip")
        calib_fold.fit(train_preds_raw, y_train, sample_weight=w_train)
        
        # Apply calibration to test sample
        pred_final = calib_fold.predict(pred_raw)
        pred_final = np.maximum(pred_final, 0.0)
        
        y_cv_preds[test_ix] = pred_final

    cv_r2, cv_rmse = evaluate_predictions(y, y_cv_preds)
    print(f"LOO-CV R2 : {cv_r2:.4f}  <-- (True Prediction Ability)")
    print(f"LOO-CV RMSE: {cv_rmse:.4f} uM")
    
    # Save CV predictions for inspection
    cv_out = args.pred_out.with_name(args.pred_out.stem + "_LOO_CV.csv")
    pd.DataFrame({
        "filename": df["filename"], 
        "conc_true": y, 
        "conc_pred_cv": y_cv_preds,
        "residual_cv": y_cv_preds - y
    }).to_csv(cv_out, index=False)
    print(f"Saved LOO-CV predictions -> {cv_out}")

    save_artifacts(
        df,
        y,
        y_pred_raw,
        y_pred_calibrated,
        model,
        calibrator,
        model_out=args.model_out,
        pred_out=args.pred_out,
        coef_out=args.coef_out,
        g_mode=g_mode,
        alpha=model.named_steps["regressor"].alpha_ if hasattr(model.named_steps["regressor"], "alpha_") else (alpha_grid[0] if alpha_grid else 0.0),
        target_transform=target_transform,
        target_eps=TARGET_EPS,
        use_bg=use_bg,
        feature_names_active=tuple(feature_names(use_bg)),
        alphas_tried=alpha_grid,
        weight_eps=WEIGHT_EPS,
    )
    print(f"Saved model bundle -> {args.model_out}")
    print(f"Saved predictions -> {args.pred_out}")
    print(f"Saved coefficients -> {args.coef_out}")


def main() -> None:
    cfg = TrainConfig()
    args = parse_args(cfg)

    if args.method == "exponential":
        run_exponential(args, cfg)
    else:
        run_ridge(args, cfg)


if __name__ == "__main__":
    main()
