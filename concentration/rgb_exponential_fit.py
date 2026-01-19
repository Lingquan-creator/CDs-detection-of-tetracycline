"""
Train a physics-informed Ridge regression model that maps ROI RGB statistics to concentration.

Improvement highlights (see improve4.md):
- 背景扣除：始终使用信号=ROI-BG 的 ln(signal) 特征 (bg_* 列必须存在)。
- 目标：直接预测浓度 (Identity fit)。
- 低浓度加权：样本权重 1/(conc+eps) 强调低浓度精度。
- 调参与单调校准：RidgeCV 搜索 alpha（并可扫 g-mode），后置 IsotonicRegression 保证非负、单调输出。

Outputs (under concentration/rgb_fit_output/):
- concentration_model_v1.pkl   : joblib bundle containing the pipeline、校准器与元数据
- ridge_predictions.csv        : per-sample predicted concentration（含原始/校准列）
- ridge_coefficients.csv       : feature weights of the trained Ridge regressor (on standardized features)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
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
    parser = argparse.ArgumentParser(description="Train physics-informed Ridge model for concentration estimation.")
    parser.add_argument("--roi-csv", type=Path, default=default_cfg.roi_csv, help="Path to roi_detailed_statistics.csv")
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


def main() -> None:
    cfg = TrainConfig()
    args = parse_args(cfg)

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


if __name__ == "__main__":
    main()
