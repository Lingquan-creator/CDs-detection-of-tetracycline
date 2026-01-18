"""
Predict concentration from ROI RGB statistics using the physics-informed Ridge model.

Key behaviors (see improve4.md):
- 自动读取模型元数据：背景模式(use_bg)、目标变换(target_transform)、alpha 与单调校准器。
- 有 bg_* 列时使用背景扣除特征；否则退回原始 ln/ratio 特征。
- 非负约束：log-target 逆变换 + 单调校准器后再裁剪到 0 以上。

Usage examples:
- Batch mode:   python rgb_to_concentration.py --roi-csv pic_RGB/roi_results/roi_detailed_statistics.csv
- Single point: python rgb_to_concentration.py --model concentration/rgb_fit_output/concentration_model_v1.pkl --r 20 --g1 45 --b 10 --var-g 300
  (背景模式下可额外传入 --bg-r --bg-g1 --bg-b 等)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from config import CLIP_MIN, PredictConfig, TARGET_EPS
from physics_features import (
    DEFAULT_G_MODE,
    build_feature_frame,
    build_feature_vector,
    extract_concentration_from_name,
    has_background_columns,
)


def load_model_bundle(path: Path, g_mode_override: Optional[str]) -> Tuple[object, str, str, float, bool, Optional[object]]:
    obj = joblib.load(path)
    target_transform: str = "identity"
    target_eps: float = TARGET_EPS
    use_bg: bool = False
    calibrator: Optional[object] = None
    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
        g_mode = obj.get("g_mode", DEFAULT_G_MODE)
        target_transform = obj.get("target_transform", target_transform)
        target_eps = obj.get("target_eps", TARGET_EPS)
        use_bg = obj.get("use_bg", use_bg)
        calibrator = obj.get("isotonic", None)
    else:
        model = obj
        g_mode = DEFAULT_G_MODE

    if g_mode_override:
        g_mode = g_mode_override
    if g_mode not in ("g1", "avg"):
        g_mode = DEFAULT_G_MODE
    return model, g_mode, target_transform, target_eps, use_bg, calibrator


def inverse_target(pred: np.ndarray, transform: Optional[str], target_eps: float) -> np.ndarray:
    """Reverse log-target transform (if present) and enforce non-negative concentration."""
    if transform == "log":
        conc = np.exp(pred) - target_eps
    else:
        conc = pred
    return np.maximum(conc, 0.0)


def apply_calibrator(pred: np.ndarray, calibrator: Optional[object]) -> np.ndarray:
    if calibrator is None:
        return np.maximum(pred, 0.0)
    calibrated = calibrator.predict(pred)
    return np.maximum(calibrated, 0.0)


def predict_batch(
    model: object,
    roi_csv: Path,
    out_csv: Path,
    g_mode: str,
    *,
    target_transform: Optional[str],
    target_eps: float,
    use_bg: bool,
    calibrator: Optional[object],
) -> None:
    df = pd.read_csv(roi_csv)
    
    # Handle background column availability
    dataset_use_bg = use_bg
    if use_bg and not has_background_columns(df, g_mode):
        print("WARNING: Model trained with background subtraction, but input CSV lacks bg_* columns.")
        print("         Falling back to raw features (assuming bg=0/negligible).")
        dataset_use_bg = False

    features = build_feature_frame(df, g_mode=g_mode, use_bg=dataset_use_bg, clip_min=CLIP_MIN)
    preds_target = model.predict(features.to_numpy())
    preds = inverse_target(np.asarray(preds_target), target_transform, target_eps)
    preds = apply_calibrator(preds, calibrator)

    out_df = pd.DataFrame({"filename": df["filename"], "conc_pred": preds})

    conc_true = df["filename"].apply(extract_concentration_from_name)
    if conc_true.notna().any():
        out_df["conc_true"] = conc_true
        out_df["residual"] = preds - conc_true

    out_df.to_csv(out_csv, index=False)
    print(f"Saved predictions -> {out_csv}")


def predict_single(
    model: object,
    g_mode: str,
    args: argparse.Namespace,
    *,
    target_transform: Optional[str],
    target_eps: float,
    use_bg: bool,
    calibrator: Optional[object],
) -> None:
    if args.r is None or args.g1 is None or args.b is None:
        raise ValueError("Single-sample mode requires --r, --g1, and --b.")
    if args.var_g is None:
        raise ValueError("Single-sample mode requires --var-g to build Var_G feature.")

    feature_row = build_feature_vector(
        args.r,
        args.g1,
        args.b,
        var_g1=args.var_g,
        g2=args.g2,
        var_g2=args.var_g2,
        g_mode=g_mode,
        use_bg=use_bg,
        clip_min=CLIP_MIN,
        bg_r=args.bg_r or 0.0,
        bg_g1=args.bg_g1 or 0.0,
        bg_g2=args.bg_g2 or 0.0,
        bg_b=args.bg_b or 0.0,
    )
    pred_target = model.predict(feature_row)[0]
    pred = float(inverse_target(np.array([pred_target]), target_transform, target_eps)[0])
    pred = float(apply_calibrator(np.array([pred]), calibrator)[0])
    print(
        f"[Single] Predicted concentration: {pred:.4f} uM "
        f"(g-mode={g_mode}, target={target_transform or 'identity'}, use_bg={use_bg})"
    )


def parse_args(default_cfg: PredictConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict concentration using the trained Ridge model.")
    parser.add_argument("--model", type=Path, default=default_cfg.model_path, help="Path to joblib model bundle.")
    parser.add_argument("--roi-csv", type=Path, help="ROI CSV for batch prediction.")
    parser.add_argument("--out", type=Path, help="Output CSV path for batch mode.")
    parser.add_argument("--g-mode", choices=["g1", "avg"], help="Override green feature mode (defaults to model metadata).")
    parser.add_argument("--r", type=float, help="Single-sample R channel mean.")
    parser.add_argument("--g1", type=float, help="Single-sample G1 channel mean.")
    parser.add_argument("--g2", type=float, help="Single-sample G2 channel mean (optional, used if g-mode=avg).")
    parser.add_argument("--b", type=float, help="Single-sample B channel mean.")
    parser.add_argument("--var-g", type=float, help="Single-sample variance for green channel (G1 or average).")
    parser.add_argument("--var-g2", type=float, help="Single-sample variance for G2 (optional, used if g-mode=avg).")
    parser.add_argument("--bg-r", type=float, help="Background R mean (for background mode).")
    parser.add_argument("--bg-g1", type=float, help="Background G1 mean (for background mode).")
    parser.add_argument("--bg-g2", type=float, help="Background G2 mean (for background mode, g-mode=avg).")
    parser.add_argument("--bg-b", type=float, help="Background B mean (for background mode).")
    return parser.parse_args()


def main() -> None:
    cfg = PredictConfig()
    args = parse_args(cfg)
    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")
    model, g_mode, target_transform, target_eps, use_bg, calibrator = load_model_bundle(args.model, args.g_mode)
    print(
        f"Loaded model from {args.model} "
        f"(g-mode={g_mode}, target={target_transform or 'identity'}, use_bg={use_bg}, calibrator={'yes' if calibrator else 'no'})"
    )

    if args.roi_csv:
        out_csv = args.out or cfg.output_path_for(args.roi_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        predict_batch(
            model,
            args.roi_csv,
            out_csv,
            g_mode,
            target_transform=target_transform,
            target_eps=target_eps,
            use_bg=use_bg,
            calibrator=calibrator,
        )
    elif args.r is not None or args.g1 is not None or args.b is not None:
        predict_single(
            model,
            g_mode,
            args,
            target_transform=target_transform,
            target_eps=target_eps,
            use_bg=use_bg,
            calibrator=calibrator,
        )
    else:
        default_csv = cfg.default_roi_csv
        if default_csv.exists():
            print("No inputs provided, running batch prediction on default roi_detailed_statistics.csv ...")
            out_csv = cfg.output_path_for(default_csv)
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            predict_batch(
                model,
                default_csv,
                out_csv,
                g_mode,
                target_transform=target_transform,
                target_eps=target_eps,
                use_bg=use_bg,
                calibrator=calibrator,
            )
        else:
            print("No work to do: provide --roi-csv for batch mode or --r/--g1/--b/--var-g for single mode.")


if __name__ == "__main__":
    main()
