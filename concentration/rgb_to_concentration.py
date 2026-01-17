"""
Predict concentration from ROI RGB statistics using the physics-informed Ridge model.

Usage examples:
- Batch mode (ROI CSV): python rgb_to_concentration.py --roi-csv ../pic_RGB/roi_results/roi_detailed_statistics.csv
- Single sample:       python rgb_to_concentration.py --model rgb_fit_output/concentration_model_v1.pkl --r 20 --g1 45 --b 10 --var-g 300
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import joblib
import pandas as pd

from config import PredictConfig
from physics_features import DEFAULT_G_MODE, build_feature_frame, build_feature_vector, extract_concentration_from_name


def load_model_bundle(path: Path, g_mode_override: Optional[str]) -> Tuple[object, str]:
    obj = joblib.load(path)
    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
        g_mode = obj.get("g_mode", DEFAULT_G_MODE)
    else:
        model = obj
        g_mode = DEFAULT_G_MODE

    if g_mode_override:
        g_mode = g_mode_override
    if g_mode not in ("g1", "avg"):
        g_mode = DEFAULT_G_MODE
    return model, g_mode


def predict_batch(model: object, roi_csv: Path, out_csv: Path, g_mode: str) -> None:
    df = pd.read_csv(roi_csv)
    features = build_feature_frame(df, g_mode=g_mode)
    preds = model.predict(features.to_numpy())

    out_df = pd.DataFrame({"filename": df["filename"], "conc_pred": preds})

    conc_true = df["filename"].apply(extract_concentration_from_name)
    if conc_true.notna().any():
        out_df["conc_true"] = conc_true
        out_df["residual"] = preds - conc_true

    out_df.to_csv(out_csv, index=False)
    print(f"Saved predictions -> {out_csv}")


def predict_single(model: object, g_mode: str, args: argparse.Namespace) -> None:
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
    )
    pred = float(model.predict(feature_row)[0])
    print(f"[Single] Predicted concentration: {pred:.4f} uM (g-mode={g_mode})")


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
    return parser.parse_args()


def main() -> None:
    cfg = PredictConfig()
    args = parse_args(cfg)
    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")
    model, g_mode = load_model_bundle(args.model, args.g_mode)
    print(f"Loaded model from {args.model} (g-mode={g_mode})")

    if args.roi_csv:
        out_csv = args.out or cfg.output_path_for(args.roi_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        predict_batch(model, args.roi_csv, out_csv, g_mode)
    elif args.r is not None or args.g1 is not None or args.b is not None:
        predict_single(model, g_mode, args)
    else:
        default_csv = cfg.default_roi_csv
        if default_csv.exists():
            print("No inputs provided, running batch prediction on default roi_detailed_statistics.csv ...")
            out_csv = cfg.output_path_for(default_csv)
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            predict_batch(model, default_csv, out_csv, g_mode)
        else:
            print("No work to do: provide --roi-csv for batch mode or --r/--g1/--b/--var-g for single mode.")


if __name__ == "__main__":
    main()
