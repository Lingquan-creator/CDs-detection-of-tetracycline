"""Train polynomial ridge regressors to map pic_2 RGB space to pic_1 RGB space.

Usage (from repo root):
    python -m multi_device_rgb.device_mapping.train_mapper
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

if __name__ == "__main__" and __package__ is None:
    # Allow running as a script (python device_mapping/train_mapper.py) by
    # ensuring the package root is on sys.path.
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    __package__ = "multi_device_rgb.device_mapping"

import joblib
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .config import MappingConfig
from .data import CHANNELS_4, CHANNELS_3, load_aligned_arrays


def build_pipeline(
    degree: int, alpha_grid: Tuple[float, ...], n_samples: int
) -> Pipeline:
    """Create the preprocessing + regressor pipeline with graceful fallback for tiny datasets."""
    steps = [
        ("poly", PolynomialFeatures(degree=degree, include_bias=True)),
        ("scaler", StandardScaler()),
    ]

    if n_samples >= 2:
        ridge = RidgeCV(
            alphas=alpha_grid,
            cv=LeaveOneOut(),
            scoring="neg_mean_squared_error",
        )
    else:
        # RidgeCV cannot run LOO when only one sample; pick a reasonable default alpha
        ridge = Ridge(alpha=alpha_grid[len(alpha_grid) // 2])
    steps.append(("ridge", ridge))
    return Pipeline(steps)


def train_models(config: MappingConfig, allow_single_sample: bool = False):
    (
        X_pic2,
        y_pic1,
        concentrations,
        df1_aligned,
        df2_aligned,
        df1_full,
        df2_full,
    ) = load_aligned_arrays(config)

    feature_names = list(CHANNELS_4 if config.use_four_channels else CHANNELS_3)
    n_samples = X_pic2.shape[0]

    if n_samples < 2 and not allow_single_sample:
        raise ValueError(
            "Need at least two overlapping concentrations to train a mapper. "
            f"Found {n_samples} overlap: {concentrations.tolist()}. "
            f"pic_1 concentrations: {sorted(df1_full['concentration'].unique().tolist())}; "
            f"pic_2 concentrations: {sorted(df2_full['concentration'].unique().tolist())}. "
            "Add more matched samples or set MappingConfig.matching_concentrations to a valid intersection."
        )

    models: Dict[str, Pipeline] = {}
    metrics = []
    preds = np.zeros_like(y_pic1)

    for idx, ch in enumerate(feature_names):
        pipeline = build_pipeline(config.polynomial_degree, config.alpha_grid, n_samples)
        pipeline.fit(X_pic2, y_pic1[:, idx])

        y_pred = pipeline.predict(X_pic2)
        preds[:, idx] = y_pred

        # Extract chosen alpha when possible
        alpha = None
        ridge = pipeline.named_steps["ridge"]
        if isinstance(ridge, RidgeCV):
            alpha = float(ridge.alpha_)
        else:
            alpha = float(ridge.alpha)

        rmse = float(np.sqrt(mean_squared_error(y_pic1[:, idx], y_pred)))
        r2 = float(r2_score(y_pic1[:, idx], y_pred)) if n_samples > 1 else float("nan")

        metrics.append({"channel": ch, "alpha": alpha, "rmse": rmse, "r2": r2})
        models[ch] = pipeline

    bundle = {
        "models": models,
        "feature_names": feature_names,
        "use_four_channels": config.use_four_channels,
        "polynomial_degree": config.polynomial_degree,
        "alpha_grid": config.alpha_grid,
        "min_clip": config.min_clip,
        "concentrations": concentrations.tolist(),
        "train_pic2": X_pic2,
        "train_pic1": y_pic1,
        "train_metrics": metrics,
    }

    out_dir = config.ensure_results_dir()
    model_path = out_dir / "ridge_mapper.pkl"
    joblib.dump(bundle, model_path)

    # Save a simple report for quick inspection
    report = {
        "n_samples": int(n_samples),
        "feature_names": feature_names,
        "metrics": metrics,
        "concentrations": concentrations.tolist(),
    }
    (out_dir / "ridge_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return model_path, report


def parse_args():
    parser = argparse.ArgumentParser(description="Train polynomial ridge color mapper.")
    parser.add_argument("--degree", type=int, default=None, help="Polynomial degree (default from config).")
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=None,
        help="Alpha grid for RidgeCV (default from config).",
    )
    parser.add_argument(
        "--three-channels",
        action="store_true",
        help="Use 3-channel RGB (G = (G1+G2)/2) instead of 4-channel Bayer.",
    )
    parser.add_argument(
        "--allow-single-sample",
        action="store_true",
        help="Skip the overlap size check (keeps previous behavior when only one matching concentration exists).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = MappingConfig()
    if args.degree is not None:
        config.polynomial_degree = args.degree
    if args.alphas is not None:
        config.alpha_grid = tuple(args.alphas)
    if args.three_channels:
        config.use_four_channels = False

    model_path, report = train_models(config, allow_single_sample=args.allow_single_sample)
    print(f"[OK] Saved mapper to: {model_path}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
