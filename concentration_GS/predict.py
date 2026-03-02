"""
Batch inference for the concentration_GS amortized posterior model.

Outputs:
- summary CSV with posterior mean/std/credible interval
- probability CSV with discrete posterior mass over reference concentration bins
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from config import PredictConfig
from physics_features import build_feature_frame, extract_concentration_from_name, has_background_columns
from train import build_probability_frame, load_model_bundle, lognormal_summary, posterior_from_bundle, shifted_lognormal_logpdf


def parse_args(default_cfg: PredictConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict concentration posterior using the amortized GS model.")
    parser.add_argument("--model", type=Path, default=default_cfg.model_path, help="Model bundle path.")
    parser.add_argument("--roi-csv", type=Path, default=default_cfg.default_roi_csv, help="Input ROI statistics CSV.")
    parser.add_argument("--out", type=Path, help="Summary CSV output path.")
    parser.add_argument("--grid-out", type=Path, help="Probability CSV output path.")
    parser.add_argument("--confidence", type=float, help="Credible interval coverage. Default: use model bundle value.")
    return parser.parse_args()


def ensure_filename_column(df: pd.DataFrame) -> pd.DataFrame:
    if "filename" in df.columns:
        return df
    df = df.copy()
    df["filename"] = [f"sample_{idx:04d}" for idx in range(len(df))]
    return df


def select_model_features(feature_df: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    missing = [name for name in feature_names if name not in feature_df.columns]
    if missing:
        raise ValueError(f"Input CSV cannot build required model features: {missing}")
    return feature_df.loc[:, list(feature_names)]


def build_inference_summary(
    df: pd.DataFrame,
    mu_log: np.ndarray,
    std_log: np.ndarray,
    *,
    reference_concentrations: Sequence[float],
    confidence: float,
    target_eps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = lognormal_summary(mu_log, std_log, confidence=confidence, target_eps=target_eps)
    prob_df = build_probability_frame(
        df["filename"].tolist(),
        mu_log,
        std_log,
        reference_concentrations,
        target_eps=target_eps,
    )

    most_prob = (
        prob_df.sort_values(["filename", "bin_probability"], ascending=[True, False])
        .drop_duplicates(subset=["filename"])
        .rename(columns={"ref_concentration": "map_bin_concentration", "bin_probability": "map_bin_probability"})
        .loc[:, ["filename", "map_bin_concentration", "map_bin_probability"]]
    )

    out_df = pd.DataFrame(
        {
            "filename": df["filename"],
            "posterior_log_mean": mu_log,
            "posterior_log_std": std_log,
            "posterior_mean": summary["posterior_mean"],
            "posterior_median": summary["posterior_median"],
            "posterior_mode": summary["posterior_mode"],
            "posterior_std": summary["posterior_std"],
            f"ci_{int(confidence * 100):02d}_low": summary["ci_low"],
            f"ci_{int(confidence * 100):02d}_high": summary["ci_high"],
        }
    )
    out_df = out_df.merge(most_prob, on="filename", how="left")

    conc_true = df["filename"].apply(extract_concentration_from_name)
    if conc_true.notna().any():
        conc_true_np = conc_true.to_numpy(dtype=np.float64)
        out_df["conc_true"] = conc_true_np
        out_df["residual_mean"] = out_df["posterior_mean"] - out_df["conc_true"]
        out_df["abs_error_mean"] = np.abs(out_df["residual_mean"])
        out_df["nll"] = -shifted_lognormal_logpdf(conc_true_np, mu_log, std_log, target_eps)

    return out_df, prob_df


def main() -> None:
    cfg = PredictConfig()
    args = parse_args(cfg)

    bundle = load_model_bundle(args.model)
    confidence = float(args.confidence if args.confidence is not None else bundle.get("confidence", cfg.confidence))
    if not (0.0 < confidence < 1.0):
        raise ValueError("--confidence must be between 0 and 1.")

    df = pd.read_csv(args.roi_csv)
    if df.empty:
        raise ValueError("ROI CSV is empty.")
    df = ensure_filename_column(df)

    g_mode = str(bundle["g_mode"])
    use_bg = bool(bundle["use_bg"])
    if use_bg and not has_background_columns(df, g_mode):
        raise ValueError("Model expects background columns, but the input CSV does not contain them.")

    feature_df = build_feature_frame(df, g_mode=g_mode, use_bg=use_bg)
    expected_features = [str(name) for name in bundle["feature_names"]]
    model_features = select_model_features(feature_df, expected_features)

    mu_log, std_log = posterior_from_bundle(bundle, model_features.to_numpy(dtype=np.float64))
    summary_df, prob_df = build_inference_summary(
        df,
        mu_log,
        std_log,
        reference_concentrations=np.asarray(bundle["reference_concentrations"], dtype=np.float64),
        confidence=confidence,
        target_eps=float(bundle["target_eps"]),
    )

    out_path = args.out or cfg.summary_output_path_for(args.roi_csv)
    grid_out_path = args.grid_out or cfg.probability_output_path_for(args.roi_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid_out_path.parent.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(out_path, index=False)
    prob_df.to_csv(grid_out_path, index=False)

    print(
        f"Loaded model: {args.model} "
        f"(g_mode={g_mode}, use_bg={use_bg}, bins={len(bundle['reference_concentrations'])})"
    )
    print(f"Saved summary CSV -> {out_path}")
    print(f"Saved prob CSV    -> {grid_out_path}")


if __name__ == "__main__":
    main()
