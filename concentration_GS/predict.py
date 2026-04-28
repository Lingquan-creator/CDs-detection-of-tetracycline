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

import matplotlib
import numpy as np
import pandas as pd

from config import PredictConfig
from physics_features import build_feature_frame, extract_concentration_from_name, has_background_columns
from train import build_probability_frame, load_model_bundle, lognormal_summary, posterior_from_bundle, shifted_lognormal_logpdf

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


CSV_FLOAT_FORMAT = "%.6g"
FONT_FALLBACKS = [
    "Droid Sans Fallback",
    "AR PL UMing CN",
    "AR PL SungtiL GB",
    "Noto Sans CJK SC",
    "WenQuanYi Zen Hei",
    "SimHei",
    "Microsoft YaHei",
    "DejaVu Sans",
]


def configure_matplotlib_output() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FALLBACKS,
            "axes.unicode_minus": False,
        }
    )


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


configure_matplotlib_output()


def parse_args(default_cfg: PredictConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict concentration posterior using the amortized GS model.")
    parser.add_argument("--model", type=Path, default=default_cfg.model_path, help="Model bundle path.")
    parser.add_argument("--roi-csv", type=Path, default=default_cfg.default_roi_csv, help="Input ROI statistics CSV.")
    parser.add_argument("--out", type=Path, help="Summary CSV output path.")
    parser.add_argument("--grid-out", type=Path, help="Probability CSV output path.")
    parser.add_argument("--plot-out", type=Path, help="Prediction visualization PNG output path.")
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


def resolve_output_path(path: Path | None, default_path: Path, output_dir: Path) -> Path:
    chosen = (path or default_path).expanduser()
    if not chosen.is_absolute():
        chosen = output_dir / chosen
    return chosen.resolve()


def draw_prediction_plot(summary_df: pd.DataFrame, out_path: Path, confidence: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    mean_col = "posterior_mean"
    ci_low_col = f"ci_{int(confidence * 100):02d}_low"
    ci_high_col = f"ci_{int(confidence * 100):02d}_high"
    y_pred = summary_df[mean_col].to_numpy(dtype=np.float64)

    if "conc_true" in summary_df.columns:
        y_true = summary_df["conc_true"].to_numpy(dtype=np.float64)
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        if np.any(valid):
            x = y_true[valid]
            y = y_pred[valid]
            axes[0].scatter(x, y, s=24, alpha=0.8, color="#1f77b4")
            low = float(min(np.min(x), np.min(y)))
            high = float(max(np.max(x), np.max(y)))
            axes[0].plot([low, high], [low, high], "k--", linewidth=1.0)
            axes[0].set_xlabel("True concentration (uM)")
            axes[0].set_ylabel("Posterior mean (uM)")
            axes[0].set_title("GS Prediction vs Truth")
            axes[0].grid(alpha=0.25)

            ci_low = summary_df[ci_low_col].to_numpy(dtype=np.float64)[valid]
            ci_high = summary_df[ci_high_col].to_numpy(dtype=np.float64)[valid]
            idx = np.arange(len(x))
            axes[1].errorbar(
                idx,
                y,
                yerr=[y - ci_low, ci_high - y],
                fmt="o",
                markersize=3,
                alpha=0.7,
                ecolor="#ff7f0e",
                color="#1f77b4",
            )
            axes[1].set_xlabel("Sample index")
            axes[1].set_ylabel("Posterior mean +/- CI (uM)")
            axes[1].set_title(f"Uncertainty ({int(confidence * 100)}% CI)")
            axes[1].grid(alpha=0.25)
        else:
            axes[0].text(0.5, 0.5, "No finite labels", ha="center", va="center")
            axes[1].text(0.5, 0.5, "No finite labels", ha="center", va="center")
    else:
        axes[0].hist(y_pred[np.isfinite(y_pred)], bins=20, color="#1f77b4", alpha=0.8)
        axes[0].set_xlabel("Posterior mean (uM)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Prediction Distribution")
        axes[0].grid(alpha=0.25)

        posterior_std = summary_df["posterior_std"].to_numpy(dtype=np.float64)
        axes[1].scatter(y_pred, posterior_std, s=16, alpha=0.7, color="#d62728")
        axes[1].set_xlabel("Posterior mean (uM)")
        axes[1].set_ylabel("Posterior std (uM)")
        axes[1].set_title("Uncertainty vs Prediction")
        axes[1].grid(alpha=0.25)

    format_figure_numbers(fig)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot        -> {out_path}")


def main() -> None:
    cfg = PredictConfig()
    args = parse_args(cfg)

    model_path = args.model.expanduser()
    if not model_path.is_absolute():
        model_path = cfg.output_dir / model_path
    model_path = model_path.resolve()

    bundle = load_model_bundle(model_path)
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

    out_path = resolve_output_path(args.out, cfg.summary_output_path_for(args.roi_csv), cfg.output_dir)
    grid_out_path = resolve_output_path(args.grid_out, cfg.probability_output_path_for(args.roi_csv), cfg.output_dir)
    plot_out_path = resolve_output_path(args.plot_out, cfg.plot_output_path_for(args.roi_csv), cfg.output_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid_out_path.parent.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(out_path, index=False, float_format=CSV_FLOAT_FORMAT)
    prob_df.to_csv(grid_out_path, index=False, float_format=CSV_FLOAT_FORMAT)
    draw_prediction_plot(summary_df, plot_out_path, confidence)

    print(
        f"Loaded model: {model_path} "
        f"(g_mode={g_mode}, use_bg={use_bg}, bins={len(bundle['reference_concentrations'])})"
    )
    print(f"Saved summary CSV -> {out_path}")
    print(f"Saved prob CSV    -> {grid_out_path}")


if __name__ == "__main__":
    main()
