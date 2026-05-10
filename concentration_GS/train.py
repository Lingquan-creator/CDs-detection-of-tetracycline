"""
Train an amortized posterior model for concentration prediction and evaluate with LOO.

The model learns q_phi(log c | x) as a Gaussian whose parameters are produced by
an MLP. Predictions are converted back to concentration space as a shifted
log-normal posterior, so the output is a probability distribution instead of only
one point estimate.
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut

from config import TARGET_EPS, TrainConfig
from physics_features import (
    build_feature_frame,
    extract_concentration_from_name,
    feature_names,
    has_background_columns,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402


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


class AmortizedPosteriorNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, min_std: float) -> None:
        super().__init__()
        self.min_std = float(min_std)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.std_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(x)
        mu = self.mu_head(hidden).squeeze(-1)
        std = self.min_std + F.softplus(self.std_head(hidden).squeeze(-1))
        return mu, std


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_sample_weights(y: np.ndarray, weight_eps: float, weight_beta: float = 1.0) -> np.ndarray:
    if weight_eps < 0:
        raise ValueError("--weight-eps must be >= 0.")
    denom = weight_beta * np.asarray(y, dtype=float) + float(weight_eps)
    if np.any(denom <= 0):
        raise ValueError("weight_beta * concentration + weight_eps must stay positive.")
    weights = 1.0 / denom
    return weights / np.mean(weights)


def resolve_use_bg(df: pd.DataFrame, g_mode: str, use_bg_arg: str) -> bool:
    if use_bg_arg == "true":
        if not has_background_columns(df, g_mode):
            raise ValueError(f"--use-bg true but background columns are missing for g_mode={g_mode}.")
        return True
    if use_bg_arg == "false":
        return False
    return has_background_columns(df, g_mode)


def select_feature_set(feature_df: pd.DataFrame, *, use_bg: bool, feature_set: str) -> pd.DataFrame:
    if feature_set == "full":
        return feature_df
    if feature_set not in {
        "compact",
        "compact_exposure",
    }:
        raise ValueError(f"Unsupported feature set: {feature_set}")

    no_exposure_columns = (
        ("ln_R_sig", "ln_G_sig", "ln_B_sig", "Ratio_RG", "Delta_RG", "Delta_GB")
        if use_bg
        else ("ln_R", "ln_G", "ln_B", "Ratio_RG", "Delta_RG", "Delta_GB")
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
    selected_columns = exposure_columns if feature_set == "compact_exposure" else no_exposure_columns
    missing = [column for column in selected_columns if column not in feature_df.columns]
    if missing:
        raise ValueError(f"Cannot build {feature_set} feature set; missing columns: {missing}")
    return feature_df.loc[:, list(selected_columns)]


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

    y_true = df["conc_true"].to_numpy(dtype=float)
    return df, y_true


def standardize_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (X - mean) / std, mean, std


def standardize_target_log(y: np.ndarray) -> Tuple[np.ndarray, float, float]:
    y_log = np.log(np.asarray(y, dtype=float) + TARGET_EPS)
    mean = float(np.mean(y_log))
    std = float(np.std(y_log))
    if std < 1e-8:
        std = 1.0
    return (y_log - mean) / std, mean, std


def gaussian_nll(
    target: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
    std_reg_lambda: float = 0.0,
) -> torch.Tensor:
    var = std.pow(2)
    loss = 0.5 * torch.log(var) + 0.5 * (target - mu).pow(2) / var + 0.5 * math.log(2.0 * math.pi)
    if sample_weight is not None:
        loss = loss * sample_weight
    base_loss = loss.mean()
    if std_reg_lambda > 0.0:
        base_loss = base_loss + float(std_reg_lambda) * std.pow(2).mean()
    return base_loss


def fit_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    hidden_dim: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    min_std: float,
    std_reg_lambda: float,
    max_std_log: float,
    weight_eps: float,
    weight_beta: float,
    grad_clip_norm: float,
    seed: int,
    g_mode: str,
    use_bg: bool,
    active_feature_names: Optional[Sequence[str]] = None,
    reference_concentrations: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    set_global_seed(seed)

    X_std, feature_mean, feature_std = standardize_features(X)
    y_std, target_log_mean, target_log_std = standardize_target_log(y)
    sample_weight = make_sample_weights(y, weight_eps, weight_beta)

    x_tensor = torch.as_tensor(X_std, dtype=torch.float32)
    y_tensor = torch.as_tensor(y_std, dtype=torch.float32)
    w_tensor = torch.as_tensor(sample_weight, dtype=torch.float32)

    model = AmortizedPosteriorNet(
        input_dim=X.shape[1],
        hidden_dim=hidden_dim,
        dropout=dropout,
        min_std=min_std,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_loss = float("inf")
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(int(epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        mu, std = model(x_tensor)
        loss = gaussian_nll(y_tensor, mu, std, sample_weight=w_tensor, std_reg_lambda=std_reg_lambda)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
        optimizer.step()

        loss_value = float(loss.detach().cpu().item())
        if loss_value < best_loss - 1e-6:
            best_loss = loss_value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= int(patience):
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_mu, train_std = model(x_tensor)
        best_train_nll = float(gaussian_nll(y_tensor, train_mu, train_std, sample_weight=w_tensor).cpu().item())

    bundle = {
        "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "input_dim": int(X.shape[1]),
        "hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
        "min_std": float(min_std),
        "std_reg_lambda": float(std_reg_lambda),
        "max_std_log": float(max_std_log),
        "feature_mean": np.asarray(feature_mean, dtype=np.float64),
        "feature_std": np.asarray(feature_std, dtype=np.float64),
        "target_log_mean": float(target_log_mean),
        "target_log_std": float(target_log_std),
        "feature_names": tuple(active_feature_names if active_feature_names is not None else feature_names(use_bg)),
        "reference_concentrations": np.asarray(
            reference_concentrations if reference_concentrations is not None else np.sort(np.unique(y)),
            dtype=np.float64,
        ),
        "g_mode": str(g_mode),
        "use_bg": bool(use_bg),
        "weight_eps": float(weight_eps),
        "weight_beta": float(weight_beta),
        "target_eps": float(TARGET_EPS),
        "seed": int(seed),
        "epochs_requested": int(epochs),
        "epochs_trained": int(best_epoch + 1),
        "best_train_nll": best_train_nll,
        "best_train_loss": float(best_loss),
    }
    return bundle


def instantiate_model_from_bundle(bundle: Dict[str, object]) -> AmortizedPosteriorNet:
    model = AmortizedPosteriorNet(
        input_dim=int(bundle["input_dim"]),
        hidden_dim=int(bundle["hidden_dim"]),
        dropout=float(bundle["dropout"]),
        min_std=float(bundle["min_std"]),
    )
    state_dict = {name: tensor for name, tensor in bundle["state_dict"].items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_model_bundle(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found: {path}")
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        bundle = torch.load(path, map_location="cpu")
    if not isinstance(bundle, dict) or "state_dict" not in bundle:
        raise ValueError(f"Unexpected model bundle format: {path}")
    return bundle


def posterior_from_bundle(bundle: Dict[str, object], X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    feature_mean = np.asarray(bundle["feature_mean"], dtype=np.float64)
    feature_std = np.asarray(bundle["feature_std"], dtype=np.float64)
    X_std = (np.asarray(X, dtype=np.float64) - feature_mean) / feature_std

    model = instantiate_model_from_bundle(bundle)
    x_tensor = torch.as_tensor(X_std, dtype=torch.float32)
    with torch.no_grad():
        mu_std, std_std = model(x_tensor)

    mu_std_np = mu_std.detach().cpu().numpy().astype(np.float64)
    std_std_np = std_std.detach().cpu().numpy().astype(np.float64)

    target_log_mean = float(bundle["target_log_mean"])
    target_log_std = float(bundle["target_log_std"])
    mu_log = target_log_mean + target_log_std * mu_std_np
    max_std_log = bundle.get("max_std_log")
    std_log = np.maximum(target_log_std * std_std_np, 1e-6)
    if max_std_log is not None and np.isfinite(float(max_std_log)) and float(max_std_log) > 0.0:
        std_log = np.minimum(std_log, float(max_std_log))
    return mu_log, std_log


def lognormal_summary(
    mu_log: np.ndarray,
    std_log: np.ndarray,
    *,
    confidence: float,
    target_eps: float,
) -> Dict[str, np.ndarray]:
    alpha = (1.0 - float(confidence)) / 2.0
    z_low = float(norm.ppf(alpha))
    z_high = float(norm.ppf(1.0 - alpha))

    mean = np.exp(mu_log + 0.5 * std_log**2) - target_eps
    median = np.exp(mu_log) - target_eps
    mode = np.exp(mu_log - std_log**2) - target_eps
    variance = (np.exp(std_log**2) - 1.0) * np.exp(2.0 * mu_log + std_log**2)
    std = np.sqrt(np.maximum(variance, 0.0))
    lower = np.exp(mu_log + z_low * std_log) - target_eps
    upper = np.exp(mu_log + z_high * std_log) - target_eps

    return {
        "posterior_mean": np.maximum(mean, 0.0),
        "posterior_median": np.maximum(median, 0.0),
        "posterior_mode": np.maximum(mode, 0.0),
        "posterior_std": np.maximum(std, 0.0),
        "ci_low": np.maximum(lower, 0.0),
        "ci_high": np.maximum(upper, 0.0),
    }


def shifted_lognormal_logpdf(y: np.ndarray, mu_log: np.ndarray, std_log: np.ndarray, target_eps: float) -> np.ndarray:
    y_safe = np.maximum(np.asarray(y, dtype=np.float64), 0.0)
    transformed = np.log(y_safe + target_eps)
    log_pdf = norm.logpdf(transformed, loc=mu_log, scale=std_log) - transformed
    return log_pdf


def build_reference_bins(reference_concentrations: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref = np.sort(np.unique(np.asarray(reference_concentrations, dtype=np.float64)))
    if ref.size == 0:
        raise ValueError("reference_concentrations must not be empty.")
    if np.any(ref <= 0):
        mids = (ref[:-1] + ref[1:]) / 2.0
    else:
        mids = np.sqrt(ref[:-1] * ref[1:])
    left_edges = np.concatenate(([0.0], mids))
    right_edges = np.concatenate((mids, [np.inf]))
    return ref, left_edges, right_edges


def shifted_lognormal_cdf(x: float, mu_log: float, std_log: float, target_eps: float) -> float:
    if np.isneginf(x):
        return 0.0
    if np.isposinf(x):
        return 1.0
    if x <= 0:
        transformed = math.log(target_eps)
    else:
        transformed = math.log(x + target_eps)
    return float(norm.cdf((transformed - mu_log) / std_log))


def build_probability_frame(
    filenames: Sequence[str],
    mu_log: np.ndarray,
    std_log: np.ndarray,
    reference_concentrations: Sequence[float],
    *,
    target_eps: float,
) -> pd.DataFrame:
    ref, left_edges, right_edges = build_reference_bins(reference_concentrations)
    rows = []
    for row_idx, filename in enumerate(filenames):
        for bin_idx, ref_value in enumerate(ref):
            left = float(left_edges[bin_idx])
            right = float(right_edges[bin_idx])
            cdf_left = shifted_lognormal_cdf(left, float(mu_log[row_idx]), float(std_log[row_idx]), target_eps)
            cdf_right = shifted_lognormal_cdf(right, float(mu_log[row_idx]), float(std_log[row_idx]), target_eps)
            rows.append(
                {
                    "filename": filename,
                    "ref_concentration": ref_value,
                    "bin_left": left,
                    "bin_right": right,
                    "bin_probability": max(cdf_right - cdf_left, 0.0),
                    "cdf_right": cdf_right,
                }
            )
    return pd.DataFrame(rows)


def build_summary_frame(
    df: pd.DataFrame,
    mu_log: np.ndarray,
    std_log: np.ndarray,
    reference_concentrations: Sequence[float],
    *,
    confidence: float,
    target_eps: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
            "conc_true": df["conc_true"],
            "posterior_log_mean": mu_log,
            "posterior_log_std": std_log,
            "posterior_mean": summary["posterior_mean"],
            "posterior_median": summary["posterior_median"],
            "posterior_mode": summary["posterior_mode"],
            "posterior_std": summary["posterior_std"],
            f"ci_{int(confidence * 100):02d}_low": summary["ci_low"],
            f"ci_{int(confidence * 100):02d}_high": summary["ci_high"],
            "nll": -shifted_lognormal_logpdf(df["conc_true"].to_numpy(dtype=np.float64), mu_log, std_log, target_eps),
        }
    )
    out_df = out_df.merge(most_prob, on="filename", how="left")
    out_df["residual_mean"] = out_df["posterior_mean"] - out_df["conc_true"]
    out_df["abs_error_mean"] = np.abs(out_df["residual_mean"])
    out_df["residual_median"] = out_df["posterior_median"] - out_df["conc_true"]
    out_df["abs_error_median"] = np.abs(out_df["residual_median"])
    return out_df, prob_df


def evaluate_summary(summary_df: pd.DataFrame, point_col: str = "posterior_median") -> Dict[str, float]:
    y_true = summary_df["conc_true"].to_numpy(dtype=np.float64)
    y_pred = summary_df[point_col].to_numpy(dtype=np.float64)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mean_nll": float(np.mean(summary_df["nll"].to_numpy(dtype=np.float64))),
        "point_estimate": point_col,
    }


def run_leave_one_out(
    X: np.ndarray,
    y: np.ndarray,
    *,
    hidden_dim: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    min_std: float,
    std_reg_lambda: float,
    max_std_log: float,
    weight_eps: float,
    weight_beta: float,
    grad_clip_norm: float,
    seed: int,
    g_mode: str,
    use_bg: bool,
    active_feature_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    loo = LeaveOneOut()
    mu_log_pred = np.zeros_like(y, dtype=np.float64)
    std_log_pred = np.zeros_like(y, dtype=np.float64)

    for fold_idx, (train_idx, test_idx) in enumerate(loo.split(X)):
        fold_bundle = fit_model(
            X[train_idx],
            y[train_idx],
            hidden_dim=hidden_dim,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=epochs,
            patience=patience,
            min_std=min_std,
            std_reg_lambda=std_reg_lambda,
            max_std_log=max_std_log,
            weight_eps=weight_eps,
            weight_beta=weight_beta,
            grad_clip_norm=grad_clip_norm,
            seed=seed + fold_idx + 1,
            g_mode=g_mode,
            use_bg=use_bg,
            active_feature_names=active_feature_names,
        )
        mu_log_fold, std_log_fold = posterior_from_bundle(fold_bundle, X[test_idx])
        mu_log_pred[test_idx] = mu_log_fold
        std_log_pred[test_idx] = std_log_fold

    return mu_log_pred, std_log_pred


def draw_result_plot(
    train_summary: pd.DataFrame,
    loo_summary: pd.DataFrame,
    loo_prob_df: pd.DataFrame,
    *,
    train_metrics: Dict[str, float],
    loo_metrics: Dict[str, float],
    confidence: float,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ci_low_col = f"ci_{int(confidence * 100):02d}_low"
    ci_high_col = f"ci_{int(confidence * 100):02d}_high"

    fig = plt.figure(figsize=(14, 6), dpi=150)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.2])
    ax_train = fig.add_subplot(gs[0, 0])
    ax_loo = fig.add_subplot(gs[0, 1])
    ax_heatmap = fig.add_subplot(gs[0, 2])

    def plot_summary(ax: plt.Axes, summary_df: pd.DataFrame, title: str, metrics: Dict[str, float], color: str) -> None:
        x = np.maximum(summary_df["conc_true"].to_numpy(dtype=np.float64), TARGET_EPS)
        y = np.maximum(summary_df["posterior_median"].to_numpy(dtype=np.float64), TARGET_EPS)
        lower = np.maximum(summary_df[ci_low_col].to_numpy(dtype=np.float64), TARGET_EPS)
        upper = np.maximum(summary_df[ci_high_col].to_numpy(dtype=np.float64), TARGET_EPS)
        yerr = np.vstack((np.maximum(y - lower, 0.0), np.maximum(upper - y, 0.0)))

        ax.errorbar(x, y, yerr=yerr, fmt="o", color=color, ecolor=color, alpha=0.75, capsize=2, markersize=4)
        v_min = float(min(np.min(x), np.min(y), np.min(lower)))
        v_max = float(max(np.max(x), np.max(y), np.max(upper)))
        diag = np.geomspace(v_min, v_max, 200)
        ax.plot(diag, diag, "k--", linewidth=1.0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("True concentration (uM)")
        ax.set_ylabel("Posterior median (uM)")
        ax.set_title(
            f"{title}\nR2={metrics['r2']:.6g}, RMSE={metrics['rmse']:.6g}, "
            f"MAE={metrics['mae']:.6g}, NLL={metrics['mean_nll']:.6g}"
        )
        ax.grid(alpha=0.25)

    plot_summary(ax_train, train_summary, "Training Posterior Fit", train_metrics, "#1f77b4")
    plot_summary(ax_loo, loo_summary, "Leave-One-Out Posterior Fit", loo_metrics, "#d62728")

    sorted_loo = loo_summary.sort_values("conc_true").reset_index(drop=True)
    concentration_order = (
        loo_prob_df["ref_concentration"].drop_duplicates().sort_values().to_numpy(dtype=np.float64)
    )
    matrix = (
        loo_prob_df.pivot(index="filename", columns="ref_concentration", values="bin_probability")
        .reindex(index=sorted_loo["filename"], columns=concentration_order)
        .to_numpy(dtype=np.float64)
    )

    image = ax_heatmap.imshow(matrix, aspect="auto", cmap="magma", interpolation="nearest")
    ax_heatmap.set_xlabel("Reference concentration bin (uM)")
    ax_heatmap.set_ylabel("Sample (sorted by true concentration)")
    ax_heatmap.set_title("LOO Posterior Bin Probabilities")

    tick_positions = np.linspace(0, len(concentration_order) - 1, num=min(8, len(concentration_order)), dtype=int)
    ax_heatmap.set_xticks(tick_positions)
    ax_heatmap.set_xticklabels([format_plot_number(concentration_order[idx]) for idx in tick_positions], rotation=45, ha="right")
    ax_heatmap.set_yticks(np.arange(len(sorted_loo)))
    ax_heatmap.set_yticklabels(
        [format_plot_number(value) for value in sorted_loo["conc_true"].to_numpy(dtype=np.float64)]
    )
    colorbar = fig.colorbar(image, ax=ax_heatmap, fraction=0.046, pad=0.04, label="Probability")

    formatter = FuncFormatter(format_plot_number)
    for ax in (ax_train, ax_loo):
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
    colorbar.ax.yaxis.set_major_formatter(formatter)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args(default_cfg: TrainConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train amortized concentration posterior model and run LOO evaluation.")
    parser.add_argument("--roi-csv", type=Path, default=default_cfg.roi_csv, help="Input ROI statistics CSV.")
    parser.add_argument("--model-out", type=Path, default=default_cfg.model_path, help="Output model bundle path.")
    parser.add_argument("--train-out", type=Path, default=default_cfg.train_summary_path, help="Training summary CSV.")
    parser.add_argument("--loo-out", type=Path, default=default_cfg.loo_summary_path, help="LOO summary CSV.")
    parser.add_argument("--train-grid-out", type=Path, default=default_cfg.train_grid_path, help="Training probability CSV.")
    parser.add_argument("--loo-grid-out", type=Path, default=default_cfg.loo_grid_path, help="LOO probability CSV.")
    parser.add_argument("--plot-out", type=Path, default=default_cfg.plot_path, help="Evaluation figure output path.")
    parser.add_argument("--g-mode", choices=["g1", "avg"], default=default_cfg.g_mode, help="Green channel aggregation mode.")
    parser.add_argument(
        "--use-bg",
        choices=["auto", "true", "false"],
        default=default_cfg.use_bg,
        help="Use background-subtracted features when available.",
    )
    parser.add_argument("--hidden-dim", type=int, default=default_cfg.hidden_dim, help="Hidden layer width.")
    parser.add_argument(
        "--feature-set",
        choices=[
            "full",
            "compact",
            "compact_exposure",
        ],
        default=default_cfg.feature_set,
        help="Feature set used for training.",
    )
    parser.add_argument("--dropout", type=float, default=default_cfg.dropout, help="Dropout rate.")
    parser.add_argument("--learning-rate", type=float, default=default_cfg.learning_rate, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=default_cfg.weight_decay, help="AdamW weight decay.")
    parser.add_argument("--epochs", type=int, default=default_cfg.epochs, help="Maximum training epochs.")
    parser.add_argument("--patience", type=int, default=default_cfg.patience, help="Early stopping patience.")
    parser.add_argument("--min-std", type=float, default=default_cfg.min_std, help="Posterior std floor in z-space.")
    parser.add_argument("--std-reg-lambda", type=float, default=default_cfg.std_reg_lambda, help="L2 penalty on predicted posterior std.")
    parser.add_argument("--max-std-log", type=float, default=default_cfg.max_std_log, help="Posterior std ceiling after converting back to log-concentration space.")
    parser.add_argument("--weight-eps", type=float, default=default_cfg.weight_eps, help="Sample weight denominator offset.")
    parser.add_argument("--weight-beta", "--beta", dest="weight_beta", type=float, default=default_cfg.weight_beta, help="Sample weight beta in 1 / (weight_beta * conc + weight_eps).")
    parser.add_argument("--grad-clip-norm", type=float, default=default_cfg.grad_clip_norm, help="Gradient clipping norm.")
    parser.add_argument("--seed", type=int, default=default_cfg.seed, help="Random seed.")
    parser.add_argument("--confidence", type=float, default=default_cfg.confidence, help="Credible interval coverage.")
    return parser.parse_args()


def resolve_output_path(path: Path, output_dir: Path) -> Path:
    chosen = path.expanduser()
    if not chosen.is_absolute():
        chosen = output_dir / chosen
    return chosen.resolve()


def main() -> None:
    cfg = TrainConfig()
    args = parse_args(cfg)
    output_dir = cfg.output_dir.resolve()

    if not (0.0 < float(args.confidence) < 1.0):
        raise ValueError("--confidence must be between 0 and 1.")

    args.model_out = resolve_output_path(args.model_out, output_dir)
    args.train_out = resolve_output_path(args.train_out, output_dir)
    args.loo_out = resolve_output_path(args.loo_out, output_dir)
    args.train_grid_out = resolve_output_path(args.train_grid_out, output_dir)
    args.loo_grid_out = resolve_output_path(args.loo_grid_out, output_dir)
    args.plot_out = resolve_output_path(args.plot_out, output_dir)

    df, y_true = load_training_dataframe(args.roi_csv)
    use_bg = resolve_use_bg(df, args.g_mode, args.use_bg)
    feature_df = build_feature_frame(df, g_mode=args.g_mode, use_bg=use_bg)
    feature_df = select_feature_set(feature_df, use_bg=use_bg, feature_set=args.feature_set)
    X = feature_df.to_numpy(dtype=np.float64)
    reference_concentrations = np.sort(np.unique(y_true))

    print(f"Training data: {args.roi_csv} (n={len(df)})")
    print(f"Features: g_mode={args.g_mode}, use_bg={use_bg}, feature_set={args.feature_set}, columns={list(feature_df.columns)}")

    bundle = fit_model(
        X,
        y_true,
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        epochs=int(args.epochs),
        patience=int(args.patience),
        min_std=float(args.min_std),
        std_reg_lambda=float(args.std_reg_lambda),
        max_std_log=float(args.max_std_log),
        weight_eps=float(args.weight_eps),
        weight_beta=float(args.weight_beta),
        grad_clip_norm=float(args.grad_clip_norm),
        seed=int(args.seed),
        g_mode=args.g_mode,
        use_bg=use_bg,
        active_feature_names=list(feature_df.columns),
        reference_concentrations=reference_concentrations,
    )

    train_mu_log, train_std_log = posterior_from_bundle(bundle, X)
    train_summary, train_prob_df = build_summary_frame(
        df,
        train_mu_log,
        train_std_log,
        reference_concentrations,
        confidence=float(args.confidence),
        target_eps=float(bundle["target_eps"]),
    )

    loo_mu_log, loo_std_log = run_leave_one_out(
        X,
        y_true,
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        epochs=int(args.epochs),
        patience=int(args.patience),
        min_std=float(args.min_std),
        std_reg_lambda=float(args.std_reg_lambda),
        max_std_log=float(args.max_std_log),
        weight_eps=float(args.weight_eps),
        weight_beta=float(args.weight_beta),
        grad_clip_norm=float(args.grad_clip_norm),
        seed=int(args.seed),
        g_mode=args.g_mode,
        use_bg=use_bg,
        active_feature_names=list(feature_df.columns),
    )
    loo_summary, loo_prob_df = build_summary_frame(
        df,
        loo_mu_log,
        loo_std_log,
        reference_concentrations,
        confidence=float(args.confidence),
        target_eps=float(bundle["target_eps"]),
    )

    train_metrics = evaluate_summary(train_summary)
    loo_metrics = evaluate_summary(loo_summary)

    bundle["confidence"] = float(args.confidence)
    bundle["feature_set"] = str(args.feature_set)
    bundle["train_metrics"] = train_metrics
    bundle["loo_metrics"] = loo_metrics

    for path in (args.model_out, args.train_out, args.loo_out, args.train_grid_out, args.loo_grid_out, args.plot_out):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    torch.save(bundle, args.model_out)
    train_summary.to_csv(args.train_out, index=False, float_format=CSV_FLOAT_FORMAT)
    loo_summary.to_csv(args.loo_out, index=False, float_format=CSV_FLOAT_FORMAT)
    train_prob_df.to_csv(args.train_grid_out, index=False, float_format=CSV_FLOAT_FORMAT)
    loo_prob_df.to_csv(args.loo_grid_out, index=False, float_format=CSV_FLOAT_FORMAT)
    draw_result_plot(
        train_summary,
        loo_summary,
        loo_prob_df,
        train_metrics=train_metrics,
        loo_metrics=loo_metrics,
        confidence=float(args.confidence),
        out_path=args.plot_out,
    )

    print(f"Training metrics: R2={train_metrics['r2']:.6g}, RMSE={train_metrics['rmse']:.6g}, MAE={train_metrics['mae']:.6g}, NLL={train_metrics['mean_nll']:.6g}")
    print(f"LOO metrics:      R2={loo_metrics['r2']:.6g}, RMSE={loo_metrics['rmse']:.6g}, MAE={loo_metrics['mae']:.6g}, NLL={loo_metrics['mean_nll']:.6g}")
    print(f"Saved model       -> {args.model_out}")
    print(f"Saved train CSV   -> {args.train_out}")
    print(f"Saved LOO CSV     -> {args.loo_out}")
    print(f"Saved train probs -> {args.train_grid_out}")
    print(f"Saved LOO probs   -> {args.loo_grid_out}")
    print(f"Saved figure      -> {args.plot_out}")


if __name__ == "__main__":
    main()
