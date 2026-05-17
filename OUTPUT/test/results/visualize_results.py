from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_DIR = Path(__file__).resolve().parent
FIGURE_DIR = RESULTS_DIR / "figures"

PASS_BAND_LOW = 80.0
PASS_BAND_HIGH = 110.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize recovery and RSD result CSVs.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def _format_number(value: float) -> str:
    if not np.isfinite(value):
        return ""
    return f"{value:.3g}"


def _ordered(values: pd.Series) -> list:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        return sorted(values.unique(), key=float)
    return sorted(values.unique())


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        }
    )


def add_recovery_band(ax: plt.Axes) -> None:
    ax.axhspan(PASS_BAND_LOW, PASS_BAND_HIGH, color="#2ca02c", alpha=0.12, label="80-110%")
    ax.axhline(100, color="#2ca02c", linewidth=1.0, linestyle="--")


def save_fig(fig: plt.Figure, name: str, *, tight: bool = True) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    if tight:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="This figure includes Axes that are not compatible with tight_layout.*",
                category=UserWarning,
            )
            fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def draw_heatmap(
    ax: plt.Axes,
    table: pd.DataFrame,
    title: str,
    *,
    cmap: str = "RdYlGn_r",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    data = table.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_xticklabels([_format_number(float(v)) for v in table.columns], rotation=0)
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_yticklabels(table.index)
    ax.set_xlabel("Added concentration C3")

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = data[i, j]
            if np.isfinite(value):
                ax.text(j, i, _format_number(value), ha="center", va="center", fontsize=7)
    return im


def plot_signal_recovery(signal_summary: pd.DataFrame) -> None:
    matrices = sorted(signal_summary["matrix"].unique())
    concentrations = _ordered(signal_summary["added_concentration_c3"])

    fig, axes = plt.subplots(
        len(matrices),
        1,
        figsize=(12.2, 3.4 * len(matrices)),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, matrix in zip(axes[:, 0], matrices):
        sub = signal_summary[signal_summary["matrix"] == matrix]
        table = (
            sub.pivot_table(
                index="signal",
                columns="added_concentration_c3",
                values="recovery_mean_percent",
                aggfunc="mean",
            )
            .reindex(columns=concentrations)
        )
        im = draw_heatmap(ax, table, f"Signal Recovery Mean (%) - {matrix}", vmin=0, vmax=220)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Recovery mean (%)", fraction=0.035, pad=0.04)
    save_fig(fig, "signal_recovery_heatmaps.png", tight=False)

    g = signal_summary[signal_summary["signal"] == "G_sig"].copy()
    if g.empty:
        return

    fig, axes = plt.subplots(1, len(matrices), figsize=(5.2 * len(matrices), 4.2), sharey=True)
    if len(matrices) == 1:
        axes = [axes]
    for ax, matrix in zip(axes, matrices):
        sub = g[g["matrix"] == matrix].sort_values("added_concentration_c3")
        x = sub["added_concentration_c3"].to_numpy(dtype=float)
        y = sub["recovery_mean_percent"].to_numpy(dtype=float)
        yerr = sub["recovery_sd_percent"].to_numpy(dtype=float)
        add_recovery_band(ax)
        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, linewidth=1.6)
        ax.set_title(f"G_sig Recovery - {matrix}")
        ax.set_xlabel("Added concentration C3")
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([_format_number(v) for v in x])
        ax.set_ylim(0, max(260, np.nanmax(y + np.nan_to_num(yerr, nan=0)) * 1.08))
    axes[0].set_ylabel("Recovery (%)")
    save_fig(fig, "signal_recovery_gsig_errorbars.png")


def plot_model_recovery(model_summary: pd.DataFrame) -> None:
    matrices = sorted(model_summary["matrix"].unique())
    concentrations = _ordered(model_summary["added_concentration_c3"])
    models = list(model_summary["model"].drop_duplicates())

    fig, axes = plt.subplots(
        len(matrices),
        1,
        figsize=(12.2, 3.4 * len(matrices)),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, matrix in zip(axes[:, 0], matrices):
        sub = model_summary[model_summary["matrix"] == matrix]
        table = (
            sub.pivot_table(
                index="model",
                columns="added_concentration_c3",
                values="recovery_mean_percent",
                aggfunc="mean",
            )
            .reindex(index=models, columns=concentrations)
        )
        im = draw_heatmap(ax, table, f"Model Recovery Mean (%) - {matrix}", vmin=0, vmax=650)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Recovery mean (%)", fraction=0.035, pad=0.04)
    save_fig(fig, "model_recovery_heatmaps.png", tight=False)

    fig, axes = plt.subplots(len(matrices), 1, figsize=(11.5, 4.0 * len(matrices)), sharex=True, squeeze=False)
    for ax, matrix in zip(axes[:, 0], matrices):
        sub = model_summary[model_summary["matrix"] == matrix]
        add_recovery_band(ax)
        for model in models:
            mdf = sub[sub["model"] == model].sort_values("added_concentration_c3")
            x = mdf["added_concentration_c3"].to_numpy(dtype=float)
            y = mdf["recovery_mean_percent"].to_numpy(dtype=float)
            yerr = mdf["recovery_sd_percent"].to_numpy(dtype=float)
            ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, linewidth=1.5, label=model)
        ax.set_title(f"Model Recovery - {matrix}")
        ax.set_ylabel("Recovery (%)")
        ax.set_xscale("log")
        ax.set_xticks(np.asarray(concentrations, dtype=float))
        ax.set_xticklabels([_format_number(float(v)) for v in concentrations])
        ax.legend(ncols=2)
    axes[-1, 0].set_xlabel("Added concentration C3")
    save_fig(fig, "model_recovery_errorbars.png")


def plot_pass_counts(signal_summary: pd.DataFrame, model_summary: pd.DataFrame) -> None:
    rows = []
    for label, df in [("signal", signal_summary), ("model", model_summary)]:
        if df.empty:
            continue
        for col, name in [
            ("meets_80_110_recovery", "80-110% recovery"),
            ("meets_rsd_lt_3", "RSD < 3%"),
            ("meets_recovery_and_rsd", "both"),
        ]:
            if col not in df.columns:
                continue
            rows.append({"source": label, "criterion": name, "count": int(df[col].sum()), "total": len(df)})
    if not rows:
        return
    counts = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    criteria = counts["criterion"].drop_duplicates().tolist()
    sources = counts["source"].drop_duplicates().tolist()
    x = np.arange(len(criteria))
    width = 0.35
    for i, source in enumerate(sources):
        sub = counts[counts["source"] == source].set_index("criterion").reindex(criteria)
        offset = (i - 0.5) * width
        ax.bar(x + offset, sub["count"], width=width, label=source)
        for xpos, count, total in zip(x + offset, sub["count"], sub["total"]):
            ax.text(xpos, count, f"{int(count)}/{int(total)}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Recovery Criterion Pass Counts")
    ax.set_ylabel("Number of summary rows")
    ax.set_xticks(x)
    ax.set_xticklabels(criteria)
    ax.legend(title="Source")
    save_fig(fig, "recovery_pass_counts.png")


def plot_sample_distributions(signal_samples: pd.DataFrame, model_samples: pd.DataFrame) -> None:
    g = (
        signal_samples[signal_samples["signal"] == "G_sig"].copy()
        if "signal" in signal_samples.columns
        else pd.DataFrame()
    )
    if not g.empty:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        groups = []
        labels = []
        for matrix in sorted(g["matrix"].unique()):
            values = g[g["matrix"] == matrix]["recovery_percent"].dropna().to_numpy(dtype=float)
            groups.append(values)
            labels.append(matrix)
        add_recovery_band(ax)
        ax.boxplot(groups, tick_labels=labels, showfliers=True)
        ax.set_title("Sample-Level G_sig Recovery Distribution")
        ax.set_ylabel("Recovery (%)")
        ax.set_xlabel("Matrix")
        save_fig(fig, "signal_gsig_sample_recovery_boxplot.png")

    if not model_samples.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        groups = []
        labels = []
        for model in model_samples["model"].drop_duplicates():
            values = model_samples[model_samples["model"] == model]["recovery_percent"].dropna().to_numpy(dtype=float)
            groups.append(values)
            labels.append(model)
        add_recovery_band(ax)
        ax.boxplot(groups, tick_labels=labels, showfliers=True)
        ax.set_title("Sample-Level Model Recovery Distribution")
        ax.set_ylabel("Recovery (%)")
        ax.set_xlabel("Model")
        ax.tick_params(axis="x", rotation=20)
        save_fig(fig, "model_sample_recovery_boxplot.png")


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def main() -> None:
    global RESULTS_DIR, FIGURE_DIR

    args = parse_args()
    RESULTS_DIR = args.results_dir.expanduser().resolve()
    FIGURE_DIR = RESULTS_DIR / "figures"

    setup_style()
    signal_summary = read_csv_or_empty(RESULTS_DIR / "signal_recovery_summary.csv")
    model_summary = read_csv_or_empty(RESULTS_DIR / "model_recovery_summary.csv")
    signal_samples = read_csv_or_empty(RESULTS_DIR / "signal_recovery_samples.csv")
    model_samples = read_csv_or_empty(RESULTS_DIR / "model_recovery_samples.csv")

    if not signal_summary.empty:
        plot_signal_recovery(signal_summary)
    if not model_summary.empty:
        plot_model_recovery(model_summary)
    plot_pass_counts(signal_summary, model_summary)
    plot_sample_distributions(signal_samples, model_samples)


if __name__ == "__main__":
    main()
