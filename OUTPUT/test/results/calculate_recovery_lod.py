from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ROI_CSV = ROOT_DIR / "dataset/RAW/roi_results/roi_detailed_statistics.csv"
DEFAULT_PREDICTION_DIR = ROOT_DIR / "OUTPUT/RAW/mate20x_model_predictions"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "OUTPUT/results"

FILENAME_RE = re.compile(
    r"^(?P<matrix>.+)-(?P<replicate>\d+)-(?P<added_concentration>[-+]?\d+(?:\.\d+)?)\.dng$",
    re.IGNORECASE,
)
FOOD_CONCENTRATION_RE = re.compile(
    r"^(?P<matrix>.+)-(?P<added_concentration>\d+(?:\.\d+)?)\.dng$",
    re.IGNORECASE,
)
TRAILING_CONCENTRATION_RE = re.compile(
    r"^(?P<matrix>.*?)(?P<added_concentration>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\.dng$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate direct prediction recovery and recovery RSD from model prediction CSVs."
        )
    )
    parser.add_argument("--roi-csv", type=Path, default=DEFAULT_ROI_CSV)
    parser.add_argument("--prediction-dir", type=Path, default=DEFAULT_PREDICTION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def parse_filename_metadata(filename: object) -> dict[str, object]:
    name = Path(str(filename)).name
    match = FILENAME_RE.match(name)
    if match:
        data = match.groupdict()
        return {
            "matrix": data["matrix"],
            "replicate": int(data["replicate"]),
            "added_concentration": float(data["added_concentration"]),
        }

    match = FOOD_CONCENTRATION_RE.match(name)
    if match:
        data = match.groupdict()
        return {
            "matrix": data["matrix"],
            "replicate": math.nan,
            "added_concentration": float(data["added_concentration"]),
        }

    match = TRAILING_CONCENTRATION_RE.match(name)
    if match and match.group("matrix"):
        data = match.groupdict()
        return {
            "matrix": data["matrix"].rstrip("-_ "),
            "replicate": math.nan,
            "added_concentration": float(data["added_concentration"]),
        }

    return {
        "matrix": "unknown",
        "replicate": math.nan,
        "added_concentration": math.nan,
    }


def add_filename_metadata(df: pd.DataFrame) -> pd.DataFrame:
    meta = pd.DataFrame([parse_filename_metadata(v) for v in df["filename"]])
    out = pd.concat([df.reset_index(drop=True), meta], axis=1)
    if out["added_concentration"].isna().any():
        bad = out.loc[out["added_concentration"].isna(), "filename"].tolist()
        raise ValueError(f"Cannot parse added concentration from filenames: {bad}")
    return out


def build_signal_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["R_sig_calc"] = out["roi_R_mean"].astype(float) - out["bg_R_mean"].astype(float)
    out["G1_sig_calc"] = out["roi_G1_mean"].astype(float) - out["bg_G1_mean"].astype(float)
    out["G2_sig_calc"] = out["roi_G2_mean"].astype(float) - out["bg_G2_mean"].astype(float)
    out["G_sig_calc"] = (out["G1_sig_calc"] + out["G2_sig_calc"]) / 2.0
    out["B_sig_calc"] = out["roi_B_mean"].astype(float) - out["bg_B_mean"].astype(float)
    return out


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": math.nan, "sd": math.nan, "rsd_percent": math.nan}
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if arr.size > 1 else math.nan
    rsd = float(sd / abs(mean) * 100.0) if arr.size > 1 and not math.isclose(mean, 0.0) else math.nan
    return {"n": int(arr.size), "mean": mean, "sd": sd, "rsd_percent": rsd}


def recovery_quality_flags(recovery_mean: object, recovery_rsd: object) -> dict[str, bool]:
    mean = float(recovery_mean) if pd.notna(recovery_mean) else math.nan
    rsd = float(recovery_rsd) if pd.notna(recovery_rsd) else math.nan
    meets_recovery = bool(np.isfinite(mean) and 80.0 <= mean <= 110.0)
    meets_rsd = bool(np.isfinite(rsd) and rsd < 3.0)
    return {
        "meets_80_110_recovery": meets_recovery,
        "meets_rsd_lt_3": meets_rsd,
        "meets_recovery_and_rsd": meets_recovery and meets_rsd,
    }


def recovery_heatmap_label(recovery: float, rsd: float, n: float) -> str:
    rec_text = "" if not np.isfinite(recovery) else f"{recovery:.1f}%"
    rsd_text = "RSD n/a" if not np.isfinite(rsd) else f"RSD {rsd:.2f}%"
    n_text = "" if not np.isfinite(n) else f"n={int(n)}"
    return "\n".join(part for part in (rec_text, rsd_text, n_text) if part)


def draw_recovery_heatmap(
    summary: pd.DataFrame,
    *,
    row_cols: list[str],
    col_col: str,
    title: str,
    out_path: Path,
    table_path: Path,
) -> None:
    if summary.empty:
        return

    required = set(row_cols + [col_col, "recovery_mean_percent", "recovery_rsd_percent", "n"])
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"Cannot draw recovery heatmap, missing columns: {missing}")

    view = summary.copy()
    view["row_label"] = view[row_cols].astype(str).agg(" / ".join, axis=1)
    recovery = view.pivot_table(
        index="row_label",
        columns=col_col,
        values="recovery_mean_percent",
        aggfunc="first",
        dropna=False,
    )
    rsd = view.pivot_table(
        index="row_label",
        columns=col_col,
        values="recovery_rsd_percent",
        aggfunc="first",
        dropna=False,
    ).reindex_like(recovery)
    n_table = view.pivot_table(
        index="row_label",
        columns=col_col,
        values="n",
        aggfunc="first",
        dropna=False,
    ).reindex_like(recovery)
    recovery = recovery.dropna(axis=1, how="all")
    rsd = rsd.reindex_like(recovery)
    n_table = n_table.reindex_like(recovery)
    recovery.to_csv(table_path, float_format="%.6g")

    values = recovery.to_numpy(dtype=float)
    if values.size == 0:
        return

    fig_w = max(7.0, 1.35 * max(1, recovery.shape[1]) + 2.5)
    fig_h = max(4.5, 0.55 * max(1, recovery.shape[0]) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=160, constrained_layout=True)
    image = ax.imshow(values, cmap="RdYlGn", vmin=0, vmax=150, aspect="auto")
    ax.set_xticks(np.arange(recovery.shape[1]))
    ax.set_yticks(np.arange(recovery.shape[0]))
    ax.set_xticklabels([str(col) for col in recovery.columns], rotation=35, ha="right")
    ax.set_yticklabels([str(idx) for idx in recovery.index])
    ax.set_title(title)
    ax.set_xlabel(col_col)
    ax.set_ylabel(" / ".join(row_cols))

    for i in range(recovery.shape[0]):
        for j in range(recovery.shape[1]):
            rec = float(recovery.iloc[i, j]) if pd.notna(recovery.iloc[i, j]) else math.nan
            rec_rsd = float(rsd.iloc[i, j]) if pd.notna(rsd.iloc[i, j]) else math.nan
            n_value = float(n_table.iloc[i, j]) if pd.notna(n_table.iloc[i, j]) else math.nan
            if not np.isfinite(rec):
                continue
            text_color = "white" if rec < 45 or rec > 125 else "black"
            ax.text(
                j,
                i,
                recovery_heatmap_label(rec, rec_rsd, n_value),
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Recovery mean (%)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def draw_recovery_heatmaps(signal_summary: pd.DataFrame, model_summary: pd.DataFrame, output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    draw_recovery_heatmap(
        model_summary,
        row_cols=["matrix", "added_concentration_c3"],
        col_col="model",
        title="Model recovery mean with triplicate RSD",
        out_path=figures_dir / "model_recovery_heatmap_rsd.png",
        table_path=output_dir / "model_recovery_heatmap_table.csv",
    )


def load_prediction_sources(prediction_dir: Path) -> list[tuple[str, Path, str]]:
    return [
        ("exp", prediction_dir / "exp_predictions.csv", "conc_pred"),
        ("rgb_ridge", prediction_dir / "rgb_predictions.csv", "conc_pred"),
        ("gs_posterior_median", prediction_dir / "gs_posterior_summary.csv", "posterior_median"),
        ("gs_posterior_mean", prediction_dir / "gs_posterior_summary.csv", "posterior_mean"),
    ]


def calculate_model_recovery(
    prediction_dir: Path,
    output_dir: Path,
    *,
    c2: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_rows: list[pd.DataFrame] = []

    for model_name, csv_path, pred_col in load_prediction_sources(prediction_dir):
        if not csv_path.exists():
            continue
        pred = add_filename_metadata(pd.read_csv(csv_path))
        if pred_col not in pred.columns:
            continue
        if "conc_true" in pred.columns:
            pred["added_concentration"] = pd.to_numeric(pred["conc_true"], errors="coerce")
        pred = pred[pred["added_concentration"] >= 0].copy()
        pred["source"] = "model_prediction"
        pred["model"] = model_name
        pred["prediction_column"] = pred_col
        pred["added_concentration_c3"] = pred["added_concentration"].astype(float)
        pred["original_concentration_c2"] = c2
        pred["detected_concentration_c1"] = pred[pred_col].astype(float)
        denominator = pred["added_concentration_c3"].to_numpy(dtype=float)
        numerator = pred["detected_concentration_c1"].to_numpy(dtype=float) - c2
        recovery = np.full_like(denominator, np.nan, dtype=float)
        valid_recovery = denominator > 0
        recovery[valid_recovery] = numerator[valid_recovery] / denominator[valid_recovery] * 100.0
        pred["recovery_percent"] = recovery
        sample_rows.append(
            pred[
                [
                    "source",
                    "model",
                    "prediction_column",
                    "matrix",
                    "replicate",
                    "filename",
                    "added_concentration_c3",
                    "original_concentration_c2",
                    "detected_concentration_c1",
                    "recovery_percent",
                ]
            ]
        )

    if not sample_rows:
        empty = pd.DataFrame()
        empty.to_csv(output_dir / "model_recovery_samples.csv", index=False)
        empty.to_csv(output_dir / "model_recovery_summary.csv", index=False)
        return empty, empty

    samples = pd.concat(sample_rows, ignore_index=True)
    summary_rows: list[dict[str, object]] = []
    keys = ["model", "prediction_column", "matrix", "added_concentration_c3"]
    for key_values, group_df in samples.groupby(keys, sort=True):
        c1_stats = summarize(group_df["detected_concentration_c1"])
        rec_stats = summarize(group_df["recovery_percent"])
        flags = recovery_quality_flags(rec_stats["mean"], rec_stats["rsd_percent"])
        summary_rows.append(
            {
                "source": "model_prediction",
                "model": key_values[0],
                "prediction_column": key_values[1],
                "matrix": key_values[2],
                "added_concentration_c3": key_values[3],
                "original_concentration_c2": c2,
                "n": rec_stats["n"],
                "detected_c1_mean": c1_stats["mean"],
                "detected_c1_sd": c1_stats["sd"],
                "recovery_mean_percent": rec_stats["mean"],
                "recovery_sd_percent": rec_stats["sd"],
                "recovery_rsd_percent": rec_stats["rsd_percent"],
                **flags,
            }
        )

    summary = pd.DataFrame(summary_rows)
    samples.to_csv(output_dir / "model_recovery_samples.csv", index=False, float_format="%.6g")
    summary.to_csv(output_dir / "model_recovery_summary.csv", index=False, float_format="%.6g")
    return samples, summary


def write_report(
    output_dir: Path,
    roi_csv: Path,
    prediction_dir: Path,
    model_summary: pd.DataFrame,
) -> None:
    def markdown_table(df: pd.DataFrame) -> str:
        if df.empty:
            return ""
        view = df.copy()
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.6g}")
            else:
                view[col] = view[col].map(lambda v: "" if pd.isna(v) else str(v))
        header = "| " + " | ".join(view.columns) + " |"
        separator = "| " + " | ".join(["---"] * len(view.columns)) + " |"
        rows = ["| " + " | ".join(row) + " |" for row in view.astype(str).to_numpy()]
        return "\n".join([header, separator, *rows])

    report_path = output_dir / "recovery_report.md"

    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Recovery and RSD Results\n\n")
        f.write(f"ROI CSV: `{roi_csv}`\n\n")
        f.write(f"Prediction directory: `{prediction_dir}`\n\n")
        f.write("Formula:\n\n")
        f.write("- Recovery: `R(%) = predicted_concentration / added_concentration * 100`.\n")
        f.write("- For 0 concentration rows, recovery is undefined and written as blank, while detected concentration is retained.\n")
        f.write("- RSD: standard deviation of replicate recoveries divided by mean recovery.\n\n")
        f.write("Output files:\n\n")
        for name in [
            "model_recovery_samples.csv",
            "model_recovery_summary.csv",
            "model_recovery_heatmap_table.csv",
            "figures/model_recovery_heatmap_rsd.png",
        ]:
            f.write(f"- `{name}`\n")

        f.write("\n## Model Recovery Summary Head\n\n")
        if model_summary.empty:
            f.write("No model prediction files were found.\n")
        else:
            f.write(markdown_table(model_summary.head(24)))
            f.write("\n")


def main() -> None:
    args = parse_args()
    roi_csv = args.roi_csv.expanduser().resolve()
    prediction_dir = args.prediction_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in (
        "lod_results.csv",
        "recovery_lod_report.md",
        "signal_recovery_heatmap_table.csv",
    ):
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    figures_dir = output_dir / "figures"
    if figures_dir.exists():
        for stale_path in figures_dir.glob("signal_*"):
            stale_path.unlink()

    signal_summary = pd.DataFrame()
    pd.DataFrame().to_csv(output_dir / "signal_recovery_samples.csv", index=False)
    pd.DataFrame().to_csv(output_dir / "signal_recovery_summary.csv", index=False)
    _, model_summary = calculate_model_recovery(prediction_dir, output_dir, c2=0.0)
    draw_recovery_heatmaps(signal_summary, model_summary, output_dir)
    write_report(output_dir, roi_csv, prediction_dir, model_summary)

    print(f"Saved results to {output_dir}")
    print(f"Model recovery summary rows: {len(model_summary)}")
    print(f"Recovery heatmaps: {output_dir / 'figures'}")


if __name__ == "__main__":
    main()
