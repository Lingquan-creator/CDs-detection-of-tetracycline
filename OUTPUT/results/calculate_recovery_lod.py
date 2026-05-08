from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ROI_CSV = ROOT_DIR / "dataset/RAW/roi_results/roi_detailed_statistics.csv"
DEFAULT_PREDICTION_DIR = ROOT_DIR / "OUTPUT/RAW/mate20x_model_predictions"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "OUTPUT/results"

FILENAME_RE = re.compile(
    r"^(?P<matrix>.+)-(?P<replicate>\d+)-(?P<added_concentration>[-+]?\d+(?:\.\d+)?)\.dng$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate spike recovery with C2=0 and LOD=3*sigma/S from ROI statistics."
        )
    )
    parser.add_argument("--roi-csv", type=Path, default=DEFAULT_ROI_CSV)
    parser.add_argument("--prediction-dir", type=Path, default=DEFAULT_PREDICTION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def parse_filename_metadata(filename: object) -> dict[str, object]:
    match = FILENAME_RE.match(Path(str(filename)).name)
    if not match:
        return {
            "matrix": "unknown",
            "replicate": math.nan,
            "added_concentration": math.nan,
        }
    data = match.groupdict()
    return {
        "matrix": data["matrix"],
        "replicate": int(data["replicate"]),
        "added_concentration": float(data["added_concentration"]),
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


def finite_xy(df: pd.DataFrame, x_col: str, y_col: str) -> tuple[np.ndarray, np.ndarray]:
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    return x[valid], y[valid]


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    if x.size < 2:
        return math.nan, math.nan, math.nan
    slope, intercept = np.polyfit(x, y, 1)
    y_fit = slope * x + intercept
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = math.nan if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    return float(slope), float(intercept), float(r2)


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


def signal_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "R_sig",
        "G1_sig_calc",
        "G2_sig_calc",
        "G_sig",
        "G_sig_calc",
        "B_sig",
        "Ratio_RG",
        "Delta_RG",
        "Delta_GB",
    ]
    return [col for col in preferred if col in df.columns]


def calculate_signal_recovery_and_lod(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    c2: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    lod_rows: list[dict[str, object]] = []

    for matrix, matrix_df in df.groupby("matrix", sort=True):
        for signal in signal_columns(matrix_df):
            x, y = finite_xy(matrix_df, "added_concentration", signal)
            slope, intercept, r2 = linear_fit(x, y)

            blank = matrix_df.loc[matrix_df["added_concentration"] == 0, signal].to_numpy(dtype=float)
            blank = blank[np.isfinite(blank)]
            sigma = float(np.std(blank, ddof=1)) if blank.size > 1 else math.nan
            lod = 3.0 * sigma / abs(slope) if np.isfinite(sigma) and slope != 0 else math.nan

            lod_rows.append(
                {
                    "matrix": matrix,
                    "signal": signal,
                    "slope_S": slope,
                    "intercept": intercept,
                    "r2": r2,
                    "blank_n": int(blank.size),
                    "blank_sigma": sigma,
                    "lod": lod,
                    "lod_formula": "LOD=3*sigma/abs(S)",
                }
            )

            nonzero = matrix_df[matrix_df["added_concentration"] > 0].copy()
            if nonzero.empty or not np.isfinite(slope) or slope == 0:
                continue

            detected = (nonzero[signal].to_numpy(dtype=float) - intercept) / slope
            c3 = nonzero["added_concentration"].to_numpy(dtype=float)
            recovery = (detected - c2) / c3 * 100.0

            for row, c1, rec in zip(nonzero.itertuples(index=False), detected, recovery):
                sample_rows.append(
                    {
                        "source": "signal_linear_calibration",
                        "matrix": matrix,
                        "replicate": getattr(row, "replicate"),
                        "filename": getattr(row, "filename"),
                        "signal": signal,
                        "added_concentration_c3": getattr(row, "added_concentration"),
                        "original_concentration_c2": c2,
                        "detected_concentration_c1": c1,
                        "recovery_percent": rec,
                        "slope_S": slope,
                        "intercept": intercept,
                    }
                )

            sample_df = pd.DataFrame(
                {
                    "added_concentration_c3": c3,
                    "detected_concentration_c1": detected,
                    "recovery_percent": recovery,
                }
            )
            for c3_value, group_df in sample_df.groupby("added_concentration_c3", sort=True):
                c1_stats = summarize(group_df["detected_concentration_c1"])
                rec_stats = summarize(group_df["recovery_percent"])
                flags = recovery_quality_flags(rec_stats["mean"], rec_stats["rsd_percent"])
                summary_rows.append(
                    {
                        "source": "signal_linear_calibration",
                        "matrix": matrix,
                        "signal": signal,
                        "added_concentration_c3": c3_value,
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

    sample_out = pd.DataFrame(sample_rows)
    summary_out = pd.DataFrame(summary_rows)
    lod_out = pd.DataFrame(lod_rows)

    sample_out.to_csv(output_dir / "signal_recovery_samples.csv", index=False, float_format="%.6g")
    summary_out.to_csv(output_dir / "signal_recovery_summary.csv", index=False, float_format="%.6g")
    lod_out.to_csv(output_dir / "lod_results.csv", index=False, float_format="%.6g")
    return sample_out, summary_out, lod_out


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
        pred = pred[pred["added_concentration"] > 0].copy()
        pred["source"] = "model_prediction"
        pred["model"] = model_name
        pred["prediction_column"] = pred_col
        pred["added_concentration_c3"] = pred["added_concentration"].astype(float)
        pred["original_concentration_c2"] = c2
        pred["detected_concentration_c1"] = pred[pred_col].astype(float)
        pred["recovery_percent"] = (
            (pred["detected_concentration_c1"] - c2) / pred["added_concentration_c3"] * 100.0
        )
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
    signal_summary: pd.DataFrame,
    lod: pd.DataFrame,
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

    report_path = output_dir / "recovery_lod_report.md"
    primary_lod = lod[lod["signal"].isin(["G_sig", "G_sig_calc"])].copy()
    if not primary_lod.empty:
        primary_lod = primary_lod.sort_values(["matrix", "signal"])

    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Recovery and LOD Results\n\n")
        f.write(f"ROI CSV: `{roi_csv}`\n\n")
        f.write(f"Prediction directory: `{prediction_dir}`\n\n")
        f.write("Formulas from `temp` images:\n\n")
        f.write("- Recovery: `R(%) = (c1 - c2) / c3 * 100`; this run uses `c2 = 0`.\n")
        f.write("- LOD: `LOD = 3 * sigma / S`; this run uses `abs(S)` because several signals decrease with concentration.\n\n")
        f.write("Output files:\n\n")
        for name in [
            "signal_recovery_samples.csv",
            "signal_recovery_summary.csv",
            "lod_results.csv",
            "model_recovery_samples.csv",
            "model_recovery_summary.csv",
        ]:
            f.write(f"- `{name}`\n")
        f.write("\n## Primary LOD Rows (G signal)\n\n")
        if primary_lod.empty:
            f.write("No G-signal LOD rows were generated.\n")
        else:
            f.write(markdown_table(primary_lod))
            f.write("\n")

        f.write("\n## Model Recovery Summary Head\n\n")
        if model_summary.empty:
            f.write("No model prediction files were found.\n")
        else:
            f.write(markdown_table(model_summary.head(24)))
            f.write("\n")

        f.write("\n## Signal Recovery Summary Head\n\n")
        if signal_summary.empty:
            f.write("No signal recovery rows were generated.\n")
        else:
            f.write(markdown_table(signal_summary.head(24)))
            f.write("\n")


def main() -> None:
    args = parse_args()
    roi_csv = args.roi_csv.expanduser().resolve()
    prediction_dir = args.prediction_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    roi_df = pd.read_csv(roi_csv)
    roi_df = add_filename_metadata(roi_df)
    roi_df = build_signal_columns(roi_df)

    _, signal_summary, lod = calculate_signal_recovery_and_lod(roi_df, output_dir, c2=0.0)
    _, model_summary = calculate_model_recovery(prediction_dir, output_dir, c2=0.0)
    write_report(output_dir, roi_csv, prediction_dir, signal_summary, lod, model_summary)

    print(f"Saved results to {output_dir}")
    print(f"LOD rows: {len(lod)}")
    print(f"Signal recovery summary rows: {len(signal_summary)}")
    print(f"Model recovery summary rows: {len(model_summary)}")


if __name__ == "__main__":
    main()
