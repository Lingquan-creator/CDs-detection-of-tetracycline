from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
TEST_RESULTS_DIR = SCRIPT_DIR.parent
FIT_DATA_DIR = TEST_RESULTS_DIR.parent
REPO_ROOT = FIT_DATA_DIR.parent
DATASET_DIR = FIT_DATA_DIR / "dataset"
RESULTS_DIR = FIT_DATA_DIR / "results"
OUTPUT_DIR = SCRIPT_DIR

sys.path.insert(0, str(FIT_DATA_DIR))

from integrate_and_fit import INTEGRATION_RANGE, model_exponential  # noqa: E402


MODEL_CSV = RESULTS_DIR / "results_intensity_models.csv"
SAMPLE_FILES = {
    "egg": DATASET_DIR / "egg.csv",
    "milk": DATASET_DIR / "milk.csv",
}

SAMPLE_RESULT_CSV = OUTPUT_DIR / "sample_predictions.csv"
RECOVERY_RESULT_CSV = OUTPUT_DIR / "recovery_summary.csv"
LOD_RESULT_CSV = OUTPUT_DIR / "lod_c2_summary.csv"


def parse_model_params(path: Path = MODEL_CSV) -> tuple[float, float, float]:
    model_df = pd.read_csv(path)
    if model_df.empty:
        raise ValueError(f"No fitted model rows found in {path}")

    best = model_df.iloc[0]
    if str(best["model"]).strip().lower() != "exponential":
        raise ValueError(f"Only exponential intensity model is supported, got {best['model']!r}")

    params = [float(part.strip()) for part in str(best["params"]).split(";")]
    if len(params) != 3:
        raise ValueError(f"Expected 3 exponential params in {path}, got {params}")
    return params[0], params[1], params[2]


def parse_sample_column(label: str) -> tuple[str, int, float]:
    match = re.match(
        r"^(?P<sample>.+)-(?P<trial>\d+)-(?P<added>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$",
        str(label).strip(),
    )
    if not match:
        raise ValueError(f"Cannot parse sample column header: {label!r}")
    return (
        match.group("sample"),
        int(match.group("trial")),
        float(match.group("added")),
    )


def invert_exponential_intensity(area: float, params: tuple[float, float, float]) -> float:
    a, b, c = params
    if a == 0 or b == 0:
        return math.nan
    ratio = (area - c) / a
    if ratio <= 0 or not math.isfinite(ratio):
        return math.nan
    return math.log(ratio) / b


def exponential_slope(concentration: float, params: tuple[float, float, float]) -> float:
    a, b, _ = params
    return abs(a * b * math.exp(b * concentration))


def integrate_column(wavelengths: np.ndarray, intensity: np.ndarray) -> float:
    low, high = INTEGRATION_RANGE
    mask = (wavelengths >= low) & (wavelengths <= high)
    if not np.any(mask):
        raise ValueError(f"No wavelengths in integration range {INTEGRATION_RANGE}")
    return float(np.trapezoid(intensity[mask], wavelengths[mask]))


def rsd_percent(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return math.nan
    mean = float(np.mean(arr))
    if mean == 0:
        return math.nan
    return float(np.std(arr, ddof=1) / abs(mean) * 100.0)


def load_sample_file(path: Path, model_params: tuple[float, float, float]) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.shape[1] < 2:
        raise ValueError(f"Expected wavelength + sample columns in {path}")

    wavelengths = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    rows: list[dict] = []
    for column in df.columns[1:]:
        sample_name, trial, added_uM = parse_sample_column(column)
        intensity = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(wavelengths) & np.isfinite(intensity)
        area = integrate_column(wavelengths[valid], intensity[valid])
        detected_total_uM = invert_exponential_intensity(area, model_params)
        rows.append(
            {
                "file": path.name,
                "sample": sample_name,
                "trial": trial,
                "added_uM_c3": added_uM,
                "integrated_intensity": area,
                "detected_total_uM_c1": detected_total_uM,
            }
        )

    result = pd.DataFrame(rows)
    blanks = (
        result[result["added_uM_c3"] == 0]
        .set_index(["sample", "trial"])["detected_total_uM_c1"]
        .to_dict()
    )
    result["true_sample_uM_c2"] = [
        blanks.get((row.sample, row.trial), math.nan) for row in result.itertuples(index=False)
    ]
    result["recovery_percent"] = np.where(
        result["added_uM_c3"] > 0,
        (result["detected_total_uM_c1"] - result["true_sample_uM_c2"])
        / result["added_uM_c3"]
        * 100.0,
        np.nan,
    )
    return result


def build_recovery_summary(samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    group_cols = ["file", "sample", "added_uM_c3"]
    for keys, group in samples.groupby(group_cols, sort=True):
        file_name, sample_name, added_uM = keys
        rows.append(
            {
                "file": file_name,
                "sample": sample_name,
                "added_uM_c3": added_uM,
                "n": int(len(group)),
                "mean_detected_total_uM_c1": float(group["detected_total_uM_c1"].mean()),
                "rsd_detected_total_percent": rsd_percent(group["detected_total_uM_c1"]),
                "mean_true_sample_uM_c2": float(group["true_sample_uM_c2"].mean()),
                "mean_recovery_percent": float(group["recovery_percent"].mean())
                if added_uM > 0
                else math.nan,
                "rsd_recovery_percent": rsd_percent(group["recovery_percent"])
                if added_uM > 0
                else math.nan,
            }
        )
    return pd.DataFrame(rows)


def build_lod_summary(samples: pd.DataFrame, model_params: tuple[float, float, float]) -> pd.DataFrame:
    rows: list[dict] = []
    for (file_name, sample_name), group in samples.groupby(["file", "sample"], sort=True):
        blanks = group[group["added_uM_c3"] == 0]
        c2_values = blanks["detected_total_uM_c1"].to_numpy(dtype=float)
        c2_mean = float(np.nanmean(c2_values)) if c2_values.size else math.nan
        c2_std = float(np.nanstd(c2_values, ddof=1)) if c2_values.size > 1 else math.nan
        c2_rsd = rsd_percent(c2_values)

        blank_signal = blanks["integrated_intensity"].to_numpy(dtype=float)
        sigma_signal = (
            float(np.nanstd(blank_signal, ddof=1)) if blank_signal.size > 1 else math.nan
        )
        slope_at_zero = exponential_slope(0.0, model_params)
        slope_at_c2 = exponential_slope(c2_mean, model_params) if math.isfinite(c2_mean) else math.nan

        rows.append(
            {
                "file": file_name,
                "sample": sample_name,
                "n_blank": int(len(blanks)),
                "mean_true_sample_uM_c2": c2_mean,
                "std_true_sample_uM_c2": c2_std,
                "rsd_true_sample_percent": c2_rsd,
                "sigma_blank_integrated_intensity": sigma_signal,
                "slope_at_zero": slope_at_zero,
                "lod_uM_slope_at_zero": 3.0 * sigma_signal / slope_at_zero,
                "slope_at_c2": slope_at_c2,
                "lod_uM_slope_at_c2": 3.0 * sigma_signal / slope_at_c2,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_params = parse_model_params()

    sample_frames = [
        load_sample_file(path, model_params)
        for path in SAMPLE_FILES.values()
    ]
    samples = pd.concat(sample_frames, ignore_index=True)
    recovery_summary = build_recovery_summary(samples)
    lod_summary = build_lod_summary(samples, model_params)

    samples.to_csv(SAMPLE_RESULT_CSV, index=False)
    recovery_summary.to_csv(RECOVERY_RESULT_CSV, index=False)
    lod_summary.to_csv(LOD_RESULT_CSV, index=False)

    print(f"Model params (a, b, c): {model_params}")
    print(f"Saved sample predictions: {SAMPLE_RESULT_CSV}")
    print(f"Saved recovery summary: {RECOVERY_RESULT_CSV}")
    print(f"Saved LOD/C2 summary: {LOD_RESULT_CSV}")


if __name__ == "__main__":
    main()
