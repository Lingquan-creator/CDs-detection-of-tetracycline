from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
FIT_DATA_DIR = SCRIPT_DIR.parents[1]
PARAMETER_CSV = FIT_DATA_DIR / "results" / "results_parameter.csv"
DEFAULT_FIT_RESULT_FILES = [
    SCRIPT_DIR / "egg_results_spec.csv",
    SCRIPT_DIR / "milk_results_spec.csv",
]
OUTPUT_CSV = SCRIPT_DIR / "a1_concentration_predictions.csv"
SUMMARY_CSV = SCRIPT_DIR / "a1_concentration_summary.csv"


def parse_sample_label(label: str) -> tuple[str, int | None, float | None]:
    match = re.match(
        r"^(?P<sample>.+)-(?P<trial>\d+)-(?P<added>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$",
        str(label).strip(),
    )
    if not match:
        return str(label), None, None
    return (
        match.group("sample"),
        int(match.group("trial")),
        float(match.group("added")),
    )


def load_a1_fit_params(path: Path = PARAMETER_CSV) -> tuple[float, float]:
    df = pd.read_csv(path)
    row = df[
        (df["parameter"] == "A_prime")
        & (df["peak"].astype(int) == 1)
        & (df["fit_type"] == "exponential")
    ]
    if row.empty:
        raise ValueError(f"Cannot find exponential fit row for A_prime peak 1 in {path}")

    first = row.iloc[0]
    param_a = float(first["param_a"])
    param_b = float(first["param_b"])
    if param_a <= 0 or param_b == 0:
        raise ValueError(f"Invalid A1 exponential params: a={param_a}, b={param_b}")
    return param_a, param_b


def concentration_from_a1(a1_value: float, param_a: float, param_b: float) -> float:
    if not math.isfinite(a1_value) or a1_value <= 0:
        return math.nan
    return math.log(a1_value / param_a) / param_b


def load_predictions_from_fit_result(path: Path, param_a: float, param_b: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Concentration", "Peak1_A_prime"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    rows: list[dict] = []
    for row in df.itertuples(index=False):
        label = getattr(row, "Concentration")
        sample, trial, added_uM = parse_sample_label(label)
        a1 = float(getattr(row, "Peak1_A_prime"))
        predicted = concentration_from_a1(a1, param_a, param_b)
        rows.append(
            {
                "source_file": path.name,
                "label": label,
                "sample": sample,
                "trial": trial,
                "added_uM": added_uM,
                "peak1_A_prime": a1,
                "predicted_total_tetracycline_uM": predicted,
            }
        )
    return pd.DataFrame(rows)


def add_blank_corrected_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    predictions = predictions.copy()
    blank_lookup = (
        predictions[predictions["added_uM"] == 0]
        .set_index(["source_file", "sample", "trial"])["predicted_total_tetracycline_uM"]
        .to_dict()
    )
    predictions["sample_true_tetracycline_uM_c2"] = [
        blank_lookup.get((row.source_file, row.sample, row.trial), math.nan)
        for row in predictions.itertuples(index=False)
    ]
    predictions["spike_recovered_uM"] = (
        predictions["predicted_total_tetracycline_uM"]
        - predictions["sample_true_tetracycline_uM_c2"]
    )
    predictions["recovery_percent"] = np.where(
        predictions["added_uM"] > 0,
        predictions["spike_recovered_uM"] / predictions["added_uM"] * 100.0,
        np.nan,
    )
    return predictions


def rsd_percent(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) < 2:
        return math.nan
    mean = float(values.mean())
    if mean == 0:
        return math.nan
    return float(values.std(ddof=1) / abs(mean) * 100.0)


def build_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    group_cols = ["source_file", "sample", "added_uM"]
    for keys, group in predictions.groupby(group_cols, dropna=False, sort=True):
        source_file, sample, added_uM = keys
        rows.append(
            {
                "source_file": source_file,
                "sample": sample,
                "added_uM": added_uM,
                "n": int(len(group)),
                "mean_predicted_total_tetracycline_uM": float(
                    group["predicted_total_tetracycline_uM"].mean()
                ),
                "rsd_predicted_total_percent": rsd_percent(
                    group["predicted_total_tetracycline_uM"]
                ),
                "mean_sample_true_tetracycline_uM_c2": float(
                    group["sample_true_tetracycline_uM_c2"].mean()
                ),
                "mean_recovery_percent": float(group["recovery_percent"].mean())
                if pd.notna(added_uM) and added_uM > 0
                else math.nan,
                "rsd_recovery_percent": rsd_percent(group["recovery_percent"])
                if pd.notna(added_uM) and added_uM > 0
                else math.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    param_a, param_b = load_a1_fit_params()
    missing_files = [path for path in DEFAULT_FIT_RESULT_FILES if not path.exists()]
    if missing_files:
        missing = ", ".join(str(path) for path in missing_files)
        raise FileNotFoundError(
            "Missing fit result file(s). Run run_fit_spectrum_food.sh first: "
            f"{missing}"
        )

    predictions = pd.concat(
        [
            load_predictions_from_fit_result(path, param_a, param_b)
            for path in DEFAULT_FIT_RESULT_FILES
        ],
        ignore_index=True,
    )
    predictions = add_blank_corrected_columns(predictions)
    summary = build_summary(predictions)

    predictions.to_csv(OUTPUT_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)

    print(f"A1 exponential fit: A1 = {param_a:.8g} * exp({param_b:.8g} * C)")
    print(f"Saved predictions: {OUTPUT_CSV}")
    print(f"Saved summary: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
