"""Data loading and alignment helpers for cross-device color mapping."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import MappingConfig

CHANNELS_4 = ("R", "G1", "G2", "B")
CHANNELS_3 = ("R", "G", "B")


def extract_concentration(filename: str) -> float:
    """Extract numeric concentration from a filename like '40.dng' or '0.4.dng'."""
    match = re.search(r"[-+]?\d*\.?\d+", filename)
    if not match:
        raise ValueError(f"Cannot parse concentration from filename: {filename}")
    return float(match.group(0))


def _corrected_value(record: dict, channel: str, min_clip: float) -> float:
    """Return corrected channel value, computing it if not explicitly stored."""
    corr_key = f"{channel}_corrected"
    if corr_key in record and record[corr_key] is not None:
        value = float(record[corr_key])
    else:
        mean_key = f"{channel}_mean"
        bg_key = f"{channel}_bg"
        value = float(record.get(mean_key, 0.0)) - float(record.get(bg_key, 0.0))
    return max(value, min_clip)


def _record_to_row(record: dict, use_four_channels: bool, min_clip: float) -> dict:
    concentration = extract_concentration(record["filename"])
    row = {"concentration": concentration}

    if use_four_channels:
        for ch in CHANNELS_4:
            row[ch] = _corrected_value(record, ch, min_clip)
    else:
        r = _corrected_value(record, "R", min_clip)
        g1 = _corrected_value(record, "G1", min_clip)
        g2 = _corrected_value(record, "G2", min_clip)
        b = _corrected_value(record, "B", min_clip)
        row.update({"R": r, "G": (g1 + g2) / 2.0, "B": b})
    return row


def load_roi_json(roi_path: Path) -> List[dict]:
    with roi_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def device_dataframe(
    roi_path: Path, use_four_channels: bool, min_clip: float
) -> pd.DataFrame:
    """Load ROI json into a tidy dataframe with corrected channel columns."""
    records = load_roi_json(roi_path)
    rows = [_record_to_row(rec, use_four_channels, min_clip) for rec in records]
    df = pd.DataFrame(rows)
    df.sort_values("concentration", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _select_by_concentrations(
    df: pd.DataFrame, targets: Sequence[float], tol: float = 1e-6
) -> pd.DataFrame:
    selected_rows = []
    for c in targets:
        mask = df["concentration"].apply(lambda v: math.isclose(v, c, rel_tol=1e-6, abs_tol=tol))
        subset = df[mask]
        if subset.empty:
            continue
        # if multiple rows for same concentration, take the first
        selected_rows.append(subset.iloc[0])
    return pd.DataFrame(selected_rows)


def align_datasets(
    df_pic1: pd.DataFrame,
    df_pic2: pd.DataFrame,
    matching_concentrations: Sequence[float] | None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[float]]:
    """Align two dataframes on concentration."""
    if matching_concentrations is None:
        # intersect concentrations present in both
        common = sorted(set(df_pic1["concentration"]).intersection(df_pic2["concentration"]))
    else:
        common = list(matching_concentrations)

    if not common:
        raise ValueError("No overlapping concentrations between devices.")

    pic1_aligned = _select_by_concentrations(df_pic1, common)
    pic2_aligned = _select_by_concentrations(df_pic2, common)

    if len(pic1_aligned) != len(common) or len(pic2_aligned) != len(common):
        raise ValueError(
            f"Missing samples for some concentrations. "
            f"Expected {len(common)} entries, got pic1={len(pic1_aligned)}, pic2={len(pic2_aligned)}"
        )

    return pic1_aligned, pic2_aligned, common


def load_aligned_arrays(config: MappingConfig):
    """Load ROI data and return aligned numpy arrays ready for model training.

    Returns:
        X_pic2: np.ndarray feature matrix from pic_2.
        y_pic1: np.ndarray target matrix from pic_1.
        concentrations: np.ndarray of matched concentration values.
        df1_aligned/df2_aligned: filtered dataframes after alignment.
        df1_full/df2_full: full dataframes before alignment (for diagnostics).
    """
    roi1 = config.pic1_dir / "roi_results" / config.roi_filename
    roi2 = config.pic2_dir / "roi_results" / config.roi_filename

    df1 = device_dataframe(roi1, config.use_four_channels, config.min_clip)
    df2 = device_dataframe(roi2, config.use_four_channels, config.min_clip)

    df1_aligned, df2_aligned, concentrations = align_datasets(
        df1, df2, config.matching_concentrations
    )

    feature_cols = list(CHANNELS_4 if config.use_four_channels else CHANNELS_3)

    X_pic2 = df2_aligned[feature_cols].to_numpy(dtype=float)
    y_pic1 = df1_aligned[feature_cols].to_numpy(dtype=float)

    return (
        X_pic2,
        y_pic1,
        np.array(concentrations, dtype=float),
        df1_aligned,
        df2_aligned,
        df1,
        df2,
    )
