"""
Invert RGB intensities to concentration using fitted exponential parameters.

Supports:
- Per-channel inversion (legacy).
- Multi-channel inversion (R_corr, G1_corr, G2_corr, B_corr) with weighted averaging.

Config:
- concentration/rgb_weights.json : {"R_corr":0.1,"G1_corr":0.35,"G2_corr":0.35,"B_corr":0.2}
  Weights will be normalized over channels present in both weights and data.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FIT_DIR = ROOT / "rgb_fit_output"
PARAM_PATH = FIT_DIR / "rgb_fit_params.csv"
WEIGHT_PATH = ROOT / "rgb_weights.json"
DEFAULT_CHANNELS = ["R_corr", "G1_corr", "G2_corr", "B_corr"]


def load_params_dict(path: Path) -> Dict[str, Tuple[float, float, float]]:
    df = pd.read_csv(path)
    params = {}
    for _, row in df.iterrows():
        params[row["channel"]] = (float(row["a"]), float(row["b"]), float(row["c"]))
    return params


def load_weights(path: Path) -> Dict[str, float]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            weights = json.load(f)
    else:
        weights = {
            "R_corr": 0.1,
            "G1_corr": 0.35,
            "G2_corr": 0.35,
            "B_corr": 0.2,
        }
    # Normalize and drop non-positive
    weights = {k: v for k, v in weights.items() if v > 0}
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()} if total > 0 else {}


def invert_intensity(intensity: float, params: Tuple[float, float, float]) -> float:
    a, b, c = params
    if b == 0:
        raise ValueError("Parameter b cannot be zero.")
    if intensity <= c or a == 0:
        return math.inf
    ratio = (intensity - c) / a
    if ratio <= 0:
        return math.inf
    return float(math.log(ratio) / b)


def invert_channels(intensities: Dict[str, float], params: Dict[str, Tuple[float, float, float]]) -> Dict[str, float]:
    concs: Dict[str, float] = {}
    for ch, val in intensities.items():
        if ch in params:
            concs[ch] = invert_intensity(val, params[ch])
    return concs


def weighted_average(concs: Dict[str, float], weights: Dict[str, float]) -> float:
    valid = {ch: conc for ch, conc in concs.items() if math.isfinite(conc) and ch in weights}
    if not valid:
        return math.inf
    wsum = sum(weights[ch] for ch in valid)
    return float(sum(weights[ch] * valid[ch] for ch in valid) / wsum) if wsum > 0 else math.inf


def invert_csv_multi(
    roi_csv: Path,
    out_csv: Path,
    params: Dict[str, Tuple[float, float, float]],
    weights: Dict[str, float],
    channels: Iterable[str],
) -> None:
    df = pd.read_csv(roi_csv)
    channels = [ch for ch in channels if ch in df.columns and ch in params]
    if not channels:
        raise ValueError("No valid channels found in ROI CSV for inversion.")
    df = df.copy()
    for ch in channels:
        df[f"{ch}_pred_concentration"] = df[ch].apply(lambda x: invert_intensity(x, params[ch]))
    df["conc_weighted"] = df.apply(
        lambda row: weighted_average({ch: row[f"{ch}_pred_concentration"] for ch in channels}, weights),
        axis=1,
    )
    df.to_csv(out_csv, index=False)
    print(f"Saved inverted concentrations -> {out_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Map RGB intensity to concentration using fitted exponential params.")
    p.add_argument("--channel", help="Single channel name (legacy mode).")
    p.add_argument("--intensity", type=float, help="Single intensity value to invert (legacy mode).")
    p.add_argument("--roi-csv", type=Path, help="ROI CSV for batch inversion.")
    p.add_argument("--out", type=Path, help="Output CSV path for batch inversion.")
    p.add_argument("--params", type=Path, default=PARAM_PATH, help="Path to rgb_fit_params.csv")
    p.add_argument("--weights", type=Path, default=WEIGHT_PATH, help="Path to weights JSON")
    p.add_argument("--channels", default=",".join(DEFAULT_CHANNELS), help="Comma-separated channels to use for weighted mode.")
    p.add_argument("--r", type=float, help="Single intensity for R_corr")
    p.add_argument("--g1", type=float, help="Single intensity for G1_corr")
    p.add_argument("--g2", type=float, help="Single intensity for G2_corr")
    p.add_argument("--b", type=float, help="Single intensity for B_corr")
    return p.parse_args()


def main():
    args = parse_args()
    params = load_params_dict(args.params)
    weights = load_weights(args.weights)
    channels = [ch.strip() for ch in args.channels.split(",") if ch.strip()]

    # Legacy single-channel mode
    if args.channel and args.intensity is not None:
        if args.channel not in params:
            raise ValueError(f"Channel {args.channel} not in params file.")
        conc = invert_intensity(args.intensity, params[args.channel])
        print(f"[Single] {args.channel}: intensity {args.intensity} -> concentration {conc}")
        return

    # Single-value multi-channel mode
    single_intensities = {}
    if args.r is not None:
        single_intensities["R_corr"] = args.r
    if args.g1 is not None:
        single_intensities["G1_corr"] = args.g1
    if args.g2 is not None:
        single_intensities["G2_corr"] = args.g2
    if args.b is not None:
        single_intensities["B_corr"] = args.b
    if single_intensities:
        concs = invert_channels(single_intensities, params)
        avg = weighted_average(concs, weights)
        print("[Multi] per-channel:", concs)
        print(f"[Multi] weighted concentration: {avg}")
        return

    # Batch mode
    if args.roi_csv:
        out_path = args.out or args.roi_csv.with_name(args.roi_csv.stem + "_conc_weighted.csv")
        invert_csv_multi(args.roi_csv, out_path, params, weights, channels)
    else:
        print("No work to do: provide --intensity (single) or --roi-csv (batch).")


if __name__ == "__main__":
    main()
