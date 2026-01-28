"""Apply a trained mapper to pic_2 ROI results and produce mapped outputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

if __name__ == "__main__" and __package__ is None:
    # Allow running as a script (python device_mapping/apply_mapper.py).
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    __package__ = "multi_device_rgb.device_mapping"

import joblib
import numpy as np

from .config import MappingConfig
from .data import CHANNELS_4, CHANNELS_3, _corrected_value, load_roi_json


def load_bundle(mapper_path: Path):
    bundle = joblib.load(mapper_path)
    return bundle


def apply_mapping(bundle: Dict, roi_input: Path, output_path: Path):
    records = load_roi_json(roi_input)

    use_four_channels = bundle["use_four_channels"]
    feature_names = list(CHANNELS_4 if use_four_channels else CHANNELS_3)
    min_clip = float(bundle.get("min_clip", 0.1))

    # Prepare input matrix
    X = []
    for rec in records:
        if use_four_channels:
            row = [_corrected_value(rec, ch, min_clip=min_clip) for ch in CHANNELS_4]
        else:
            r = _corrected_value(rec, "R", min_clip)
            g = (_corrected_value(rec, "G1", min_clip) + _corrected_value(rec, "G2", min_clip)) / 2.0
            b = _corrected_value(rec, "B", min_clip)
            row = [r, g, b]
        X.append(row)
    X = np.asarray(X, dtype=float)

    # Predict per channel
    mapped_values: List[np.ndarray] = []
    for ch in feature_names:
        model = bundle["models"][ch]
        mapped_values.append(model.predict(X))
    Y = np.vstack(mapped_values).T  # shape (n, n_channels)

    # Attach mapped values to records
    for rec, mapped_row in zip(records, Y):
        if use_four_channels:
            rec["R_mapped"] = float(mapped_row[0])
            rec["G1_mapped"] = float(mapped_row[1])
            rec["G2_mapped"] = float(mapped_row[2])
            rec["B_mapped"] = float(mapped_row[3])
        else:
            rec["R_mapped"] = float(mapped_row[0])
            rec["G_mapped"] = float(mapped_row[1])
            rec["B_mapped"] = float(mapped_row[2])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return output_path


def parse_args():
    cfg = MappingConfig()
    parser = argparse.ArgumentParser(description="Map pic_2 ROI JSON into pic_1 color space.")
    parser.add_argument(
        "--mapper",
        type=Path,
        default=cfg.results_dir / "ridge_mapper.pkl",
        help="Path to trained mapper pickle.",
    )
    parser.add_argument(
        "--input_roi",
        type=Path,
        default=cfg.pic2_dir / "roi_results" / cfg.roi_filename,
        help="ROI JSON from pic_2.",
    )
    parser.add_argument(
        "--output_roi",
        type=Path,
        default=cfg.results_dir / "pic2_mapped_roi_results.json",
        help="Where to save mapped ROI JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    bundle = load_bundle(args.mapper)
    out = apply_mapping(bundle, args.input_roi, args.output_roi)
    print(f"[OK] Saved mapped ROI JSON -> {out}")


if __name__ == "__main__":
    main()
