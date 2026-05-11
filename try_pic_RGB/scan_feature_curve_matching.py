from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIC_RGB_DIR = PROJECT_ROOT / "pic_RGB"
EXP_DIR = PROJECT_ROOT / "concentration_EXP"
REGRESSION_DIR = PROJECT_ROOT / "concentration_regression"

for path in (PIC_RGB_DIR, EXP_DIR, REGRESSION_DIR):
    path_text = str(path)
    if path_text in sys.path:
        sys.path.remove(path_text)
sys.path[:0] = [str(PIC_RGB_DIR), str(EXP_DIR), str(REGRESSION_DIR)]

from roi_config import SAVE_VISUALIZATION, get_settings  # noqa: E402
from roi_processor import ROIProcessor  # noqa: E402


def load_module_from_path(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp_train = load_module_from_path("try_pic_rgb_exp_train", EXP_DIR / "train.py")
regression_features = load_module_from_path(
    "try_pic_rgb_regression_features",
    REGRESSION_DIR / "physics_features.py",
)
FIT_MODEL_NAMES = exp_train.FIT_MODEL_NAMES
fit_response_model = exp_train.fit_response_model
predict_response_model = exp_train.predict_response_model
build_feature_frame = regression_features.build_feature_frame


CSV_FLOAT_FORMAT = "%.6g"
R2_THRESHOLD = 0.7
WEIGHT_EPS = 5.0
GRID_MIN = 0.0
GRID_MAX = 100.0
GRID_POINTS = 101
MODEL_PRIORITY = {
    "linear": 0,
    "exponential_c0": 1,
    "exponential_offset": 2,
    "quadratic": 3,
}

INPUT_VECTOR_CANDIDATES: dict[str, tuple[str, ...]] = {
    "ln_rgb": ("ln_R_sig", "ln_G_sig", "ln_B_sig"),
    "ln_rgb_total": ("ln_R_sig", "ln_G_sig", "ln_B_sig", "ln_RGB_sum"),
    "ln_rgb_total_delta_ratio": (
        "ln_R_sig",
        "ln_G_sig",
        "ln_B_sig",
        "ln_RGB_sum",
        "Delta_RG",
        "Delta_RB",
        "Ratio_RG",
    ),
    "ln_rgb_log_ratios": (
        "ln_R_sig",
        "ln_G_sig",
        "ln_B_sig",
        "ln_RGB_sum",
        "ln_R_over_G",
        "ln_B_over_G",
    ),
    "ln_total_pair_sum": ("ln_RGB_sum", "ln_RG_sum", "ln_RB_sum", "ln_GB_sum"),
    "ln_total_pair_contrast": ("ln_RGB_sum", "ln_RG_over_GB", "ln_RB_over_GB"),
    "rgb_raw_ln_total_contrast": (
        "R_sig",
        "G_sig",
        "B_sig",
        "ln_RGB_sum",
        "ln_R_over_G",
        "ln_B_over_G",
    ),
    "rgb_sum_chroma_ln_total_contrast": (
        "RGB_sum",
        "R_chroma",
        "B_chroma",
        "ln_RGB_sum",
        "ln_R_over_G",
        "ln_B_over_G",
    ),
    "rgb_sum_chroma_ln_total_pair_contrast": (
        "RGB_sum",
        "R_chroma",
        "B_chroma",
        "ln_RGB_sum",
        "ln_RG_over_GB",
        "ln_RB_over_GB",
    ),
}
EXTRA_FEATURES = ("G_chroma",)
FEATURE_NAMES = tuple(
    dict.fromkeys(
        feature
        for features in INPUT_VECTOR_CANDIDATES.values()
        for feature in (*features, *EXTRA_FEATURES)
    )
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    data_dir: Path
    output_dir: Path


@dataclass
class LoadedImage:
    path: Path
    raw_channels: np.ndarray
    metadata: dict[str, Any]
    roi_selection: Any
    bg_selection: Any


class ConfigurableStatsROIProcessor(ROIProcessor):
    def __init__(self, *args: Any, statistic_method: str = "trimmed_mean", **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.statistic_method = statistic_method

    def _robust_stat(self, values: np.ndarray) -> float:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return 0.0
        method = self.statistic_method.strip().lower()
        if method == "median":
            return float(np.median(finite))
        if method == "trimmed_mean":
            return self._trimmed_mean(finite)
        raise ValueError(f"Unsupported statistic method: {self.statistic_method}")

    def compute_roi_statistics(
        self,
        raw_channels: np.ndarray,
        roi_mask: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        stats: dict[str, float] = {}
        for index, name in enumerate(("R", "G1", "G2", "B")):
            channel_data = raw_channels[:, :, index]
            values = channel_data[roi_mask > 0]
            stats[f"{name}_mean"] = self._robust_stat(values)
        return stats


def parse_float_list(text: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def parse_str_list(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def make_processor(
    *,
    width_fraction: float,
    height_fraction: float,
    trim_fraction: float,
    statistic_method: str,
    selection_cache_path: Path | None = None,
) -> ConfigurableStatsROIProcessor:
    settings = get_settings(
        {
            "measurement_roi_width_fraction": width_fraction,
            "measurement_roi_height_fraction": height_fraction,
            "trim_fraction": trim_fraction,
        }
    )
    return ConfigurableStatsROIProcessor(
        settings=settings,
        statistic_method=statistic_method,
        reuse_last_selection=True,
        selection_cache_path=selection_cache_path,
    )


def load_dataset_images(spec: DatasetSpec) -> list[LoadedImage]:
    processor = make_processor(
        width_fraction=0.4,
        height_fraction=0.4,
        trim_fraction=0.08,
        statistic_method="trimmed_mean",
        selection_cache_path=spec.output_dir / "roi_selection_cache.json",
    )
    images: list[LoadedImage] = []
    for path in sorted(spec.data_dir.glob("*.dng")):
        cached = processor._get_cached_selection(path.name)
        if cached is None:
            raise ValueError(f"{spec.name}: missing cached ROI/BG selection for {path.name}")
        raw_channels, metadata = processor.load_raw_image(path)
        images.append(
            LoadedImage(
                path=path,
                raw_channels=raw_channels,
                metadata=metadata,
                roi_selection=cached[0],
                bg_selection=cached[1],
            )
        )
    if not images:
        raise ValueError(f"{spec.name}: no .dng images found in {spec.data_dir}")
    return images


def compute_roi_dataframe(
    images: list[LoadedImage],
    *,
    width_fraction: float,
    height_fraction: float,
    trim_fraction: float,
    statistic_method: str,
) -> pd.DataFrame:
    processor = make_processor(
        width_fraction=width_fraction,
        height_fraction=height_fraction,
        trim_fraction=trim_fraction,
        statistic_method=statistic_method,
    )
    rows: list[dict[str, str]] = []
    for image in images:
        with contextlib.redirect_stdout(io.StringIO()):
            result = processor._process_loaded_image(
                filepath=image.path,
                raw_channels=image.raw_channels,
                metadata=image.metadata,
                roi_sel=image.roi_selection,
                bg_sel=image.bg_selection,
                save_visualization=False,
                output_dir=None,
            )
        rows.append(processor._build_output_record(result))
    return pd.DataFrame(rows)


def build_candidate_features(roi_df: pd.DataFrame) -> pd.DataFrame:
    roi_df = roi_df.copy()
    for column in roi_df.columns:
        if column != "filename":
            roi_df[column] = pd.to_numeric(roi_df[column], errors="coerce")
    feature_df = build_feature_frame(
        roi_df,
        g_mode="avg",
        use_bg=True,
        clip_min=0.1,
        bg_correction="subtract",
    )
    missing = [feature for feature in FEATURE_NAMES if feature not in feature_df.columns]
    if missing:
        raise ValueError(f"Cannot build candidate features: {missing}")
    out = feature_df.loc[:, list(FEATURE_NAMES)].copy()
    out.insert(0, "filename", roi_df["filename"].to_numpy())
    out.insert(1, "conc", roi_df["filename"].apply(parse_concentration).to_numpy(dtype=float))
    out = out.dropna(subset=["conc"]).sort_values(["conc", "filename"]).reset_index(drop=True)
    return out


def parse_concentration(filename: object) -> float:
    stem = Path(str(filename)).stem
    parts = stem.split("-")
    if len(parts) >= 3:
        try:
            return float(parts[-1])
        except ValueError:
            pass
    try:
        return float(stem)
    except ValueError:
        import re

        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", stem)
        return float(match.group(0)) if match else math.nan


def finite_fit(
    model_name: str,
    conc: np.ndarray,
    values: np.ndarray,
) -> dict[str, Any] | None:
    valid = np.isfinite(conc) & np.isfinite(values)
    x = conc[valid]
    y = values[valid]
    if x.size < 3:
        return None
    try:
        params, r2, rmse, equation = fit_response_model(model_name, x, y, weight_eps=WEIGHT_EPS)
    except Exception:
        return None
    if not np.isfinite(r2):
        return None
    return {
        "model": model_name,
        "params": tuple(float(value) for value in params),
        "r2": float(r2),
        "rmse": float(rmse),
        "equation": equation,
    }


def choose_feature_model(
    feature: str,
    food_features: pd.DataFrame,
    mate_features: pd.DataFrame,
) -> dict[str, Any]:
    food_x = food_features["conc"].to_numpy(dtype=float)
    mate_x = mate_features["conc"].to_numpy(dtype=float)
    food_y = food_features[feature].to_numpy(dtype=float)
    mate_y = mate_features[feature].to_numpy(dtype=float)

    candidates: list[dict[str, Any]] = []
    for model_name in FIT_MODEL_NAMES:
        food_fit = finite_fit(model_name, food_x, food_y)
        mate_fit = finite_fit(model_name, mate_x, mate_y)
        if food_fit is None or mate_fit is None:
            continue
        candidates.append(
            {
                "feature": feature,
                "model": model_name,
                "food_params": food_fit["params"],
                "mate_params": mate_fit["params"],
                "food_r2": food_fit["r2"],
                "mate_r2": mate_fit["r2"],
                "food_rmse": food_fit["rmse"],
                "mate_rmse": mate_fit["rmse"],
                "min_r2": min(food_fit["r2"], mate_fit["r2"]),
                "mean_r2": (food_fit["r2"] + mate_fit["r2"]) / 2.0,
            }
        )
    if not candidates:
        return {"feature": feature, "kept": False, "drop_reason": "all_models_failed"}

    candidates.sort(
        key=lambda row: (
            -float(row["min_r2"]),
            MODEL_PRIORITY.get(str(row["model"]), 999),
            -float(row["mean_r2"]),
        )
    )
    best = candidates[0]
    best["kept"] = bool(best["min_r2"] >= R2_THRESHOLD)
    best["drop_reason"] = "" if best["kept"] else f"best_min_r2<{R2_THRESHOLD:g}"
    best["tested_models"] = ";".join(str(row["model"]) for row in candidates)
    return best


def curve_difference(model_name: str, food_params: Iterable[float], mate_params: Iterable[float]) -> float:
    grid = np.linspace(GRID_MIN, GRID_MAX, GRID_POINTS)
    food_curve = np.asarray(predict_response_model(model_name, grid, tuple(food_params)), dtype=float)
    mate_curve = np.asarray(predict_response_model(model_name, grid, tuple(mate_params)), dtype=float)
    valid = np.isfinite(food_curve) & np.isfinite(mate_curve)
    if not np.any(valid):
        return float("nan")
    food_curve = food_curve[valid]
    mate_curve = mate_curve[valid]
    positive = (food_curve > 0.0) & (mate_curve > 0.0)
    if np.all(positive):
        return float(np.mean(np.abs(np.log(food_curve / mate_curve))))
    combined = np.concatenate([food_curve, mate_curve])
    scale = float(np.nanmax(combined) - np.nanmin(combined))
    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    return float(np.mean(np.abs(food_curve - mate_curve)) / scale)


def measured_common_feature_score(food_features: pd.DataFrame, mate_features: pd.DataFrame, features: list[str]) -> float:
    food_group = food_features.groupby("conc")[features].mean()
    mate_group = mate_features.groupby("conc")[features].mean()
    common = sorted(set(food_group.index).intersection(set(mate_group.index)))
    values: list[float] = []
    for conc in common:
        for feature in features:
            a = float(food_group.loc[conc, feature])
            b = float(mate_group.loc[conc, feature])
            if np.isfinite(a) and np.isfinite(b):
                if a > 0 and b > 0:
                    values.append(abs(math.log(a / b)))
                else:
                    scale = max(abs(a), abs(b), 1.0)
                    values.append(abs(a - b) / scale)
    return float(np.mean(values)) if values else float("nan")


def evaluate_candidate(
    *,
    food_roi_df: pd.DataFrame,
    mate_roi_df: pd.DataFrame,
    width_fraction: float,
    height_fraction: float,
    trim_fraction: float,
    statistic_method: str,
) -> dict[str, Any]:
    food_features = build_candidate_features(food_roi_df)
    mate_features = build_candidate_features(mate_roi_df)

    feature_rows = [choose_feature_model(feature, food_features, mate_features) for feature in FEATURE_NAMES]
    kept = [row for row in feature_rows if row.get("kept")]
    dropped = [row for row in feature_rows if not row.get("kept")]
    for row in kept:
        row["curve_score"] = curve_difference(
            str(row["model"]),
            row["food_params"],
            row["mate_params"],
        )
    curve_scores = [float(row["curve_score"]) for row in kept if np.isfinite(float(row["curve_score"]))]

    row: dict[str, Any] = {
        "measurement_roi_width_fraction": width_fraction,
        "measurement_roi_height_fraction": height_fraction,
        "roi_statistic_method": statistic_method,
        "trim_fraction": trim_fraction,
        "total_curve_score": float(np.mean(curve_scores)) if curve_scores else float("inf"),
        "kept_feature_count": len(kept),
        "dropped_feature_count": len(dropped),
        "measured_common_feature_score": measured_common_feature_score(
            food_features,
            mate_features,
            [str(item["feature"]) for item in kept],
        )
        if kept
        else float("nan"),
        "kept_features": ";".join(str(item["feature"]) for item in kept),
        "dropped_features": ";".join(str(item["feature"]) for item in dropped),
        "feature_fit_details_json": json.dumps(feature_rows, ensure_ascii=False, default=json_default),
    }
    for item in kept:
        prefix = f"feature_{item['feature']}"
        row[f"{prefix}_model"] = item["model"]
        row[f"{prefix}_curve_score"] = item["curve_score"]
        row[f"{prefix}_food_r2"] = item["food_r2"]
        row[f"{prefix}_mate_r2"] = item["mate_r2"]
    return row


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot JSON serialize {type(value)!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan pic_RGB ROI stats with multi-feature curve matching.")
    parser.add_argument("--food-dir", type=Path, default=PROJECT_ROOT / "dataset" / "food_2")
    parser.add_argument("--food-output-dir", type=Path, default=PROJECT_ROOT / "dataset" / "food_2" / "roi_results")
    parser.add_argument("--mate-dir", type=Path, default=PROJECT_ROOT / "dataset" / "mate20x")
    parser.add_argument("--mate-output-dir", type=Path, default=PROJECT_ROOT / "dataset" / "mate20x" / "roi_results")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "try_pic_RGB" / "feature_curve_matching_eval.csv")
    parser.add_argument("--width-fractions", default="0.25,0.30,0.35,0.40")
    parser.add_argument("--height-fractions", default="0.25,0.30,0.35,0.40")
    parser.add_argument("--trim-fractions", default="0.05,0.08,0.12")
    parser.add_argument("--methods", default="median,trimmed_mean")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    food_spec = DatasetSpec("food_2", args.food_dir.resolve(), args.food_output_dir.resolve())
    mate_spec = DatasetSpec("mate20x", args.mate_dir.resolve(), args.mate_output_dir.resolve())

    print("Loading RAW images and cached ROI/BG selections...")
    food_images = load_dataset_images(food_spec)
    mate_images = load_dataset_images(mate_spec)
    print(f"Loaded {len(food_images)} food_2 images and {len(mate_images)} mate20x images")

    width_fractions = parse_float_list(args.width_fractions)
    height_fractions = parse_float_list(args.height_fractions)
    trim_fractions = parse_float_list(args.trim_fractions)
    methods = parse_str_list(args.methods)

    rows: list[dict[str, Any]] = []
    total = len(width_fractions) * len(height_fractions) * len(trim_fractions) * len(methods)
    index = 0
    for width_fraction in width_fractions:
        for height_fraction in height_fractions:
            for trim_fraction in trim_fractions:
                for method in methods:
                    index += 1
                    food_roi_df = compute_roi_dataframe(
                        food_images,
                        width_fraction=width_fraction,
                        height_fraction=height_fraction,
                        trim_fraction=trim_fraction,
                        statistic_method=method,
                    )
                    mate_roi_df = compute_roi_dataframe(
                        mate_images,
                        width_fraction=width_fraction,
                        height_fraction=height_fraction,
                        trim_fraction=trim_fraction,
                        statistic_method=method,
                    )
                    row = evaluate_candidate(
                        food_roi_df=food_roi_df,
                        mate_roi_df=mate_roi_df,
                        width_fraction=width_fraction,
                        height_fraction=height_fraction,
                        trim_fraction=trim_fraction,
                        statistic_method=method,
                    )
                    rows.append(row)
                    print(
                        f"[{index:03d}/{total:03d}] "
                        f"wf={width_fraction:.3g} hf={height_fraction:.3g} method={method} trim={trim_fraction:.3g} "
                        f"score={row['total_curve_score']:.6g} kept={row['kept_feature_count']} "
                        f"dropped={row['dropped_feature_count']}",
                        flush=True,
                    )

    out_df = pd.DataFrame(rows).sort_values(
        ["total_curve_score", "kept_feature_count", "measured_common_feature_score"],
        ascending=[True, False, True],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False, float_format=CSV_FLOAT_FORMAT)
    best = out_df.iloc[0]
    print(f"\nSaved evaluation table -> {args.out}")
    print(
        "Best candidate: "
        f"wf={best['measurement_roi_width_fraction']:.6g}, "
        f"hf={best['measurement_roi_height_fraction']:.6g}, "
        f"method={best['roi_statistic_method']}, "
        f"trim={best['trim_fraction']:.6g}, "
        f"score={best['total_curve_score']:.6g}, "
        f"kept={int(best['kept_feature_count'])}, "
        f"dropped={int(best['dropped_feature_count'])}"
    )


if __name__ == "__main__":
    main()
