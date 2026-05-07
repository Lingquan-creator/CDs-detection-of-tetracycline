from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
FIT_DATA_DIR = SCRIPT_DIR.parents[1]
ROOT = FIT_DATA_DIR.parent
REG_DIR = ROOT / "concentration_regression"
if str(REG_DIR) not in sys.path:
    sys.path.insert(0, str(REG_DIR))

from config import TARGET_EPS, TARGET_TRANSFORM  # noqa: E402
from train import (  # noqa: E402
    apply_calibrator,
    fit_calibrator,
    inverse_target as regression_inverse_target,
    transform_target as regression_transform_target,
)

DATASET_DIR = FIT_DATA_DIR / "dataset"
OUT_DIR = SCRIPT_DIR / "OUTPUT"

TRAIN_CSV = DATASET_DIR / "data.csv"
TEST_FILES = {
    "egg": DATASET_DIR / "egg.csv",
    "milk": DATASET_DIR / "milk.csv",
}
TEST_INPUT_RANKED_CSV = ROOT / "test_input_LOO" / "OUTPUT" / "loo_all_ranked.csv"
PARAMETER_SCAN_METRICS_CSV = ROOT / "parameter_scan_LOO" / "OUTPUT" / "regression_parameter_scan_metrics.csv"

DEFAULT_FEATURE_NAMES = (
    "ln_R_sig",
    "ln_G_sig",
    "ln_B_sig",
    "Ratio_RG",
    "Ratio_BG",
    "Ratio_RB",
    "Delta_RB",
)
DEFAULT_ALPHA = 1e8
DEFAULT_BETA = 1.0
DEFAULT_WEIGHT_EPS = 50.0
EPS = 1e-6
CSV_FLOAT_FORMAT = "%.6g"
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class HyperParams:
    alpha: float
    beta: float
    weight_eps: float
    variant_id: str


@dataclass(frozen=True)
class RGBBandConfig:
    scan_id: str
    b_band: tuple[float, float]
    g_band: tuple[float, float]
    r_band: tuple[float, float]

    @property
    def bands(self) -> dict[str, tuple[float, float]]:
        return {"B": self.b_band, "G": self.g_band, "R": self.r_band}

    @property
    def b_len(self) -> float:
        return self.b_band[1] - self.b_band[0]

    @property
    def g_len(self) -> float:
        return self.g_band[1] - self.g_band[0]

    @property
    def r_len(self) -> float:
        return self.r_band[1] - self.r_band[0]


def build_rgb_band_scan(
    start_nm: float = 400.0,
    end_nm: float = 700.0,
    initial_edge_len_nm: float = 100.0,
    step_nm: float = 10.0,
    min_g_len_nm: float = 20.0,
) -> list[RGBBandConfig]:
    configs: list[RGBBandConfig] = []
    edge_len = initial_edge_len_nm
    while True:
        b_high = start_nm + edge_len
        r_low = end_nm - edge_len
        g_len = r_low - b_high
        if g_len < min_g_len_nm - 1e-9:
            break
        configs.append(
            RGBBandConfig(
                scan_id=f"B{edge_len:g}_G{g_len:g}_R{edge_len:g}",
                b_band=(start_nm, b_high),
                g_band=(b_high, r_low),
                r_band=(r_low, end_nm),
            )
        )
        if g_len <= min_g_len_nm + 1e-9:
            break
        edge_len += step_nm
    return configs


def parse_concentration(label: object) -> float:
    match = NUMBER_RE.search(str(label))
    return float(match.group(0)) if match else math.nan


def parse_food_sample_label(label: object) -> tuple[str, int | None, float | None]:
    match = re.match(
        r"^(?P<sample>.+)-(?P<trial>\d+)-(?P<added>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$",
        str(label).strip(),
    )
    if not match:
        return str(label), None, math.nan
    return match.group("sample"), int(match.group("trial")), float(match.group("added"))


def load_best_feature_names(path: Path = TEST_INPUT_RANKED_CSV) -> tuple[str, ...]:
    if not path.exists():
        return DEFAULT_FEATURE_NAMES
    ranked = pd.read_csv(path)
    if ranked.empty or "feature_names" not in ranked.columns:
        return DEFAULT_FEATURE_NAMES
    features = tuple(str(ranked.iloc[0]["feature_names"]).split(";"))
    return tuple(feature for feature in features if feature) or DEFAULT_FEATURE_NAMES


def load_best_hyperparams(path: Path = PARAMETER_SCAN_METRICS_CSV) -> HyperParams:
    if not path.exists():
        return HyperParams(DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_WEIGHT_EPS, "defaults")
    metrics_df = pd.read_csv(path)
    rows = metrics_df[(metrics_df["phase"] == "loo") & (metrics_df["sample_group"] == "all")].copy()
    if rows.empty:
        return HyperParams(DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_WEIGHT_EPS, "defaults")
    best = rows.sort_values(["r2", "rmse"], ascending=[False, True]).iloc[0]
    return HyperParams(
        alpha=float(best["alpha"]),
        beta=float(best["beta"]),
        weight_eps=float(best["weight_eps"]),
        variant_id=str(best["variant_id"]),
    )


def integrate_band(wavelengths: np.ndarray, intensity: np.ndarray, band: tuple[float, float]) -> float:
    low, high = band
    mask = (wavelengths >= low) & (wavelengths <= high)
    if not np.any(mask):
        raise ValueError(f"No wavelength points in band {band}")
    return float(np.trapezoid(intensity[mask], wavelengths[mask]))


def build_feature_frame(rgb_df: pd.DataFrame) -> pd.DataFrame:
    r = np.maximum(rgb_df["R_sig"].to_numpy(dtype=float), EPS)
    g = np.maximum(rgb_df["G_sig"].to_numpy(dtype=float), EPS)
    b = np.maximum(rgb_df["B_sig"].to_numpy(dtype=float), EPS)
    ratio_rg = r / (g + EPS)
    ratio_bg = b / (g + EPS)
    feature_df = pd.DataFrame(
        {
            "ln_R_sig": np.log(r + EPS),
            "ln_G_sig": np.log(g + EPS),
            "ln_B_sig": np.log(b + EPS),
            "Ratio_RG": ratio_rg,
            "Ratio_BG": ratio_bg,
            "Ratio_RB": ratio_rg / (ratio_bg + EPS),
            "Delta_RG": (g - r) / (g + r + EPS),
            "Delta_GB": (ratio_bg - 1.0) / (ratio_bg + 1.0 + EPS),
            "Delta_RB": (ratio_bg - ratio_rg) / (ratio_bg + ratio_rg + EPS),
        }
    )
    return feature_df


def select_features(feature_df: pd.DataFrame, feature_names: Iterable[str]) -> pd.DataFrame:
    feature_names = tuple(feature_names)
    missing = [feature for feature in feature_names if feature not in feature_df.columns]
    if missing:
        raise ValueError(f"Selected input vector requires missing features: {missing}")
    return feature_df.loc[:, list(feature_names)]


def add_band_metadata(df: pd.DataFrame, band_config: RGBBandConfig) -> pd.DataFrame:
    out = df.copy()
    out["scan_id"] = band_config.scan_id
    out["B_low_nm"] = band_config.b_band[0]
    out["B_high_nm"] = band_config.b_band[1]
    out["G_low_nm"] = band_config.g_band[0]
    out["G_high_nm"] = band_config.g_band[1]
    out["R_low_nm"] = band_config.r_band[0]
    out["R_high_nm"] = band_config.r_band[1]
    out["B_len_nm"] = band_config.b_len
    out["G_len_nm"] = band_config.g_len
    out["R_len_nm"] = band_config.r_len
    return out


def load_training_spectra(path: Path, band_config: RGBBandConfig) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    wavelengths = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    for column in df.columns[1:]:
        concentration = parse_concentration(column)
        intensity = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(wavelengths) & np.isfinite(intensity)
        row = {
            "source_file": path.name,
            "label": column,
            "sample": "standard",
            "trial": math.nan,
            "conc_true": concentration,
        }
        for channel, band in band_config.bands.items():
            row[f"{channel}_sig"] = integrate_band(wavelengths[valid], intensity[valid], band)
        rows.append(row)
    out = pd.DataFrame(rows).dropna(subset=["conc_true"]).sort_values("conc_true").reset_index(drop=True)
    out = add_band_metadata(out, band_config)
    return out, out["conc_true"].to_numpy(dtype=float)


def load_food_spectra(path: Path, band_config: RGBBandConfig) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    wavelengths = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    for column in df.columns[1:]:
        sample, trial, added = parse_food_sample_label(column)
        intensity = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(wavelengths) & np.isfinite(intensity)
        row = {
            "source_file": path.name,
            "label": column,
            "sample": sample,
            "trial": trial,
            "conc_true": added,
        }
        for channel, band in band_config.bands.items():
            row[f"{channel}_sig"] = integrate_band(wavelengths[valid], intensity[valid], band)
        rows.append(row)
    return add_band_metadata(pd.DataFrame(rows), band_config)


def transform_target(y: np.ndarray) -> np.ndarray:
    return regression_transform_target(np.asarray(y, dtype=float), TARGET_TRANSFORM, TARGET_EPS)


def inverse_target(pred: np.ndarray) -> np.ndarray:
    return regression_inverse_target(np.asarray(pred, dtype=float), TARGET_TRANSFORM, TARGET_EPS)


def make_sample_weights(y: np.ndarray, hyperparams: HyperParams) -> np.ndarray:
    denom = hyperparams.beta * np.asarray(y, dtype=float) + hyperparams.weight_eps
    if np.any(denom <= 0):
        raise ValueError("beta * concentration + weight_eps must stay positive.")
    weights = 1.0 / denom
    return weights / np.mean(weights)


def train_model(X: np.ndarray, y_true: np.ndarray, hyperparams: HyperParams) -> tuple[Pipeline, IsotonicRegression | None, np.ndarray, np.ndarray]:
    sample_weight = make_sample_weights(y_true, hyperparams)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=hyperparams.alpha, fit_intercept=True)),
        ]
    )
    model.fit(X, transform_target(y_true), regressor__sample_weight=sample_weight)
    pred_raw = inverse_target(model.predict(X))
    calibrator = fit_calibrator(pred_raw, y_true, sample_weight)
    pred = apply_calibrator(pred_raw, calibrator)
    return model, calibrator, pred_raw, pred


def run_leave_one_out(X: np.ndarray, y_true: np.ndarray, hyperparams: HyperParams) -> np.ndarray:
    pred = np.zeros_like(y_true, dtype=float)
    for train_idx, test_idx in LeaveOneOut().split(X):
        fold_model, fold_calibrator, _, _ = train_model(X[train_idx], y_true[train_idx], hyperparams)
        raw = inverse_target(fold_model.predict(X[test_idx]))
        pred[test_idx] = apply_calibrator(raw, fold_calibrator)
    return pred


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        return {"n": 0, "r2": math.nan, "rmse": math.nan, "mae": math.nan, "mean_error": math.nan, "median_abs_error": math.nan, "max_abs_error": math.nan}
    y = y_true[valid]
    pred = y_pred[valid]
    residual = pred - y
    return {
        "n": int(valid.sum()),
        "r2": float(r2_score(y, pred)) if y.size >= 2 else math.nan,
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "mean_error": float(np.mean(residual)),
        "median_abs_error": float(np.median(np.abs(residual))),
        "max_abs_error": float(np.max(np.abs(residual))),
    }


def build_prediction_frame(
    rows: pd.DataFrame,
    pred_raw: np.ndarray,
    pred: np.ndarray,
    phase: str,
    feature_names: tuple[str, ...],
    hyperparams: HyperParams,
) -> pd.DataFrame:
    columns = [
        "scan_id",
        "B_low_nm",
        "B_high_nm",
        "G_low_nm",
        "G_high_nm",
        "R_low_nm",
        "R_high_nm",
        "B_len_nm",
        "G_len_nm",
        "R_len_nm",
        "source_file",
        "label",
        "sample",
        "trial",
        "conc_true",
        "R_sig",
        "G_sig",
        "B_sig",
    ]
    out = rows[columns].copy()
    out.insert(0, "phase", phase)
    out["feature_names"] = ";".join(feature_names)
    out["alpha"] = hyperparams.alpha
    out["beta"] = hyperparams.beta
    out["weight_eps"] = hyperparams.weight_eps
    out["conc_pred_raw"] = pred_raw
    out["conc_pred"] = pred
    out["residual"] = out["conc_pred"] - out["conc_true"]
    out["abs_error"] = out["residual"].abs()
    return out


def build_metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scan_id, scan_part in predictions.groupby("scan_id", sort=False):
        metadata = scan_part.iloc[0]
        for phase, part in scan_part.groupby("phase", sort=False):
            groups = ["all"] if phase != "food" else ["all", "egg", "milk"]
            for group in groups:
                source = part if group == "all" else part[part["sample"] == group]
                row = {
                    "scan_id": scan_id,
                    "B_low_nm": metadata["B_low_nm"],
                    "B_high_nm": metadata["B_high_nm"],
                    "G_low_nm": metadata["G_low_nm"],
                    "G_high_nm": metadata["G_high_nm"],
                    "R_low_nm": metadata["R_low_nm"],
                    "R_high_nm": metadata["R_high_nm"],
                    "B_len_nm": metadata["B_len_nm"],
                    "G_len_nm": metadata["G_len_nm"],
                    "R_len_nm": metadata["R_len_nm"],
                    "phase": phase,
                    "sample": group,
                }
                row.update(metrics(source["conc_true"].to_numpy(float), source["conc_pred"].to_numpy(float)))
                rows.append(row)
    return pd.DataFrame(rows)


def rank_scans_by_loo(metrics_df: pd.DataFrame) -> pd.DataFrame:
    ranked = metrics_df[(metrics_df["phase"] == "loo") & (metrics_df["sample"] == "all")].copy()
    return ranked.sort_values(["r2", "rmse", "mae"], ascending=[False, True, True]).reset_index(drop=True)


def plot_predictions(predictions: pd.DataFrame, metrics_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), dpi=150)
    for ax, phase, title in ((axes[0], "loo", "Training LOO"), (axes[1], "food", "Egg/Milk test")):
        df = predictions[predictions["phase"] == phase]
        colors = {"standard": "#1f77b4", "egg": "#ff7f0e", "milk": "#2ca02c"}
        for sample, part in df.groupby("sample"):
            ax.scatter(part["conc_true"], part["conc_pred"], s=42, alpha=0.85, label=sample, color=colors.get(sample, "#777777"), edgecolor="white", linewidth=0.4)
        low = float(min(df["conc_true"].min(), df["conc_pred"].min()))
        high = float(max(df["conc_true"].max(), df["conc_pred"].max()))
        pad = max((high - low) * 0.05, 1.0)
        ax.plot([low - pad, high + pad], [low - pad, high + pad], "k--", linewidth=1.0)
        row = metrics_df[(metrics_df["phase"] == phase) & (metrics_df["sample"] == "all")].iloc[0]
        ax.set_title(f"{title}\nR2={row['r2']:.4g}, RMSE={row['rmse']:.4g}, MAE={row['mae']:.4g}")
        ax.set_xlabel("True concentration (uM)")
        ax.set_ylabel("Predicted concentration (uM)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feature_names = load_best_feature_names()
    hyperparams = load_best_hyperparams()
    band_configs = build_rgb_band_scan()

    prediction_frames: list[pd.DataFrame] = []
    train_feature_frames: list[pd.DataFrame] = []
    food_feature_frames: list[pd.DataFrame] = []
    for band_config in band_configs:
        train_rows, y_train = load_training_spectra(TRAIN_CSV, band_config)
        train_features = select_features(build_feature_frame(train_rows), feature_names).to_numpy(dtype=float)
        model, calibrator, train_pred_raw, train_pred = train_model(train_features, y_train, hyperparams)
        loo_pred = run_leave_one_out(train_features, y_train, hyperparams)

        prediction_frames.extend(
            [
                build_prediction_frame(train_rows, train_pred_raw, train_pred, "train", feature_names, hyperparams),
                build_prediction_frame(train_rows, np.full_like(loo_pred, np.nan), loo_pred, "loo", feature_names, hyperparams),
            ]
        )

        food_rows = pd.concat([load_food_spectra(path, band_config) for path in TEST_FILES.values()], ignore_index=True)
        food_features = select_features(build_feature_frame(food_rows), feature_names).to_numpy(dtype=float)
        food_pred_raw = inverse_target(model.predict(food_features))
        food_pred = apply_calibrator(food_pred_raw, calibrator)
        prediction_frames.append(build_prediction_frame(food_rows, food_pred_raw, food_pred, "food", feature_names, hyperparams))
        train_feature_frames.append(train_rows.copy())
        food_feature_frames.append(food_rows.copy())

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics_df = build_metric_rows(predictions)
    ranked_scans = rank_scans_by_loo(metrics_df)

    rgb_train = pd.concat(train_feature_frames, ignore_index=True)
    rgb_food = pd.concat(food_feature_frames, ignore_index=True)
    rgb_train.to_csv(OUT_DIR / "spectral_rgb_training_features.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    rgb_food.to_csv(OUT_DIR / "spectral_rgb_food_features.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    predictions.to_csv(OUT_DIR / "spectral_rgb_predictions.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    metrics_df.to_csv(OUT_DIR / "spectral_rgb_metrics.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    ranked_scans.to_csv(OUT_DIR / "spectral_rgb_scan_ranked_by_loo.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    best_scan_id = str(ranked_scans.iloc[0]["scan_id"])
    best_predictions = predictions[predictions["scan_id"] == best_scan_id].copy()
    best_metrics = metrics_df[metrics_df["scan_id"] == best_scan_id].copy()
    best_predictions.to_csv(OUT_DIR / "spectral_rgb_best_loo_predictions.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    best_metrics.to_csv(OUT_DIR / "spectral_rgb_best_loo_metrics.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    plot_predictions(best_predictions, best_metrics, OUT_DIR / "spectral_rgb_prediction_plot.png")

    with (OUT_DIR / "spectral_rgb_summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"Training CSV: {TRAIN_CSV}\n")
        f.write("Test CSVs:\n")
        for path in TEST_FILES.values():
            f.write(f"  {path}\n")
        f.write("RGB scan rule: start at B/G/R = 100/100/100 nm over 400-700 nm; increase B and R by 10 nm each step until G is 20 nm.\n")
        f.write("Scanned RGB bands (nm):\n")
        for config in band_configs:
            f.write(f"  {config.scan_id}: B={config.b_band}, G={config.g_band}, R={config.r_band}\n")
        f.write(f"Feature source: {TEST_INPUT_RANKED_CSV}\n")
        f.write(f"Features: {';'.join(feature_names)}\n")
        f.write(f"Hyperparameter source: {PARAMETER_SCAN_METRICS_CSV}\n")
        f.write(f"Hyperparameters: variant={hyperparams.variant_id}, alpha={hyperparams.alpha:g}, beta={hyperparams.beta:g}, weight_eps={hyperparams.weight_eps:g}\n\n")
        f.write(f"Best scan by training LOO: {best_scan_id}\n\n")
        f.write("Ranked scans by training LOO:\n")
        f.write(ranked_scans.to_string(index=False))
        f.write("\n\nAll metrics:\n")
        f.write(metrics_df.to_string(index=False))
        f.write("\n")

    print(f"Saved outputs -> {OUT_DIR}")
    print(f"Best scan by training LOO -> {best_scan_id}")
    print(ranked_scans.to_string(index=False))


if __name__ == "__main__":
    main()
