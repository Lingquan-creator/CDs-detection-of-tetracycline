from __future__ import annotations

import json
import math
import os
import platform
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OPT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.environ.get("OUT_DIR", OPT_ROOT / "test_input" / "OUTPUT")).resolve()
TRAIN_CSV = Path(
    os.environ.get("TRAIN_CSV", PROJECT_ROOT / "dataset" / "mate20x" / "roi_results" / "roi_detailed_statistics.csv")
).resolve()
TEST_CSV = Path(
    os.environ.get("TEST_CSV", PROJECT_ROOT / "dataset" / "food_2" / "roi_results" / "roi_detailed_statistics.csv")
).resolve()

RAW_NAME_RE = re.compile(r"^(?P<sample>.+)-(?P<batch>\d+)-(?P<concentration>[-+]?\d+(?:\.\d+)?)$")
TRAILING_CONCENTRATION_RE = re.compile(
    r"^(?P<sample>.*?)(?P<concentration>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")

sys.path.insert(0, str(PROJECT_ROOT / "concentration_regression"))
from config import ALPHAS, CLIP_MIN, TARGET_EPS, WEIGHT_BETA, WEIGHT_EPS  # noqa: E402
from physics_features import build_feature_frame  # noqa: E402
from predict import draw_prediction_plot, predict_batch  # noqa: E402


BG_CORRECTION = "subtract"
G_MODE = "avg"
USE_BG = True
TARGET_TRANSFORM = "identity"
CSV_FLOAT_FORMAT = "%.6g"

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


def concentration_from_filename(filename: object) -> float:
    stem = Path(str(filename)).stem
    raw_match = RAW_NAME_RE.fullmatch(stem)
    if raw_match:
        return float(raw_match.group("concentration"))
    trailing_match = TRAILING_CONCENTRATION_RE.fullmatch(stem)
    if trailing_match and trailing_match.group("sample"):
        return float(trailing_match.group("concentration"))
    match = NUMBER_RE.search(stem)
    return float(match.group(0)) if match else math.nan


def load_dataframe(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"ROI CSV not found: {path}")
    df = pd.read_csv(path).copy()
    if "filename" not in df.columns:
        raise ValueError(f"ROI CSV must include filename column: {path}")
    df["conc_true"] = df["filename"].apply(concentration_from_filename)
    df = df.dropna(subset=["conc_true"]).sort_values(["conc_true", "filename"]).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No valid concentration labels parsed from {path}")
    return df, df["conc_true"].to_numpy(dtype=float)


def make_sample_weights(y: np.ndarray) -> np.ndarray:
    denom = WEIGHT_BETA * np.asarray(y, dtype=float) + WEIGHT_EPS
    weights = 1.0 / denom
    return weights / np.mean(weights)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if y_true.size == 0:
        return {"n": 0, "r2": math.nan, "rmse": math.nan, "mae": math.nan, "mean_error": math.nan}
    residual = y_pred - y_true
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "n": int(y_true.size),
        "r2": math.nan if ss_tot <= 0 else 1.0 - ss_res / ss_tot,
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mean_error": float(np.mean(residual)),
    }


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return build_feature_frame(
        df,
        g_mode=G_MODE,
        use_bg=USE_BG,
        clip_min=CLIP_MIN,
        bg_correction=BG_CORRECTION,
    ).copy()


def train_ridge(X: np.ndarray, y_true: np.ndarray) -> tuple[Pipeline, np.ndarray, np.ndarray]:
    sample_weight = make_sample_weights(y_true)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", RidgeCV(alphas=list(ALPHAS), scoring="neg_mean_squared_error")),
        ]
    )
    model.fit(X, y_true, regressor__sample_weight=sample_weight)
    pred_raw = np.maximum(model.predict(X), 0.0)
    pred = pred_raw
    return model, pred_raw, pred


def save_training_outputs(
    *,
    vector_dir: Path,
    variant_id: str,
    feature_names: tuple[str, ...],
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    model: Pipeline,
    train_pred_raw: np.ndarray,
    train_pred: np.ndarray,
) -> Path:
    vector_dir.mkdir(parents=True, exist_ok=True)
    regressor = model.named_steps["regressor"]
    selected_alpha = float(getattr(regressor, "alpha_", ALPHAS[0]))
    model_path = vector_dir / "concentration_model_v1.pkl"
    bundle = {
        "model": model,
        "g_mode": G_MODE,
        "alpha": selected_alpha,
        "alphas_tried": tuple(ALPHAS),
        "feature_names": feature_names,
        "target_transform": TARGET_TRANSFORM,
        "target_eps": TARGET_EPS,
        "use_bg": USE_BG,
        "feature_set": f"custom:{variant_id}",
        "bg_correction": BG_CORRECTION,
        "weight_eps": WEIGHT_EPS,
        "weight_beta": WEIGHT_BETA,
        "monotonic_calibration": False,
        "sklearn_version": sklearn_version,
        "python_version": platform.python_version(),
        "bundle_created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    joblib.dump(bundle, model_path)

    train_out = pd.DataFrame(
        {
            "filename": train_df["filename"],
            "conc_true": y_train,
            "conc_pred_raw": train_pred_raw,
            "conc_pred": train_pred,
            "residual": train_pred - y_train,
        }
    )
    train_out.to_csv(vector_dir / "ridge_predictions.csv", index=False, float_format=CSV_FLOAT_FORMAT)

    coef_out = pd.DataFrame({"feature": feature_names, "coefficient": regressor.coef_})
    coef_out.to_csv(vector_dir / "ridge_coefficients.csv", index=False, float_format=CSV_FLOAT_FORMAT)

    metadata = {
        "variant_id": variant_id,
        "feature_names": list(feature_names),
        "train_csv": str(TRAIN_CSV),
        "test_csv": str(TEST_CSV),
        "target_transform": TARGET_TRANSFORM,
        "monotonic_calibration": False,
        "selected_alpha": selected_alpha,
        "train_metrics": metrics(y_train, train_pred),
    }
    (vector_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    draw_train_plot(y_train, train_pred, vector_dir / "ridge_training_fit.png", variant_id)
    return model_path


def draw_train_plot(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str) -> None:
    row = metrics(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6.5, 5.4), dpi=150)
    low = float(min(np.min(y_true), np.min(y_pred)))
    high = float(max(np.max(y_true), np.max(y_pred)))
    pad = max((high - low) * 0.05, 1.0)
    low -= pad
    high += pad
    ax.scatter(y_true, y_pred, s=30, alpha=0.85)
    ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1.0)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("True concentration (uM)")
    ax.set_ylabel("Predicted concentration (uM)")
    ax.set_title(f"{title}\ntrain R2={row['r2']:.4g}, RMSE={row['rmse']:.4g}, MAE={row['mae']:.4g}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def prepare_output_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old_file in OUT_DIR.iterdir():
        if old_file.is_file():
            old_file.unlink()
    for variant_id in INPUT_VECTOR_CANDIDATES:
        variant_dir = OUT_DIR / variant_id
        if variant_dir.exists():
            shutil.rmtree(variant_dir)


def main() -> None:
    prepare_output_dir()
    train_df, y_train = load_dataframe(TRAIN_CSV)
    test_df, y_test = load_dataframe(TEST_CSV)
    train_features = build_features(train_df)
    test_features = build_features(test_df)

    print(f"Train CSV: {TRAIN_CSV} (n={len(train_df)})")
    print(f"Test CSV:  {TEST_CSV} (n={len(test_df)})")
    print(f"Output:    {OUT_DIR}")

    rows: list[dict[str, object]] = []
    for variant_id, feature_names in INPUT_VECTOR_CANDIDATES.items():
        missing_train = [name for name in feature_names if name not in train_features.columns]
        missing_test = [name for name in feature_names if name not in test_features.columns]
        if missing_train or missing_test:
            raise ValueError(f"{variant_id} missing features: train={missing_train}, test={missing_test}")

        vector_dir = OUT_DIR / variant_id
        X_train = train_features.loc[:, list(feature_names)].to_numpy(dtype=float)
        model, train_pred_raw, train_pred = train_ridge(X_train, y_train)
        model_path = save_training_outputs(
            vector_dir=vector_dir,
            variant_id=variant_id,
            feature_names=feature_names,
            train_df=train_df,
            y_train=y_train,
            model=model,
            train_pred_raw=train_pred_raw,
            train_pred=train_pred,
        )

        pred_csv = vector_dir / "food_2_predictions.csv"
        pred_png = vector_dir / "food_2_prediction.png"
        pred_df = predict_batch(
            model,
            TEST_CSV,
            pred_csv,
            G_MODE,
            target_transform=TARGET_TRANSFORM,
            target_eps=TARGET_EPS,
            use_bg=USE_BG,
            bg_correction=BG_CORRECTION,
            expected_feature_names=feature_names,
        )
        draw_prediction_plot(pred_df, pred_png)

        row = metrics(y_test, pred_df["conc_pred"].to_numpy(dtype=float))
        rows.append(
            {
                "variant_id": variant_id,
                "n_features": len(feature_names),
                "r2": row["r2"],
                "rmse": row["rmse"],
                "mae": row["mae"],
                "mean_error": row["mean_error"],
                "n": row["n"],
                "feature_names": ";".join(feature_names),
                "model_path": str(model_path),
                "prediction_csv": str(pred_csv),
            }
        )
        print(
            f"{variant_id}: R2={row['r2']:.6g}, RMSE={row['rmse']:.6g}, "
            f"MAE={row['mae']:.6g}"
        )

    r2_df = pd.DataFrame(rows).sort_values(["r2", "rmse"], ascending=[False, True]).reset_index(drop=True)
    r2_df.to_csv(OUT_DIR / "r2_by_input_vector.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    best = r2_df.iloc[0].to_dict()
    (OUT_DIR / "best_r2.txt").write_text(
        "\n".join(
            [
                f"variant_id={best['variant_id']}",
                f"r2={best['r2']:.6g}",
                f"rmse={best['rmse']:.6g}",
                f"mae={best['mae']:.6g}",
                f"feature_names={best['feature_names']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nBest input vectors:")
    print(r2_df[["variant_id", "n_features", "r2", "rmse", "mae", "mean_error", "feature_names"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
