"""
Run concentration inference from ROI statistics using a trained Ridge model bundle.

Defaults are read from config.PredictConfig:
- model path: concentration_regression/rgb_fit_output/concentration_model_v1.pkl
- input CSV: pic_RGB/roi_results/roi_detailed_statistics.csv
"""

from __future__ import annotations

import argparse
import re
import shutil
import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import joblib
import matplotlib
import numpy as np
import pandas as pd

from config import CLIP_MIN, PredictConfig, TARGET_EPS
from physics_features import DEFAULT_G_MODE, build_feature_frame, extract_concentration_from_name, has_background_columns

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


CSV_FLOAT_FORMAT = "%.6g"


def format_plot_number(value: float, _pos: object = None) -> str:
    if not np.isfinite(value):
        return ""
    text = f"{value:.6g}"
    return "0" if text in ("", "-0") else text


def format_figure_numbers(fig: plt.Figure) -> None:
    formatter = FuncFormatter(format_plot_number)
    for ax in fig.axes:
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)


plt.rcParams.update({"axes.unicode_minus": False})

_INCONSISTENT_VERSION_RE = re.compile(
    r"Trying to unpickle estimator\s+(?P<estimator>[A-Za-z0-9_]+)\s+from version\s+(?P<saved>[A-Za-z0-9_.+-]+)\s+when using version\s+(?P<runtime>[A-Za-z0-9_.+-]+)"
)


def _summarize_inconsistent_version(messages: List[str]) -> Tuple[Optional[str], Optional[str], List[str]]:
    saved_versions = set()
    runtime_versions = set()
    estimators = set()

    for msg in messages:
        match = _INCONSISTENT_VERSION_RE.search(msg)
        if not match:
            continue
        estimators.add(match.group("estimator"))
        saved_versions.add(match.group("saved").rstrip("."))
        runtime_versions.add(match.group("runtime").rstrip("."))

    saved = ", ".join(sorted(saved_versions)) if saved_versions else None
    runtime = ", ".join(sorted(runtime_versions)) if runtime_versions else None
    return saved, runtime, sorted(estimators)


def _joblib_load_with_sklearn_version_handling(
    path: Path,
    *,
    strict_sklearn_version: bool,
    suppress_sklearn_version_warning: bool,
) -> object:
    try:
        from sklearn.exceptions import InconsistentVersionWarning
    except Exception:
        return joblib.load(path)

    captured: List[str] = []
    original_showwarning = warnings.showwarning

    def custom_showwarning(message, category, filename, lineno, file=None, line=None):  # type: ignore[no-untyped-def]
        if issubclass(category, InconsistentVersionWarning):
            captured.append(str(message))
            return
        return original_showwarning(message, category, filename, lineno, file=file, line=line)

    try:
        with warnings.catch_warnings():
            warnings.showwarning = custom_showwarning
            warnings.simplefilter("default")
            warnings.simplefilter("always", InconsistentVersionWarning)
            obj = joblib.load(path)
    finally:
        warnings.showwarning = original_showwarning

    if captured:
        saved, runtime, estimators = _summarize_inconsistent_version(captured)
        details = ""
        if saved or runtime:
            details = f" (saved={saved or 'unknown'}, runtime={runtime or 'unknown'})"
        est_text = f" [{', '.join(estimators)}]" if estimators else ""
        msg = (
            "scikit-learn version mismatch detected while loading model"
            f"{details}{est_text}."
        )

        if strict_sklearn_version:
            raise RuntimeError(msg)
        if not suppress_sklearn_version_warning:
            print(f"WARNING: {msg}")

    return obj


def _resave_model_bundle(bundle: object, *, in_path: Path, out_path: Optional[Path]) -> Path:
    if out_path is None:
        backup = Path(str(in_path) + ".bak")
        if not backup.exists():
            shutil.copy2(in_path, backup)
        out_path = in_path

    joblib.dump(bundle, out_path)
    return out_path


def load_model_bundle(
    path: Path,
    g_mode_override: Optional[str] = None,
    *,
    strict_sklearn_version: bool = False,
    suppress_sklearn_version_warning: bool = False,
) -> Tuple[object, object, str, str, float, bool, Optional[object]]:
    obj = _joblib_load_with_sklearn_version_handling(
        path,
        strict_sklearn_version=strict_sklearn_version,
        suppress_sklearn_version_warning=suppress_sklearn_version_warning,
    )

    target_transform = "identity"
    target_eps = TARGET_EPS
    use_bg = False
    calibrator: Optional[object] = None

    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
        g_mode = obj.get("g_mode", DEFAULT_G_MODE)
        target_transform = obj.get("target_transform", target_transform)
        target_eps = obj.get("target_eps", TARGET_EPS)
        use_bg = bool(obj.get("use_bg", use_bg))
        calibrator = obj.get("isotonic", None)
    else:
        model = obj
        g_mode = DEFAULT_G_MODE

    if g_mode_override:
        g_mode = g_mode_override
    if g_mode not in ("g1", "avg"):
        g_mode = DEFAULT_G_MODE

    return obj, model, g_mode, str(target_transform), float(target_eps), use_bg, calibrator


def inverse_target(pred: np.ndarray, transform: Optional[str], target_eps: float) -> np.ndarray:
    if transform == "log":
        pred = np.exp(pred) - target_eps
    return np.maximum(np.asarray(pred, dtype=float), 0.0)


def apply_calibrator(pred: np.ndarray, calibrator: Optional[object]) -> np.ndarray:
    if calibrator is None:
        return np.maximum(np.asarray(pred, dtype=float), 0.0)
    calibrated = calibrator.predict(np.asarray(pred, dtype=float))
    return np.maximum(np.asarray(calibrated, dtype=float), 0.0)


def _select_model_features(feature_df: pd.DataFrame, feature_names: Optional[Sequence[str]]) -> pd.DataFrame:
    if not feature_names:
        return feature_df

    missing = [name for name in feature_names if name not in feature_df.columns]
    if missing:
        raise ValueError(f"Input CSV cannot build required model features: {missing}")
    return feature_df.loc[:, list(feature_names)]


def predict_batch(
    model: object,
    roi_csv: Path,
    out_csv: Path,
    g_mode: str,
    *,
    target_transform: Optional[str],
    target_eps: float,
    use_bg: bool,
    calibrator: Optional[object],
    expected_feature_names: Optional[Sequence[str]],
) -> pd.DataFrame:
    if not roi_csv.exists():
        raise FileNotFoundError(f"ROI CSV not found: {roi_csv}")

    df = pd.read_csv(roi_csv)
    if df.empty:
        raise ValueError("ROI CSV is empty.")

    dataset_use_bg = use_bg
    if use_bg and not has_background_columns(df, g_mode):
        print("WARNING: model expects bg_* columns, input CSV has none, fallback to non-bg features.")
        dataset_use_bg = False

    feature_df = build_feature_frame(df, g_mode=g_mode, use_bg=dataset_use_bg, clip_min=CLIP_MIN)
    model_features = _select_model_features(feature_df, expected_feature_names)
    pred_target = model.predict(model_features.to_numpy(dtype=float))
    pred_raw = inverse_target(np.asarray(pred_target, dtype=float), target_transform, target_eps)
    pred = apply_calibrator(pred_raw, calibrator)

    out_df = pd.DataFrame({"filename": df["filename"], "conc_pred": pred})
    conc_true = df["filename"].apply(extract_concentration_from_name)
    if conc_true.notna().any():
        out_df["conc_true"] = conc_true
        out_df["residual"] = pred - conc_true

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False, float_format=CSV_FLOAT_FORMAT)
    print(f"Saved predictions -> {out_csv}")
    return out_df


def draw_prediction_plot(out_df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    conc_pred = out_df["conc_pred"].to_numpy(dtype=float)
    if "conc_true" in out_df.columns:
        conc_true = out_df["conc_true"].to_numpy(dtype=float)
        valid = np.isfinite(conc_true) & np.isfinite(conc_pred)
        if np.any(valid):
            x = conc_true[valid]
            y = conc_pred[valid]
            axes[0].scatter(x, y, s=24, alpha=0.8, color="#1f77b4")
            low = float(min(np.min(x), np.min(y)))
            high = float(max(np.max(x), np.max(y)))
            axes[0].plot([low, high], [low, high], "k--", linewidth=1.0)
            axes[0].set_xlabel("True concentration (uM)")
            axes[0].set_ylabel("Pred concentration (uM)")
            axes[0].set_title("Prediction vs Truth")
            axes[0].grid(alpha=0.25)

            residual = y - x
            axes[1].hist(residual, bins=20, color="#ff7f0e", alpha=0.8)
            axes[1].axvline(0.0, color="k", linestyle="--", linewidth=1.0)
            axes[1].set_xlabel("Residual (pred - true)")
            axes[1].set_ylabel("Count")
            axes[1].set_title("Residual Distribution")
            axes[1].grid(alpha=0.25)
        else:
            axes[0].text(0.5, 0.5, "No finite labels", ha="center", va="center")
            axes[1].text(0.5, 0.5, "No finite labels", ha="center", va="center")
    else:
        axes[0].hist(conc_pred[np.isfinite(conc_pred)], bins=20, color="#1f77b4", alpha=0.8)
        axes[0].set_xlabel("Pred concentration (uM)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Prediction Distribution")
        axes[0].grid(alpha=0.25)

        axes[1].plot(np.arange(len(conc_pred)), conc_pred, ".", markersize=4, color="#d62728")
        axes[1].set_xlabel("Sample index")
        axes[1].set_ylabel("Pred concentration (uM)")
        axes[1].set_title("Prediction by Sample")
        axes[1].grid(alpha=0.25)

    format_figure_numbers(fig)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure      -> {out_path}")


def resolve_output_path(path: Optional[Path], default_path: Path, output_dir: Path) -> Path:
    chosen = (path or default_path).expanduser()
    if not chosen.is_absolute():
        chosen = output_dir / chosen
    return chosen.resolve()


def parse_args(default_cfg: PredictConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict concentration using trained model + config defaults.")
    parser.add_argument("--model", type=Path, default=default_cfg.model_path, help="Model bundle (.pkl) path.")
    parser.add_argument("--roi-csv", type=Path, default=default_cfg.default_roi_csv, help="Input ROI CSV path.")
    parser.add_argument("--out", type=Path, help="Output CSV path. Default: auto by config output suffix.")
    parser.add_argument("--plot-out", type=Path, help="Output PNG path. Default: auto by config output suffix.")
    parser.add_argument("--g-mode", choices=["g1", "avg"], help="Override model g-mode.")
    parser.add_argument(
        "--strict-sklearn-version",
        action="store_true",
        help="Fail if sklearn runtime version mismatches model bundle version.",
    )
    parser.add_argument(
        "--suppress-sklearn-version-warning",
        action="store_true",
        help="Suppress version mismatch warning while loading model bundle.",
    )
    parser.add_argument(
        "--resave-model",
        action="store_true",
        help="Resave model bundle with current sklearn runtime to reduce version warnings next time.",
    )
    parser.add_argument("--resave-model-out", type=Path, help="Output path for --resave-model.")
    return parser.parse_args()


def main() -> None:
    cfg = PredictConfig()
    args = parse_args(cfg)

    model_path = args.model.expanduser()
    if not model_path.is_absolute():
        model_path = cfg.output_dir / model_path
    model_path = model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    bundle, model, g_mode, target_transform, target_eps, use_bg, calibrator = load_model_bundle(
        model_path,
        args.g_mode,
        strict_sklearn_version=bool(args.strict_sklearn_version),
        suppress_sklearn_version_warning=bool(args.suppress_sklearn_version_warning),
    )

    if args.resave_model:
        resave_out = None
        if args.resave_model_out is not None:
            resave_out = resolve_output_path(args.resave_model_out, model_path, cfg.output_dir)
        resaved_path = _resave_model_bundle(bundle, in_path=model_path, out_path=resave_out)
        print(f"Resaved model bundle -> {resaved_path}")

    print(
        f"Loaded model: {model_path} "
        f"(g_mode={g_mode}, target={target_transform}, use_bg={use_bg}, calibrator={'yes' if calibrator else 'no'})"
    )

    expected_feature_names: Optional[Sequence[str]] = None
    if isinstance(bundle, dict):
        names = bundle.get("feature_names")
        if isinstance(names, (tuple, list)):
            expected_feature_names = [str(name) for name in names]

    out_csv = resolve_output_path(args.out, cfg.output_path_for(args.roi_csv), cfg.output_dir)
    out_df = predict_batch(
        model,
        args.roi_csv,
        out_csv,
        g_mode,
        target_transform=target_transform,
        target_eps=target_eps,
        use_bg=use_bg,
        calibrator=calibrator,
        expected_feature_names=expected_feature_names,
    )
    plot_out = resolve_output_path(args.plot_out, cfg.plot_path_for(args.roi_csv), cfg.output_dir)
    draw_prediction_plot(out_df, plot_out)


if __name__ == "__main__":
    main()
