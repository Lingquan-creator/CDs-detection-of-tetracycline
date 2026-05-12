from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRY_DIR = PROJECT_ROOT / "try_pic_RGB"

if str(TRY_DIR) not in sys.path:
    sys.path.insert(0, str(TRY_DIR))

from apply_best_and_rerun import rerun_pic_rgb_dataset, select_best  # noqa: E402


CSV_FLOAT_FORMAT = "%.6g"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the best pic_RGB ROI statistic candidate, then train mate20x models "
            "with available feature-vector terms and evaluate on food_2."
        )
    )
    parser.add_argument("--eval-csv", type=Path, default=TRY_DIR / "feature_curve_matching_eval.csv")
    parser.add_argument("--food-dir", type=Path, default=PROJECT_ROOT / "dataset" / "food_2")
    parser.add_argument("--food-output-dir", type=Path, default=PROJECT_ROOT / "dataset" / "food_2" / "roi_results")
    parser.add_argument("--mate-dir", type=Path, default=PROJECT_ROOT / "dataset" / "mate20x")
    parser.add_argument("--mate-output-dir", type=Path, default=PROJECT_ROOT / "dataset" / "mate20x" / "roi_results")
    parser.add_argument("--output-dir", type=Path, default=TRY_DIR / "model_outputs" / "available_features")
    parser.add_argument("--feature-list-out", type=Path, default=TRY_DIR / "available_input_features.txt")
    parser.add_argument("--summary-out", type=Path, default=TRY_DIR / "available_feature_model_summary.txt")
    return parser.parse_args()


def extract_kept_features(best: pd.Series) -> list[str]:
    details = json.loads(str(best["feature_fit_details_json"]))
    features: list[str] = []
    seen: set[str] = set()
    for item in details:
        if not item.get("kept"):
            continue
        feature = str(item.get("feature", "")).strip()
        if feature and feature not in seen:
            features.append(feature)
            seen.add(feature)
    if not features:
        raise RuntimeError("No kept features found in best evaluation row.")
    return features


def run_command(cmd: list[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        quoted = " ".join(cmd)
        raise RuntimeError(f"Command failed ({completed.returncode}): {quoted}\nSee {log_path}")


def r2_from_prediction_csv(path: Path) -> tuple[float, float, int]:
    df = pd.read_csv(path)
    required = {"conc_true", "conc_pred"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    y = df["conc_true"].to_numpy(dtype=float)
    pred = df["conc_pred"].to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(pred)
    if not np.any(valid):
        return float("nan"), float("nan"), 0
    y = y[valid]
    pred = pred[valid]
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    rmse = float(np.sqrt(ss_res / len(y)))
    return r2, rmse, int(len(y))


def main() -> None:
    args = parse_args()
    best, _baseline = select_best(args.eval_csv)
    features = extract_kept_features(best)

    args.feature_list_out.parent.mkdir(parents=True, exist_ok=True)
    args.feature_list_out.write_text("\n".join(features) + "\n", encoding="utf-8")

    width_fraction = float(best["measurement_roi_width_fraction"])
    height_fraction = float(best["measurement_roi_height_fraction"])
    trim_fraction = float(best["trim_fraction"])
    statistic_method = str(best["roi_statistic_method"])

    print("Rerunning pic_RGB statistics for food_2...")
    food_count = rerun_pic_rgb_dataset(
        data_dir=args.food_dir.resolve(),
        output_dir=args.food_output_dir.resolve(),
        width_fraction=width_fraction,
        height_fraction=height_fraction,
        trim_fraction=trim_fraction,
        statistic_method=statistic_method,
    )
    print("Rerunning pic_RGB statistics for mate20x...")
    mate_count = rerun_pic_rgb_dataset(
        data_dir=args.mate_dir.resolve(),
        output_dir=args.mate_output_dir.resolve(),
        width_fraction=width_fraction,
        height_fraction=height_fraction,
        trim_fraction=trim_fraction,
        statistic_method=statistic_method,
    )

    out_dir = args.output_dir.resolve()
    reg_dir = out_dir / "concentration_regression"
    exp_dir = out_dir / "concentration_EXP"
    reg_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    mate_roi = args.mate_output_dir.resolve() / "roi_detailed_statistics.csv"
    food_roi = args.food_output_dir.resolve() / "roi_detailed_statistics.csv"
    feature_arg = str(args.feature_list_out.resolve())

    print("Training concentration_regression on mate20x...")
    run_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "concentration_regression" / "train.py"),
            "--roi-csv",
            str(mate_roi),
            "--model-out",
            str(reg_dir / "ridge_model.pkl"),
            "--pred-out",
            str(reg_dir / "ridge_train_predictions.csv"),
            "--loo-out",
            str(reg_dir / "ridge_loo_predictions.csv"),
            "--coef-out",
            str(reg_dir / "ridge_coefficients.csv"),
            "--plot-out",
            str(reg_dir / "ridge_evaluation.png"),
            "--g-mode",
            "avg",
            "--use-bg",
            "true",
            "--bg-correction",
            "subtract",
            "--target-transform",
            "identity",
            "--alpha",
            "100",
            "--weight-eps",
            "50",
            "--feature-list",
            feature_arg,
        ],
        cwd=PROJECT_ROOT,
        log_path=reg_dir / "train.log",
    )

    print("Predicting food_2 with concentration_regression mate20x model...")
    reg_food_pred = reg_dir / "food_2_ridge_predictions.csv"
    run_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "concentration_regression" / "predict.py"),
            "--model",
            str(reg_dir / "ridge_model.pkl"),
            "--roi-csv",
            str(food_roi),
            "--out",
            str(reg_food_pred),
            "--plot-out",
            str(reg_dir / "food_2_ridge_predictions.png"),
        ],
        cwd=PROJECT_ROOT,
        log_path=reg_dir / "predict_food_2.log",
    )

    print("Training concentration_EXP on mate20x feature responses...")
    run_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "concentration_EXP" / "train.py"),
            "--roi-csv",
            str(mate_roi),
            "--params-out",
            str(exp_dir / "exp_fit_params.csv"),
            "--pred-out",
            str(exp_dir / "exp_fit_predictions.csv"),
            "--conc-out",
            str(exp_dir / "exp_train_concentration_predictions.csv"),
            "--plot-out",
            str(exp_dir / "exp_fit_evaluation.png"),
            "--feature-list",
            feature_arg,
        ],
        cwd=PROJECT_ROOT,
        log_path=exp_dir / "train.log",
    )

    print("Predicting food_2 with concentration_EXP mate20x feature-response model...")
    exp_food_pred = exp_dir / "food_2_exp_predictions.csv"
    run_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "concentration_EXP" / "predict.py"),
            "--params",
            str(exp_dir / "exp_fit_params.csv"),
            "--roi-csv",
            str(food_roi),
            "--output",
            str(exp_food_pred),
            "--plot-out",
            str(exp_dir / "food_2_exp_predictions.png"),
        ],
        cwd=PROJECT_ROOT,
        log_path=exp_dir / "predict_food_2.log",
    )

    reg_r2, reg_rmse, reg_n = r2_from_prediction_csv(reg_food_pred)
    exp_r2, exp_rmse, exp_n = r2_from_prediction_csv(exp_food_pred)

    summary_rows = [
        {
            "model": "concentration_regression_ridge",
            "train_dataset": "dataset/mate20x",
            "test_dataset": "dataset/food_2",
            "r2": reg_r2,
            "rmse": reg_rmse,
            "n_test": reg_n,
            "prediction_csv": str(reg_food_pred),
        },
        {
            "model": "concentration_EXP_feature_response",
            "train_dataset": "dataset/mate20x",
            "test_dataset": "dataset/food_2",
            "r2": exp_r2,
            "rmse": exp_rmse,
            "n_test": exp_n,
            "prediction_csv": str(exp_food_pred),
        },
    ]
    summary_csv = out_dir / "food_2_r2_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False, float_format=CSV_FLOAT_FORMAT)

    lines = [
        "available feature model summary",
        "",
        "ROI statistics candidate:",
        f"  measurement_roi_width_fraction={width_fraction:.6g}",
        f"  measurement_roi_height_fraction={height_fraction:.6g}",
        f"  roi_statistic_method={statistic_method}",
        f"  trim_fraction={trim_fraction:.6g}",
        f"  total_curve_score={float(best['total_curve_score']):.6g}",
        "",
        "Datasets:",
        f"  food_2_roi_rows={food_count}",
        f"  mate20x_roi_rows={mate_count}",
        "",
        "Available input features:",
        *(f"  - {feature}" for feature in features),
        "",
        "food_2 prediction from mate20x-trained models:",
        f"  concentration_regression Ridge: R2={reg_r2:.6g}, RMSE={reg_rmse:.6g}, n={reg_n}",
        f"  concentration_EXP feature response: R2={exp_r2:.6g}, RMSE={exp_rmse:.6g}, n={exp_n}",
        "",
        f"Summary CSV: {summary_csv}",
    ]
    args.summary_out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
