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
PIC_RGB_DIR = PROJECT_ROOT / "pic_RGB"

if str(TRY_DIR) not in sys.path:
    sys.path.insert(0, str(TRY_DIR))
if str(PIC_RGB_DIR) not in sys.path:
    sys.path.insert(0, str(PIC_RGB_DIR))

from roi_config import SAVE_VISUALIZATION, get_settings  # noqa: E402
from scan_feature_curve_matching import ConfigurableStatsROIProcessor  # noqa: E402


CSV_FLOAT_FORMAT = "%.6g"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply best try_pic_RGB scan candidate and rerun EXP fits.")
    parser.add_argument("--eval-csv", type=Path, default=TRY_DIR / "feature_curve_matching_eval.csv")
    parser.add_argument("--summary-out", type=Path, default=TRY_DIR / "final_summary.txt")
    parser.add_argument("--food-dir", type=Path, default=PROJECT_ROOT / "dataset" / "food_2")
    parser.add_argument("--food-output-dir", type=Path, default=PROJECT_ROOT / "dataset" / "food_2" / "roi_results")
    parser.add_argument("--food-func-dir", type=Path, default=PROJECT_ROOT / "dataset" / "food_2" / "func")
    parser.add_argument("--mate-dir", type=Path, default=PROJECT_ROOT / "dataset" / "mate20x")
    parser.add_argument("--mate-output-dir", type=Path, default=PROJECT_ROOT / "dataset" / "mate20x" / "roi_results")
    parser.add_argument("--mate-func-dir", type=Path, default=PROJECT_ROOT / "dataset" / "mate20x" / "func")
    return parser.parse_args()


def select_best(eval_csv: Path) -> tuple[pd.Series, pd.Series | None]:
    if not eval_csv.exists():
        raise FileNotFoundError(f"Evaluation CSV not found: {eval_csv}")
    df = pd.read_csv(eval_csv)
    required = {
        "total_curve_score",
        "measurement_roi_width_fraction",
        "measurement_roi_height_fraction",
        "roi_statistic_method",
        "trim_fraction",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Evaluation CSV missing required columns: {missing}")
    ranked = df.sort_values(
        ["total_curve_score", "kept_feature_count", "measured_common_feature_score"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    best = ranked.iloc[0]
    baseline_mask = (
        np.isclose(df["measurement_roi_width_fraction"].astype(float), 0.4)
        & np.isclose(df["measurement_roi_height_fraction"].astype(float), 0.4)
        & np.isclose(df["trim_fraction"].astype(float), 0.08)
        & (df["roi_statistic_method"].astype(str) == "trimmed_mean")
    )
    baseline = df[baseline_mask].iloc[0] if baseline_mask.any() else None
    return best, baseline


def rerun_pic_rgb_dataset(
    *,
    data_dir: Path,
    output_dir: Path,
    width_fraction: float,
    height_fraction: float,
    trim_fraction: float,
    statistic_method: str,
) -> int:
    settings = get_settings(
        {
            "measurement_roi_width_fraction": width_fraction,
            "measurement_roi_height_fraction": height_fraction,
            "trim_fraction": trim_fraction,
        }
    )
    processor = ConfigurableStatsROIProcessor(
        settings=settings,
        statistic_method=statistic_method,
        reuse_last_selection=True,
        selection_cache_path=output_dir / "roi_selection_cache.json",
    )
    results = processor.batch_process(
        image_dir=data_dir,
        output_dir=output_dir,
        save_visualization=SAVE_VISUALIZATION,
        interactive=False,
    )
    return len(results)


def run_exp_fit(*, roi_csv: Path, func_dir: Path) -> None:
    func_dir.mkdir(parents=True, exist_ok=True)
    log_path = func_dir / "exp_train.log"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "concentration_EXP" / "train.py"),
        "--roi-csv",
        str(roi_csv),
        "--params-out",
        str(func_dir / "exp_fit_params.csv"),
        "--pred-out",
        str(func_dir / "exp_fit_predictions.csv"),
        "--conc-out",
        str(func_dir / "exp_concentration_predictions.csv"),
        "--plot-out",
        str(func_dir / "exp_fit_evaluation.png"),
    ]
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"concentration_EXP failed for {roi_csv}; see {log_path}")


def summarize_feature_details(best: pd.Series) -> tuple[list[str], list[str]]:
    raw = best.get("feature_fit_details_json", "[]")
    try:
        details: list[dict[str, Any]] = json.loads(str(raw))
    except json.JSONDecodeError:
        return [], []
    kept: list[str] = []
    dropped: list[str] = []
    for item in details:
        feature = str(item.get("feature", ""))
        if item.get("kept"):
            kept.append(
                f"{feature}:{item.get('model')} "
                f"food_r2={float(item.get('food_r2', float('nan'))):.6g} "
                f"mate_r2={float(item.get('mate_r2', float('nan'))):.6g} "
                f"curve_score={float(item.get('curve_score', float('nan'))):.6g}"
            )
        else:
            dropped.append(
                f"{feature}:{item.get('drop_reason', 'dropped')} "
                f"best_model={item.get('model', '')} "
                f"min_r2={float(item.get('min_r2', float('nan'))):.6g}"
            )
    return kept, dropped


def read_best_exp_fit(func_dir: Path) -> str:
    params_path = func_dir / "exp_fit_params.csv"
    if not params_path.exists():
        return f"{params_path} missing"
    df = pd.read_csv(params_path)
    if df.empty:
        return f"{params_path} empty"
    row = df.sort_values(["inv_r2", "inv_rmse"], ascending=[False, True]).iloc[0]
    return (
        f"{row['fit_id']} inv_r2={float(row['inv_r2']):.6g} "
        f"inv_rmse={float(row['inv_rmse']):.6g} r2={float(row['r2']):.6g}"
    )


def write_summary(
    *,
    summary_out: Path,
    best: pd.Series,
    baseline: pd.Series | None,
    food_count: int,
    mate_count: int,
    food_func_dir: Path,
    mate_func_dir: Path,
) -> None:
    kept, dropped = summarize_feature_details(best)
    lines = [
        "try_pic_RGB final summary",
        "",
        "Best ROI statistics candidate:",
        f"  measurement_roi_width_fraction={float(best['measurement_roi_width_fraction']):.6g}",
        f"  measurement_roi_height_fraction={float(best['measurement_roi_height_fraction']):.6g}",
        f"  roi_statistic_method={best['roi_statistic_method']}",
        f"  trim_fraction={float(best['trim_fraction']):.6g}",
        f"  total_curve_score={float(best['total_curve_score']):.6g}",
        f"  measured_common_feature_score={float(best['measured_common_feature_score']):.6g}",
        f"  kept_feature_count={int(best['kept_feature_count'])}",
        f"  dropped_feature_count={int(best['dropped_feature_count'])}",
        "",
    ]
    if baseline is not None:
        lines += [
            "Baseline candidate:",
            "  measurement_roi_width_fraction=0.4",
            "  measurement_roi_height_fraction=0.4",
            "  roi_statistic_method=trimmed_mean",
            "  trim_fraction=0.08",
            f"  total_curve_score={float(baseline['total_curve_score']):.6g}",
            f"  measured_common_feature_score={float(baseline['measured_common_feature_score']):.6g}",
            "",
        ]
    lines += [
        "Rerun results:",
        f"  food_2_roi_rows={food_count}",
        f"  mate20x_roi_rows={mate_count}",
        f"  food_2_best_exp_fit={read_best_exp_fit(food_func_dir)}",
        f"  mate20x_best_exp_fit={read_best_exp_fit(mate_func_dir)}",
        "",
        "Kept features:",
        *(f"  - {item}" for item in kept),
        "",
        "Dropped features:",
        *(f"  - {item}" for item in dropped),
        "",
    ]
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    best, baseline = select_best(args.eval_csv)
    width_fraction = float(best["measurement_roi_width_fraction"])
    height_fraction = float(best["measurement_roi_height_fraction"])
    trim_fraction = float(best["trim_fraction"])
    statistic_method = str(best["roi_statistic_method"])

    print("Applying best candidate to food_2...")
    food_count = rerun_pic_rgb_dataset(
        data_dir=args.food_dir.resolve(),
        output_dir=args.food_output_dir.resolve(),
        width_fraction=width_fraction,
        height_fraction=height_fraction,
        trim_fraction=trim_fraction,
        statistic_method=statistic_method,
    )
    print("Applying best candidate to mate20x...")
    mate_count = rerun_pic_rgb_dataset(
        data_dir=args.mate_dir.resolve(),
        output_dir=args.mate_output_dir.resolve(),
        width_fraction=width_fraction,
        height_fraction=height_fraction,
        trim_fraction=trim_fraction,
        statistic_method=statistic_method,
    )

    print("Rerunning concentration_EXP for food_2...")
    run_exp_fit(roi_csv=args.food_output_dir.resolve() / "roi_detailed_statistics.csv", func_dir=args.food_func_dir.resolve())
    print("Rerunning concentration_EXP for mate20x...")
    run_exp_fit(roi_csv=args.mate_output_dir.resolve() / "roi_detailed_statistics.csv", func_dir=args.mate_func_dir.resolve())

    write_summary(
        summary_out=args.summary_out.resolve(),
        best=best,
        baseline=baseline,
        food_count=food_count,
        mate_count=mate_count,
        food_func_dir=args.food_func_dir.resolve(),
        mate_func_dir=args.mate_func_dir.resolve(),
    )
    print(f"Saved final summary -> {args.summary_out.resolve()}")


if __name__ == "__main__":
    main()
