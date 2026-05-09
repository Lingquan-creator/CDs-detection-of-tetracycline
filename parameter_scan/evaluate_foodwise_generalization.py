from __future__ import annotations

from pathlib import Path

import pandas as pd

from run_regression_parameter_scan import (
    HyperParams,
    build_all_features,
    evaluate_raw_prediction,
    group_from_filename,
    load_labeled_roi,
    metrics,
)


ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "dataset" / "mate20x" / "roi_detailed_statistics.csv"
TEST_CSV = ROOT / "dataset" / "RAW" / "roi_results" / "roi_detailed_statistics.csv"
FOODWISE_BEST = ROOT / "parameter_scan" / "OUTPUT" / "foodwise_iterative" / "foodwise_best_summary.csv"
OUT_DIR = ROOT / "parameter_scan" / "OUTPUT" / "foodwise_generalization"
CSV_FLOAT_FORMAT = "%.6g"


def metric_row(source_food: str, predictions: pd.DataFrame, group: str) -> dict[str, object]:
    part = predictions if group == "all" else predictions[predictions["sample_group"] == group]
    row: dict[str, object] = {
        "source_food": source_food,
        "eval_group": group,
        "n": len(part),
    }
    row.update(metrics(part["conc_true"].to_numpy(float), part["conc_pred"].to_numpy(float)))
    return row


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df, y_train = load_labeled_roi(TRAIN_CSV)
    test_df, y_test = load_labeled_roi(TEST_CSV)
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["sample_group"] = train_df["filename"].apply(group_from_filename)
    test_df["sample_group"] = test_df["filename"].apply(group_from_filename)

    train_features = build_all_features(train_df)
    test_features = build_all_features(test_df)
    best_df = pd.read_csv(FOODWISE_BEST)

    all_metric_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for _, config in best_df.sort_values("food").iterrows():
        source_food = str(config["food"])
        feature_names = tuple(str(config["feature_names"]).split(";"))
        params = HyperParams(
            alpha=float(config["alpha"]),
            beta=float(config["beta"]),
            weight_eps=float(config["weight_eps"]),
        )

        _, predictions = evaluate_raw_prediction(
            train_df,
            y_train,
            test_df,
            y_test,
            train_features,
            test_features,
            feature_names,
            params,
            food=source_food,
            step="generalization",
            scan="foodwise_best",
            variant_id=f"{source_food}_best_on_all_foods",
        )

        config_dir = OUT_DIR / source_food
        config_dir.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(config_dir / "all_food_predictions.csv", index=False, float_format=CSV_FLOAT_FORMAT)

        groups = ["all", *sorted(test_df["sample_group"].unique())]
        config_metrics = []
        for group in groups:
            row = metric_row(source_food, predictions, group)
            row.update(
                {
                    "alpha": params.alpha,
                    "beta": params.beta,
                    "weight_eps": params.weight_eps,
                    "n_features": len(feature_names),
                    "feature_names": ";".join(feature_names),
                }
            )
            config_metrics.append(row)
            all_metric_rows.append(row)
        pd.DataFrame(config_metrics).to_csv(config_dir / "generalization_metrics.csv", index=False, float_format=CSV_FLOAT_FORMAT)

        metrics_df = pd.DataFrame(config_metrics)
        all_row = metrics_df[metrics_df["eval_group"] == "all"].iloc[0]
        food_rows = metrics_df[metrics_df["eval_group"] != "all"].copy()
        summary_rows.append(
            {
                "source_food": source_food,
                "alpha": params.alpha,
                "beta": params.beta,
                "weight_eps": params.weight_eps,
                "n_features": len(feature_names),
                "feature_names": ";".join(feature_names),
                "all_r2": all_row["r2"],
                "all_rmse": all_row["rmse"],
                "all_mae": all_row["mae"],
                "mean_food_r2": food_rows["r2"].mean(),
                "min_food_r2": food_rows["r2"].min(),
                "mean_food_rmse": food_rows["rmse"].mean(),
                "max_food_rmse": food_rows["rmse"].max(),
                "worst_food_by_r2": food_rows.sort_values(["r2", "rmse"], ascending=[True, False]).iloc[0]["eval_group"],
            }
        )

    metrics_out = pd.DataFrame(all_metric_rows)
    summary_out = pd.DataFrame(summary_rows)
    robust_rank = summary_out.sort_values(["min_food_r2", "all_r2", "all_rmse"], ascending=[False, False, True]).reset_index(drop=True)
    all_rank = summary_out.sort_values(["all_r2", "all_rmse"], ascending=[False, True]).reset_index(drop=True)

    metrics_out.to_csv(OUT_DIR / "foodwise_generalization_metrics.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    robust_rank.to_csv(OUT_DIR / "foodwise_generalization_rank_robust.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    all_rank.to_csv(OUT_DIR / "foodwise_generalization_rank_all_r2.csv", index=False, float_format=CSV_FLOAT_FORMAT)

    best = robust_rank.iloc[0]
    with (OUT_DIR / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("Food-wise best combinations evaluated on all RAW foods\n")
        f.write(f"Train CSV: {TRAIN_CSV}\n")
        f.write(f"Test CSV: {TEST_CSV}\n")
        f.write("Robust ranking: max min_food_r2, then max all_r2, then min all_rmse.\n\n")
        f.write("[robust rank]\n")
        f.write(
            robust_rank[
                [
                    "source_food",
                    "all_r2",
                    "all_rmse",
                    "mean_food_r2",
                    "min_food_r2",
                    "worst_food_by_r2",
                    "alpha",
                    "beta",
                    "weight_eps",
                    "feature_names",
                ]
            ].to_string(index=False)
        )
        f.write("\n\n[best robust generalization]\n")
        f.write(best.to_string())
        f.write("\n")

    print(f"Saved generalization outputs -> {OUT_DIR}")
    print(
        robust_rank[
            [
                "source_food",
                "all_r2",
                "all_rmse",
                "mean_food_r2",
                "min_food_r2",
                "worst_food_by_r2",
                "feature_names",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
