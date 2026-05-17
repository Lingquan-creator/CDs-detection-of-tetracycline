# OUTPUT

本目录保存当前最佳模型的统一预测、指标、回收率和 RSD 输出。它不是单个模型模块的训练目录，而是把 `optimization` 选出的 Ridge、EXP、GS 最优组合整理后统一复评的结果目录。

## 主入口

```bash
python OUTPUT/run_best_models_and_recovery.py
```

该脚本会：

1. 读取 `optimization/test` 和 `optimization/LOO` 中各模型族的最佳配置。
2. 用 `dataset/train` 重新训练或整理最佳模型。
3. 对 train、test、LOO 三种模式输出预测。
4. 计算每种食物、每个浓度的平均回收率和 RSD。
5. 把最佳模型同步到 `concentration_regression`、`concentration_EXP` 和 `concentration_GS`。

## 输入

```text
dataset/train/roi_results/roi_detailed_statistics.csv
dataset/test/roi_results/roi_detailed_statistics.csv
optimization/test/{ridge,exp,gs}/OUTPUT/best_summary.csv
optimization/LOO/{ridge,exp,gs}/OUTPUT/best_summary.csv
```

## 目录结构

```text
OUTPUT/train/
OUTPUT/test/
OUTPUT/LOO/
```

每个模式下都有：

```text
predict/       Ridge 预测和指标
predict_exp/   EXP 预测和指标
predict_gs/    GS 后验预测和指标
results/       回收率、RSD 和热图
```

历史旧结果保存在：

```text
OUTPUT/.old/
```

## 预测输出

Ridge:

```text
predict/rgb_predictions.csv
predict/prediction_metrics.csv
```

EXP:

```text
predict_exp/exp_predictions.csv
predict_exp/prediction_metrics.csv
```

GS:

```text
predict_gs/gs_posterior_summary.csv
predict_gs/prediction_metrics.csv
```

`train`、`test`、`LOO` 目录中还会保留对应模式命名的指标文件，例如 `test_prediction_metrics.csv` 或 `loo_prediction_metrics.csv`。

## 回收率和 RSD

每个 `results/` 下的核心文件：

```text
model_recovery_samples.csv
model_recovery_summary.csv
model_recovery_heatmap_table.csv
figures/model_recovery_heatmap_rsd.png
```

当前用户需要的精简结果主要看：

- `model_recovery_summary.csv`: 每个模型、每种食物、每个浓度的平均回收率和 RSD。
- `figures/model_recovery_heatmap_rsd.png`: 按原表格风格输出的回收率/RSD 热图。

## 与模型目录的关系

脚本运行后会同步最佳模型到：

```text
concentration_regression/rgb_fit_output/best_train|best_test|best_loo/
concentration_EXP/exp_fit_output/best_train|best_test|best_loo/
concentration_GS/amortized_output/best_train|best_test|best_loo/
```

最佳组合索引位于各模型目录的 `best_optimized_combinations.csv`。
