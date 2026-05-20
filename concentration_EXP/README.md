# concentration_EXP

本目录使用响应函数拟合 ROI/BG 统计中的光强特征，并通过反演得到浓度预测。它是 Ridge 主线之外的可解释基线模型。

支持的响应函数：

```text
I(C) = a * exp(b*C)
I(C) = a * exp(b*C) + c
I(C) = m*C + k
I(C) = q*C^2 + m*C + k
```

当前最佳 EXP 组合已由 `optimization` 搜索后整理到 `exp_fit_output/best_*`。

## 主要文件

- `config.py`: 默认路径、输出文件名和拟合默认参数。
- `physics_features.py`: 浓度/曝光解析和基础特征工具。
- `train.py`: 拟合响应函数并保存 `exp_fit_params.csv`。
- `predict.py`: 读取 `exp_fit_params.csv` 对 ROI CSV 反演浓度。
- `algorithm.md`: 算法细节说明。

## 输入

## 当前默认配置

`config.py` 当前默认值已同步为 `optimization/test/exp/OUTPUT/best_summary.csv` 中的 train->test 最优组合：

```text
roi_csv: dataset/train/roi_results/roi_detailed_statistics.csv
output: concentration_EXP/exp_fit_output/
feature_list: ln_R_sig, ln_G_sig, ln_B_sig
channel: ln_G_sig
weight_eps: 0.0
```

`train.py` 默认会使用上述 `feature_list` 和 `channel`，并对支持的响应函数全部拟合；当前最优响应形式为 `exponential_offset`。预测阶段如果要直接使用已整理的 test 最优参数，可指定 `--params concentration_EXP/exp_fit_output/best_test/exp_fit_params.csv`。

当前主流程使用：

```text
dataset/train/roi_results/roi_detailed_statistics.csv
dataset/test/roi_results/roi_detailed_statistics.csv
```

输入 CSV 至少需要：

```text
filename
roi_R_mean, roi_G1_mean, roi_B_mean
bg_R_mean, bg_G1_mean, bg_B_mean
```

若存在 `roi_G2_mean/bg_G2_mean`，会构造平均绿色通道 `G_sig`。当前优化流程也会使用 `concentration_regression.physics_features` 中的 ln 派生特征。

## 当前最佳模型

当前整理后的 EXP 参数：

```text
concentration_EXP/exp_fit_output/best_train/exp_fit_params.csv
concentration_EXP/exp_fit_output/best_test/exp_fit_params.csv
concentration_EXP/exp_fit_output/best_loo/exp_fit_params.csv
```

最佳组合索引：

```text
concentration_EXP/exp_fit_output/best_optimized_combinations.csv
```

test 模式当前 EXP 最优组合：

```text
feature_names: ln_R_sig;ln_G_sig;ln_B_sig
response_model: exponential_offset
selected_channel: ln_G_sig
weight_eps: 0.0
```

LOO 模式当前 EXP 最优组合：

```text
feature_names: ln_R_sig;ln_G_sig;ln_B_sig
response_model: exponential_offset
selected_channel: ln_B_sig
weight_eps: 1.0
```

## 常用命令

模块默认训练：

```bash
python concentration_EXP/train.py
```

指定输入、特征和最终反演通道：

```bash
python concentration_EXP/train.py \
  --roi-csv dataset/train/roi_results/roi_detailed_statistics.csv \
  --feature-list "ln_R_sig ln_G_sig ln_B_sig" \
  --channel ln_G_sig \
  --weight-eps 0
```

`train.py` 会对支持的响应函数全部拟合；最终使用哪个响应函数由训练结果中的最佳 `fit_id` 或优化整理脚本决定。

预测：

```bash
python concentration_EXP/predict.py \
  --params concentration_EXP/exp_fit_output/best_test/exp_fit_params.csv \
  --roi-csv dataset/test/roi_results/roi_detailed_statistics.csv
```

最终 train/test/LOO 的统一训练和预测由：

```bash
python OUTPUT/run_best_models_and_recovery.py
```

## 输出

默认输出目录：

```text
concentration_EXP/exp_fit_output/
```

主要文件：

- `exp_fit_params.csv`: 每个响应模型的参数、函数形式和反演指标。
- `exp_fit_predictions.csv`: 训练集光强拟合结果。
- `<roi_csv_stem>_pred_concentration_exp.csv`: 预测阶段浓度反演结果。
- `exp_fit_evaluation.png`: 训练阶段评估图。
- `<roi_csv_stem>_pred_concentration_exp.png`: 预测阶段评估图。

当前精简后的最终预测和回收率汇总位于 `OUTPUT/train/predict_exp`、`OUTPUT/test/predict_exp` 和 `OUTPUT/LOO/predict_exp`。
