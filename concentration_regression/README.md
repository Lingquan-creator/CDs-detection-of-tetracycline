# concentration_regression

本目录使用 ROI/BG RAW 统计构造物理特征，并用 Ridge 回归做浓度反演。当前工作区的最佳 Ridge 组合已由 `optimization` 寻优后整理到 `rgb_fit_output/best_*`。

## 主要文件

- `config.py`: 默认路径、目标变换、背景校正、特征集合和正则参数。
- `physics_features.py`: 从 ROI/BG 统计构造 `R_sig/G_sig/B_sig`、ln 加和项、ln 比值项等特征。
- `train.py`: 训练 Ridge 模型，输出训练集和 LOO 预测。
- `predict.py`: 加载 `concentration_model_v1.pkl` 对新的 ROI CSV 预测。
- `algorithm.md`: 算法细节说明。

## 当前默认配置

`config.py` 当前默认值：

```text
roi_csv: dataset/train/roi_results/roi_detailed_statistics.csv
output: concentration_regression/rgb_fit_output/
predict_model: concentration_regression/rgb_fit_output/best_test/concentration_model_v1.pkl
predict_roi_csv: dataset/test/roi_results/roi_detailed_statistics.csv
target_transform: identity
use_bg: true
bg_correction: subtract
feature_set: compact
feature_list: ln_G_sig, ln_RB_over_GB
alphas: (0.03,)
weight_eps: 5.0
weight_beta: 0.0
clip_min: 0.1
g_mode: avg
```

这些默认值已同步为 `optimization/test/ridge/OUTPUT/best_summary.csv` 中的当前 train->test 最优组合。LOO 最优组合仍记录在 `best_optimized_combinations.csv` 中。

## 输入

训练 CSV 至少需要：

```text
filename
roi_R_mean, roi_G1_mean, roi_B_mean
bg_R_mean, bg_G1_mean, bg_B_mean
```

当前主流程使用：

```text
dataset/train/roi_results/roi_detailed_statistics.csv
dataset/test/roi_results/roi_detailed_statistics.csv
```

默认 `g_mode=avg`，通常也需要 `roi_G2_mean` 和 `bg_G2_mean`。`filename` 用于解析真实浓度；无法解析浓度的样品会在训练时被丢弃。

## 特征

背景扣除信号：

```text
R_sig = max(roi_R_mean - bg_R_mean, 0.1)
G_sig = average(max(roi_G1_mean - bg_G1_mean, 0.1),
                max(roi_G2_mean - bg_G2_mean, 0.1))
B_sig = max(roi_B_mean - bg_B_mean, 0.1)
```

`physics_features.py` 会进一步构造 ln 单通道、ln 加和和 ln 比值特征。当前优化流程只允许使用 ln 形式输入向量，例如：

```text
ln_R_sig
ln_G_sig
ln_B_sig
ln_RGB_sum
ln_RG_sum
ln_RB_sum
ln_GB_sum
ln_R_over_G
ln_B_over_G
ln_RG_over_GB
ln_RB_over_GB
```

## 当前最佳模型

当前整理后的最佳 Ridge 模型：

```text
concentration_regression/rgb_fit_output/best_train/concentration_model_v1.pkl
concentration_regression/rgb_fit_output/best_test/concentration_model_v1.pkl
concentration_regression/rgb_fit_output/best_loo/concentration_model_v1.pkl
```

最佳组合索引：

```text
concentration_regression/rgb_fit_output/best_optimized_combinations.csv
```

其中 test 模式当前 Ridge 最优组合为：

```text
feature_names: ln_G_sig;ln_RB_over_GB
alpha: 0.03
target_transform: identity
weight_beta: 0.0
weight_eps: 5.0
```

LOO 模式当前 Ridge 最优组合为：

```text
feature_names: ln_RGB_sum;ln_R_over_G;ln_B_over_G
alpha: 1.0
target_transform: identity
weight_beta: 0.0
weight_eps: 50.0
```

## 常用命令

模块默认训练：

```bash
python concentration_regression/train.py
```

指定 ROI CSV 和特征：

```bash
python concentration_regression/train.py \
  --roi-csv dataset/train/roi_results/roi_detailed_statistics.csv \
  --feature-list "ln_G_sig ln_RB_over_GB" \
  --target-transform identity \
  --alpha 0.03 \
  --weight-eps 5 \
  --weight-beta 0
```

预测：

```bash
python concentration_regression/predict.py \
  --model concentration_regression/rgb_fit_output/best_test/concentration_model_v1.pkl \
  --roi-csv dataset/test/roi_results/roi_detailed_statistics.csv
```

最终 train/test/LOO 的统一训练和预测由根目录下的脚本执行：

```bash
python OUTPUT/run_best_models_and_recovery.py
```

## 输出

默认输出目录：

```text
concentration_regression/rgb_fit_output/
```

训练输出包括：

- `concentration_model_v1.pkl`
- `ridge_predictions.csv`
- `ridge_predictions_loo.csv`
- `ridge_coefficients.csv`
- `concentration_model_v1_evaluation.png`

最终预测和回收率汇总位于 `OUTPUT/train`、`OUTPUT/test` 和 `OUTPUT/LOO`。
