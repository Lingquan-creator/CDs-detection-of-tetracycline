# concentration_GS

本目录使用小型神经网络做摊销后验浓度预测。模型输出的是浓度后验分布，而不是单个点估计：

```text
q(log(C + eps) | x) = Normal(mu_phi(x), sigma_phi(x)^2)
```

转换回浓度空间后会得到后验均值、中位数、众数、标准差、置信区间和离散浓度 bin 概率。当前点估计默认使用 `posterior_median`。

## 主要文件

- `config.py`: 默认输入输出路径和训练超参数。
- `physics_features.py`: ROI/BG 物理特征构造。
- `train.py`: 训练摊销后验模型并执行 LOO 评估。
- `predict.py`: 加载 `.pt` 模型，对新的 ROI CSV 输出后验预测。
- `visualize_results.py`: 基于训练输出生成额外可视化。
- `algorithm.md`: 算法细节说明。

## 默认配置

模块 CLI 当前默认使用：

```text
输入: dataset/train/roi_results/roi_detailed_statistics.csv
输出: concentration_GS/amortized_output/
predict_model: concentration_GS/amortized_output/best_test/amortized_posterior_model.pt
predict_roi_csv: dataset/test/roi_results/roi_detailed_statistics.csv
g_mode: avg
use_bg: true
feature_set: compact
feature_list: ln_R_sig, ln_G_sig, ln_B_sig, ln_RG_sum, ln_RB_sum, ln_GB_sum
hidden_dim: 8
dropout: 0.05
learning_rate: 0.01
weight_decay: 0.001
weight_eps: 50.0
weight_beta: 1.0
min_std: 0.08
std_reg_lambda: 0.01
epochs: 800
patience: 100
confidence: 0.90
```

这些默认值已同步为 `optimization/test/gs/OUTPUT/best_summary.csv` 中的当前 train->test 最优组合。LOO 最优组合仍记录在 `amortized_output/best_optimized_combinations.csv` 中。

## 输入

当前主流程使用：

```text
dataset/train/roi_results/roi_detailed_statistics.csv
dataset/test/roi_results/roi_detailed_statistics.csv
```

训练 CSV 至少需要：

```text
filename
roi_R_mean, roi_G1_mean, roi_B_mean
bg_R_mean, bg_G1_mean, bg_B_mean
```

默认 `g_mode=avg`，通常还需要 `roi_G2_mean` 和 `bg_G2_mean`。`filename` 会用于解析真实浓度。

## 特征

`train.py` 的 CLI 支持 `full`、`compact`、`compact_exposure` 等特征集合，也支持 `--feature-list` 直接指定自定义输入向量。当前默认 `feature_list` 已设为 test 模式最优 ln 输入向量。

当前可用的 ln 输入向量来自：

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

当前整理后的 GS 模型：

```text
concentration_GS/amortized_output/best_train/amortized_posterior_model.pt
concentration_GS/amortized_output/best_test/amortized_posterior_model.pt
concentration_GS/amortized_output/best_loo/amortized_posterior_model.pt
```

最佳组合索引：

```text
concentration_GS/amortized_output/best_optimized_combinations.csv
```

test 模式当前 GS 最优组合：

```text
feature_names: ln_R_sig;ln_G_sig;ln_B_sig;ln_RG_sum;ln_RB_sum;ln_GB_sum
hidden_dim: 8
dropout: 0.05
learning_rate: 0.01
weight_decay: 0.001
weight_eps: 50.0
weight_beta: 1.0
min_std: 0.08
std_reg_lambda: 0.01
epochs: 800
patience: 100
seed: 42
```

LOO 模式当前 GS 最优组合：

```text
feature_names: ln_R_sig;ln_G_sig;ln_B_sig
hidden_dim: 16
dropout: 0.05
learning_rate: 0.01
weight_decay: 0.001
weight_eps: 50.0
weight_beta: 1.0
min_std: 0.08
std_reg_lambda: 0.01
epochs: 80
patience: 20
seed: 42
```

## 常用命令

模块默认训练：

```bash
python concentration_GS/train.py
```

指定输入：

```bash
python concentration_GS/train.py \
  --roi-csv dataset/train/roi_results/roi_detailed_statistics.csv \
  --feature-set compact \
  --use-bg true
```

预测：

```bash
python concentration_GS/predict.py \
  --model concentration_GS/amortized_output/best_test/amortized_posterior_model.pt \
  --roi-csv dataset/test/roi_results/roi_detailed_statistics.csv
```

最终 train/test/LOO 的统一训练和预测由：

```bash
python OUTPUT/run_best_models_and_recovery.py
```

## 输出

默认输出目录：

```text
concentration_GS/amortized_output/
```

训练输出：

- `amortized_posterior_model.pt`
- `train_posterior_summary.csv`
- `loo_posterior_summary.csv`
- `train_posterior_probabilities.csv`
- `loo_posterior_probabilities.csv`
- `posterior_evaluation.png`

预测输出：

- `<roi_csv_stem>_pred_concentration_gs.csv`
- `<roi_csv_stem>_pred_concentration_gs_prob.csv`
- `<roi_csv_stem>_pred_concentration_gs.png`

当前精简后的最终预测和回收率汇总位于 `OUTPUT/train/predict_gs`、`OUTPUT/test/predict_gs` 和 `OUTPUT/LOO/predict_gs`。
