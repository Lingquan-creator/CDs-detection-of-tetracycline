# concentration_GS

本目录使用小型神经网络做摊销后验浓度预测。它输出的是浓度的概率分布，而不是单个点估计：

```text
q(log(C + eps) | x) = Normal(mu_phi(x), sigma_phi(x)^2)
```

转换回浓度空间后得到移位对数正态后验，并输出后验均值、中位数、众数、标准差、置信区间和离散浓度 bin 概率。默认点估计使用 `posterior_median`。

## 主要文件

- `config.py`: 默认输入输出路径和训练超参数。
- `physics_features.py`: ROI/BG 物理特征构造。
- `train.py`: 训练摊销后验模型，执行 Leave-One-Out 评估。
- `predict.py`: 加载 `.pt` 模型，对新的 ROI CSV 输出后验预测。
- `visualize_results.py`: 基于训练输出生成额外可视化。
- `algorithm.md`: 更完整的代码和算法说明。

## 默认配置

```text
输入: dataset/food/roi_detailed_statistics.csv
输出: concentration_GS/amortized_output/
g_mode: avg
use_bg: true
feature_set: compact
hidden_dim: 8
dropout: 0.05
epochs: 2500
patience: 250
confidence: 0.90
```

## 输入

训练 CSV 需要包含：

```text
filename
roi_R_mean, roi_G1_mean, roi_B_mean
bg_R_mean, bg_G1_mean, bg_B_mean
```

默认 `g_mode=avg` 且 `use_bg=true`，因此通常还需要：

```text
roi_G2_mean, bg_G2_mean
```

可选列：

- `roi_G1_var`, `roi_G2_var`: 用于 `Var_G`。
- `exposure_time`: 用于曝光相关特征。
- `roi_*_gauss_mu`, `roi_*_gauss_shape` 或 `roi_*_gamma_shape`, `roi_*_gamma_scale`: `feature_set=full` 时使用。

`filename` 会用于解析真实浓度；训练时无法解析浓度的样品会被丢弃。

## 特征集合

默认 `feature_set=compact`，使用 6 个背景扣除特征：

```text
ln_R_sig, ln_G_sig, ln_B_sig, Ratio_RG, Delta_RG, Delta_GB
```

其他选项：

- `compact_exposure`: 加入曝光归一化、`Ratio_BG`、`Var_G`、曝光时间等基础特征。
- `full`: 使用全部可构造特征，包括可用 ROI 分布参数。

背景扣除信号为：

```text
R_sig = max(roi_R_mean - bg_R_mean, 0.1)
G_sig = average(max(roi_G1_mean - bg_G1_mean, 0.1),
                max(roi_G2_mean - bg_G2_mean, 0.1))
B_sig = max(roi_B_mean - bg_B_mean, 0.1)
```

## 常用命令

训练和 LOO 评估：

```bash
python concentration_GS/train.py
```

指定输入和特征：

```bash
python concentration_GS/train.py \
  --roi-csv dataset/food/roi_detailed_statistics.csv \
  --feature-set compact \
  --use-bg true
```

预测：

```bash
python concentration_GS/predict.py \
  --model concentration_GS/amortized_output/amortized_posterior_model.pt \
  --roi-csv dataset/food/roi_detailed_statistics.csv
```

额外可视化：

```bash
python concentration_GS/visualize_results.py
```

## 输出

默认输出目录：

```text
concentration_GS/amortized_output/
```

训练输出：

- `amortized_posterior_model.pt`: PyTorch 模型 bundle，包含网络权重、标准化参数、特征名、参考浓度、配置和指标。
- `train_posterior_summary.csv`: 训练集后验摘要。
- `loo_posterior_summary.csv`: Leave-One-Out 后验摘要，泛化评估优先看这个文件。
- `train_posterior_probabilities.csv`: 训练集离散浓度 bin 概率。
- `loo_posterior_probabilities.csv`: LOO 离散浓度 bin 概率。
- `posterior_evaluation.png`: 训练/LOO 后验散点图和概率热图。

预测输出：

- `<roi_csv_stem>_pred_concentration_gs.csv`
- `<roi_csv_stem>_pred_concentration_gs_prob.csv`
- `<roi_csv_stem>_pred_concentration_gs.png`

## 关键字段

summary CSV 常用字段：

- `posterior_log_mean`, `posterior_log_std`: log 浓度后验参数。
- `posterior_mean`, `posterior_median`, `posterior_mode`: 浓度空间点摘要。
- `posterior_std`: 浓度空间标准差。
- `ci_90_low`, `ci_90_high`: 默认 90% 区间。
- `map_bin_concentration`, `map_bin_probability`: 概率最大的参考浓度 bin。
- `nll`: 对真实浓度的负对数似然。

probability CSV 常用字段：

- `filename`
- `ref_concentration`
- `bin_left`, `bin_right`
- `bin_probability`
- `cdf_right`

## 依赖

```bash
pip install numpy pandas matplotlib scipy scikit-learn torch
```
