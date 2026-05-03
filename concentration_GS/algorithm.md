# concentration_GS 算法说明

本目录实现基于神经网络的摊销后验浓度反演。它输出的不是单个点估计，而是条件后验分布：

```text
q(log(C + eps) | x) = Normal(mu_phi(x), sigma_phi(x)^2)
```

其中 `x` 是从 ROI 统计构造的物理特征，`mu_phi` 和 `sigma_phi` 由小型 MLP 输出。浓度空间中的预测分布是移位对数正态分布。默认点预测使用 `posterior_median`，避免 `posterior_mean` 被大方差放大。

## 主要代码

- `train.py`: 训练摊销后验模型，输出训练集评估和 Leave-One-Out 评估。
- `predict.py`: 加载模型，对新的 ROI 统计 CSV 输出后验分布。
- `visualize_results.py`: 读取训练输出 CSV，生成额外可视化。
- `physics_features.py`: 从 ROI 统计构造物理特征。
- `config.py`: 默认路径、网络超参数和输出文件名。

## 算法输入

算法层面的输入是：

- 样品真实浓度 `C`，从 `filename` 解析。
- ROI 和背景通道统计。
- 可选曝光时间。

实际默认输入文件：

- `dataset/food/roi_detailed_statistics.csv`

当前默认配置：

- `use_bg=true`: 默认要求并使用背景扣除。
- `feature_set=compact`: 输入向量参考 `concentration_regression`，使用跨数据集更稳的 4 个 ROI-BG 特征。
- `hidden_dim=8`
- `max_std_log=2.0`: 保留后验 log 标准差上限。

## 特征构造

背景扣除信号：

```text
R_sig = max(roi_R_mean - bg_R_mean, clip_min)
G_sig = max(avg(roi_G1_mean, roi_G2_mean) - avg(bg_G1_mean, bg_G2_mean), clip_min)
B_sig = max(roi_B_mean - bg_B_mean, clip_min)
```

完整基础物理特征包含 12 维：

```text
ln_R_sig, ln_G_sig, ln_B_sig
ln_R_sig_per_exp, ln_G_sig_per_exp, ln_B_sig_per_exp
Ratio_RG, Delta_RG, Ratio_BG
Var_G
exposure_time, ln_exposure
```

如果 CSV 中存在 ROI 分布参数，`full` 特征还会追加：

```text
exp_roi_*_gauss_mu      = exp(roi_*_gauss_mu / 1000)
log_roi_*_gamma_shape   = log(max(roi_*_gamma_shape, eps))
sqrt_roi_*_gamma_scale  = sqrt(max(roi_*_gamma_scale, 0))
```

当前默认 `compact` 实际训练列是 4 维，不包含 `Var_G`：

```text
ln_R_sig, ln_G_sig, ln_B_sig, Ratio_RG
```

如果使用 `full`，会使用所有基础物理特征和可用的 ROI 分布变换特征；如果使用 `compact_exposure`，则只使用 12 维基础物理特征。若 `use_bg=false`，对应特征名会从 `*_sig` 变为原始 ROI 通道，例如 `ln_R`。

## 处理流程

1. 读取 ROI CSV，并从 `filename` 解析 `conc_true`。
2. 根据 `g_mode` 选择 G1 或 G1/G2 平均值。
3. 根据 `use_bg` 决定是否使用背景扣除。
4. 构造特征矩阵 `X`，再按 `feature_set` 选择列。
5. 标准化特征：

   ```text
   X_std = (X - mean(X)) / std(X)
   ```

6. 对目标做对数变换和标准化：

   ```text
   y_log = log(C + eps)
   y_std = (y_log - mean(y_log)) / std(y_log)
   ```

7. 训练 MLP。输入维度由 `feature_set` 和 CSV 中可用列决定；当前默认 compact 是 4 维。网络结构为：

   ```text
   input_dim -> hidden_dim -> hidden_dim -> mu/std heads
   ```

   网络输出：

   ```text
   mu = mu_head(backbone(X_std))
   std = min_std + softplus(std_head(backbone(X_std)))
   ```

8. 损失函数是加权高斯负对数似然，并额外惩罚过大的预测标准差：

   ```text
   loss = weighted_NLL + std_reg_lambda * mean(std^2)
   ```

   样品权重约为：

   ```text
   weight = 1 / (C + weight_eps)
   ```

9. 将标准化空间的 `mu/std` 转回 `log(C + eps)` 空间。转换后对 `std_log` 使用上限：

   ```text
   std_log = min(std_log, max_std_log)
   ```

10. 计算移位对数正态分布摘要：
    - `posterior_mean`
    - `posterior_median`
    - `posterior_mode`
    - `posterior_std`
    - 置信区间
    - 离散参考浓度 bin 的概率质量
11. 训练时执行 Leave-One-Out：每次留出一个样品，用剩余样品重新训练并预测留出样品。
12. 评估和默认图使用 `posterior_median`。

## 算法输出

算法层面的输出是：

- 模型参数和标准化参数。
- 每个样品的后验分布参数 `posterior_log_mean`, `posterior_log_std`。
- 后验点估计、区间估计和概率表。
- 训练集和 LOO 指标：R2、RMSE、MAE、平均 NLL。

实际默认输出目录：

- `concentration_GS/amortized_output/`

默认输出文件：

- `amortized_posterior_model.pt`: PyTorch 模型 bundle。
- `train_posterior_summary.csv`: 训练集后验摘要。
- `loo_posterior_summary.csv`: LOO 后验摘要。
- `train_posterior_probabilities.csv`: 训练集离散浓度概率。
- `loo_posterior_probabilities.csv`: LOO 离散浓度概率。
- `posterior_evaluation.png`: 训练/LOO 散点图和概率热图。

预测新 CSV 时默认输出：

- `<roi_csv_stem>_pred_concentration_gs.csv`
- `<roi_csv_stem>_pred_concentration_gs_prob.csv`
- `<roi_csv_stem>_pred_concentration_gs.png`

## 使用方式

```bash
python concentration_GS/train.py
python concentration_GS/predict.py --model concentration_GS/amortized_output/amortized_posterior_model.pt --roi-csv dataset/food/roi_detailed_statistics.csv
```
