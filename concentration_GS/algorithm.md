# concentration_GS 算法说明

本目录实现基于神经网络的摊销后验浓度反演。它不只输出一个浓度点估计，而是学习条件分布：

```text
q(log(C + eps) | x) = Normal(mu_phi(x), sigma_phi(x)^2)
```

其中 `x` 是从 ROI 统计中构造的物理特征，`mu_phi` 和 `sigma_phi` 由一个小型 MLP 输出。浓度空间中的预测分布是移位对数正态分布。当前默认点预测使用 `posterior_median`，避免对数正态均值在方差较大时被指数项放大。

## 主要代码

- `train.py`: 训练摊销后验模型，输出训练集评估和 Leave-One-Out 评估。
- `predict.py`: 加载模型，对新的 ROI 统计 CSV 输出后验分布。
- `visualize_results.py`: 读取训练输出 CSV，生成额外可视化。
- `physics_features.py`: 从 ROI 统计构造物理特征。
- `config.py`: 默认路径、网络超参数和输出文件名。

## 算法输入

算法层面的输入是：

- 样品真实浓度 `C`，从 `filename` 解析。
- ROI 和背景通道统计，用于构造物理特征。
- 可选曝光时间。

实际默认输入文件：

- `pic_RGB/roi_results/roi_detailed_statistics.csv`

默认使用背景扣除特征。如果 CSV 中存在背景列，`--use-bg auto` 会启用背景扣除。完整特征集包括：

- `ln_R_sig`, `ln_G_sig`, `ln_B_sig`
- `ln_R_sig_per_exp`, `ln_G_sig_per_exp`, `ln_B_sig_per_exp`
- `Ratio_RG`, `Delta_RG`, `Ratio_BG`
- `Var_G`
- `exposure_time`, `ln_exposure`

其中：

```text
R_sig = max(roi_R_mean - bg_R_mean, clip_min)
G_sig = max(roi_G_mean - bg_G_mean, clip_min)
B_sig = max(roi_B_mean - bg_B_mean, clip_min)
```

当前默认 `feature_set=compact`，对应无曝光时长的物理特征，训练时只保留：

```text
ln_R_sig, ln_G_sig, ln_B_sig, Ratio_RG, Delta_RG, Ratio_BG, Var_G
```

如果需要加入曝光时长，使用 `feature_set=compact_exposure`：

```text
ln_R_sig, ln_G_sig, ln_B_sig,
ln_R_sig_per_exp, ln_G_sig_per_exp, ln_B_sig_per_exp,
Ratio_RG, Delta_RG, Ratio_BG, Var_G,
exposure_time, ln_exposure
```

如果不用背景扣除，对应为：

```text
ln_R, ln_G, ln_B, Ratio_RG, Delta_RG, Ratio_BG, Var_G
```

无背景扣除且带曝光时，对应使用 `ln_R_per_exp`, `ln_G_per_exp`, `ln_B_per_exp`, `exposure_time`, `ln_exposure`。

## 处理流程

1. 读取 ROI CSV，并从文件名解析 `conc_true`。
2. 根据 `g_mode` 选择 G1 或双绿平均值，根据 `use_bg` 决定是否使用背景扣除。
3. 构造 12 维物理特征矩阵 `X`。
4. 对特征做标准化：

   ```text
   X_std = (X - mean(X)) / std(X)
   ```

5. 对目标做对数变换和标准化：

   ```text
   y_log = log(C + eps)
   y_std = (y_log - mean(y_log)) / std(y_log)
   ```

6. 训练 MLP。默认网络宽度 `hidden_dim=8`。使用 compact 特征集时结构为 `7 -> 8 -> 8 -> 2 heads`。网络共享一个 backbone，并有两个输出头：

   ```text
   mu = mu_head(backbone(X_std))
   std = min_std + softplus(std_head(backbone(X_std)))
   ```

7. 损失函数是加权高斯负对数似然，并额外惩罚过大的预测标准差：

   ```text
   NLL = 0.5*log(std^2) + 0.5*(y_std - mu)^2/std^2
   loss = weighted_NLL + std_reg_lambda * mean(std^2)
   ```

   样品权重约为 `1 / (concentration + weight_eps)`。默认 `std_reg_lambda=0.01`，用于降低小样本 LOO 中方差头失控的概率。

8. 将标准化空间的 `mu/std` 转回 `log(C + eps)` 空间。
   转换后会对 `std_log` 使用默认上限 `max_std_log=2.0`，避免极端折中方差头输出导致 `posterior_mean` 和 `posterior_std` 数值溢出。
9. 用移位对数正态分布计算：

   - `posterior_mean`
   - `posterior_median`
   - `posterior_mode`
   - `posterior_std`
   - 置信区间
   - 每个参考浓度区间的概率质量

10. 训练时额外执行 Leave-One-Out：每次留出一个样品，用剩余样品重新训练模型，再预测留出样品。
11. 训练指标、LOO 指标和默认图中的点预测使用 `posterior_median`，`posterior_mean` 仍保留在 CSV 中作为分布摘要字段。

## 算法输出

算法层面的输出是：

- 模型参数和标准化参数。
- 每个样品的后验分布参数 `posterior_log_mean`, `posterior_log_std`。
- 后验点估计和区间估计。
- 每个离散浓度 bin 的概率。
- 训练集和 LOO 指标：R2、RMSE、MAE、平均 NLL。

实际默认输出文件位于 `concentration_GS/amortized_output/`：

- `amortized_posterior_model.pt`: PyTorch 模型 bundle，包含权重、特征均值方差、目标均值方差等。
- `train_posterior_summary.csv`: 训练集后验摘要。
- `loo_posterior_summary.csv`: Leave-One-Out 后验摘要。
- `train_posterior_probabilities.csv`: 训练集离散浓度概率表。
- `loo_posterior_probabilities.csv`: LOO 离散浓度概率表。
- `posterior_evaluation.png`: 训练/LOO 散点图和概率热图。

预测新 CSV 时默认输出到 `concentration_GS/amortized_output/`：

- `<roi_csv_stem>_pred_concentration_gs.csv`
- `<roi_csv_stem>_pred_concentration_gs_prob.csv`
- `<roi_csv_stem>_pred_concentration_gs.png`

## 注意事项

`posterior_mean = exp(mu_log + 0.5 * std_log^2) - eps` 对 `std_log` 很敏感。当 LOO 小样本训练导致 `std_log` 变大时，`posterior_mean` 可能出现极大值。因此当前代码把评估和绘图默认切换到 `posterior_median`，在损失函数中加入 `std_reg_lambda * mean(std^2)` 约束方差头，并用 `max_std_log=2.0` 给后验 log 标准差加数值上限。解释结果时仍应同时查看 `posterior_log_std` 和概率分布。

## 使用方式

训练默认 CSV：

```bash
python concentration_GS/train.py
```

指定输入输出：

```bash
python concentration_GS/train.py \
  --roi-csv pic_RGB/roi_results/roi_detailed_statistics.csv \
  --model-out concentration_GS/amortized_output/amortized_posterior_model.pt
```

预测：

```bash
python concentration_GS/predict.py \
  --model concentration_GS/amortized_output/amortized_posterior_model.pt \
  --roi-csv pic_RGB/roi_results/roi_detailed_statistics.csv
```
