# concentration_GS 代码说明

本目录实现基于小型神经网络的摊销后验浓度反演。模型不只输出单个浓度值，而是学习条件后验分布：

```text
q(log(C + eps) | x) = Normal(mu_phi(x), sigma_phi(x)^2)
```

其中 `x` 是从 ROI/BG 统计构造的物理特征，`mu_phi` 和 `sigma_phi` 由 MLP 输出。转换回浓度空间后，预测分布是移位对数正态分布。默认点估计使用 `posterior_median`。

## 代码文件

- `config.py`: 定义默认路径、输出文件名和训练超参数。默认输入为 `dataset/food/roi_detailed_statistics.csv`，输出目录为 `concentration_GS/amortized_output/`。
- `physics_features.py`: 构造物理特征，解析浓度和曝光时间，检查背景列。
- `train.py`: 训练入口。训练摊销后验模型，输出训练集和 Leave-One-Out 后验摘要、离散概率表和评估图。
- `predict.py`: 预测入口。加载 `.pt` 模型 bundle，对新的 ROI CSV 输出后验摘要、离散概率和预测图。
- `visualize_results.py`: 读取训练输出，生成散点图、残差图、不确定性图和后验分布图。

## 输入数据

训练和预测使用 `roi_detailed_statistics.csv`。核心列包括：

- `filename`: 用于解析浓度标签，训练时必须能解析出有效浓度。
- `roi_R_mean`, `roi_G1_mean`, `roi_G2_mean`, `roi_B_mean`: ROI 四通道均值。
- `bg_R_mean`, `bg_G1_mean`, `bg_G2_mean`, `bg_B_mean`: 背景四通道均值。默认 `--use-bg true`，因此训练输入默认必须包含这些列。
- `roi_G1_var`, `roi_G2_var`: 可选，构造 `Var_G` 时使用；缺失则置为 0。
- `exposure_time`: 可选，缺失时从文件名或表内有效值兜底。
- ROI 分布参数列可选。使用 `feature_set=full` 时会追加变换后的分布特征。

## 特征构造

默认 `g_mode=avg`，绿色通道使用 G1/G2 平均。背景扣除模式下：

```text
R_sig = max(roi_R_mean - bg_R_mean, 0.1)
G_sig = (max(roi_G1_mean - bg_G1_mean, 0.1)
       + max(roi_G2_mean - bg_G2_mean, 0.1)) / 2
B_sig = max(roi_B_mean - bg_B_mean, 0.1)
```

基础物理特征为：

```text
ln_R_sig, ln_G_sig, ln_B_sig
ln_R_sig_per_exp, ln_G_sig_per_exp, ln_B_sig_per_exp
Ratio_RG, Delta_RG, Ratio_BG, Delta_GB
Var_G
exposure_time, ln_exposure
```

如果 `use_bg=false`，对应的 `ln_*_sig` 会变为原始 ROI 光强的 `ln_R/ln_G/ln_B`。

`feature_set` 控制实际输入列：

- `compact`: 默认 6 维，使用 `ln_R_sig, ln_G_sig, ln_B_sig, Ratio_RG, Delta_RG, Delta_GB`。
- `compact_exposure`: 使用 12 维基础特征中的曝光相关列和 `Ratio_BG/Var_G`，但不含 `Delta_GB`。
- `full`: 使用 `build_feature_frame()` 生成的全部列，包括可用的 ROI 分布特征。

可选分布特征包括：

```text
exp_roi_*_gauss_mu
log_roi_*_gauss_shape 或 log_roi_*_gamma_shape
sqrt_roi_*_gamma_scale
```

## 模型结构和训练

神经网络定义在 `AmortizedPosteriorNet`：

```text
input_dim -> Linear(hidden_dim) -> SiLU -> Dropout
          -> Linear(hidden_dim) -> SiLU
          -> mu_head, std_head
```

默认超参数：

- `hidden_dim=8`
- `dropout=0.05`
- `learning_rate=1e-2`
- `weight_decay=1e-3`
- `epochs=2500`
- `patience=250`
- `min_std=0.08`
- `std_reg_lambda=0.01`
- `max_std_log=2.0`
- `weight_eps=25.0`

训练流程：

1. 读取 ROI CSV 并解析 `conc_true`。
2. 选择 `use_bg`、`g_mode` 和 `feature_set`，构造特征矩阵 `X`。
3. 标准化特征：

   ```text
   X_std = (X - mean(X)) / std(X)
   ```

4. 将目标变换到 log 空间并标准化：

   ```text
   y_log = log(C + eps)
   y_std = (y_log - mean(y_log)) / std(y_log)
   ```

5. 用加权高斯负对数似然训练：

   ```text
   loss = weighted_NLL + std_reg_lambda * mean(std^2)
   weight = 1 / (C + weight_eps)
   ```

6. 使用 early stopping 保存训练损失最低的参数。
7. 将输出从标准化 log 空间还原为 `posterior_log_mean` 和 `posterior_log_std`，并用 `max_std_log` 限制过大的 log 标准差。
8. 计算后验均值、中位数、众数、标准差、置信区间和 NLL。
9. 以训练集中出现过的浓度作为参考 bin，计算每个样品落入各浓度 bin 的概率质量。
10. 执行 Leave-One-Out。每折用剩余样品重新训练一个模型，再预测留出样品。

## 输出文件

默认输出目录：

```text
concentration_GS/amortized_output/
```

训练输出：

- `amortized_posterior_model.pt`: PyTorch bundle，包含网络参数、标准化参数、特征名、参考浓度、训练配置和评估指标。
- `train_posterior_summary.csv`: 训练集后验摘要。
- `loo_posterior_summary.csv`: Leave-One-Out 后验摘要。
- `train_posterior_probabilities.csv`: 训练集离散浓度 bin 概率。
- `loo_posterior_probabilities.csv`: LOO 离散浓度 bin 概率。
- `posterior_evaluation.png`: 训练/LOO 后验散点图和 LOO 概率热图。

预测输出：

- `<roi_csv_stem>_pred_concentration_gs.csv`: 后验摘要。
- `<roi_csv_stem>_pred_concentration_gs_prob.csv`: 参考浓度 bin 概率。
- `<roi_csv_stem>_pred_concentration_gs.png`: 预测散点或不确定性图。

## 使用方式

```bash
python concentration_GS/train.py
python concentration_GS/train.py --feature-set compact_exposure --use-bg true
python concentration_GS/predict.py --model concentration_GS/amortized_output/amortized_posterior_model.pt --roi-csv dataset/food/roi_detailed_statistics.csv
python concentration_GS/visualize_results.py
```
