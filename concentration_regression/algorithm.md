# concentration_regression 算法说明

本目录实现基于物理特征和 Ridge 回归的浓度反演。模型把 ROI/BG 统计转换成 Beer-Lambert 风格的对数强度和通道差分特征，再用带强正则的线性模型预测浓度。

## 主要代码

- `train.py`: 训练 Ridge 模型，输出训练集和 Leave-One-Out 评估。
- `predict.py`: 加载保存的模型，对新的 ROI 统计 CSV 预测浓度。
- `physics_features.py`: 从 ROI CSV 构造物理特征。
- `config.py`: 默认路径、正则强度、目标变换和输出文件名。

## 算法输入

算法层面的输入是：

- 样品真实浓度 `C`，从 `filename` 中解析。
- ROI 与背景通道统计：`roi_*_mean`, `bg_*_mean`, `roi_G*_var`。
- 可选曝光时间 `exposure_time`。
- 可选 ROI 分布参数：`roi_*_gauss_mu`, `roi_*_gamma_shape`, `roi_*_gamma_scale`。

实际默认输入文件：

- `dataset/food/roi_detailed_statistics.csv`

当前默认：

- `use_bg=True`
- `feature_set=compact`
- `bg_correction=subtract`: 背景校正统一使用 `ROI - BG`
- `target_transform=log`

批处理脚本会用训练集下的 `roi_detailed_statistics.csv` 作为输入，`dataset/food` 保留为测试集。

## 特征构造

背景校正信号：

```text
R_sig = max(roi_R_mean - bg_R_mean, clip_min)
G_sig = max(avg(roi_G1_mean - bg_G1_mean,
                roi_G2_mean - bg_G2_mean), clip_min)
B_sig = max(roi_B_mean - bg_B_mean, clip_min)
```

基础物理特征：

```text
ln_R_sig, ln_G_sig, ln_B_sig
ln_R_sig_per_exp, ln_G_sig_per_exp, ln_B_sig_per_exp
Ratio_RG, Delta_RG, Delta_GB
Var_G
exposure_time, ln_exposure
```

如果 CSV 包含 ROI 分布参数，`full` 特征还会追加：

```text
exp_roi_*_gauss_mu      = exp(roi_*_gauss_mu / 1000)
log_roi_*_gamma_shape   = log(max(roi_*_gamma_shape, eps))
sqrt_roi_*_gamma_scale  = sqrt(max(roi_*_gamma_scale, 0))
```

`compact` 是当前默认，使用最近外部测试中最稳的 6 个无曝光物理特征，不包含 `Var_G`：

```text
ln_R_sig, ln_G_sig, ln_B_sig, Ratio_RG, Delta_RG, Delta_GB
```

`compact_exposure` 保留 12 个基础物理特征，仍包含 `Var_G`。

## 处理流程

1. 读取 ROI CSV，并从文件名解析 `conc_true`。
2. 根据 `g_mode` 选择 G1 或 G1/G2 平均值。
3. 根据 `use_bg` 决定是否使用背景扣除信号。
4. 构造特征矩阵 `X`，并按 `feature_set` 选择列。
5. 目标浓度默认做 log 变换：

   ```text
   y_target = log(C + eps)
   ```

6. 构造样品权重：

   ```text
   weight = 1 / (C + weight_eps)
   ```

7. 训练模型：

   ```text
   StandardScaler -> RidgeCV(alphas=(10000, 100000))
   ```

8. 将预测从 log 空间反变换回浓度：

   ```text
   C_pred = exp(y_pred) - eps
   ```

9. 使用 isotonic regression 对训练预测做单调校准；LOO 中每折单独拟合校准器。
10. 执行 Leave-One-Out，输出训练集和 LOO 指标。

## 算法输出

算法层面的输出是：

- Ridge pipeline 和校准器。
- 使用的特征名、背景设置、目标变换。
- 选中的正则强度 alpha。
- 每个特征的 Ridge 系数。
- 训练集和 LOO 预测浓度、残差、R2/RMSE/MAE。

实际默认输出目录：

- `concentration_regression/rgb_fit_output/`

默认输出文件：

- `concentration_model_v1.pkl`: `joblib` 模型 bundle。
- `ridge_predictions.csv`: 训练集预测结果。
- `ridge_predictions_loo.csv`: Leave-One-Out 预测结果。
- `ridge_coefficients.csv`: 特征系数。
- `concentration_model_v1_evaluation.png`: 训练和 LOO 评估图。

批处理训练时实际输出示例：

- `OUTPUT/mate20x/rgb_concentration_model.pkl`
- `OUTPUT/mate20x/rgb_ridge_predictions.csv`
- `OUTPUT/mate20x/rgb_ridge_predictions_loo.csv`
- `OUTPUT/mate20x/rgb_ridge_coefficients.csv`
- `OUTPUT/mate20x/rgb_ridge_evaluation.png`

## 使用方式

```bash
python concentration_regression/train.py
python concentration_regression/predict.py --model concentration_regression/rgb_fit_output/concentration_model_v1.pkl --roi-csv dataset/food/roi_detailed_statistics.csv
```
