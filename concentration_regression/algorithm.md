# concentration_regression 算法说明

本目录实现基于物理特征和 Ridge 回归的浓度反演。核心思想是根据 Beer-Lambert 类关系，把背景扣除后的通道强度取对数，再用带强正则的线性模型预测浓度。

## 主要代码

- `train.py`: 训练 Ridge 模型，输出训练集和 Leave-One-Out 评估。
- `predict.py`: 加载保存的模型，对新的 ROI 统计 CSV 预测浓度。
- `physics_features.py`: 从 ROI CSV 构造物理特征。
- `config.py`: 默认路径、正则强度、目标变换和输出文件名。

## 算法输入

算法层面的输入是：

- 样品真实浓度 `C`，从 `filename` 解析。
- ROI 与背景通道统计。
- 可选曝光时间。

实际默认输入文件：

- `pic_RGB/roi_results/roi_detailed_statistics.csv`

默认先由 `physics_features.build_feature_frame()` 生成完整特征，再根据 `--feature-set` 选择实际训练列。当前支持：

```text
compact:
  ln_R_sig, ln_G_sig, ln_B_sig, Ratio_RG, Delta_RG, Ratio_BG, Var_G

compact_exposure:
  ln_R_sig, ln_G_sig, ln_B_sig,
  ln_R_sig_per_exp, ln_G_sig_per_exp, ln_B_sig_per_exp,
  Ratio_RG, Delta_RG, Ratio_BG, Var_G,
  exposure_time, ln_exposure
```

如果不用背景扣除，`ln_*_sig` 会替换为 `ln_R`, `ln_G`, `ln_B`，`ln_*_sig_per_exp` 会替换为 `ln_*_per_exp`。

完整候选列包括：

- 对数强度：`ln_R_sig`, `ln_G_sig`, `ln_B_sig`
- 曝光归一化对数强度：`ln_R_sig_per_exp`, `ln_G_sig_per_exp`, `ln_B_sig_per_exp`
- 比值特征：`Ratio_RG`, `Ratio_BG`, `Delta_RG`
- 绿色通道方差：`Var_G`
- 曝光特征：`exposure_time`, `ln_exposure`

如果不使用背景扣除，对应特征名会从 `*_sig` 变为原始通道名，例如 `ln_R`。

## 处理流程

1. 读取 ROI CSV，并从 `filename` 解析真实浓度 `conc_true`。
2. 根据 `g_mode` 使用 G1 或 G1/G2 平均值。
3. 根据 `use_bg` 决定使用原始 ROI 均值还是 `ROI - BG` 信号。
4. 构造物理特征矩阵 `X`，再按 `feature_set` 选择方案B或方案B加曝光时间。
5. 对目标浓度做变换，默认是：

   ```text
   y_target = log(concentration + eps)
   ```

6. 设置样品权重：

   ```text
   weight = 1 / (concentration + weight_eps)
   ```

   这样低浓度样品不会被高浓度样品完全淹没，但 `weight_eps` 较大时权重会更平滑。

7. 训练模型：

   ```text
   StandardScaler() -> RidgeCV(alphas=(10000, 100000))
   ```

   `StandardScaler` 先标准化特征，`RidgeCV` 在候选正则强度里选内部 CV 分数更好的 alpha。

8. 将预测值从 log 空间反变换回浓度：

   ```text
   concentration_pred = exp(y_pred) - eps
   ```

9. 使用 isotonic regression 对训练集预测值做单调校准，并在 LOO 中对每一折用训练折单独拟合校准器。
10. 执行 Leave-One-Out：每次留出一个样品，重新训练 Ridge 和校准器，再预测留出样品。

## 算法输出

算法层面的输出是：

- Ridge 模型和 scaler。
- 选中的正则强度 alpha。
- 特征系数。
- 训练集预测浓度、LOO 预测浓度和残差。
- R2、RMSE、MAE。

实际默认输出文件位于 `concentration_regression/rgb_fit_output/`：

- `concentration_model_v1.pkl`: `joblib` 保存的模型 bundle，包含 pipeline、特征设置、target transform、校准器等。
- `ridge_predictions.csv`: 训练集预测结果。
- `ridge_predictions_loo.csv`: Leave-One-Out 预测结果。
- `ridge_coefficients.csv`: 每个特征的 Ridge 系数。
- `concentration_model_v1_evaluation.png`: 训练集和 LOO 预测散点图。

预测新 CSV 时默认输出：

- `<roi_csv_stem>_pred_concentration.csv`
- `<roi_csv_stem>_pred_concentration.png`

## 使用方式

训练默认 CSV：

```bash
python concentration_regression/train.py
```

指定输入：

```bash
python concentration_regression/train.py \
  --roi-csv pic_RGB/roi_results/roi_detailed_statistics.csv
```

使用带曝光时间的方案：

```bash
python concentration_regression/train.py \
  --roi-csv pic_RGB/roi_results/roi_detailed_statistics.csv \
  --feature-set compact_exposure
```

预测：

```bash
python concentration_regression/predict.py \
  --model concentration_regression/rgb_fit_output/concentration_model_v1.pkl \
  --roi-csv pic_RGB/roi_results/roi_detailed_statistics.csv
```
