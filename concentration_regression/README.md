# concentration_regression

本目录使用物理特征加 Ridge 回归做浓度反演。代码将 ROI/BG RAW 通道统计转换为对数光强、通道比值和差分特征，再用标准化后的 RidgeCV 预测浓度，并用 isotonic regression 做单调校准。

## 主要文件

- `config.py`: 默认路径、目标变换、背景校正、特征集合和正则网格。
- `physics_features.py`: ROI/BG 物理特征构造。
- `train.py`: 训练 RidgeCV，执行 Leave-One-Out，保存模型、预测、系数和评估图。
- `predict.py`: 加载模型 bundle，对新的 ROI CSV 输出浓度预测。
- `algorithm.md`: 更完整的代码和算法说明。

## 默认配置

```text
输入: dataset/food/roi_detailed_statistics.csv
输出: concentration_regression/rgb_fit_output/
target_transform: log
use_bg: true
bg_correction: subtract
feature_set: compact
alphas: (10000, 100000)
weight_eps: 30.0
clip_min: 0.1
g_mode: avg
```

## 输入

训练 CSV 需要包含：

```text
filename
roi_R_mean, roi_G1_mean, roi_B_mean
bg_R_mean, bg_G1_mean, bg_B_mean
```

默认 `g_mode=avg` 且 `use_bg=true`，通常还需要：

```text
roi_G2_mean, bg_G2_mean
```

可选列：

- `roi_G1_var`, `roi_G2_var`: 用于 `Var_G`。
- `exposure_time`: 用于曝光相关特征。
- ROI 分布参数列: `feature_set=full` 时使用。

`filename` 会用于解析真实浓度；训练时无法解析浓度的样品会被丢弃。

## 特征集合

默认 `feature_set=compact`，使用：

```text
ln_R_sig, ln_G_sig, ln_B_sig, Ratio_RG, Delta_RG, Delta_GB
```

其他选项：

- `compact_exposure`: 使用曝光归一化、`Ratio_BG`、`Var_G`、曝光时间等基础特征。
- `ln_rgb`: 只使用 `ln_R_sig/ln_G_sig/ln_B_sig`。
- `full`: 使用全部可构造特征，包括可用 ROI 分布参数。

背景校正支持：

```text
subtract: ROI - BG
divide: ROI / (BG + eps)
relative: (ROI - BG) / (BG + eps)
```

所有背景校正信号都会裁剪到不小于 `clip_min=0.1`。

## 训练流程

1. 读取 ROI CSV，从 `filename` 解析 `conc_true`。
2. 构造特征并按 `feature_set` 选择训练列。
3. 默认对目标做 log 变换：

   ```text
   y_target = log(C + 1e-6)
   ```

4. 样品权重：

   ```text
   weight = 1 / (C + weight_eps)
   ```

5. 训练：

   ```text
   StandardScaler -> RidgeCV(alphas=(10000, 100000))
   ```

6. 将预测反变换回浓度，并用 `IsotonicRegression(out_of_bounds="clip")` 校准。
7. 执行 Leave-One-Out，输出训练集和 LOO 的 R2、RMSE、MAE。

## 常用命令

训练：

```bash
python concentration_regression/train.py
```

指定输入和特征：

```bash
python concentration_regression/train.py \
  --roi-csv dataset/food/roi_detailed_statistics.csv \
  --feature-set compact \
  --use-bg true \
  --bg-correction subtract
```

预测：

```bash
python concentration_regression/predict.py \
  --model concentration_regression/rgb_fit_output/concentration_model_v1.pkl \
  --roi-csv dataset/food/roi_detailed_statistics.csv
```

处理 scikit-learn 版本警告：

```bash
python concentration_regression/predict.py \
  --model concentration_regression/rgb_fit_output/concentration_model_v1.pkl \
  --roi-csv dataset/food/roi_detailed_statistics.csv \
  --suppress-sklearn-version-warning
```

## 输出

默认输出目录：

```text
concentration_regression/rgb_fit_output/
```

训练输出：

- `concentration_model_v1.pkl`: `joblib` 模型 bundle。
- `ridge_predictions.csv`: 训练集原始预测、校准预测和残差。
- `ridge_predictions_loo.csv`: Leave-One-Out 预测和残差。
- `ridge_coefficients.csv`: Ridge 特征系数。
- `concentration_model_v1_evaluation.png`: 训练和 LOO 评估图。

预测输出：

- `<roi_csv_stem>_pred_concentration.csv`
- `<roi_csv_stem>_pred_concentration.png`

## 依赖

```bash
pip install numpy pandas matplotlib scikit-learn joblib
```
