# concentration_EXP 算法说明

本目录实现基于指数衰减模型的浓度反演。核心假设是 ROI 的背景扣除后光强会随浓度单调衰减，可用

```text
I(C) = a * exp(b * C) + c
```

拟合，其中 `C` 是浓度，`I` 是某个颜色通道的校正光强，约束上通常希望 `a > 0`、`b <= 0`。

## 主要代码

- `train.py`: 训练指数模型，并用训练集反演浓度。
- `predict.py`: 加载已拟合参数，对新的 ROI 统计 CSV 反演浓度。
- `config.py`: 默认输入输出路径与默认通道。
- `physics_features.py`: 从文件名解析浓度/曝光时间，并提供共享特征工具。

## 算法输入

算法层面的输入是每个样品的：

- 已知浓度 `C`，从 `filename` 文件名解析得到，例如 `12.dng` 表示 12 uM。
- ROI 与背景的 RAW 通道均值：`roi_R_mean`、`roi_G1_mean`、`roi_G2_mean`、`roi_B_mean` 以及对应的 `bg_*_mean`。
- 可选曝光时间 `exposure_time`，或从文件名 `浓度_曝光时间.dng` 解析；缺失时使用默认曝光。

实际默认输入文件：

- `pic_RGB/roi_results/roi_detailed_statistics.csv`

这个 CSV 必须至少包含：

- `filename`
- `roi_R_mean`, `roi_G1_mean`, `roi_B_mean`
- `bg_R_mean`, `bg_G1_mean`, `bg_B_mean`
- 如果使用双绿通道，还会使用 `roi_G2_mean`, `bg_G2_mean`

## 处理流程

1. 读取 `roi_detailed_statistics.csv`。
2. 从 `filename` 解析真实浓度 `conc_true`，丢弃无法解析浓度的行。
3. 构造背景扣除通道：

   ```text
   R_corr  = max(roi_R_mean  - bg_R_mean,  0)
   G1_corr = max(roi_G1_mean - bg_G1_mean, 0)
   G2_corr = max(roi_G2_mean - bg_G2_mean, 0)
   G_corr  = (G1_corr + G2_corr) / 2
   B_corr  = max(roi_B_mean  - bg_B_mean,  0)
   ```

4. 对每个通道再做曝光归一化：

   ```text
   channel_per_exp = channel_corr / exposure_time
   ```

5. 对 `R_corr`、`G1_corr`、`G2_corr`、`G_corr`、`B_corr` 分别拟合指数模型。拟合时使用 `scipy.optimize.curve_fit`，并用 `weight_eps` 让低浓度样品在拟合中有更高权重。
6. 默认选定 `G1_corr` 通道，将指数模型反解为浓度：

   ```text
   C_pred = ln((I - c) / a) / b
   ```

   其中 `(I - c) / a` 会被裁剪到合法范围，避免 `log(0)` 或负数。
7. 计算残差、R2、RMSE，并绘制通道拟合曲线和浓度预测图。

## 算法输出

算法层面的输出是：

- 每个通道的指数模型参数 `a`, `b`, `c`。
- 每个样品的反演浓度 `conc_pred`。
- 与真实浓度的误差 `residual`。
- 拟合质量指标 `r2`, `rmse`。

实际默认输出文件位于 `concentration_EXP/exp_fit_output/`：

- `exp_fit_params.csv`: 每个通道的 `a`, `b`, `c`, `r2`, `rmse`。
- `exp_fit_predictions.csv`: 每个通道的实测曝光归一化光强和模型拟合光强。
- `roi_detailed_statistics_pred_concentration_exp.csv`: 选定通道反演出的浓度结果。
- `exp_fit_evaluation.png`: 各通道指数拟合曲线和浓度反演效果图。

## 使用方式

训练并反演默认 CSV：

```bash
python concentration_EXP/train.py
```

指定输入和通道：

```bash
python concentration_EXP/train.py \
  --roi-csv pic_RGB/roi_results/roi_detailed_statistics.csv \
  --channel G_corr
```

用已保存参数预测新 CSV：

```bash
python concentration_EXP/predict.py \
  --params concentration_EXP/exp_fit_output/exp_fit_params.csv \
  --roi-csv pic_RGB/roi_results/roi_detailed_statistics.csv \
  --channel G1_corr
```
