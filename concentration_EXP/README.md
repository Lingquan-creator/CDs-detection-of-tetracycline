# concentration_EXP

本目录使用指数衰减模型从 ROI/BG 统计中反演浓度。模型直接拟合背景扣除后的 RAW 光强通道：

```text
I(C) = a * exp(b * C) + c
C_pred = log((I - c) / a) / b
```

当前默认会分别拟合 `R_sig/G1_sig/G2_sig/G_sig/B_sig`，并用训练集反演浓度 `inv_r2` 最高的通道作为最终预测通道。

## 主要文件

- `config.py`: 默认路径、输出文件名和指数拟合默认参数。
- `physics_features.py`: 浓度/曝光时间解析工具。
- `train.py`: 训练指数模型并输出参数、训练集拟合结果、浓度反演结果和图。
- `predict.py`: 加载 `exp_fit_params.csv`，对新的 ROI CSV 预测浓度。
- `algorithm.md`: 更完整的代码和算法说明。

## 输入

默认输入：

```text
dataset/food/roi_detailed_statistics.csv
```

输入 CSV 至少需要：

```text
filename
roi_R_mean, roi_G1_mean, roi_B_mean
bg_R_mean, bg_G1_mean, bg_B_mean
```

如果存在 `roi_G2_mean/bg_G2_mean`，代码会同时构造 `G2_sig` 和平均绿色通道 `G_sig`。`exposure_time` 可选，缺失时会从文件名或表内有效值兜底。

## 处理流程

1. 从 `filename` 解析真实浓度。
2. 构造背景扣除光强：

   ```text
   R_sig  = max(roi_R_mean  - bg_R_mean,  0.1)
   G1_sig = max(roi_G1_mean - bg_G1_mean, 0.1)
   G2_sig = max(roi_G2_mean - bg_G2_mean, 0.1)
   B_sig  = max(roi_B_mean  - bg_B_mean,  0.1)
   G_sig  = (G1_sig + G2_sig) / 2
   ```

3. 用 `scipy.optimize.curve_fit` 对每个光强通道拟合指数模型。
4. 反演浓度并计算 `inv_r2/inv_rmse`。
5. 按 `--channel` 选择单通道或聚合输出。

## 常用命令

```bash
python concentration_EXP/train.py
python concentration_EXP/train.py --roi-csv dataset/food/roi_detailed_statistics.csv --channel G_sig
python concentration_EXP/train.py --weight-eps 10.0

python concentration_EXP/predict.py \
  --params concentration_EXP/exp_fit_output/exp_fit_params.csv \
  --roi-csv dataset/food/roi_detailed_statistics.csv
```

`--channel` 支持：

- `light_best`: 默认，选择 `inv_r2` 最高的拟合通道。
- `R_sig`, `G1_sig`, `G2_sig`, `G_sig`, `B_sig`: 指定单通道。
- `light_mean`, `light_median`: 聚合所有拟合成功通道。

## 输出

默认输出目录：

```text
concentration_EXP/exp_fit_output/
```

主要输出：

- `exp_fit_params.csv`: 每个通道的 `a/b/c`、光强拟合指标和浓度反演指标。
- `exp_fit_predictions.csv`: 训练集实测光强、曝光归一化光强和拟合光强。
- `<roi_csv_stem>_pred_concentration_exp.csv`: 最终浓度反演结果和各通道反演结果。
- `exp_fit_evaluation.png`: 通道拟合曲线和浓度反演评估图。
- `<roi_csv_stem>_pred_concentration_exp.png`: 预测阶段的结果图。

## 依赖

```bash
pip install numpy pandas scipy matplotlib
```
