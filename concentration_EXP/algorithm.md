# concentration_EXP 算法说明

本目录实现基于指数衰减模型的浓度反演。核心思想是把 ROI 的背景扣除信号视为随浓度单调变化的响应，并对每个候选通道分别拟合：

```text
I(C) = a * exp(b * C) + c
```

其中 `C` 是浓度，`I` 是曝光归一化后的通道信号。预测时使用选定通道把指数函数反解成浓度。

## 主要代码

- `train.py`: 读取 ROI 统计 CSV，拟合指数模型，输出训练集反演结果和图。
- `predict.py`: 读取已保存的指数参数，对新 ROI CSV 做浓度反演。
- `config.py`: 默认输入输出路径、默认反演通道和拟合权重参数。
- `physics_features.py`: 解析文件名中的浓度/曝光时间，并提供通用物理特征工具。

## 算法输入

算法层面的输入是：

- 样品真实浓度 `C`，从 `filename` 中解析，例如 `120.dng` 或 `chicken120.dng`。
- ROI 与背景 RAW 通道均值：`roi_R_mean`, `roi_G1_mean`, `roi_G2_mean`, `roi_B_mean` 和对应 `bg_*_mean`。
- 可选曝光时间 `exposure_time`，缺失时使用默认曝光。
- 可选 ROI 分布参数：`roi_*_gauss_mu`, `roi_*_gamma_shape`, `roi_*_gamma_scale`。

实际默认输入文件：

- `dataset/food/roi_detailed_statistics.csv`

批处理脚本 `OUTPUT/run_dataset_concentration_training.sh` 会递归查找 `dataset/**/roi_detailed_statistics.csv`，跳过 `dataset/food` 作为测试集，并把每个训练集的输出写到对应 `OUTPUT/<dataset>/...` 目录。

## 处理流程

1. 读取 ROI CSV，要求至少包含 `filename` 和 ROI/BG 均值列。
2. 从 `filename` 提取真实浓度 `conc_true`，丢弃无法解析浓度的样品。
3. 构造背景扣除通道：

   ```text
   R_corr  = max(roi_R_mean  - bg_R_mean,  0)
   G1_corr = max(roi_G1_mean - bg_G1_mean, 0)
   G2_corr = max(roi_G2_mean - bg_G2_mean, 0)
   G_corr  = (G1_corr + G2_corr) / 2
   B_corr  = max(roi_B_mean  - bg_B_mean,  0)
   ```

4. 对每个强度通道做曝光归一化：

   ```text
   channel_per_exp = channel / exposure_time
   ```

5. 如果 CSV 中存在 ROI 分布参数，还会追加变换通道：

   ```text
   exp_roi_*_gauss_mu      = exp(roi_*_gauss_mu / 1000)
   log_roi_*_gamma_shape   = log(max(roi_*_gamma_shape, eps))
   sqrt_roi_*_gamma_scale  = sqrt(max(roi_*_gamma_scale, 0))
   ```

   如果存在 `roi_*_gauss_shape`，代码会优先使用它；当前数据通常是 `gamma_shape`。

6. 对所有候选通道分别拟合 `I(C) = a * exp(b*C) + c`。拟合使用 `scipy.optimize.curve_fit`，并用 `weight_eps` 提高低浓度样品权重。
7. 默认使用 `G1_corr` 反演浓度：

   ```text
   C_pred = log((I - c) / a) / b
   ```

   反演前会裁剪 `(I - c) / a`，避免负数或 `log(0)`。
8. 输出每个样品的预测浓度、残差、R2/RMSE，并绘制拟合图。

## 算法输出

算法层面的输出是：

- 每个候选通道的指数参数 `a`, `b`, `c`。
- 每个通道的拟合质量 `r2`, `rmse`。
- 选定通道的反演浓度 `conc_pred` 和残差。

实际默认输出目录：

- `concentration_EXP/exp_fit_output/`

默认输出文件：

- `exp_fit_params.csv`: 每个通道的指数参数和拟合指标。
- `exp_fit_predictions.csv`: 各通道实测信号和拟合信号。
- `<roi_csv_stem>_pred_concentration_exp.csv`: 选定通道反演浓度。
- `exp_fit_evaluation.png`: 通道拟合曲线和浓度反演图。

批处理训练时实际输出文件名由脚本指定，例如：

- `OUTPUT/mate20x/exp_fit_params.csv`
- `OUTPUT/mate20x/exp_concentration_predictions.csv`
- `OUTPUT/mate20x/exp_fit_evaluation.png`

## 使用方式

```bash
python concentration_EXP/train.py
python concentration_EXP/predict.py --params concentration_EXP/exp_fit_output/exp_fit_params.csv --roi-csv dataset/food/roi_detailed_statistics.csv
```
