# concentration_EXP 算法说明

本目录实现基于指数衰减模型的浓度反演。核心思想是把 ROI 的背景扣除信号视为随浓度单调变化的响应，并对每个候选通道分别拟合：

```text
I(C) = a * exp(b * C) + c
```

其中 `C` 是浓度，`I` 是 ROI-BG 后的光强信号。当前默认直接使用 `R_sig/G1_sig/G2_sig/G_sig/B_sig` 光强分别拟合，并选择训练集反演浓度 R2 最高的光强通道做最终反演。

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

实际默认输入文件：

- `dataset/food/roi_detailed_statistics.csv`

批处理脚本 `OUTPUT/run_dataset_concentration_training.sh` 会递归查找 `dataset/**/roi_detailed_statistics.csv`，跳过 `dataset/food` 作为测试集，并把每个训练集的输出写到对应 `OUTPUT/<dataset>/...` 目录。

## 处理流程

1. 读取 ROI CSV，要求至少包含 `filename` 和 ROI/BG 均值列。
2. 从 `filename` 提取真实浓度 `conc_true`，丢弃无法解析浓度的样品。
3. 构造 ROI-BG 背景扣除信号：

```text
R_sig = max(roi_R_mean - bg_R_mean, clip_min)
G_sig = max(avg(roi_G1_mean - bg_G1_mean,
                roi_G2_mean - bg_G2_mean), clip_min)
B_sig = max(roi_B_mean - bg_B_mean, clip_min)
```

4. 直接使用 ROI-BG 光强通道作为指数拟合的 `I`：

```text
R_sig, G1_sig, G2_sig, G_sig, B_sig
```

5. 对各个光强通道分别拟合 `I(C) = a * exp(b*C) + c`。拟合使用 `scipy.optimize.curve_fit`，并用 `weight_eps` 提高低浓度样品权重。
6. 对每个拟合成功的光强通道分别反演浓度：

   ```text
   C_pred = log((I - c) / a) / b
   ```

   反演前会裁剪 `(I - c) / a`，避免负数或 `log(0)`。
7. 默认 `--channel light_best`，即选择训练集反演浓度 R2 最高的光强通道做最终反演；也可以指定某一个通道名，或使用 `light_median` / `light_mean` 聚合各通道反演结果。`feature_best/feature_median/feature_mean` 作为旧脚本兼容别名保留。
8. 输出每个样品的预测浓度、各光强通道反演浓度、残差、R2/RMSE，并绘制拟合图。

## 算法输出

算法层面的输出是：

- 每个光强通道的指数参数 `a`, `b`, `c`。
- 每个光强通道的拟合质量 `r2`, `rmse`。
- 默认最优光强通道反演浓度 `conc_pred`、各通道反演浓度和残差。

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
