# concentration_EXP 代码说明

本目录实现基于指数衰减模型的浓度反演。代码直接使用 ROI-BG 后的 RAW 通道光强拟合浓度响应，并通过指数函数反解浓度：

```text
I(C) = a * exp(b * C) + c
C_pred = log((I - c) / a) / b
```

其中 `C` 为浓度，`I` 为背景扣除后的通道光强。当前实现将每个光强通道单独拟合，并默认选择训练集反演浓度 `inv_r2` 最高的通道作为最终预测通道。

## 代码文件

- `config.py`: 定义默认输入、输出目录和文件名。默认输入是 `dataset/food/roi_detailed_statistics.csv`，默认输出目录是 `concentration_EXP/exp_fit_output/`，默认通道策略是 `light_best`。
- `physics_features.py`: 解析文件名中的浓度和曝光时间，并提供通用物理特征工具。指数模型主要使用其中的浓度解析和曝光时间解析函数。
- `train.py`: 训练入口。读取 ROI 统计 CSV，构造背景扣除光强，拟合各通道指数模型，保存参数、训练集拟合值、浓度反演结果和评估图。
- `predict.py`: 预测入口。读取训练阶段保存的 `exp_fit_params.csv`，对新的 ROI CSV 构造同样的背景扣除光强并反演浓度。
- `example.py`: 示例调用脚本，用于演示训练或预测流程。

## 输入数据

训练和预测都依赖 `roi_detailed_statistics.csv`。核心列包括：

- `filename`: 用于解析真实浓度。支持纯数字文件名，也支持从文件名中提取第一个数字。
- `roi_R_mean`, `roi_G1_mean`, `roi_G2_mean`, `roi_B_mean`: ROI 区域四个 Bayer 通道均值。
- `bg_R_mean`, `bg_G1_mean`, `bg_G2_mean`, `bg_B_mean`: 背景区域四个 Bayer 通道均值。
- `exposure_time`: 可选。缺失时会尝试从文件名解析，仍缺失则使用同表中第一个有效曝光时间，最终兜底为 `1.0`。

## 光强构造

`train.py` 和 `predict.py` 通过 `build_corrected_channels()` 生成拟合用光强：

```text
R_sig  = max(roi_R_mean  - bg_R_mean,  0.1)
G1_sig = max(roi_G1_mean - bg_G1_mean, 0.1)
G2_sig = max(roi_G2_mean - bg_G2_mean, 0.1)
B_sig  = max(roi_B_mean  - bg_B_mean,  0.1)
G_sig  = (G1_sig + G2_sig) / 2
```

实际参与指数拟合的通道为：

```text
R_sig, G1_sig, G2_sig, G_sig, B_sig
```

代码也会生成 `*_per_exp` 作为诊断输出，但默认不参与指数拟合和通道选择。

## 训练流程

1. 读取 ROI CSV，检查空表和必要列。
2. 从 `filename` 解析 `conc_true`，丢弃无法解析浓度的样品，并按浓度排序。
3. 计算 `R_sig/G1_sig/G2_sig/G_sig/B_sig`。
4. 对每个通道拟合指数模型。实际拟合使用重参数形式：

   ```text
   I(C) = (i0 - c) * exp(b * C) + c
   a = i0 - c
   ```

   约束为 `i0 >= max(I)`、`b <= 0`、`c <= min(I)`，使曲线保持递减且便于反演。
5. 样品权重通过 `sigma = sqrt(C + weight_eps)` 传入 `curve_fit`。默认 `weight_eps=5.0`，低浓度样品获得相对更高权重。
6. 对每个通道计算光强拟合指标 `r2/rmse`，再用反演浓度计算 `inv_r2/inv_rmse`。
7. 根据 `--channel` 决定最终浓度：
   - `light_best`: 选择 `inv_r2` 最高的通道。
   - 指定通道名，如 `G_sig`。
   - `light_mean` 或 `light_median`: 聚合所有拟合成功通道的反演值。
   - `feature_*` 名称保留为兼容别名。
8. 保存参数、训练集光强拟合结果、浓度反演结果和图像。

## 预测流程

`predict.py` 的流程与训练阶段保持一致：

1. 读取 `exp_fit_params.csv` 中各通道的 `a/b/c`。
2. 若请求 `light_best`，根据参数文件中的 `inv_r2` 选择通道。
3. 读取新的 ROI CSV 并重新计算背景扣除光强。
4. 用保存的指数参数反演浓度。
5. 如果文件名能解析出真实浓度，则额外输出 `conc_true` 和 `residual`。

## 输出文件

默认输出目录：

```text
concentration_EXP/exp_fit_output/
```

训练输出：

- `exp_fit_params.csv`: 每个通道的 `a`, `b`, `c`, `r2`, `rmse`, `inv_r2`, `inv_rmse`。
- `exp_fit_predictions.csv`: 训练样品的实测光强、曝光归一化光强和拟合光强。
- `<roi_csv_stem>_pred_concentration_exp.csv`: 最终反演浓度和各通道反演浓度。
- `exp_fit_evaluation.png`: 各通道指数曲线和最终浓度反演散点图。

预测输出：

- `<roi_csv_stem>_pred_concentration_exp.csv`
- `<roi_csv_stem>_pred_concentration_exp.png`

## 使用方式

```bash
python concentration_EXP/train.py
python concentration_EXP/train.py --roi-csv dataset/food/roi_detailed_statistics.csv --channel G_sig
python concentration_EXP/predict.py --params concentration_EXP/exp_fit_output/exp_fit_params.csv --roi-csv dataset/food/roi_detailed_statistics.csv
```
