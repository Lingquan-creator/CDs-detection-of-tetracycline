# concentration_GS

`concentration_GS` 使用摊销推断做浓度概率预测，而不是只输出一个点估计值。

当前实现将 `log(concentration)` 的条件后验建模为高斯分布：

```text
q_phi(log c | x) = Normal(mu_phi(x), sigma_phi(x)^2)
```

其中：

- `x` 来自 ROI 统计量构造的物理特征
- `mu_phi(x)` 和 `sigma_phi(x)` 由一个小型 MLP 直接输出
- 最终在浓度空间得到一个移位对数正态后验，可导出：
  - 后验均值
  - 后验中位数
  - 后验众数
  - 后验标准差
  - 置信区间
  - 离散浓度 bins 上的概率分布

这和 `concentration` 中的岭回归不同。`concentration` 更偏向点预测；`concentration_GS` 的目标是直接给出 `p(c | x)` 的近似结果。

## 目录结构

```text
concentration_GS/
├── config.py
├── train.py
├── predict.py
└── amortized_output/
```

- `config.py`
  统一管理输入路径、输出路径和默认超参数。
- `train.py`
  训练摊销推断模型，执行全量训练和留一法评估，并输出 CSV 与图片。
- `predict.py`
  加载训练好的模型，对新的 ROI 统计 CSV 执行批量推理。
- `amortized_output/`
  保存训练得到的模型和评估结果。

## 方法概述

### 1. 特征

特征构造复用了 [physics_features.py](/h:/000/CDs/concentration/physics_features.py) 中的逻辑。

默认使用以下 7 个物理特征：

- `ln_R_sig` 或 `ln_R`
- `ln_G_sig` 或 `ln_G`
- `ln_B_sig` 或 `ln_B`
- `Ratio_RG`
- `Delta_RG`
- `Ratio_BG`
- `Var_G`

如果输入 CSV 里存在背景列，且 `--use-bg auto` 或 `--use-bg true`，则使用背景扣除后的信号特征：

```text
Signal = ROI - BG
```

### 2. 标签

训练标签由 `filename` 自动解析。例如：

- `0.2.dng -> 0.2`
- `12.0.dng -> 12.0`
- `100.dng -> 100`

因此训练 CSV 必须包含 `filename` 列，并且文件名 stem 能解析成浓度数值。

### 3. 模型

模型是一个小型 MLP：

- 输入维度：7
- 隐藏层宽度：默认 `16`
- 激活函数：`SiLU`
- 输出：
  - `mu`
  - `std`

训练目标是高斯负对数似然（Gaussian NLL），并对低浓度样本给予更高权重：

```text
weight = 1 / (concentration + WEIGHT_EPS)
```

这样做的目的，是在样本数很少时尽量稳定训练，同时保留低浓度区间的分辨能力。

### 4. 输出概率

网络输出的是 `log(concentration)` 空间中的高斯分布。转换回浓度空间后，会得到一个移位对数正态分布。

代码会基于训练集中的参考浓度点自动构建离散 bins，并给出每个样本在这些 bins 上的概率质量。

## 输入数据要求

默认输入文件：

[`pic_RGB/roi_results/roi_detailed_statistics.csv`](/h:/000/CDs/pic_RGB/roi_results/roi_detailed_statistics.csv)

至少需要这些列：

- `filename`
- `roi_R_mean`
- `roi_G1_mean`
- `roi_B_mean`
- `roi_G1_var`

当使用背景扣除时，还需要：

- `bg_R_mean`
- `bg_G1_mean`
- `bg_B_mean`

当 `g_mode=avg` 且使用双绿通道平均时，还需要：

- `roi_G2_mean`
- `bg_G2_mean`

## 环境依赖

项目里已有基础依赖：

```bash
pip install numpy pandas matplotlib scipy scikit-learn opencv-python rawpy PySide6 joblib
```

`concentration_GS` 额外依赖 `torch`：

```bash
pip install torch
```

## 快速开始

### 训练 + 留一法评估

在仓库根目录执行：

```bash
python concentration_GS/train.py
```

默认行为：

- 读取 `pic_RGB/roi_results/roi_detailed_statistics.csv`
- 自动判断是否使用背景扣除特征
- 训练全量模型
- 执行 Leave-One-Out 评估
- 输出模型、CSV 和评估图到 `concentration_GS/amortized_output/`

### 推理

```bash
python concentration_GS/predict.py ^
  --model concentration_GS/amortized_output/amortized_posterior_model.pt ^
  --roi-csv pic_RGB/roi_results/roi_detailed_statistics.csv
```

默认会在输入 CSV 同目录下输出：

- `*_pred_concentration_gs.csv`
- `*_pred_concentration_gs_prob.csv`

## 常用参数

`train.py` 常用参数：

- `--roi-csv`
  指定训练输入 CSV。
- `--g-mode {g1,avg}`
  绿通道使用 `G1` 或 `G1/G2` 平均值。
- `--use-bg {auto,true,false}`
  是否使用背景扣除特征。
- `--hidden-dim`
  隐藏层宽度。
- `--dropout`
  Dropout 比例。
- `--learning-rate`
  AdamW 学习率。
- `--weight-decay`
  AdamW 权重衰减。
- `--epochs`
  最大训练轮数。
- `--patience`
  早停耐心轮数。
- `--min-std`
  后验标准差下界。
- `--weight-eps`
  样本权重平滑项。
- `--confidence`
  输出置信区间覆盖率，默认 `0.90`。

`predict.py` 常用参数：

- `--model`
  训练好的模型文件。
- `--roi-csv`
  待预测 CSV。
- `--out`
  summary CSV 输出路径。
- `--grid-out`
  概率 CSV 输出路径。
- `--confidence`
  推理阶段覆盖率。未指定时优先使用模型内保存的值。

## 输出文件说明

训练完成后，默认输出到：

[`amortized_output`](/h:/000/CDs/concentration_GS/amortized_output)

主要文件包括：

- `amortized_posterior_model.pt`
  PyTorch 模型包，包含网络权重、特征标准化参数、目标变换参数、参考浓度 bins、训练配置和评估指标。
- `train_posterior_summary.csv`
  全量训练集上的后验摘要结果。
- `loo_posterior_summary.csv`
  留一法评估结果。实际泛化能力优先看这个文件。
- `train_posterior_probabilities.csv`
  全量训练集对应的离散浓度 bin 概率。
- `loo_posterior_probabilities.csv`
  留一法对应的离散浓度 bin 概率。
- `posterior_evaluation.png`
  训练集拟合图、留一法拟合图，以及 LOO 概率热图。

### summary CSV 字段

以 [`loo_posterior_summary.csv`](/h:/000/CDs/concentration_GS/amortized_output/loo_posterior_summary.csv) 为例，关键列如下：

- `filename`
  样本名。
- `conc_true`
  真实浓度。
- `posterior_log_mean`
  `log(concentration)` 后验均值。
- `posterior_log_std`
  `log(concentration)` 后验标准差。
- `posterior_mean`
  浓度空间的后验均值。
- `posterior_median`
  浓度空间的后验中位数。
- `posterior_mode`
  浓度空间的后验众数。
- `posterior_std`
  浓度空间的后验标准差。
- `ci_90_low`, `ci_90_high`
  90% 置信区间上下界。若修改 `--confidence`，列名会随之变化。
- `nll`
  对真实浓度的负对数似然。越小通常越好。
- `map_bin_concentration`
  概率最大的离散浓度 bin 对应的参考浓度。
- `map_bin_probability`
  上述 bin 的概率质量。
- `residual_mean`
  `posterior_mean - conc_true`
- `abs_error_mean`
  后验均值的绝对误差。

### probability CSV 字段

- `filename`
  样本名。
- `ref_concentration`
  参考浓度点。
- `bin_left`
  bin 左边界。
- `bin_right`
  bin 右边界。
- `bin_probability`
  该样本落在该浓度 bin 内的概率质量。
- `cdf_right`
  到右边界的累计分布值。

## 如何解读结果

建议优先看留一法文件，而不是训练集文件。

- 评估精度：
  看 `loo_posterior_summary.csv` 里的 `posterior_mean` 与 `conc_true` 的偏差。
- 评估不确定性：
  看 `posterior_std` 和 `ci_*` 区间宽度。
- 评估概率分布是否集中：
  看 `map_bin_probability` 是否足够高。
- 评估概率校准：
  看 `nll` 是否稳定偏低，以及热图中概率是否集中在真实浓度附近。

## 与岭回归版本的区别

相对 [concentration](/h:/000/CDs/concentration)，本方法的主要区别是：

- 岭回归输出的是点预测；这里输出的是条件后验分布。
- 岭回归更适合直接回归；这里更适合表达预测不确定性。
- 岭回归的 LOO 文件关注回归误差；这里同时保留了概率质量和负对数似然。

## 注意事项

- 当前样本量很小，后验分布的平滑程度会明显受超参数影响。
- `LOO` 是当前最重要的评估结果，不要只看训练集拟合。
- 如果输入 CSV 缺少训练时使用的背景列，推理会报错。
- 如果训练文件名不能解析出浓度数值，训练脚本不会把该样本纳入训练。

## 相关文件

- [config.py](/h:/000/CDs/concentration_GS/config.py)
- [train.py](/h:/000/CDs/concentration_GS/train.py)
- [predict.py](/h:/000/CDs/concentration_GS/predict.py)
- [physics_features.py](/h:/000/CDs/concentration/physics_features.py)
