# 手机 DNG 图像荧光浓度反演工作区

本工作区用于从手机直接拍摄的 DNG 原始图像中提取样品 ROI/BG 信号，并建立荧光浓度反演模型。当前主线流程是：

```text
DNG 图像 -> ROI/BG 选择 -> RAW Bayer 四通道统计 -> 物理特征构造 -> 岭回归浓度预测
```

项目重点使用 DNG 原始图像而不是 JPEG/PNG 等压缩图像。DNG 保留更接近传感器线性响应的 Bayer RAW 数据，适合做定量分析。当前建模主线为 `concentration_regression` 中的岭回归；`concentration_EXP` 作为通道响应函数拟合基线，`concentration_GS` 作为概率后验模型实验。

## 目录结构

```text
application/                 桌面应用，集成 DNG 打开、ROI/BG 选择和浓度预测
article/                     项目说明、论文式文档和参考文献材料
concentration_regression/    当前主线：物理特征 + Ridge 回归 + 单调校准
concentration_EXP/           通道响应函数拟合浓度反演基线
concentration_GS/            PyTorch 摊销后验浓度模型
dataset/                     DNG 图像、ROI 缓存和 ROI 统计输出
fit_data/                    光谱 CSV 的峰拟合、积分拟合和参数趋势分析
parameter_scan/              Ridge 正则强度、样品权重等超参数扫描
pic_RGB/                     DNG/RAW 读取、ROI/BG 选择和特征 CSV 生成
test_input/                  输入特征组合扫描实验
requirements.txt             根工作区依赖
```

更细的实现说明见各目录下的 `README.md` 和 `algorithm.md`。

## 环境安装

建议在仓库根目录创建虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果只运行桌面应用，也可以参考：

```bash
pip install -r application/requirements.txt
```

`torch` 的 CPU/GPU 版本可能需要按本机 CUDA 环境调整；根目录 `requirements.txt` 记录的是当前代码直接需要的依赖入口。

## 数据约定

图像数据默认放在 `dataset/<dataset_name>/` 下，文件通常为 `.dng`。ROI 主流程会生成或复用以下文件：

```text
roi_results.json             每张图的 ROI/BG 几何、通道均值、元数据和源文件路径
roi_selection_cache.json     ROI/BG 选择缓存，可用于复用已有选区
roi_detailed_statistics.csv  后续浓度模型的主要输入表
```

模型训练通常依赖 `roi_detailed_statistics.csv`。该表至少需要包含 `filename`、ROI 通道均值和背景通道均值；文件名会用于解析真实浓度，例如 `0.2.dng`、`120.dng`、`chicken120.dng`。

## 推荐流程

### 1. 提取 DNG ROI 特征

默认处理 `pic_RGB/roi_config.py` 中配置的数据目录：

```bash
python pic_RGB/roi_processor.py
```

指定输入和输出目录：

```bash
PIC_RGB_DATA_DIR=dataset/mate20x \
PIC_RGB_OUTPUT_DIR=dataset/mate20x \
python pic_RGB/roi_processor.py
```

如果已有 `roi_selection_cache.json`，可以在配置中开启复用，或通过现有代码调用 `ROIProcessor(reuse_last_selection=True)` 批量更新 ROI 输出。当前 `pic_RGB` 主流程只计算 ROI/BG 统计和 6 维物理特征，不调用高斯、泊松等统计分布拟合。

### 2. 训练主线 Ridge 浓度模型

```bash
python concentration_regression/train.py
```

常用指定参数：

```bash
python concentration_regression/train.py \
  --roi-csv dataset/food/roi_detailed_statistics.csv \
  --feature-set compact \
  --use-bg true \
  --bg-correction subtract
```

默认输出在：

```text
concentration_regression/rgb_fit_output/
```

主要文件包括模型 `concentration_model_v1.pkl`、训练预测 `ridge_predictions.csv`、留一法预测 `ridge_predictions_loo.csv`、系数表和评估图。

### 3. 使用 Ridge 模型预测

```bash
python concentration_regression/predict.py \
  --model concentration_regression/rgb_fit_output/concentration_model_v1.pkl \
  --roi-csv dataset/food/roi_detailed_statistics.csv
```

预测输出会写入模型配置对应的输出目录，包含预测 CSV 和散点图。

### 4. 扫描输入特征和超参数

输入特征组合扫描：

```bash
python test_input/run_input_vector_2pow6.py
```

Ridge 超参数扫描：

```bash
python parameter_scan/run_regression_parameter_scan.py \
  --train-csv dataset/mate20x/roi_detailed_statistics.csv \
  --test-csv dataset/food/roi_detailed_statistics.csv
```

输出分别位于：

```text
test_input/OUTPUT/
parameter_scan/OUTPUT/
```

这些实验用于比较 6 维输入向量、Ridge 正则强度、样品权重形式和外部测试集表现。

## 其他模型和分析模块

### 响应函数拟合基线

```bash
python concentration_EXP/train.py
python concentration_EXP/predict.py \
  --params concentration_EXP/exp_fit_output/exp_fit_params.csv \
  --roi-csv dataset/food/roi_detailed_statistics.csv
```

该模块用于把单通道背景扣除光强拟合为指数、线性、二次等响应函数，主要作为岭回归方法的基线对比。

### GS 后验模型

```bash
python concentration_GS/train.py
python concentration_GS/predict.py \
  --model concentration_GS/amortized_output/amortized_posterior_model.pt \
  --roi-csv dataset/food/roi_detailed_statistics.csv
```

该模块输出浓度后验分布摘要，包括后验均值、中位数、众数、置信区间和离散浓度 bin 概率。

### 光谱数据拟合

```bash
python fit_data/fit_spectrum.py
python fit_data/integrate_and_fit.py
python fit_data/fit_parameters.py
python fit_data/plot_fit_results.py
```

`fit_data` 面向光谱 CSV，不直接参与手机 DNG 图像的 Ridge 主线训练。

## 桌面应用

从源码运行：

```bash
./.venv/bin/python application/app.py
```

应用支持打开 `.dng` 图像、手动选择 ROI/BG、选择手机数据集和模型类型，并输出浓度预测。可用模型类型包括：

```text
Regression    Ridge 回归模型
GS posterior  摊销后验模型
EXP           响应函数拟合基线模型
```

打包脚本：

```bash
cd application
bash build_linux.sh
```

Windows PowerShell:

```powershell
cd application
.\build_windows.ps1
```

## 当前主线特征

Ridge 主线默认使用 6 维紧凑特征：

```text
ln_R_sig
ln_G_sig
ln_B_sig
Ratio_RG
Delta_RG
Delta_GB
```

其中 `R_sig/G_sig/B_sig` 来自 ROI 均值减去局部背景均值，并裁剪到不小于 `0.1`。`G_sig` 默认取 Bayer 两个绿色通道 `G1/G2` 的平均值。

## 文档入口

- `pic_RGB/README.md`: DNG 解析、ROI/BG 选择和特征输出。
- `concentration_regression/README.md`: Ridge 训练、预测、特征和输出。
- `concentration_EXP/README.md`: 响应函数拟合基线。
- `concentration_GS/README.md`: 后验模型训练和预测。
- `fit_data/README.md`: 光谱 CSV 拟合分析。
- `application/README.md`: 桌面应用运行和打包。
- `article/artical.md`: 项目说明文档。

## 注意事项

- 当前主线不使用从 RAW 再转换得到的 DNG，而是直接使用手机拍摄保存的 DNG 图像。
- ROI 选择记录应优先保留在 `roi_selection_cache.json`，后续重复分析时可复用，减少人工误差。
- 训练、预测和扫描脚本默认会覆盖各自输出目录中的同名结果文件，运行前请确认是否需要保留旧结果。
- `dataset/` 中存在历史输出文件，如 `feature_matrix.npy`、分布特征 CSV 和 ROI 可视化 PNG；当前精简 ROI 主流程的关键输出是 `roi_results.json`、`roi_selection_cache.json` 和 `roi_detailed_statistics.csv`。
