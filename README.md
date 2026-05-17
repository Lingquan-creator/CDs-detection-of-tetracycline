# 手机 DNG 图像荧光浓度反演工作区

本工作区用于从手机 DNG 原始图像中提取样品 ROI/BG 信号，并建立浓度反演模型。当前主线流程为：

```text
DNG 图像 -> ROI/BG 统计 -> ln 物理特征 -> 输入向量/超参数优化 -> Ridge/EXP/GS 预测 -> 回收率和 RSD 汇总
```

项目仍以 DNG RAW 数据为基础，`pic_RGB` 负责提取 Bayer 四通道统计；`concentration_regression`、`concentration_EXP` 和 `concentration_GS` 分别保存三类模型；`optimization` 负责只使用 ln 特征的寻优；`OUTPUT` 保存训练集、测试集和留一法的最终预测与回收率结果。

## 目录结构

```text
application/                 桌面应用和当前最佳模型打包结果
article/                     当前工作区中用于整理文章材料的 CSV 和图片副本
concentration_EXP/           响应函数拟合和反演模型
concentration_GS/            PyTorch 摊销后验浓度模型
concentration_regression/    物理特征 + Ridge 回归模型
dataset/                     DNG 数据、ROI 缓存和 ROI 统计表
fit_data/                    光谱 CSV 峰拟合与积分拟合分析
optimization/                ln 输入向量和超参数寻优
OUTPUT/                      最终训练/测试/LOO 预测、回收率、RSD 输出
pic_RGB/                     DNG/RAW 读取、ROI/BG 选择和统计输出
requirements.txt             根工作区依赖
```

更细的说明见各目录下的 `README.md` 和 `algorithm.md`。

## 环境

建议在仓库根目录使用当前虚拟环境：

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

也可以直接用：

```bash
./.venv/bin/python <script.py>
```

`torch` 的 CPU/GPU 版本可能需要按本机 CUDA 环境调整。

## 数据约定

当前主要数据集为：

```text
dataset/train/roi_results/roi_detailed_statistics.csv
dataset/test/roi_results/roi_detailed_statistics.csv
```

文件名通常按 `食品_批次_浓度` 或类似结构解析，例如 `chicken-1-30.dng`、`egg-3-9.dng`。测试集中曾将误写的 `egg-3-8` 修正为 `egg-3-9`，相关 CSV/JSON 已同步。

ROI 输出的核心文件：

```text
roi_results.json
roi_selection_cache.json
roi_detailed_statistics.csv
*_roi.png
```

## 推荐流程

提取或刷新 DNG ROI 统计：

```bash
PIC_RGB_DATA_DIR=dataset/test \
PIC_RGB_OUTPUT_DIR=dataset/test/roi_results \
python pic_RGB/roi_processor.py
```

运行 ln 输入向量和超参数寻优：

```bash
python optimization/test/ridge/run_ln_hyperopt.py
python optimization/test/exp/run_ln_hyperopt.py
python optimization/test/gs/run_ln_hyperopt.py
python optimization/LOO/ridge/run_ln_hyperopt.py
python optimization/LOO/exp/run_ln_hyperopt.py
python optimization/LOO/gs/run_ln_hyperopt.py
```

整理当前最佳组合、重新训练并输出 train/test/LOO 预测与回收率：

```bash
python OUTPUT/run_best_models_and_recovery.py
```

该脚本会把最佳模型同步到：

```text
concentration_regression/rgb_fit_output/best_train|best_test|best_loo/
concentration_EXP/exp_fit_output/best_train|best_test|best_loo/
concentration_GS/amortized_output/best_train|best_test|best_loo/
```

最佳输入向量和超参数索引保存在：

```text
concentration_regression/rgb_fit_output/best_optimized_combinations.csv
concentration_EXP/exp_fit_output/best_optimized_combinations.csv
concentration_GS/amortized_output/best_optimized_combinations.csv
```

## 最终输出

`OUTPUT/train`、`OUTPUT/test` 和 `OUTPUT/LOO` 下分别包含：

```text
predict/       Ridge 预测
predict_exp/   EXP 预测
predict_gs/    GS 后验预测
results/       每种食物、每个浓度的平均回收率和 RSD
```

回收率汇总表和图片：

```text
model_recovery_summary.csv
model_recovery_heatmap_table.csv
figures/model_recovery_heatmap_rsd.png
```

## 桌面应用

从源码运行：

```bash
./.venv/bin/python application/app.py
```

应用支持打开 `.dng` 图像、手动选择 ROI/BG、选择模型包和预测模型。当前已加入 `current_work` 模型包，包含本轮优化后的 Ridge、EXP 和 GS 模型。
