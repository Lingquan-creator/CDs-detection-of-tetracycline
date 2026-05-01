# fit_data README

量子点光谱处理与拟合工具集，输入/输出统一至 `fit_data/results/`，便于批量运行和下游（如 RGB 浓度映射）使用。

## 数据与目录
- 原始光谱：`fit_data/data.csv`（首列波长，后续列为不同浓度光谱）
- 结果目录：`fit_data/results/`
  - `results_spec.csv` / `results_spec.txt` / `results_spec.png`：多峰拟合结果及可视化
  - `results_parameter.csv`：参数趋势拟合（线性/指数）
  - `results_integrated_intensity.csv`：400–700 nm 积分强度
  - `results_intensity_models.csv`：积分强度 vs 浓度模型摘要（AIC 选择）
  - `fit_plots/`：全谱、积分拟合等辅助图
  - `parameter_plots/`：参数趋势图（由 `fit_parameters.py` 生成）

## 核心脚本
- `fit_config.py`：全局配置（数据/结果路径、峰数、初值、边界等，已默认指向 `results/`）。
- `fit_spectrum.py`：多峰拟合批处理，产出 `results_spec.*`。
- `spectrum_fitter.py`：峰建模、优化与指标计算的核心逻辑。
- `integrate_and_fit.py`：光谱积分并对积分强度做模型选择（AIC），产出积分相关 CSV。
- `plot_fit_results.py`：根据 `results_spec.csv` 绘制各浓度拟合/原始光谱对比。
- `fit_parameters.py`：绘制/拟合参数随浓度趋势，写出 `results_parameter.csv` 与参数图。

## 运行顺序示例
```bash
cd fit_data
# 1) 多峰拟合
python fit_spectrum.py
# 2) 光谱积分 + 模型选择
python integrate_and_fit.py
# 3) 绘制拟合光谱对比
python plot_fit_results.py
# 4) 参数趋势/线性-指数拟合
python fit_parameters.py
```
所有输出将落到 `results/` 或对应图像目录（`fit_plots/`, `parameter_plots/`）。

## 与浓度映射衔接
`results_intensity_models.csv`、`results_integrated_intensity.csv`、`results_parameter.csv` 等文件可被浓度映射脚本引用，路径已统一到 `fit_data/results/`，无需额外配置。

## 依赖
- Python 3.8+
- numpy, pandas, matplotlib, scipy
