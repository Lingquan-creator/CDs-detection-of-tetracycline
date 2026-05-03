# fit_data 算法说明

本目录处理光谱 CSV 数据，包含两条路线：多峰光谱拟合，以及对指定波段积分后拟合“积分强度-浓度”关系。

## 主要代码

- `fit_spectrum.py`: 批量读取 `data.csv`，对每个样品列做多峰光谱拟合。
- `spectrum_fitter.py`: 多峰模型、峰检测、初值生成、非线性最小二乘拟合等核心算法。
- `fit_config.py`: 光谱拟合配置，包括峰数、初值、边界、输入输出文件。
- `plot_fit_results.py`: 根据 `results_spec.csv` 重建拟合曲线并逐浓度画图。
- `fit_parameters.py`: 读取多峰拟合参数，分析参数随浓度变化的趋势。
- `integrate_and_fit.py`: 对光谱积分，并拟合积分强度和浓度之间的模型。

## 算法输入

算法层面的输入是：

- 波长数组 `lambda_nm`。
- 每个浓度对应的一列光谱强度 `I(lambda)`。
- 峰模型配置：峰数、峰位初值、能量宽度、背景、参数边界。
- 积分拟合时使用的波段范围。

实际默认输入文件：

- `fit_data/data.csv`

默认约定：

- 第 1 列是波长。
- 其余列是不同浓度或样品的光谱强度。
- 表头用于识别样品名或浓度。

## 多峰光谱拟合流程

1. `fit_spectrum.py` 读取 `fit_data/data.csv`，得到波长列和多个强度列。
2. 对每个样品列过滤 NaN/Inf。
3. 可选进行平滑和离群点移除。
4. 构造多峰模型。每个峰使用能量域高斯形式，并包含波长相关强度项：

   ```text
   I(lambda) = B + sum_m A_m * exp(-k_m * lambda) *
               exp(-0.5 * ((E(lambda) - E0_m) / sigma_E_m)^2)
   E(lambda) = 1239.8 / lambda
   ```

5. 自动峰检测或使用 `fit_config.py` 中的手动初值。
6. 用 `scipy.optimize.least_squares` 做非线性最小二乘拟合。
7. 如果 `FIT_K=True`，先固定 `k=0` 做第一阶段拟合，再释放 `k` 做第二阶段拟合。
8. 计算每个样品的 R2、RMSE、背景和每个峰的参数。
9. 保存批量结果 CSV 和总览图。

## 光谱积分拟合流程

`integrate_and_fit.py` 走更简单的积分路线：

1. 读取 `fit_data/data.csv`。
2. 绘制所有浓度的原始光谱。
3. 在默认 `400-700 nm` 范围内对每个浓度积分：

   ```text
   area(C) = integral I(lambda, C) d lambda
   ```

4. 去掉 0 浓度点。
5. 拟合积分强度与浓度的关系，例如指数模型：

   ```text
   area(C) = a * exp(b * C) + c
   ```

6. 用 R2、RMSE、AIC 评价模型，并输出拟合图。

## 算法输出

多峰拟合的算法输出：

- 每个样品的背景强度。
- 每个峰的 `A_prime`, `k`, `lambda0_nm`, `sigma_E_eV`。
- 派生峰参数：真实峰位、中心能量、最大强度、积分强度。
- 拟合指标 R2、RMSE。

实际输出文件：

- `fit_data/results/results_spec.csv`: 多峰拟合参数表。
- `fit_data/results/results_spec.txt`: 文本摘要，只有 `SAVE_RESULTS=True` 时保存。
- `fit_data/results/results_spec.png`: 所有样品的拟合对比图。
- `fit_data/results/results_parameter.csv`: 参数-浓度趋势拟合结果。
- `fit_data/fit_plots/`: 单浓度拟合图和积分拟合图。
- `fit_data/parameter_plots/`: 参数趋势图。

光谱积分拟合的实际输出：

- `fit_data/results/results_integrated_intensity.csv`: 每个浓度的积分强度。
- `fit_data/results/results_intensity_models.csv`: 积分强度模型参数和指标。
- `fit_data/fit_plots/spectra_by_concentration.png`: 全部光谱图。
- `fit_data/fit_plots/integrated_intensity_fit.png`: 积分强度拟合图。

## 使用方式

```bash
python fit_data/fit_spectrum.py
python fit_data/integrate_and_fit.py
python fit_data/fit_parameters.py
python fit_data/plot_fit_results.py
```
