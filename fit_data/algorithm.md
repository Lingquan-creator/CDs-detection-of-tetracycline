# fit_data 代码说明

本目录处理光谱 CSV 数据，包含两条主要路线：

1. `fit_spectrum.py` 对每个浓度的光谱曲线做多峰非线性拟合。
2. `integrate_and_fit.py` 对指定波段积分，再拟合“积分强度-浓度”关系。

另外，`fit_parameters.py` 会分析多峰拟合得到的峰参数如何随浓度变化，`plot_fit_results.py` 会按浓度重建并绘制单张拟合图。

## 代码文件

- `fit_config.py`: 多峰拟合配置，包括输入文件、峰数、初值、边界、多起点、是否拟合 `k`、输出文件等。
- `spectrum_fitter.py`: 光谱拟合核心算法。包含峰检测、初值和边界生成、单峰/多峰模型、最小二乘拟合、多起点优化和指标计算。
- `fit_spectrum.py`: 多峰批量拟合入口。读取 `data.csv` 的所有样品列，逐列拟合并保存参数。
- `integrate_and_fit.py`: 光谱积分和积分强度模型拟合入口。
- `fit_parameters.py`: 从 `results_spec.csv` 读取峰参数，绘制参数随浓度变化的趋势，并保存参数趋势拟合结果。
- `plot_fit_results.py`: 读取原始光谱和 `results_spec.csv`，为每个浓度绘制原始曲线、总拟合曲线和各单峰曲线。

## 输入数据

默认输入文件：

```text
fit_data/data.csv
```

当前配置约定：

- 第一行是列名。
- 第 1 列是波长，单位 nm。
- 后续列是不同浓度下的光谱强度，列名如 `0.2μM`, `20μM`。
- `fit_config.py` 中 `SKIP_ROWS=1`，`X_COLUMN=1`，`Y_COLUMNS="all"`。

## 多峰拟合模型

核心模型在 `spectrum_fitter.py` 中。单峰为：

```text
lambda_peak = lambda0,                  if |k| < 1e-10
lambda_peak = k * A_prime + lambda0,    otherwise

E0 = C_eV_nm / lambda_peak - 2 * sigma_E^2 / C_eV_nm * lambda_peak

I_peak(lambda) = A_prime / lambda^2
                 * exp(-((C_eV_nm / lambda - E0)^2) / (2 * sigma_E^2))
```

多峰模型为所有单峰加背景：

```text
I(lambda) = B + sum(I_peak_m(lambda))
```

其中 `C_eV_nm=1239.8` 是 eV-nm 转换常数，不是样品浓度。

每个峰的拟合参数是：

```text
A_prime, k, lambda0_nm, sigma_E_eV
```

最后一个参数是背景 `B`。

## 多峰拟合配置

当前 `fit_config.py` 的关键默认值：

```text
N_PEAKS = 3
FIT_K = True
PEAK_PARAMS = [(0.0, nan, nan)]
MANUAL_INITIAL_GUESSES = [(20000, nan, 420, nan), (35000, nan, 570, nan)]
ALLOW_NEGATIVE_A = False
K_MIN = -0.1
K_MAX = 0.1
SIGMA_E_MIN = 0.03
SIGMA_E_MAX = 0.8
B_MIN = 0.0
B_MAX_RATIO = 0.5
N_MULTI_START = 3
SMOOTH_DATA = False
REMOVE_OUTLIERS = False
SAVE_CSV = True
SAVE_FIGURE = True
SAVE_RESULTS = False
```

`PEAK_SPECIFIC_BOUNDS` 可对某个峰单独限制 `A/k/lambda0/sigma_E` 的范围。

## 多峰拟合流程

1. `fit_spectrum.py` 读取 `data.csv`，得到波长数组和所有强度列。
2. 对每个浓度列过滤 NaN/Inf。
3. 如果启用 `SMOOTH_DATA` 或 `REMOVE_OUTLIERS`，调用 `preprocess_data()` 做 Savitzky-Golay 平滑和 IQR 异常点过滤。
4. 第一阶段强制所有峰 `k=0`，根据手动初值或自动峰检测生成初始参数。
5. 调用 `scipy.optimize.least_squares(method="trf")` 做有边界非线性最小二乘拟合。
6. 如果 `FIT_K=True`，用第一阶段拟合结果作为第二阶段初值，释放所有 `k` 再拟合一次。
7. 如果 `N_MULTI_START > 1`，每阶段会尝试多个起点，包括原始起点、峰位扰动和宽度扰动，选择残差平方和最低的结果。
8. 对每个样品计算：

   ```text
   R_squared, RMSE, Relative_RMSE_%
   ```

9. 计算每个峰的派生参数：

   ```text
   peak_position_nm
   E0_eV
   max_intensity = A_prime / lambda0^2
   integrated_intensity = integral(I_peak(lambda)) d lambda
   ```

10. 按实际峰位从小波长到大波长排序后保存结果。

## 光谱积分拟合流程

`integrate_and_fit.py` 走更简化的路线：

1. 读取 `data.csv`，将每列列名解析为浓度数值。
2. 绘制所有浓度的原始光谱。
3. 在默认 `400-700 nm` 范围内积分：

   ```text
   area(C) = integral I(lambda, C) d lambda
   ```

4. 丢弃 0 浓度点。
5. 当前只拟合一个指数候选模型：

   ```text
   area(C) = a * exp(b * C) + c
   ```

   其中约束 `a >= 0` 且 `b <= 0`。
6. 计算 R2、RMSE、AIC，按 AIC 排序保存模型结果并绘图。

## 参数趋势分析

`fit_parameters.py` 读取 `results_spec.csv` 后：

1. 从 `Concentration` 字符串中解析数值浓度。
2. 自动检测结果表中包含多少个峰。
3. 对每个峰的 `A_prime`, `lambda0_nm`, `sigma_E_eV`, `peak_position_nm`, `E0_eV`, `max_intensity`, `integrated_intensity` 作图。
4. 默认忽略浓度 `0` 和 `0.2` 后拟合参数趋势。
5. 根据参数类型使用预设模型：
   - `A_prime`, `max_intensity`, `integrated_intensity`: 指数拟合。
   - `lambda0_nm`, `sigma_E_eV`, `peak_position_nm`, `E0_eV`: 线性拟合。
6. 将趋势拟合参数和指标保存到 `results_parameter.csv`。

## 输出文件

多峰拟合输出：

- `fit_data/results/results_spec.csv`: 每个浓度的多峰参数、背景和拟合指标。
- `fit_data/results/results_spec.png`: 所有样品的拟合总览图。
- `fit_data/results/results_spec.txt`: 文本摘要，仅当 `SAVE_RESULTS=True` 时生成。

积分拟合输出：

- `fit_data/results/results_integrated_intensity.csv`: 每个浓度的积分强度。
- `fit_data/results/results_intensity_models.csv`: 积分强度模型参数和指标。
- `fit_data/fit_plots/spectra_by_concentration.png`: 全部光谱图。
- `fit_data/fit_plots/integrated_intensity_fit.png`: 积分强度拟合图。

参数和单浓度可视化输出：

- `fit_data/results/results_parameter.csv`: 峰参数随浓度变化的拟合结果。
- `fit_data/parameter_plots/`: 参数趋势图。
- `fit_data/fit_plots/fit_<concentration>.png`: 每个浓度的拟合重建图。

## 使用方式

```bash
python fit_data/fit_spectrum.py
python fit_data/integrate_and_fit.py
python fit_data/fit_parameters.py
python fit_data/plot_fit_results.py
```
