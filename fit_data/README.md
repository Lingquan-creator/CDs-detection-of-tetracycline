# fit_data

本目录处理光谱 CSV 数据，包含多峰光谱拟合、光谱积分拟合，以及峰参数随浓度变化的趋势分析。

## 主要文件

- `data.csv`: 原始光谱数据。首列为波长，后续列为不同浓度的光谱强度。
- `fit_config.py`: 多峰拟合配置，包括峰数、初值、边界、多起点和输出路径。
- `spectrum_fitter.py`: 峰检测、初值生成、多峰模型、非线性最小二乘和多起点优化。
- `fit_spectrum.py`: 多峰批量拟合入口。
- `integrate_and_fit.py`: 光谱积分和积分强度-浓度模型拟合入口。
- `fit_parameters.py`: 分析 `results_spec.csv` 中峰参数随浓度变化的趋势。
- `plot_fit_results.py`: 按浓度重建拟合曲线并输出单张图。
- `algorithm.md`: 更完整的代码和算法说明。

## 输入格式

默认输入：

```text
fit_data/data.csv
```

当前配置：

```text
SKIP_ROWS = 1
X_COLUMN = 1
Y_COLUMNS = "all"
```

也就是第一行为列名，第 1 列为波长 nm，后续列为不同浓度或样品的强度。列名会用于识别浓度，例如 `0.2μM`、`20μM`。

## 多峰拟合

运行：

```bash
python fit_data/fit_spectrum.py
```

当前默认关键参数：

```text
N_PEAKS = 3
FIT_K = True
N_MULTI_START = 3
ALLOW_NEGATIVE_A = False
SIGMA_E_MIN = 0.03
SIGMA_E_MAX = 0.8
SAVE_CSV = True
SAVE_FIGURE = True
SAVE_RESULTS = False
```

模型为多个能量域峰加背景：

```text
I(lambda) = B + sum(I_peak_m(lambda))
```

每个峰参数：

```text
A_prime, k, lambda0_nm, sigma_E_eV
```

输出：

- `fit_data/results/results_spec.csv`: 每个浓度的多峰参数和拟合指标。
- `fit_data/results/results_spec.png`: 所有样品拟合总览图。
- `fit_data/results/results_spec.txt`: 文本摘要，仅当 `SAVE_RESULTS=True` 时生成。

## 光谱积分拟合

运行：

```bash
python fit_data/integrate_and_fit.py
```

流程：

1. 读取 `data.csv`。
2. 绘制所有浓度的光谱。
3. 在默认 `400-700 nm` 内积分。
4. 去掉 0 浓度点。
5. 拟合指数模型：

   ```text
   area(C) = a * exp(b * C) + c
   ```

6. 输出 R2、RMSE 和 AIC。

输出：

- `fit_data/results/results_integrated_intensity.csv`
- `fit_data/results/results_intensity_models.csv`
- `fit_data/fit_plots/spectra_by_concentration.png`
- `fit_data/fit_plots/integrated_intensity_fit.png`

## 参数趋势分析

运行：

```bash
python fit_data/fit_parameters.py
```

该脚本读取 `results_spec.csv`，对每个峰的参数随浓度变化作图并拟合趋势。默认忽略浓度 `0` 和 `0.2`。

输出：

- `fit_data/results/results_parameter.csv`
- `fit_data/parameter_plots/`

## 单浓度拟合图

运行：

```bash
python fit_data/plot_fit_results.py
```

输出：

- `fit_data/fit_plots/fit_<concentration>.png`

## 推荐运行顺序

```bash
python fit_data/fit_spectrum.py
python fit_data/integrate_and_fit.py
python fit_data/fit_parameters.py
python fit_data/plot_fit_results.py
```

## 依赖

```bash
pip install numpy pandas matplotlib scipy
```
