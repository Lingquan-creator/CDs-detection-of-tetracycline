# fit_data

`fit_data/` 用于“量子点多峰光谱拟合 → 参数随浓度变化建模（标定曲线）→ 结果可视化/导出”这一套流程。

## 目录结构（当前仓库）

- `fit_spectrum.py`：批量光谱拟合主程序（读入 `y4.csv`，逐列拟合，输出结果 CSV/图）
- `spectrum_fitter.py`：拟合核心算法（峰检测、参数初始化、非线性最小二乘、多起点拟合等）
- `fit_config.py`：拟合配置（输入文件名、峰数、边界、是否平滑/去异常点、输出文件名等）
- `fit_parameters.py`：参数可视化 + 参数-浓度拟合（生成 `parameter_plots/`，并输出“拟合关系表”CSV）
- `y4.csv`：示例/实验数据（第一列为波长，后续每列为一个样品/浓度的强度）
- `parameter_plots/`：由 `fit_parameters.py` 生成的图
- `results_spec.csv`、`results_spec.png`：一次运行产生的拟合结果示例（文件名由配置决定）
- `改进说明.md`、`模型配置说明.md`、`拟合分析报告.md`、`可视化使用说明.md`：过程说明与分析文档

> 说明：部分文档里仍沿用旧文件名 `fit_spectrum_general.py` / `plot_parameters.py`，对应当前目录下的 `fit_spectrum.py` / `fit_parameters.py`。

## 环境依赖

- Python 3.10+（建议）
- 依赖包：`numpy`、`scipy`、`matplotlib`（可视化还需要 `pandas`）

安装示例：
```bash
pip install numpy scipy matplotlib pandas
```

## 1) 批量拟合（生成每个浓度的峰参数）

1. 配置 `fit_config.py`
   - `DATA_FILE`：输入数据文件名（默认 `y4.csv`）
   - `SKIP_ROWS`：跳过头部行数（包含表头）
   - `X_COLUMN`：波长列（从 1 开始）
   - `N_PEAKS`：拟合峰数量
   - `PEAK_PARAMS` / `MANUAL_INITIAL_GUESSES`：固定参数/初值策略
   - `ALLOW_NEGATIVE_A`：是否允许出现负峰（吸收/差分）
   - `SMOOTH_DATA`、`REMOVE_OUTLIERS`：预处理开关
   - `N_MULTI_START`：多起点拟合次数（>1 更稳但更慢）
   - `CSV_OUTPUT_FILE`、`FIGURE_FILE`：输出文件名

2. 在 `fit_data/` 目录运行：
```bash
python fit_spectrum.py
```

3. 主要输出（取决于 `fit_config.py` 的开关与文件名）：
   - 结果表：`CSV_OUTPUT_FILE`（示例：`results_spec.csv`）
   - 拟合对比图：`FIGURE_FILE`（示例：`results_spec.png`）
   - （可选）文本摘要：`RESULTS_FILE`

结果 CSV 的字段包含：
- 每个样品：`Concentration`、`R_squared`、`RMSE`、`Relative_RMSE_%`、`Background_B`
- 每个峰：`Peak{i}_A_prime`、`Peak{i}_k`、`Peak{i}_lambda0_nm`、`Peak{i}_sigma_E_eV`、`Peak{i}_peak_position_nm`、`Peak{i}_E0_eV`、`Peak{i}_max_intensity`、`Peak{i}_integrated_intensity`

## 2) 参数可视化 + 生成“参数-浓度”拟合表

`fit_parameters.py` 用于：
- 读取上一步结果 CSV
- 绘制“参数 vs 浓度”的曲线图（输出到 `parameter_plots/`）
- 对每个参数/每个峰，拟合一条“参数-浓度关系”（输出为一张 CSV 表，后续可用于反解浓度）

使用方法：
1. 打开 `fit_parameters.py`，把 `INPUT_CSV` 改成你的拟合结果 CSV（例如 `results_spec.csv` 或你在 `fit_config.py` 配置的 `CSV_OUTPUT_FILE`）。
2. 在 `fit_data/` 目录运行：
```bash
python fit_parameters.py
```

输出：
- `parameter_plots/*.png`：各参数随浓度变化趋势图 + 汇总图
- `LINEAR_FIT_CSV`（脚本内配置的输出名）：每个参数/峰的拟合类型、公式、系数、R2、RMSE、有效浓度范围等

关于“用标定曲线反解浓度”的写法，见仓库根目录的 `量子点.md`（浓度计算部分）。

## 输入数据格式说明（y4.csv）

默认假设：
- 第一行是表头：第 1 列可为空/波长名，后面每一列是样品/浓度标签（如 `0.2μM`）
- 从第 `SKIP_ROWS+1` 行开始为数值数据
- 第 `X_COLUMN` 列为波长（单位 nm）
- 其他列为同一波长下的强度

编码注意：
- 建议把 `y4.csv` 存为 UTF-8 编码；若用 ANSI/GBK，表头里的 `μ` 可能在程序输出/结果 CSV 里显示成乱码，但不影响数值拟合。
