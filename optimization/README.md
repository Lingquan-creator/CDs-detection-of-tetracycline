# optimization

本目录保存当前 ln-only 输入向量和超参数寻优流程。它使用 `dataset/train` 作为训练集，并分别评估：

- `optimization/LOO`: 在 `dataset/train` 上做留一法，指标为 LOO 结果。
- `optimization/test`: 使用 `dataset/train` 训练，使用 `dataset/test` 测试。

每个评估模式下直接分为三类模型：

```text
exp/
gs/
ridge/
```

## 主要文件

- `ln_hyperopt.py`: 统一寻优实现，负责构造 ln 特征、评估 Ridge/EXP/GS、记录所有组合。
- `LOO/*/run_ln_hyperopt.py`: LOO 模式的模型族入口。
- `test/*/run_ln_hyperopt.py`: train->test 模式的模型族入口。

## 输入数据

```text
dataset/train/roi_results/roi_detailed_statistics.csv
dataset/test/roi_results/roi_detailed_statistics.csv
```

统一特征设置：

```text
g_mode = avg
use_bg = true
bg_correction = subtract
clip_min = 0.1
```

## 允许的输入向量

当前寻优只允许 `concentration_regression.physics_features.build_feature_frame()` 可生成的 ln 特征：

```text
ln_R_sig
ln_G_sig
ln_B_sig
ln_RGB_sum
ln_RG_sum
ln_RB_sum
ln_GB_sum
ln_R_over_G
ln_B_over_G
ln_RG_over_GB
ln_RB_over_GB
```

明确不使用非 ln 特征，例如 `R_sig/G_sig/B_sig`、`RGB_sum`、`Ratio_*`、`Delta_*`、`*_chroma`、`exposure_time` 或分布拟合特征。

## 寻优流程

每个模型族、每个评估模式使用同一结构：

1. 用当前参数扫描候选输入向量。
2. 固定最佳输入向量扫描超参数。
3. 交替循环多轮，把每次尝试追加到 CSV。
4. 按 `r2` 降序、`rmse` 升序、`mae` 升序排序。

Ridge 优先复用 `concentration_regression` 的特征和训练逻辑；EXP 优先复用 `concentration_EXP` 的响应拟合和反演逻辑；GS 优先复用 `concentration_GS` 的模型训练、LOO 和后验预测逻辑。

## 运行

示例：

```bash
python optimization/test/ridge/run_ln_hyperopt.py
python optimization/test/exp/run_ln_hyperopt.py
python optimization/test/gs/run_ln_hyperopt.py
python optimization/LOO/ridge/run_ln_hyperopt.py
python optimization/LOO/exp/run_ln_hyperopt.py
python optimization/LOO/gs/run_ln_hyperopt.py
```

GS 训练耗时较长，LOO 模式历史运行曾使用较小训练预算筛选，例如 `epochs=80`、`patience=20`；正式配置以对应输出 CSV 的 `hyperparams` 字段为准。

## 输出

每个子目录会生成 `OUTPUT/`：

```text
optimization/test/ridge/OUTPUT/
optimization/test/exp/OUTPUT/
optimization/test/gs/OUTPUT/
optimization/LOO/ridge/OUTPUT/
optimization/LOO/exp/OUTPUT/
optimization/LOO/gs/OUTPUT/
```

每个 `OUTPUT/` 包含：

- `input_vector_trials.csv`: 输入向量筛选记录。
- `parameter_trials.csv`: 超参数扫描记录。
- `all_trials.csv`: 所有输入向量和超参数组合的完整记录。
- `all_trials_ranked.csv`: 按指标排序后的完整记录。
- `best_summary.csv`: 最佳输入向量、超参数和指标。
- `best_summary.txt`: 便于人工查看的最佳结果。
- `best_predictions.csv`: 最佳配置下的预测结果。

所有记录至少包含 `model_family`、`eval_mode`、`stage`、`iteration`、`feature_names`、`hyperparams`、`r2`、`rmse`、`mae`、`n_train` 和 `n_eval`。

## 后续整理

当前最佳组合会由：

```bash
python OUTPUT/run_best_models_and_recovery.py
```

整理到：

```text
concentration_regression/rgb_fit_output/best_optimized_combinations.csv
concentration_EXP/exp_fit_output/best_optimized_combinations.csv
concentration_GS/amortized_output/best_optimized_combinations.csv
```
