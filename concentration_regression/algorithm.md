# concentration_regression 代码说明

本目录实现基于物理特征和 Ridge 回归的浓度反演。整体思路是把 ROI/BG 的 RAW 通道统计转换成 Beer-Lambert 风格的对数光强、通道比值和差分特征，再用标准化后的 RidgeCV 预测浓度。默认目标为 `log(C + eps)`，预测后再反变换到浓度空间。

## 代码文件

- `config.py`: 定义默认路径、输出文件名、目标变换、背景校正方式、特征集合和 Ridge 正则网格。
- `physics_features.py`: 从 ROI CSV 构造物理特征，支持背景扣除、背景相除、相对背景校正和曝光时间特征。
- `train.py`: 训练入口。训练 RidgeCV，执行 Leave-One-Out，保存模型、预测、系数和评估图。
- `predict.py`: 预测入口。加载 `joblib` 模型 bundle，对新的 ROI CSV 构造同名特征并输出浓度预测。
- `rgb_fit_output/plot_comparison.py`, `rgb_fit_output/plot_ridge_results.py`: 历史结果可视化辅助脚本。

## 默认配置

当前默认配置来自 `config.py`：

```text
roi_csv = dataset/food/roi_detailed_statistics.csv
output_dir = concentration_regression/rgb_fit_output/
target_transform = log
use_bg = true
bg_correction = subtract
feature_set = compact
alphas = (10000, 100000)
weight_eps = 30.0
clip_min = 0.1
g_mode = avg
```

`TrainConfig.alpha=0.5` 只保留在 dataclass 中，实际训练默认使用 `ALPHAS`，除非命令行传入 `--alpha`。

## 输入数据

训练和预测都读取 `roi_detailed_statistics.csv`。核心列包括：

- `filename`: 训练时从中解析 `conc_true`。
- `roi_R_mean`, `roi_G1_mean`, `roi_G2_mean`, `roi_B_mean`: ROI 四通道均值。
- `bg_R_mean`, `bg_G1_mean`, `bg_G2_mean`, `bg_B_mean`: 背景四通道均值。默认 `--use-bg true` 时必须存在。
- `roi_G1_var`, `roi_G2_var`: 可选，用于 `Var_G`；缺失则为 0。
- `exposure_time`: 可选，用于曝光归一化特征。
- `roi_*_gauss_mu`, `roi_*_gauss_shape` 或 `roi_*_gamma_shape`, `roi_*_gamma_scale`: 可选，`feature_set=full` 时追加。

## 背景校正和特征构造

默认 `g_mode=avg`，绿色通道使用 G1/G2 平均。`use_bg=true` 时可选三种背景校正：

```text
subtract: R_sig = ROI - BG
divide:   R_sig = ROI / (BG + eps)
relative: R_sig = (ROI - BG) / (BG + eps)
```

所有背景校正后的信号都会裁剪到不小于 `clip_min=0.1`，避免 `log(0)` 或负值。

基础特征为：

```text
ln_R_sig, ln_G_sig, ln_B_sig
ln_R_sig_per_exp, ln_G_sig_per_exp, ln_B_sig_per_exp
Ratio_RG, Delta_RG, Ratio_BG, Delta_GB
Var_G
exposure_time, ln_exposure
```

如果 `use_bg=false`，`ln_*_sig` 会对应替换为 `ln_R/ln_G/ln_B`。

`feature_set` 控制实际训练列：

- `compact`: 默认 6 维，使用 `ln_R_sig, ln_G_sig, ln_B_sig, Ratio_RG, Delta_RG, Delta_GB`。
- `compact_exposure`: 使用 12 维曝光相关特征和 `Ratio_BG/Var_G`，不包含 `Delta_GB`。
- `ln_rgb`: 只使用 `ln_R_sig, ln_G_sig, ln_B_sig`。
- `full`: 使用 `build_feature_frame()` 生成的全部特征，包括可用分布参数变换。

## 训练流程

1. 读取 ROI CSV，检查 `filename`，解析 `conc_true`，并按浓度排序。
2. 根据 `--g-mode` 选择 `g1`、`avg`，或在 `auto` 时分别训练候选模型并比较内部 CV 分数。
3. 根据 `--use-bg` 决定是否使用背景校正；`auto` 会在背景列存在时启用。
4. 构造特征并按 `feature_set` 选择列。
5. 目标默认进行 log 变换：

   ```text
   y_target = log(C + 1e-6)
   ```

6. 样品权重为：

   ```text
   weight = 1 / (C + weight_eps)
   ```

   并归一化到平均权重为 1。
7. 训练模型：

   ```text
   StandardScaler -> RidgeCV(alphas=(10000, 100000))
   ```

8. 将训练预测从目标空间反变换回浓度：

   ```text
   C_raw = exp(y_pred) - eps
   ```

9. 将反变换后的 `C_raw` 直接作为 `conc_pred`。
10. 执行 Leave-One-Out。每折在训练子集上重新训练 RidgeCV，并直接输出该折预测。
11. 输出训练集和 LOO 的 R2、RMSE、MAE。

## 预测流程

`predict.py` 会读取模型 bundle 中保存的：

- `model`
- `feature_names`
- `g_mode`
- `target_transform`
- `target_eps`
- `use_bg`
- `bg_correction`
- `monotonic_calibration`

预测时按模型保存的特征名从新 CSV 中重建特征。如果模型需要背景列但新 CSV 缺失，代码会退回非背景特征并打印警告；但如果保存的 `feature_names` 仍要求背景特征，最终会报缺失特征错误。

`predict.py` 还处理 scikit-learn 版本不一致警告，并支持 `--strict-sklearn-version`、`--suppress-sklearn-version-warning` 和 `--resave-model`。

## 输出文件

默认输出目录：

```text
concentration_regression/rgb_fit_output/
```

训练输出：

- `concentration_model_v1.pkl`: `joblib` 模型 bundle。
- `ridge_predictions.csv`: 训练集 `conc_pred_raw`、最终 `conc_pred` 和残差；当前二者相同。
- `ridge_predictions_loo.csv`: Leave-One-Out 预测和残差。
- `ridge_coefficients.csv`: Ridge 系数，列名为保存的特征名。
- `concentration_model_v1_evaluation.png`: 训练集和 LOO 预测散点图。

预测输出：

- `<roi_csv_stem>_pred_concentration.csv`
- `<roi_csv_stem>_pred_concentration.png`

## 使用方式

```bash
python concentration_regression/train.py
python concentration_regression/train.py --feature-set compact --use-bg true --bg-correction subtract
python concentration_regression/predict.py --model concentration_regression/rgb_fit_output/concentration_model_v1.pkl --roi-csv dataset/food/roi_detailed_statistics.csv
```
