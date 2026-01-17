# 浓度反演流程使用说明（中文）

基于 `improve.md` 的物理先验 + 岭回归，将 ROI RGB 统计量映射为浓度。

## 目录与功能
- `config.py`：统一默认配置（路径、alpha、g-mode 等）。
- `physics_features.py`：特征构造（ln_R/ln_G/ln_B，Ratio_RG/Ratio_BG，Var_G）。
- `rgb_exponential_fit.py`：训练模型，输出模型/预测/系数。
- `rgb_to_concentration.py`：推理，支持批量 CSV 与单样本输入。
- `rgb_fit_output/`：模型与输出文件目录。

## 环境
- Python 3.9+，依赖：pandas、numpy、scikit-learn、joblib、matplotlib（绘图可选）。
- 安装（仓库根目录执行）：
```bash
pip install -r pic_RGB/requirements.txt
```

## 数据输入
- ROI CSV 默认位置：`pic_RGB/roi_results/roi_detailed_statistics.csv`（由 ROI 流程生成）。
- 浓度值从文件名数字解析（如 `12.0.dng` -> 12.0），无法解析的样本会在训练中跳过。

## 训练
使用 `config.py` 默认值：
```bash
python concentration/rgb_exponential_fit.py
```
生成文件：
- `concentration/rgb_fit_output/concentration_model_v1.pkl`
- `concentration/rgb_fit_output/ridge_predictions.csv`
- `concentration/rgb_fit_output/ridge_coefficients.csv`

常用覆盖示例：
```bash
# 用 G1/G2 均值，并增大正则
python concentration/rgb_exponential_fit.py --g-mode avg --alpha 1.0

# 指定其他 ROI CSV
python concentration/rgb_exponential_fit.py --roi-csv path/to/roi.csv --model-out my_model.pkl
```

可在 `config.py` 持久化修改：
- `TrainConfig.alpha`：正则强度，噪声大时可增大。
- `TrainConfig.g_mode`：`g1`（默认）或 `avg`（G1/G2 均值）。
- ROI CSV 路径与输出目录。

## 推理
批量（CSV）：
```bash
python concentration/rgb_to_concentration.py --roi-csv pic_RGB/roi_results/roi_detailed_statistics.csv
```
输出默认写到同目录 `<roi_csv>_pred_concentration.csv`（可在配置或 CLI 覆盖）。

单样本手动输入：
```bash
python concentration/rgb_to_concentration.py \
  --model concentration/rgb_fit_output/concentration_model_v1.pkl \
  --r 20 --g1 45 --b 10 --var-g 300
```
若使用 G1/G2 均值，需额外传 `--g2`、`--var-g2` 并指定 `--g-mode avg`。

## 备注
- 特征设计：ln_R/ln_G/ln_B（线性化衰减）+ Ratio_RG/Ratio_BG（抗亮度漂移）+ Var_G（噪声指示）。
- CLI 参数优先级最高；修改 `config.py` 可持久化默认值。
