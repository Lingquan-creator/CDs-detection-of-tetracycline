# 浓度反演流程使用说明

基于物理先验 + 岭回归，将 ROI RGB 统计量映射为浓度。本版本（v4）增强了低浓度精度与单调性约束，并提供了防过拟合的评估机制。

## 核心改进 (v4)
1. **真实评估 (LOO-CV)**：输出留一法交叉验证结果 (`_LOO_CV.csv`)，杜绝因极高拟合精度产生的“过拟合错觉”，真实反映模型对未知样品的预测能力。
2. **低浓度优化**：通过 `WEIGHT_EPS` (默认50.0) 平滑样本权重，并使用大数值 Alpha 正则化，有效抑制低浓度段因噪声导致的剧烈震荡。
3. **背景扣除 (强制)**：始终采用背景扣除模式 (`Signal = ROI - BG`)，直接建立信号与浓度的映射，不再支持无背景模式。
4. **单调性约束**：集成 `IsotonicRegression`，强制输出非负且单调递增。

---

## 输入与配置详解 (Input & Configuration)

### 1. 数据输入 (CSV)
默认路径：`pic_RGB/roi_results/roi_detailed_statistics.csv`
必需列（由 `pic_RGB` 流程生成）：
- **基础列**：`filename` (必须包含浓度数值，如 `12.0.dng`)
- **ROI 统计**：`roi_R_mean`, `roi_G1_mean`, `roi_B_mean`, `roi_G1_var`
- **背景统计 (必需)**：`bg_R_mean`, `bg_G1_mean`, `bg_B_mean`
  - *注：必须存在背景列，模型会自动进行信号扣除 (Signal = ROI - BG)。*

### 2. 参数配置 (`config.py`)
主要超参数均在 `concentration/config.py` 中定义，修改此处即可影响训练行为：

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `ALPHAS` | `(10000, 100000.0)` | 正则化网格。强制使用高正则化项以应对共线性并平滑模型。 |
| `WEIGHT_EPS` | `50.0` | 权重平滑项。权重 $w = 1/(y + \epsilon)$。值越大，低浓度样本被“特殊对待”的程度越低，曲线越平滑。 |
| `TARGET_TRANSFORM` | `"identity"` | 目标变换。默认直接预测浓度，配合背景扣除物理模型。 |
| `CLIP_MIN` | `0.1` | 信号阈值。防止 `log(Signal)` 出现负无穷。 |

---

## 输出产物 (Outputs)

所有输出文件均位于 `concentration/rgb_fit_output/` 目录：

### 1. 模型文件
- **`concentration_model_v1.pkl`**
  - **内容**：Python `dict` 对象，包含：
    - `model`: 训练好的 Pipeline (Scaler + Ridge)。
    - `isotonic`: 训练好的保序回归校准器 (IsotonicRegression)。
    - `g_mode`, `use_bg`, `feature_names`: 训练时选定的最佳策略元数据。
    - `sklearn_version`, `python_version`, `bundle_created_utc`: 训练环境信息（用于排查版本一致性问题）。
  - **用途**：推理脚本 `predict.py` 的主要输入。
  - **版本兼容提示**：模型文件使用 joblib/pickle 序列化，建议训练与推理使用相同的 scikit-learn 版本。
    - 出现版本不一致警告时：优先重新训练（`python concentration/rgb_exponential_fit.py --method ridge`）。
    - 若仅需减少/消除重复警告（并理解风险）：优先重新训练并重新保存模型 bundle。

### 1.1 指数拟合输出（无需模型文件）
- **`rgb_fit_params.csv`**：指数拟合参数（按通道）。
- **`rgb_fit_predictions.csv`**：指数拟合曲线在各浓度点的强度预测（用于检查拟合质量）。
- **`pic_RGB/roi_results/roi_detailed_statistics_pred_concentration_exp.csv`**：指数拟合反推的浓度预测结果（默认使用 `G1_corr`）。

### 2. 评估报告 (重要)
- **`ridge_predictions_LOO_CV.csv` (真实精度)**
  - **含义**：留一法交叉验证结果。每个样本的预测值是在“该样本未参与训练”的模型上跑出来的。
  - **用途**：**请看这个文件的 RMSE 和 R2 来评估模型好坏**。
- **`ridge_predictions.csv` (拟合精度)**
  - **含义**：全量数据训练后的回测结果。
  - **用途**：仅用于检查标准曲线是否“做”得足够光滑，不代表预测能力。

### 3. 分析辅助
- **`ridge_coefficients.csv`**：特征权重表，用于分析哪些颜色通道对浓度的贡献最大。

---

## 快速使用 (Quick Start)

### 训练
```bash
# 指数拟合（默认）：拟合 I = a*exp(b*C)+c 并反推浓度
python concentration/rgb_exponential_fit.py

# 岭回归训练：生成 concentration_model_v1.pkl 等
python concentration/rgb_exponential_fit.py --method ridge
```

### 推理
**批量预测 (Batch Mode)**：
```bash
# 读取 ROI 统计 CSV，生成预测结果
python concentration_regression/train.py --roi-csv pic_RGB/roi_results/roi_detailed_statistics.csv
```
*注：输入 CSV 必须包含背景统计列 (bg_*)，否则脚本会报错。*

**单点预测 (Single Mode, 调试用)**：
```bash
python concentration_regression/train.py \
  --model concentration/rgb_fit_output/concentration_model_v1.pkl \
  --r 25.5 --g1 40.2 --b 10.1 --var-g 205.0
```

---

## 目录结构
```text
concentration/
├── config.py               # [配置] 路径、超参数 (Alpha, Eps)
├── physics_features.py     # [核心] 物理特征计算逻辑
├── rgb_exponential_fit.py  # [拟合] 指数拟合(默认) / 岭回归训练(--method ridge)
├── train.py              # [训练] 训练 Ridge 模型并做 LOO 验证
├── METHODOLOGY.md          # [文档] 算法原理白皮书
└── rgb_fit_output/         # [输出] 产物目录
```
