折中方案：兼容有/无背景字段、保留 log 目标可选，同时加强低浓度与单调性
==================================================

### 目标
- 兼容两条训练线：有背景的「信号特征 + 线性目标」与无背景的「原特征 + log 目标」。
- 强化低浓度精度（样本权重）与区间单调性（后置校准），确保非负。
- 自动调参 alpha、g-mode，减少手工试错。

### 特征与目标策略
1) 背景扣除可用时（CSV 含 `bg_*_mean`）：
   - 信号 = `roi_*_mean - bg_*_mean`，裁剪到 `_CLIP_MIN` 避免 log 负值。
   - 特征：`ln_R_sig/ln_G_sig/ln_B_sig`、信号比值 `Ratio_RG/Ratio_BG`、`Var_G`。
   - 目标：直接预测浓度（identity）。
2) 无背景列时：
   - 保持现有特征 `ln_R/ln_G/ln_B` + 比值 + `Var_G`。
   - 目标：log-target 回归，推理时 exp 逆变换并裁剪非负。

### 训练流水线（建议写入 rgb_exponential_fit.py）
1) 数据加载：解析浓度，按浓度分组可留一验证（可选）。
2) 特征构建：根据是否有 `bg_*` 列选择策略 1/2；记录元数据 `use_bg`、`target_transform`、`target_eps`。
3) 样本权重：`w = 1 / (conc + eps)`（或 sqrt 版本），归一化后用于回归与校准器。
4) 模型：`StandardScaler` + `RidgeCV`（alpha 网格如 [0.01, 0.1, 0.3, 0.5, 1, 3, 10, 30]），同时扫 g-mode（g1/avg），选最优组合。
5) 预测与校准：在浓度空间拟合 `IsotonicRegression(out_of_bounds="clip")`，输入为模型输出，目标为真实浓度，使用相同样本权重；可选留一浓度验证以防过拟合。
6) 约束：预测结果裁剪到非负。
7) 产物：
   - joblib bundle：`model`、`g_mode`、`feature_names`、`use_bg`、`target_transform`、`target_eps`、`alpha`、`isotonic` 校准器。
   - CSV：预测、残差；可增加分浓度段的相对误差统计。

### 推理流水线（rgb_to_concentration.py）
1) 加载 bundle 获取 `use_bg`、`target_transform`、`target_eps`、`g_mode`、`isotonic`。
2) 特征构建：若 `use_bg` 为真且 CSV/单点提供 bg_*，用信号特征，否则退回原特征。
3) 模型预测 -> 目标逆变换（log 则 exp）-> 非负裁剪 -> 单调校准器（若存在）。
4) 批量模式输出 conc_pred/conc_true/residual；单点打印并指示 g-mode/target/use_bg。

### 配置与兼容
- `config.py`：保留 `TARGET_TRANSFORM`/`TARGET_EPS` 作为默认；增加 `USE_BG_DEFAULT=False`、`ALPHAS` 网格配置。
- 兼容旧模型：若 bundle 无 `use_bg`/`isotonic`/`target_transform`，推理端按默认（原逻辑 + 非负裁剪）运行。

### 建议的验证
- 低浓度段（≤8 uM）相对误差均值/最大值。
- 全段 RMSE / R²；残差是否单调、是否存在反转（conc_pred 随 conc_true 递增）。
- 可视化：conc_true vs conc_pred 折线（按真值排序）和残差分布。

### 下一步落地任务（按优先级）
1) 在 physics_features.py 增加背景信号特征构建（可检测列自动切换）；为单点接口添加 bg_* 可选参数。
2) 在 rgb_exponential_fit.py 实现双策略训练、样本权重、RidgeCV alpha/g-mode 搜索、Isotonic 校准与元数据持久化。
3) 在 rgb_to_concentration.py 应用元数据驱动的特征/逆变换/校准与非负裁剪。
4) 更新 README.md 记录两种模式、权重与校准说明；提供训练/推理示例（有背景、无背景各一）。
