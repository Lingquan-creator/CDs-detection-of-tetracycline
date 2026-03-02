# Physics Features 模块统一说明

## 概述

`physics_features.py` 模块现已在三个浓度预测目录中统一：

- `concentration/physics_features.py` - 原始完整版本
- `concentration_EXP/physics_features.py` - 指数拟合版本（已更新）
- `concentration_GS/physics_features.py` - 高斯摊销后验版本（新创建）

## 统一内容

所有三个版本现在包含相同的功能：

### 核心函数

1. **`extract_concentration_from_name(name: str) -> Optional[float]`**
   - 从文件名中提取浓度值
   - 例：`"12.5.dng"` → `12.5`

2. **`build_feature_frame(df, g_mode, use_bg, clip_min) -> DataFrame`**
   - 从 ROI 统计数据构建物理特征
   - 支持背景扣除模式和原始模式
   - 支持单绿通道 (G1) 或双绿平均 (avg)

3. **`build_feature_vector(...) -> ndarray`**
   - 为单个样本创建特征向量
   - 用于实时推理

4. **`feature_names(use_bg: bool) -> Tuple[str, ...]`**
   - 返回活跃的特征名称列表

5. **`has_background_columns(df, g_mode) -> bool`**
   - 检查 DataFrame 是否包含所需的背景列

### 特征集

#### 原始特征（无背景扣除）
- `ln_R`, `ln_G`, `ln_B`: 对数变换的强度
- `Ratio_RG`, `Ratio_BG`: 通道比率
- `Delta_RG`: 归一化颜色差异
- `Var_G`: 绿色通道方差

#### 背景扣除特征
- `ln_R_sig`, `ln_G_sig`, `ln_B_sig`: 背景扣除后的对数强度
- 其余特征相同

### 常量

- `FEATURE_NAMES_RAW`: 原始特征名称
- `FEATURE_NAMES_BG`: 背景扣除特征名称
- `DEFAULT_G_MODE = "avg"`: 默认绿色通道模式
- `_EPS = 1e-6`: 防止 log(0)
- `_CLIP_MIN = 0.1`: 背景扣除后的最小信号值

## 更改日志

### concentration_GS
- ✅ 创建独立的 `physics_features.py`
- ✅ 更新 `train.py` 导入语句（移除对 `concentration` 目录的依赖）
- ✅ 更新 `predict.py` 导入语句

### concentration_EXP
- ✅ 扩展 `physics_features.py` 以包含完整功能
- ✅ 添加 `build_feature_frame()`, `feature_names()`, `has_background_columns()` 等函数

### concentration
- ✅ 更新文档字符串，说明其作为统一模块的角色

## 使用示例

### 基本用法

```python
from physics_features import build_feature_frame, extract_concentration_from_name

# 提取浓度
conc = extract_concentration_from_name("12.5.dng")  # 12.5

# 构建特征
features = build_feature_frame(
    df,
    g_mode="avg",      # 使用平均绿色通道
    use_bg=True,       # 启用背景扣除
    clip_min=0.1       # 最小信号值
)
```

### 高级用法

```python
from physics_features import build_feature_vector

# 单个样本特征构建
feature_vec = build_feature_vector(
    r=123.4, g1=98.5, b=45.2,
    var_g1=205.0,
    bg_r=10.0, bg_g1=8.0, bg_b=5.0,
    g_mode="g1",
    use_bg=True
)
```

## 兼容性

所有三个模块现在完全兼容，可以互换使用。每个目录都有独立的副本，避免跨目录依赖。

## 物理背景

基于**比尔-朗伯定律**：

$$I = I_0 \cdot e^{-\epsilon l C}$$

因此 $\ln(I)$ 与浓度 $C$ 应呈线性关系。

## 维护建议

如需更新功能，请同时更新三个版本：
1. `concentration/physics_features.py`
2. `concentration_EXP/physics_features.py`
3. `concentration_GS/physics_features.py`

建议使用版本控制系统跟踪更改。
