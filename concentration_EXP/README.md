# 指数拟合浓度反演 (Exponential Fitting)

基于指数模型 `I = a * exp(b * C) + c` 从 ROI RGB 统计量反演化学浓度。

## 📋 概述

本目录包含简化的指数拟合流程，从 `concentration` 目录迁移而来，专注于指数模型的拟合和反演。

### 核心原理

根据比尔-朗伯定律，光强 $I$ 随浓度 $C$ 呈指数衰减：

$$I = a \cdot e^{b \cdot C} + c$$

其中：
- $a$: 幅度参数
- $b$: 衰减系数（通常为负值）
- $c$: 基线强度（背景+暗电流）

拟合后可反推浓度：

$$C = \frac{\ln\left(\frac{I - c}{a}\right)}{b}$$

## 🚀 快速开始

### 安装依赖

```bash
pip install numpy pandas scipy
```

### 基本用法

```bash
# 使用默认配置拟合
python concentration_EXP/fit_exponential.py

# 指定输入文件
python concentration_EXP/fit_exponential.py --roi-csv path/to/roi_statistics.csv

# 使用平均绿色通道进行反演
python concentration_EXP/fit_exponential.py --channel G_corr

# 调整低浓度加权参数
python concentration_EXP/fit_exponential.py --weight-eps 10.0
```

## 📁 输入要求

输入 CSV 必须包含以下列：

### 必需列
- `filename`: 文件名（文件名stem必须是浓度数值，如 `12.5.dng`）
- `roi_R_mean`, `roi_G1_mean`, `roi_B_mean`: ROI 区域各通道均值
- `bg_R_mean`, `bg_G1_mean`, `bg_B_mean`: 背景区域各通道均值
- `roi_G1_var`: ROI 区域 G1 通道方差

### 可选列（推荐）
- `roi_G2_mean`, `bg_G2_mean`: 第二个绿色通道（Bayer 阵列传感器）
- `roi_G2_var`: G2 通道方差

默认输入路径：`pic_RGB/roi_results/roi_detailed_statistics.csv`

## 📊 输出文件

所有输出保存在 `concentration_EXP/exp_fit_output/` 目录：

### 1. `exp_fit_params.csv`
每个通道的拟合参数：

| channel | a | b | c | r2 | rmse |
|---------|---|---|---|-------|------|
| R_corr | 245.3 | -0.0123 | 10.5 | 0.9876 | 2.34 |
| G1_corr | 189.7 | -0.0145 | 8.2 | 0.9912 | 1.87 |
| ... | ... | ... | ... | ... | ... |

### 2. `exp_fit_predictions.csv`
各通道在训练集上的拟合质量：

| filename | concentration_uM | R_corr | R_corr_pred | G1_corr | G1_corr_pred | ... |
|----------|------------------|--------|-------------|---------|--------------|-----|
| 10.0.dng | 10.0 | 123.4 | 122.8 | 98.5 | 97.9 | ... |

### 3. `<input>_pred_concentration_exp.csv`
最终浓度预测结果：

| filename | conc_pred | conc_true | residual |
|----------|-----------|-----------|----------|
| 10.0.dng | 10.23 | 10.0 | 0.23 |
| 25.0.dng | 24.87 | 25.0 | -0.13 |

## ⚙️ 参数调优

### `--channel` (默认: `G1_corr`)
选择用于最终浓度反演的通道：
- `G1_corr`: 第一个绿色通道（单绿传感器）
- `G_corr`: 平均绿色通道（双绿传感器，推荐）
- `R_corr`, `B_corr`: 红/蓝通道

**建议**：绿色通道通常有最好的信噪比。双绿传感器使用 `G_corr`。

### `--weight-eps` (默认: `5.0`)
低浓度加权参数。权重 $w_i = 1/\sqrt{C_i + \epsilon}$

- **较小值** (1.0-3.0): 强调低浓度拟合，可能在高浓度段产生更大误差
- **中等值** (5.0-10.0): 平衡的权重分布（推荐）
- **较大值** (>20.0): 接近均匀加权，可能牺牲低浓度精度

## 📈 评估指标

### R² (决定系数)
- 范围: (-∞, 1]
- 值越接近 1 表示拟合越好
- 通常期望 > 0.95

### RMSE (均方根误差)
- 单位: µM (微摩尔)
- 值越小越好
- 应与浓度量程对比（如 RMSE < 5% 最大浓度）

## 🔧 常见问题

### Q: 拟合失败或精度很低
**可能原因**：
1. 数据点太少（<3 个浓度）
2. 浓度分布不均（集中在某个范围）
3. 信噪比太低或背景扣除不准确

**解决方案**：
- 增加浓度梯度样本（建议至少 5-7 个点）
- 确保浓度跨越感兴趣的范围（如 1-100 µM）
- 检查 ROI 和背景区域选择是否合理

### Q: 某些通道 R² 很高，但另一些很低
**原因**：
- 不同通道对浓度变化的响应灵敏度不同
- 某些通道可能饱和或信号太弱

**建议**：
- 选择 R² 最高的通道用于反演
- 通常绿色通道效果最好

### Q: 预测浓度出现负值或 NaN
**原因**：
- 测量强度低于拟合的基线 $c$
- 参数 $a$ 或 $b$ 不合理

**解决方案**：
- 代码会自动将负值裁剪为 0
- 检查拟合参数是否物理合理（$a > 0$, $b < 0$, $c \geq 0$）

## 📚 相关文档

- 完整特征工程和岭回归方法：见 `concentration/` 目录
- ROI 提取流程：见 `pic_RGB/` 目录
- 物理原理详解：见 `concentration/METHODOLOGY.md`

## 🔬 技术细节

### 拟合算法
使用 `scipy.optimize.curve_fit` 进行非线性最小二乘拟合。

**参数化技巧**：
内部使用重参数化 $y = (i_0 - c) \cdot e^{bx} + c$，其中 $i_0 \geq \max(y)$。
这样可以确保逆函数不会产生负浓度（因为 $I(C=0) = i_0 \geq$ 任何观测值）。

### 数值稳定性
- 指数项限制在 $[-700, 50]$ 范围内避免溢出
- 对数项中的比值限制在 $[10^{-12}, 1]$ 避免 log(0)

## 📝 文件说明

```
concentration_EXP/
├── config.py              # 配置参数
├── physics_features.py    # 浓度提取工具
├── fit_exponential.py     # 主程序：拟合和反演
├── README.md             # 本文档
└── exp_fit_output/       # 输出目录（自动创建）
    ├── exp_fit_params.csv
    ├── exp_fit_predictions.csv
    └── *_pred_concentration_exp.csv
```

## 版本历史

- **v1.0** (2026-03-02): 从 `concentration` 目录迁移指数拟合代码
  - 简化为单一模型（仅指数拟合）
  - 保留核心功能和参数调优选项
  - 移除岭回归和复杂特征工程
