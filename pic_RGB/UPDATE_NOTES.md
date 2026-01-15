# GUI 手动框选更新 (2026-01)

- 自动/高级 ROI 流程已移除，改为在 OpenCV 窗口中手动画矩形选择 ROI。
- 运行 `roi_processor.py` 或 `test_roi.py` 时，每张图都会弹出 GUI；按 Enter/Space 确认、Esc/C 取消。
- `roi_config.py` 只保留路径、背景环厚度、预览缩放等参数；输出格式保持兼容。

# ROI处理器详细统计功能更新

## 更新内容

为 `roi_processor.py` 添加了详细的统计信息输出功能，生成包含每个ROI区域完整统计数据的CSV文件。

## 主要改动

### 1. `compute_roi_statistics()` 方法增强

新增以下统计量计算：

- **方差 (variance)**：除了标准差外，新增方差计算
- **p50分位数**：新增50%分位数（中位数的另一种表示）
- **饱和比例 (saturation_ratio)**：计算接近白电平（95%阈值）的像素占比
- **通道比值**：
  - R/G 比值
  - B/G 比值
  - R/B 比值
  - G1/G2 比值

### 2. 新增 `_save_detailed_statistics()` 方法

自动生成 `roi_detailed_statistics.csv` 文件，包含：

- ROI区域的所有统计量（前缀 `roi_`）
- 背景区域的所有统计量（前缀 `bg_`）
- 每个通道的均值、方差、标准差、分位数
- 饱和比例
- 通道比值

## 新增文件

### 1. 脚本文件

- **test_detailed_stats.py** - 测试详细统计功能的脚本
- **analyze_statistics.py** - 分析和可视化详细统计数据的脚本
- **详细统计说明.md** - 详细功能说明文档

### 2. 输出文件

运行 `roi_processor.py` 后会生成：

- `roi_detailed_statistics.csv` - 包含所有详细统计信息的CSV文件

## 使用方法

### 方法1：运行原有的批量处理

```bash
cd h:/000/CDs/pic_RGB
python roi_processor.py
```

这会自动生成所有输出文件，包括新的 `roi_detailed_statistics.csv`。

### 方法2：运行测试脚本

```bash
cd h:/000/CDs/pic_RGB
python test_detailed_stats.py
```

这会处理前3张图像进行快速测试。

### 方法3：分析已生成的统计数据

```bash
cd h:/000/CDs/pic_RGB
python analyze_statistics.py
```

这会读取并分析 `roi_detailed_statistics.csv`，显示各种统计摘要和质量检查结果。

## CSV列说明

### 文件标识
- `filename` - 图像文件名

### ROI区域统计（每个通道：R, G1, G2, B）
- `roi_[通道]_mean` - 均值
- `roi_[通道]_var` - 方差
- `roi_[通道]_std` - 标准差
- `roi_[通道]_median` - 中位数
- `roi_[通道]_p10` - 10%分位数
- `roi_[通道]_p50` - 50%分位数
- `roi_[通道]_p90` - 90%分位数
- `roi_[通道]_saturation_ratio` - 饱和比例

### 通道比值
- `roi_R_G_ratio` - R/G
- `roi_B_G_ratio` - B/G
- `roi_R_B_ratio` - R/B
- `roi_G1_G2_ratio` - G1/G2

### 背景区域统计
所有上述统计量都有对应的背景版本（`bg_` 前缀）

## 应用场景

### 1. 质量控制

```python
import pandas as pd

df = pd.read_csv('roi_results/roi_detailed_statistics.csv')

# 检查过曝图像
over_saturated = df[df['roi_R_saturation_ratio'] > 0.05]
print(f"过曝图像: {len(over_saturated)}")

# 检查G1/G2一致性
g_ratio_outliers = df[(df['roi_G1_G2_ratio'] < 0.8) | 
                       (df['roi_G1_G2_ratio'] > 1.2)]
print(f"G1/G2异常图像: {len(g_ratio_outliers)}")
```

### 2. 信号稳定性分析

```python
# 使用方差评估信号稳定性
df['signal_stability'] = 1.0 / (1.0 + df['roi_R_var'])

# 使用分位数范围评估分布
df['R_range'] = df['roi_R_p90'] - df['roi_R_p10']
```

### 3. 颜色比值归一化

```python
# 使用通道比值进行归一化，抗整体亮度漂移
normalized_features = df[['roi_R_G_ratio', 'roi_B_G_ratio']]
```

## 技术细节

### 饱和比例计算

```python
white_level = 16383.0  # 从RAW元数据读取，默认14-bit
saturation_threshold = white_level * 0.95
saturation_ratio = saturated_pixels / total_pixels
```

### 通道比值计算

```python
epsilon = 1e-6  # 防止除零
G_mean = (G1_mean + G2_mean) / 2.0
R_G_ratio = R_mean / (G_mean + epsilon)
```

## 质量指标建议

| 指标 | 正常范围 | 异常阈值 | 说明 |
|------|----------|----------|------|
| saturation_ratio | < 0.01 | > 0.05 | 饱和比例过高表示过曝 |
| G1_G2_ratio | 0.9 - 1.1 | < 0.8 或 > 1.2 | G1/G2应接近1 |
| 方差 | - | 过大 | 方差过大可能表示噪声或不均匀 |
| 背景均值 | 接近黑电平 | 过高 | 背景过亮可能有环境光干扰 |

## 兼容性

- **向后兼容**：原有的 `roi_results.json`, `roi_features.csv`, `feature_matrix.npy` 保持不变
- **可选功能**：详细统计作为额外输出，不影响现有流程
- **依赖包**：无新增依赖，使用现有的 numpy, pandas, rawpy, cv2

## 性能说明

由于需要重新读取图像并计算额外的统计量，批量处理时间可能略有增加（约10-20%）。如果只需要基本特征，可以注释掉 `batch_process()` 方法中的这一行：

```python
# self._save_detailed_statistics(results, output_dir)
```

## 下一步建议

1. **实时质量监控**：基于饱和比例和方差实现自动质量分级
2. **异常检测**：使用统计分布识别异常样本
3. **特征工程**：利用通道比值和分位数构建更稳健的特征
4. **可视化工具**：开发统计数据的交互式可视化界面

## 相关文档

- [详细统计说明.md](详细统计说明.md) - 功能详细说明
- [README_ROI.md](README_ROI.md) - ROI处理器总体说明
- [QUICKSTART.md](QUICKSTART.md) - 快速开始指南
