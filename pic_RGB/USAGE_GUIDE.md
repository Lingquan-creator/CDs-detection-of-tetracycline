# ROI处理系统使用指南

## 快速开始

### 1. 测试系统

```bash
# 激活虚拟环境
& H:/000/CDs/.CDs_venv/Scripts/Activate.ps1

# 运行测试（处理前3个DNG文件）
python pic_RGB/test_gui.py
```

### 2. 批量处理所有文件

```bash
python pic_RGB/roi_processor.py
```

### 3. 分析结果

```bash
python pic_RGB/analyze_statistics.py
```

## GUI操作指南

### 第一步：选择ROI
1. 弹出窗口显示图像
2. 用鼠标拖拽绘制矩形框
3. 按 Enter 或 Space 确认

### 第二步：调整参数
- **左侧控制窗口**：11个滑动条
  - Angle: 旋转角度 (-90° ~ 90°)
  - Width %: 宽度缩放 (10% ~ 300%)
  - Height %: 高度缩放 (10% ~ 300%)
  - BG inner H/V: 背景环内边距（水平/垂直）
  - BG outer H/V: 背景环外边距（水平/垂直）
  - ROI X/Y: ROI中心位置
  - BG X/Y: 背景环中心位置

- **右侧预览窗口**：实时显示
  - 绿色框：ROI区域
  - 黄色框：背景环

### 第三步：精细调整
**键盘控制**：
- `方向键` 或 `W/A/S/D`：移动当前目标（1像素步长）
- `Tab`：切换控制目标（ROI ↔ 背景环）
- `Enter`：确认并保存
- `Esc` 或 `C`：取消

## 输出文件说明

处理后在 `roi_results/` 目录生成：

### 1. roi_results.json
完整的JSON格式结果，包含所有参数和统计值。

### 2. roi_features.csv
基本特征CSV，包含：
- ROI位置、尺寸、角度
- 背景环参数
- 通道均值（原始、背景、校正后）

### 3. roi_detailed_statistics.csv
详细统计CSV，包含：
- 均值、方差、标准差
- 分位数（p10, p50, p90）
- 饱和比例（过曝检测）
- 通道比值（R/G, B/G, R/B, G1/G2）

### 4. feature_matrix.npy
NumPy格式特征矩阵，用于机器学习：
```python
import numpy as np
features = np.load('roi_results/feature_matrix.npy')
# shape: (N, 4) - N个样本，每个4个特征 [R, G1, G2, B]
```

### 5. *_roi_analysis.png
可视化图，包含6个子图：
1. 强度图（最大通道）
2. 选择的ROI mask
3. ROI区域叠加
4. 背景环区域
5. RGB预览
6. 通道信号柱状图

## Python API

### 基本用法

```python
from roi_processor import ROIProcessor
from pathlib import Path

# 创建处理器
processor = ROIProcessor()

# 处理单张图像
result = processor.process_image(
    filepath=Path("data_pic/image.dng"),
    save_visualization=True,
    output_dir=Path("roi_results")
)

# 访问结果
print(f"R校正值: {result.R_corrected}")
print(f"G1校正值: {result.G1_corrected}")
```

### 批量处理

```python
results = processor.batch_process(
    image_dir=Path("data_pic"),
    output_dir=Path("roi_results"),
    save_visualization=True
)

# 结果分析
import numpy as np
features = np.array([r.get_feature_vector() for r in results])
print(f"特征矩阵形状: {features.shape}")
```

### 自定义配置

```python
from roi_config import ROISettings

# 自定义设置
settings = ROISettings(
    bg_ring_inner=100,      # 背景环内边距
    bg_ring_outer=300,      # 背景环外边距
    preview_max_dim=1200,   # 预览最大尺寸
    preview_percentile=99   # 对比度拉伸
)

processor = ROIProcessor(settings=settings)
```

## 配置文件

编辑 `roi_config.py` 修改默认值：

```python
# 数据路径
DATA_DIR = Path(r"h:\000\CDs\pic_RGB\data_pic")
OUTPUT_DIR = Path(r"h:\000\CDs\pic_RGB\roi_results")

# 背景环默认参数
DEFAULT_PARAMS = {
    "bg_ring_inner": 80,   # 内边距（像素）
    "bg_ring_outer": 240,  # 外边距（像素）
}

# GUI预览参数
GUI_PARAMS = {
    "preview_max_dim": 1000,      # 最大尺寸
    "preview_percentile": 99.5,   # 亮度百分位
}
```

## 常见问题

### Q1: 方向键不响应？
A: 确保图像预览窗口处于激活状态（点击一下窗口）。

### Q2: 如何调整背景环？
A: 使用左侧控制窗口的4个背景环滑动条：
- BG inner H/V：内边距（ROI边缘到内环的距离）
- BG outer H/V：外边距（ROI边缘到外环的距离）
- H = 水平方向，V = 垂直方向

### Q3: ROI和背景环可以不同心吗？
A: 可以。按 Tab 键切换到"BG"模式，然后用方向键移动背景环中心。

### Q4: 如何保存ROI设置？
A: 当前版本通过GUI每次选择。可以查看生成的CSV文件获取之前的参数值。

### Q5: 支持哪些图像格式？
A: 当前仅支持DNG格式的RAW图像。

## 技术细节

### 坐标系统
- 原点：图像左上角
- X轴：向右为正
- Y轴：向下为正
- 角度：逆时针为正（-90° ~ 90°）

### 通道定义
RAW Bayer阵列：
```
R  G1    →  [0] R, [1] G1
G2 B         [2] G2, [3] B
```

### 背景校正
```
校正值 = ROI均值 - 背景均值
显示值 = max(0, 校正值)
```

### 统计指标
- **均值**：算术平均
- **方差/标准差**：信号稳定性
- **分位数**：分布特征
- **饱和比例**：>95%白电平的像素占比
- **通道比值**：归一化特征，抗整体亮度漂移

## 性能优化

### 加快处理速度
1. 关闭可视化：`save_visualization=False`
2. 减小预览尺寸：`preview_max_dim=800`
3. 批量处理时避免频繁磁盘写入

### 内存优化
处理大量图像时，考虑分批处理：
```python
batch_size = 10
for i in range(0, len(files), batch_size):
    batch = files[i:i+batch_size]
    results = [processor.process_image(f) for f in batch]
    processor._save_results(results, output_dir)
```

## 依赖环境

```
Python 3.7+
opencv-python (cv2)
numpy
rawpy
matplotlib
pandas (仅 analyze_statistics.py)
```

## 获取帮助

查看代码注释或联系维护者。
