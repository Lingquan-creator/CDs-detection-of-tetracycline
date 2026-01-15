# ROI处理系统 - 重构版

## 代码结构

### 核心文件

1. **roi_config.py** - 配置管理
   - 数据路径、输出路径
   - GUI预览参数
   - 背景环默认参数

2. **roi_gui.py** - GUI ROI选择
   - `ROISelection`: ROI选择结果数据类
   - `ROISelector`: OpenCV GUI交互界面
   - 功能：
     - 拖拽选择矩形ROI
     - 滑动条调整角度、尺寸、背景环参数
     - 键盘控制精细移动（方向键/WASD）
     - 独立控制ROI和背景环中心

3. **roi_processor.py** - ROI数据提取与处理
   - `ROIResult`: ROI处理结果数据类
   - `ROIProcessor`: 核心处理器
   - 功能：
     - 加载RAW图像（DNG格式）
     - 创建ROI和背景mask
     - 计算统计数据（均值、方差、分位数、饱和比例、通道比值）
     - 背景校正
     - 批量处理
     - 保存结果（JSON、CSV、NPY、PNG）

4. **analyze_statistics.py** - 统计分析工具
   - 读取并分析详细统计CSV
   - 显示通道均值、比值、饱和度等信息

### 辅助文件

- **test_processing.py** - 测试脚本

## 使用方法

### 基本流程

```python
from roi_processor import ROIProcessor
from pathlib import Path

# 创建处理器
processor = ROIProcessor()

# 处理单张图像
result = processor.process_image(
    filepath=Path("image.dng"),
    save_visualization=True,
    output_dir=Path("output")
)

# 批量处理
results = processor.batch_process(
    image_dir=Path("data_pic"),
    output_dir=Path("roi_results"),
    save_visualization=True
)
```

### GUI操作

1. **初始选择**：拖拽矩形框选择ROI区域
2. **参数调整**：
   - 左侧控制窗口：11个滑动条调整参数
   - 右侧预览窗口：实时显示ROI和背景环
3. **键盘控制**：
   - `方向键/WASD`：移动当前目标（1像素）
   - `Tab`：切换ROI/背景环控制
   - `Enter`：确认选择
   - `Esc/C`：取消

### 输出文件

- `roi_results.json` - 完整结果（JSON格式）
- `roi_features.csv` - 基本特征（ROI中心、尺寸、通道均值等）
- `roi_detailed_statistics.csv` - 详细统计（包含方差、分位数、饱和比例、通道比值）
- `feature_matrix.npy` - NumPy特征矩阵（用于机器学习）
- `*_roi_analysis.png` - 可视化图（6个子图展示ROI、背景、通道信号等）

## 配置修改

编辑 `roi_config.py`：

```python
DATA_DIR = Path(r"h:\000\CDs\pic_RGB\data_pic")      # 输入目录
OUTPUT_DIR = Path(r"h:\000\CDs\pic_RGB\roi_results") # 输出目录

DEFAULT_PARAMS = {
    "bg_ring_inner": 80,   # 背景环内边距（默认值）
    "bg_ring_outer": 240,  # 背景环外边距（默认值）
}

GUI_PARAMS = {
    "preview_max_dim": 1000,    # 预览图最大尺寸
    "preview_percentile": 99.5, # 对比度拉伸百分位
}
```

## 特性

### 已移除
- ❌ 自动ROI检测
- ❌ 固定尺寸ROI模式
- ❌ 手动坐标指定

### 保留并优化
- ✅ GUI交互式ROI选择
- ✅ 键盘精细控制
- ✅ 独立水平/垂直背景环参数
- ✅ ROI和背景环独立中心
- ✅ 详细统计分析
- ✅ 批量处理
- ✅ 完整可视化

## 依赖

- Python 3.7+
- OpenCV (cv2)
- NumPy
- rawpy
- matplotlib
- pandas (仅analyze_statistics.py)

## 注意事项

1. 所有ROI参数通过GUI交互式确定，不再需要预设尺寸
2. 背景环支持独立的水平/垂直边距（H/V）
3. 详细统计包含质量检查指标（饱和比例、通道比值）
4. 适配机器学习：特征矩阵格式统一
