# K-means图像分割

使用K-means聚类方法将量子点发光图像分割为背景和高亮区域。

## 功能特点

### 基础版本 (`kmeans_segmentation.py`)
- 使用K-means将图像分为2个聚类（背景和高亮区域）
- 自动识别高亮区域（基于亮度值）
- 计算各区域的统计信息（面积、RGB均值等）
- 生成可视化结果和掩码图像

### 高级版本 (`kmeans_advanced.py`)
除了基础功能外，还包括：
- **形态学操作**：使用闭运算和开运算优化掩码，去除噪点和填充小孔
- **连通组件分析**：识别高亮区域中的独立组件，统计数量和位置
- **详细统计**：计算每个区域的均值、标准差、最小值、最大值、中位数
- **颜色通道分析**：分别分析R、G、B通道的统计信息
- **可配置参数**：通过`config.py`灵活调整各项参数

## 物理背景

量子点溶液的发光强度 $I$ 与浓度 $c$ 呈负指数关系：

$$I = I_0 \cdot e^{-\alpha c}$$

其中：
- $I_0$ 是初始发光强度
- $\alpha$ 是衰减系数
- $c$ 是浓度

通过K-means分割，可以：
1. 分离高亮区域（发光液体）和背景
2. 提取高亮区域的强度信息
3. 为后续浓度计算提供数据基础

## 目录结构

```
pic_Kmeans/
├── config.py                    # 配置文件
├── kmeans_segmentation.py       # 基础版本
├── kmeans_advanced.py           # 高级版本
├── README.md                    # 本文件
├── data_pic/                    # 输入DNG图像
│   ├── 0.2.dng
│   ├── 0.4.dng
│   └── ...
└── roi_results/                 # 输出结果
    ├── *_segmentation.png       # 可视化结果
    ├── *_background.png         # 背景掩码
    ├── *_bright.png             # 高亮区域掩码
    ├── kmeans_results.json      # 详细结果（JSON）
    └── kmeans_summary.csv       # 汇总表（CSV）
```

## 使用方法

### 1. 安装依赖

```bash
pip install numpy opencv-python scikit-learn matplotlib scipy rawpy
```

### 2. 配置参数（可选）

编辑 `config.py` 调整参数：

```python
# 路径
DATA_DIR = Path(r"h:\000\CDs\pic_Kmeans\data_pic")
OUTPUT_DIR = Path(r"h:\000\CDs\pic_Kmeans\roi_results")

# K-means参数
N_CLUSTERS = 2  # 聚类数量

# 形态学操作
MORPHOLOGY_KERNEL_SIZE = 5  # 核大小
APPLY_MORPHOLOGY = True  # 是否应用

# 连通组件分析
ANALYZE_CONNECTED_COMPONENTS = True
MIN_COMPONENT_AREA = 100  # 最小组件面积

# 可视化
SAVE_VISUALIZATION = True
SAVE_MASKS = True
```

### 3. 运行分割

**基础版本**（快速分割）：
```bash
python kmeans_segmentation.py
```

**高级版本**（推荐，包含更多分析）：
```bash
python kmeans_advanced.py
```

## 输出说明

### 可视化图像 (`*_segmentation.png`)
包含6个子图：
1. **原始RGB图像**：输入的原始图像
2. **灰度图**：预处理后的灰度图
3. **K-means标签**：聚类结果，不同颜色表示不同聚类
4. **背景掩码**：背景区域（黑色=背景，白色=前景）
5. **高亮区域（原始）**：未经优化的高亮区域掩码
6. **高亮区域（优化）**：经过形态学操作优化的掩码

### 掩码图像
- `*_background.png`：背景区域掩码
- `*_bright.png`：高亮区域掩码

可用于：
- 后续图像处理
- 区域特征提取
- 与其他分析工具集成

### JSON结果 (`kmeans_results.json`)
每个图像的详细信息：
```json
{
  "filename": "0.2.dng",
  "cluster_centers": [50.2, 180.5],
  "bright_label": 1,
  "background_stats": {
    "area": 1234567,
    "mean_intensity": 45.2,
    "R_mean": 42.3,
    "G_mean": 45.8,
    "B_mean": 47.5
  },
  "bright_region_stats": {
    "area": 123456,
    "mean_intensity": 185.6,
    "R_mean": 180.2,
    "G_mean": 188.3,
    "B_mean": 188.9
  },
  "bright_components": {
    "num_components": 1,
    "largest_component_area": 123000
  },
  "intensity_contrast": 140.4
}
```

### CSV汇总 (`kmeans_summary.csv`)
适合用Excel打开，包含关键指标：
- 文件名
- 背景/高亮区域占比
- 各区域平均亮度
- 强度对比度
- 连通组件数量

## 参数调优建议

### K-means聚类数量
- **N_CLUSTERS = 2**：适合简单场景（背景 vs 高亮）
- **N_CLUSTERS = 3**：如果存在中间亮度区域
- **N_CLUSTERS > 3**：复杂场景，需要更细的分割

### 形态学核大小
- **MORPHOLOGY_KERNEL_SIZE = 3**：轻微平滑
- **MORPHOLOGY_KERNEL_SIZE = 5**：标准平滑（推荐）
- **MORPHOLOGY_KERNEL_SIZE = 7+**：强力平滑，可能丢失细节

### 连通组件最小面积
- **MIN_COMPONENT_AREA = 50**：保留较小的发光点
- **MIN_COMPONENT_AREA = 100**：标准设置（推荐）
- **MIN_COMPONENT_AREA = 500+**：只保留大面积区域

## 算法流程

```
输入DNG图像
    ↓
1. 加载并转换为RGB
    ↓
2. 转换为灰度图
    ↓
3. K-means聚类（k=2）
    ↓
4. 识别高亮区域（最大聚类中心）
    ↓
5. 生成掩码
    ↓
6. 形态学优化（闭运算+开运算）
    ↓
7. 连通组件分析
    ↓
8. 统计分析
    ↓
输出：可视化图、掩码、统计结果
```

## 与其他模块的关系

- **pic_freqence/** 和 **pic_RGB/**：手动/自动ROI提取
- **pic_Kmeans/**（本模块）：自动K-means分割
- **concentration/**：基于RGB进行浓度拟合
- **fit_data/**：基于光谱数据进行拟合

## 常见问题

### Q1: 高亮区域被错误分类为背景？
**A**: 可能是图像对比度不足。尝试：
- 检查图像是否正常加载
- 调整DNG处理参数（`NO_AUTO_BRIGHT`）
- 增加聚类数量（`N_CLUSTERS = 3`）

### Q2: 掩码有很多噪点？
**A**: 调整形态学参数：
- 增大 `MORPHOLOGY_KERNEL_SIZE`
- 确保 `APPLY_MORPHOLOGY = True`

### Q3: 处理速度慢？
**A**: 
- 使用基础版本 `kmeans_segmentation.py`
- 关闭不需要的分析：`ANALYZE_CONNECTED_COMPONENTS = False`
- 关闭可视化：`SAVE_VISUALIZATION = False`

### Q4: 内存不足？
**A**: 
- 分批处理图像
- 降低输出DPI：`VISUALIZATION_DPI = 100`
- 处理完一张后立即释放内存

## 示例结果

对于附件中的图像（量子点溶液在紫外光下发光）：
- **背景区域**：深紫色/黑色区域，低亮度
- **高亮区域**：白色发光的试管部分，高亮度
- **分割效果**：清晰分离试管和背景

## 后续分析

分割后的掩码可用于：
1. **提取发光强度**：计算高亮区域的平均/积分强度
2. **浓度计算**：结合负指数模型估算浓度
3. **时间序列分析**：跟踪不同时刻的发光变化
4. **空间分布**：分析发光的空间分布特征

## 作者与日期

- 创建日期：2026年1月25日
- 用途：量子点浓度检测实验的图像分割
