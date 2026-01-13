# ROI自动识别和特征提取

## 功能说明

本模块用于从荧光图像（DNG格式RAW图像）中自动识别中央黄白色高亮区域（荧光试剂），并提取输入特征向量。

## 核心功能

### 1. RAW图像处理
- 读取DNG格式RAW图像
- 线性化处理（黑电平扣除）
- 保持在线性域，不做gamma校正和sRGB转换
- 按Bayer模式拆分为4个通道：R, G1, G2, B

### 2. 自动ROI识别
- **强度图构建**：使用 Y = max(R, G1, G2, B) 对颜色变化更鲁棒
- **荧光区域分割**：Otsu阈值 + 形态学处理 + 连通域分析
- **固定ROI定位**：使用质心定位，采用固定半径圆形ROI避免统计偏差
- **背景ROI生成**：环形区域，排除荧光污染

### 3. 特征向量提取
- 计算ROI内各通道统计值（均值、中位数、分位数等）
- 背景扣除：样品值 - 背景值
- 输出4维特征向量：[R, G1, G2, B]（背景校正后）
- 可选RGB显示向量：[R, (G1+G2)/2, B]

## 使用方法

### 基本用法

```python
from pathlib import Path
from roi_extractor import ROIExtractor

# 1. 创建提取器
extractor = ROIExtractor(
    fixed_roi_radius=50,      # 固定ROI半径（像素）
    bg_ring_inner=80,         # 背景环内径
    bg_ring_outer=120,        # 背景环外径
    min_area=1000             # 最小荧光区域面积
)

# 2. 批量处理图像
data_dir = Path(r'pic_RGB\data_pic')
output_dir = Path(r'pic_RGB\roi_results')

results = extractor.batch_process(
    image_dir=data_dir,
    output_dir=output_dir,
    save_visualization=True   # 保存可视化结果
)

# 3. 访问结果
for result in results:
    print(f"文件: {result.filename}")
    print(f"中心: ({result.center_x:.1f}, {result.center_y:.1f})")
    print(f"特征向量: {result.get_feature_vector()}")
    print(f"RGB向量: {result.get_rgb_vector()}")
```

### 处理单张图像

```python
# 处理单张图像
filepath = Path(r'pic_RGB\data_pic\IMG_20260112_213237.dng')
result = extractor.process_image(filepath, save_visualization=True, output_dir=output_dir)

# 获取特征向量（用于后续建模）
feature_vector = result.get_feature_vector()  # [R, G1, G2, B]
print(f"特征向量: {feature_vector}")

# 获取RGB向量（用于显示）
rgb_vector = result.get_rgb_vector()  # [R, G, B]
print(f"RGB向量: {rgb_vector}")
```

## 输出结果

### 1. 可视化图像
- 每张图像生成一个可视化PNG文件，包含6个子图：
  - 强度图
  - 荧光区域分割
  - 固定ROI标注
  - 背景ROI标注
  - RGB伪彩色 + ROI叠加
  - 通道值柱状图

### 2. 数据文件
- **roi_results.json**: 完整的ROI提取结果（JSON格式）
- **roi_features.csv**: 特征数据表（CSV格式，易于分析）
- **feature_matrix.npy**: 特征矩阵（NumPy数组，shape: [N, 4]）

### 3. ROIResult对象属性

```python
result.filename          # 文件名
result.center_x, center_y  # ROI中心坐标
result.radius            # ROI半径
result.area              # 荧光区域面积

# 原始通道值
result.R_mean, G1_mean, G2_mean, B_mean

# 背景值
result.R_bg, G1_bg, G2_bg, B_bg

# 背景校正后的值（特征向量）
result.R_corrected, G1_corrected, G2_corrected, B_corrected

# RGB显示值
result.R_display, G_display, B_display

# 元信息
result.exposure_time     # 曝光时间
result.iso               # ISO感光度
```

## 参数调节指南

### ROI尺寸参数
- `fixed_roi_radius=50`: 根据荧光区域大小调整，确保覆盖主要发光区
- `bg_ring_inner=80`: 应大于fixed_roi_radius，避免荧光散射污染
- `bg_ring_outer=120`: 确保有足够的背景采样区域

### 分割参数
- `min_area=1000`: 最小荧光区域面积，过滤噪声
- `morph_kernel_size=5`: 形态学操作核大小，调整去噪强度

### 如何调整
1. **ROI太小/太大**：调整`fixed_roi_radius`
2. **背景区域被污染**：增大`bg_ring_inner`
3. **分割失败**：减小`min_area`或调整`morph_kernel_size`

## 与量子点.md的对应关系

本代码实现了文档中"三、RAW 图像处理"部分的内容：

1. **3.2 RAW解码与线性化** → `load_raw_image()`方法
2. **3.3 自动识别ROI** → `segment_fluorescence()` + `create_fixed_roi()`
3. **3.4 背景ROI生成** → `create_background_roi()`
4. **3.5 ROI统计与背景扣除** → `compute_roi_statistics()`

## 依赖包

```bash
pip install numpy opencv-python rawpy matplotlib
```

## 注意事项

1. **DNG格式**: 确保图像为RAW格式（.dng），避免JPEG等压缩格式
2. **线性处理**: 代码保持在线性域，不做gamma校正
3. **固定ROI**: 使用固定半径避免亮区面积变化导致的统计偏差
4. **背景扣除**: 确保背景ROI足够暗且不被荧光污染

## 后续应用

提取的特征向量可用于：
- 光谱反演（结合fit_spectrum_general.py）
- 浓度预测建模
- 深度学习输入（DNN文件夹）
- 参数拟合分析
