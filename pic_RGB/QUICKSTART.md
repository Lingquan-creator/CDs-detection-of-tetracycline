# ROI自动识别 - 快速开始

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- `rawpy`: RAW图像读取
- `opencv-python`: 图像处理和分割
- `numpy`: 数值计算
- `matplotlib`: 可视化

## 2. 数据准备

确保你的DNG图像文件在 `data_pic` 文件夹中：

```
pic_RGB/
  ├── data_pic/
  │   ├── IMG_20260112_213237.dng
  │   ├── IMG_20260112_213433.dng
  │   └── ...
  ├── roi_extractor.py
  └── test_roi.py
```

## 3. 运行测试

### 方式1：交互式测试

```bash
python test_roi.py
```

然后根据提示选择：
- `1`: 测试单张图像（快速验证）
- `2`: 批量处理所有图像
- `3`: 测试不同参数设置
- `4`: 运行所有测试

### 方式2：直接批量处理

```python
from pathlib import Path
from roi_extractor import ROIExtractor

# 创建提取器
extractor = ROIExtractor()

# 批量处理
results = extractor.batch_process(
    image_dir=Path(r'data_pic'),
    output_dir=Path(r'roi_results')
)
```

## 4. 查看结果

处理完成后，在 `roi_results` 文件夹中查看：

### 可视化图像
每张原始图像对应一个 `*_roi.png` 文件，包含：
- 强度图
- 荧光区域分割
- 固定ROI标注
- 背景ROI标注
- RGB伪彩色
- 通道值柱状图

### 数据文件
- `roi_results.json`: 完整结果（所有信息）
- `roi_features.csv`: 特征数据表（方便Excel打开）
- `feature_matrix.npy`: 特征矩阵（NumPy格式，用于建模）

## 5. 使用提取的特征

### 读取特征矩阵

```python
import numpy as np

# 加载特征矩阵
features = np.load('roi_results/feature_matrix.npy')
print(f"特征形状: {features.shape}")  # (N, 4) - N张图像，4个通道

# 每行是一个样本的特征向量 [R, G1, G2, B]
sample_features = features[0]
print(f"第一个样本: {sample_features}")
```

### 读取CSV数据

```python
import pandas as pd

# 使用pandas读取
df = pd.read_csv('roi_results/roi_features.csv')

# 查看数据
print(df.head())

# 访问特定列
r_values = df['R_corrected'].values
g1_values = df['G1_corrected'].values
```

### 使用ROI结果对象

```python
import json

# 读取JSON结果
with open('roi_results/roi_results.json', 'r') as f:
    results_data = json.load(f)

# 访问第一个结果
first_result = results_data[0]
print(f"文件名: {first_result['filename']}")
print(f"R通道: {first_result['R_corrected']}")
```

## 6. 参数调整

如果默认参数效果不好，可以调整：

```python
extractor = ROIExtractor(
    fixed_roi_radius=50,      # ROI圆形半径（调整以覆盖荧光区）
    bg_ring_inner=80,         # 背景环内径（应大于ROI半径）
    bg_ring_outer=120,        # 背景环外径（确保足够背景采样）
    min_area=1000,            # 最小荧光区域面积（过滤噪声）
    morph_kernel_size=5       # 形态学操作核大小（去噪强度）
)
```

### 常见问题

**问题1: ROI太小，未覆盖全部荧光区**
- 解决：增大 `fixed_roi_radius`

**问题2: 背景区域被荧光污染**
- 解决：增大 `bg_ring_inner`

**问题3: 分割失败（找不到荧光区域）**
- 解决：减小 `min_area` 或调整 `morph_kernel_size`

**问题4: 检测到错误的区域**
- 解决：增大 `min_area` 过滤小噪声区域

## 7. 进阶使用

### 处理特定文件列表

```python
from pathlib import Path

files_to_process = [
    Path('data_pic/IMG_20260112_213237.dng'),
    Path('data_pic/IMG_20260112_213433.dng'),
]

results = []
for filepath in files_to_process:
    result = extractor.process_image(filepath)
    results.append(result)
```

### 提取扩展特征

```python
# ROIResult对象还包含统计信息
result = results[0]

# 原始通道值（背景扣除前）
raw_values = [result.R_mean, result.G1_mean, result.G2_mean, result.B_mean]

# 背景值
bg_values = [result.R_bg, result.G1_bg, result.G2_bg, result.B_bg]

# 几何信息
geometry = {
    'center': (result.center_x, result.center_y),
    'radius': result.radius,
    'area': result.area,
    'bbox': result.bbox
}
```

### 自定义ROI统计方法

如果需要更复杂的统计（如分位数、方差等），可以修改 `roi_extractor.py` 中的 `compute_roi_statistics()` 方法。

## 8. 与其他模块集成

提取的特征可以用于：

- **光谱拟合**: 传递给 `fit_spectrum_general.py`
- **浓度预测**: 结合 `linear_fit_*.py` 进行线性回归
- **深度学习**: 作为 `DNN/` 模块的输入
- **参数分析**: 使用 `plot_parameters.py` 可视化

示例：将特征用于后续分析

```python
import numpy as np

# 加载特征
features = np.load('roi_results/feature_matrix.npy')

# 归一化（可选）
features_norm = features / features.max(axis=0)

# 计算RGB比值（常用特征）
R, G1, G2, B = features.T
G = (G1 + G2) / 2
RG_ratio = R / (G + 1e-6)
GB_ratio = G / (B + 1e-6)

print(f"R/G比值范围: [{RG_ratio.min():.3f}, {RG_ratio.max():.3f}]")
print(f"G/B比值范围: [{GB_ratio.min():.3f}, {GB_ratio.max():.3f}]")
```

## 9. 故障排查

### LibRaw相关错误
如果遇到 `rawpy` 导入错误：
```bash
pip uninstall rawpy
pip install rawpy --no-binary rawpy
```

### OpenCV导入问题
```bash
pip install opencv-python-headless  # 无GUI版本
```

### 内存不足
处理大量图像时：
```python
# 分批处理
batch_size = 10
for i in range(0, len(image_files), batch_size):
    batch = image_files[i:i+batch_size]
    # 处理批次
```

## 10. 完整示例

```python
from pathlib import Path
from roi_extractor import ROIExtractor
import numpy as np
import matplotlib.pyplot as plt

# 1. 创建提取器
extractor = ROIExtractor(fixed_roi_radius=50)

# 2. 批量处理
results = extractor.batch_process(
    image_dir=Path('data_pic'),
    output_dir=Path('roi_results'),
    save_visualization=True
)

# 3. 加载特征矩阵
features = np.load('roi_results/feature_matrix.npy')

# 4. 分析特征
R, G1, G2, B = features.T
G = (G1 + G2) / 2

# 5. 可视化
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.scatter(R, G, alpha=0.6)
plt.xlabel('R通道')
plt.ylabel('G通道')
plt.title('R vs G')

plt.subplot(132)
plt.scatter(G, B, alpha=0.6)
plt.xlabel('G通道')
plt.ylabel('B通道')
plt.title('G vs B')

plt.subplot(133)
plt.scatter(R/G, G/B, alpha=0.6)
plt.xlabel('R/G比值')
plt.ylabel('G/B比值')
plt.title('颜色比值空间')

plt.tight_layout()
plt.savefig('roi_results/feature_analysis.png', dpi=150)
plt.show()

print(f"处理完成: {len(results)} 个样本")
print(f"特征维度: {features.shape}")
```

---

更多详细信息请参考 `README_ROI.md`
