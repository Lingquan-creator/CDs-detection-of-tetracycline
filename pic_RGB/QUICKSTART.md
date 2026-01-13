# ROI 自动识别 – 快速上手

本指南帮你用最少步骤跑通 ROI 自动识别与特征提取流程。

## 1) 环境与依赖
- Python 3.10+，建议使用虚拟环境。
- 在 `pic_RGB` 目录执行：
```bash
pip install -r requirements.txt
```

## 2) 准备数据
- 将待处理的 DNG 原始图片放入 `pic_RGB/data_pic/`。
- 输出默认写入 `pic_RGB/roi_results/`（可在 `roi_config.py` 修改输入输出路径）。

目录示例：
```
pic_RGB/
  ├─ data_pic/
  │   ├─ IMG_20260112_213237.dng
  │   └─ ...
  ├─ roi_config.py
  ├─ roi_extractor.py
  └─ test_roi.py
```

## 3) 一键跑通
### 方式 A：直接批量处理
```bash
python roi_extractor.py
```
使用 `roi_config.py` 中的路径与参数，批量处理 `data_pic` 下所有 DNG。

### 方式 B：交互式测试
```bash
python test_roi.py
```
按提示选择：
- `1` 单张测试
- `2` 批量处理
- `3` 参数试验
- `4` 运行全部测试

### 方式 C：在代码中调用
```python
from pathlib import Path
from roi_extractor import ROIExtractor

extractor = ROIExtractor(
    fixed_roi_width=100,
    fixed_roi_height=100,
    bg_ring_inner=80,
    bg_ring_outer=120
)

results = extractor.batch_process(
    image_dir=Path("pic_RGB/data_pic"),
    output_dir=Path("pic_RGB/roi_results"),
    save_visualization=True
)
```

## 4) 查看输出
- `roi_results.json`：完整 ROI 结果列表。
- `roi_features.csv`：特征表（R/G1/G2/B 背景校正后）。
- `feature_matrix.npy`：特征矩阵 (N, 4)。
- `*_roi.png`：可视化（强度图、分割、固定 ROI、背景环、伪彩等）。

快速读取特征矩阵：
```python
import numpy as np
features = np.load("pic_RGB/roi_results/feature_matrix.npy")
print(features.shape)  # (N, 4)
```

## 5) 调参快捷入口
在 `roi_config.py` 修改：
- `FIXED_ROI_WIDTH`, `FIXED_ROI_HEIGHT`：固定 ROI 矩形尺寸。
- `ROI_ANGLE_DEG`：ROI 旋转角度（度，逆时针为正）。
- `BG_RING_INNER`, `BG_RING_OUTER`：背景环内外半径。
- `MIN_AREA`：最小荧光区域面积，过滤噪声。
- `MORPH_KERNEL_SIZE`：形态学核大小，调节去噪力度。

调完后重新运行上面的命令即可。***
