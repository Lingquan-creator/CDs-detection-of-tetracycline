# ROI 自动识别与特征提取

从 DNG 原始图片中自动定位荧光试剂区域，生成可用于光谱拟合、建模或可视化的特征数据。

## 模块概览
- RAW 读取与线性化：用 `rawpy` 读取 DNG，分离 RGGB 四通道，做黑电平扣除。
- 强度图构建：`Y = max(R, G1, G2, B)`，中值滤波抑制热噪声。
- 荧光区域分割：Otsu 阈值 + 形态学开闭运算 + 连通域筛选。
- ROI 设计：以质心为中心的固定矩形 ROI；同心环作为背景 ROI，自动避开荧光主体。
- 特征输出：背景校正后的 [R, G1, G2, B]，可选 RGB 展示向量；同时生成可视化图、JSON、CSV、NumPy 矩阵。

## 快速开始
1. 安装依赖（在 `pic_RGB` 下）：`pip install -r requirements.txt`
2. 将 DNG 放入 `pic_RGB/data_pic/`，如需修改路径或参数，编辑 `roi_config.py`。
3. 批量处理：`python roi_extractor.py`  
   交互测试：`python test_roi.py`
4. 在 `pic_RGB/roi_results/` 查看输出（`*.png`、`roi_results.json`、`roi_features.csv`、`feature_matrix.npy`）。
更多细节见 `QUICKSTART.md`。

## 代码调用示例
```python
from pathlib import Path
from roi_extractor import ROIExtractor

extractor = ROIExtractor(
    fixed_roi_width=100,
    fixed_roi_height=100,
    roi_angle_deg=0.0,   # 需要旋转时调整角度（度，逆时针为正）
    bg_ring_inner=80,
    bg_ring_outer=120,
    morph_kernel_size=5,
    min_area=1000,
)

results = extractor.batch_process(
    image_dir=Path("pic_RGB/data_pic"),
    output_dir=Path("pic_RGB/roi_results"),
    save_visualization=True,
)

first = results[0]
print(first.filename, first.get_feature_vector())  # [R, G1, G2, B] 背景校正后
```

处理单张图片：
```python
result = extractor.process_image(
    Path("pic_RGB/data_pic/IMG_001.dng"),
    save_visualization=True,
    output_dir=Path("pic_RGB/roi_results"),
)
print(result.center_x, result.center_y, result.area)
```

## 配置参数（`roi_config.py`）
- 路径：`DATA_DIR`（输入 DNG 目录），`OUTPUT_DIR`（输出目录）。
- 固定 ROI 尺寸：`FIXED_ROI_WIDTH=100`，`FIXED_ROI_HEIGHT=100`。
- ROI 旋转：`ROI_ANGLE_DEG=0.0`（度，逆时针为正，可用于对齐倾斜样品）。
- 背景环：`BG_RING_INNER=80`，`BG_RING_OUTER=120`（应大于 ROI 尺寸以避免污染）。
- 分割与去噪：`MIN_AREA=1000`（最小连通域），`MORPH_KERNEL_SIZE=5`（形态学核大小）。
- 输出：`SAVE_VISUALIZATION=True`，`VISUALIZATION_DPI=150`，`VISUALIZATION_FIGSIZE=(15, 10)`。

调参提示：
- ROI 太小覆盖不全 → 增大 `FIXED_ROI_WIDTH/HEIGHT`。
- 背景被荧光污染 → 增大 `BG_RING_INNER`。
- 找不到荧光区域 → 减小 `MIN_AREA` 或 `MORPH_KERNEL_SIZE`。
- 噪声太多 → 略微增大 `MORPH_KERNEL_SIZE`。

## 输出文件说明
- `roi_results.json`：完整 ROI 记录列表（含位置、通道统计、背景校正值、曝光/ISO 等）。
- `roi_features.csv`：同上但以表格形式便于 Excel/分析。
- `feature_matrix.npy`：形状 `(N, 4)`，顺序 `[R, G1, G2, B]`，背景校正后。
- `*_roi.png`：六宫格可视化（强度图、分割、固定 ROI、背景环、伪彩+ROI、通道柱状图）。

`ROIResult` 主要字段：
- `filename`、`center_x/center_y`、`width/height`、`bbox`、`area`
- `R_mean/G1_mean/G2_mean/B_mean` 与对应背景值 `*_bg`
- 背景校正后 `R_corrected/G1_corrected/G2_corrected/B_corrected`
- 展示用 `R_display/G_display/B_display`
- 元信息 `exposure_time`, `iso`

## 常见问题
- rawpy 安装失败：`pip install rawpy --no-binary rawpy`
- OpenCV GUI 冲突：在无显示环境使用 `pip install opencv-python-headless`
- 结果全零或全空：检查 `data_pic` 是否是 DNG，或调低 `MIN_AREA`/`MORPH_KERNEL_SIZE`。
- 可视化缺失：确认 `SAVE_VISUALIZATION=True` 且 `output_dir` 可写。

## 后续应用
生成的特征可直接用于：
- 光谱拟合/线性回归（与上游 `fit_*` 模块结合）
- 深度学习输入（`DNN/` 目录）
- 参数分析与可视化（`fit_data/plot_*` 等脚本）
