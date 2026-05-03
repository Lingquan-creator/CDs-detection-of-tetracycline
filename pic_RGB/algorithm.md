# pic_RGB 算法说明

本目录负责从 DNG/RAW 图像中选取样品 ROI 和背景区域，提取 R/G1/G2/B 四通道统计量，并生成浓度反演使用的 CSV/NPY/JSON 文件。

## 主要代码

- `roi_processor.py`: 主流程，批量读取 DNG，复用缓存或调用 GUI 选择 ROI/BG，计算统计量并保存结果。
- `roi_gui.py`: 交互式 ROI、背景内框、背景外框选择界面。
- `roi_manual.py`: 根据中心、宽高、角度生成旋转矩形 ROI mask 和背景环 mask。
- `roi_distribution_features.py`: 计算高斯/泊松/Gamma、矩、熵、FFT、MGF 等分布特征。
- `roi_config.py`: 输入输出路径、默认 ROI 参数、缓存复用和可视化配置。
- `analyze_statistics.py`: 查看和分析统计 CSV 的辅助脚本。

## 算法输入

算法层面的输入是：

- 一张或多张 Bayer DNG 图像。
- ROI 几何参数：中心、宽、高、旋转角。
- 背景内框和外框几何参数：各自中心、宽、高、旋转角。
- 可选元数据：浓度、曝光时间、ISO、黑电平、白电平、显示增益等。

当前实际默认输入/输出路径来自 `roi_config.py`：

- `DATA_DIR = dataset/xiaomi`
- `OUTPUT_DIR = dataset/xiaomi`
- `REUSE_LAST_SELECTION = True`
- `SELECTION_CACHE_PATH = dataset/xiaomi/roi_selection_cache.json`

路径可用环境变量覆盖：

```bash
PIC_RGB_DATA_DIR=/path/to/dngs
PIC_RGB_OUTPUT_DIR=/path/to/output
```

常见实际输入文件：

- `dataset/xiaomi/*.dng`
- `dataset/mate3_2/*.dng`
- `dataset/<name>/roi_selection_cache.json`
- 可选 `dataset/<name>/metadata.txt`

## 处理流程

1. 使用 `rawpy` 读取 DNG 的 Bayer RAW 图。
2. 扣除相机黑电平，并按 Bayer 排列拆成四个通道：

   ```text
   R  = raw[0::2, 0::2]
   G1 = raw[0::2, 1::2]
   G2 = raw[1::2, 0::2]
   B  = raw[1::2, 1::2]
   ```

3. 从文件名和目录元数据解析浓度、曝光时间、ISO 等信息。
4. 确定 ROI/BG：
   - 若缓存存在并启用 `REUSE_LAST_SELECTION`，直接复用缓存。
   - 否则打开 GUI，由用户拖动选择当前激活框：ROI、BG inner 或 BG outer。
5. 用 `roi_manual.py` 生成 ROI mask 和背景环 mask。背景环会排除 ROI 区域。
6. 对 ROI 和 BG 分别计算四通道统计：
   - mean, var, std
   - median, p10, p50, p90
   - saturation_ratio
   - R/G, B/G, R/B, G1/G2 比值
7. 计算背景扣除值：

   ```text
   R_corrected  = roi_R_mean  - bg_R_mean
   G1_corrected = roi_G1_mean - bg_G1_mean
   G2_corrected = roi_G2_mean - bg_G2_mean
   B_corrected  = roi_B_mean  - bg_B_mean
   ```

8. 计算 RAW 四通道和显示 RGB 的分布特征。
9. 保存 JSON、CSV、NPY 和每张图的 ROI 可视化 PNG。

## 算法输出

算法层面的输出是：

- 每张图的 ROI/BG 几何位置。
- ROI 和 BG 的四通道统计量。
- 背景扣除后的 R/G1/G2/B 特征。
- RAW/RGB 分布统计特征。
- 用于建模的特征矩阵。

实际输出目录是 `OUTPUT_DIR`。当前默认是：

- `dataset/xiaomi`

主要输出文件：

- `roi_results.json`: ROI/BG 几何、通道均值、背景扣除值和元数据。
- `roi_selection_cache.json`: 可复用的 ROI/BG 选择缓存。
- `roi_features.csv`: 简化特征表。
- `feature_matrix.npy`: 4 维背景扣除特征矩阵 `[R, G1, G2, B]`。
- `feature_matrix_full.npy`: RAW 均值 + RAW 分布 + RGB 分布的完整特征矩阵。
- `feature_matrix_rgb_full.npy`: 显示 RGB + RGB 分布特征矩阵。
- `roi_distribution_features.csv`: RAW 四通道分布特征。
- `roi_rgb_distribution_features.csv`: RGB 三通道分布特征。
- `roi_detailed_statistics.csv`: 详细统计表，后续浓度反演主要使用。
- `<filename>_roi.png`: 每张图的 ROI/BG 可视化。

## 使用方式

默认路径批量处理：

```bash
python pic_RGB/roi_processor.py
```

指定数据集：

```bash
PIC_RGB_DATA_DIR=dataset/mate3_2 \
PIC_RGB_OUTPUT_DIR=dataset/mate3_2 \
python pic_RGB/roi_processor.py
```
