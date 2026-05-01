# pic_RGB 算法说明

本目录负责从 DNG/RAW 图像中选取样品 ROI 和背景区域，提取 R/G1/G2/B 四通道统计量，并生成后续浓度反演使用的 `roi_detailed_statistics.csv`。

## 主要代码

- `roi_processor.py`: 主流程，批量读取 DNG、调用 GUI 选择 ROI/BG、计算统计量并保存结果。
- `roi_gui.py`: 交互式 ROI 和背景环选择界面。
- `roi_manual.py`: 根据中心、宽高、角度生成旋转矩形 ROI mask 和背景环 mask。
- `roi_distribution_features.py`: 计算强度分布特征，例如高斯/泊松/Gamma 拟合、矩、熵、FFT、MGF。
- `roi_config.py`: 输入输出目录、ROI 尺寸、背景环尺寸、GUI 步长等配置。
- `analyze_statistics.py`: 查看和分析详细统计 CSV 的示例脚本。

## 算法输入

算法层面的输入是：

- 一张或多张 Bayer DNG 图像。
- ROI 几何参数：中心、宽、高、旋转角。
- 背景环几何参数：中心、内外扩展尺寸、旋转角。
- 可选元数据：浓度、曝光时间、ISO、白电平、黑电平、显示增益等。

实际默认输入：

- 图像目录：`pic_RGB/dataset2/`
- 文件类型：`*.dng`
- 可选目录元数据：输入图像同目录下的 `metadata.txt`

默认路径可通过环境变量覆盖：

```bash
PIC_RGB_DATA_DIR=/path/to/dngs
PIC_RGB_OUTPUT_DIR=/path/to/output
```

## 处理流程

1. 使用 `rawpy` 读取 DNG 的 Bayer RAW 图。
2. 扣除相机黑电平，并按 Bayer 排列拆成四个通道：

   ```text
   R  = raw[0::2, 0::2]
   G1 = raw[0::2, 1::2]
   G2 = raw[1::2, 0::2]
   B  = raw[1::2, 1::2]
   ```

3. 从文件名解析浓度和曝光时间。支持类似：

   ```text
   12.dng
   12_0.01.dng
   12.1000.dng
   ```

4. 通过 GUI 或缓存确定 ROI 和背景环。ROI 是可旋转矩形，背景是扣除 ROI 后的空心矩形环。
5. 为 ROI 和 BG 分别生成 mask。
6. 对每个 RAW 通道计算统计量：

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

8. 可选计算显示 RGB：使用目录元数据中的 `R_gain`, `G_gain`, `B_gain`, `gamma`, `saturability` 对背景扣除值做显示变换。
9. 计算 RAW/RGB 强度分布特征，用于更复杂的建模实验。
10. 保存 JSON、CSV、NPY 和每张图的 ROI 可视化 PNG。

## 算法输出

算法层面的输出是：

- 每张图的 ROI/BG 几何位置。
- ROI 和 BG 的四通道统计量。
- 背景扣除后的 R/G1/G2/B 特征。
- 分布统计特征。
- 用于建模的特征矩阵。

实际默认输出目录：

- `pic_RGB/roi_results2/`

主要输出文件：

- `roi_results.json`: 每张图的 ROI/BG 几何、通道均值、背景扣除值和元数据。
- `roi_features.csv`: 简化特征表，包含背景扣除均值和显示 RGB。
- `feature_matrix.npy`: 4 维背景扣除特征矩阵 `[R, G1, G2, B]`。
- `feature_matrix_full.npy`: RAW 均值 + RAW 分布 + RGB 分布的完整特征矩阵。
- `feature_matrix_rgb_full.npy`: 显示 RGB + RGB 分布特征矩阵。
- `roi_distribution_features.csv`: RAW 四通道分布特征。
- `roi_rgb_distribution_features.csv`: RGB 三通道分布特征。
- `roi_detailed_statistics.csv`: 后续浓度反演主要使用的详细统计 CSV。
- `<filename>_roi.png`: 每张图的 ROI/BG 可视化。

## 使用方式

批量处理默认目录：

```bash
python pic_RGB/roi_processor.py
```

指定输入输出目录：

```bash
PIC_RGB_DATA_DIR=pic_RGB/dataset2 \
PIC_RGB_OUTPUT_DIR=pic_RGB/roi_results2 \
python pic_RGB/roi_processor.py
```

后续三个浓度反演目录通常读取：

```text
pic_RGB/roi_results/roi_detailed_statistics.csv
```

或由批处理脚本读取 `dataset/*/roi_results/roi_detailed_statistics.csv`。
