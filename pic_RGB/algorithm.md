# pic_RGB 算法说明

本目录负责从 DNG/RAW 图像中选取样品 ROI 和背景区域，只提取当前浓度反演输入向量需要的 R/G1/G2/B 均值参数，并保存 ROI/BG 选择记录。

## 主要代码

- `roi_processor.py`: 主流程，批量读取 DNG，复用缓存或调用 GUI 选择 ROI/BG，计算统计量并保存结果。
- `roi_gui.py`: 交互式 ROI、背景内框、背景外框选择界面。
- `roi_manual.py`: 根据中心、宽高、角度生成旋转矩形 ROI mask 和背景环 mask。
- `roi_distribution_features.py`: 历史分布特征工具；当前精简流程不再调用。
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
6. 对 ROI 和 BG 分别只计算四通道均值：
   - `R_mean`, `G1_mean`, `G2_mean`, `B_mean`
7. 计算背景扣除值：

   ```text
   R_corrected  = roi_R_mean  - bg_R_mean
   G1_corrected = roi_G1_mean - bg_G1_mean
   G2_corrected = roi_G2_mean - bg_G2_mean
   B_corrected  = roi_B_mean  - bg_B_mean
   ```

8. 构造当前使用的输入向量：

   ```text
   ln_R_sig, ln_G_sig, ln_B_sig, Ratio_RG
   ```

9. 保存 ROI/BG 选择记录和输入向量所需参数。默认不再输出 NPY、分布统计 CSV 和每张图的 PNG。

## 算法输出

算法层面的输出是：

- 每张图的 ROI/BG 几何位置。
- ROI/BG 几何位置。
- ROI 和 BG 的四通道均值。
- 背景扣除后的 `R_sig/G_sig/B_sig`。
- 当前 4 维输入向量。

实际输出目录是 `OUTPUT_DIR`。当前默认是：

- `dataset/xiaomi`

主要输出文件：

- `roi_results.json`: ROI/BG 几何、通道均值、背景扣除值和元数据。
- `roi_selection_cache.json`: 可复用的 ROI/BG 选择缓存。
- `roi_detailed_statistics.csv`: 当前训练/预测需要的 ROI/BG 均值、`R_sig/G_sig/B_sig` 和 `ln_R_sig/ln_G_sig/ln_B_sig/Ratio_RG`。

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
