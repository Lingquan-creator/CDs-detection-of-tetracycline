# pic_RGB 方法说明 (Methodology)

本文档对应 `extraction/roi_processor.py` 的实现，说明 RAW 读取、ROI/BG 定义、统计量与输出字段。

## 1. 数据读取与预处理

- 使用 `rawpy.imread()` 读取 DNG。
- 从 `raw.raw_image` 拿到 Bayer 原始阵列，并按 RGGB 位置拆分为 4 通道：
  - `R = raw_data[0::2, 0::2]`
  - `G1 = raw_data[0::2, 1::2]`
  - `G2 = raw_data[1::2, 0::2]`
  - `B = raw_data[1::2, 1::2]`
- 黑电平：优先读取 `raw.black_level_per_channel` 或 `raw.camera_black_level_per_channel`；逐通道扣除后截断到 `>= 0`。
- 白电平：优先读取 `raw.white_level` 或 `raw.camera_white_level_per_channel`；用于饱和比例计算（默认 `16383`，14-bit）。

## 2. ROI 与背景环（mask 定义）

### ROI（矩形，可旋转）

ROI 用 `(center_x, center_y, width, height, angle_deg)` 描述，通过 `cv2.boxPoints()` 得到 4 个顶点并填充生成 mask。

### 背景环（空心矩形）

背景环由 “外框 - 内框” 构成，并可与 ROI 使用不同中心：

- 外框尺寸：`(width + 2*outer_x, height + 2*outer_y)`
- 内框尺寸：`(width + 2*inner_x, height + 2*inner_y)`
- `bg_mask = outer_mask & ~inner_mask`
- 额外排除 ROI：`bg_mask &= ~roi_mask`

BG 的中心允许独立于 ROI（便于避开局部杂散光）；GUI 中按 `B` 可把 BG 中心对齐到 ROI 中心。

## 3. 统计量（`compute_roi_statistics`）

对 ROI mask 与 BG mask 分别计算四通道统计，字段命名以 `R/G1/G2/B` 为前缀：

- `*_mean`：均值
- `*_var`：方差
- `*_std`：标准差
- `*_median`：中位数
- `*_p10` / `*_p50` / `*_p90`：10/50/90 分位数
- `*_saturation_ratio`：饱和比例（像素值 `>= 0.95*white_level` 的占比）

通道比值（用于抗整体亮度漂移）：

- `G_mean = (G1_mean + G2_mean) / 2`
- `R_G_ratio = R_mean / (G_mean + ε)`
- `B_G_ratio = B_mean / (G_mean + ε)`
- `R_B_ratio = R_mean / (B_mean + ε)`
- `G1_G2_ratio = G1_mean / (G2_mean + ε)`

其中 `ε = 1e-6` 防止除零。

## 4. 背景扣除与特征

- 背景扣除：`ch_corrected = max(0, roi_ch_mean - bg_ch_mean)`（`R/G1/G2/B` 同理）
- 特征向量：`[R_corrected, G1_corrected, G2_corrected, B_corrected]`
- RGB 显示向量：
  - `R_display = R_corrected`
  - `G_display = max(0, (G1_corrected + G2_corrected) / 2)`
  - `B_display = B_corrected`

## 5. 输出文件与字段

### 5.1 `roi_results.json`

JSON list；每个元素对应一张图，包含几何信息、ROI/BG 均值与扣除值、显示值，以及可选元数据（`exposure_time`, `iso`）。默认不再输出额外 CSV。

### 5.2 `roi_detailed_statistics.csv`

每行对应一张图，包含：

- `filename`
- ROI 详细统计：`roi_` 前缀 + 第 3 节所有统计量与比值（如 `roi_R_mean`, `roi_R_var`, `roi_R_p10`, `roi_R_saturation_ratio`, `roi_R_G_ratio` 等）
- BG 详细统计：`bg_` 前缀 + 同一套统计量与比值

浮点数按 `%.6f` 写入。

### 5.3（可选）`*_roi.png`

开启 `SAVE_VISUALIZATION=True` 时才生成；默认关闭以精简输出。

- 强度图 `Y = max(R, G1, G2, B)`
- ROI/BG mask 可视化
- RGB 伪彩叠加 ROI/BG 轮廓
- 背景扣除后的通道柱状图

## 6. 质量检查建议（常用）

- 过曝提示：任一通道 `*_saturation_ratio > 0.05`
- 绿色通道一致性：`G1_G2_ratio` 明显偏离 1（例如 `<0.8` 或 `>1.2`）
- 噪声/不均匀：`*_var` / `*_std` 异常偏大
- 背景干扰：`bg_*_mean` 偏高

## 7. 复用策略

- 历史选择从 `roi_results.json` 与 `roi_selection_cache.json` 读取（按文件名匹配）。
- `REUSE_LAST_SELECTION=True` 时，若存在同名文件历史选择则跳过 GUI。
- ?????????????????????????????????????????`select_roi_with_gui()` ??? `(roi_sel, bg_sel)`???????????? `process_image(..., roi_selection=roi_sel, bg_selection=bg_sel, interactive=False)`??

