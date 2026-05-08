# pic_RGB 代码说明

本目录负责从 DNG/RAW 图像中选取样品 ROI 和背景区域，提取后续浓度反演需要的 Bayer 四通道均值、背景扣除信号和 6 维输入特征。当前主流程是 GUI 辅助的手动 ROI/BG 选择，选择结果可缓存复用。

## 代码文件

- `roi_config.py`: 配置输入输出目录、默认 ROI/BG 尺寸、GUI 步长、是否保存可视化图和是否复用历史选择。
- `roi_processor.py`: 主入口。批量读取 DNG，解析元数据，调用 GUI 或缓存确定 ROI/BG，计算通道均值和特征，保存 JSON/CSV。
- `roi_gui.py`: PySide6 交互界面。支持 ROI、背景内框、背景外框独立拖拽、移动、缩放和旋转。
- `roi_manual.py`: 根据中心、宽、高和角度生成旋转矩形 ROI mask，以及背景空心矩形 mask。
- `roi_auto.py`: 自动 ROI 分割和估计工具，当前主流程没有调用。
- `roi_distribution_features.py`: ROI 像素分布特征工具，当前精简输出流程没有调用。
- `analyze_statistics.py`: 旧统计 CSV 的分析示例脚本，路径和列名偏历史格式。
- `test_roi.py`, `test_detailed_stats.py`: 测试/示例脚本。

## 默认配置

`roi_config.py` 当前默认：

```text
DATA_DIR = dataset/food
OUTPUT_DIR = dataset/food
IMAGE_EXTENSION = *.dng
SAVE_VISUALIZATION = False
REUSE_LAST_SELECTION = False
SELECTION_CACHE_PATH = OUTPUT_DIR / roi_selection_cache.json
```

默认 ROI/BG 几何参数：

```text
fixed_roi_width = 160
fixed_roi_height = 400
roi_angle_deg = 0.0
bg_ring_inner_x = 80
bg_ring_inner_y = 80
bg_ring_outer_x = 240
bg_ring_outer_y = 240
move_step = 1
size_step = 1
spin_repeat_interval_ms = 100
angle_step_deg = 0.5
```

输入输出目录可用环境变量覆盖：

```bash
PIC_RGB_DATA_DIR=/path/to/dngs
PIC_RGB_OUTPUT_DIR=/path/to/output
python pic_RGB/roi_processor.py
```

## 输入数据

主流程输入为 `DATA_DIR` 下的 `.dng` 文件。元数据来源按优先级组合：

- 文件名：支持 `浓度_曝光时间.dng`，也兼容 `浓度.曝光时间.dng`。若不能完整解析，浓度可由后续模型从文件名中提取数字。
- 同目录 `metadata.txt`: 支持 `key=value`，可提供 `exposure_time`, `iso`, `R_gain`, `G_gain`, `B_gain`, `gamma/gama`, `saturability` 等。
- DNG 元数据：尝试读取 shutter、ISO、黑电平和白电平。

## RAW 通道读取

`load_raw_image()` 使用 `rawpy` 读取 Bayer RAW 数据，扣除每通道黑电平并裁剪到非负，然后按 RGGB 方式拆分四个通道：

```text
R  = raw[0::2, 0::2] - black_level[0]
G1 = raw[0::2, 1::2] - black_level[1]
G2 = raw[1::2, 0::2] - black_level[2]
B  = raw[1::2, 1::2] - black_level[3]
```

四通道随后堆叠为：

```text
raw_channels[:, :, 0] = R
raw_channels[:, :, 1] = G1
raw_channels[:, :, 2] = G2
raw_channels[:, :, 3] = B
```

## ROI/BG 选择

每张图像的选择流程：

1. 默认 ROI 初始化在图像中心，尺寸来自 `fixed_roi_width/fixed_roi_height`。
2. 默认背景由内框和外框构成，中心默认与 ROI 相同，尺寸为 ROI 尺寸加上对应的 inner/outer offset。
3. 如果存在 `roi_results.json` 或 `roi_selection_cache.json`，会读取上次选择作为初值。
4. 当 `REUSE_LAST_SELECTION=True` 且当前文件名存在缓存时，跳过 GUI 直接复用。
5. 否则打开 GUI 重新确认 ROI、背景内框和背景外框。
6. 每次处理成功后都会把选择写回 `roi_selection_cache.json`。

背景 mask 由外框减去内框得到，并额外排除 ROI 区域：

```text
bg_mask = outer_rect_mask & ~inner_rect_mask & ~roi_mask
```

## 统计和特征

`compute_roi_statistics()` 当前只计算四通道均值：

```text
R_mean, G1_mean, G2_mean, B_mean
```

主流程分别对 ROI mask 和 BG mask 计算均值，并得到背景扣除值：

```text
R_corrected  = roi_R_mean  - bg_R_mean
G1_corrected = roi_G1_mean - bg_G1_mean
G2_corrected = roi_G2_mean - bg_G2_mean
B_corrected  = roi_B_mean  - bg_B_mean
G_corrected  = (G1_corrected + G2_corrected) / 2
```

写入 CSV 前会将信号裁剪到不小于 `0.1`：

```text
R_sig = max(R_corrected, 0.1)
G_sig = max((G1_corrected + G2_corrected) / 2, 0.1)
B_sig = max(B_corrected, 0.1)
```

当前保存的 6 维输入特征为：

```text
ln_R_sig = log(R_sig + 1e-6)
ln_G_sig = log(G_sig + 1e-6)
ln_B_sig = log(B_sig + 1e-6)
Ratio_RG = R_sig / (G_sig + 1e-6)
Delta_RG = (G_sig - R_sig) / (G_sig + R_sig + 1e-6)
Delta_GB = (B_sig - G_sig) / (B_sig + G_sig + 1e-6)
```

注意：CSV 中保留了 `roi_*_mean` 和 `bg_*_mean`，因此 `concentration_regression` 和 `concentration_GS` 仍可在训练时重新构造 `Ratio_BG`、曝光归一化特征等未直接写出的特征。

## 可视化

当 `SAVE_VISUALIZATION=True` 时，每张图会输出 `<stem>_roi.png`，内容包括：

- `Y = max(R,G1,G2,B)` 强度图。
- ROI mask。
- 背景环 mask。
- RGB 伪彩预览和 ROI/BG 叠加。
- 背景校正后的四通道柱状图。

默认关闭可视化，只保存 JSON/CSV。

## 输出文件

输出目录为 `OUTPUT_DIR`。主流程生成：

- `roi_results.json`: 每张图的 ROI/BG 几何信息、通道均值、背景扣除值、显示值、元数据和源文件路径。
- `roi_selection_cache.json`: 每张图的 ROI/BG 选择缓存。
- `roi_detailed_statistics.csv`: 后续浓度模型的主要输入文件。

`roi_detailed_statistics.csv` 当前列包括：

```text
filename, concentration, exposure_time, iso
center_x, center_y, width, height, angle_deg
bg_inner_center_x, bg_inner_center_y
bg_outer_center_x, bg_outer_center_y
bg_inner_x, bg_inner_y, bg_outer_x, bg_outer_y, bg_angle_deg
roi_R_mean, roi_G1_mean, roi_G2_mean, roi_B_mean
bg_R_mean, bg_G1_mean, bg_G2_mean, bg_B_mean
R_sig, G_sig, B_sig
ln_R_sig, ln_G_sig, ln_B_sig
Ratio_RG, Delta_RG, Delta_GB
```

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
