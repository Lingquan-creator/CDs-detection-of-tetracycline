# pic_RGB

本目录从 DNG/RAW 图像中选择样品 ROI 和背景区域，提取后续浓度模型需要的 Bayer 四通道均值、背景扣除信号和基础 ln 特征。

当前主流程是 GUI 辅助的手动 ROI/BG 选择，选择结果可缓存复用。代码默认保存 JSON、CSV 和 ROI 可视化 PNG。

## 主要文件

- `roi_config.py`: 输入输出路径、默认 ROI/BG 尺寸、GUI 步长、可视化和缓存设置。
- `roi_processor.py`: 主入口。批量读取 DNG，选择 ROI/BG，计算通道均值和特征，保存结果。
- `roi_gui.py`: PySide6 交互式 ROI/BG 选择界面。
- `roi_manual.py`: 生成旋转矩形 ROI mask 和背景空心矩形 mask。
- `roi_auto.py`: 自动 ROI 分割工具，当前主流程未调用。
- `roi_distribution_features.py`: 分布特征工具，当前精简输出流程未调用。
- `analyze_statistics.py`: 历史统计 CSV 分析示例。
- `algorithm.md`: 更完整的代码和算法说明。

## 当前默认配置

`roi_config.py` 当前默认：

```text
DATA_DIR = dataset/test
OUTPUT_DIR = dataset/test/roi_results
IMAGE_EXTENSION = *.dng
SAVE_VISUALIZATION = True
VISUALIZATION_MAX_SIDE = 1600
REUSE_LAST_SELECTION = False
SELECTION_CACHE_PATH = OUTPUT_DIR / roi_selection_cache.json
```

默认 ROI/BG 参数：

```text
fixed_roi_width = 160
fixed_roi_height = 400
measurement_roi_width_fraction = 0.4
measurement_roi_height_fraction = 0.4
trim_fraction = 0.08
roi_angle_deg = 0.0
bg_ring_inner_x = 80
bg_ring_inner_y = 80
bg_ring_outer_x = 240
bg_ring_outer_y = 240
move_step = 1
size_step = 1
angle_step_deg = 0.5
```

## 运行

默认路径批量处理：

```bash
python pic_RGB/roi_processor.py
```

指定输入输出目录：

```bash
PIC_RGB_DATA_DIR=dataset/train \
PIC_RGB_OUTPUT_DIR=dataset/train/roi_results \
python pic_RGB/roi_processor.py
```

```bash
PIC_RGB_DATA_DIR=dataset/test \
PIC_RGB_OUTPUT_DIR=dataset/test/roi_results \
python pic_RGB/roi_processor.py
```

## 输入

主流程读取 `DATA_DIR` 下的 `.dng` 文件。元数据可来自：

- 文件名：支持 `食品-批次-浓度.dng`、`浓度_曝光时间.dng` 等形式。
- 同目录 `metadata.txt`: 支持 `key=value`，可提供 `exposure_time`, `iso`, `R_gain`, `G_gain`, `B_gain`, `gamma/gama`, `saturability` 等。
- DNG 元数据：尝试读取 shutter、ISO、黑电平和白电平。

## RAW 通道拆分

`roi_processor.py` 使用 `rawpy` 读取 RAW，并按 RGGB 方式拆成四通道：

```text
R  = raw[0::2, 0::2] - black_level[0]
G1 = raw[0::2, 1::2] - black_level[1]
G2 = raw[1::2, 0::2] - black_level[2]
B  = raw[1::2, 1::2] - black_level[3]
```

当前实现会直接创建最终四通道数组，逐通道扣黑电平并裁剪到非负，避免先复制整张 RAW 再转 float32 带来的内存峰值。

## GUI 操作

- 鼠标拖拽：绘制当前激活框。
- Tab：切换 ROI / BG inner / BG outer。
- W/A/S/D：移动当前激活区域中心。
- 方向键：调整当前激活框宽高。
- Q/E：按 `angle_step_deg` 旋转。
- B：将背景框中心对齐到 ROI。
- Enter：确认。
- ESC：取消。

## ROI/BG 复用

程序会读取历史 `roi_results.json` 和 `roi_selection_cache.json` 作为初始选择。

- `REUSE_LAST_SELECTION=True`: 若同名文件已有缓存，跳过 GUI 直接复用。
- `REUSE_LAST_SELECTION=False`: 若同名文件已有缓存，只作为 GUI 初始值，仍可调整。

每张图处理成功后都会更新 `roi_selection_cache.json`。

## 输出

输出目录为 `OUTPUT_DIR`。主流程生成：

- `roi_results.json`: 每张图的 ROI/BG 几何、通道均值、背景扣除值、显示值、元数据和源文件路径。
- `roi_selection_cache.json`: ROI/BG 选择缓存。
- `roi_detailed_statistics.csv`: 后续浓度模型的主要输入表。
- `<stem>_roi.png`: ROI/BG 可视化图。

`roi_detailed_statistics.csv` 当前核心列：

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

下游 `concentration_regression.physics_features` 会在此基础上继续构造 `ln_RGB_sum`、`ln_RG_sum`、`ln_RB_sum`、`ln_GB_sum` 和 ln 比值项。

## 可视化和内存

`SAVE_VISUALIZATION=True` 时会输出 `<stem>_roi.png`。当前可视化会先按 `VISUALIZATION_MAX_SIDE=1600` 生成缩略图，再缩放 ROI/BG 框坐标，降低 matplotlib 渲染阶段的内存峰值。保存后会关闭 figure，批处理每张图结束后会释放主要数组引用。

## 方法细节

更详细的实现说明见 `pic_RGB/algorithm.md`。历史方法说明可参考 `pic_RGB/METHODOLOGY.md`。
