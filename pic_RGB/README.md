# pic_RGB

本目录从 DNG/RAW 图像中选择样品 ROI 和背景区域，提取后续浓度模型需要的 Bayer 四通道均值、背景扣除信号和 6 维输入特征。

当前主流程是 GUI 辅助的手动 ROI/BG 选择，选择结果可缓存复用。代码只保存当前浓度反演需要的 JSON/CSV，默认不再输出 NPY、分布特征 CSV 或每张图的 PNG。

## 主要文件

- `roi_config.py`: 输入输出路径、默认 ROI/BG 尺寸、GUI 步长、可视化和缓存设置。
- `roi_processor.py`: 主入口。批量读取 DNG，选择 ROI/BG，计算通道均值和特征，保存结果。
- `roi_gui.py`: PySide6 交互式 ROI/BG 选择界面。
- `roi_manual.py`: 生成旋转矩形 ROI mask 和背景空心矩形 mask。
- `roi_auto.py`: 自动 ROI 分割工具，当前主流程未调用。
- `roi_distribution_features.py`: 分布特征工具，当前精简输出流程未调用。
- `analyze_statistics.py`: 历史统计 CSV 分析示例。
- `algorithm.md`: 更完整的代码和算法说明。

## 默认配置

当前 `roi_config.py` 默认：

```text
DATA_DIR = dataset/food
OUTPUT_DIR = dataset/food
IMAGE_EXTENSION = *.dng
SAVE_VISUALIZATION = False
REUSE_LAST_SELECTION = False
SELECTION_CACHE_PATH = OUTPUT_DIR / roi_selection_cache.json
```

默认 ROI/BG 参数：

```text
fixed_roi_width = 160
fixed_roi_height = 400
roi_angle_deg = 0.0
bg_ring_inner_x = 80
bg_ring_inner_y = 80
bg_ring_outer_x = 240
bg_ring_outer_y = 240
move_step = 2
size_step = 20
angle_step_deg = 0.5
```

## 安装依赖

```bash
pip install numpy opencv-python matplotlib rawpy pyside6
```

## 运行

默认路径批量处理：

```bash
python pic_RGB/roi_processor.py
```

指定输入输出目录：

```bash
PIC_RGB_DATA_DIR=dataset/mate3_2 \
PIC_RGB_OUTPUT_DIR=dataset/mate3_2 \
python pic_RGB/roi_processor.py
```

## 输入

主流程读取 `DATA_DIR` 下的 `.dng` 文件。元数据可来自：

- 文件名：支持 `浓度_曝光时间.dng`，兼容 `浓度.曝光时间.dng`。
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

扣黑电平后会裁剪到非负。

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

`roi_detailed_statistics.csv` 当前列：

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

当前保存的 6 维输入特征：

```text
ln_R_sig, ln_G_sig, ln_B_sig, Ratio_RG, Delta_RG, Delta_GB
```

## 可视化

默认 `SAVE_VISUALIZATION=False`，不保存每张图的可视化 PNG。若改为 `True`，会输出 `<stem>_roi.png`，包含强度图、ROI/BG mask、伪彩预览和背景校正通道柱状图。

## 方法细节

更详细的实现说明见 `pic_RGB/algorithm.md`。如果需要历史方法说明，可参考 `pic_RGB/METHODOLOGY.md`。
