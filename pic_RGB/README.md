# pic_RGB — DNG ROI 选区与 RGB 特征提取

本目录提供一套基于 GUI 的 ROI（可旋转矩形）+ 背景环（空心矩形）选区工具，用于从 DNG/RAW 中提取 R/G1/G2/B 通道统计、做背景扣除，并输出 CSV/JSON/NPY 与可视化 PNG。

- 主脚本：`roi_processor.py`（批处理）
- GUI：`roi_gui.py`（PySide6）
- 配置：`roi_config.py`（输入/输出路径、默认参数、是否保存可视化、复用策略）

## 安装与运行

1. 准备 Python 3.10+
2. 安装依赖（示例）：
   ```bash
   pip install numpy opencv-python matplotlib rawpy pyside6 pandas
   ```
3. 确认/修改 `roi_config.py` 中的路径：
   - `DATA_DIR`：输入 DNG 目录（默认 `h:\000\CDs\pic_RGB\data_pic`）
   - `OUTPUT_DIR`：输出目录（默认 `h:\000\CDs\pic_RGB\roi_results`）
4. 在 `pic_RGB` 目录运行：
   ```bash
   python roi_processor.py
   ```

说明：当前实现按“文件名”复用历史选择。第一次处理一批新文件时，可能会逐张弹出 GUI；若希望“一次选区复用整批”，参考下方 `test_detailed_stats.py` 的用法。

## 常用脚本

- `python roi_processor.py`：批量处理 `DATA_DIR` 下所有 `*.dng`，生成输出文件
- `python test_roi.py`：交互测试菜单
  - 1 单张（GUI）
  - 2 批量（逐张处理；在需要时弹 GUI）
  - 3 参数对比（无 GUI，使用图像中心固定 ROI/BG）
  - 4 运行全部
- `python test_detailed_stats.py`：示例——先在第一张图上用 GUI 选一次 ROI/BG，然后复用到后续图像（默认只跑前 3 张，可改 `test_count`）
- `python analyze_statistics.py`：读取并打印 `roi_detailed_statistics.csv` 的分析示例

## GUI 操作（`roi_gui.py`）

- 鼠标左键拖拽：绘制 ROI（保留当前角度）
- Tab：切换当前激活区域（ROI / 背景环）
- W/A/S/D：移动当前激活区域中心
- 方向键：
  - BG 激活：调整背景环外边距（`outer_x/outer_y`）
  - ROI 激活：调整 ROI 宽高（`width/height`）
- E / R：旋转 ROI（并同步背景环角度，步长来自 `roi_config.py` 的 `angle_step_deg`）
- B：将背景环中心对齐到 ROI 中心
- Enter：确认并返回；Q / ESC：取消

## 输出文件

输出目录默认是 `OUTPUT_DIR`：

- `roi_results.json`：每张图的完整结果（几何 + 通道值 + 元数据）
- `roi_features.csv`：扁平化特征表（适合直接用 Excel / pandas）
- `feature_matrix.npy`：形状 `(N, 4)` 的特征矩阵 `[R,G1,G2,B]`（背景扣除后）
- `feature_matrix_full.npy`：增强输入向量，包含 4 维 RAW 均值 + RAW 四通道分布参数 + RGB 三通道分布参数
- `feature_matrix_rgb_full.npy`：RGB 输入向量，包含 3 维 `[R,G,B]` + ROI 内 RGB 三通道分布参数
- `roi_distribution_features.csv`：RAW 四通道（`R/G1/G2/B`）ROI 强度分布参数
- `roi_rgb_distribution_features.csv`：RGB 三通道（`R/G/B`）ROI 强度分布参数
- `roi_detailed_statistics.csv`：ROI 与 BG 的详细统计（均值/方差/分位数/饱和比例/通道比值）
- `*_roi.png`：可视化覆盖图（由 `SAVE_VISUALIZATION=True` 控制）

新增的分布参数会同时尝试 `Gaussian / Poisson / Gamma` 三种模型，并输出：

- `best_model_code`：最佳模型编码（`1=Gaussian`, `2=Poisson`, `3=Gamma`）
- `best_model_margin`：最佳模型相对第二名的 AIC 优势
- 拟合参数：`gauss_mu / gauss_sigma / poisson_lambda / gamma_shape / gamma_scale`
- 分布形状与频域特征：`fano_factor / skewness / kurtosis / entropy / fft_* / mgf_t1~t4`

## 复用与缓存

- 选择缓存文件：`roi_results/roi_selection_cache.json`
- 历史结果也可作为复用来源：`roi_results/roi_results.json`
- `roi_config.py` 的 `REUSE_LAST_SELECTION`：
  - `True`：若存在同名文件的历史选择，则跳过 GUI 直接计算
  - `False`：若存在同名文件的历史选择，则以其作为 GUI 初始值（仍可调整）

## 方法细节

见 `pic_RGB/METHODOLOGY.md`。

