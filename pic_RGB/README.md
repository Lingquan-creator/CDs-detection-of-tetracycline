# pic_RGB – DNG ROI 提取到 RGB 信号

面向 DNG/RAW 的 ROI+背景选择与统计工具，支持 PySide6 GUI/可视化输出，导出 CSV/JSON/NPY/PNG。

## 模块
- 配置：`config/roi_config.py`
- ROI 提取入口：`extraction/roi_processor.py`（批处理与输出）
- ROI GUI：`extraction/roi_gui.py`（PySide6） / `extraction/roi_gui_legacy.py`（OpenCV）
- ROI/BG 可复用工具：`processing/roi_manual.py`（遮罩/旋转框）、`processing/roi_auto.py`（自动检测）
- 统计分析：`stats/analyze_statistics.py`

## 安装与运行
1. Python 3.10+
2. 依赖安装：
   ```bash
   pip install numpy opencv-python matplotlib rawpy pyside6 pandas
   ```
3. 在 `config/roi_config.py` 中确认路径：
   - `DATA_DIR`: 输入 DNG 目录（默认 `h:\000\CDs\pic_RGB\data_pic`）
   - `OUTPUT_DIR`: 输出目录（默认 `h:\000\CDs\pic_RGB\roi_results`）
4. 在仓库根目录运行 ROI 批处理（需以包形式运行）：
   ```bash
   python -m pic_RGB.extraction.roi_processor
   ```
5. 查看详细统计 CSV：
   ```bash
   python -m pic_RGB.stats.analyze_statistics
   ```

说明：目前按“文件名复用”逻辑选区，首张新文件会弹出 GUI，后续可复用缓存或再次手动调整。

## GUI 操作（`extraction/roi_gui.py`）
- 鼠标左键拖拽：绘制 ROI（保持当前角度）
- Tab：切换 ROI/BG
- W/A/S/D：移动当前区域中心；方向键（BG 模式）调整 outer_x/outer_y；方向键（ROI 模式）调整宽高
- E/R：旋转 ROI；B：对齐 BG 与 ROI 中心
- Enter：确认；Q/ESC：取消

## 输出文件（默认 `OUTPUT_DIR`）
- `roi_results.json`：每张图的完整结果（多边形、通道值、原始统计）
- `roi_features.csv`：归一化/矫正后的通道特征
- `feature_matrix.npy`：`(N,4)` 通道矩阵 `[R, G1, G2, B]`
- `roi_detailed_statistics.csv`：ROI/BG 详细统计（均值/方差/分位/饱和度/通道比）
- `*_roi.png`：可视化叠加图（受 `SAVE_VISUALIZATION` 控制）

## 复用与缓存
- 快速缓存：`roi_results/roi_selection_cache.json`
- 历史结果：`roi_results/roi_results.json`
- `config/roi_config.py` 的 `REUSE_LAST_SELECTION`：
  - `True`：如存在同名文件的选区，跳过 GUI 直接复用
  - `False`：加载历史作为初始值，可继续调整

## 方法细节
见 `pic_RGB/METHODOLOGY.md`。
