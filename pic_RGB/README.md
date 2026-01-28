# pic_RGB – DNG ROI 提取到 RGB 信号

面向 DNG/RAW 的 ROI+背景选择与统计工具。内置 K-means 预分割生成 ROI/BG 初值，支持 PySide6 GUI 精调、批/单张处理、缓存复用，导出 CSV/JSON/NPY/PNG。

## 功能亮点
- 自动 + 手动：`roi_init/initializer.py` 用内置 K-means 分割结果生成 ROI/BG 初值，GUI 再微调；也可纯手动。
- 批/单张入口：`main.run_single`、`main.run_batch` 统一封装；保留 `pipeline/roi_processor` 批处理。
- 可复用与无界面：同名文件可复用缓存；`interactive=False` 时直接采用自动结果，无需 GUI。
- 完整产物：ROI/BG 统计、通道比、矫正特征矩阵、可视化叠加图。

## 关键模块
- 配置：`config/settings.py`（路径、ROI 参数、输出/缓存/GUI 开关）
- ROI 处理：`pipeline/roi_processor.py`（读取、GUI、统计、批处理主入口）
- GUI 启动器：`gui/launcher.py`（可切换无界面），`gui/roi_gui.py`
- 自动初始化：`roi_init/initializer.py`（K-means 分割推 ROI/BG）
- 封装流程：`pipeline/processor.py`（`ROIWorkflow`），`main/single_process.py`（单张/批量便捷函数）
- K-means 包装：`kmeans/algorithm.py`（封装本仓 `kmeans/advanced.py`）
- 特征与持久化：`features/vector_builder.py`，`persistence/selection_io.py`
- 统计分析：`stats/analyze_statistics.py`

## 快速开始
1) Python 3.10+  
2) 安装依赖（含 K-means 所需）：
```bash
pip install numpy opencv-python matplotlib rawpy pyside6 pandas scikit-learn scipy
```
3) 配置路径  
   - `config/settings.py`: `DATA_DIR`（默认 `h:\000\CDs\pic_RGB\data_set_2`）、`OUTPUT_DIR`（默认 `h:\000\CDs\pic_RGB\data_set_2\roi_results`）、输出开关（`EXPORT_JSON`=True、`EXPORT_CSV`、`EXPORT_PNG`）、GUI/缓存开关（`DEFAULT_INTERACTIVE`、`REUSE_LAST_SELECTION`、`SELECTION_CACHE_PATH`、`OVERWRITE_SELECTION`）。  
   - K-means 逻辑已内置，无需额外依赖外部仓库。

4) 运行方式  
- 经典批处理（保留原入口）  
```bash
python -m pic_RGB.pipeline.roi_processor
```
- 一体化单张（K-means 初值 + GUI）：默认处理 `DATA_DIR` 下首个 DNG  
```bash
python -m pic_RGB.main.single_process
```
- 批量调用（可选无界面模式）  
```powershell
python -c "from pathlib import Path; from pic_RGB.main import run_batch; run_batch(image_dir=Path(r'h:\000\CDs\pic_RGB\data_set_2'), interactive=True)"
```
  其中 `interactive=False` 可跳过 GUI，直接使用自动初值。
- 查看详细统计摘要  
```bash
python -m pic_RGB.stats.analyze_statistics
```

## GUI 快捷键（`gui/roi_gui.py`）
- 鼠标左键拖拽：ROI 模式绘制 ROI；BG 模式移动 BG 中心
- Tab：切换 ROI/BG
- W/A/S/D：移动当前区域中心（ROI 或 BG）
- 方向键：BG 模式调 outer_x/outer_y；ROI 模式调宽高
- Q/E：旋转当前激活区域；B：BG 中心对齐 ROI
- Enter：确认；Q/ESC：取消

## 输出文件（默认写入 `OUTPUT_DIR`）
- `roi_results.json`：每张图的完整结果（几何、通道值、原始统计）
- `roi_detailed_statistics.csv`：ROI/BG 详细统计（均值/方差/分位/饱和度/通道比）
- `roi_selection_cache.json`：最新的 ROI/BG 选择缓存（`config.settings.SELECTION_CACHE_PATH`）
（默认输出：JSON 必开；CSV 由 `EXPORT_CSV` 控制〈默认开启〉；PNG 可视化由 `EXPORT_PNG` 控制〈默认关闭〉）

## 复用与缓存
- 历史结果：`roi_results/roi_results.json`；快速缓存：`roi_results/roi_selection_cache.json`
- `REUSE_LAST_SELECTION=True` 时，如存在同名文件选区会直接复用并跳过 GUI；否则加载历史作为初始值再调整。
- 顶层入口在每次处理后调用 `persistence.save_selection` 更新缓存，可通过 `config.OVERWRITE_SELECTION` 控制是否覆盖。

## 工作流概览
1. 读取 DNG，拆分 Bayer 通道并做黑白电平校正。  
2. 内置 K-means 预分割液体/瓶身，生成 ROI/BG 初值。  
3. GUI 允许用户微调（或 `interactive=False` 直接采用自动结果）。  
4. 计算 ROI/BG 统计，导出 CSV/JSON/NPY/PNG 并缓存选区，后续同名文件可复用。

方法细节见 `pic_RGB/METHODOLOGY.md`。
