# ROI GUI 手动选取与特征提取

基于 GUI 手动框选 ROI 与背景环，提取通道特征并输出可视化/CSV/JSON。

## 功能概览
- DNG 读取：使用 `rawpy` 拆分 RGGB 四通道并扣除黑电平
- GUI 选区：矩形 ROI 可调位置/尺寸/角度，背景环可独立调整中心与厚度
- 快速批量：首张图确定 ROI/BG 后复用到整批图像
- 输出：背景校正后的 R/G1/G2/B，RGB 显示向量，可视化 PNG、JSON、CSV、NPY

## 使用方法
1. 安装依赖（在 `pic_RGB` 目录）  
   `pip install -r requirements.txt`
2. 把 DNG 放到 `data_pic/`，输出默认写到 `roi_results/`
3. 运行：
   - `python test_roi.py` 交互选择 1/2/3（单张/批量/参数对比）
   - 或 `python roi_processor.py` 直接批量（首张弹出 GUI）

### GUI 快捷键
- 鼠标拖拽：绘制 ROI（保持当前角度）
- Tab：切换 ROI / 背景环
- AWSD：移动当前激活区域的中心；方向键（仅 BG 激活时）：调整外边距（上下=外高，左右=外宽）
- E/R：ROI 角度 ±（更细步长 0.5°，滑块精度 0.1°）；B：背景环中心对齐 ROI
- Enter：确认；Q/ESC：取消

## 输出文件
- `roi_results.json`：完整记录
- `roi_features.csv`：特征表（含 ROI/BG 几何信息）
- `roi_detailed_statistics.csv`：详细统计（运行 detailed 流程时生成）
- `feature_matrix.npy`：形状 (N, 4) 的特征矩阵
- `*_roi.png`：可视化覆盖图

提示：选定的 ROI/BG 会在同一批处理过程中复用，避免重复操作。
`roi_config.py` 中的 `REUSE_LAST_SELECTION` 可控制对同名文件的行为：为 True 时若在 `roi_results/roi_results.json`（或缓存）找到上次选择则直接计算不弹 GUI；为 False 时若有历史选择则作为初始值打开 GUI 供微调。缓存文件：`roi_results/roi_selection_cache.json`（单次运行的轻量记录）。
