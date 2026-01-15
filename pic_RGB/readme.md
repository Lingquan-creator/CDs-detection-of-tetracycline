# ROI GUI Pipeline (All-in-One Guide)

整合了原有说明文件（README_ROI, QUICKSTART, UPDATE_NOTES, 详细统计说明, 键盘/控制说明），方便一次查阅。

## 快速开始
1) 安装依赖（在 `pic_RGB`）：`pip install -r requirements.txt`  
2) 将 DNG 放入 `pic_RGB/data_pic/`（路径可改 `roi_config.py`）  
3) 运行批处理（每张图弹出 GUI：先拖矩形，再在第二窗口调角度/位置/背景环中心与厚度，Enter/Space 确认）：  
   `python roi_processor.py`  
   - 单张/批量引导脚本：`python test_roi.py`  
4) 输出默认写入 `pic_RGB/roi_results/`：`*_roi.png`, `roi_results.json`, `roi_features.csv`, `feature_matrix.npy`, `roi_detailed_statistics.csv`

配置要点（`roi_config.py`）：`DATA_DIR`/`OUTPUT_DIR`，`bg_ring_inner/bg_ring_outer`，`preview_max_dim/preview_percentile`，`SAVE_VISUALIZATION`、`VISUALIZATION_*`。

## GUI 操作
- 窗口1：拖出矩形 ROI（Enter/Space 确认，Esc/C 取消）。  
- 窗口2：滑条微调 **角度/尺寸/位置/背景环中心与厚度**；ROI/BG 中心独立。`+/-` 可微调基准尺寸。  
- 预览会自动按 `preview_max_dim` 缩放。批处理模式会逐张弹窗。

## 键盘/控制（若开启键盘支持）
- ROI 移动：W/A/S/D 或方向键；背景环中心：I/J/K/L；`+/-` 缩放；Enter/Space 确认，Esc/C 取消。实际是否启用键盘取决于当前 GUI 版本（最新默认用滑条；键盘可作为辅助）。

## 更新摘要
- 已移除自动/高级 ROI 模式，改为全 GUI 手动矩形。  
- 背景环可独立调节中心与厚度。  
- 输出格式保持兼容（JSON/CSV/NPY/PNG），新增 `bg_center_x/bg_center_y`、背景环厚度字段。

## 详细统计说明
- `roi_detailed_statistics.csv`：ROI 与背景的均值/方差/中位数/分位数(p10/p50/p90)/饱和比例/通道比值 (R/G, B/G, R/B, G1/G2)。  
- 生成方式：运行 `roi_processor.py` 或 `test_detailed_stats.py`。  
- 典型质量检查：饱和比例 >0.05 可能过曝；G1/G2 明显偏离 1.0 需关注。

## 工作流简述
1) `rawpy` 读取 DNG，拆分 RGGB，做黑电平校正。  
2) 预览强度图 `Y=max(R,G1,G2,B)`，GUI 手动框选 ROI。  
3) 生成 ROI 与同心（可独立中心）的背景环 mask，计算通道统计与背景校正。  
4) 保存可视化与 JSON/CSV/NPY 结果，可选详细统计 CSV。
