# ROI GUI 手动选取 — 快速上手

新版流程使用 GUI 手动框选 ROI 与背景环，选区确定后即可复用到整批图像。

## 1) 环境与依赖
- Python 3.10+，建议使用虚拟环境  
- 在 `pic_RGB` 目录执行：
```bash
pip install -r requirements.txt
```

## 2) 准备数据
- 将待处理的 DNG 放入 `pic_RGB/data_pic/`
- 输出默认写入 `pic_RGB/roi_results/`（可在 `roi_config.py` 修改）

## 3) 运行方式
- 交互式单张/批量/参数对比：`python test_roi.py`，按提示选择 1/2/3
  - 选项 1：单张，弹出 GUI 选区
  - 选项 2：批量，首张弹出 GUI，后续沿用相同 ROI/BG
  - 选项 3：参数对比，使用图像中心固定选区，不弹 GUI
- 直接批量（同样会在首张弹出 GUI）：`python roi_processor.py`

## 4) GUI 操作提示
- 鼠标左键拖拽：绘制 ROI 矩形（保持当前角度）
- Tab/Space：切换正在调整的区域（ROI / 背景环）
- AWSD：移动 ROI 中心；方向键：背景环激活时调节外边距（上下=外高，左右=外宽），ROI 激活时也可移动
- E/R：ROI 角度 ±；B：将背景环中心对齐到 ROI
- Enter：确认并计算；Q/ESC：取消

## 5) 输出
- `roi_results.json` / `roi_features.csv` / `feature_matrix.npy`
- `roi_detailed_statistics.csv`（运行 detailed 统计时生成）
- `*_roi.png` 可视化

提示：选择好的 ROI/BG 会沿用到同一批次的其余图像，避免重复操作。
