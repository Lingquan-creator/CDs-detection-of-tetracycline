# dataset

本目录保存 DNG 图像、ROI 选择缓存和 ROI 统计输出。当前建模主流程使用 `train` 和 `test` 两个数据集。

## 当前主数据

```text
dataset/train/
dataset/test/
```

每个目录下包含 `.dng` 原始图像和 `roi_results/`：

```text
roi_results.json
roi_selection_cache.json
roi_detailed_statistics.csv
*_roi.png
```

`roi_detailed_statistics.csv` 是 `concentration_regression`、`concentration_EXP`、`concentration_GS` 和 `optimization` 的主要输入表。

## 文件命名

当前样品文件名通常包含食品、批次和浓度：

```text
chicken-1-30.dng
egg-3-9.dng
```

下游脚本会从文件名解析：

- `sample_group`: 食品类别，例如 `chicken`、`egg`。
- `batch`: 批次编号。
- `concentration`: 浓度，通常取最后一个数字 token。

测试集中原先误写的 `egg-3-8` 已修正为 `egg-3-9`，对应 DNG、ROI PNG、CSV 和 JSON 引用已同步。

## ROI 输出字段

`roi_detailed_statistics.csv` 至少包含：

```text
filename, concentration, exposure_time, iso
roi_R_mean, roi_G1_mean, roi_G2_mean, roi_B_mean
bg_R_mean, bg_G1_mean, bg_G2_mean, bg_B_mean
R_sig, G_sig, B_sig
ln_R_sig, ln_G_sig, ln_B_sig
Ratio_RG, Delta_RG, Delta_GB
```

下游 `concentration_regression.physics_features` 会继续派生当前优化使用的 ln 加和和 ln 比值特征。

## 生成或刷新 ROI 统计

训练集：

```bash
PIC_RGB_DATA_DIR=dataset/train \
PIC_RGB_OUTPUT_DIR=dataset/train/roi_results \
python pic_RGB/roi_processor.py
```

测试集：

```bash
PIC_RGB_DATA_DIR=dataset/test \
PIC_RGB_OUTPUT_DIR=dataset/test/roi_results \
python pic_RGB/roi_processor.py
```

默认 `pic_RGB` 会保存 `<stem>_roi.png` 可视化图，并复用 `roi_selection_cache.json` 作为 GUI 初始框。

## 其他数据

- `dataset/data/`: 额外 DNG 数据。
- `dataset/old/`: 历史数据和历史 ROI 输出。
- `dataset/000/`: 压缩归档副本。

这些目录不作为当前 `optimization` 和 `OUTPUT/run_best_models_and_recovery.py` 的默认输入。
