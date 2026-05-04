# OUTPUT 批处理脚本说明

本目录主要保存训练和预测输出，同时包含两个批处理脚本，用来把 `dataset` 下的 ROI 统计文件批量送入三个浓度反演方法。

## 主要代码

- `run_dataset_concentration_training.sh`: 递归查找训练集 ROI CSV，依次训练 EXP、GS、RGB/Ridge 三类模型。
- `predict_dataset_concentration_training.sh`: 使用指定训练集模型预测 `food` 测试集，并按鸡肉/鸡蛋分组评估。

## 训练脚本输入

算法层面的输入是：

- 多个训练数据集的 `roi_detailed_statistics.csv`。
- 每行样品的 `filename`，用于解析真实浓度。
- 三个反演方法需要的 ROI/BG 统计列。

实际默认输入：

- 搜索目录：`dataset`
- 匹配文件：`roi_detailed_statistics.csv`
- 测试集名称：`food`
- Python 环境：`.venv/bin/python`

环境变量：

- `DATASET_DIR`: 覆盖数据集根目录。
- `OUTPUT_DIR`: 覆盖输出根目录。
- `VENV_DIR`: 覆盖虚拟环境目录。
- `TEST_DATASET_NAME`: 默认 `food`，该数据集会跳过训练。
- `ROI_CSV_NAME`: 默认 `roi_detailed_statistics.csv`。

## 训练脚本流程

1. 在 `DATASET_DIR` 下递归查找 `ROI_CSV_NAME`。
2. 跳过路径第一层目录名等于 `TEST_DATASET_NAME` 的 CSV。
3. 对每个训练 CSV 建立对应输出目录：

   ```text
   OUTPUT/<relative_csv_directory>/
   ```

4. 调用：
   - `concentration_EXP/train.py`
   - `concentration_GS/train.py`
   - `concentration_regression/train.py`
5. 分别保存模型、预测、LOO、系数、概率表、图和日志。

## 训练脚本输出

算法层面的输出是：

- 每个训练数据集对应的三类模型。
- 训练集预测和 LOO 评估。
- 可视化图和日志。

实际输出示例：

- `OUTPUT/mate20x/exp_fit_params.csv`
- `OUTPUT/mate20x/gs_amortized_posterior_model.pt`
- `OUTPUT/mate20x/rgb_concentration_model.pkl`
- `OUTPUT/mate20x/rgb_ridge_predictions_loo.csv`
- `OUTPUT/mate20x/*_train.log`

## 预测脚本输入

算法层面的输入是：

- 一个测试集 ROI CSV。
- 已训练的 EXP 参数、RGB/Ridge 模型和 GS 模型。
- 文件名中的真实浓度和样品类型。

实际默认输入：

- 测试 CSV：`dataset/food/roi_detailed_statistics.csv`
- 模型目录：`OUTPUT/mate20x`
- 输出目录：`OUTPUT/food/mate20x_model_predictions`

环境变量：

- `ROI_CSV`: 覆盖测试 CSV。
- `MODEL_DIR`: 覆盖模型目录。
- `PRED_OUTPUT_DIR`: 覆盖预测输出目录。
- `EXP_PARAMS`, `RGB_MODEL`, `GS_MODEL`: 分别覆盖三个模型文件。

## 预测脚本流程

1. 使用 EXP 参数预测测试 CSV，输出 `exp_predictions.csv`。
2. 使用 Ridge 模型预测，输出 `rgb_predictions.csv`。
3. 使用 GS 模型预测，输出后验摘要和概率表。
4. 从 `filename` 提取真实浓度：

   ```text
   chicken120.dng -> 120
   egg90.dng -> 90
   ```

5. 从 `filename` 提取样品组：

   ```text
   chicken* -> chicken
   egg*     -> egg
   ```

6. 分别对 `all`、`chicken`、`egg` 计算 R2、RMSE、MAE、mean error、median abs error、max abs error。
7. 输出总图和分组图。

## 预测脚本输出

算法层面的输出是：

- 三类模型的测试集预测浓度。
- GS 的后验分布和概率表。
- 总体和分组预测精度。

实际默认输出：

- `OUTPUT/food/mate20x_model_predictions/exp_predictions.csv`
- `OUTPUT/food/mate20x_model_predictions/rgb_predictions.csv`
- `OUTPUT/food/mate20x_model_predictions/gs_posterior_summary.csv`
- `OUTPUT/food/mate20x_model_predictions/gs_posterior_probabilities.csv`
- `OUTPUT/food/mate20x_model_predictions/prediction_metrics.csv`
- `OUTPUT/food/mate20x_model_predictions/prediction_metrics.txt`
- `OUTPUT/food/mate20x_model_predictions/prediction_evaluation.png`
- `OUTPUT/food/mate20x_model_predictions/group_evaluation_plots/prediction_evaluation_chicken.png`
- `OUTPUT/food/mate20x_model_predictions/group_evaluation_plots/prediction_evaluation_egg.png`

## 使用方式

```bash
./OUTPUT/run_dataset_concentration_training.sh
./OUTPUT/predict_dataset_concentration_training.sh
```
