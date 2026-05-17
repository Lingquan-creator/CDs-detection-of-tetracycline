# application

本目录是本地桌面应用，用于打开 DNG 图像、手动选择 ROI/BG，并调用已保存模型输出浓度预测。

## 运行

```bash
cd /home/jzh/CDs
./.venv/bin/python application/app.py
```

基本流程：

1. 打开 `.dng` 图像。
2. 在图像上调整 ROI、BG inner、BG outer。
3. 选择模型包和模型类型。
4. 点击 `Predict` 输出浓度预测。

## 模型包

模型注册表为：

```text
application/model_registry.json
```

当前包含历史手机模型包和本轮整理的模型包：

```text
mate20x
mate3_2
xiaomi
vivo100x
current_work
```

`current_work` 来自当前 `dataset/train`、`dataset/test` 和 `optimization` 结果，文件位于：

```text
application/models/current_work/
```

包含：

```text
rgb_concentration_model.pkl
exp_fit_params.csv
gs_amortized_posterior_model.pt
best_optimized_combinations.csv
```

## 模型类型

- `Regression`: `concentration_regression` 保存的 Ridge 模型。
- `EXP`: `concentration_EXP` 保存的响应函数反演参数。
- `GS posterior`: `concentration_GS` 保存的摊销后验模型，应用中默认显示后验中位数。

应用已支持当前优化中使用的 ln 派生特征，包括 `ln_RGB_sum`、`ln_RG_sum`、`ln_RB_sum`、`ln_GB_sum`、`ln_R_over_G`、`ln_B_over_G`、`ln_RG_over_GB` 和 `ln_RB_over_GB`。

## 结果材料

当前工作区结果副本位于：

```text
application/results/current_work/
```

其中 `csv/` 保存汇总表和预测表，`fig/` 保存回收率热图，`README.md` 说明复制来源。

## 打包

Linux:

```bash
cd application
bash build_linux.sh
```

Windows PowerShell:

```powershell
cd application
.\build_windows.ps1
```
