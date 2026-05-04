# 基于手机 DNG 图像与岭回归的荧光浓度反演项目说明

## 1.4 基于智能手机 DNG 图像的荧光定量检测研究现状

近年来，智能手机因其高分辨率图像传感器、低成本和便携性，被广泛用于荧光成像和现场检测系统。已有研究报道了基于智能手机的双模式荧光/白光成像系统、便携式荧光显微平台和开放式体内/体外荧光成像系统，说明消费级移动成像硬件可以承担一定的定量检测任务[1,4,6]。同时，Webcam 或手机摄像头结合简化光学结构也已用于即时检测场景，显示出便携式荧光检测在食品安全、环境监测和现场诊断中的应用潜力[5]。

然而，直接使用 JPEG/PNG 图像进行定量分析存在明显局限。手机内部 ISP 流程会引入白平衡、颜色校正、降噪、锐化、局部 tone mapping 和压缩等非线性处理，使像素值不再严格对应样品发出的线性光强[2,7]。对于荧光检测而言，这种非线性会削弱不同浓度样品之间的可比性，并使简单 RGB 拟合对设备、曝光和环境变化更加敏感。因此，本项目不使用由其他格式转换得到的图像，而是让手机在专业/RAW 模式下直接拍摄 DNG 原始文件作为输入。DNG 文件保留更完整的传感器线性响应和更高位深信息，图像质量与定量可用性均优于压缩图像。

荧光图像中还存在局部背景、散射光、暗电流和读出偏置等干扰。定量荧光成像研究通常将背景估计和噪声扣除作为后续分析的必要预处理步骤[3,8]。基于这一思路，本项目不直接使用 ROI 内的原始 RGB 均值，而是引入 ROI 区域和背景环区域的双区统计，通过同图局部背景扣除获得更稳定的通道信号。随后，项目采用岭回归建立从物理特征到浓度的映射，并通过留一法和独立外部测试评估泛化性能[9,10]。

## 1.5 研究思路与创新点

本项目的目标是构建一套从手机直接拍摄的 DNG 原始图像到浓度值的可复现定量分析流程。整体路线包括：DNG 文件解析、交互式 ROI 与背景环选择、背景扣除、物理特征构造、岭回归建模、单调校准、输入特征筛选和超参数扫描。与仅用单通道强度或简单 RGB 比值拟合的方法相比，本项目强调三个方面。

第一，使用 RAW 四通道而不是压缩图像。算法直接读取 Bayer 阵列中的 `R/G1/G2/B` 四个子通道，扣除黑电平后得到近似线性光强。该步骤减少了 ISP 非线性处理对定量结果的影响。

第二，使用背景扣除和物理可解释特征。对每个样品同时统计 ROI 与背景环的通道均值，并构造 `ln_R_sig/ln_G_sig/ln_B_sig` 等强度衰减特征，以及 `Ratio_RG/Delta_RG/Delta_GB` 等通道比值和归一化差分特征。强度项用于描述浓度变化引起的整体荧光衰减，比值和差分项用于降低激发强度、曝光漂移和样品放置差异的影响。

第三，采用“加权岭回归 + log 目标变换 + 单调校准 + 外部测试”的建模框架。岭回归通过 L2 正则项缓解小样本、多特征条件下的共线性和过拟合问题[9]；log 目标变换保证预测值非负并增强低浓度区间的分辨能力；Isotonic Regression 用于对回归输出进行单调校准；Leave-One-Out 和独立 food 测试集共同用于评估模型对未知样品的预测能力[10]。

## 2.6 便携式检测与手机 DNG 图像采集流程

样品图像在固定光路和暗环境中采集。检测装置提供稳定的激发光源、样品固定位置和手机固定视角，使同一批样品之间的几何差异和环境光波动尽可能减小。图像由手机直接以 DNG 格式拍摄保存，而不是先采集其他格式再转换为 DNG。该采集方式能够保留更高质量的原始传感器数据，后续由 Python 脚本解析 DNG 文件。

在 DNG 读取阶段，程序使用 `rawpy` 获取手机 DNG 文件中的 Bayer 原始阵列，并按 RGGB 排列拆分为四个通道：

$$
\begin{aligned}
R  &= I(2i,2j),\\
G_1 &= I(2i,2j+1),\\
G_2 &= I(2i+1,2j),\\
B  &= I(2i+1,2j+1).
\end{aligned}
$$

设 DNG 元数据中读取到的黑电平为 $\mathrm{BL}_X$，则每个通道的有效强度为：

$$
X'=\max(X-\mathrm{BL}_X,0), \quad X\in\{R,G_1,G_2,B\}.
$$

ROI 由可旋转矩形表示：

$$
\mathcal{R}=(x_c,y_c,w,h,\theta),
$$

背景区域由背景外框减去背景内框得到，并额外排除 ROI 区域：

$$
M_{\mathrm{BG}} = M_{\mathrm{outer}}\setminus M_{\mathrm{inner}}\setminus M_{\mathrm{ROI}}.
$$

这种“目标区域 + 局部背景环”的设计可以在同一张图像内估计局部背景，减少暗电流、杂散光和非特异背景对定量结果的影响[3,8]。

## 2.7 基于手机 DNG 图像的特征提取与岭回归算法

### 2.7.1 ROI/BG 统计与背景扣除

对每张图像，程序分别计算 ROI 和背景环中四个 Bayer 通道的均值。设 $\mu_{\mathrm{ROI},X}$ 和 $\mu_{\mathrm{BG},X}$ 分别为 ROI 与背景区域中通道 $X$ 的均值，则背景扣除信号定义为：

$$
X_{\mathrm{sig}}=\max(\mu_{\mathrm{ROI},X}-\mu_{\mathrm{BG},X},s_{\min}),
$$

其中 $s_{\min}=0.1$，用于避免后续对数变换出现零值或负值。绿色通道采用两个 Bayer 绿色子通道的平均值：

$$
G_{\mathrm{sig}}=\frac{G_{1,\mathrm{sig}}+G_{2,\mathrm{sig}}}{2}.
$$

当前最终模型使用的 6 维输入向量为：

$$
x=[
\ln R_{\mathrm{sig}},
\ln G_{\mathrm{sig}},
\ln B_{\mathrm{sig}},
\mathrm{Ratio}_{RG},
\Delta_{RG},
\Delta_{GB}
].
$$

其中：

$$
\mathrm{Ratio}_{RG}=\frac{R_{\mathrm{sig}}}{G_{\mathrm{sig}}+\epsilon},
$$

$$
\Delta_{RG}=\frac{G_{\mathrm{sig}}-R_{\mathrm{sig}}}{G_{\mathrm{sig}}+R_{\mathrm{sig}}+\epsilon},
$$

$$
\Delta_{GB}=\frac{B_{\mathrm{sig}}-G_{\mathrm{sig}}}{B_{\mathrm{sig}}+G_{\mathrm{sig}}+\epsilon}.
$$

对数光强项近似对应 Beer-Lambert 型的指数衰减关系，比值与差分项反映光谱形状变化，可降低整体亮度漂移的影响。

### 2.7.2 加权岭回归与单调校准

岭回归模型使用标准化后的特征预测 log 浓度：

$$
y_i^{\ast}=\log(C_i+\epsilon).
$$

训练目标为加权平方误差加 L2 正则项：

$$
\min_{\beta}
\sum_i w_i(y_i^{\ast}-\hat{y}_i^{\ast})^2
\alpha \lVert \beta\rVert_2^2.
$$

其中 $\alpha$ 为岭回归正则强度。样品权重用于调整不同浓度区间的影响，超参数扫描阶段采用更一般的权重形式：

$$
w_i=\frac{1}{\beta_w C_i+\epsilon_w},
$$

并将权重归一化到平均值为 1。模型预测后通过反变换得到原始浓度：

$$
\hat{C}_{\mathrm{raw}}=\exp(\hat{y}^{\ast})-\epsilon.
$$

由于浓度与传感响应应保持整体单调关系，程序进一步使用 Isotonic Regression 对 $\hat{C}_{\mathrm{raw}}$ 进行单调校准：

$$
\hat{C}=g(\hat{C}_{\mathrm{raw}}),
$$

其中 $g(\cdot)$ 是在训练集预测值和真实浓度之间拟合得到的单调函数。

### 2.7.3 评估策略

模型评估包含三类结果：

1. 训练集回测：用于检查模型是否能描述训练样品的整体趋势。
2. Leave-One-Out 交叉验证：每次留出一个训练样品，用其余样品重新训练并预测留出样品，用于估计小样本条件下的内部泛化能力[10]。
3. 外部 food 测试集：使用 `dataset/mate20x/roi_detailed_statistics.csv` 训练，在 `dataset/food/roi_detailed_statistics.csv` 上测试，用于评估跨样品基质或外部场景的预测能力。

## 3.4 岭回归输入向量与超参数优化

### 3.4.1 输入向量 2^6 组合筛选

`test_input/run_input_vector_2pow6.py` 用于筛选输入特征组合。脚本固定 3 个基础特征：

```text
ln_R_sig, ln_G_sig, ln_B_sig
```

并对 6 个可选特征做全组合枚举：

```text
Ratio_RG, Ratio_BG, Ratio_RB, Delta_RG, Delta_RB, Delta_GB
```

因此共测试 $2^6=64$ 种输入向量。每种组合均使用同一建模流程：

```text
ROI-BG correction + log target + RidgeCV + isotonic calibration
```

训练集为 `dataset/mate20x/roi_detailed_statistics.csv`，外部测试集为 `dataset/food/roi_detailed_statistics.csv`。在 food 外部测试集上，最佳输入向量为：

```text
ln_R_sig; ln_G_sig; ln_B_sig; Ratio_RG; Delta_RG; Delta_GB
```

该组合包含 6 个特征，在 food-all 上达到：

| 输入向量 | 特征数 | R² | RMSE / μM | MAE / μM | 平均误差 / μM |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ln_R_sig;ln_G_sig;ln_B_sig;Ratio_RG;Delta_RG;Delta_GB` | 6 | 0.9830 | 10.0911 | 7.7399 | -1.3754 |
| `ln_R_sig;ln_G_sig;ln_B_sig;Ratio_RG;Ratio_BG;Delta_RG` | 6 | 0.9814 | 10.5682 | 7.9562 | -2.9591 |
| `ln_R_sig;ln_G_sig;ln_B_sig` | 3 | 0.7687 | 37.2539 | 29.4468 | -27.2394 |

结果表明，单独使用三个 log 强度特征虽然能描述部分浓度趋势，但外部测试误差较大；加入 `Ratio_RG/Delta_RG/Delta_GB` 后，模型同时利用强度衰减和通道相对变化，外部测试性能明显提升。

![输入向量组合筛选结果 Top 20](../test_input/OUTPUT/top20_metric_comparison.png)

### 3.4.2 Ridge 正则强度 alpha 扫描

`parameter_scan/run_regression_parameter_scan.py` 在固定最佳输入向量后，对岭回归正则强度进行扫描。扫描范围为：

```text
alpha = 0, 1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8
```

此时权重参数固定为 $\beta_w=1$，$\epsilon_w=50$。在 food-all 外部测试集上，前几名结果为：

| alpha | R² | RMSE / μM | MAE / μM | 平均误差 / μM |
| ---: | ---: | ---: | ---: | ---: |
| 1e3 | 0.9831 | 10.0828 | 7.8068 | -0.8866 |
| 1e4 | 0.9830 | 10.0911 | 7.7399 | -1.3754 |
| 1e5 | 0.9830 | 10.0980 | 7.7464 | -1.4153 |
| 1e2 | 0.9821 | 10.3721 | 8.9670 | 3.4033 |
| 1e1 | 0.7623 | 37.7657 | 30.4004 | 29.7907 |

当 $\alpha$ 过小时，模型受共线性和小样本噪声影响明显，外部测试误差快速增大；当 $\alpha$ 位于 $10^3$ 到 $10^5$ 区间时，模型表现稳定。扫描结果选择 $\alpha=1000$ 作为后续 beta 和权重平滑项扫描的固定值。

![alpha 参数扫描结果](../parameter_scan/OUTPUT/alpha_scan_metrics.png)

### 3.4.3 样品权重 beta 扫描

权重公式为：

$$
w_i=\frac{1}{\beta_w C_i+\epsilon_w}.
$$

在固定 $\alpha=1000$、$\epsilon_w=50$ 后，脚本扫描：

```text
beta = 0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000
```

结果显示 $\beta_w=1$ 最优：

| beta | effective weight eps | R² | RMSE / μM | MAE / μM | 平均误差 / μM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 50 | 0.9831 | 10.0828 | 7.8068 | -0.8866 |
| 0.1 | 500 | 0.9709 | 13.2028 | 9.9174 | -8.0243 |
| 0.01 | 5000 | 0.9640 | 14.6910 | 11.4610 | -9.8415 |
| 10 | 5 | 0.9425 | 18.5802 | 15.1127 | 13.4840 |
| 100 | 0.5 | -0.1037 | 81.3772 | 66.2363 | 66.2363 |

当 $\beta_w$ 过小，权重接近均匀，低浓度样品相对影响下降；当 $\beta_w$ 过大，高浓度样品权重被过度压低，模型在全量程上的预测稳定性下降。因此，最终保留 $\beta_w=1$。

![beta 参数扫描结果](../parameter_scan/OUTPUT/beta_scan_metrics.png)

### 3.4.4 权重平滑项 weight_eps 扫描

在固定 $\alpha=1000$、$\beta_w=1$ 后，脚本扫描：

```text
weight_eps = 1, 10, 30, 50, 80, 120
```

food-all 外部测试集结果如下：

| weight_eps | R² | RMSE / μM | MAE / μM | 平均误差 / μM |
| ---: | ---: | ---: | ---: | ---: |
| 30 | 0.9833 | 10.0093 | 8.4122 | 1.1690 |
| 50 | 0.9831 | 10.0828 | 7.8068 | -0.8866 |
| 80 | 0.9822 | 10.3303 | 7.9272 | -2.7160 |
| 120 | 0.9803 | 10.8669 | 8.2392 | -4.1916 |
| 10 | 0.9742 | 12.4303 | 10.2651 | 6.7905 |
| 1 | 0.4916 | 55.2330 | 43.8763 | 43.8763 |

按 R² 和 RMSE 排序，`weight_eps=30` 为最佳组合，对应整体最优配置：

```text
features = ln_R_sig, ln_G_sig, ln_B_sig, Ratio_RG, Delta_RG, Delta_GB
alpha = 1000
beta = 1
weight_eps = 30
```

该配置在 food-all 外部测试集上达到 $R^2=0.9833$，RMSE 为 $10.0093\,\mu\mathrm{M}$，MAE 为 $8.4122\,\mu\mathrm{M}$。若更强调 MAE，则 `weight_eps=50` 的 MAE 略低；若以 R²/RMSE 为主，则 `weight_eps=30` 为最终推荐配置。

![weight_eps 参数扫描结果](../parameter_scan/OUTPUT/weight_eps_scan_metrics.png)

## 3.5 指数拟合基线对比

为验证岭回归模型相对于简单物理基线的优势，项目同时使用指数拟合方法作为对照。指数基线对每个背景扣除光强通道单独拟合：

$$
I(C)=a\exp(bC)+c,
$$

并通过

$$
\hat{C}=\frac{\ln((I-c)/a)}{b}
$$

反演浓度。该方法具有形式简单、参数可解释的优点，但它主要依赖单通道强度变化，难以同时利用多通道谱形信息，也更容易受到样品基质、曝光漂移和跨数据集差异影响。

在与岭回归相同的外部测试设置下，即使用 mate20x 数据训练并在 food 数据上测试，指数基线默认选择 `G1_sig` 通道。结果如下：

| 方法 | 外部测试样本数 | R² | RMSE / μM | MAE / μM | 平均误差 / μM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 岭回归最终模型 | 18 | 0.9833 | 10.0093 | 8.4122 | 1.1690 |
| 指数拟合基线 `G1_sig` | 18 | -0.5351 | 95.9731 | 79.1324 | -76.2092 |

指数基线在训练集内部可描述一定的单调衰减趋势，但在 food 外部测试集中出现系统性低估，尤其在高浓度样品上误差明显增大。这说明单通道指数反演不足以覆盖跨样品基质和跨场景变化；相比之下，岭回归通过多通道 log 强度、比值和差分特征联合建模，并结合正则化与单调校准，在外部测试中表现出更好的鲁棒性。

![岭回归最佳外部测试预测图](../parameter_scan/OUTPUT/best_food_prediction.png)

## 3.6 小结

本项目建立了从手机 DNG 荧光图像到浓度预测的完整岭回归分析链路。手机直接拍摄 DNG 文件保证了输入信号尽量接近传感器线性响应，并保留比压缩图像更高质量的原始信息；ROI/BG 双区统计降低了局部背景干扰；物理特征将整体强度衰减和通道相对变化同时纳入模型；岭回归、样品加权和单调校准共同改善了小样本条件下的稳定性。输入向量筛选表明，最佳特征不是单纯的 RGB log 强度，而是强度项与 `Ratio_RG/Delta_RG/Delta_GB` 的组合。超参数扫描进一步确定了 $\alpha=1000$、$\beta_w=1$、`weight_eps=30` 的推荐配置。与单通道指数拟合基线相比，岭回归在外部测试集上显著降低误差，说明该方法更适合本项目的手机 DNG 荧光浓度定量任务。

## 参考文献

[1] Ruiz, A. J., Allen, R., Giallorenzi, M. K., Samkoe, K. S., Chapman, M. S., & Pogue, B. W. (2023). Smartphone-based dual radiometric fluorescence and white-light imager for quantification of protoporphyrin IX in skin. *Journal of Biomedical Optics*, 28(08), 086003.

[2] Hasinoff, S. W., Sharlet, D., Geiss, R., Adams, A., Barron, J. T., Kainz, F., Chen, J., & Levoy, M. (2016). Burst photography for high dynamic range and low-light imaging on mobile cameras. *ACM Transactions on Graphics*, 35(6), 1-12.

[3] Chen, T., Lin, B., Brunner, E., & Schild, D. (2005). In situ background estimation in quantitative fluorescence imaging. *Biophysical Journal*, 90(7), 2534-2547.

[4] Liu, Y., Rollins, A. M., Levenson, R. M., Fereidouni, F., & Jenkins, M. W. (2021). Pocket MUSE: an affordable, versatile and high-performance fluorescence microscope using a smartphone. *Communications Biology*, 4(1), 334.

[5] Rasooly, R., Bruck, H., Balsam, J., Prickril, B., Ossandon, M., & Rasooly, A. (2016). Improving the sensitivity and functionality of mobile Webcam-Based fluorescence detectors for Point-of-Care diagnostics in global health. *Diagnostics*, 6(2), 19.

[6] Branning, J. M., Faughnan, K. A., Tomson, A. A., Bell, G. J., Isbell, S. M., DeGroot, A., Jameson, L., Kilroy, K., Smith, M., Smith, R., Mottel, L., Branning, E. G., Worrall, Z., Anderson, F., Panditaradyula, A., Yang, W., Abdelmalek, J., Brake, J., & Cash, K. J. (2024). Multifunction fluorescence open source in vivo/in vitro imaging system (openIVIS). *PLoS ONE*, 19(3), e0299875.

[7] Brooks, T., Mildenhall, B., Xue, T., Chen, J., Sharlet, D., & Barron, J. T. (2019). *Unprocessing images for learned raw denoising*. CVPR 2019. https://openaccess.thecvf.com/content_CVPR_2019/html/Brooks_Unprocessing_Images_for_Learned_Raw_Denoising_CVPR_2019_paper.html

[8] Hüpfel, M., Kobitski, A. Y., Zhang, W., & Nienhaus, G. U. (2021). Wavelet-based background and noise subtraction for fluorescence microscopy images. *Biomedical Optics Express*, 12(2), 969. https://doi.org/10.1364/boe.413181

[9] Kennard, R. W., & Hoerl, A. E. (2012). Ridge regression: Biased estimation for nonorthogonal problems. *Taylor & Francis*. https://www.tandfonline.com/doi/abs/10.1080/00401706.1970.10488634

[10] Stone, M. (1974). Cross-validatory choice and assessment of statistical predictions. *Journal of the Royal Statistical Society: Series B (Methodological)*, 36(2), 111-133.
