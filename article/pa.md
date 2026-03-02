本研究使用固定摄像头进行荧光强度检测，研发了一套基于 RAW 数据的浓度定量分析软件。Tim Brooks等人明确指出，图像处理流水线中的 gain、color correction、tone mapping 等步骤会显著影响 RAW 测量值到最终成像结果的变换。[^1]因此如果做定量分析，使用原始的RAW数据，绕开这些处理更稳妥。通过图形化界面 ROI 交互选区和背景环扣除机制，构建了低设备依赖、可批量复用的数据处理链路。在固定光路与固定视角条件下，系统能够稳定快速输出跨样本可比较的统计量，为后续建模提供一致的数据基础。

软件首先对 RAW Bayer 阵列进行 RGGB 四通道拆分，提取 $R$、$G_1$、$G_2$、$B$ 子通道信号，并结合黑电平扣除与饱和阈值控制完成基础质量处理。随后基于“ROI 区域 + 背景环区域”的双区统计策略，计算各通道均值、方差、分位数与饱和比例等指标。Hüpfel 等在 2021 年指出，荧光图像中的背景贡献主要来自离焦荧光与散射光（形成缓慢变化的低频背景），而检测系统还会引入像素级噪声；他们明确把背景/噪声扣除视为后续定量分析的预处理步骤。这个来源很适合支撑你说的“把局部散射光和暗背景影响分离出来”。[^2]我们通过 $X_{corr}=\max(\mu_{ROI,X}-\mu_{BG,X},0)$ 完成背景校正。该定义对应于文献中常见的 ROI 平均强度减局部背景平均强度的校正方式，并在实现上加入零截断以避免负值；此步骤可削弱局部散射背景、非特异本底及检测偏置对信号的影响，从而使不同批次样本在更一致的基线下进入后续定量分析。

在输入特征构建时，软件围绕“双峰”发射谱机理建立物理可解释特征集合：一类为强度衰减特征（ $\ln(R_{corr}+\epsilon)$、$\ln(G_{corr}+\epsilon)$、$\ln(B_{corr}+\epsilon)$），用于刻画浓度变化引起的整体信号衰减；另一类为谱形变化特征（如 $R/G$、$B/G$、$\rho_{RG}$、$\delta_{RG}$），用于表征红绿峰相对贡献变化。同时引入 $G_1/G_2$ 一致性和饱和比例作为质量约束指标，用于识别照明不均、局部遮挡和异常成像样本。通过“强度信息 + 比值信息”联合建模，系统在亮度漂移条件下保持较好的稳健性。

在建模与推理实现上，软件采用“加权岭回归 + 单调校准 + 留一法交叉验证（LOO-CV）”的流程。训练阶段对低浓度样本施加更高权重，提升全量程尤其低浓区的拟合稳定性；在岭回归输出后[^3]使用 Isotonic Regression 进行单调校准，约束预测结果与真实浓度保持一致单调关系。评估阶段采用留一交叉验证（LOOCV），对每个样本轮流留出并使用其余样本建模，从而估计未见样本上的预测误差；在样本量有限且样本近似同分布时，该指标可作为模型泛化性能的近似表征。若样本存在批次、年份或来源差异，则更适合采用按组留出的交叉验证或独立外部测试集进行评估。[^4] 推理阶段依据模型元数据自动匹配特征模式与参数配置，支持批量 CSV 输入和浓度结果导出。

软件工程实现方面，本系统将 RAW 解析、ROI 统计、特征构建、模型训练、批量推理与结果可视化解耦为独立模块，并提供图形化操作接口与命令行接口两种使用方式。用户可在 GUI 中完成 ROI/BG 设定与浓度预测批处理，也可在脚本模式下完成模型重训与自动推理；结果以 CSV、模型文件和评估图的形式保存，便于后续复核、版本管理和论文复现。该设计保证了研究验证与工程落地之间的一致性。

实验结果表明，在 $240\,\mu\mathrm{M}$ 满量程下，本系统在 LOO-CV 条件下达到 $R^2=0.983$、RMSE $=9.40\,\mu\mathrm{M}$，相对误差约 $3.9\%$；相较单通道 $G_1$ 指数反演方法（$R^2=0.949$，RMSE $=12.61\,\mu\mathrm{M}$），精度与鲁棒性均有提升。上述结果说明，所研发软件能够在固定摄像头平台上实现稳定、可复现、可解释的荧光定量可视化检测，为低成本定量分析应用提供了可行的工程方案。

[1] Brooks, T., Mildenhall, B., Xue, T., Chen, J., Sharlet, D., & Barron, J. T. (2019). *Unprocessing images for learned raw denoising*. https://openaccess.thecvf.com/content_CVPR_2019/html/Brooks_Unprocessing_Images_for_Learned_Raw_Denoising_CVPR_2019_paper.html

[2]Hüpfel, M., Kobitski, A. Y., Zhang, W., & Nienhaus, G. U. (2021). Wavelet-based background and noise subtraction for fluorescence microscopy images. *Biomedical Optics Express*, *12*(2), 969. https://doi.org/10.1364/boe.413181

[3]Kennard, R. W., & Hoerl, A. E. (2012). Ridge regression: Biased estimation for nonorthogonal problems. *Taylor & Francis*. https://www.tandfonline.com/doi/abs/10.1080/00401706.1970.10488634?utm_source=chatgpt.com

[4]Stone, M. (1974). Cross‐validatory choice and assessment of statistical predictions. Journal of the royal statistical society: Series B (Methodological), 36(2), 111-133.

