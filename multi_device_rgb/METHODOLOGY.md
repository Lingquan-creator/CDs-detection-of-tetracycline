# 跨设备RGB颜色空间映射方法说明

## 1. 问题背景与动机

### 1.1 问题定义

在量子点浓度检测系统中，不同手机设备拍摄相同浓度样品时，由于以下因素会产生不同的RGB响应：

- **传感器差异**：不同的CMOS传感器、Bayer滤色片光谱响应曲线不同
- **光学系统**：镜头的色散、透射率、畸变特性各异
- **ISP处理**：图像信号处理器的去马赛克、白平衡、色彩校正算法不同
- **制造差异**：即使同型号手机，个体间也存在制造误差

这导致在设备A（pic_1）上训练的浓度预测模型无法直接应用于设备B（pic_2）的数据。

### 1.2 目标

建立从设备B（pic_2）到设备A（pic_1）的RGB颜色空间映射函数 $f: \mathbb{R}^4 \to \mathbb{R}^4$，使得：

$$
\mathbf{y}_{\text{pic1}} = f(\mathbf{x}_{\text{pic2}})
$$

其中 $\mathbf{x}_{\text{pic2}} = [R_2, G1_2, G2_2, B_2]^T$ 和 $\mathbf{y}_{\text{pic1}} = [R_1, G1_1, G2_1, B_1]^T$ 分别是两个设备的4通道背景扣除特征。

**最终目标**：映射后的特征能够准确预测浓度，即：

$$
c_{\text{pred}} = g(f(\mathbf{x}_{\text{pic2}})) \approx c_{\text{true}}
$$

其中 $g$ 是在pic_1上训练的浓度预测模型。

### 1.3 数据情况

- **pic_1设备**：7个DNG文件，浓度为 40, 80, 120, 140, 160, 200, 240 µM
- **pic_2设备**：7个DNG文件，浓度为 0, 40, 80, 120, 160, 200, 240 µM
- **匹配样本对**：6对（浓度：40, 80, 120, 160, 200, 240 µM）
- **测试样本**：pic_2的浓度0样本（用于外推测试）

### 1.4 核心挑战

1. **小样本问题**：仅有6对匹配样本，容易过拟合
2. **非线性映射**：传感器响应通常是非线性的（gamma曲线、色彩矩阵）
3. **外推能力**：需要对训练范围外的浓度（如0 µM）有合理预测
4. **物理约束**：映射应保持单调性、非负性等物理特性
5. **多通道耦合**：4个通道之间可能存在相关性

---

## 2. 候选算法综述

本节详细介绍所有考虑的算法，包括**已实现**和**未实现**的方法。

### 2.1 线性方法

#### 2.1.1 颜色校正矩阵（Color Correction Matrix, CCM）

**状态**：❌ 未实现（作为对比基准）

**数学原理**：

使用3×3或3×4矩阵进行线性变换：

$$
\begin{bmatrix} R_1 \\ G_1 \\ B_1 \end{bmatrix} =
\mathbf{M} \begin{bmatrix} R_2 \\ G_2 \\ B_2 \\ 1 \end{bmatrix}
$$

其中 $\mathbf{M} \in \mathbb{R}^{3 \times 4}$ 是颜色校正矩阵。

**求解方法**：

最小二乘法：

$$
\mathbf{M}^* = \arg\min_{\mathbf{M}} \|\mathbf{Y}_1 - \mathbf{M}\mathbf{X}_2\|_F^2
$$

其中 $\|\cdot\|_F$ 是Frobenius范数。

**优点**：
- 计算极快（矩阵乘法）
- 参数少（12个参数）
- 广泛应用于相机标定
- 物理意义明确

**缺点**：
- 仅能捕捉线性关系
- 无法处理gamma非线性
- 对传感器饱和、暗电流等非线性效应无能为力
- 在我们的场景中，传感器响应明显非线性

**为何未实现**：
传感器的gamma曲线、色彩空间转换等都是非线性的，线性CCM无法充分建模这些效应。初步分析显示pic_1和pic_2的RGB关系存在明显的二次非线性。

---

#### 2.1.2 岭回归（Ridge Regression）

**状态**：✅ 已实现（作为线性基准）

**数学原理**：

对每个输出通道独立建模：

$$
y_c = \mathbf{w}_c^T \mathbf{x} + b_c, \quad c \in \{R, G1, G2, B\}
$$

带L2正则化的损失函数：

$$
\mathcal{L}(\mathbf{w}_c) = \sum_{i=1}^{N} (y_{c,i} - \mathbf{w}_c^T \mathbf{x}_i - b_c)^2 + \alpha \|\mathbf{w}_c\|^2
$$

**求解方法**：

闭式解：

$$
\mathbf{w}_c^* = (\mathbf{X}^T\mathbf{X} + \alpha\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}_c
$$

**优点**：
- 正则化防止过拟合（关键！N=6）
- 闭式解，训练快速
- 超参数α通过交叉验证自动选择
- 稳定性好

**缺点**：
- 线性假设过强
- 无法捕捉传感器的非线性响应

**实现细节**：
使用`sklearn.linear_model.RidgeCV`，α网格为[0.01, 0.1, 1, 10, 100, 1000]，交叉验证策略为LeaveOneOut。

---

### 2.2 非线性方法（多项式扩展）

#### 2.2.1 多项式岭回归（Polynomial Ridge Regression）⭐

**状态**：✅ 已实现（主推荐方案）

**数学原理**：

先对输入特征进行多项式扩展，再应用岭回归。

**特征扩展**（2次多项式）：

$$
\phi(\mathbf{x}) = [1, R, G1, G2, B, R^2, G1^2, G2^2, B^2, R \cdot G1, R \cdot G2, R \cdot B, G1 \cdot G2, G1 \cdot B, G2 \cdot B]^T
$$

维度：4 → 15（包含常数项）

**模型**：

$$
y_c = \mathbf{w}_c^T \phi(\mathbf{x})
$$

**损失函数**：

$$
\mathcal{L}(\mathbf{w}_c) = \sum_{i=1}^{N} (y_{c,i} - \mathbf{w}_c^T \phi(\mathbf{x}_i))^2 + \alpha \|\mathbf{w}_c\|^2
$$

**物理解释**：
- **线性项**：基础颜色校正矩阵
- **二次项**：gamma曲线校正（$R^2, G^2, B^2$）
- **交叉项**：通道间耦合（色彩串扰、去马赛克效应）

**优点**：
- 捕捉非线性传感器响应
- 岭正则化防止过拟合（N=6时至关重要）
- 物理可解释性强
- 与现有代码库一致（量子点_2.md中使用岭回归）
- 训练和推理速度快

**缺点**：
- 多项式阶数需要手动选择
- 高阶多项式可能导致数值不稳定
- 外推能力有限（多项式在边界外可能震荡）

**超参数选择**：
- **多项式阶数**：2（平衡复杂度和过拟合风险）
- **Alpha网格**：[0.01, 0.1, 1, 10, 100, 1000]
- **交叉验证**：LeaveOneOut（LOO-CV，最适合N=6）
- **特征标准化**：StandardScaler（防止数值问题）

**为何选择此方案**：
1. 小样本鲁棒性最佳
2. 物理可解释性强
3. 计算效率高
4. 与现有代码库一致

---

#### 2.2.2 多项式Lasso回归（Polynomial Lasso Regression）

**状态**：❌ 未实现

**数学原理**：

与多项式岭回归类似，但使用L1正则化：

$$
\mathcal{L}(\mathbf{w}_c) = \sum_{i=1}^{N} (y_{c,i} - \mathbf{w}_c^T \phi(\mathbf{x}_i))^2 + \alpha \|\mathbf{w}_c\|_1
$$

**优点**：
- L1正则化产生稀疏解（特征选择）
- 可以自动识别重要的多项式项
- 模型更简洁

**缺点**：
- 无闭式解，需要迭代优化
- 在高度相关的特征中表现不稳定
- 对α的选择更敏感

**为何未实现**：
1. 多项式特征之间高度相关（如$R$和$R^2$），Lasso在这种情况下不稳定
2. 岭回归的L2正则化更适合处理多重共线性
3. 我们更关心预测精度而非特征选择

---

#### 2.2.3 弹性网络（Elastic Net）

**状态**：❌ 未实现

**数学原理**：

结合L1和L2正则化：

$$
\mathcal{L}(\mathbf{w}_c) = \sum_{i=1}^{N} (y_{c,i} - \mathbf{w}_c^T \phi(\mathbf{x}_i))^2 + \alpha_1 \|\mathbf{w}_c\|_1 + \alpha_2 \|\mathbf{w}_c\|^2
$$

**优点**：
- 结合Lasso和Ridge的优点
- 在相关特征中比Lasso稳定
- 仍能进行特征选择

**缺点**：
- 两个超参数需要调优（计算成本高）
- 对于N=6的小样本，调优困难

**为何未实现**：
超参数空间增大（α1, α2），在仅有6个样本时，交叉验证的可靠性下降。岭回归已经足够鲁棒。

---

### 2.3 核方法

#### 2.3.1 支持向量回归（Support Vector Regression, SVR）

**状态**：❌ 未实现

**数学原理**：

使用核函数将输入映射到高维空间，然后在该空间进行线性回归。

**RBF核**：

$$
k(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right)
$$

**优化目标**（ε-SVR）：

$$
\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{N}\max(0, |y_i - f(\mathbf{x}_i)| - \epsilon)
$$

**优点**：
- 核技巧可以捕捉复杂非线性
- 对异常值鲁棒（ε-不敏感损失）
- 理论基础扎实（统计学习理论）

**缺点**：
- 超参数多（C, γ, ε）
- 训练时间长（二次规划）
- 对小样本（N=6）可能过拟合
- 预测时需要存储支持向量

**为何未实现**：
1. 超参数调优在N=6时不可靠
2. 计算成本高
3. 多项式岭回归已经能捕捉主要非线性

---

#### 2.3.2 核岭回归（Kernel Ridge Regression）

**状态**：❌ 未实现

**数学原理**：

岭回归的核化版本：

$$
f(\mathbf{x}) = \sum_{i=1}^{N} \alpha_i k(\mathbf{x}, \mathbf{x}_i)
$$

其中 $\boldsymbol{\alpha} = (\mathbf{K} + \alpha\mathbf{I})^{-1}\mathbf{y}$，$\mathbf{K}_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$。

**优点**：
- 闭式解（比SVR快）
- 可以使用任意核函数
- 正则化防止过拟合

**缺点**：
- 预测时需要计算与所有训练样本的核函数（O(N)）
- 核参数（如RBF的γ）需要调优
- 对于N=6，多项式核等价于显式多项式特征

**为何未实现**：
多项式核的核岭回归等价于显式多项式特征+岭回归，后者更直观且易于解释。

---

### 2.4 概率方法

#### 2.4.1 高斯过程回归（Gaussian Process Regression, GPR）⭐

**状态**：✅ 已实现（备选方案）

**数学原理**：

假设函数 $f$ 服从高斯过程：

$$
f(\mathbf{x}) \sim \mathcal{GP}(\mu(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))
$$

**预测分布**：

给定训练数据 $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$，对新输入 $\mathbf{x}_*$ 的预测：

$$
p(f_* | \mathbf{x}_*, \mathcal{D}) = \mathcal{N}(\mu_*, \sigma_*^2)
$$

其中：

$$
\mu_* = \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2\mathbf{I})^{-1} \mathbf{y}
$$

$$
\sigma_*^2 = k(\mathbf{x}_*, \mathbf{x}_*) - \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2\mathbf{I})^{-1} \mathbf{k}_*
$$

**核函数**（RBF + 白噪声）：

$$
k(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp\left(-\frac{\|\mathbf{x} - \mathbf{x}'\|^2}{2l^2}\right) + \sigma_n^2 \delta(\mathbf{x}, \mathbf{x}')
$$

**超参数**：
- $l$：长度尺度（控制平滑度）
- $\sigma_f^2$：信号方差
- $\sigma_n^2$：噪声方差

**优点**：
- **不确定性量化**：提供预测置信区间（关键优势！）
- 小样本表现优秀（N=6是合理的）
- 非参数方法，无需假设函数形式
- 自然强制平滑性
- 超参数通过最大似然自动优化

**缺点**：
- 计算复杂度O(N³)（对N=6可接受）
- 核函数选择需要领域知识
- 外推时不确定性快速增大
- 对每个通道需要独立训练

**实现细节**：
- 使用`sklearn.gaussian_process.GaussianProcessRegressor`
- 核函数：`ConstantKernel * RBF + WhiteKernel`
- 超参数优化：最大似然估计（MLE）+ 多次随机重启
- 对4个通道分别建模

**为何作为备选**：
1. 提供不确定性估计（可用于质量控制）
2. 小样本性能优秀
3. 但计算成本略高于多项式岭回归
4. 可解释性不如多项式模型

---

#### 2.4.2 贝叶斯岭回归（Bayesian Ridge Regression）

**状态**：❌ 未实现

**数学原理**：

对权重 $\mathbf{w}$ 和噪声精度 $\lambda$ 施加先验分布：

$$
p(\mathbf{w} | \alpha) = \mathcal{N}(\mathbf{0}, \alpha^{-1}\mathbf{I})
$$

$$
p(\lambda) = \text{Gamma}(\lambda | a_0, b_0)
$$

通过变分推断或EM算法估计后验分布。

**优点**：
- 自动确定正则化强度（α）
- 提供预测不确定性
- 防止过拟合

**缺点**：
- 计算成本高于普通岭回归
- 对于N=6，交叉验证已经足够选择α
- 不确定性估计不如GP准确

**为何未实现**：
RidgeCV通过交叉验证选择α已经很有效，贝叶斯方法的额外计算成本不值得。

---

### 2.5 非参数方法

#### 2.5.1 分段线性插值（Piecewise Linear Interpolation）⭐

**状态**：✅ 已实现（保底方案）

**数学原理**：

对每个通道独立，使用训练数据点作为控制点进行线性插值：

$$
y_c = \text{interp1d}(x_{2,c}, y_{1,c}, \text{kind}='linear')
$$

**可选约束**：
- **单调性**：使用保序回归（Isotonic Regression）确保 $y$ 随 $x$ 单调
- **边界外推**：线性外推或常数外推

**优点**：
- 极其简单，无需训练
- 保证通过所有训练点（训练误差为0）
- 可强制单调性（符合物理直觉）
- 计算极快

**缺点**：
- 无平滑性（分段点处不可导）
- 外推能力差（线性外推可能不合理）
- 无正则化（对噪声敏感）
- 不利用通道间相关性

**实现细节**：
- 使用`scipy.interpolate.interp1d`
- 外推策略：`fill_value='extrapolate'`
- 可选：使用`sklearn.isotonic.IsotonicRegression`强制单调性

**为何作为保底**：
1. 当其他方法失败时的最简单备选
2. 可用于快速原型验证
3. 作为性能下界基准

---

#### 2.5.2 K近邻回归（K-Nearest Neighbors Regression）

**状态**：❌ 未实现

**数学原理**：

预测为K个最近邻的平均值：

$$
\hat{y} = \frac{1}{K}\sum_{i \in \mathcal{N}_K(\mathbf{x})} y_i
$$

其中 $\mathcal{N}_K(\mathbf{x})$ 是 $\mathbf{x}$ 的K个最近邻。

**优点**：
- 非参数，无需训练
- 可以捕捉局部非线性
- 简单直观

**缺点**：
- 对N=6，K的选择困难（K=1或2？）
- 对维度诅咒敏感（4维输入）
- 无外推能力（只能在训练数据凸包内预测）
- 对噪声敏感

**为何未实现**：
样本量太小（N=6），KNN无法发挥优势。外推能力差是致命缺陷。

---

### 2.6 集成方法

#### 2.6.1 随机森林回归（Random Forest Regression）

**状态**：❌ 未实现

**数学原理**：

集成多个决策树的预测：

$$
\hat{y} = \frac{1}{T}\sum_{t=1}^{T} f_t(\mathbf{x})
$$

其中 $f_t$ 是第t棵决策树。

**优点**：
- 可以捕捉非线性和交互作用
- 对异常值鲁棒
- 特征重要性分析

**缺点**：
- 对N=6，每棵树只能看到更少的样本（bootstrap）
- 外推能力差（只能预测训练范围内的值）
- 超参数多（树的数量、深度、最小样本数）

**为何未实现**：
样本量太小，无法有效bootstrap。外推能力差是致命缺陷（需要预测浓度0）。

---

#### 2.6.2 梯度提升回归（Gradient Boosting Regression）

**状态**：❌ 未实现

**数学原理**：

顺序训练多个弱学习器，每个学习器拟合前一个的残差：

$$
f_m(\mathbf{x}) = f_{m-1}(\mathbf{x}) + \nu \cdot h_m(\mathbf{x})
$$

其中 $h_m$ 拟合残差 $r_m = y - f_{m-1}(\mathbf{x})$。

**优点**：
- 通常比随机森林更准确
- 可以处理非线性

**缺点**：
- 对N=6，容易过拟合
- 超参数多且敏感
- 外推能力差
- 训练时间长

**为何未实现**：
与随机森林类似的问题，且更容易过拟合。

---

### 2.7 深度学习方法

#### 2.7.1 多层感知机（Multi-Layer Perceptron, MLP）

**状态**：❌ 未实现

**数学原理**：

多层非线性变换：

$$
\mathbf{h}_1 = \sigma(\mathbf{W}_1\mathbf{x} + \mathbf{b}_1)
$$

$$
\mathbf{h}_2 = \sigma(\mathbf{W}_2\mathbf{h}_1 + \mathbf{b}_2)
$$

$$
\mathbf{y} = \mathbf{W}_3\mathbf{h}_2 + \mathbf{b}_3
$$

其中 $\sigma$ 是激活函数（如ReLU）。

**优点**：
- 理论上可以逼近任意函数（万能逼近定理）
- 可以学习复杂的非线性映射
- 端到端学习

**缺点**：
- **严重过拟合**：N=6远远不足以训练神经网络
- 超参数多（层数、神经元数、学习率、正则化）
- 需要大量数据
- 训练不稳定（局部最优）
- 可解释性差

**为何未实现**：
样本量太小（N=6），神经网络必然过拟合。即使使用强正则化（Dropout、L2），也无法保证泛化性能。

---

## 3. 特征工程策略

### 3.1 输入特征选择

#### 选项A：4通道原始特征（推荐）⭐

```python
X = [R_corrected, G1_corrected, G2_corrected, B_corrected]  # Shape: (N, 4)
```

**理由**：
- 保留Bayer传感器的最大信息
- G1和G2可能有不同的响应特性
- 允许模型学习通道间的相关性

#### 选项B：3通道RGB特征

```python
X = [R_corrected, G_avg_corrected, B_corrected]  # Shape: (N, 3)
其中 G_avg = (G1 + G2) / 2
```

**理由**：
- 更简单（3个输出vs 4个）
- 标准RGB表示
- 如果G1和G2差异很小，可以简化模型

### 3.2 多项式特征扩展（用于岭回归）

**2次多项式**（推荐）：

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=True)
X_poly = poly.fit_transform(X)  # Shape: (N, 15)
```

**特征列表**：
```
[1, R, G1, G2, B, R², G1², G2², B², R·G1, R·G2, R·B, G1·G2, G1·B, G2·B]
```

**物理意义**：
- **常数项**：偏移量
- **线性项**：基础颜色校正矩阵
- **二次项**：gamma曲线校正
- **交叉项**：通道间耦合（色彩串扰、去马赛克效应）

**3次多项式**（可选，谨慎使用）：

```python
poly = PolynomialFeatures(degree=3, include_bias=True)
X_poly = poly.fit_transform(X)  # Shape: (N, 35)
```

**风险**：
- 特征数（35）远大于样本数（6），严重过拟合风险
- 需要更强的正则化（更大的α）
- 数值不稳定

### 3.3 特征标准化

**必须使用**：

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)
```

**理由**：
- 多项式特征的尺度差异巨大（$R$ vs $R^2$）
- 标准化确保所有特征对损失函数的贡献相当
- 提高数值稳定性
- 使正则化更有效（L2惩罚对所有特征一视同仁）

### 3.4 其他特征（未使用）

#### 比值特征

```python
R/G, B/G, R/B, G1/G2
```

**优点**：光照不变性
**缺点**：除零风险，非线性难以建模

#### 对数特征

```python
log(R+ε), log(G+ε), log(B+ε)
```

**优点**：符合Beer-Lambert定律
**缺点**：对小值敏感，需要选择ε

**为何未使用**：
多项式特征已经能捕捉主要非线性，额外特征增加过拟合风险。

---

## 4. 验证策略

### 4.1 留一法交叉验证（LOO-CV）⭐

**方法**：

```python
from sklearn.model_selection import LeaveOneOut

cv = LeaveOneOut()
y_pred_cv = np.zeros_like(y_true)

for train_idx, test_idx in cv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    mapper.fit(X_train, y_train)
    y_pred_cv[test_idx] = mapper.predict(X_test)
```

**特点**：
- 6个样本 → 6折交叉验证
- 每折训练5个样本，测试1个样本
- 无偏估计泛化误差
- 最适合小样本场景

**评估指标**：

```python
from sklearn.metrics import mean_squared_error, r2_score

rmse_cv = np.sqrt(mean_squared_error(y_true, y_pred_cv))
r2_cv = r2_score(y_true, y_pred_cv)
```

### 4.2 外推测试（浓度0）

**目的**：测试模型对训练范围外数据的泛化能力

**方法**：

```python
# pic_2中有浓度=0的样本（不在训练集中）
X_zero = load_concentration_zero_sample()
y_mapped = mapper.predict(X_zero)

# 检查映射后的特征是否合理
print(f"Mapped features for conc=0: {y_mapped}")
```

**评估**：
- 映射后的特征应该接近pic_1的低浓度样本
- 不应出现负值或异常大的值
- 可以使用pic_1的浓度模型预测，看是否接近0

### 4.3 端到端浓度预测测试（最关键）⭐

**目的**：验证映射后的特征能否准确预测浓度

**方法**：

```python
# 1. 加载pic_1训练的浓度模型
conc_model = load_concentration_model()

# 2. 映射pic_2特征到pic_1空间
X_pic2_mapped = mapper.predict(X_pic2)

# 3. 预测浓度
conc_pred = conc_model.predict(X_pic2_mapped)

# 4. 评估
rmse_conc = np.sqrt(mean_squared_error(conc_true, conc_pred))
r2_conc = r2_score(conc_true, conc_pred)
```

**成功标准**：
- R² > 0.95（浓度预测）
- RMSE < 10 µM（相对误差 < 5%）

**这是最终验证标准**：通道级指标是手段，浓度预测是目的。

### 4.4 对比基准

**基准1**：直接使用pic_2数据（未映射）
- 预期：浓度预测性能差

**基准2**：线性岭回归（无多项式）
- 预期：性能不如多项式岭回归

**基准3**：pic_1原始数据的浓度模型性能
- 预期：映射后应接近此性能

---

## 5. 评估指标

### 5.1 通道级指标

对每个通道 $c \in \{R, G1, G2, B\}$：

**均方根误差（RMSE）**：

$$
\text{RMSE}_c = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_{c,i} - \hat{y}_{c,i})^2}
$$

**平均绝对误差（MAE）**：

$$
\text{MAE}_c = \frac{1}{N}\sum_{i=1}^{N}|y_{c,i} - \hat{y}_{c,i}|
$$

**决定系数（R²）**：

$$
R^2_c = 1 - \frac{\sum_{i=1}^{N}(y_{c,i} - \hat{y}_{c,i})^2}{\sum_{i=1}^{N}(y_{c,i} - \bar{y}_c)^2}
$$

**平均绝对百分比误差（MAPE）**：

$$
\text{MAPE}_c = \frac{100\%}{N}\sum_{i=1}^{N}\frac{|y_{c,i} - \hat{y}_{c,i}|}{y_{c,i} + \epsilon}
$$

### 5.2 颜色差异指标（ΔE）

**CIEDE2000**：感知均匀的颜色差异

```python
from skimage.color import rgb2lab, deltaE_ciede2000

# 转换到LAB空间
lab1 = rgb2lab(rgb1_normalized)
lab2 = rgb2lab(rgb2_normalized)

# 计算ΔE
delta_e = deltaE_ciede2000(lab1, lab2)
```

**解释**：
- ΔE < 1：不可察觉
- ΔE 1-2：仔细观察可察觉
- ΔE 2-10：一眼可见
- ΔE > 10：显著差异

**目标**：ΔE < 3（一眼可见以下）

### 5.3 浓度预测指标（最关键）⭐

**RMSE（浓度）**：

$$
\text{RMSE}_{\text{conc}} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(c_i - \hat{c}_i)^2}
$$

**R²（浓度）**：

$$
R^2_{\text{conc}} = 1 - \frac{\sum_{i=1}^{N}(c_i - \hat{c}_i)^2}{\sum_{i=1}^{N}(c_i - \bar{c})^2}
$$

**相对误差**：

$$
\text{RelErr} = \frac{\text{RMSE}_{\text{conc}}}{\bar{c}} \times 100\%
$$

**目标**：
- R² > 0.95
- RMSE < 10 µM
- 相对误差 < 5%

---

## 6. 实现细节

### 6.1 数据加载

**前提**：pic_1和pic_2已运行pic_RGB流程，生成`roi_detailed_statistics.csv`

**加载代码**：

```python
import pandas as pd
import numpy as np

def load_device_data(device_dir):
    """加载设备的ROI统计数据"""
    stats_file = device_dir / "roi_results" / "roi_detailed_statistics.csv"
    df = pd.read_csv(stats_file)

    # 从文件名提取浓度
    df['concentration'] = df['filename'].str.extract(r'(\d+(?:\.\d+)?)').astype(float)

    return df

def align_datasets(pic1_dir, pic2_dir, matching_concentrations):
    """对齐两个设备的数据"""
    df1 = load_device_data(pic1_dir)
    df2 = load_device_data(pic2_dir)

    # 筛选匹配浓度
    df1_matched = df1[df1['concentration'].isin(matching_concentrations)].sort_values('concentration')
    df2_matched = df2[df2['concentration'].isin(matching_concentrations)].sort_values('concentration')

    # 提取4通道特征（背景扣除）
    feature_cols = ['roi_R_mean', 'roi_G1_mean', 'roi_G2_mean', 'roi_B_mean']
    bg_cols = ['bg_R_mean', 'bg_G1_mean', 'bg_G2_mean', 'bg_B_mean']

    X_pic2 = (df2_matched[feature_cols].values - df2_matched[bg_cols].values).clip(min=0.1)
    y_pic1 = (df1_matched[feature_cols].values - df1_matched[bg_cols].values).clip(min=0.1)

    concentrations = df1_matched['concentration'].values

    return X_pic2, y_pic1, concentrations
```

### 6.2 训练流程

```python
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut

# 构建pipeline
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=True)),
    ('scaler', StandardScaler()),
    ('ridge', RidgeCV(
        alphas=[0.01, 0.1, 1, 10, 100, 1000],
        cv=LeaveOneOut(),
        scoring='neg_mean_squared_error'
    ))
])

# 对每个通道独立训练
models = {}
for i, ch_name in enumerate(['R', 'G1', 'G2', 'B']):
    print(f"Training mapper for channel {ch_name}...")
    pipeline.fit(X_pic2, y_pic1[:, i])

    # 提取选择的alpha
    best_alpha = pipeline.named_steps['ridge'].alpha_
    print(f"  Best alpha: {best_alpha:.4f}")

    models[ch_name] = pipeline

# 保存模型
import joblib
joblib.dump(models, 'color_mapper_ridge.pkl')
```

### 6.3 推理流程

```python
# 加载模型
models = joblib.load('color_mapper_ridge.pkl')

# 映射新数据
def map_colors(X_pic2_new):
    y_pred = np.zeros((X_pic2_new.shape[0], 4))
    for i, ch_name in enumerate(['R', 'G1', 'G2', 'B']):
        y_pred[:, i] = models[ch_name].predict(X_pic2_new)
    return y_pred

# 使用
X_pic2_test = load_new_pic2_data()
X_pic1_mapped = map_colors(X_pic2_test)
```

---

## 7. 使用指南

### 7.1 快速开始

**步骤1**：确保数据准备完毕

```bash
# 对pic_1运行ROI提取
python -m pic_RGB.extraction.roi_processor --data_dir multi_device_rgb/pic_1

# 对pic_2运行ROI提取
python -m pic_RGB.extraction.roi_processor --data_dir multi_device_rgb/pic_2
```

**步骤2**：训练颜色映射器

```bash
cd multi_device_rgb/device_mapping
python train_mapper.py --mapper_type ridge --poly_degree 2
```

**步骤3**：验证映射器

```bash
python validate_mapper.py --mapper_path results/ridge_mapper.pkl
```

**步骤4**：应用映射器

```bash
python apply_mapper.py --mapper_path results/ridge_mapper.pkl \
                       --input_dir ../pic_2 \
                       --output_dir ../pic_2_mapped
```

### 7.2 配置选项

编辑`device_mapping/config.py`：

```python
@dataclass
class MappingConfig:
    # 数据路径
    pic1_dir: Path = Path("h:/000/CDs/multi_device_rgb/pic_1")
    pic2_dir: Path = Path("h:/000/CDs/multi_device_rgb/pic_2")

    # 匹配浓度
    matching_concentrations: tuple = (40, 80, 120, 160, 200, 240)

    # 特征设置
    use_4_channels: bool = True  # True: R,G1,G2,B; False: R,G,B
    polynomial_degree: int = 2

    # 模型设置
    mapper_type: str = "ridge"  # "ridge", "gp", "interpolation"
    alpha_grid: tuple = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
```

### 7.3 输出文件

训练后生成：

```
multi_device_rgb/device_mapping/results/
├── ridge_mapper.pkl              # 训练好的模型
├── validation_report.csv         # 验证指标
├── loo_cv_predictions.csv        # LOO-CV预测结果
├── channel_scatter_plots.png     # 通道级散点图
├── residual_analysis.png         # 残差分析图
└── concentration_prediction.png  # 浓度预测对比图
```

---

## 8. 总结与建议

### 8.1 推荐方案

**主方案**：多项式岭回归（Polynomial Ridge Regression）⭐
- 简单、可靠、可解释
- 适合小样本（N=6）
- 与现有代码库一致
- 快速训练和推理

**备选方案**：高斯过程回归（Gaussian Process Regression）
- 提供不确定性估计
- 小样本性能优秀
- 计算成本略高

**保底方案**：分段线性插值（Piecewise Linear Interpolation）
- 最简单
- 作为性能下界基准

### 8.2 验证重点

**最关键**：端到端浓度预测准确性⭐
- 这是最终目标
- 通道级指标是手段，浓度预测是目的

**次要**：通道级RMSE、R²、ΔE

### 8.3 成功标准

1. **通道级RMSE < 5%**（相对于通道均值）
2. **颜色差异ΔE < 3**（一眼可见以下）
3. **浓度预测R² > 0.95**（端到端测试）⭐
4. **LOO-CV稳定**（6折结果方差小）

### 8.4 风险缓解

**风险1**：样本量太小（N=6）
- **缓解**：LOO-CV + 强正则化 + 限制多项式阶数

**风险2**：外推到浓度0表现差
- **缓解**：测试外推性能 + 考虑物理约束（单调性）

**风险3**：不同浓度范围映射质量不一致
- **缓解**：残差分析 + 考虑分段映射或加权回归

---

## 9. 参考文献

1. **颜色校正**：
   - Finlayson, G. D., & Drew, M. S. (2016). "The maximum ignorance assumption with positivity." Color and Imaging Conference.

2. **岭回归**：
   - Hoerl, A. E., & Kennard, R. W. (1970). "Ridge regression: Biased estimation for nonorthogonal problems." Technometrics.

3. **高斯过程**：
   - Rasmussen, C. E., & Williams, C. K. I. (2006). "Gaussian Processes for Machine Learning." MIT Press.

4. **小样本学习**：
   - Raudys, S. J., & Jain, A. K. (1991). "Small sample size effects in statistical pattern recognition." IEEE TPAMI.

5. **颜色科学**：
   - Sharma, G., Wu, W., & Dalal, E. N. (2005). "The CIEDE2000 color-difference formula." Color Research & Application.

---

**文档版本**：v1.0
**最后更新**：2026-01-27
**作者**：Claude Sonnet 4.5


