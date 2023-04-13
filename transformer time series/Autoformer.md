# Autoformer

作为时间序列分析中的一种标准方法，时间序列分解 [1, 33] 将时间序列解构为几个部分，每个部分代表一种更可预测的基本模式类别。它主要用于探索随着时间推移的历史变化。对于预测任务，在预测未来序列 [20, 2] 之前，总是使用分解作为历史序列的预处理，例如带有 **trend-seasonality 分解的 Prophet [39] 和带有基扩展（basis expansion）**的 N-BEATS [29] 和 使用矩阵分解的 DeepGLO [35] 。然而，这种预处理受限于历史序列的简单分解效应，并忽略了长期未来序列底层模式之间的层次交互。本文从一个新的渐进维度提出分解思想。我们的 Autoformer 将分解作为深度模型的内部块，它可以在整个预测过程中逐步分解隐藏序列，包括过去的序列和预测的中间结果。

我们将 Transformer [41] 更新为深度分解架构（图 1），包括内置序列分解模块、Auto-Correlation 机制以及相应的 Encoder and Decoder。

![img](https://img-blog.csdnimg.cn/70ab912a1705487a83df332d022c52af.png)

**Encoder 通过序列分解模块（蓝色模块）消除了 long-term trend-cyclical 部分，并专注于 seasonal 模式建模。Decoder 逐步累积从隐藏变量中提取的 trend 部分。 encoder-decoder Auto-Correlation（Decoder 中的中间绿色块）利用来自 encoder 的过去 seasonal 信息。**

时间序列一般可以分为  趋势 +季节性+残差。"Long-term trend-cyclical" 通常被认为是趋势成分的一部分，因为它反映了较长时间尺度上的趋势变化，但也包括一些周期性的变化。

**序列分解模块**   

为了在 long-term 预测上下文中学习复杂的时间模式，我们采用分解的思想 [1, 33]，它可以将序列分为 trend-cyclical 部分和 seasonal 部分。 这两个部分分别反映了该序列的 long-term 发展和 seasonality。

对于未来的序列来说，直接分解是无法实现的，因为未来是未知的。 为了解决这个难题，我们提出了一个序列分解模块作为 Autoformer 的内部操作（图 1），它可以**从预测的中间隐藏变量中逐步提取 long-term 平稳 trend** 。 具体而言，**我们采用 moving average 以平滑周期性波动并突出 long-term trends。** 对于 length-L 的输入序列 $X \in \mathbb{R}^{L \times d}$ 过程为:
$$
\begin{aligned}
& \mathcal{X}_{\mathrm{t}}=\operatorname{Avg} \operatorname{Pool}(\operatorname{Padding}(\mathcal{X})) \\
& \mathcal{X}_{\mathrm{s}}=\mathcal{X}-\mathcal{X}_{\mathrm{t}}
\end{aligned}
$$
这里 $X_t, X_s \in \mathbb{R}^{L \times d}$ 分别代表 **seasonal 部分和 提取的 trend-cyclical 部分。**我们采用 带有 padding 操作的 AvgPool (・) 进行 moving average ，填充操作用来保持序列长度不变。我们用$X_s, X_t=\operatorname{SeriesDecomp}(X)$ 来总结上面的方程，这是一个模型内部模块

**模型输入**

Encoder 部分的输入是过去的 $I$个时间步 $X_{e n} \in \mathbb{R}^{I \times d}$ 作为一种分解架构（图 1），**Autoformer decoder 的输入包含要细化的 seasonal 部分**$X_{\text {des }} \in \mathbb{R}^{\left(\frac{I}{2}+O\right) \times d}$ 和 trend-cyclical 部分 

