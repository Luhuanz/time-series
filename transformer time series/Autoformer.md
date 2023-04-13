# Autoformer

作为时间序列分析中的一种标准方法，时间序列分解 [1, 33] 将时间序列解构为几个部分，每个部分代表一种更可预测的基本模式类别。它主要用于探索随着时间推移的历史变化。对于预测任务，在预测未来序列 [20, 2] 之前，总是使用分解作为历史序列的预处理，例如带有 **trend-seasonality 分解的 Prophet [39] 和带有基扩展（basis expansion）**的 N-BEATS [29] 和 使用矩阵分解的 DeepGLO [35] 。然而，这种预处理受限于历史序列的简单分解效应，并忽略了长期未来序列底层模式之间的层次交互。本文从一个新的渐进维度提出分解思想。我们的 Autoformer 将分解作为深度模型的内部块，它可以在整个预测过程中逐步分解隐藏序列，包括过去的序列和预测的中间结果。

我们将 Transformer [41] 更新为深度分解架构（图 1），包括内置序列分解模块、Auto-Correlation 机制以及相应的 Encoder and Decoder。

![img](https://img-blog.csdnimg.cn/70ab912a1705487a83df332d022c52af.png)

**Encoder 通过序列分解模块（蓝色模块）消除了 long-term trend-cyclical 部分，并专注于 seasonal 模式建模。Decoder 逐步累积从隐藏变量中提取的 trend 部分。 encoder-decoder Auto-Correlation（Decoder 中的中间绿色块）利用来自 encoder 的过去 seasonal 信息。**

时间序列一般可以分为  趋势 +季节性+残差。"Long-term trend-cyclical" 通常被认为是趋势成分的一部分，因为它反映了较长时间尺度上的趋势变化，但也包括一些周期性的变化。

**序列分解模块**  