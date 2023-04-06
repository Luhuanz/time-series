# Autoformer

AutoFormer 旨在研究长时间序列的预测问题。先前的基于 Transformer 的模型采用各种 self-attention 来挖掘长期依赖。长期复杂的时间模式阻碍了模型找到可靠的依赖关系。此外，Transformer 必须采用稀疏版的 point-wise self-attentions 来捕捉较长时间的关联，这也造成了信息利用的瓶颈。AutoFormer 基于 Auto-Correlation 机制的新型分解架构。我们将 series 的分解更新为模型的一个内部模块。这种设计使 Autoformer 具有复杂时间序列的渐进分解能力。另外，受随机过程理论的启发，我们设计了基于序列周期性的自相关机制，在子序列级进行依赖项发现和表示聚合。Auto-Correlation 在效率和准确性上都优于 self-attention。在长期预测中，Autoformer 具有最好的精度，在六个基准上相对提高了 38%。

- 长期序列预测的困难：处理复杂的时间模式，打破计算效率和信息利用的瓶颈。为了解决这两个难题，我们将分解作为一个内置模块引入深度预测模型，并提出 Autoformer 作为一种分解架构。此外，我们还设计了自相关机制来发现基于周期的依赖关系，并聚合来自底层周期的相似子序列。



## 框架分解

###### **Series decomposition block**

我们将序列分解为 trend-cyslical 和周期的模块，这两个模块可以反映序列的长期的 progression 和周期性。于是我们提出下面的框架：

![img](https://filescdn.proginn.com/9d66e901fd192ed3ef2a04d7f3efd362/f18f622826f70e8d54e139c62421ea63.webp)

对于长度为$L$ 的序列$X \in \mathcal{R}^{L * d}$ ,处理方式为:

其中$X_s, X_t \in \mathcal{R}^{L * d}$ 表示**周期和抽取的rend-cyclical 部分 **$\operatorname{AvgPool}(\cdot)$表示 moving average。

###### Model inputs

encoder 部分是过去$I$个时间段,$X_{e n} \in \mathcal{R}^{I * d}$  输入到 Autoformer **decoder** 的部分包含周期性部分

$X_{\text {des }} \in R^{(0.5 * I+O) * d}$ 以及 trend-cyclical 部分$X_{d e t} \in R^{(0.5 * I+O) * d}$

其中：

$X_{e n s}, X_{e n t} \in \mathcal{R}^{\frac{1}{2} * d}$ 表示周期的和 trend-cyclical 部分；

$X_0, X_{\text {Mean }} \in \mathcal{R}^{O * d}$ 表示填充了 0 的 placeholders 和$X_{en}的均值$

###### **Encoder**

Encoder 侧重于季节性建模。Encoder 的输出包含过去的季节性信息，并将用作交叉信息，以帮助 Decoder 优化预测结果。

假设我们有 N 个编码器层。第 l 个编码器层的总体方程如下：

- ”_“表示删除的 trend 部分；

###### **Decoder**

decoder 包含两部分，trend-cyclical 成份的累积结构和周期性分量的叠加自相关机制。每个 decoder 层包含内在的 Auto-Correlation 和 encoder-decoder Auto-Correlation 部分。这样就可以微调预测并且利用过去的周期性信息。

 

![img](Autoformer.assets/v2-29a3c803f6d240d1715e66d96bb5e067_r.jpg)



https://jishuin.proginn.com/p/763bfbd78adf