# Non-stationary  

时间序列的不平稳性（non-stationarity）是一个比较难处理，且真实世界中很常见的问题。**时间序列的不平稳性指的是随着时间的变化，观测值的均值、方差等统计量发生变化**。不平稳性会导致在训练集训练的模型，在测试集上效果较差，因为训练集和测试集属于不同时间，而不同时间的数据分布差异较大。业内解决这种统计量随时间变化的不平稳问题主要方法是，对时间序列数据做一些诸如归一化等平稳化处理。例如对每个序列样本使用 z-normalization 处理成 0 均值 1 方差的，这样就可以解决不同时间步的样本统计量不一致的问题。**但是这种解决方法会对 Transformer 模型带来一个负面影响：平稳化后的序列虽然统计量一致了，但是这个过程中也让数据损失了一些个性化的信息，导致不同序列的 Transformer 中的 attention 矩阵趋同**。文中将这个现象叫作 **over-stationarization**。

将非平稳序列**平稳化**（如序列分解、归一化）是时间序列预测中的常见手段，平稳后更好预测一些，一般会提高预测性能。但这篇文章认为，**平稳化会存在过渡平稳化的问题，导致对于有着不同属性特征的序列，Transformer 模型学到的 Attention Map 很相似。**

对于一个序列的 3 个时间窗口的子序列，不进行归一化处理的 attention 分布差别很大，而使用了归一化处理后，3 个序列的 attention 分布趋同了。这也是导致 Transformer 模型在一些 non-stationary 数据上效果不好的原因之一。

![image-20230413063736858](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20230413063736858.png)

这篇文章提出了一个框架如下图，包括两个主要模块。一个是平稳化模块（Series Stationarization）用来进行序列平稳化（提高可预测性），一个是去平稳化注意力机制（De-stationary Attention）用来缓解过渡平稳化问题。

![image-20230413065734644](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20230413065734644.png)

### Series Stationarization

序列平稳化由两个模块来实现，分别是输入时归一化（**Normalization** module）和输出时逆归一化（**De-normalization** module），归一化时算出来的均值和方差会送到逆归一化层，来还原序列的统计量特征。

#### Normalization module

 求均值$\mu_x$和方差$\sigma_x^2$ ，将序列归一化再送入模型。

- 输入归一化

$$
\mu_{\mathbf{x}}=\frac{1}{S} \sum_{i=1}^S x_i, \sigma_{\mathbf{x}}^2=\frac{1}{S} \sum_{i=1}^S\left(x_i-\mu_{\mathbf{x}}\right)^2, x_i^{\prime}=\frac{1}{\sigma_{\mathbf{x}}} \odot\left(x_i-\mu_{\mathbf{x}}\right)
$$
####  De-normalization module

 利用上面的均值和方差逆归一化。

-  输出逆归一化

$$
\mathbf{y}^{\prime}=\mathcal{H}\left(\mathbf{x}^{\prime}\right), \hat{y}_i=\sigma_{\mathbf{x}} \odot\left(y_i^{\prime}+\mu_{\mathbf{x}}\right)
$$

正如 main idea 提到的，这样归一化（平稳化）后，可能会提升预测能力，但也可能会带来过渡平稳化现象。因此，又引入了 De-stationary Attention 来缓解过渡平稳化，提升模型能力。

### De-stationary Attention

> 将序列平稳化了再输入到模型中，那算出来的 Attention 肯定是由平稳化后的序列计算得到的，存在过渡平稳化问题。但我们希望，模型中的 Attention Matrix 实际上还是算的是非平稳序列的 Attention Matrix，也就是说，**De-stationary Attention 的目的就是想通过平稳后序列的 Attention Matrix 来近似原始非平稳序列的** Attention Matrix。

$Q^{\prime}, K^{\prime}, V^{\prime}$ 是由平稳化后序列得到的， 而 $Q, K, V$ 是由平稳化前序列得到的。

平稳化前的序列计算 Attention 如下，这其实就是我们的目标：
$$
\operatorname{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V})=\operatorname{Softmax}\left(\frac{\mathbf{Q K}^{\mathrm{T}}}{\sqrt{d_k}}\right) \mathbf{V}
$$
由于平稳化过程是一个归一化的过程，可以将**平稳化后**的$Q^{\prime} K^{\prime T}$ 展开为：
$$
\mathbf{Q}^{\prime} \mathbf{K}^{\prime \mathrm{T}}=\frac{1}{\sigma_{\mathbf{x}}^2}\left(\mathbf{Q K}^{\mathrm{T}}-\mathbf{1}\left(\mu_{\mathbf{Q}}^{\mathrm{T}} \mathbf{K}^{\mathrm{T}}\right)-\left(\mathbf{Q} \mu_{\mathbf{K}}\right) \mathbf{1}^{\mathrm{T}}+\mathbf{1}\left(\mu_{\mathbf{Q}}^{\mathrm{T}} \mu_{\mathbf{K}}\right) \mathbf{1}^{\mathrm{T}}\right)
$$
然后将**平稳化后**的$Q^{\prime} K^{\prime T}$ 带入我们的目标可以将我们目标的 Attention Matrix 改写为：
$$
\operatorname{Softmax}\left(\frac{\mathbf{Q K}^{\mathrm{T}}}{\sqrt{d_k}}\right)=\operatorname{Softmax}\left(\frac{\sigma_{\mathrm{x}}^2 \mathbf{Q}^{\prime} \mathbf{K}^{\prime \mathrm{T}}+\mathbf{1}\left(\mu_{\mathbf{Q}}^{\mathrm{T}} \mathbf{K}^{\mathrm{T}}\right)+\left(\mathbf{Q} \mu_{\mathbf{K}}\right) \mathbf{1}^{\mathrm{T}}-\mathbf{1}\left(\mu_{\mathbf{Q}}^{\mathrm{T}} \mu_{\mathbf{K}}\right) \mathbf{1}^{\mathrm{T}}}{\sqrt{d_k}}\right)
$$
其中后两项是重复在每列操作，都不影响 Softmax 后的结果。比如对矩阵的任意一行来说，后两项就相当于为该行的每一个元素加上相同的值，并且由于 Softmax 是对矩阵的每一行操作，是否加这个相同的值对 Softmax 的结果没有影响。因此，可以直接去掉后面两项，上面的式子就可以化简为：
$$
\operatorname{Softmax}\left(\frac{\mathbf{Q K}^{\mathrm{T}}}{\sqrt{d_k}}\right)=\operatorname{Softmax}\left(\frac{\sigma_{\mathbf{x}}^2 \mathbf{Q}^{\prime} \mathbf{K}^{\prime \mathrm{T}}+\mathbf{1} \mu_{\mathbf{Q}}^{\mathrm{T}} \mathbf{K}^{\mathrm{T}}}{\sqrt{d_k}}\right)
$$
搭建了一个从 **平稳后序列的 Attention Matrix** 来得到 **平稳前原始序列的 Attention Matrix**（即我们的目标）的桥梁。

$\sigma_x^2,\left(K \mu_Q\right)^T$ 无法从平稳后序列中得到的。因此，可以使用 MLP 来学习这两个量，即使用额外的两个 MLP，一个用来学 $\tau=\sigma_x^2$ 注意这个量是正数，因此可以学它的对数），另一个用来学 $\Delta=K \mu_Q$ 这里的 $\tau, \Delta$ 也被称为 **去平稳因子（de-stationary factors）**

整个的 De-stationary Attention 可以写为：
$$
\begin{gathered}
\log \tau=\operatorname{MLP}\left(\sigma_{\mathbf{x}}, \mathbf{x}\right), \boldsymbol{\Delta}=\operatorname{MLP}\left(\mu_{\mathbf{x}}, \mathbf{x}\right) \\
\operatorname{Attn}\left(\mathbf{Q}^{\prime}, \mathbf{K}^{\prime}, \mathbf{V}^{\prime}, \tau, \boldsymbol{\Delta}\right)=\operatorname{Softmax}\left(\frac{\tau \mathbf{Q}^{\prime} \mathbf{K}^{\prime \top}+\mathbf{1} \boldsymbol{\Delta}^{\top}}{\sqrt{d_k}}\right) \mathbf{V}^{\prime}
\end{gathered}
$$