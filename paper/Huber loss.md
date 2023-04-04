Huber loss是一种用于回归问题的损失函数。它的设计目的是在损失函数中同时考虑MSE（均方误差）和MAE（平均绝对误差）的优点，对于存在噪声或离群点的数据集，表现更为鲁棒。

在Huber loss中，对于预测值与真实值之差小于一个设定的阈值$\delta$，使用MSE来计算损失，对于差值大于阈值$\delta$，则使用MAE来计算损失，这样可以在一定程度上抑制离群点的影响。因此，Huber loss在一些噪声较大或含有离群点的数据集中比MSE更为合适。

Huber loss的数学表达式如下：
$$
L_\delta(y, \hat{y})= \begin{cases}\frac{1}{2}(y-\hat{y})^2, & \text { if }|y-\hat{y}| \leq \delta \\ \delta|y-\hat{y}|-\frac{1}{2} \delta^2, & \text { otherwise }\end{cases}
$$
其中，$y$为真实值，$\hat{y}$为模型预测值，$\delta$为设定的阈值。