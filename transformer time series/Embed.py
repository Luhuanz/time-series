import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
#d_model是嵌入向量的维度，max_len是最大序列长度，因为在训练过程中，位置嵌入向量的维度和序列长度都是固定的，所以需要一个预先指定的最大序列长度。
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
#这里定义了一个形状为[max_len, d_model]的全零张量pe，该张量的数据类型为float，并且设置了require_grad=False，意味着它不需要梯度
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
#[max_len, 1]的张量position，表示序列中每个位置的索引，即[0, 1, 2, ..., max_len-1]。
        position = torch.arange(0, max_len).float().unsqueeze(1)
#定义一个形状为[d_model/2]的张量div_term，用于计算位置嵌入向量中奇偶位置的权重。具体来说，它计算了一个分母为10000的指数函数，然后将其拆分成了两个部分，分别作为奇偶位置的权重。
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp() #/omiga
#这里根据位置嵌入向量的定义，对全零张量pe进行修改，将每个位置嵌入向量的奇偶位置用正弦函数和余弦函数计算出来。
        pe[:, 0::2] = torch.sin(position * div_term) # 所有行偶数列
        pe[:, 1::2] = torch.cos(position * div_term) # 所有行奇数列
#pe张量加一维，在第0维插入了一个维度，变成了形状为[1, max_len, d_model]，然后将其注册为该类的一个缓存变量。
        pe = pe.unsqueeze(0) # batch max_len d_model
#self.register_buffer('pe', pe) 是将 pe tensor 注册为一个 buffer，这样它会被视为模型参数的一部分，并在反向传播过程中自动更新。
     # 这是因为在 PyTorch 中，只有 nn.Module 类的属性才会被视为模型参数，在反向传播时会被自动更新，而不是像普通 tensor 一样只是记录数据。
        self.register_buffer('pe', pe)
#用于计算输入序列的位置嵌入向量。输入x是形状为[batch_size, seq_len, d_model]的张量，它的第二个
    def forward(self, x):
        #pe[:, :x.size(1)] 是在将位置嵌入张量 pe 中提取与输入张量 x 大小匹配的子张量
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    # c_in 时间步子不是sen_length 而是 nums_dature 如 年月日时分 5个时间步，d_model 表示词嵌入的维度或者说通道数。
#输入token序列转换为词嵌入的模块。它采用一个一维卷积层来处理输入，并将输出转换为与模型中的其他张量相同大小的张量。
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
    #遍历当前 TokenEmbedding 模块中的所有子模块（包括 TokenEmbedding 自身
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
    #对网络权重进行初始化，有助于网络更快地收敛并提高训练效果。
#其作用是初始化卷积层或线性层的权重，使其服从均值为0、标准差为$\sqrt{\frac{2}{(1+a^2)\times fan_{in}}}$的正态分布，
# 其中 $a$ 为 LeakyReLU 的负斜率，$fan_{in}$ 是权重张量输入通道的数量。这种初始化方法有助于保持数据在网络中的方差，
    # 从而避免了梯度消失或梯度爆炸的问题。
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
      # print(x.shape) #torch.Size([2, 10, 5])
      #   print(x.permute(0, 2, 1).shape) #torch.Size([2, 5, 10])
      #   exit()
      #   print(self.tokenConv(x.permute(0, 2, 1)).shape) #10-3+4 10   torch.Size([2, 16, 10])
      #   exit()
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

#FixedEmbedding是将token映射到固定的embedding向量上，这些embedding向量通常是在训练过程中预定义的，并且不会因为输入序列的不同而改变
class FixedEmbedding(nn.Module): # 整数转词向量 c_in 表示时间维度
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
#将整数序列映射为向量序列的类，通常用于 NLP 中将词汇表中的每个词映射为向量表示。
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False) #w 转换成了一个不需要梯度更新的可训练张量

    def forward(self, x):
#detach()方法可以使梯度不会在该层被传递到之前的层。这个类的好处在于，它可以在每次前向传递时使用相同的位置编码矩阵，而不是在每次传递时重新计算位置编码。
        return self.emb(x).detach()

#该类通过将时间维度中的年、月、日、小时、星期等信息嵌入到模型中来帮助模型更好地理解时间特征。
class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()
        '''
        d_model: 嵌入向量的维度大小。
        embed_type: 嵌入类型，可以是 'fixed' 或 'nn'。
        freq: 时间频率，可以是 'h'（小时）或 't'（分钟）。
        '''
        #时间特征中每个维度的取值范围
        minute_size = 4 #表示分钟维度的取值范围，这里是4，因此表示分钟可能的取值为0, 1, 2, 3。
        hour_size = 24 #表示小时维度的取值范围，这里是24，表示小时可能的取值为0, 1, 2, ..., 23。
        weekday_size = 7
        day_size = 32 #表示月中的天数维度的取值范围，这里是32，表示一个月中的天数可能的取值为1, 2, ..., 31
        month_size = 13 #表示月份维度的取值范围，这里是13，表示月份可能的取值为1, 2, ..., 12

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        #freq 表示时间序列的频率
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model) #24 16
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):

        x = x.long() #torch.Size([2, 10, 5])
        #print(x[:, :, 3].shape) #torch.Size([2, 10])
        # #x[:, :, 4]表示选取x张量中的所有batch的所有时间步的第5个特征，即第5列数据。结果是一个二维张量，维度为(batch_size, seq_len)。
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(  #0
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3]) #torch.Size([2, 10, 16])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()
#示不同的时间频率对应的时间特征的长度。例如，'h'表示每小时的时间特征，长度为4，包含小时数、分钟数、秒数和毫秒数。
        # 'm'表示每分钟的时间特征，长度为1，只包含分钟数。
        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()
        #固定嵌入（fixed）
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        #时间嵌入（temporal） #时间特征嵌入（timeF） 二选一 如果fixed 使用   temporal_embedding用来嵌入时间信息。
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)

#DataEmbedding_wo_pos 表示不使用位置编码
class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    '''
    patch_len 表示每个 patch（块）的长度；
    stride 表示每个 patch 的步长；
    padding 表示对于输入序列两端的 padding 的大小；
    value_embedding 是将每个 patch 映射到 d_model 维空间的线性变换；
    position_embedding 是用于对每个 patch 的位置进行编码的位置编码器；
    最终返回的结果是将每个 patch 进行 value_embedding 和 position_embedding 之后，再经过 dropout 的输出结果以及输入序列的长度 n_vars。
    '''

    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        #一种1D卷积神经网络中的padding方式，它可以将一维的序列在两端进行复制扩展以达到指定的 padding 大小。
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
#，输入通常是一个1D的序列，每个元素代表一个特征。因此，padding操作需要在这个维度上进行。
# 而在图像的情况下，padding操作通常是在width和height这两个维度上进行，因为图像是由一个个像素组成的。
        x = self.padding_patch_layer(x) #torch.Size([2, 10, 256])
 # 最后一个维度即上，步长为
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride) #torch.Size([2, 10, 31, 16])

        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])) #orch.Size([20, 31, 16])

        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x) #torch.Size([20, 31, 512])
        return self.dropout(x), n_vars
if __name__ == '__main__':
    ########################################posembedding################################
#     batch_size = 2
#     seq_len = 10
#     d_model = 6 # 得是偶数
#     x = torch.randn(batch_size, seq_len, d_model)
#     # pos_embedding = PositionalEmbedding(d_model)
#     # pos_embed = pos_embedding(x)
#     # print(pos_embed)
# #######################Tokebembedding################################################################
    # # 实例化 TokenEmbedding 类
    #  batch_size = 2
    #  seq_len = 10
    #  num_features=5
    #  c_in= num_features
    # #c_in 是 input_channel
    #  d_model = 16 #
    #  x = torch.randn(batch_size, seq_len, num_features)
    #  embe = TokenEmbedding(c_in=c_in,d_model=d_model) #torch.Size([2, 16, 10])
    # # 将输入数据传递给 TokenEmbedding 类的 forward() 方法
    #  output = embe(x)
    #
    # # 查看输出张量的形状
    #  print(output.shape)
########################################Fixembedding####################################
    # c_in = 12  # 输入的词汇表大小
    # d_model = 20  # 词向量的维度
    #
    # # 随机生成一个形状为 (2, 5) 的整数张量，模拟两个句子，每个句子长度为 5
    # x = torch.randint(0, c_in, (2, 5)) #元素都是在 [0, c_in) 范围内随机生成的整数。
    #     #tensor([[7, 9, 6, 3, 8],
    #          # [7, 3, 3, 1, 8]])
    #
    # # 实例化 FixedEmbedding 类
    # embedding = FixedEmbedding(c_in, d_model)
    #
    # # 对输入数据进行编码，得到输出张量
    # output = embedding(x)
    #
    # # 输出张量的形状为 (2, 5, 20)
    # print(output.shape)
##############################################TemporalEmbedding###########################
    # import torch
    # from torch import nn
    #
    # d_model = 16
    # embed_type = 'fixed'
    # freq = 'h'
    #
    # batch_size = 2
    # seq_len = 10
    # num_features = 5
    #
    # x1 = torch.randint(0,4, (batch_size, seq_len)) #生成的数据不能超过表的大小
    # x2 = torch.randint(0,24, (batch_size, seq_len))
    # x3 = torch.randint(0,7, (batch_size, seq_len))
    # x4 = torch.randint(0,32, (batch_size, seq_len))
    # x5 = torch.randint(0, 13, (batch_size, seq_len))
    # x = torch.zeros((batch_size, seq_len,num_features))
    # x[:,:,4]=x1
    # x[:,:,3]=x2
    # x[:,:,2]=x3
    # x[:, :, 1]=x4
    # x[:, :, 0]=x5
    # temporal_embedding = TemporalEmbedding(d_model, embed_type, freq)
    # output = temporal_embedding(x)
    # print(output.shape)  # should be (batch_size, seq_len, d_model) ##torch.Size([2, 10, 16])
###################################################TimeFeatureEmbedding################################
    #
    # import torch
    #
    # batch_size = 2
    # seq_len = 3
    # num_features = 4 # 表示 h  如果 5表示 t
    #
    # x = torch.zeros((batch_size, seq_len, num_features))
    # d_model = 8
    # freq = 'h'
    # time_embed = TimeFeatureEmbedding(d_model, freq)
    #
    # output = time_embed(x)
    # print(output.shape) #torch.Size([2, 3, 8])
#################################################DataEmbedding#################################
        # import torch
        # from torch import nn
        #
        # d_model = 16
        # embed_type = 'fixed'
        # freq = 'h'
        #
        # batch_size = 2
        # seq_len = 10
        # num_features = 5
        # c_in=num_features
        # x=x = torch.randn(batch_size, seq_len, num_features)
        # x_ = torch.zeros((batch_size, seq_len, num_features)) #时间信息
        # x1 = torch.randint(0,4, (batch_size, seq_len)) #生成的数据不能超过表的大小
        # x2 = torch.randint(0,24, (batch_size, seq_len))
        # x3 = torch.randint(0,7, (batch_size, seq_len))
        # x4 = torch.randint(0,32, (batch_size, seq_len))
        # x5 = torch.randint(0, 13, (batch_size, seq_len))
        # x = torch.zeros((batch_size, seq_len,num_features))
        # x_[:,:,4]=x1
        # x_[:,:,3]=x2
        # x_[:,:,2]=x3
        # x_[:, :, 1]=x4
        # x_[:, :, 0]=x5
        # # 随机生成额外的时间特征信息
        # # x_mark = None
        # x_mark =x_
        #
        # # 实例化 DataEmbedding 类
        # embedding = DataEmbedding(c_in=c_in, d_model=d_model, embed_type='fixed', freq='h', dropout=0.1)
        #
        # # 对输入数据进行嵌入操作
        # embedded_x = embedding(x, x_mark)
        #
        # print(embedded_x.shape) #torch.Size([2, 10, 16])

####################################PatchEmbedding######################################################
    import torch

    d_model = 512
    patch_len = 16
    stride = 8
    padding = 2
    dropout = 0.1

    batch_size = 2
    seq_len = 10
    input_dim = 256 # dim 是 stridex patch_len的倍数 因为用了unfold

    # 随机生成输入数据，形状为 [batch_size, n_vars]
    x = torch.randn(batch_size, seq_len, input_dim)
    patch_embedding = PatchEmbedding(d_model, patch_len, stride, padding, dropout)
    x, n_vars = patch_embedding(x)

    # 输出处理后的结果
    print('Processed input shape:', x.shape)
    print('Number of variables after patching:', n_vars)