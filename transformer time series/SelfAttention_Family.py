import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange, repeat


class DSAttention(nn.Module):
    '''De-stationary Attention'''

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        '''
        初始化 De-stationary Attention 模块。这个方法包括以下输入参数：
        mask_flag：控制是否使用掩码，缺省值为 True。
        factor：调整缩放因子的因子，缺省值为 5。
        scale：缩放因子，缺省值为 None。
        attention_dropout：dropout 比率，缺省值为 0.1。
        output_attention：控制是否输出 attention，缺省值为 False。
        forward(self, queries, keys, values, attn_mask, tau=None, delta=None)：
        执行 De-stationary Attention。这个方法包括以下输入参数：
        queries：查询张量，大小为 (B, L, H, E)，其中 `B'
        '''
        #Non-stationary attention是一个在自然语言处理和计算机视觉中用于解决非平稳时间序列建模问题的注意力层。
        # 其特点是在注意力得分计算中引入了可学习的非平稳性因素，
        # 从而提高了模型在处理非平稳时间序列数据时的表现。在传统的自注意力机制中，得分函数通常是基于查询和键之间的点积或矩阵乘法计算的，
        # 这种方法不考虑时间上的变化，对于非平稳时间序列数据建模效果较差。
        # 而在Non-stationary attention中，得分函数的计算包括可学习的因素，这些因素是与时间变化有关的，
        #从而使得注意力机制可以更好地适应时间序列的变化。
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape  # 获取输入张量的形状 torch.Size([4, 10, 4, 16])
        _, S, _, D = values.shape #torch.Size([4, 10, 4, 16])

        scale = self.scale or 1. / sqrt(E)  # 计算缩放因子
#假设tau的shape为(batch_size,)，则经过如下操作后，tau的shape会变成(batch_size, 1, 1)，
        # 以便与queries和keys的shape进行broadcasting后进行矩阵乘法。
        tau = 1.0 if tau is None else tau.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x 1 # 将 tau 改变形状 torch.Size([4, 1, 1]) no MLP
        delta = 0.0 if delta is None else delta.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x S  # 将 delta 改变形状 no MLP

        # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta # 矩阵乘法，加法，求得 scores torch.Size([4, 4, 10, 10])
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device) # 如果有掩码则应用它 生成mask矩阵
            # print(attn_mask.mask.shape) #torch.Size([4, 1, 10, 10])
            # exit()
            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1)) # 应用缩放因子并进行 softmax，随后进行 dropout
        V = torch.einsum("bhls,bshd->blhd", A, values) # 矩阵乘法，得到输出特征 torch.Size([4, 10, 4, 16])
# self.output_attention 是一个布尔值，表示是否需要输出attention。如果需要输出attention，则返回一个元组(V.contiguous(), A)，
        # 其中 V.contiguous()表示对输出特征进行内存连续化，
        # A表示计算得到的注意力矩阵。如果不需要输出
        # attention，则只返回一个元组(V.contiguous(), None)，其中 None表示没有计算注意力矩阵。
        if self.output_attention:
            return (V.contiguous(), A) # 如果需要输出 attention，将 attention 返回
        else:
            return (V.contiguous(), None) # 只返回输出特征

#Vanila attention block
class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

#概率注意力机制 topK Q
class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        #(B, H, L_K, E) 的 K 张量在倒数第三个维度上添加一个新的维度，变为 (B, H, 1, L_K, E)
        #然后用 expand() 方法将其复制扩展为 (B, H, L_Q, L_K, E)
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
#torch.randint 函数从区间 [0, L_K) 中随机生成 L_Q x sample_k 个整数，
        index_sample = torch.randint(L_K, (L_Q, sample_k))
#从K中抽取一些列作为样本，以构成一个大小为(L_Q, sample_k, E)的张量K_sample。具体来说，torch.arange(L_Q).unsqueeze(1）
        #(B, H, L_Q, sample_k, E)
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        #(B, H, L_Q, L_Q)
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()
        #表示在 K_sample 中每个 query 对应的 top-k 的 attention scores 的最大值，
        # 即 Q 和 K_sample 中每个 query 和 top-k keys 的 attention score 的最大值。
        #torch.div(Q_K_sample.sum(-1), L_K) 表示 Q 和 K_sample 中每个 query 和 top-k keys 的 attention score 的平均值。
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
#是取M矩阵在最后一个维度上的前n_top个最大值，返回的是这些最大值在最后一个维度上的索引，即表示最相关的top-k个键值对的位置。
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                                                  None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
                 np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
#u 是用于控制 Q 的行数的参数c 是一个可调参数，通常取值为 $1/\sqrt{d_k}$，$L_q$ 是 Q 的序列长度
#它的作用是控制 Q 的行数，使得在多头注意力中，每个头得到的 Q 矩阵的行数相同。
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)
#对 U_part 和 u 进行了一个限制，确保它们不超过 L_K 和 L_Q。
        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q
        #前 n_top 个最大的注意力得分和
        # sample_k控制对于每个 query，只考虑前 sample_k 个 key，从而加速 attention 的计算。
        # 而 n_top 控制输出最大的前几个 attention score，只有这些 score 才会被保留下来
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
#取topk操作虽然可以减少计算量，但是也可能会损失一些信息，因此需要引入上下文信息来弥补这种损失。
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None


class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, configs,
                 seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        ## TSA层中的3个attention层，分别用于时间维度、维度sender和维度receiver
        self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=configs.output_attention), d_model, n_heads)
        self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=configs.output_attention), d_model, n_heads)
        self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                         output_attention=configs.output_attention), d_model, n_heads)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))  #seg_num 表示数据张量 Data_dim 的分段数  router 是一个可训练的张量；

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None):

        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        #重新排列张量的维度，使得张量变为指定维度的形状。 [batch_size, ts*D, seg_num, d_model]-》 [(batch_size*ts*D), seg_num, d_model]
        #对于一个形状为(a, b, c)的张量，可以使用以下的字符串来将其转换为一个形状为(a*c, b)的张量：
        #rearrange(x, 'a b c -> (a c) b')

        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc, attn = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)
        dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)

        return final_out

if __name__ == '__main__':
########################################################DSAttention #####################################
    # import torch
    # # 模拟数据
    # batch_size = 4
    # seq_len = 10
    # hidden_size = 16
    # num_heads = 4
    #
    # queries = torch.randn(batch_size, seq_len, num_heads, hidden_size) #torch.Size([4, 10, 4, 16])
    # keys = torch.randn(batch_size, seq_len, num_heads, hidden_size)
    # values = torch.randn(batch_size, seq_len, num_heads, hidden_size)
    # attn_mask = None
    # # tau = torch.randn(batch_size)
    # # tau=1
    # # # delta = torch.randn(batch_size, seq_len, seq_len)
    # # delta=0
    #
    # # 实例化 DSAttention
    # attention = DSAttention(mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False)
    #
    # # 使用 DSAttention 进行前向传播
    # outputs = attention(queries, keys, values, attn_mask)
    #
    # print(outputs[0].shape)  # 输出特征的形状
##########################################Fullattetion#####################################################
    # import torch
    #
    # # 模拟数据
    # batch_size = 4
    # seq_len = 10
    # hidden_size = 16
    # num_heads = 4
    #
    # queries = torch.randn(batch_size, seq_len, num_heads, hidden_size)  # torch.Size([4, 10, 4, 16])
    # keys = torch.randn(batch_size, seq_len, num_heads, hidden_size)
    # values = torch.randn(batch_size, seq_len, num_heads, hidden_size)
    # attn_mask = None
    # # tau = torch.randn(batch_size)
    # # tau=1
    # # # delta = torch.randn(batch_size, seq_len, seq_len)
    # # delta=0
    #
    # # 实例化 DSAttention
    # attention = FullAttention(mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False)
    #
    # # 使用 DSAttention 进行前向传播
    # outputs = attention(queries, keys, values, attn_mask) #torch.Size([4, 10, 4, 16])
    # print(outputs[0].shape)  # 输出特征的形状
    pass
