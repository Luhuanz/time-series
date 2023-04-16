import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Transformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')
    # data loader
    #required=True 表示这个参数是必需的，也就是说在使用脚本时必须要提供这个参数，否则程序就会报错并提醒用户需要提供这个参数。 不设置就是默认
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
  #M：多变量预测多变量。在这种情况下，模型需要使用多个输入变量来预测多个输出变量，例如使用多个传感器读数来预测某个物理过程中多个变量的变化。
    #S：单变量预测单变量。在这种情况下，模型使用一个输入变量来预测一个输出变量，例如使用历史气温来预测未来气温。
    #MS：多变量预测单变量。在这种情况下，模型需要使用多个输入变量来预测一个输出变量，例如使用历史气温、湿度和风速来预测未来气温。
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
  #告诉程序在对时间特征进行编码时使用何种频率。例如，指定 freq='h' 表示使用小时作为时间特征编码的频率，指定 freq='d' 表示使用天作为频率。
# 另外，freq 参数也支持更详细的设置，例如 freq='15min' 表示使用 15 分钟作为时间特征编码的频率。
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
 #--label_len 是一个参数名，表示模型输出序列的长度（标签长度），类型为整数，缺省值为 48
    # ，帮助信息为 “start token length”。这个参数是用于控制模型生成序列的长度的，
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    #预测序列的长度，即需要预测多少个时间步长（或者说多少个数据点）。
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
 #M4数据集的子集名称
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # inputation task
#mask_rate 指的是需要被遮蔽（即被置为 NaN 或其他特殊值）的数据比例。
# 在数据填充任务中，我们会随机遮蔽一定比例的数据，然后使用模型预测遮蔽的数据，从而实现数据填充
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
#异常检测任务，指定先验异常率，以便在训练期间对数据进行标记并进行异常检测。假设数据集中有一定比例的异常值，使用此参数可以让模型更好地学习如何检测异常值。
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    # TimesBlock中卷积核的数量。
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    #num_kernels: Inception模型中卷积核的数量。
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    #enc_in和dec_in: 编码器和解码器输入的大小。
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    #c_out: 模型输出的大小。
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    #d_model: 模型中的向量维度。
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    #n_heads: 注意力机制中的头数。
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    #e_layers和d_layers: 编码器和解码器中的层数。
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    #d_ff: 模型中全连接层的维度。
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    #moving_avg: 移动平均窗口的大小。
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    #factor: 注意力机制的缩放因子。
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    #distil: 是否在编码器中使用知识蒸馏。如果使用，则为True，否则为False。
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)

    #一些深度学习模型的超参数，用于指定模型架构、优化器、训练方式等方面。

    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    #embed: 时间特征编码方式，有三种选项：timeF、fixed、learned。
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    #是否输出encoder中的attention权重矩阵。
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    #itr: 实验重复次数。
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    #参数用于设置实验的描述或名称，方便在实验记录或结果输出中进行标识和区分。默认值为test。
    # 例如，如果要运行多个不同的实验，可以使用不同的--des参数值来区分它们
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    # 该参数用于控制学习率的调整方式，可选值包括：
    # type1: 按照预设的epoch进行学习率调整
    # type2: 当验证集上的指标连续多个epoch没有提升时，降低学习率
    # type3: 当验证集上的指标连续多个epoch没有提升时，降低学习率，并且在指标连续多个epoch没有提升的情况下，提前终止训练
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    #使用自动混合精度训练（Automatic Mixed Precision，简称 AMP）来加速训练并减少内存使用。如果该参数被设置为 False 或未设置，默认不使用 AMP。
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')


    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
