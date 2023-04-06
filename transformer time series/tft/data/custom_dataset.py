import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.utils import get_single_col_by_input_type
from utils.utils import extract_cols_from_data_type
from data_formatters.electricity import ElectricityFormatter
from data_formatters.base import DataTypes, InputTypes

class TFTDataset(Dataset, ElectricityFormatter):
    """Dataset Basic Structure for Temporal Fusion Transformer"""

    def __init__(self, 
                 data_df):
        super(ElectricityFormatter, self).__init__()
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        # Attribute loading the data
        self.data = data_df.reset_index(drop=True)
        
        self.id_col = get_single_col_by_input_type(InputTypes.ID, self._column_definition)
        self.time_col = get_single_col_by_input_type(InputTypes.TIME, self._column_definition)
        self.target_col = get_single_col_by_input_type(InputTypes.TARGET, self._column_definition)
        self.input_cols = [
                            tup[0]
                            for tup in self._column_definition
                            if tup[2] not in {InputTypes.ID, InputTypes.TIME}
                          ]
        self.col_mappings = {
                              'identifier': [self.id_col],
                              'time': [self.time_col],
                              'outputs': [self.target_col],
                              'inputs': self.input_cols
                          }
        self.lookback = self.get_time_steps()
        self.num_encoder_steps = self.get_num_encoder_steps()
        
        self.data_index = self.get_index_filtering()
        self.group_size = self.data.groupby([self.id_col]).apply(lambda x: x.shape[0]).mean()
        self.data_index = self.data_index[self.data_index.end_rel < self.group_size].reset_index()
        
    def get_index_filtering(self):
        
        g = self.data.groupby([self.id_col])
        
        df_index_abs = g[[self.target_col]].transform(lambda x: x.index+self.lookback) \
                        .reset_index() \
                        .rename(columns={'index':'init_abs',
                                         self.target_col:'end_abs'})
        df_index_rel_init = g[[self.target_col]].transform(lambda x: x.reset_index(drop=True).index) \
                        .rename(columns={self.target_col:'init_rel'})
        df_index_rel_end = g[[self.target_col]].transform(lambda x: x.reset_index(drop=True).index+self.lookback) \
                        .rename(columns={self.target_col:'end_rel'})
        df_total_count = g[[self.target_col]].transform(lambda x: x.shape[0] - self.lookback + 1) \
                        .rename(columns = {self.target_col:'group_count'})
        
        return pd.concat([df_index_abs, 
                          df_index_rel_init,
                          df_index_rel_end,
                          self.data[[self.id_col]], 
                          df_total_count], axis = 1).reset_index(drop = True)

    def __len__(self):
        # In this case, the length of the dataset is not the length of the training data, 
        # rather the ammount of unique sequences in the data
        return self.data_index.shape[0]

    def __getitem__(self, idx):
        
        data_index = self.data.iloc[self.data_index.init_abs.iloc[idx]:self.data_index.end_abs.iloc[idx]]
        
        data_map = {}
        for k in self.col_mappings:
            cols = self.col_mappings[k]
            
            if k not in data_map:
                data_map[k] = [data_index[cols].values]
            else:
                data_map[k].append(data_index[cols].values)
                
        # Combine all data
        for k in data_map:
            data_map[k] = np.concatenate(data_map[k], axis=0)
        # Shorten target so we only get decoder steps
        data_map['outputs'] = data_map['outputs'][self.num_encoder_steps:, :]
        
        active_entries = np.ones_like(data_map['outputs'])
        if 'active_entries' not in data_map:
            data_map['active_entries'] = active_entries
        else:
            data_map['active_entries'].append(active_entries)
                
        return data_map['inputs'], data_map['outputs'], data_map['active_entries']

if __name__ == '__main__':
    import pandas as pd
    # 随机生成数据 13取 5
    # power_usage: 电力功率使用量，这个数据集中最主要的一个特征。
    # t: 时间戳或时间步，表示该数据点所对应的时间。
    # days_from_start: 从数据集开始时间到当前时间的天数。
    # categorical_id: 对某个实体或对象的分类，可能是离散化后的某个特征。
    # date: 具体的日期，是由时间戳转换而来的。
    # id: 对某个实体或对象的唯一标识。
    # hour: 具体的小时数。
    # day: 具体的天数。
    # day_of_week: 星期几，可能是离散化后的某个特征。
    # month: 具体的月份。
    # hours_from_start: 从数据集开始时间到当前时间的小时数。
    # categorical_day_of_week: 星期几，可能是离散化后的某个特征。
    # categorical_hour: 具体的小时数，可能是离散化后的某个特征。
    data_df = pd.DataFrame({
        'id': ['id_1'] * 1000,
        'power_usage': np.random.normal(0, 1, 1000),  #下面是输入神经网络的特征
        'hours_from_start': np.random.normal(0, 1, 1000),
        'hour': np.random.normal(0, 1, 1000),
        'day_of_week': np.random.normal(0, 1, 1000),
        'categorical_id': np.zeros(1000)
    })

    # 初始化TFTDataset类
    dataset = TFTDataset(data_df)
    from torch.utils.data import DataLoader

    # 定义数据加载器
    batch_size = 64
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 遍历数据集
    for batch in data_loader:
        inputs, outputs, active_entries = batch
        print(inputs.shape)  # (batch_size, num_inputs, num_encoder_steps) torch.Size([64, 192, 5])
        print(outputs.shape)  # (batch_size, num_decoder_steps, num_outputs) torch.Size([64, 24, 1])
        print(active_entries.shape)  # (batch_size, num_decoder_steps, num_outputs) torch.Size([64, 24, 1])
        break# 只打印第一个batch



