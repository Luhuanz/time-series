import os
import wget
import pyunpack
import pandas as pd
import numpy as np

def download_from_url(url, output_path):
    """Downloads a file froma url."""

    print('Pulling data from {} to {}'.format(url, output_path))
    # wget.download(url, output_path)
    print('done')

def recreate_folder(path):
    """Deletes and recreates folder."""

    shutil.rmtree(path)
    os.makedirs(path)

def unzip(zip_path, output_file, data_folder):
    """Unzips files and checks successful completion."""
#用于解压指定的zip文件并将解压后的文件保存到指定的目录（即data_folder）。
# 其中，zip_path指定了待解压的zip文件路径。如果zip文件中包含多个文件，extractall方法会将这些文件都解压出来并保存到指定目录下。
    print('Unzipping file: {}'.format(zip_path))
    pyunpack.Archive(zip_path).extractall(data_folder)

    # Checks if unzip was successful
    if not os.path.exists(output_file):
        raise ValueError(
            'Error in unzipping process! {} not found.'.format(output_file))

def download_and_unzip(url, zip_path, csv_path, data_folder):
    """Downloads and unzips an online csv file.
    Args:
    url: Web address
    zip_path: Path to download zip file
    csv_path: Expected path to csv file
    data_folder: Folder in which data is stored.
    """

    download_from_url(url, zip_path)

    unzip(zip_path, csv_path, data_folder)

    print('Done.')

def download_electricity(config):
    """Downloads electricity dataset from UCI repository."""

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip'

    data_folder = config.data_folder
    #LD2011_2014.txt是一个数据文件，里面包含一些列时间序列数据，通常用于机器学习和时间序列预测的实验和研究。
    #./data1/LD2011_2014.txt
    csv_path = os.path.join(data_folder, 'LD2011_2014.txt')#拼接成一个完整的路径字符串，返回这个完整路径字符串。
    #'data1/LD2011_2014.txt.zip'
    zip_path = csv_path + '.zip'

    download_and_unzip(url, zip_path, csv_path, data_folder)

    print('Aggregating to hourly data')
#将第一列作为index，设置时间戳格式并按时间顺序排序。index_col=0指定将第一列作为数据的index；sep=';'指定分隔符为分号；
    df = pd.read_csv(csv_path, index_col=0, sep=';', decimal=',')
    #将index转换为时间戳格式
    df.index = pd.to_datetime(df.index)
    #sort_index函数按照时间顺序对数据进行排序。最终得到的是一个按照时间顺序排列的DataFrame对象。
    df.sort_index(inplace=True)

    # Used to determine the start and end dates of a series
    # resample 函数对数据进行重采样，将其按照每小时进行平均值计算。这里的 '1h' 表示以 1 小时为时间间隔进行重采样。
    #
    # 2023 - 04 - 01  00: 00:00, 10
    # 2023 - 04 - 01 06: 00:00, 12
    # ->
    # 2023 - 04 - 01 00: 00:00, 10
    # 2023 - 04 - 01 01: 00:00, NaN
    # 2023 - 04 - 01 02: 00:00, NaN
    # ...
    # 2023 - 04 - 01 05: 00:00, NaN
    # 2023 - 04 - 01 06: 00:00, 12

    output = df.resample('1h').mean().replace(0., np.nan) # 0换为nan
#output.index是以小时为间隔重采样后的时间戳索引，.min()方法返回最小时间戳，也就是最早的时间。
    earliest_time = output.index.min()

    df_list = []
    for label in output:
        print('Processing {}'.format(label))
        srs = output[label]

        start_date = min(srs.fillna(method='ffill').dropna().index)
        end_date = max(srs.fillna(method='bfill').dropna().index)

        active_range = (srs.index >= start_date) & (srs.index <= end_date)
        srs = srs[active_range].fillna(0.)

        tmp = pd.DataFrame({'power_usage': srs})
        date = tmp.index
        tmp['t'] = (date - earliest_time).seconds / 60 / 60 + (
            date - earliest_time).days * 24
        tmp['days_from_start'] = (date - earliest_time).days
        tmp['categorical_id'] = label
        tmp['date'] = date
        tmp['id'] = label
        tmp['hour'] = date.hour
        tmp['day'] = date.day
        tmp['day_of_week'] = date.dayofweek
        tmp['month'] = date.month

        df_list.append(tmp)

    output = pd.concat(df_list, axis=0, join='outer').reset_index(drop=True)

    output['categorical_id'] = output['id'].copy()
    output['hours_from_start'] = output['t']
    output['categorical_day_of_week'] = output['day_of_week'].copy()
    output['categorical_hour'] = output['hour'].copy()

    # Filter to match range used by other academic papers
    output = output[(output['days_from_start'] >= 1096)
                  & (output['days_from_start'] < 1346)].copy()

    output.to_csv(config.data_csv_path)

    print('Done.')
    
class Config():
    def __init__(self, data_folder, csv_path):
        self.data_folder = data_folder #'data1'
        self.data_csv_path = csv_path #'data1/electricity.csv'


if __name__ == '__main__':
    config = Config('data1', 'data1/electricity.csv')
    download_electricity(config)
    electricity = pd.read_csv('data1/electricity.csv', index_col=0)
    data_formatter = ElectricityFormatter()
    train, valid, test = data_formatter.split_data(electricity)
