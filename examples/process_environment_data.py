import numpy as np
import pandas as pd
from utils.utils import time_alignment

if __name__ == '__main__':
    # 读取数据
    position_data = pd.read_parquet('../data/position_precise.parquet')
    environment_data = pd.read_parquet('../data/environment_data.parquet')

    # 基于时间戳精确对齐数据集
    position_precise, environment_precise = time_alignment(position_data, environment_data)

    # 删除所有NAN值 - 移除任何包含NaN值的时间点
    combined_data = pd.concat([position_precise, environment_precise], axis=1)
    combined_data = combined_data.loc[:, ~combined_data.columns.duplicated()]
    cleaned_data = combined_data.dropna()       # 找出包含NaN值的行并删除

    # 分离回原来的数据结构
    position_precise = cleaned_data[position_precise.columns]
    environment_precise = cleaned_data[environment_precise.columns]

    ## 重新设置segment
    # 计算时间差分并识别断点
    time_diff = position_precise['时间'].diff()
    # 使用numpy的timedelta64进行比较，避免类型不匹配问题
    break_points = time_diff > pd.Timedelta(minutes=10)
    # 处理可能的NaN值
    break_points = break_points.fillna(False)
    # 添加段落编号
    position_precise.loc[:, 'segment'] = break_points.cumsum()
    environment_precise.loc[:, 'segment'] = break_points.cumsum()

    # 保存为parquet文件
    environment_precise.to_parquet('../data/environment_precise.parquet', index=False)

    # 对时间列进行处理，修改为Modified Julian Date格式
    # 最终保存为txt格式
    time_column = environment_precise['时间']

    # 将时间列转换为Modified Julian Date格式
    # MJD = JD - 2400000.5
    # JD = 367*year - floor(7*(year+floor((month+9)/12))/4) + floor(275*month/9) + day + 1721013.5 + UT/24.0
    formatted_time = (time_column - pd.Timestamp('1858-11-17')).dt.total_seconds() / (24*60*60)

    # 替换 environment_precise 中的时间列为新格式
    environment_precise = environment_precise.copy()
    environment_precise['时间'] = formatted_time

    # 删除 segment 列
    environment_precise = environment_precise.drop(columns=['segment'])

    # 使用逗号分割并保存为 txt 格式文件
    environment_precise.to_csv('../data/environment_precise.txt', sep=',', index=False, header=False)

