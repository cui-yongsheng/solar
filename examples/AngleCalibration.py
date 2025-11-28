import pandas as pd
from utils.angle_calculator import calculate_solar_power

if __name__ == '__main__':
    # 读取数据
    position_precise = pd.read_parquet('../data/position_precise.parquet')
    attitude_precise = pd.read_parquet('../data/attitude_precise.parquet')
    # 计算姿态调整后因子
    columns_needed = ["姿态四元数Q1", "姿态四元数Q2", "姿态四元数Q3", "姿态四元数Q4", ]  # 请替换为实际需要的列名
    position_precise = position_precise.merge(
        attitude_precise[columns_needed],
        left_index=True,
        right_index=True,
        how='left'
    )
    position_precise = calculate_solar_power(position_precise)
    position_precise.to_parquet('../data/power_precise.parquet', index=False)