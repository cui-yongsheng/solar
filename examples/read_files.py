from utils.file_processor import FileProcessor
from utils.angle_calculator import calculate_angles,calculate_solar_power
from utils.utils import time_alignment
import pandas as pd
# 定义数据配置
DATA_CONFIGS = {
    'state': {
        'columns': ["方阵电流", "帆板温度[+X]", "帆板温度[-X]", "帆板温度[+Y]", "帆板温度[-Y]", "28V母线电压-TTC采集", "时间"],
        'target_file': "卫星健康数据.csv",
        'file_pattern': None,
        'output_path': "./state_data.parquet"
    },
    'position': {
        'columns': ["轨道位置X", "轨道位置Y", "轨道位置Z", "轨道速度X", "轨道速度Y", "轨道速度Z",
                   "太阳矢量计算X", "太阳矢量计算Y", "太阳矢量计算Z", "时间"],
        'target_file': None,
        'file_pattern': "{date}ADCS丙1.csv",
        'output_path': "./position_data.parquet"
    },
    'attitude': {
        'columns': ["滚动角", "俯仰角", "偏航角", "姿态四元数Q1", "姿态四元数Q2", "姿态四元数Q3", "姿态四元数Q4", "时间"],
        'target_file': None,
        'file_pattern': "{date}ADCS丙2.csv",
        'output_path': "./attitude_data.parquet"
    },
    'environment': {
        'columns': ["时间", "轨道位置X", "轨道位置Y", "轨道位置Z"],
        'target_file': None,
        'file_pattern': "{date}ADCS丙1.csv",
        'output_path': "./environment_data.parquet"
    }
}

def process_satellite_data(data_type, start_date, end_date, root_path="G:/实验室基础/灵巧卫星/网盘数据全部/SepnetLog/SepnetLog"):
    """
    通用卫星数据处理函数
    Args:
        data_type (str): 数据类型 ('state', 'position', 'attitude')
        start_date (str): 开始日期
        end_date (str): 结束日期
        root_path (str): 数据根路径
    """
    if data_type not in DATA_CONFIGS:
        raise ValueError(f"不支持的数据类型: {data_type}")

    config = DATA_CONFIGS[data_type]
    file_processor = FileProcessor()

    # 查找文件
    found_files = file_processor.find_files_by_date_range(
        root_path,
        start_date,
        end_date,
        target_file=config['target_file'],
        file_pattern=config['file_pattern']
    )

    if not found_files:
        print(f"未找到 {data_type} 类型的数据文件")
        return None

    # 读取和处理数据
    df = file_processor.read_files(found_files, config['columns'])

    if df.empty:
        print(f"{data_type} 数据为空")
        return None

    # 验证时间序列
    df = FileProcessor.validate_and_clean_time_series(df)
    
    # 计算时间差分并识别断点
    time_diff = df['时间'].diff()
    # 使用numpy的timedelta64进行比较，避免类型不匹配问题
    break_points = time_diff > pd.Timedelta(minutes=10)
    # 处理可能的NaN值
    break_points = break_points.fillna(False)
    df['segment'] = break_points.cumsum()
    # 保存数据
    output_path = config['output_path']
    success = file_processor.save_dataframe(df, output_path, "parquet")

    if success:
        print(f"{data_type} 数据处理完成，保存至 {config['output_path']}")

    return df

def state_data(start_date, end_date):
    """处理状态数据"""
    return process_satellite_data('state', start_date, end_date)

def position_data(start_date, end_date):
    """处理角度数据"""
    return process_satellite_data('position', start_date, end_date)

def attitude_data(start_date, end_date):
    """处理姿态数据"""
    return process_satellite_data('attitude', start_date, end_date)

def environment_data(start_date, end_date):
    """处理环境数据"""
    return process_satellite_data('environment', start_date, end_date)

if __name__ == '__main__':
    start_date = "20140916"
    end_date = "20211209"
    ## 1.处理状态数据
    state_data(start_date, end_date)
    ## 2.处理位置数据
    df = position_data(start_date, end_date)
    df = calculate_angles(df)
    df.to_parquet("./position_data.parquet")
    ## 3.处理姿态数据
    attitude_data(start_date, end_date)
    ## 4.处理环境数据
    environment_data(start_date, end_date)

