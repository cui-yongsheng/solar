import pandas as pd
from skyfield.api import load
from astropy.time import Time
import torch
import random
import shutil
import os

def time_alignment(*data_frames, tolerance='1s'):
    """
    基于时间戳精确对齐多个数据集，以第一个数据集为基准

    参数:
    *data_frames: 可变数量的数据DataFrame
    tolerance: 时间容忍度，默认为1秒

    返回:
    tuple: 对齐后的数据DataFrame元组
    """
    if len(data_frames) < 2:
        raise ValueError("至少需要两个数据集进行对齐")
    
    # 复制所有数据帧避免修改原始数据
    copied_frames = [df.copy() for df in data_frames]
    
    # 确保所有数据帧的时间列都是datetime类型
    for df in copied_frames:
        df['时间'] = pd.to_datetime(df['时间'])
    
    # 以第一个数据集为基准
    base_times = copied_frames[0][['时间']].copy()
    
    # 对齐其余所有数据集
    aligned_frames = [copied_frames[0]]  # 第一个数据集不需要对齐
    
    for i in range(1, len(copied_frames)):
        aligned_frame = pd.merge_asof(
            base_times,
            copied_frames[i],
            on='时间',
            direction='nearest',
            tolerance=pd.Timedelta(tolerance)
        )
        aligned_frames.append(aligned_frame)
    
    return tuple(aligned_frames)


def sun_earth_distance(times):
    """
    向量化计算多个时间的日地距离

    参数:
    times: 时间列表，可以是datetime对象列表、时间字符串列表或skyfield.Time对象

    返回:
    包含所有时间点日地距离的数组
    """
    # 加载星历数据
    eph = load('de421.bsp')
    sun = eph['sun']
    earth = eph['earth']
    ts = load.timescale()
    astropy_times = Time(times, scale='utc')
    skyfield_times = ts.from_astropy(astropy_times)
    # 向量化计算地球相对于太阳的位置
    astrometric = earth.at(skyfield_times).observe(sun)
    # 获取距离数组（天文单位AU）
    distances_au = astrometric.distance().au
    # 转换为千米
    distances_km = distances_au * 149597870.7
    return distances_km

def set_random_seed(seed):
    """
    设置全局随机种子以确保实验可复现性

    Args:
        seed (int): 随机种子值
    """
    # 设置Python内置random模块的种子
    random.seed(seed)

    # 设置PyTorch的随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # 设置cuDNN相关参数确保确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_code(save_path):
    """
    递归复制项目中所有.py文件到保存路径，保持目录结构，排除__init__.py文件和results目录
    Args:
        save_path (str): 目标保存路径
    """
    # 使用shutil.copytree配合ignore参数实现递归复制
    save_path = os.path.join(save_path, "src")
    current_dir = os.getcwd()
    
    # 先处理根目录下的.py文件
    for file in os.listdir(current_dir):
        if file.endswith('.py'):
            src_path = os.path.join(current_dir, file)
            dst_path = os.path.join(save_path, file)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)
    
    # 遍历当前目录及子目录，复制所有.py文件
    for root, dirs, files in os.walk(current_dir):
        # 排除__pycache__和results目录
        dirs[:] = [d for d in dirs if d in ('dataset', 'examples', 'model', 'utils')]
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                # 构造源文件路径
                src_path = os.path.join(root, file)
                # 计算相对于当前工作目录的路径
                rel_path = os.path.relpath(src_path, current_dir)
                # 构造目标文件路径
                dst_path = os.path.join(save_path, rel_path)
                # 创建目标目录
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                # 复制文件
                shutil.copy2(src_path, dst_path)

def get_directory_by_model_and_dataset(csv_file_path, model, dataset_name):
    """
    根据model和dataset_name获取对应的directory值

    参数:
    csv_file_path (str): CSV文件路径
    model (str): 模型名称
    dataset_name (str): 数据集名称

    返回:
    str or None: 对应的directory值，如果未找到则返回None
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)

    # 根据model和dataset_name筛选行
    filtered_df = df[(df['model'] == model) & (df['dataset_name'] == dataset_name)]

    # 如果找到了匹配的行，返回directory值
    if not filtered_df.empty:
        return filtered_df.iloc[0]['directory']
    else:
        return None
