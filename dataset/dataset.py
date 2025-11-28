# 基于torch框架定义数据集文件夹
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

class SatelliteDataset(Dataset):
    def __init__(self, data_dir='../data', normalize=True,
                 feature_columns=None, custom_raw_features=None):
        """
        初始化卫星数据集

        参数:
        data_dir: 数据目录路径
        normalize: 是否进行数据归一化
        feature_columns: 特征列配置，如果为None则使用默认配置
        custom_raw_features: 自定义原始特征列配置
        """
        self.data_dir = data_dir
        self.normalize = normalize
        self.feature_columns = feature_columns
        self.custom_raw_features = custom_raw_features
        
        # 初始化归一化参数为None
        self.feature_min = None
        self.feature_max = None
        
        # 显式声明数据属性，避免IDE无法解析属性引用
        self.position_data = None
        self.state_data = None
        self.attitude_data = None
        self.distance_data = None
        self.nn_features = None        # 神经网络部分使用维度
        self.physical_features = None  # 物理模型使用维度
        self.processed_features = None # 处理后的特征数据（神经网络部分使用维度）
        self.labels = None
        self.total_dim = 0

        # 检查数据目录是否存在
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")

        # 加载已对齐的数据
        self._load_data()
        
        # 预处理特征数据
        self._preprocess_features()
        
        # 验证数据完整性
        if self.nn_features is None or self.physical_features is None or self.labels is None:
            raise ValueError("数据预处理失败，特征或标签为None")
        
        # 计算标准化参数
        if self.normalize:
            self.compute_normalization_params()
            
        # 预处理所有特征数据
        self.prepare_all_features()

    def _load_data(self):
        """加载所有数据文件"""
        files = {
            'position': 'position_precise.parquet',
            'state': 'state_precise.parquet', 
            'attitude': 'attitude_precise.parquet',
            'distance': 'distance.parquet'
        }
        
        # 检查并读取文件
        loaded_data = {}
        for name, filename in files.items():
            path = os.path.join(self.data_dir, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name}数据文件不存在: {path}")
                
            loaded_data[name] = pd.read_parquet(path)
            
        # 确保所有数据集具有相同的时间戳
        lengths = [len(data) for data in loaded_data.values()]
        if len(set(lengths)) > 1:
            raise ValueError("数据集长度不一致，请检查数据对齐")
            
        # 设置数据属性并重置索引以确保正确对齐
        for name, data in loaded_data.items():
            setattr(self, f"{name}_data", data.reset_index(drop=True))

    def _preprocess_features(self):
        """预处理特征数据"""
        # 定义默认特征列配置
        default_feature_columns = {
            'position': ['轨道位置X', '轨道速度X', '太阳矢量计算X', '轨道位置Y', '轨道速度Y', '太阳矢量计算Y', '轨道位置Z', '轨道速度Z', '太阳矢量计算Z'],
            'state': ['帆板温度[+X]', '帆板温度[-X]', '帆板温度[+Y]', '帆板温度[-Y]', '28V母线电压-TTC采集'],
            'attitude': ['滚动角', '俯仰角', '偏航角'],
            'distance': []
        }
        
        # 使用自定义配置或默认配置
        columns_config = self.feature_columns or default_feature_columns
        
        # 定义默认自定义特征列配置
        default_custom_raw_features = {
            'position': ['太阳矢量计算X', '太阳矢量计算Y'],
            'state': ['帆板温度[+X]', '帆板温度[-X]', '帆板温度[+Y]', '帆板温度[-Y]', '28V母线电压-TTC采集'],
            'attitude': [],
            'distance': ['distance']
        }
        
        # 使用自定义配置或默认配置
        custom_columns_config = self.custom_raw_features or default_custom_raw_features
        
        # 合并所有数据为一个DataFrame
        data_frames = [self.position_data, self.state_data, self.attitude_data, self.distance_data]
        combined_data = pd.concat(data_frames, axis=1)
        
        # 提取神经网络特征数据
        all_feature_columns = []
        for name, columns in columns_config.items():
            all_feature_columns.extend(columns)
            
        self.nn_features = combined_data[all_feature_columns].values.astype(np.float32)
        
        # 处理时间特征，将其拆分为年、月、日并进行编码
        time_features, time_features_raw = self._process_time_features(combined_data)
        
        # 将时间特征与原始特征合并
        if time_features is not None:
            self.nn_features = np.concatenate([self.nn_features, time_features], axis=1)
            
        self.total_dim = self.nn_features.shape[1]

        # 提取物理模型使用的原始特征数据
        all_custom_columns = []
        for name, columns in custom_columns_config.items():
            all_custom_columns.extend(columns)

        self.physical_features = combined_data[all_custom_columns].values.astype(np.float32)

        # 添加原始时间特征
        if time_features_raw is not None:
            self.physical_features = np.concatenate([self.physical_features, time_features_raw], axis=1)

        # 提取标签
        if '方阵电流' not in self.state_data.columns:
            raise ValueError("标签列 '方阵电流' 不存在于状态数据中")
        self.labels = self.state_data['方阵电流'].values.astype(np.float32)

    def _process_time_features(self, combined_data):
        """
        处理时间特征，将其拆分为年、月、日并进行编码
        
        参数:
        combined_data: 合并后的数据DataFrame
        
        返回:
        处理后的时间特征数组
        """
        # 检查时间列
        time_column = None
        for col in ['time', '时间']:
            if col in combined_data.columns:
                time_column = col
                break
            
        if time_column is None:
            return None
            
        # 将时间列转换为datetime类型
        time_data = pd.to_datetime(combined_data[time_column])
        
        # 提取年、月、日
        years = time_data.dt.year.values
        months = time_data.dt.month.values
        days = time_data.dt.day.values
        
        # 对年、月、日进行编码
        # 年份标准化（减去基准年份2014）
        year_features = (years - 2014).astype(np.float32)
        
        # 月份进行周期编码（sin和cos）
        month_sin = np.sin(2 * np.pi * months / 12).astype(np.float32)
        month_cos = np.cos(2 * np.pi * months / 12).astype(np.float32)
        
        # 日期进行周期编码（sin和cos）
        day_sin = np.sin(2 * np.pi * days / 31).astype(np.float32)
        day_cos = np.cos(2 * np.pi * days / 31).astype(np.float32)
        
        # 合并所有时间特征
        time_features = np.column_stack([year_features, month_sin, month_cos, day_sin, day_cos])

        # 保存原始时间特征以供后续使用
        time_features_raw = np.column_stack([years, months, days]).astype(np.float32)

        return time_features,time_features_raw

    def compute_normalization_params(self, indices=None):
        """
        计算归一化所需的最小值和最大值
        参数:
        indices: 用于计算归一化参数的索引列表，默认为None表示使用全部数据
        """
        if self.nn_features is None:
            raise ValueError("特征数据未初始化")
            
        if indices is None:
            features = self.nn_features
        else:
            features = self.nn_features[indices]
            
        if len(features) == 0:
            raise ValueError("用于计算归一化参数的特征数据为空")

        # 计算最小值和最大值
        self.feature_min = np.min(features, axis=0).astype(np.float32)
        self.feature_max = np.max(features, axis=0).astype(np.float32)
        
        # 避免除零错误
        zero_range = self.feature_max == self.feature_min
        self.feature_max[zero_range] = self.feature_min[zero_range] + 1.0

    def prepare_all_features(self):
        """准备所有特征数据"""
        if self.nn_features is None:
            raise ValueError("神经网络特征未初始化")
            
        # 使用已提取的特征
        self.processed_features = self.nn_features
        
        # 如果需要标准化，则进行min-max归一化
        if self.feature_min is not None and self.feature_max is not None:
            self.processed_features = (self.processed_features - self.feature_min) / (self.feature_max - self.feature_min)

    def __len__(self):
        """返回数据集长度"""
        if self.position_data is None:
            raise ValueError("数据未加载")
        return len(self.position_data)

    def __getitem__(self, idx):
        """
        获取指定索引的数据样本

        参数:
        idx: 数据索引

        返回:
        (features, label) 特征向量和标签（方阵电流）
        """
        if self.processed_features is None or self.labels is None:
            raise ValueError("数据未正确初始化")
            
        # 检查索引是否有效
        if idx >= len(self):
            raise IndexError(f"索引 {idx} 超出范围，数据集长度为 {len(self)}")
            
        # 直接从预处理的数据中获取特征
        features = self.processed_features[idx]

        # 获取标签
        label = self.labels[idx]

        return features, label

    def get_item_with_raw_features(self, idx):
        """
        获取指定索引的数据样本，同时返回归一化后的特征和原始特征

        参数:
        idx: 数据索引

        返回:
        (normalized_features, raw_features, label) 归一化特征、原始特征和标签
        """
        if self.processed_features is None or self.physical_features is None or self.labels is None:
            raise ValueError("数据未正确初始化")
            
        # 检查索引是否有效
        if idx >= len(self):
            raise IndexError(f"索引 {idx} 超出范围，数据集长度为 {len(self)}")
            
        # 获取归一化后的特征
        normalized_features = self.processed_features[idx]
        
        # 获取原始特征
        raw_features = self.physical_features[idx]

        # 获取标签
        label = self.labels[idx]

        return normalized_features, raw_features, label
        
    def get_feature_dim(self):
        """
        获取总特征维度

        返回:
        int: 总特征维度
        """
        return self.total_dim