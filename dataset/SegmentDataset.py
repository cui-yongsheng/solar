from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import os

class SegmentDataset(Dataset):
    def __init__(self, data_dir='../data', normalize=True,
                 segment_length=None, feature_columns=None, custom_raw_features=None):
        """
        初始化分段数据集
        
        参数:
        data_dir: 数据目录路径
        normalize: 是否进行数据归一化
        segment_length: 每段数据的长度，如果为None则根据数据特征自动确定
        feature_columns: 特征列配置，如果为None则使用默认配置
        custom_raw_features: 自定义原始特征列配置
        """

        super().__init__()
        self.data_dir = data_dir
        self.normalize = normalize
        self.segment_length = segment_length
        self.feature_columns = feature_columns
        self.custom_raw_features = custom_raw_features
        
        # 初始化归一化参数为None
        self.feature_min = None
        self.feature_max = None
        
        # 显式声明数据属性，避免IDE无法解析属性引用
        self.labels = None
        self.nn_features = None        # 神经网络部分使用维度
        self.physical_features = None  # 物理模型使用维度
        self.processed_features = None # 处理后的特征数据（神经网络部分使用维度）
        self.num_days = 0
        self.T = segment_length # segment特征长度（神经网络部分）
        self.F = 0              # segment特征维度（神经网络部分）
        self.mask = None        # 每个样本在序列维度上的真实数据掩码（1=真实,0=填充）
        
        # 检查数据目录是否存在
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")

        # 加载所有数据文件
        data_dict = self._load_data_files()

        # 处理特征数据
        self._process_features(data_dict)

        # 按天和段组织数据
        segment_features, segment_current = self._organize_segment_data(data_dict)

        # 标准化序列长度并转换为张量
        self._standardize_and_tensorize(segment_features, segment_current)

        # 计算标准化参数（如果需要）
        if self.normalize:
            self.compute_normalization_params()

        # 预处理所有特征数据
        self.prepare_all_features()

    def _load_data_files(self):
        """加载所有数据文件"""
        # 定义需要的数据文件
        files = {
            'position': 'position_precise.parquet',
            'state': 'state_precise.parquet', 
            'attitude': 'attitude_precise.parquet',
            'distance': 'distance.parquet'
        }
        
        # 检查并读取文件
        data_dict = {}
        for name, filename in files.items():
            path = os.path.join(self.data_dir, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name}数据文件不存在: {path}")
            data_dict[name] = pd.read_parquet(path)
            
        # 重置索引以确保正确对齐
        for name in files.keys():
            data_dict[name] = data_dict[name].reset_index(drop=True)
            
        return data_dict

    def _process_features(self, data_dict):
        """处理特征数据"""
        # 合并所有数据
        combined_data = pd.concat([data_dict['position'], data_dict['state'], 
                                  data_dict['attitude'], data_dict['distance']], axis=1)
                                  
        # 定义默认特征列
        default_feature_columns = {
            'position': ['轨道位置X', '轨道速度X', '太阳矢量计算X', '轨道位置Y', '轨道速度Y', '太阳矢量计算Y', '轨道位置Z', '轨道速度Z', '太阳矢量计算Z'],
            'state': ['帆板温度[+X]', '帆板温度[-X]', '帆板温度[+Y]', '帆板温度[-Y]', '28V母线电压-TTC采集'],
            'attitude': ['滚动角', '俯仰角', '偏航角'],
            'distance': []
        }

        # 定义默认自定义原始特征列配置
        default_custom_raw_features = {
            'position': ['太阳矢量计算X', '太阳矢量计算Y'],
            'state': ['帆板温度[+X]', '帆板温度[-X]', '帆板温度[+Y]', '帆板温度[-Y]', '28V母线电压-TTC采集'],
            'attitude': [],
            'distance': ['distance']
        }

        # 提取自定义原始特征数据（用于物理计算）
        custom_columns_config = self.custom_raw_features or default_custom_raw_features
        all_custom_columns = []
        for _ , columns in custom_columns_config.items():
            all_custom_columns.extend(columns)
        self.physical_features = combined_data[all_custom_columns].values.astype(np.float32)
        # 处理时间特征，将其拆分为年、月、日并进行编码
        _ , time_features_raw = self._process_time_features(combined_data)
        if time_features_raw is not None:
            self.physical_features = np.concatenate([self.physical_features, time_features_raw], axis=1)

        # 提取神经网络特征数据（初步处理）
        columns_config = self.feature_columns or default_feature_columns
        all_feature_columns = []
        for _ , columns in columns_config.items():
            all_feature_columns.extend(columns)
        features = combined_data[all_feature_columns].values.astype(np.float32)
        self.F = features.shape[1]
        # 存储处理后的特征以供后续使用
        self._processed_features = features

    def _organize_segment_data(self, data_dict):
        """按天和段组织数据"""
        # 提取电流数据
        current = data_dict['state']['方阵电流'].values.astype(np.float32)
        # 根据segment列进行分段处理，使用state数据中的segment列
        segments = data_dict['state']['segment'].values
        # 从时间列提取日期信息以确定天数
        time_column = None
        for col in ['time', '时间']:
            if col in data_dict['state'].columns:
                time_column = col
                break
        if time_column is None:
            raise ValueError("找不到时间列")
            
        time_data = pd.to_datetime(data_dict['state'][time_column])
        dates = time_data.dt.date
        unique_dates = np.unique(dates)
        self.num_days = len(unique_dates)

        # 按天和段组织数据
        segment_features = []
        segment_current = []
        
        # 记录实际存在的(day, segment)组合
        self.valid_samples = []

        # 创建一个DataFrame来简化数据处理
        temp_df = pd.DataFrame({
            'date': dates,
            'segment': segments,
            'features': list(self._processed_features),  # 将二维数组转为列表，每个元素是一行特征数据
            'current': list(current)     # 将一维数组转为列表
        })
        
        # 按日期和分段分组
        grouped = temp_df.groupby(['date', 'segment'])
        
        for (date, seg_id), group in grouped:
            date_idx = np.where(unique_dates == date)[0][0]
            
            # 提取特征和电流数据
            feat_seg = np.array(group['features'].tolist())
            curr_seg = np.array(group['current'].tolist())
            
            segment_features.append(feat_seg)
            segment_current.append(curr_seg)
            
            # 记录这是一个有效的(day, segment)组合
            self.valid_samples.append((date_idx, seg_id))
            
        return segment_features, segment_current

    def _standardize_and_tensorize(self, segment_features, segment_current):
        """标准化序列长度并转换为PyTorch张量"""
        # 根据数据特征确定序列长度
        if not segment_features:
            raise ValueError("没有有效的数据段")
            
        # 计算所有序列长度的统计信息
        lengths = [len(seg) for seg in segment_features]
        min_length = min(lengths)
        max_length = max(lengths)
        median_length = int(np.median(lengths))
        
        print(f"序列长度统计: 最小={min_length}, 最大={max_length}, 中位数={median_length}")
        
        # 如果未指定segment_length，则使用中位数长度
        # 如果指定了segment_length，则使用指定值，但确保不超过最大长度且不小于最小长度
        if self.segment_length is None:
            self.T = median_length
        else:
            if self.segment_length <= 0:
                raise ValueError("segment_length必须为正整数")
            self.T = max(min(self.segment_length, max_length), min_length)
        
        print(f"设置序列长度为: {self.T}")

        total_samples = len(segment_features)
        # 预先分配numpy数组以提高性能
        feature_np = np.empty((total_samples, self.T, self.F), dtype=np.float32)
        current_np = np.empty((total_samples, self.T), dtype=np.float32)
        # 掩码：1 表示真实数据位置，0 表示填充位置
        mask_np = np.zeros((total_samples, self.T), dtype=np.float32)
        
        for i, (feat, curr) in enumerate(zip(segment_features, segment_current)):
            seq_len = len(feat)
            if seq_len >= self.T:
                # 如果序列长度大于等于目标长度，则截断
                feature_np[i, :, :] = feat[:self.T, :]
                current_np[i, :] = curr[:self.T]
                mask_np[i, :] = 1.0
            else:
                # 如果序列长度小于目标长度，则进行填充
                # 使用最后一行进行填充（复制最后一行）
                feature_np[i, :seq_len, :] = feat
                feature_np[i, seq_len:, :] = np.tile(feat[-1:], (self.T - seq_len, 1))  # 用最后一行填充剩余部分
                
                current_np[i, :seq_len] = curr
                current_np[i, seq_len:] = np.full(self.T - seq_len, curr[-1])  # 用最后一个值填充剩余部分
                mask_np[i, :seq_len] = 1.0
        
        # 一次性转换为torch tensor以减少开销
        self.nn_features = torch.from_numpy(feature_np)
        self.labels = torch.from_numpy(current_np)
        # 转换 mask
        self.mask = torch.from_numpy(mask_np)
        
        # 处理物理特征数据
        physical_features = np.empty((total_samples, self.T, self.physical_features.shape[1]), dtype=np.float32)
        
        # 跟踪physical_features中的当前索引
        current_physical_idx = 0
        
        for i, feat in enumerate(segment_features):
            seq_len = len(feat)
            if seq_len >= self.T:
                # 如果序列长度大于等于目标长度，则截断
                physical_features[i, :, :] = self.physical_features[current_physical_idx:current_physical_idx+self.T, :]
            else:
                # 如果序列长度小于目标长度，则进行填充
                # 使用最后一行进行填充（复制最后一行）
                physical_features[i, :seq_len, :] = self.physical_features[current_physical_idx:current_physical_idx+seq_len, :]
                physical_features[i, seq_len:, :] = np.tile(self.physical_features[current_physical_idx+seq_len-1:current_physical_idx+seq_len, :], (self.T - seq_len, 1))
            current_physical_idx += seq_len
        
        self.physical_features = torch.from_numpy(physical_features)

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        """
        获取指定索引的数据样本，同时返回归一化后的特征
        
        参数:
        idx: 数据索引
        
        返回:
        (normalized_features, label, day_id, mask) 归一化特征、标签、day_id 和掩码（mask 形状为 [T,1]）
        """
        # 检查索引是否有效
        if idx < 0 or idx >= len(self.valid_samples):
            raise IndexError(f"索引 {idx} 超出范围 [0, {len(self.valid_samples)-1}]")
            
        features = self.processed_features[idx]  # [T, F]
        label = self.labels[idx].unsqueeze(-1)  # [T, 1]
        mask = self.mask[idx].unsqueeze(-1) if self.mask is not None else None  # [T, 1]
        # 直接从valid_samples获取day_id以减少解包开销
        day_id = self.valid_samples[idx][0]  # int, 用于 health embedding

        return features, label, day_id, mask
        
    def get_item_with_raw_features(self, idx):
        """
        获取指定索引的数据样本，同时返回归一化后的特征和原始特征
        
        参数:
        idx: 数据索引
        
        返回:
        (normalized_features, raw_features, label, day_id, mask) 归一化特征、原始特征、标签、day_id 和掩码（mask 形状为 [T,1]）
        """
        # 检查索引是否有效
        if idx < 0 or idx >= len(self.valid_samples):
            raise IndexError(f"索引 {idx} 超出范围 [0, {len(self.valid_samples)-1}]")
            
        normalized_features = self.processed_features[idx]  # [T, F]
        label = self.labels[idx].unsqueeze(-1)  # [T, 1]
        raw_features = self.physical_features[idx] if self.physical_features is not None else None  # [T, F_raw]
        mask = self.mask[idx].unsqueeze(-1) if self.mask is not None else None  # [T, 1]
        # 直接从valid_samples获取day_id以减少解包开销
        day_id = self.valid_samples[idx][0]  # int, 用于 health embedding

        return normalized_features, raw_features, label, day_id, mask
        
    def compute_normalization_params(self, indices=None):
        """
        计算归一化所需的最小值和最大值
        参数:
        indices: 用于计算归一化参数的索引列表，默认为None表示使用全部数据
        """
        if indices is None:
            # 使用所有样本的所有数据计算归一化参数
            if self.nn_features is None:
                raise ValueError("X_all 未初始化")
            # 展平特征和 mask
            features = self.nn_features.reshape(-1, self.F)
            if self.mask is not None:
                mask_flat = self.mask.reshape(-1).bool()
                # 只保留真实数据位置
                if mask_flat.sum().item() == 0:
                    raise ValueError("没有真实数据用于计算归一化参数")
                features = features[mask_flat]

        else:
            # 根据指定索引计算归一化参数
            if self.nn_features is None:
                raise ValueError("X_all 未初始化")
            # 检查 indices 是否为空
            if len(indices) == 0:
                raise ValueError("indices 不能为空")
            selected_features = self.nn_features[indices]
            features = selected_features.reshape(-1, self.F)
            # 如果有 mask，则只使用被选中样本中的真实位置
            if self.mask is not None:
                selected_mask = self.mask[indices].reshape(-1).bool()
                if selected_mask.sum().item() == 0:
                    raise ValueError("在所选索引中没有真实数据用于计算归一化参数")
                features = features[selected_mask]

        # 计算最小值和最大值，使用更高效的torch函数
        self.feature_min = torch.amin(features, dim=0)
        self.feature_max = torch.amax(features, dim=0)
        
        # 避免除零错误
        self.feature_max = torch.where(self.feature_max == self.feature_min, 
                                      self.feature_min + 1.0, 
                                      self.feature_max)

    def prepare_all_features(self):
        """准备所有特征数据"""
        # 如果特征最小值和最大值已计算，则进行min-max归一化（无论normalize设置如何）
        if self.feature_min is not None and self.feature_max is not None:
            self.processed_features = (self.nn_features - self.feature_min) / (self.feature_max - self.feature_min)
            
    def get_feature_dim(self):
        """
        获取总特征维度

        返回:
        int: 总特征维度
        """
        return self.F

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
        self.time_features_raw = time_features_raw
        
        return time_features, time_features_raw
    
# 定义RawFeatureDatasetWrapper类
class RawFeatureDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        # 获取包含原始特征的数据
        normalized_features, raw_features, label, day_id, mask = self.dataset.get_item_with_raw_features(self.indices[idx])
        return normalized_features, raw_features, label, day_id, mask