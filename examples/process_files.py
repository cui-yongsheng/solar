import numpy as np
import pandas as pd
from utils.utils import time_alignment

def pre_process_by_segment(data):
    """
    按segment分别对数据进行预处理

    参数:
    data: 包含'segment'列的DataFrame

    返回:
    processed_data: 处理后的数据
    """
    processed_segments = []

    # 按segment分组处理
    for segment_id, segment_data in data.groupby('segment', group_keys=False):
        # 深拷贝避免修改原始数据
        seg_copy = segment_data.copy()
        # 处理无穷值并删除NaN
        seg_copy.replace([np.inf, -np.inf], np.nan, inplace=True)
        seg_clean = seg_copy.dropna().reset_index(drop=True)
        # 异常值检测（排除时间列）
        features = seg_clean.select_dtypes(include=[np.number]).columns.difference(['时间'])
        if not features.empty:
            # 向量化计算Z-score
            z_scores = (seg_clean[features] - seg_clean[features].mean()) / seg_clean[features].std()
            # 标记超过3σ的异常点
            outlier_mask = (np.abs(z_scores) > 3).any(axis=1)
            seg_final = seg_clean[~outlier_mask].reset_index(drop=True)
        else:
            seg_final = seg_clean
        # 添加segment标识
        seg_final['segment'] = segment_id
        processed_segments.append(seg_final)

    # 合并所有处理后的segment
    processed_data = pd.concat(processed_segments, axis=0, ignore_index=True) if processed_segments else pd.DataFrame()
    return processed_data


def remove_abnormal_segments(data, method='zscore', threshold=3.0):
    """
    基于整体统计特征检测并删除异常segment

    参数:
    data: 包含'segment'列的DataFrame
    method: 异常检测方法 ('zscore' 或 'iqr')
    threshold: 异常检测阈值

    返回:
    cleaned_data: 删除异常segment后的数据
    """
    # 检查数据是否为空
    if data.empty or 'segment' not in data.columns:
        return data

    # 按segment分组计算统计特征
    segment_stats = data.groupby('segment').agg({
        col: ['mean', 'std', 'median', 'min', 'max']
        for col in data.select_dtypes(include=np.number).columns
        if col != 'segment'  # 排除segment列本身
    })

    # 展平多级列索引
    segment_stats.columns = ['_'.join(col).strip() for col in segment_stats.columns.values]

    # 异常检测方法选择
    if method == 'zscore':
        # 计算每个统计特征的Z-score
        z_scores = (segment_stats - segment_stats.mean()) / segment_stats.std()
        # 标记异常segment (任何特征的Z-score超过阈值)
        abnormal_segments = z_scores[(np.abs(z_scores) > threshold).any(axis=1)].index

    elif method == 'iqr':
        # 使用IQR方法检测异常
        Q1 = segment_stats.quantile(0.25)
        Q3 = segment_stats.quantile(0.75)
        IQR = Q3 - Q1

        # 标记异常segment (任何特征超出IQR范围)
        outlier_mask = ((segment_stats < (Q1 - threshold * IQR)) |
                        (segment_stats > (Q3 + threshold * IQR))).any(axis=1)
        abnormal_segments = segment_stats[outlier_mask].index

    else:
        raise ValueError(f"不支持的异常检测方法: {method}. 请选择 'zscore' 或 'iqr'")

    # 删除异常segment
    cleaned_data = data[~data['segment'].isin(abnormal_segments)]
    return cleaned_data.reset_index(drop=True)

if __name__ == '__main__':
    # 读取数据
    position_data = pd.read_parquet('../data/position_data.parquet')
    state_data = pd.read_parquet('../data/state_data.parquet')
    attitude_data = pd.read_parquet('../data/attitude_data.parquet')

    # 应用段落内异常处理
    position_data = pre_process_by_segment(position_data)
    attitude_data = pre_process_by_segment(attitude_data)
    state_data = pre_process_by_segment(state_data)

    # 应用段落间异常处理
    position_data = remove_abnormal_segments(position_data, method='zscore', threshold=3.0)
    attitude_data = remove_abnormal_segments(attitude_data, method='zscore', threshold=3.0)
    state_data = remove_abnormal_segments(state_data, method='zscore', threshold=3.0)

    # 基于时间戳精确对齐数据集
    position_precise, attitude_precise, state_precise = time_alignment(position_data, attitude_data, state_data)

    # 删除所有NAN值 - 移除任何包含NaN值的时间点
    combined_data = pd.concat([position_precise, attitude_precise, state_precise], axis=1)
    combined_data = combined_data.loc[:, ~combined_data.columns.duplicated()]
    cleaned_data = combined_data.dropna()       # 找出包含NaN值的行并删除

    # 分离回原来的数据结构
    position_precise = cleaned_data[position_precise.columns]
    attitude_precise = cleaned_data[attitude_precise.columns]
    state_precise = cleaned_data[state_precise.columns]

    ## 重新设置segment
    # 计算时间差分并识别断点
    time_diff = position_precise['时间'].diff()
    # 使用numpy的timedelta64进行比较，避免类型不匹配问题
    break_points = time_diff > pd.Timedelta(minutes=10)
    # 处理可能的NaN值
    break_points = break_points.fillna(False)
    # 添加段落编号
    position_precise.loc[:, 'segment'] = break_points.cumsum()
    attitude_precise.loc[:, 'segment'] = break_points.cumsum()
    state_precise.loc[:, 'segment'] = break_points.cumsum()

    # 保存为parquet文件
    position_precise.to_parquet('../data/position_precise.parquet', index=False)
    attitude_precise.to_parquet('../data/attitude_precise.parquet', index=False)
    state_precise.to_parquet('../data/state_precise.parquet', index=False)
