import torch
import numpy as np
import pandas as pd
import json
from model.physics_model import PhysicsModel
from dataset.SegmentDataset import SegmentDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, Subset
from utils.args import get_args
from utils.utils import set_random_seed, save_code
import random

def create_data_loaders(args):
    """创建数据加载器，只返回测试集"""
    print("数据集加载...")

    dataset = SegmentDataset(data_dir='./data', normalize=False)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    # 按时间顺序划分数据集，只获取测试集
    test_indices = list(range(train_size + val_size, len(dataset)))
    test_dataset = Subset(dataset, test_indices)
    
    # 现在仅基于训练集计算归一化参数，避免数据泄露
    train_indices = list(range(0, train_size))
    dataset.compute_normalization_params(indices=train_indices)
    
    # 对所有数据集应用相同地归一化参数
    dataset.prepare_all_features()  # 重新处理特征并应用归一化
    
    # 为测试创建支持原始特征的数据集包装器
    class RawFeatureDatasetWrapper(torch.utils.data.Dataset):
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            # 获取包含原始特征的数据
            normalized_features, raw_features, label, day_id = self.dataset.get_item_with_raw_features(self.indices[idx])
            return normalized_features, raw_features, label, day_id
    
    # 创建包装后的测试数据集
    test_dataset_with_raw = RawFeatureDatasetWrapper(dataset, test_dataset.indices)
    
    # 创建测试数据加载器
    test_loader = DataLoader(test_dataset_with_raw, batch_size=args.test_batch_size, shuffle=False)
    
    return test_loader, dataset

def test_physics_model(save_path, args):
    """
    测试纯物理模型的效果（只在测试集上）
    """
    # 如果没有提供保存路径，则使用默认路径
    if save_path is None:
        save_path = './results'
    
    # 确保保存目录存在
    os.makedirs(save_path, exist_ok=True)
    
    # 创建数据加载器
    test_loader, dataset = create_data_loaders(args)
    
    # 初始化物理模型
    physics_model = PhysicsModel()
    
    # 获取所有数据的物理特征和真实标签
    all_predictions = []
    all_labels = []
    all_day_ids = []  # 保存segment信息
    
    print("开始测试物理模型...")
    
    # 批量处理数据
    for i, (normalized_features, raw_features, labels, day_ids) in enumerate(test_loader):
        # 转换为torch tensor
        raw_features_tensor = raw_features  # 已经是torch tensor
        
        # 使用物理模型进行预测
        with torch.no_grad():
            predictions = physics_model.compute_current(raw_features_tensor)
            
        # 存储预测结果和真实标签
        # 确保所有张量都在CPU上并转换为numpy数组
        predictions_np = predictions.cpu().numpy()
        labels_np = labels.cpu().numpy()
        day_ids_np = day_ids.cpu().numpy()
        
        # 展平所有数组
        predictions_flat = predictions_np.flatten()
        labels_flat = labels_np.flatten()
        day_ids_flat = np.repeat(day_ids_np, predictions_np.shape[1])  # 每个day_id重复序列长度次
        
        all_predictions.extend(predictions_flat)
        all_labels.extend(labels_flat)
        all_day_ids.extend(day_ids_flat)
        
        # 显示进度
        if (i + 1) % 10 == 0:
            print(f"已处理 {i + 1} 个批次，共 {len(test_loader)} 个批次")
    
    # 转换为numpy数组
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    day_ids = np.array(all_day_ids)  # segment信息
    
    print(f"预测值数量: {len(predictions)}, 标签数量: {len(labels)}, day_ids数量: {len(day_ids)}")
    
    # 确保长度一致
    min_length = min(len(predictions), len(labels), len(day_ids))
    predictions = predictions[:min_length]
    labels = labels[:min_length]
    day_ids = day_ids[:min_length]
    
    print(f"截断后长度: 预测值={len(predictions)}, 标签={len(labels)}, day_ids={len(day_ids)}")
    
    # 检查数组维度
    print(f"预测值维度: {predictions.shape}, 标签维度: {labels.shape}")
    
    # 过滤掉无效值（NaN或无穷大）
    valid_mask = np.isfinite(predictions) & np.isfinite(labels)
    predictions = predictions[valid_mask]
    labels = labels[valid_mask]
    day_ids = day_ids[valid_mask]
    
    print(f"过滤无效值后长度: 预测值={len(predictions)}, 标签={len(labels)}, day_ids={len(day_ids)}")
    
    # 如果没有有效数据，直接返回
    if len(predictions) == 0:
        print("没有有效的预测数据")
        return None
    
    # 计算评估指标
    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    
    # 打印评估结果（参考BaseTrainer中的_print_test_results方法）
    print('\n' + '=' * 50)
    print('模型测试结果')
    print('=' * 50)
    print(f'测试集平均损失:     {mse:.6f}')
    print(f'均方误差 (MSE):     {mse:.6f}')
    print(f'均方根误差 (RMSE):  {rmse:.6f}')
    print(f'平均绝对误差 (MAE): {mae:.6f}')
    print(f'决定系数 (R²):      {r2:.6f}')
    print('=' * 50)
    
    # 保存预测结果，包含segment信息（参考BaseTrainer中的_save_test_results方法）
    # 保存指标到JSON文件
    test_metrics = {
        'test_loss': float(mse),  # 与模型训练保持一致的命名
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
    }
    
    with open(os.path.join(save_path, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=4)
    print(f"测试指标已保存至 {os.path.join(save_path, 'test_metrics.json')}")
    
    # 保存预测值和实际值到CSV文件 (参考BaseTrainer中的_save_test_results方法)
    csv_file = os.path.join(save_path, 'predictions_vs_actuals.csv')
    # 创建一个空的object类型数组来容纳混合数据类型
    data = np.empty((len(labels), 3), dtype=object)
    data[:, 0] = labels
    data[:, 1] = predictions
    data[:, 2] = day_ids
    np.savetxt(csv_file, data,
              delimiter=',', fmt=['%.6f', '%.6f', '%s'],
              header='actual,predicted,segment', comments='')
    print(f"\n预测结果已保存至 {os.path.join(save_path, 'predictions_vs_actuals.csv')}")

    # 绘制预测值与真实值的对比图（参考BaseTrainer中的_plot_predictions方法）
    if args.show_plot:
        plt.figure(figsize=(10, 6))
        plt.scatter(labels, predictions, alpha=0.5, s=1)
        plt.plot([labels.min(), labels.max()], [labels.min(), labels.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Physics Model: Predicted vs Actual Values')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'physics_model_scatter.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    # 绘制时间序列图（前1000个点）
    if args.show_plot:
        plt.figure(figsize=(12, 6))
        sample_count = min(1000, len(labels))
        plt.plot(labels[:sample_count], label='Actual Values', linewidth=1)
        plt.plot(predictions[:sample_count], label='Predicted Values', linewidth=1)
        plt.xlabel('Time Step')
        plt.ylabel('Current (A)')
        plt.title('Physics Model Prediction Time Series Comparison (First 1000 Points)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'physics_model_timeseries.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    # 绘制误差分布图（参考BaseTrainer中的_plot_error_distribution方法）
    if args.show_plot:
        errors = predictions - labels
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Physics Model Prediction Error Distribution')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'physics_model_error_distribution.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': predictions,
        'labels': labels
    }

def main():
    # 获取命令行参数
    args = get_args()
    
    # 检查模型类型是否为物理模型
    if args.model_type != 'physics_model':
        print(f"警告: 当前脚本是为物理模型设计的，但模型类型设置为 {args.model_type}")
        return
    
    # 设置随机种子
    set_random_seed(args.seed)
    
    # 保存源代码
    save_code(args.save_path)
    
    # 运行测试
    results = test_physics_model(args.save_path, args)

if __name__ == "__main__":
    main()