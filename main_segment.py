from sklearn.model_selection import TimeSeriesSplit
from dataset.SegmentDataset import SegmentDataset, RawFeatureDatasetWrapper
from model.ml_models import MLRandomForest, MLSVM, MLLinearRegression, convert_torch_dataset_to_numpy, evaluate_ml_model, save_ml_test_results
from model.EnhancedTrainer import EnhancedSegmentPINNTrainer,EnhancedSegmentTrainer, PhysicsModelTrainer
from model.segment_nn import SolarHealthModel, SimpleSolarModel, GRUSolarModel, CNNLSTMSolarModel, TransformerSolarModel
from model.segment_pinn import PhysicsInformedSolarHealthModel, PhysicsInformedSolarHealthModelNoDegradation
from model.segment_pinn import NeuralODEHealthModel, TransformerNeuralODE
from model.segment_physics import PhysicsOnlyModel
from utils.utils import set_random_seed, save_code
from torch.utils.data import DataLoader, Subset
from utils.args import get_args
import numpy as np
import random
import torch
import os

def create_data_loaders(args):
    """创建数据加载器"""
    print("数据集加载...")

    dataset = SegmentDataset(data_dir='./data', normalize=False)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    if args.few_shot:
        # Few-shot learning: randomly sample a small subset for training
        print("使用小样本训练模式")
        few_shot_size = max(int(0.2 * train_size), 50)  # Use 20% of training data or 50 samples, whichever is smaller
        print(f"小样本数量: {few_shot_size}")
        
        # Randomly select few-shot samples from the training set
        full_train_indices = list(range(0, train_size)) if not args.shuffle_data else list(range(len(dataset)))[:train_size]
        few_shot_indices = random.sample(full_train_indices, few_shot_size)
        
        if args.shuffle_data:
            # 随机打乱数据集
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            
            # Split the shuffled indices
            train_indices = few_shot_indices  # Use the few-shot sampled indices
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]
        else:
            # 按时间顺序划分数据集
            train_indices = few_shot_indices  # Use the few-shot sampled indices
            val_indices = list(range(train_size, train_size + val_size))
            test_indices = list(range(train_size + val_size, len(dataset)))
    else:
        if args.shuffle_data:
            # 随机打乱数据集
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]
        else:
            # 按时间顺序划分数据集
            train_indices = list(range(0, train_size))
            val_indices = list(range(train_size, train_size + val_size))
            test_indices = list(range(train_size + val_size, len(dataset)))
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # 现在仅基于训练集计算归一化参数，避免数据泄露
    dataset.compute_normalization_params(indices=train_indices)
    
    # 对所有数据集应用相同地归一化参数
    dataset.prepare_all_features()  # 重新处理特征并应用归一化
    
    # 为需要原始特征的模型创建支持原始特征的数据集包装器
    if args.model_type in ['segment_pinn', 'neural_ode', 'transformer_neural_ode', 'cnn_gru_neural_ode', 'segment_pinn_nodegradation', 'physics_model']:
        # 为训练器创建支持原始特征的数据集包装器

        # 创建包装后的数据集
        train_dataset_with_raw = RawFeatureDatasetWrapper(dataset, train_dataset.indices)
        val_dataset_with_raw = RawFeatureDatasetWrapper(dataset, val_dataset.indices)
        test_dataset_with_raw = RawFeatureDatasetWrapper(dataset, test_dataset.indices)

        # 创建数据加载器
        train_loader = DataLoader(train_dataset_with_raw, batch_size=args.train_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset_with_raw, batch_size=args.val_batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset_with_raw, batch_size=args.test_batch_size, shuffle=False)
    elif args.model_type in ['segment', 'simple_segment', 'gru_segment', 'cnn_lstm_segment', 'transformer_segment']:
        # 对于其他模型，直接使用普通的DataLoader
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
    elif args.model_type in ['ml_rf', 'ml_svm', 'ml_lr']:
        # 对于机器学习模型，不需要DataLoader
        train_loader = train_dataset
        val_loader = val_dataset
        test_loader = test_dataset
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")
    return train_loader, val_loader, test_loader, dataset


def create_model(args, dataset):
    """创建模型"""
    if args.model_type == 'segment_pinn':
        model = PhysicsInformedSolarHealthModel(
            num_days=dataset.num_days,
            input_dim=dataset.F,
        )
    elif args.model_type == 'segment_pinn_nodegradation':  # 新增不带退化因子的模型选项
        model = PhysicsInformedSolarHealthModelNoDegradation(
            num_days=dataset.num_days,
            input_dim=dataset.F,
        )
    elif args.model_type == 'neural_ode':  # Added Neural ODE model option
        model = NeuralODEHealthModel(
            num_days=dataset.num_days,
            input_dim=dataset.F,
            lstm_hidden_dim=args.lstm_hidden_dim,
            lstm_layers=args.lstm_layers,
            health_dim=args.health_dim,
            feat_hidden_dim=args.feat_hidden_dim,
        )
    elif args.model_type == 'transformer_neural_ode':  # 新增Transformer Neural ODE模型选项
        model = TransformerNeuralODE(
            num_days=dataset.num_days,
            input_dim=dataset.F,
            d_model=args.transformer_d_model,
            nhead=args.transformer_nhead,
            num_layers=args.transformer_num_layers,
            health_dim=args.health_dim,
            feat_hidden_dim=args.feat_hidden_dim,
        )
    elif args.model_type == 'physics_model':  # 新增纯物理模型选项
        model = PhysicsOnlyModel(num_days=dataset.num_days)
    elif args.model_type == 'segment':
        model = SolarHealthModel(
            num_days=dataset.num_days,
            input_dim=dataset.F,
        )
    elif args.model_type == 'simple_segment':
        model = SimpleSolarModel(
            num_days=dataset.num_days,
            input_dim=dataset.F,
        )
    elif args.model_type == 'gru_segment':
        model = GRUSolarModel(
            num_days=dataset.num_days,
            input_dim=dataset.F,
            gru_hidden_dim=args.gru_hidden_dim,
            gru_layers=args.gru_layers,
        )
    elif args.model_type == 'cnn_lstm_segment':
        model = CNNLSTMSolarModel(
            num_days=dataset.num_days,
            input_dim=dataset.F,
            cnn_channels=args.cnn_channels,
            cnn_kernel_size=args.cnn_kernel_size,
            lstm_hidden_dim=args.lstm_hidden_dim,
            lstm_layers=args.lstm_layers,
        )
    elif args.model_type == 'transformer_segment':
        model = TransformerSolarModel(
            num_days=dataset.num_days,
            input_dim=dataset.F,
            d_model=args.transformer_d_model,
            nhead=args.transformer_nhead,
            num_layers=args.transformer_num_layers,
            dim_feedforward=args.transformer_dim_feedforward,
            dropout=args.transformer_dropout,
        )
    elif args.model_type == 'ml_rf':
        model = MLRandomForest(n_estimators=100, random_state=42)
    elif args.model_type == 'ml_svm':
        model = MLSVM(kernel='rbf', C=1.0)
    elif args.model_type == 'ml_lr':
        model = MLLinearRegression()
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")
    return model

def create_trainer(args, model, device):
    """创建训练器"""
    if args.model_type in ['segment_pinn', 'segment_pinn_nodegradation', 'neural_ode', 'transformer_neural_ode', 'cnn_gru_neural_ode']:
        trainer = EnhancedSegmentPINNTrainer(model, save_path=args.save_path, device=device)
    elif args.model_type in ['segment', 'simple_segment', 'gru_segment', 'cnn_lstm_segment', 'transformer_segment']:
        trainer = EnhancedSegmentTrainer(model, save_path=args.save_path, device=device)
    elif args.model_type == 'physics_model':
        # 物理模型使用专用训练器
        trainer = PhysicsModelTrainer(model, save_path=args.save_path, device=device)
    else:
        trainer = None
    return trainer

def perform_cross_validation(args, dataset, device):
    """执行时间序列交叉验证"""
    print("执行时间序列交叉验证...")
    
    # 获取完整的数据集索引
    all_indices = np.array(list(range(len(dataset))))
    
    # 初始化TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=args.n_splits)
    
    # 存储每次交叉验证的结果
    cv_results = []
    
    # 为每次交叉验证创建模型和训练器
    for fold, (train_indices, val_indices) in enumerate(tscv.split(all_indices)):
        print(f"\n执行第 {fold + 1} 折交叉验证...")
        
        # 转换索引为列表格式
        train_indices_list = train_indices.tolist()
        val_indices_list = val_indices.tolist()
        
        # 创建训练集和验证集
        train_dataset = Subset(dataset, train_indices_list)
        val_dataset = Subset(dataset, val_indices_list)
        
        # 计算归一化参数（仅基于训练集）
        dataset.compute_normalization_params(indices=train_indices_list)
        dataset.prepare_all_features()
        
        # 创建数据加载器
        if args.model_type in ['segment_pinn', 'neural_ode', 'transformer_neural_ode', 'cnn_gru_neural_ode', 'segment_pinn_nodegradation', 'physics_model']:
            train_dataset_with_raw = RawFeatureDatasetWrapper(dataset, train_dataset.indices)
            val_dataset_with_raw = RawFeatureDatasetWrapper(dataset, val_dataset.indices)

            train_loader = DataLoader(train_dataset_with_raw, batch_size=args.train_batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset_with_raw, batch_size=args.val_batch_size, shuffle=False)
        elif args.model_type in ['segment', 'simple_segment', 'gru_segment', 'cnn_lstm_segment', 'transformer_segment']:
            train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False)
        else:
            raise ValueError(f"不支持交叉验证的模型类型: {args.model_type}")
        
        # 创建模型和训练器
        model = create_model(args, dataset)
        trainer = create_trainer(args, model, device)
        
        if trainer is None:
            raise ValueError(f"无法为模型类型 {args.model_type} 创建训练器")
        
        # 训练模型
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            config=args
        )
        
        # 评估模型
        val_results = trainer.test(val_loader, config=args)
        
        # 存储结果
        cv_results.append({
            'fold': fold + 1,
            'val_loss': val_results['test_loss'],
            'mse': val_results['mse'],
            'rmse': val_results['rmse'],
            'mae': val_results['mae'],
            'mape': val_results['mape'],
            'r2': val_results['r2']
        })
        
        print(f"第 {fold + 1} 折验证损失: {val_results['test_loss']:.6f}")
    
    # 计算平均结果
    avg_results = {
        'avg_val_loss': np.mean([r['val_loss'] for r in cv_results]),
        'avg_mse': np.mean([r['mse'] for r in cv_results]),
        'avg_rmse': np.mean([r['rmse'] for r in cv_results]),
        'avg_mae': np.mean([r['mae'] for r in cv_results]),
        'avg_mape': np.mean([r['mape'] for r in cv_results]),
        'avg_r2': np.mean([r['r2'] for r in cv_results]),
        'std_val_loss': np.std([r['val_loss'] for r in cv_results]),
        'std_mse': np.std([r['mse'] for r in cv_results]),
        'std_rmse': np.std([r['rmse'] for r in cv_results]),
        'std_mae': np.std([r['mae'] for r in cv_results]),
        'std_mape': np.std([r['mape'] for r in cv_results]),
        'std_r2': np.std([r['r2'] for r in cv_results])
    }
    
    # 打印交叉验证结果
    print("\n交叉验证结果:")
    print("-" * 50)
    for result in cv_results:
        print(f"第 {result['fold']} 折 - 验证损失: {result['val_loss']:.6f}, RMSE: {result['rmse']:.6f}")
    
    print("-" * 50)
    print(f"平均验证损失: {avg_results['avg_val_loss']:.6f} ± {avg_results['std_val_loss']:.6f}")
    print(f"平均MSE: {avg_results['avg_mse']:.6f} ± {avg_results['std_mse']:.6f}")
    print(f"平均RMSE: {avg_results['avg_rmse']:.6f} ± {avg_results['std_rmse']:.6f}")
    print(f"平均MAE: {avg_results['avg_mae']:.6f} ± {avg_results['std_mae']:.6f}")
    print(f"平均MAPE: {avg_results['avg_mape']:.6f} ± {avg_results['std_mape']:.6f}")
    print(f"平均R²: {avg_results['avg_r2']:.6f} ± {avg_results['std_r2']:.6f}")
    
    # 保存交叉验证结果
    cv_results_file = os.path.join(args.save_path, 'cross_validation_results.txt')
    with open(cv_results_file, 'w', encoding='utf-8') as f:
        f.write("交叉验证详细结果:\n")
        f.write("-" * 50 + "\n")
        for result in cv_results:
            f.write(f"第 {result['fold']} 折 - 验证损失: {result['val_loss']:.6f}, "
                    f"MSE: {result['mse']:.6f}, RMSE: {result['rmse']:.6f}, "
                    f"MAE: {result['mae']:.6f}, MAPE: {result['mape']:.6f}, R²: {result['r2']:.6f}\n")
        
        f.write("-" * 50 + "\n")
        f.write(f"平均验证损失: {avg_results['avg_val_loss']:.6f} ± {avg_results['std_val_loss']:.6f}\n")
        f.write(f"平均MSE: {avg_results['avg_mse']:.6f} ± {avg_results['std_mse']:.6f}\n")
        f.write(f"平均RMSE: {avg_results['avg_rmse']:.6f} ± {avg_results['std_rmse']:.6f}\n")
        f.write(f"平均MAE: {avg_results['avg_mae']:.6f} ± {avg_results['std_mae']:.6f}\n")
        f.write(f"平均MAPE: {avg_results['avg_mape']:.6f} ± {avg_results['std_mape']:.6f}\n")
        f.write(f"平均R²: {avg_results['avg_r2']:.6f} ± {avg_results['std_r2']:.6f}\n")
    
    print(f"\n交叉验证结果已保存到: {cv_results_file}")
    
    return cv_results, avg_results

def train_and_evaluate(args, trainer, train_loader, val_loader, test_loader, model, dataset):
    """训练和评估模型"""
    # 创建训练配置
    
    if args.model_type in ['ml_rf', 'ml_svm', 'ml_lr']:
        # 处理机器学习模型
        print("开始训练传统机器学习模型...")
        
        # 转换训练数据
        X_train, y_train = convert_torch_dataset_to_numpy(train_loader)
        X_val, y_val = convert_torch_dataset_to_numpy(val_loader)
        X_test, y_test = convert_torch_dataset_to_numpy(test_loader)
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 评估模型
        train_results = evaluate_ml_model(model, X_train, y_train)
        val_results = evaluate_ml_model(model, X_val, y_val)
        test_results = evaluate_ml_model(model, X_test, y_test)
        
        print("\n训练集结果:")
        print(f"训练损失: {train_results['test_loss']:.6f}")
        print(f"均方误差 (MSE): {train_results['mse']:.6f}")
        print(f"均方根误差 (RMSE): {train_results['rmse']:.6f}")
        print(f"平均绝对误差 (MAE): {train_results['mae']:.6f}")
        print(f"平均绝对百分比误差 (MAPE): {train_results['mape']:.6f}")
        print(f"决定系数 (R²): {train_results['r2']:.6f}")
        
        print("\n验证集结果:")
        print(f"验证损失: {val_results['test_loss']:.6f}")
        print(f"均方误差 (MSE): {val_results['mse']:.6f}")
        print(f"均方根误差 (RMSE): {val_results['rmse']:.6f}")
        print(f"平均绝对误差 (MAE): {val_results['mae']:.6f}")
        print(f"平均绝对百分比误差 (MAPE): {val_results['mape']:.6f}")
        print(f"决定系数 (R²): {val_results['r2']:.6f}")
        
        print("\n测试集结果:")
        print(f"测试损失: {test_results['test_loss']:.6f}")
        print(f"均方误差 (MSE): {test_results['mse']:.6f}")
        print(f"均方根误差 (RMSE): {test_results['rmse']:.6f}")
        print(f"平均绝对误差 (MAE): {test_results['mae']:.6f}")
        print(f"平均绝对百分比误差 (MAPE): {test_results['mape']:.6f}")
        print(f"决定系数 (R²): {test_results['r2']:.6f}")
        
        # 保存模型
        model_path = os.path.join(args.save_path, f'{args.model_type}_model.pkl')
        model.save_model(model_path)
        print(f"\n模型已保存至: {model_path}")
        
        # 保存测试结果
        save_ml_test_results(args.save_path, test_results, y_test)
        print(f"\n测试结果已保存至: {args.save_path}")
        
        return None, test_results
    else:
        print("开始训练...")
            
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            config=args
        )
        
        # 绘制损失曲线
        if args.show_plot:
            trainer.plot_losses()
        
        # 测试模型
        print("\n开始测试模型...")
        test_results = trainer.test(test_loader, config=args)

        
        # 打印详细的测试结果
        print("\n详细测试结果:")
        print(f"测试损失: {test_results['test_loss']:.6f}")
        print(f"均方误差 (MSE): {test_results['mse']:.6f}")
        print(f"均方根误差 (RMSE): {test_results['rmse']:.6f}")
        print(f"平均绝对误差 (MAE): {test_results['mae']:.6f}")
        print(f"平均绝对百分比误差 (MAPE): {test_results['mape']:.6f}")
        print(f"决定系数 (R²): {test_results['r2']:.6f}")
        
        # 绘制退化曲线（仅适用于特定模型）
        if args.show_plot and (args.model_type in ['segment_pinn', 'neural_ode']):
            trainer.plot_degradation_curve()
        
        return history, test_results

def main():
    args = get_args()
    set_random_seed(args.seed)
    save_code(args.save_path)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 如果启用交叉验证，则执行交叉验证
    if args.cross_validation:
        dataset = SegmentDataset(data_dir='./data', normalize=False)
        cv_results, avg_results = perform_cross_validation(args, dataset, device)
        return None, None, {'cv_results': cv_results, 'avg_results': avg_results}
    
    # 创建数据加载器
    train_loader, val_loader, test_loader, dataset = create_data_loaders(args)
    
    # 创建模型
    model = create_model(args, dataset)
    
    # 创建训练器
    trainer = create_trainer(args, model, device)
    
    # 训练和评估模型
    history, test_results = train_and_evaluate(args, trainer, train_loader, val_loader, test_loader, model, dataset)

    return model, history, test_results


if __name__ == "__main__":
    main()