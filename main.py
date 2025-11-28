import torch
import random
from torch.utils.data import DataLoader
from dataset.dataset import SatelliteDataset
from model.nn import SimpleNN, LSTMModel, ResNetModel, AttentionModel, AutoencoderModel
from model.EnhancedTrainer import EnhancedNeuralNetworkTrainer, EnhancedPINNTrainer
from utils.args import get_args
from utils.utils import set_random_seed, save_code
from model.pinn import PINN


def create_data_loaders(args):
    """创建数据加载器"""
    print("数据集加载...")
    dataset = SatelliteDataset(data_dir='./data', normalize=False)  # 初始化时不进行归一化
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    if args.few_shot:
        # Few-shot learning: randomly sample a small subset for training
        print("使用小样本训练模式")
        few_shot_size = min(int(0.2 * train_size), 50)  # Use 20% of training data or 50 samples, whichever is smaller
        print(f"小样本数量: {few_shot_size}")
        
        # Randomly select few-shot samples from the training set
        full_train_indices = list(range(0, train_size))
        few_shot_indices = random.sample(full_train_indices, few_shot_size)
        
        # Use few-shot indices for training
        train_indices = few_shot_indices
        val_indices = list(range(train_size, train_size + val_size))
        test_indices = list(range(train_size + val_size, len(dataset)))
    else:
        # 前80%用于训练，中间10%用于验证，后10%用于测试
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
    
    # 仅为PINN模型创建自定义数据集包装器以支持原始特征
    if args.model_type == 'pinn':
        # 创建自定义数据集包装器以支持原始特征
        class RawFeatureDatasetWrapper(torch.utils.data.Dataset):
            def __init__(self, dataset, indices):
                self.dataset = dataset
                self.indices = indices
                
            def __len__(self):
                return len(self.indices)
                
            def __getitem__(self, idx):
                # 使用新的方法获取包含原始特征的数据
                normalized_features, raw_features, label = self.dataset.get_item_with_raw_features(self.indices[idx])
                return normalized_features, raw_features, label
        
        # 创建包装后的数据集，用于提供原始特征
        train_dataset_with_raw = RawFeatureDatasetWrapper(dataset, train_dataset.indices)
        val_dataset_with_raw = RawFeatureDatasetWrapper(dataset, val_dataset.indices)
        test_dataset_with_raw = RawFeatureDatasetWrapper(dataset, test_dataset.indices)
        
        train_loader = DataLoader(train_dataset_with_raw, batch_size=args.train_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset_with_raw, batch_size=args.val_batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset_with_raw, batch_size=args.test_batch_size, shuffle=False)
    else:
        # 对于其他模型，直接使用普通的DataLoader
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, dataset

def create_model(args, dataset):
    """创建模型"""
    input_dim = dataset.get_feature_dim()
    # 根据args参数选择模型
    if args.model_type == 'pinn':
        model = PINN(input_dim=input_dim)
    elif args.model_type == 'lstm':
        model = LSTMModel(input_dim=input_dim, output_dim=1)
    elif args.model_type == 'resnet':
        model = ResNetModel(input_dim=input_dim, output_dim=1)
    elif args.model_type == 'attention':
        model = AttentionModel(input_dim=input_dim, output_dim=1)
    elif args.model_type == 'autoencoder':
        model = AutoencoderModel(input_dim=input_dim, output_dim=1)
    else:
        model = SimpleNN(input_dim=input_dim, output_dim=1)
    return model


def create_trainer(args, model, device):
    """创建训练器"""
    if args.model_type == 'pinn':
        trainer = EnhancedPINNTrainer(model, args.save_path, device=device)
    else:
        trainer = EnhancedNeuralNetworkTrainer(model, args.save_path, device=device)
    return trainer


def train_and_evaluate(args, trainer, train_loader, val_loader, test_loader):
    """训练和评估模型"""
    # 创建训练配置
    
    # 训练模型
    print("开始训练...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        config=args
    )

    # 绘制损失曲线
    trainer.plot_losses()
    # 测试模型
    print("\n开始测试模型...")
    test_results = trainer.test(test_loader)
    return history, test_results

def main():
    args = get_args()
    set_random_seed(args.seed)
    save_code(args.save_path)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    train_loader, val_loader, test_loader, dataset = create_data_loaders(args)
    
    # 创建模型
    model = create_model(args, dataset)
    
    # 创建训练器
    trainer = create_trainer(args, model, device)
    
    # 训练和评估模型
    history, test_results = train_and_evaluate(args, trainer, train_loader, val_loader, test_loader)

    return model, history, test_results


if __name__ == "__main__":
    main()