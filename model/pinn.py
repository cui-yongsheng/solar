import torch
import torch.nn as nn
from model.physics_model import PhysicsModel

class PINN(nn.Module):
    """
    物理信息神经网络（Physics-Informed Neural Network）模型
    用于预测卫星太阳能电池板的方阵电流
    """
    
    def __init__(self, input_dim, hidden_dims=None, output_dim=1, dropout_rate=0.1):
        """
        初始化PINN模型
        
        参数:
        input_dim: 输入维度
        hidden_dims: 隐藏层维度列表
        output_dim: 输出维度
        dropout_rate: Dropout比率
        """
        super(PINN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        # 初始化神经网络
        self.network = self._create_network(input_dim, hidden_dims, output_dim)
        
        # 添加门控机制网络 - 用于结合神经网络和物理模型输出
        self.gate_network = self._create_gate_network(input_dim, output_dim)
        
        # 初始化物理模型
        self.physics_model = PhysicsModel()

        # 各方向的电流修正因子
        self.x_neg_factor = nn.Parameter(torch.tensor(1.0))
        self.x_pos_factor = nn.Parameter(torch.tensor(1.0))
        self.y_neg_factor = nn.Parameter(torch.tensor(1.0))
        self.y_pos_factor = nn.Parameter(torch.tensor(1.0))
        
    def _create_network(self, input_dim, hidden_dims, output_dim):
        """
        创建主神经网络
        
        参数:
        input_dim: 输入维度
        hidden_dims: 隐藏层维度列表
        output_dim: 输出维度
        
        返回:
        神经网络模型
        """
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
            
        layers = []
        prev_dim = input_dim
        
        # 添加隐藏层
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            # 在隐藏层之间添加Dropout
            layers.append(nn.Dropout(self.dropout_rate))
            prev_dim = hidden_dim
            
        # 添加输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # 组合为Sequential模型
        return nn.Sequential(*layers)
        
    def _create_gate_network(self, input_dim, output_dim):
        """
        创建门控网络，用于动态结合神经网络和物理模型输出
        
        参数:
        input_dim: 输入维度
        output_dim: 输出维度
        
        返回:
        门控网络模型
        """
        return nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(32, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x, raw_features):
        """
        前向传播

        参数:
        x: 输入特征张量
        raw_features: 物理张量

        返回:
        方阵电流预测值
        """
        # 神经网络预测
        nn_output = self.network(x)

        
        # 物理模型预测
        physics_output = self.physics_model.compute_current(
            raw_features,
            self.x_neg_factor.item(),
            self.y_neg_factor.item(),
            self.x_pos_factor.item(),
            self.y_pos_factor.item(),
        )
        
        # 门控机制结合两者
        gate = self.gate_network(x)
        
        # 使用门控机制结合神经网络和物理模型输出
        combined_output = gate * nn_output + (1 - gate) * physics_output
        return combined_output
        
    def physics_loss(self, x, target_current, raw_features=None, physics_weight=0.1):
        """
        物理损失函数，结合神经网络预测和物理模型
        
        参数:
        x: 输入特征 (包含位置、状态[不含方阵电流]、距离等信息)
        target_current: 目标方阵电流
        raw_features: 原始（未归一化）特征，用于物理计算
        physics_weight: 物理一致性损失的权重
        
        返回:
        结合物理约束的损失值
        """
        # 神经网络预测的方阵电流
        predicted_current = self.forward(x, raw_features)
        
        # 计算物理模型预测的电流
        physics_current = self.physics_model.compute_current(
            raw_features,
            self.x_neg_factor.item(),
            self.y_neg_factor.item(),
            self.x_pos_factor.item(),
            self.y_pos_factor.item(),
        )
        
        # 计算数据损失（均方误差）
        data_loss = nn.MSELoss()(predicted_current.squeeze(), target_current.squeeze())
        
        # 计算物理一致性损失
        physics_loss = nn.MSELoss()(predicted_current.squeeze(), physics_current.squeeze())
        
        # 组合损失
        total_loss = data_loss + physics_weight * physics_loss  # 可调节权重
        return total_loss