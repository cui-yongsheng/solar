from model.physics_model import SegmentPhysicsModel
import torch
import torch.nn as nn

class PhysicsOnlyModel(nn.Module):
    """
    仅使用物理模型的简化版本
    直接使用物理模型进行电流预测，可选择添加基于天数的简单衰减参数
    """

    def __init__(self, num_days=None):
        """
        初始化仅物理模型
        
        参数:
        num_days: 总天数，用于计算衰减因子
        """
        super().__init__()
        # 物理模型
        self.physics_model = SegmentPhysicsModel()
        
        # 注册衰减率参数，初始值设为接近0的小值，表示轻微衰减
        self.degradation_rate = nn.Parameter(torch.tensor(0.1))
        
        # 保存总天数
        self.num_days = num_days

    def forward(self, X=None, day_ids=None, raw_features=None):
        """
        使用物理模型直接计算电流值，并可选地应用基于天数的衰减
        
        参数:
        X: 输入特征 [B, T, F] (用于保持接口一致性)
        day_ids: 天数索引 [B] (用于计算衰减因子)
        raw_features: [B, T, F_raw] 用于物理计算的原始特征
        
        返回:
        I_pred: [B, T, 1] 预测电流值
        """
        # 直接使用物理模型计算电流
        I_pred = self.physics_model.compute_current(raw_features)  # [B, T, 1]
        
        # 计算每个样本的衰减因子
        # 衰减因子 = exp(-degradation_rate * days)
        # 归一化天数以确保衰减率在合理范围内
        normalized_days = day_ids.float() / (self.num_days if self.num_days is not None and self.num_days > 0 else 365.0)
        
        # 计算衰减因子
        degradation_factors = torch.exp(-self.degradation_rate * normalized_days)  # [B]
        degradation_factors = degradation_factors.unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
        
        # 应用衰减因子
        I_pred = I_pred * degradation_factors  # [B, T, 1]
        return {
            "I_pred": I_pred,
            "I_phys": I_pred,  # 在此模型中，预测值就是物理值
        }

    def get_degradation_curve(self, device, num_days=1000):
        """
        获取衰减曲线
        
        参数:
        device: 计算设备
        num_days: 曲线包含的天数
        
        返回:
        degradation_curve: 衰减因子曲线
        """
        # 创建天数张量
        days = torch.arange(num_days, device=device, dtype=torch.float)
        
        # 归一化天数
        normalized_days = days / (self.num_days if self.num_days is not None and self.num_days > 0 else 365.0)
            
        # 计算衰减曲线
        degradation_curve = torch.exp(-self.degradation_rate * normalized_days)
        
        return degradation_curve