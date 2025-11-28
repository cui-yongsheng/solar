from .BaseTrainer import BaseTrainer
from torch.nn import functional as F
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

## 针对点输入时的训练方式，目前不需要
class EnhancedNeuralNetworkTrainer(BaseTrainer):
    def __init__(self, model, save_path="./", device=torch.device("cpu")):
        super().__init__(model, save_path, device)
        
    def _compute_loss(self, batch_data, criterion, config=None):
        """计算神经网络损失"""
        data, target = batch_data
        data, target = data.to(self.device), target.to(self.device)
        
        output = self.model(data)
        # 确保output和target形状一致
        if output.shape != target.shape:
            target = target.view_as(output)
        loss = criterion(output, target)
        return loss

    def _predict_and_evaluate(self, batch_data, criterion, config=None):
        """预测并评估单个批次"""
        data, target = batch_data
        data, target = data.to(self.device), target.to(self.device)
        
        output = self.model(data)
        # 确保output和target形状一致
        if output.shape != target.shape:
            target = target.view_as(output)
        loss = criterion(output, target)
        
        return output, target, loss

    def test(self, test_loader, config=None):
        """测试模型"""
        return super().test(test_loader, plot_predictions=config.show_plot ,config=config)

class EnhancedPINNTrainer(BaseTrainer):
    def __init__(self, model, save_path="./", device=torch.device("cpu")):
        super().__init__(model, save_path, device)
        
    def _compute_loss(self, batch_data, criterion, config=None):
        """计算PINN损失"""
        normalized_features, raw_features, target = batch_data
        normalized_features = normalized_features.to(self.device)
        raw_features = raw_features.to(self.device)
        target = target.to(self.device)
        
        # 计算数据损失
        loss = self.model.physics_loss(normalized_features, target, raw_features, config.physics_weight)
            
        return loss

    def _predict_and_evaluate(self, batch_data, criterion, config=None):
        """预测并评估单个批次"""
        normalized_features, raw_features, target = batch_data
        normalized_features = normalized_features.to(self.device)
        raw_features = raw_features.to(self.device)
        target = target.to(self.device)
        
        output = self.model(normalized_features, raw_features)
        # 确保output和target形状一致
        if output.shape != target.shape:
            target = target.view_as(output)
        data_loss = nn.MSELoss()(output, target)
        
        return output, target, data_loss

    def test(self, test_loader, config=None):
        """测试模型"""
        return super().test(test_loader, plot_predictions=config.show_plot ,config=config)

## 针对分段输入时的训练方式
class EnhancedSegmentTrainer(BaseTrainer):
    def __init__(self, model, save_path="./", device=torch.device("cpu")):
        super().__init__(model, save_path, device)
        
    def _compute_loss(self, batch_data, criterion, config=None):
        """计算分段模型损失"""
        data, target, day_ids = batch_data
        data, target, day_ids = data.to(self.device), target.to(self.device), day_ids.to(self.device)
        
        output = self.model(data, day_ids)
        # 确保output和target形状一致
        if output.shape != target.shape:
            target = target.view_as(output)
        loss = criterion(output, target)
            
        return loss

    def _predict_and_evaluate(self, batch_data, criterion, config=None):
        """预测并评估单个批次"""
        data, target, day_ids = batch_data
        data, target, day_ids = data.to(self.device), target.to(self.device), day_ids.to(self.device)
        
        output = self.model(data, day_ids)
        # 确保output和target形状一致
        if output.shape != target.shape:
            target = target.view_as(output)
        data_loss = criterion(output, target)
        
        return output, target, data_loss

    def test(self, test_loader, config=None):
        """测试模型"""
        return super().test(test_loader, plot_predictions=config.show_plot ,config=config)


class EnhancedSegmentPINNTrainer(BaseTrainer):
    def __init__(self, model, save_path="./", device=torch.device("cpu")):
        super().__init__(model, save_path, device)

    def _compute_loss(self, batch_data, criterion=None, config=None):
        normalized_features, raw_features, target, day_ids = batch_data
        normalized_features = normalized_features.to(self.device)
        raw_features = raw_features.to(self.device)
        target = target.to(self.device)
        day_ids = day_ids.to(self.device)

        # ---------- Forward ----------
        output_dict = self.model(normalized_features, day_ids, raw_features)
        I_pred = output_dict["I_pred"]
        I_phys = output_dict["I_phys"]
        r_pred = output_dict["r_pred"]

        if I_pred.shape != target.shape:
            target = target.view_as(I_pred)

        # ============================================================
        # 1) 主数据损失（最终预测 I_pred）
        # ============================================================
        data_loss = F.mse_loss(I_pred, target)

        # 初始化总损失
        loss = data_loss

        # 计算正则化项
        res_reg = (r_pred ** 2).mean()
        lambda_r = getattr(config, 'lambda_r', 1e-2)
        
        # ============================================================
        # 2) D 的弱监督（帮助 D 收敛）
        #    D_tilde ≈ target / I_phys
        # ============================================================
        # 检查模型是否有退化因子D
        has_degradation = "D" in output_dict and "I_base" in output_dict
        if has_degradation:
            D = output_dict["D"]
            eps = 1e-6
            mask = (I_phys.abs() > eps)
            D_tilde = torch.zeros_like(D)
            D_tilde[mask] = (target[mask] / (I_phys[mask] + eps)).clamp(0.1, 2.0)

            loss_D_supervise = F.mse_loss(D[mask], D_tilde[mask])
            lambda_D_supervise = getattr(config, 'lambda_D_supervise', 1e-1)
            loss = (
                    data_loss
                    + lambda_D_supervise * loss_D_supervise
                    + lambda_r * res_reg
            )
        else:
            # 对于无退化因子的模型，只添加残差正则项
            loss = data_loss + lambda_r * res_reg
        return loss

    def _predict_and_evaluate(self, batch_data, criterion, config=None):
        """预测并评估单个批次"""
        normalized_features, raw_features, target, day_ids = batch_data
        normalized_features = normalized_features.to(self.device)
        raw_features = raw_features.to(self.device)
        target = target.to(self.device)
        day_ids = day_ids.to(self.device)
        
        # 模型输出
        output_dict = self.model(normalized_features, day_ids, raw_features)
        output = output_dict["I_pred"]
        # 确保output和target形状一致
        if output.shape != target.shape:
            target = target.view_as(output)
        data_loss = nn.MSELoss()(output, target)
        
        return output, target, data_loss
        
    def test(self, test_loader, config=None):
        """测试模型"""
        # 在测试时启用缓存
        if hasattr(self.model, 'enable_cache'):
            self.model.enable_cache()
        
        result = super().test(test_loader, plot_predictions=config.show_plot ,config=config)
        
        # 测试完成后禁用缓存
        if hasattr(self.model, 'disable_cache'):
            self.model.disable_cache()
            
        return result

    def plot_degradation_curve(self, save_path=None):
        """获取D衰减曲线并可选择保存结果
        
        Args:
            save_path (str, optional): 保存路径，如果提供则保存图像和数据
        """
        # 只有当模型具有退化曲线功能时才绘制
        if not hasattr(self.model, 'get_degradation_curve'):
            print("当前模型不支持退化曲线绘制")
            return None
            
        self.model.eval()
        with torch.no_grad():
            D_curve = self.model.get_degradation_curve(self.device)
            D_curve_np = D_curve.cpu().numpy()
            
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(D_curve_np)), D_curve_np)
            plt.xlabel('Days')
            plt.ylabel('Degradation Factor (D)')
            plt.title('Solar Panel Degradation Curve')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            default_save_path = os.path.join(self.save_path, "degradation_curve.png")
            plt.savefig(default_save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
        return D_curve_np


class PhysicsModelTrainer(BaseTrainer):
    """
    物理模型专用训练器，用于保持与其它模型一致的接口
    """

    def __init__(self, model, save_path="./", device=None):
        """
        初始化物理模型训练器

        参数:
        model: 物理模型
        save_path: 保存路径
        device: 计算设备
        """
        super().__init__(model, save_path, device)
        
    def _compute_loss(self, batch_data, criterion, config=None):
        """
        计算物理模型的损失

        参数:
        batch_data: 批次数据 (normalized_features, raw_features, target, day_ids)
        criterion: 损失函数
        config: 配置对象

        返回:
        损失值
        """
        normalized_features, raw_features, target, day_ids = batch_data
        normalized_features = normalized_features.to(self.device)
        raw_features = raw_features.to(self.device)
        target = target.to(self.device)
        day_ids = day_ids.to(self.device)

        # 模型前向传播
        output_dict = self.model(normalized_features, day_ids, raw_features)
        output = output_dict["I_pred"]
        
        # 确保输出和目标形状一致
        if output.shape != target.shape:
            target = target.view_as(output)
            
        # 计算损失
        loss = criterion(output, target)
        return loss

    def _predict_and_evaluate(self, batch_data, criterion, config=None):
        """
        预测并评估单个批次

        参数:
        batch_data: 批次数据 (normalized_features, raw_features, target, day_ids)
        criterion: 损失函数
        config: 配置对象

        返回:
        (预测值, 实际值, 损失)
        """
        normalized_features, raw_features, target, day_ids = batch_data
        normalized_features = normalized_features.to(self.device)
        raw_features = raw_features.to(self.device)
        target = target.to(self.device)
        day_ids = day_ids.to(self.device)
        
        # 模型前向传播
        output_dict = self.model(normalized_features, day_ids, raw_features)
        output = output_dict["I_pred"]
        
        # 确保输出和目标形状一致
        if output.shape != target.shape:
            target = target.view_as(output)
            
        # 计算损失
        loss = criterion(output, target)
        
        return output, target, loss

    def train(self, train_loader, val_loader, config=None):
        """
        物理模型训练方法，继承BaseTrainer的训练方式

        参数:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        config: 训练配置对象

        返回:
        训练历史记录
        """
        # 直接使用BaseTrainer的训练方法
        return super().train(train_loader, val_loader, config)
        
    def plot_degradation_curve(self, save_path=None):
        """
        绘制退化曲线
        
        参数:
        save_path: 保存路径
        """
        # 检查模型是否有退化曲线功能
        if not hasattr(self.model, 'get_degradation_curve'):
            print("当前模型不支持退化曲线绘制")
            return None
            
        self.model.eval()
        with torch.no_grad():
            D_curve = self.model.get_degradation_curve(self.device)
            D_curve_np = D_curve.cpu().numpy()
            
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(D_curve_np)), D_curve_np)
            plt.xlabel('Days')
            plt.ylabel('Degradation Factor')
            plt.title('Solar Panel Degradation Curve')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 保存图像
            default_save_path = os.path.join(self.save_path, "degradation_curve.png") if self.save_path else "degradation_curve.png"
            plt.savefig(default_save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
        return D_curve_np
