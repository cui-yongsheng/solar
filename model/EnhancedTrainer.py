from .BaseTrainer import BaseTrainer
from torch.nn import functional as F
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

## 针对点输入时的训练方式
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
        # 支持 dataset 返回 (data, target, day_ids) 或 (data, target, day_ids, mask)
        data, target, day_ids, mask = batch_data
        data = data.to(self.device)
        target = target.to(self.device)
        day_ids = day_ids.to(self.device)
        mask = mask.to(self.device)

        output = self.model(data, day_ids)
        # 确保output和target形状一致
        if output.shape != target.shape:
            target = target.view_as(output)

        # 按真实位置计算MSE
        eps = 1e-8
        mse = (output - target) ** 2
        masked = mse * mask
        loss = masked.sum() / (mask.sum() + eps)

        return loss

    def _predict_and_evaluate(self, batch_data, criterion, config=None):
        """预测并评估单个批次"""
        data, target, day_ids, mask = batch_data
        data = data.to(self.device)
        target = target.to(self.device)
        day_ids = day_ids.to(self.device)
        mask = mask.to(self.device)

        output = self.model(data, day_ids)
        # 确保output和target形状一致
        if output.shape != target.shape:
            target = target.view_as(output)
        # 使用 mask 布尔索引截取真实位置的预测与标签，然后计算损失
        # mask 可能形状为 [B, T, 1] 或 [B, T]
        # 如果最后一维是 1，则去掉以匹配 output/target 的时间维
        if mask.dim() == output.dim() and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        mask = mask.bool()
        output_mask = output[mask]
        target_mask = target[mask]
        data_loss = ((output_mask - target_mask) ** 2).mean()

        return output, target, mask, day_ids, data_loss

    def test(self, test_loader, config=None):
        """测试模型"""
        return super().test(test_loader, plot_predictions=config.show_plot ,config=config)


class EnhancedSegmentPINNTrainer(BaseTrainer):
    def __init__(self, model, save_path="./", device=torch.device("cpu")):
        super().__init__(model, save_path, device)

    def _compute_loss(self, batch_data, criterion=None, config=None):
        normalized_features, raw_features, target, day_ids, mask = batch_data
        normalized_features = normalized_features.to(self.device)
        raw_features = raw_features.to(self.device)
        target = target.to(self.device)
        day_ids = day_ids.to(self.device)
        mask = mask.to(self.device)

        # ---------- Forward ----------
        output_dict = self.model(normalized_features, day_ids, raw_features)
        I_pred = output_dict["I_pred"]
        I_phys = output_dict["I_phys"]
        r_pred = output_dict["r_pred"]

        if I_pred.shape != target.shape:
            target = target.view_as(I_pred)

        # ============================================================
        # 1) 主数据损失（最终预测 I_pred），按真实位置计算
        # ============================================================
        eps = 1e-8
        mse = (I_pred - target) ** 2
        masked = mse * mask
        data_loss = masked.sum() / (mask.sum() + eps)

        # 初始化总损失
        loss = data_loss

        # 计算正则化项，应用mask
        # 对r_pred应用mask以只考虑真实位置的残差
        r_pred_masked = r_pred * mask
        res_reg = (r_pred_masked ** 2).sum() / (mask.sum() + eps)
        lambda_r = getattr(config, 'lambda_r', 1e-2)
        
        # ============================================================
        # 2) D 的弱监督（帮助 D 收敛）
        #    D_tilde ≈ target / I_phys
        # ============================================================
        # 检查模型是否有退化因子D
        has_degradation = "D" in output_dict and "I_base" in output_dict
        if has_degradation:
            D = output_dict["D"]
            eps = 3.5
            mask_phy = (I_phys.abs() < eps)
            # 结合数据mask和物理mask，确保使用布尔类型进行位运算
            if mask.dim() == D.dim() and mask.shape[-1] == 1:
                combined_mask = mask.squeeze(-1).bool() & mask_phy.squeeze(-1).bool()
            else:
                combined_mask = mask.bool() & mask_phy.bool()
            
            D_tilde = torch.zeros_like(D)
            D_tilde[mask_phy] = (target[mask_phy] / (I_phys[mask_phy] + eps)).clamp(0.1, 2.0)
            D_curve = self.model.get_degradation_curve(self.device)
            # 添加衰减约束：鼓励D_curve随时间递减
            D_decay_loss = 0.0
            if len(D_curve) > 1:
                # 计算相邻时间步的差异
                D_curve_diff = D_curve[1:] - D_curve[:-1]
                # 鼓励负差异（即衰减），使用relu函数惩罚正向变化
                D_decay_loss = F.relu(D_curve_diff).mean()
            
            # 添加平滑约束：限制D_curve的变化幅度
            D_smooth_loss = 0.0
            if len(D_curve) > 2:
                # 计算二阶差异（加速度）
                D_curve_second_diff = D_curve[2:] - 2 * D_curve[1:-1] + D_curve[:-2]
                D_smooth_loss = D_curve_second_diff.abs().mean()
            # 从配置中获取权重
            lambda_D_decay = getattr(config, 'lambda_D_decay', 1e-3)
            lambda_D_smooth = getattr(config, 'lambda_D_smooth', 1e-3)
            # 应用combined_mask计算损失
            if combined_mask.sum() > 0:
                loss_D_supervise = F.mse_loss(D[combined_mask], D_tilde[combined_mask])
                lambda_D_supervise = getattr(config, 'lambda_D_supervise', 1e-2)
                loss = (
                        data_loss
                        + lambda_D_supervise * loss_D_supervise
                        + lambda_r * res_reg
                        + lambda_D_decay * D_decay_loss
                        + lambda_D_smooth * D_smooth_loss
                )
            else:
                # 如果没有有效的mask位置，则只使用data_loss和res_reg
                loss = data_loss + lambda_r * res_reg
        else:
            # 对于无退化因子的模型，只添加残差正则项
            loss = data_loss + lambda_r * res_reg
        return loss

    def _predict_and_evaluate(self, batch_data, criterion, config=None):
        """预测并评估单个批次"""
        normalized_features, raw_features, target, day_ids, mask = batch_data
        normalized_features = normalized_features.to(self.device)
        raw_features = raw_features.to(self.device)
        target = target.to(self.device)
        day_ids = day_ids.to(self.device)
        mask = mask.to(self.device)

        # 模型输出
        output_dict = self.model(normalized_features, day_ids, raw_features)
        output = output_dict["I_pred"]
        # 确保output和target形状一致
        if output.shape != target.shape:
            target = target.view_as(output)

        # 使用 mask 布尔索引截取真实位置的预测与标签，然后计算损失
        # mask 可能形状为 [B, T, 1] 或 [B, T]
        # 如果最后一维是 1，则去掉以匹配 output/target 的时间维
        if mask.dim() == output.dim() and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        mask = mask.bool()
        output_mask = output[mask]
        target_mask = target[mask]
        data_loss = ((output_mask - target_mask) ** 2).mean()

        return output, target, mask, day_ids, data_loss
        
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

    def plot_degradation_curve(self, show_plot):
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
            if show_plot:
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
        normalized_features, raw_features, target, day_ids, mask = batch_data
        normalized_features = normalized_features.to(self.device)
        raw_features = raw_features.to(self.device)
        target = target.to(self.device)
        day_ids = day_ids.to(self.device)
        mask = mask.to(self.device)

        # 模型前向传播
        output_dict = self.model(normalized_features, day_ids, raw_features)
        output = output_dict["I_pred"]

        # 确保输出和目标形状一致
        if output.shape != target.shape:
            target = target.view_as(output)

        eps = 1e-8
        mse = (output - target) ** 2
        masked = mse * mask
        loss = masked.sum() / (mask.sum() + eps)
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
        normalized_features, raw_features, target, day_ids, mask = batch_data
        normalized_features = normalized_features.to(self.device)
        raw_features = raw_features.to(self.device)
        target = target.to(self.device)
        day_ids = day_ids.to(self.device)
        mask = mask.to(self.device)
        
        # 模型前向传播
        output_dict = self.model(normalized_features, day_ids, raw_features)
        output = output_dict["I_pred"]
        
        # 确保输出和目标形状一致
        if output.shape != target.shape:
            target = target.view_as(output)
            
        # 计算损失
        # 使用 mask 布尔索引截取真实位置的预测与标签，然后计算损失
        # mask 可能形状为 [B, T, 1] 或 [B, T]
        # 如果最后一维是 1，则去掉以匹配 output/target 的时间维
        if mask.dim() == output.dim() and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        mask = mask.bool()
        output_mask = output[mask]
        target_mask = target[mask]
        data_loss = ((output_mask - target_mask) ** 2).mean()

        return output, target, mask, day_ids, data_loss

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
        
    def plot_degradation_curve(self, show_plot):
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
            if show_plot:
                plt.show()
            
        return D_curve_np
