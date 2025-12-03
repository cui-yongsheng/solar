from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch
import json
import os

class EarlyStopping:
    """
    早停机制类，用于在验证损失不再改善时停止训练
    """
    def __init__(self, patience=10, min_delta=1e-4, save_best_model=True, save_path=None):
        """
        初始化早停机制
        
        参数:
        patience: 容忍多少个epoch没有改善
        min_delta: 最小改善阈值
        save_best_model: 是否保存最佳模型
        save_path: 模型保存路径
        """
        self.patience = patience
        self.min_delta = min_delta
        self.save_best_model = save_best_model
        self.save_path = save_path
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, val_loss, model):
        """
        检查是否应该早停
        
        参数:
        val_loss: 当前验证损失
        model: 当前模型
        
        返回:
        是否应该早停
        """
        if val_loss < self.best_loss - self.min_delta:
            # 损失有显著改善
            self.best_loss = val_loss
            self.counter = 0
            # 保存最佳模型
            if self.save_best_model and self.save_path:
                self._save_model(model)
        else:
            # 损失没有显著改善
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop
    
    def _save_model(self, model):
        """保存最佳模型"""
        if self.save_path:
            best_model_path = os.path.join(self.save_path, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)


class BaseTrainer:
    """
    统一的训练器基类，为所有模型训练提供通用功能
    """

    def __init__(self, model, save_path=None, device=None):
        """
        初始化训练器

        参数:
        model: 要训练的模型
        save_path: 模型和结果保存路径
        device: 计算设备
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = save_path
        self.model.to(self.device)

        # 存储训练历史
        self.train_losses = []
        self.val_losses = []

    def train(self, train_loader, val_loader, config=None):
        """
        通用训练方法

        参数:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        config: 训练配置对象

        返回:
        训练历史记录
        """
        # 默认优化器和损失函数
        optimizer = self._get_optimizer(config)
        criterion = nn.MSELoss()
            
        # 初始化早停机制
        early_stopper = None
        if config.early_stopping:
            early_stopper = EarlyStopping(
                patience=getattr(config, 'patience', 10),
                min_delta=getattr(config, 'min_delta', 1e-4),
                save_best_model=getattr(config, 'save_best_model', True),
                save_path=self.save_path
            )

        # 训练模型
        for epoch in range(config.num_epochs):
            # 训练阶段
            train_loss = self._train_epoch(train_loader, optimizer, criterion, config)

            # 验证阶段
            val_loss = self._validate_epoch(val_loader, criterion, config)

            # 记录损失
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # 早停检查
            early_stop = False
            if early_stopper is not None:
                early_stop = early_stopper(val_loss, self.model)
            
            # 打印信息
            if config.verbose:
                print(f'Epoch [{epoch + 1}/{config.num_epochs}]')
                print(f'  Train Loss: {train_loss:.6f}')
                print(f'  Val Loss:   {val_loss:.6f}')
                print(f'  Loss Diff:  {abs(train_loss - val_loss):.6f}')
                if early_stopper is not None:
                    print(f'  Best Val Loss: {early_stopper.best_loss:.6f}')
                    print(f'  Patience Counter: {early_stopper.counter}/{getattr(config, "patience", 10)}')
                print('-' * 40)
            else:
                # 默认显示基本的轮次信息
                print(f'Epoch [{epoch + 1}/{config.num_epochs if config else 50}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            # 检查是否早停
            if early_stop:
                print(f"早停触发！在第 {epoch+1} 轮停止训练。")
                print(f"最佳验证损失: {early_stopper.best_loss:.6f}")
                break

        # 保存最终模型
        self._save_model()
        if config.verbose:
            print(f"模型已保存到 {self.save_path}")

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

    def _train_epoch(self, train_loader, optimizer, criterion, config=None):
        """
        训练一个epoch

        参数:
        train_loader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        config: 配置对象

        返回:
        训练损失
        """
        self.model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc='Training', leave=False)

        for batch_data in train_pbar:
            loss = self._compute_loss(batch_data, criterion, config)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
                
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({'Batch Loss': f'{loss.item():.6f}'})

        train_loss /= len(train_loader)
        return train_loss

    def _validate_epoch(self, val_loader, criterion, config=None):
        """
        验证一个epoch

        参数:
        val_loader: 验证数据加载器
        criterion: 损失函数
        config: 配置对象

        返回:
        验证损失
        """
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc='Validation', leave=False)
            for batch_data in val_pbar:
                loss = self._compute_loss(batch_data, criterion, config)
                val_loss += loss.item()
                val_pbar.set_postfix({'Batch Loss': f'{loss.item():.6f}'})

        val_loss /= len(val_loader)
        return val_loss

    def _compute_loss(self, batch_data, criterion, config=None):
        """
        计算单个批次的损失，子类需要根据具体情况重写此方法

        参数:
        batch_data: 批次数据
        criterion: 损失函数
        config: 配置对象

        返回:
        损失值
        """
        raise NotImplementedError("子类必须实现 _compute_loss 方法")

    def test(self, test_loader, show_plot=True, 
             save_results=True, config=None, use_best_model=True):
        """
        测试模型

        参数:
        test_loader: 测试数据加载器
        show_plot: 是否绘制预测结果图
        save_plots: 是否保存图表
        save_results: 是否保存测试结果到文件
        config: 测试配置
        use_best_model: 是否优先使用最优模型（如果存在）

        返回:
        测试结果字典
        """
        # 如果指定了使用最优模型且存在，则加载它
        if use_best_model and self.save_path:
            best_model_path = os.path.join(self.save_path, 'best_model.pth')
            if os.path.exists(best_model_path):
                self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
                print("已加载最优模型进行测试")
            else:
                print("未找到最优模型，使用最终模型进行测试")

        criterion = nn.MSELoss()

        self.model.eval()
        test_loss = 0.0
        predictions = []
        actuals = []
        sample_indices = []  # 仅存储样本索引

        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc='Testing')
            # 用于追踪每个 day_id 的段落计数（用以区分相同 day_id 的不同段落）
            day_segment_counter = {}
            
            for _ , batch_data in enumerate(test_pbar):
                pred, actual, mask, day_ids, loss = self._predict_and_evaluate(batch_data, criterion, config)
                test_loss += loss.item()

                # 转换为 numpy
                pred_np = pred.cpu().numpy()
                actual_np = actual.cpu().numpy()
                
                mask_np = mask.cpu().numpy()
                # 处理 mask 维度：可能为 [B, T, 1] 或 [B, T]
                # 如果最后一维为1（例如 [B, T, 1]），则去掉最后一维，保证 mask 与 pred 的时间维对齐
                # 采用基于最后一维是否为1的判断比比较 ndim 更稳健，能处理 (B,T,1) -> (B,T) 的情况
                if mask_np.ndim >= 1 and mask_np.shape[-1] == 1:
                    mask_np = mask_np.squeeze(-1)
                
                day_ids_np = day_ids.cpu().numpy() if day_ids is not None else None
                
                # 按样本遍历，只保留真实位置（mask==1）的数据
                B = pred_np.shape[0]
                for i in range(B):
                    valid = mask_np[i].astype(bool)
                    pred_i = pred_np[i][valid]
                    actual_i = actual_np[i][valid]
                    
                    predictions.extend(pred_i.flatten())
                    actuals.extend(actual_i.flatten())
                    
                    # 生成复合标识符：day_id_segment_id（用以区分相同 day_id 的不同段落）
                    if day_ids_np is not None:
                        day_id_val = day_ids_np[i]
                    else:
                        day_id_val = i
                    
                    # 追踪该 day_id 的段落计数
                    if day_id_val not in day_segment_counter:
                        day_segment_counter[day_id_val] = 0
                    seg_count = day_segment_counter[day_id_val]
                    day_segment_counter[day_id_val] += 1
                    
                    # 生成复合标识符：day_id_segment_id
                    composite_id = f"{int(day_id_val)}_{seg_count}"
                    sample_indices.extend([composite_id] * len(pred_i))

                test_pbar.set_postfix({'Batch Loss': f'{loss.item():.6f}'})

        test_loss /= len(test_loader)
        
        # 展平预测值和实际值
        predictions_flat = np.array(predictions).flatten()
        actuals_flat = np.array(actuals).flatten()
        sample_ids = np.array(sample_indices)

        # 计算评估指标
        mse = mean_squared_error(actuals_flat, predictions_flat)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals_flat, predictions_flat)
        r2 = r2_score(actuals_flat, predictions_flat)
        mape = mean_absolute_percentage_error(actuals_flat, predictions_flat)

        # 格式化输出结果
        self._print_test_results(test_loss, mse, rmse, mae, r2, mape)

        # 绘制预测vs实际值图
        self._plot_predictions(actuals_flat, predictions_flat, show_plot)

        # 绘制误差分布图
        self._plot_error_distribution(actuals_flat, predictions_flat, show_plot)
        
        # 保存测试结果到文件
        if save_results and self.save_path:
            self._save_test_results(test_loss, mse, rmse, mae, r2, mape, predictions_flat, actuals_flat, sample_ids)

        return {
            'test_loss': test_loss,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'predictions': predictions_flat,
            'actuals': actuals_flat
        }

    def _predict_and_evaluate(self, batch_data, criterion, config=None):
        """
        预测并评估单个批次，子类需要根据具体情况重写此方法

        参数:
        batch_data: 批次数据
        criterion: 损失函数
        config: 配置对象

        返回:
        一个包含五个元素的元组: `(pred, actual, mask, day_ids, loss)`。
        - `pred`: 模型预测值的 Tensor，形状通常为 [B, T] 或 [B, T, 1]
        - `actual`: 真实值的 Tensor，形状与 `pred` 对应
        - `mask`: 表示有效位置的掩码 Tensor，形状可能为 [B, T] 或 [B, T, 1]
        - `day_ids`: 每个样本对应的 day_id（可选），形状通常为 [B,]
        - `loss`: 本批次的损失（标量 Tensor）
        """
        raise NotImplementedError("子类必须实现 _predict_and_evaluate 方法")

    def _get_optimizer(self, config=None):
        """
        获取优化器
        
        参数:
        config: 配置对象
        
        返回:
        优化器实例
        """
        if config is None:
            return torch.optim.Adam(self.model.parameters(), lr=0.001)
            
        if not hasattr(config, 'optimizer_type') or config.optimizer_type == "adam":
            return torch.optim.Adam(
                self.model.parameters(), 
                lr=getattr(config, 'learning_rate', 0.001), 
                weight_decay=getattr(config, 'weight_decay', 0.0)
            )
        elif config.optimizer_type == "sgd":
            return torch.optim.SGD(
                self.model.parameters(), 
                lr=getattr(config, 'learning_rate', 0.001), 
                weight_decay=getattr(config, 'weight_decay', 0.0)
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {config.optimizer_type}")

    def _save_model(self):
        """保存模型"""
        if self.save_path:
            torch.save(self.model.state_dict(), os.path.join(self.save_path, 'model.pth'))

    def _save_test_results(self, test_loss, mse, rmse, mae, r2, mape, predictions, actuals, sample_ids=None):
        """
        保存测试结果到文件

        参数:
        test_loss: 测试损失
        mse: 均方误差
        rmse: 均方根误差
        mae: 平均绝对误差
        r2: 决定系数
        mape: 平均绝对百分比误差
        predictions: 预测值
        actuals: 实际值
        sample_ids: 样本标识符
        """
        # 保存指标到JSON文件（易读取和处理的格式）
        metrics_file = os.path.join(self.save_path, 'test_metrics.json')
        metrics_data = {
            'test_loss': float(test_loss),
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape)
        }
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=4, ensure_ascii=False)

        # 保存预测值和实际值到CSV文件
        csv_file = os.path.join(self.save_path, 'predictions_vs_actuals.csv')
        if sample_ids is not None:
            # 如果提供了样本ID，则包含在保存的数据中
            # 创建一个空的object类型数组来容纳混合数据类型
            data = np.empty((len(actuals), 3), dtype=object)
            data[:, 0] = actuals
            data[:, 1] = predictions
            data[:, 2] = sample_ids
            np.savetxt(csv_file, data,
                      delimiter=',', fmt=['%.6f', '%.6f', '%s'],
                      header='actual,predicted,sample_id', comments='')
        else:
            # 如果没有提供样本ID，使用原来的方式保存
            data = np.column_stack((actuals, predictions))
            np.savetxt(csv_file, data,
                       delimiter=',', fmt='%.6f',
                       header='actual,predicted', comments='')

    def _print_test_results(self, test_loss, mse, rmse, mae, r2, mape):
        """
        格式化打印测试结果
        """
        print('\n' + '=' * 50)
        print('模型测试结果')
        print('=' * 50)
        print(f'测试集平均损失:     {test_loss:.6f}')
        print(f'均方误差 (MSE):     {mse:.6f}')
        print(f'均方根误差 (RMSE):  {rmse:.6f}')
        print(f'平均绝对误差 (MAE): {mae:.6f}')
        print(f'平均绝对百分比误差 (MAPE): {mape:.6f}')
        print(f'决定系数 (R²):      {r2:.6f}')
        print('=' * 50)

    def _plot_predictions(self, actuals, predictions, show_plot=True):
        """
        绘制预测值与实际值的对比图
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(actuals, predictions, alpha=0.5)
        plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs Actual Values')
        plt.grid(True)
        if self.save_path:
            plt.savefig(os.path.join(self.save_path, 'prediction_vs_actual.png'), dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()

    def _plot_error_distribution(self, actuals, predictions, show_plot=True):
        """
        绘制误差分布图
        """
        errors = predictions - actuals
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Prediction Error Distribution')
        plt.grid(True)
        if self.save_path:
            plt.savefig(os.path.join(self.save_path, 'error_distribution.png'), dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()

    def plot_losses(self, show_plot=True):
        """
        绘制训练和验证损失曲线
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Loss Curve')
        plt.grid(True)
        if self.save_path:
            plt.savefig(os.path.join(self.save_path, 'training_loss.png'), dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()