import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import torch
import pickle
import os
import json


class BaseMLModel:
    """
    传统机器学习模型基类
    """
    def __init__(self):
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = None
        self.is_fitted = False

    def fit(self, X, y):
        """
        训练模型
        
        参数:
        X: 特征矩阵 (n_samples, n_features)
        y: 目标值 (n_samples,)
        """
        # 标准化特征和目标值
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # 训练模型
        self.model.fit(X_scaled, y_scaled)
        self.is_fitted = True

    def predict(self, X):
        """
        预测
        
        参数:
        X: 特征矩阵 (n_samples, n_features)
        
        返回:
        预测值 (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
            
        # 标准化特征
        X_scaled = self.scaler_X.transform(X)
        
        # 预测
        y_pred_scaled = self.model.predict(X_scaled)
        
        # 反标准化预测值
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        return y_pred

    def save_model(self, filepath):
        """
        保存模型
        
        参数:
        filepath: 保存路径
        """
        model_data = {
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'model': self.model,
            'is_fitted': self.is_fitted
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath):
        """
        加载模型
        
        参数:
        filepath: 模型路径
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.scaler_X = model_data['scaler_X']
        self.scaler_y = model_data['scaler_y']
        self.model = model_data['model']
        self.is_fitted = model_data['is_fitted']


class MLRandomForest(BaseMLModel):
    """
    随机森林回归模型
    """
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        super().__init__()
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )


class MLSVM(BaseMLModel):
    """
    支持向量机回归模型
    """
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        super().__init__()
        self.model = SVR(kernel=kernel, C=C, gamma=gamma)


class MLLinearRegression(BaseMLModel):
    """
    线性回归模型
    """
    def __init__(self):
        super().__init__()
        self.model = LinearRegression()


def evaluate_ml_model(model, X_test, y_test):
    """
    评估机器学习模型性能
    
    参数:
    model: 已训练的模型
    X_test: 测试特征
    y_test: 测试标签
    
    返回:
    包含各种评估指标的字典
    """
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算评估指标
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred))
    
    # 计算R2分数
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'test_loss': mse,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': y_pred
    }


def save_ml_test_results(save_path, test_results, y_test):
    """
    保存机器学习模型的测试结果
    
    参数:
    save_path: 保存路径
    test_results: 测试结果字典
    y_test: 测试集真实值
    """
    # 保存指标到JSON文件
    metrics_file = os.path.join(save_path, 'test_metrics.json')
    metrics_data = {
        'test_loss': float(test_results['test_loss']),
        'mse': float(test_results['mse']),
        'rmse': float(test_results['rmse']),
        'mae': float(test_results['mae']),
        'r2': float(test_results['r2'])
    }
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=4, ensure_ascii=False)

    # 保存预测值和实际值到CSV文件
    csv_file = os.path.join(save_path, 'predictions_vs_actuals.csv')
    predictions = test_results['predictions']
    np.savetxt(csv_file, np.column_stack((y_test, predictions)),
               delimiter=',', header='actual,predicted', comments='')


def convert_torch_dataset_to_numpy(dataset):
    """
    将PyTorch数据集转换为NumPy数组
    
    参数:
    dataset: PyTorch数据集
    
    返回:
    features: 特征数组
    labels: 标签数组
    """
    features_list = []
    labels_list = []
    
    for i in range(len(dataset)):
        data = dataset[i]
        if len(data) == 2:
            # 普通数据集 (features, label)
            features, label = data
        elif len(data) == 3:
            # 带day_id的数据集 (features, label, day_id)
            features, label, day_id = data
        elif len(data) == 4:
            # 带原始特征的数据集 (normalized_features, raw_features, label, day_id)
            features, raw_features, label, day_id = data
            # 对于机器学习模型，我们使用归一化后的特征
        else:
            raise ValueError(f"Unexpected data format with {len(data)} elements")
            
        # 对于SegmentDataset，特征是3D张量 [T, F]，需要展平为2D [T, F]
        # 标签也是2D张量 [T, 1]，需要展平为1D [T]
        if isinstance(features, torch.Tensor):
            features = features.numpy()
        if isinstance(label, torch.Tensor):
            label = label.numpy()
            
        # 展平特征和标签
        features = features.reshape(-1, features.shape[-1])  # [T, F]
        label = label.flatten()  # [T]
        
        features_list.append(features)
        labels_list.append(label)
    
    # 合并所有样本
    features = np.vstack(features_list)  # [N*T, F]
    labels = np.concatenate(labels_list)  # [N*T]
    
    return features, labels