import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

try:
    import scienceplots
    plt.style.use(['science', 'no-latex'])
    SCIENCE_PLOTS_AVAILABLE = True
except ImportError:
    SCIENCE_PLOTS_AVAILABLE = False
    print("SciencePlots not available, using default matplotlib style")

class ModelVisualizer:
    def __init__(self, csv_path='results_summary.csv'):
        """
        初始化模型可视化器
        
        参数:
            csv_path: 结果CSV文件路径，默认为'results_summary.csv'
        """
        self.csv_path = csv_path
        self.results_df = pd.read_csv(self.csv_path)
        print(f"成功加载 {len(self.results_df)} 条实验记录")
            
    def find_model(self, model_type=None, **kwargs):
        filtered_df = self.results_df.copy()
        
        if model_type:
            filtered_df = filtered_df[filtered_df['model_type'] == model_type]
            
        for key, value in kwargs.items():
            if key in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[key] == value]
                
        return filtered_df
    
    def plot_model_performance(self, folder_name, show_plot=True, sample_idx=None):
        """
        绘制模型性能图并保存到fig目录
        
        参数:
            folder_name: 实验文件夹名称
            show_plot: 是否显示图形，默认为True
            sample_idx: 指定要绘制的样本索引，如果为None则绘制所有数据
        """
        record = self.results_df[self.results_df['folder_name'] == folder_name]
        
        # 获取预测数据
        actual_folder_path = Path(folder_name)
        pred_files = list(actual_folder_path.glob("predictions_vs_actuals.csv"))
        data = pd.read_csv(pred_files[0])
        
        # 检查是否有样本索引列
        if len(data.columns) >= 3 and sample_idx is not None:
            # 如果指定了样本索引，则只选择对应的数据
            filtered_data = data[data.iloc[:, 2] == sample_idx]
            true_data = filtered_data.iloc[:, 0].values
            pred_data = filtered_data.iloc[:, 1].values
        else:
            # 否则使用所有数据
            true_data = data.iloc[:, 0].values
            pred_data = data.iloc[:, 1].values
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Model Performance: {record.iloc[0]["model_type"]} ({folder_name})', fontsize=16)
        
        axes[0, 0].scatter(true_data, pred_data, alpha=0.5)
        axes[0, 0].plot([true_data.min(), true_data.max()], [true_data.min(), true_data.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('True Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('Predicted vs True Values')
        
        residuals = pred_data - true_data
        axes[0, 1].scatter(pred_data, residuals, alpha=0.5)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        
        # 在时间序列图中添加信息
        title_suffix = f" (Sample {sample_idx})" if sample_idx is not None and len(data.columns) >= 3 else ""
        axes[1, 0].plot(true_data[:min(1000, len(true_data))], label='True', alpha=0.7)
        axes[1, 0].plot(pred_data[:min(1000, len(pred_data))], label='Predicted', alpha=0.7)
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Values')
        axes[1, 0].set_title(f'Time Series Comparison (First 1000 points){title_suffix}')
        axes[1, 0].legend()
        
        axes[1, 1].hist(residuals, bins=50, alpha=0.7)
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Residual Distribution')
        
        plt.tight_layout()
        
        # 保存图形到fig目录
        fig_folder = Path("fig")
        fig_folder.mkdir(exist_ok=True)
        
        # 根据是否指定样本索引确定文件名
        if sample_idx is not None and len(data.columns) >= 3:
            plot_path = fig_folder / f"{folder_name}_performance_plot_sample_{sample_idx}.png"
        else:
            plot_path = fig_folder / f"{folder_name}_performance_plot.png"
            
        plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
        print(f"性能图已保存到: {plot_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
            
    def compare_models(self, model_types, show_plot=True):
        """
        比较多个模型的性能并保存到fig目录
        
        参数:
            model_types: 要比较的模型类型列表
            show_plot: 是否显示图形，默认为True
        """
        filtered_df = self.results_df[self.results_df['model_type'].isin(model_types)]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        metrics = ['r2', 'mse', 'rmse', 'mae']
        x = np.arange(len(model_types))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = []
            for model_type in model_types:
                model_record = filtered_df[filtered_df['model_type'] == model_type]
                values.append(model_record.iloc[0][metric])
                    
            ax.bar(x + i*width, values, width, label=metric)
            
        ax.set_xlabel('Model Types')
        ax.set_ylabel('Metric Values')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(model_types, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图形到fig目录
        fig_folder = Path("fig")
        fig_folder.mkdir(exist_ok=True)
        
        plot_path = fig_folder / "model_comparison.png"
        plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
        print(f"模型比较图已保存到: {plot_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
            
    def plot_time_series(self, folder_name, sample_idx=None, show_plot=True):
        """
        单独绘制真实值和预测值的时间序列曲线图
        
        参数:
            folder_name: 实验文件夹名称
            sample_idx: 指定要绘制的样本索引，如果为None则绘制所有数据
            show_plot: 是否显示图形，默认为True
        """
        # 获取预测数据
        actual_folder_path = Path(folder_name)
        pred_files = list(actual_folder_path.glob("predictions_vs_actuals.csv"))
        data = pd.read_csv(pred_files[0])
        
        # 检查是否有样本索引列
        if len(data.columns) >= 3 and sample_idx is not None:
            # 如果指定了样本索引，则只选择对应的数据
            filtered_data = data[data.iloc[:, 2] == sample_idx]
            true_data = filtered_data.iloc[:, 0].values
            pred_data = filtered_data.iloc[:, 1].values
        else:
            # 否则使用所有数据
            true_data = data.iloc[:, 0].values
            pred_data = data.iloc[:, 1].values
            
        # 创建图形
        fig, ax = plt.subplots(figsize=(15, 6))
        
        ax.plot(true_data, label='True Values', alpha=0.7)
        ax.plot(pred_data, label='Predicted Values', alpha=0.7)
        ax.set_xlabel('Time')
        ax.set_ylabel('Values')
        ax.set_title(f'Time Series Comparison - {folder_name}' + (f' (Sample {sample_idx})' if sample_idx is not None else ''))
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图形到fig目录
        fig_folder = Path("fig")
        fig_folder.mkdir(exist_ok=True)
        
        # 根据是否指定样本索引确定文件名
        if sample_idx is not None and len(data.columns) >= 3:
            plot_path = fig_folder / f"{folder_name}_time_series_sample_{sample_idx}.png"
        else:
            plot_path = fig_folder / f"{folder_name}_time_series.png"
            
        plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
        print(f"时间序列图已保存到: {plot_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()

if __name__ == "__main__":
    pass