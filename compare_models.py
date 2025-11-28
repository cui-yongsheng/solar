import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def compare_models_time_series(folder_names, labels=None, sample_idx=None, show_plot=True):
    """
    对比多个模型的时间序列预测结果
    
    参数:
        folder_names: 实验文件夹名称列表
        labels: 模型标签列表，如果为None则使用文件夹名称
        sample_idx: 指定要绘制的样本索引，如果为None则绘制所有数据
        show_plot: 是否显示图形，默认为True
    """
    if labels is None:
        labels = folder_names
        
    # 确保标签数量与文件夹数量一致
    if len(labels) != len(folder_names):
        raise ValueError("标签数量必须与文件夹数量一致")
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # 为每个模型绘制时间序列
    colors = plt.cm.Set1(np.linspace(0, 1, len(folder_names) + 1))  # +1 为真实值留出颜色
    
    # 绘制真实值（从第一个模型中获取）
    if folder_names:
        first_folder_path = Path(folder_names[0])
        first_pred_files = list(first_folder_path.glob("predictions_vs_actuals.csv"))
        if first_pred_files:
            first_data = pd.read_csv(first_pred_files[0])
            
            if len(first_data.columns) >= 3 and sample_idx is not None:
                # 查找对应的样本数据
                sample_identifier = f"sample_{sample_idx}_"
                mask = first_data.iloc[:, 2].astype(str).str.contains(sample_identifier)
                filtered_data = first_data[mask]
                if len(filtered_data) > 0:
                    true_data = filtered_data.iloc[:, 0].values
                else:
                    # 如果找不到特定样本，使用原始方法
                    filtered_data = first_data[first_data.iloc[:, 2] == sample_idx]
                    true_data = filtered_data.iloc[:, 0].values
            else:
                # 否则使用所有数据
                true_data = first_data.iloc[:, 0].values
                
            ax.plot(true_data, label='True Values', color=colors[0], linewidth=2, alpha=0.8)
    
    # 绘制每个模型的预测值
    for i, (folder_name, label) in enumerate(zip(folder_names, labels)):
        folder_path = Path(folder_name)
        pred_files = list(folder_path.glob("predictions_vs_actuals.csv"))
        
        if not pred_files:
            print(f"警告: 在 {folder_name} 中未找到预测结果文件")
            continue
            
        data = pd.read_csv(pred_files[0])
        
        # 检查是否有样本索引列
        if len(data.columns) >= 3 and sample_idx is not None:
            # 查找对应的样本数据
            sample_identifier = f"sample_{sample_idx}_"
            mask = data.iloc[:, 2].astype(str).str.contains(sample_identifier)
            filtered_data = data[mask]
            if len(filtered_data) > 0:
                pred_data = filtered_data.iloc[:, 1].values
            else:
                # 如果找不到特定样本，使用原始方法
                filtered_data = data[data.iloc[:, 2] == sample_idx]
                pred_data = filtered_data.iloc[:, 1].values
        else:
            # 否则使用所有数据
            pred_data = data.iloc[:, 1].values
            
        ax.plot(pred_data, label=f'{label} Predicted', color=colors[i + 1], linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Values')
    title_suffix = f' (Sample {sample_idx})' if sample_idx is not None else ''
    ax.set_title(f'Model Time Series Comparison{title_suffix}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图形到fig目录
    fig_folder = Path("fig")
    fig_folder.mkdir(exist_ok=True)
    
    # 根据是否指定样本索引确定文件名
    if sample_idx is not None:
        plot_path = fig_folder / f"model_comparison_time_series_sample_{sample_idx}.png"
    else:
        plot_path = fig_folder / "model_comparison_time_series.png"
        
    plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
    print(f"模型对比时间序列图已保存到: {plot_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

# 使用示例
if __name__ == "__main__":
    # 指定要对比的模型文件夹
    model_folders = [
        "results/11/22-162359",
        "results/11/22-163555", 
        "results/11/22-170531"
    ]
    
    # 指定模型标签
    model_labels = [
        "PINN Model",
        "Neural Network",
        "Physics Only"
    ]
    
    # 对比所有数据的时间序列
    compare_models_time_series(model_folders, model_labels, sample_idx=None, show_plot=True)
    
    # 对比特定样本的时间序列（例如样本0）
    # compare_models_time_series(model_folders, model_labels, sample_idx=0, show_plot=True)