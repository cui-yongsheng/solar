import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def compare_model_time_series(model_folders, sample_idx=None, show_plot=True, save_plot=True):
    """
    对比多个模型的时间序列预测结果
    
    参数:
        model_folders: 包含实验文件夹名称的列表
        sample_idx: 指定要绘制的样本索引，如果为None则绘制所有数据
        show_plot: 是否显示图形，默认为True
        save_plot: 是否保存图形，默认为True
    """
    # 存储所有模型的数据
    model_data = {}
    
    # 读取每个模型的预测数据
    for folder_name in model_folders:
        actual_folder_path = Path(folder_name)
        pred_files = list(actual_folder_path.glob("predictions_vs_actuals.csv"))
        
        if not pred_files:
            print(f"警告: 在 {folder_name} 中未找到预测结果文件")
            continue
            
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
            
        model_data[folder_name] = {
            'true': true_data,
            'pred': pred_data
        }
    
    if not model_data:
        print("错误: 没有找到任何有效的模型数据")
        return
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # 绘制真实值（只需要绘制一次）
    first_model = list(model_data.keys())[0]
    ax.plot(model_data[first_model]['true'], label='True Values', linewidth=2, color='black', alpha=0.8)
    
    # 为每个模型绘制预测值
    colors = plt.cm.Set1(np.linspace(0, 1, len(model_data)))
    for i, (folder_name, data) in enumerate(model_data.items()):
        ax.plot(data['pred'], label=f'{folder_name} Predictions', 
                linewidth=1.5, alpha=0.7, color=colors[i])
    
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Values')
    ax.set_title(f'Time Series Comparison Across Models' + 
                (f' (Sample {sample_idx})' if sample_idx is not None else ''))
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图形
    if save_plot:
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

def compare_model_metrics(model_folders):
    """
    对比多个模型的评估指标
    
    参数:
        model_folders: 包含实验文件夹名称的列表
    """
    # 存储所有模型的指标
    metrics_data = {}
    
    # 读取每个模型的指标数据
    for folder_name in model_folders:
        actual_folder_path = Path(folder_name)
        metric_files = list(actual_folder_path.glob("test_metrics.json"))
        
        if not metric_files:
            print(f"警告: 在 {folder_name} 中未找到评估指标文件")
            continue
            
        import json
        with open(metric_files[0], 'r') as f:
            metrics = json.load(f)
            
        metrics_data[folder_name] = metrics
    
    if not metrics_data:
        print("错误: 没有找到任何有效的模型指标数据")
        return
    
    # 创建指标对比图
    metrics_names = list(list(metrics_data.values())[0].keys())
    model_names = list(metrics_data.keys())
    
    # 过滤掉非数值指标
    numeric_metrics = []
    for metric in metrics_names:
        try:
            float(list(metrics_data.values())[0][metric])
            numeric_metrics.append(metric)
        except (ValueError, TypeError):
            continue
    
    if not numeric_metrics:
        print("错误: 没有找到任何数值型指标")
        return
    
    # 创建子图
    n_metrics = len(numeric_metrics)
    cols = min(3, n_metrics)
    rows = (n_metrics + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if n_metrics == 1:
        axes = [axes]
    elif rows > 1:
        axes = axes.flatten()
    
    # 为每个指标创建条形图
    for i, metric in enumerate(numeric_metrics):
        ax = axes[i] if i < len(axes) else None
        if ax is None:
            break
            
        values = [float(metrics_data[model][metric]) for model in model_names]
        
        bars = ax.bar(range(len(model_names)), values, 
                     color=plt.cm.Set1(np.linspace(0, 1, len(model_names))))
        ax.set_xlabel('Models')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Comparison')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        
        # 在条形图上添加数值标签
        for j, (bar, value) in enumerate(zip(bars, values)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                   f'{value:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 隐藏多余的子图
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # 保存图形
    fig_folder = Path("fig")
    fig_folder.mkdir(exist_ok=True)
    plot_path = fig_folder / "model_metrics_comparison.png"
    plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
    print(f"模型指标对比图已保存到: {plot_path}")
    
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 示例用法
    model_folders = [
        "results/11/22-162359",
        "results/11/22-163555", 
        "results/11/22-170531"
    ]
    
    # 对比时间序列
    compare_model_time_series(model_folders, sample_idx=None, show_plot=True)
    
    # 对比评估指标
    compare_model_metrics(model_folders)