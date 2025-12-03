"""
运行主要的分段模型脚本
支持多种模型类型和配置组合
"""

import subprocess
import sys
import os
from typing import List, Dict, Any

class ModelRunner:
    """模型运行器，支持不同的模型配置"""
    
    def __init__(self):
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        
    def run_model(self, model_type: str, **kwargs) -> None:
        """
        运行指定类型的模型
        
        参数:
        - model_type: 模型类型
        - **kwargs: 其他参数
        """
        cmd = ['python', 'main_segment.py', '--model_type', model_type]
        
        # 添加其他参数
        for key, value in kwargs.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f'--{key}')
            else:
                cmd.append(f'--{key}')
                cmd.append(str(value))
        
        print(f"\n{'='*80}")
        print(f"运行命令: {' '.join(cmd)}")
        print(f"{'='*80}\n")
        
        result = subprocess.run(cmd, cwd=self.project_root)
        return result.returncode == 0
    
    def run_multiple_models(self, model_configs: List[Dict[str, Any]]) -> Dict[str, bool]:
        """
        依次运行多个模型
        
        参数:
        - model_configs: 模型配置列表，每个配置是一个字典
        
        返回:
        - 运行结果字典，key为模型类型，value为是否成功
        """
        results = {}
        
        for config in model_configs:
            model_type = config.pop('model_type')
            success = self.run_model(model_type, **config)
            results[model_type] = success
        
        return results
    
    def print_summary(self, results: Dict[str, bool]) -> None:
        """打印运行总结"""
        print(f"\n{'='*80}")
        print("运行结果总结:")
        print(f"{'='*80}")
        
        for model_type, success in results.items():
            status = "✓ 成功" if success else "✗ 失败"
            print(f"{model_type:30} {status}")
        
        total = len(results)
        successful = sum(1 for v in results.values() if v)
        print(f"\n总计: {successful}/{total} 模型运行成功")


def run_single_model(model_type: str, **kwargs) -> None:
    """
    快速运行单个模型
    
    使用方法:
    python run_main_segment.py --model segment_pinn --epochs 50 --lr 0.001
    """
    runner = ModelRunner()
    runner.run_model(model_type, **kwargs)


def run_preset_combinations() -> None:
    """运行预设的模型组合"""
    runner = ModelRunner()
    
    # 预设配置1: 快速测试 (所有模型, 少数epoch)
    quick_test_config = [
        {'model_type': 'segment_pinn', 'num_epochs': 10, 'learning_rate': 1e-3},
        {'model_type': 'segment', 'num_epochs': 10, 'learning_rate': 1e-3},
        {'model_type': 'gru_segment', 'num_epochs': 10, 'learning_rate': 1e-3},
        {'model_type': 'transformer_segment', 'num_epochs': 10, 'learning_rate': 1e-3},
    ]
    
    print("\n运行快速测试配置...")
    results = runner.run_multiple_models(quick_test_config)
    runner.print_summary(results)


def run_neural_ode_models() -> None:
    """运行所有神经ODE相关模型"""
    runner = ModelRunner()
    
    config = [
        {'model_type': 'neural_ode', 'num_epochs': 50, 'learning_rate': 1e-3},
        {'model_type': 'transformer_neural_ode', 'num_epochs': 50, 'learning_rate': 1e-3},
    ]
    
    print("\n运行神经ODE模型...")
    results = runner.run_multiple_models(config)
    runner.print_summary(results)


def run_all_deep_learning_models() -> None:
    """运行所有深度学习模型"""
    runner = ModelRunner()
    
    config = [
        {'model_type': 'segment_pinn', 'num_epochs': 50, 'learning_rate': 1e-3, 'physics_weight': 0.001},
        {'model_type': 'segment_pinn_nodegradation', 'num_epochs': 50, 'learning_rate': 1e-3, 'physics_weight': 0.001},
        {'model_type': 'segment', 'num_epochs': 50, 'learning_rate': 1e-3},
        {'model_type': 'simple_segment', 'num_epochs': 50, 'learning_rate': 1e-3},
        {'model_type': 'gru_segment', 'num_epochs': 50, 'learning_rate': 1e-3, 'gru_hidden_dim': 64, 'gru_layers': 2},
        {'model_type': 'cnn_lstm_segment', 'num_epochs': 50, 'learning_rate': 1e-3, 'cnn_channels': 32, 'lstm_hidden_dim': 64},
        {'model_type': 'transformer_segment', 'num_epochs': 50, 'learning_rate': 1e-3, 'transformer_d_model': 64, 'transformer_nhead': 8},
        {'model_type': 'neural_ode', 'num_epochs': 50, 'learning_rate': 1e-3},
        {'model_type': 'transformer_neural_ode', 'num_epochs': 50, 'learning_rate': 1e-3},
        {'model_type': 'physics_model', 'num_epochs': 50, 'learning_rate': 1e-3},
    ]
    
    print("\n运行所有深度学习模型...")
    results = runner.run_multiple_models(config)
    runner.print_summary(results)


def run_all_ml_models() -> None:
    """运行所有传统机器学习模型"""
    runner = ModelRunner()
    
    config = [
        {'model_type': 'ml_rf', 'num_epochs': 1},  # ML模型不使用epoch
        {'model_type': 'ml_svm', 'num_epochs': 1},
        {'model_type': 'ml_lr', 'num_epochs': 1},
    ]
    
    print("\n运行所有机器学习模型...")
    results = runner.run_multiple_models(config)
    runner.print_summary(results)


def run_comparison_study() -> None:
    """运行对比研究 - 不同模型在相同设置下的性能对比"""
    runner = ModelRunner()
    
    # 所有模型都使用相同的基础参数
    base_config = {
        'num_epochs': 30,
        'learning_rate': 1e-3,
        'seed': 42,
        'shuffle_data': False,  # 使用时间顺序分割
    }
    
    models = [
        'segment_pinn',
        'segment',
        'gru_segment',
        'transformer_segment',
        'neural_ode',
    ]
    
    config = [
        {**base_config, 'model_type': model}
        for model in models
    ]
    
    print("\n运行对比研究...")
    results = runner.run_multiple_models(config)
    runner.print_summary(results)


def run_hyperparameter_search() -> None:
    """运行超参数搜索"""
    runner = ModelRunner()
    
    # 测试不同的学习率
    learning_rates = [1e-4, 1e-3, 1e-2]
    config = [
        {'model_type': 'segment_pinn', 'num_epochs': 30, 'learning_rate': lr}
        for lr in learning_rates
    ]
    
    print("\n运行超参数搜索 (不同学习率)...")
    results = runner.run_multiple_models(config)
    runner.print_summary(results)


def show_usage():
    """显示使用说明"""
    usage_text = """
使用方法:

1. 运行单个模型:
   python run_main_segment.py --model segment_pinn --epochs 50 --lr 0.001

2. 运行预设配置 (快速测试):
   python run_main_segment.py --preset quick

3. 运行所有深度学习模型:
   python run_main_segment.py --preset all_dl

4. 运行所有机器学习模型:
   python run_main_segment.py --preset all_ml

5. 运行对比研究:
   python run_main_segment.py --preset comparison

6. 运行超参数搜索:
   python run_main_segment.py --preset hyper

7. 运行所有神经ODE模型:
   python run_main_segment.py --preset ode

可用的模型类型:
  - segment_pinn              : 物理信息神经网络
  - segment_pinn_nodegradation: 不带退化因子的PINN
  - segment                   : 基础分段模型
  - simple_segment            : 简化分段模型
  - gru_segment               : GRU分段模型
  - cnn_lstm_segment          : CNN-LSTM分段模型
  - transformer_segment       : Transformer分段模型
  - neural_ode                : 神经ODE模型
  - transformer_neural_ode    : Transformer神经ODE模型
  - physics_model             : 纯物理模型
  - ml_rf                     : 随机森林
  - ml_svm                    : 支持向量机
  - ml_lr                     : 线性回归

常见参数:
  --num_epochs          训练轮数 (默认: 50)
  --learning_rate       学习率 (默认: 1e-3)
  --train_batch_size    训练批次大小 (默认: 256)
  --val_batch_size      验证批次大小 (默认: 1024)
  --test_batch_size     测试批次大小 (默认: 4096)
  --seed               随机种子 (默认: 2)
  --few_shot           是否使用小样本训练
  --shuffle_data       是否随机打乱数据集
  --physics_weight     物理损失权重 (PINN模型, 默认: 0.001)
  --show_plot          是否显示绘图
    """
    print(usage_text)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        show_usage()
        sys.exit(1)
    
    if sys.argv[1] == '--preset':
        if len(sys.argv) < 3:
            print("错误: 必须指定预设名称")
            print("可用预设: quick, all_dl, all_ml, comparison, hyper, ode")
            sys.exit(1)
        
        preset = sys.argv[2]
        
        if preset == 'quick':
            run_preset_combinations()
        elif preset == 'all_dl':
            run_all_deep_learning_models()
        elif preset == 'all_ml':
            run_all_ml_models()
        elif preset == 'comparison':
            run_comparison_study()
        elif preset == 'hyper':
            run_hyperparameter_search()
        elif preset == 'ode':
            run_neural_ode_models()
        else:
            print(f"未知的预设: {preset}")
            print("可用预设: quick, all_dl, all_ml, comparison, hyper, ode")
            sys.exit(1)
    
    elif sys.argv[1] == '--help':
        show_usage()
    
    elif sys.argv[1] == '--model':
        if len(sys.argv) < 3:
            print("错误: 必须指定模型类型")
            sys.exit(1)
        
        model_type = sys.argv[2]
        
        # 构建其他参数
        kwargs = {}
        i = 3
        while i < len(sys.argv):
            if sys.argv[i].startswith('--'):
                key = sys.argv[i][2:]
                if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('--'):
                    value = sys.argv[i + 1]
                    # 尝试转换为适当的类型
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            if value.lower() in ('true', 'false'):
                                value = value.lower() == 'true'
                    kwargs[key] = value
                    i += 2
                else:
                    kwargs[key] = True
                    i += 1
            else:
                i += 1
        
        run_single_model(model_type, **kwargs)
    
    else:
        print(f"未知的命令: {sys.argv[1]}")
        show_usage()
        sys.exit(1)
