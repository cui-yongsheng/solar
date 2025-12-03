import argparse
import json
import time
import os
import sys

def get_args():
    """
    解析命令行参数
    
    返回:
    解析后的命令行参数对象
    """
    parser = argparse.ArgumentParser(description='Model Arguments')
    
    # 基础训练参数
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='学习率')
    parser.add_argument('--optimizer_type', type=str, default="adam",
                              choices=['adam', 'sgd'], help='优化器类型')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='权重衰减')
    
    # 数据集参数
    parser.add_argument('--few_shot', action='store_true', default=False, help='是否使用小样本训练')
    parser.add_argument('--shuffle_data', action='store_true', default=False, help='是否随机打乱数据集')
    
    # 交叉验证参数
    parser.add_argument('--cross_validation', action='store_true', default=True, help='启用时间序列交叉验证')
    parser.add_argument('--n_splits', type=int, default=5, help='交叉验证折数')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='segment_pinn',
                           choices=['simple', 'lstm', 'resnet', 'attention', 'autoencoder', 'pinn', 'segment_pinn', 'segment_pinn_nodegradation',
                                    'segment', 'simple_segment', 'gru_segment', 'cnn_lstm_segment', 'transformer_segment',
                                    'ml_rf', 'ml_svm', 'ml_lr', 'neural_ode', 'physics_model', 'transformer_neural_ode'],
                           help='模型类型: simple, lstm, resnet, attention, autoencoder, pinn, segment_pinn, segment_pinn_nodegradation'
                                'segment, simple_segment, gru_segment, cnn_lstm_segment, transformer_segment, '
                                'ml_rf, ml_svm, ml_lr, neural_ode, physics_model, transformer_neural_ode')

    # 早停参数
    parser.add_argument('--early_stopping', action='store_true', default=True, help='是否使用早停机制')
    parser.add_argument('--save_best_model', action='store_true', default=True, help='是否保存最佳模型')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--min_delta', type=float, default=1e-5, help='早停最小变化量')

    # 其他参数
    parser.add_argument('--verbose', action='store_true', default=False, help='是否显示结果')
    parser.add_argument('--train_batch_size', type=int, default=256, help='训练批次大小')
    parser.add_argument('--val_batch_size', type=int, default=1024, help='验证批次大小')
    parser.add_argument('--test_batch_size', type=int, default=4096, help='测试批次大小')
    parser.add_argument('--seed', type=int, default=2, help='随机种子')
    parser.add_argument('--show_plot', action='store_true', default=True, help='是否显示绘图')
    
    # 根据模型类型添加特定参数
    args, _ = parser.parse_known_args()
    
    # 添加模型特定参数
    add_model_specific_args(parser, args.model_type)
    
    try:
        arg = parser.parse_args()
    except SystemExit:
        # 处理参数解析错误
        raise ValueError("参数解析失败，请检查输入参数")
    
    save_path = get_save_path()
    arg.save_path = save_path
    save_args(arg)
    return arg

def add_model_specific_args(parser, model_type):
    """
    根据模型类型添加特定参数
    """
    # 模型特定参数
    if model_type in ['pinn', 'segment_pinn', 'segment_pinn_nodegradation']:
        parser.add_argument('--physics_weight', type=float, default=0.001, help='物理损失权重 (用于PINN模型)')
    
    if model_type in ['segment', 'segment_pinn', 'segment_pinn_nodegradation', 'simple_segment', 'gru_segment', 
                      'cnn_lstm_segment', 'transformer_segment', 'neural_ode', 'transformer_neural_ode', 'cnn_gru_neural_ode']:
        parser.add_argument('--lambda_smooth', type=float, default=0.001, help='平滑损失权重 (用于分段模型)')
        parser.add_argument('--lambda_monotonicity', type=float, default=0.001, help='单调性损失权重 (用于分段模型)')
        parser.add_argument('--health_dim', type=int, default=8, help='健康向量维度')

    # GRU模型参数
    if model_type in ['gru_segment']:
        parser.add_argument('--gru_hidden_dim', type=int, default=64, help='GRU隐藏层维度')
        parser.add_argument('--gru_layers', type=int, default=2, help='GRU层数')

    # CNN-LSTM模型参数
    if model_type in ['cnn_lstm_segment']:
        parser.add_argument('--cnn_channels', type=int, default=32, help='CNN通道数')
        parser.add_argument('--cnn_kernel_size', type=int, default=3, help='CNN卷积核大小')
        parser.add_argument('--lstm_hidden_dim', type=int, default=64, help='LSTM隐藏层维度')
        parser.add_argument('--lstm_layers', type=int, default=2, help='LSTM层数')
    
    # Neural ODE和特征融合参数
    if model_type in ['neural_ode', 'segment_pinn_nodegradation']:
        parser.add_argument('--lstm_hidden_dim', type=int, default=64, help='LSTM隐藏层维度')
        parser.add_argument('--lstm_layers', type=int, default=2, help='LSTM层数')
        parser.add_argument('--feat_hidden_dim', type=int, default=64, help='特征融合隐藏层维度')

    # Transformer Neural ODE参数
    if model_type in ['transformer_neural_ode']:
        parser.add_argument('--transformer_d_model', type=int, default=64, help='Transformer模型维度')
        parser.add_argument('--transformer_nhead', type=int, default=8, help='Transformer注意力头数')
        parser.add_argument('--transformer_num_layers', type=int, default=2, help='Transformer层数')
        parser.add_argument('--feat_hidden_dim', type=int, default=64, help='特征融合隐藏层维度')

    # CNN-GRU Neural ODE参数
    if model_type in ['cnn_gru_neural_ode']:
        parser.add_argument('--cnn_channels', type=int, default=32, help='CNN通道数')
        parser.add_argument('--cnn_kernel_size', type=int, default=3, help='CNN卷积核大小')
        parser.add_argument('--gru_hidden_dim', type=int, default=64, help='GRU隐藏层维度')
        parser.add_argument('--gru_layers', type=int, default=2, help='GRU层数')
        parser.add_argument('--feat_hidden_dim', type=int, default=64, help='特征融合隐藏层维度')

    # Transformer模型参数
    if model_type in ['transformer_segment']:
        parser.add_argument('--transformer_d_model', type=int, default=64, help='Transformer模型维度')
        parser.add_argument('--transformer_nhead', type=int, default=8, help='Transformer注意力头数')
        parser.add_argument('--transformer_num_layers', type=int, default=2, help='Transformer层数')
        parser.add_argument('--transformer_dim_feedforward', type=int, default=128, help='Transformer前馈网络维度')
        parser.add_argument('--transformer_dropout', type=float, default=0.1, help='Transformer Dropout率')

def save_args(arg):
    """
    将参数保存到文件
    
    参数:
    arg: 命令行参数对象
    """
    try:
        filename = os.path.join(arg.save_path, 'args.json')
        with open(filename, 'w') as f:
            json.dump(vars(arg), f, indent=4)
        if hasattr(arg, 'verbose') and arg.verbose:
            print(f"Arguments saved to {filename}")
    except Exception as e:
        print(f"Failed to save arguments: {e}", file=sys.stderr)

def get_save_path():
    """
    获取保存路径
    
    返回:
    保存路径字符串
    """
    try:
        # 月日建立一级目录，时分秒是另外一级目录
        save_path = os.path.join(
            os.getcwd(),
            "results",
            time.strftime("%m", time.localtime()),
            time.strftime("%d-%H%M%S", time.localtime())
        )
        os.makedirs(save_path, exist_ok=True)
        return save_path
    except Exception as e:
        raise RuntimeError(f"Failed to create save path: {e}")

if __name__ == '__main__':
    try:
        args = get_args()
        print(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)