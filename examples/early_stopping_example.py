"""
早停功能使用示例
"""

import argparse
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def demonstrate_early_stopping_options():
    """
    展示早停功能的各种使用方式
    """
    print("早停功能使用示例")
    print("=" * 50)
    
    print("\n1. 默认使用早停功能:")
    print("   python main.py")
    print("   或")
    print("   python main_segment.py")
    
    print("\n2. 禁用早停功能:")
    print("   python main.py --early_stopping False")
    
    print("\n3. 调整早停容忍度为20个epoch:")
    print("   python main.py --patience 20")
    
    print("\n4. 调整最小改善阈值:")
    print("   python main.py --min_delta 1e-5")
    
    print("\n5. 禁用保存最佳模型:")
    print("   python main.py --save_best_model False")
    
    print("\n6. 组合使用多个早停参数:")
    print("   python main.py --early_stopping --patience 15 --min_delta 1e-5 --save_best_model")
    
    print("\n注意:")
    print("- 早停参数会自动从命令行参数传递到TrainingConfig")
    print("- 最佳模型将保存为'best_model.pth'")
    print("- 早停触发时会显示相关信息")

if __name__ == "__main__":
    demonstrate_early_stopping_options()