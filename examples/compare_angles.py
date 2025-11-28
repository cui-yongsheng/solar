import pandas as pd
from matplotlib import pyplot as plt
from utils.angle_calculator import calculate_solar_power
import scienceplots
import numpy as np
plt.style.use(['science', 'no-latex', 'cjk-sc-font'])


if __name__ == '__main__':
    # 读取数据
    position_precise = pd.read_parquet('../data/position_precise.parquet')
    attitude_precise = pd.read_parquet('../data/attitude_precise.parquet')
    power_precise = pd.read_parquet('../data/power_precise.parquet')

    # 绘制beta角增益因子
    plt.figure()
    plt.plot(position_precise['时间'], np.sin(np.radians(position_precise['Beta角(度)'])), label='Beta角增益因子')
    plt.plot(position_precise['时间'], -position_precise['太阳矢量计算Y'], label='太阳矢量计算Y')
    plt.plot(position_precise['时间'], power_precise['负Y板比例因子'], label='Beta角矫正')
    plt.xlabel('时间')
    plt.ylabel('增益因子')
    plt.legend()
    plt.title('-Y面太阳能板')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 绘制beta角增益因子
    plt.figure()
    plt.plot(position_precise['时间'], np.sin(np.radians(position_precise['Beta角(度)']))-power_precise['负Y板比例因子'], label='增益因子差')
    plt.xlabel('时间')
    plt.ylabel('增益因子')
    plt.legend()
    plt.title('-Y面太阳能板')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


    # 绘制alpha角增益因子
    plt.figure()
    plt.plot(position_precise['时间'], np.sin(np.radians(position_precise['Alpha角(度)'])), label='Alpha角增益因子')
    plt.plot(position_precise['时间'], position_precise['太阳矢量计算X'], label='太阳矢量计算X')
    plt.plot(position_precise['时间'], power_precise['正X板比例因子'], label='Alpha角矫正')
    plt.xlabel('时间')
    plt.ylabel('增益因子')
    plt.legend()
    plt.title('X面太阳能板')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 绘制alpha角增益因子
    plt.figure()
    plt.plot(position_precise['时间'], np.sin(np.radians(position_precise['Alpha角(度)'])) - power_precise['正X板比例因子'], label='增益因子差')
    plt.xlabel('时间')
    plt.ylabel('增益因子')
    plt.legend()
    plt.title('X面太阳能板')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()