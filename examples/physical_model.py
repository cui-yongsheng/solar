import matplotlib.pyplot as plt
from utils.PV_array import PVModule
import pandas as pd
import scienceplots
import numpy as np
plt.style.use(['science', 'no-latex', 'cjk-sc-font'])

if __name__ == '__main__':
    # 读取数据
    position_precise = pd.read_parquet('../data/position_precise.parquet')
    state_precise = pd.read_parquet('../data/state_precise.parquet')
    distance_precise = pd.read_parquet('../data/distance.parquet')
    # 筛选前面部分数据，基于时间数据筛选
    position_precise = position_precise[position_precise['时间'] < '20141021']
    state_precise = state_precise[state_precise['时间'] < '20141021']
    distance_precise = distance_precise[distance_precise['time'] < '20141021']


    distance = distance_precise['distance'].values
    # 计算光照强度
    average_solar_constant = 1366  # W/m²
    average_distance = np.mean(distance)
    irradiance = average_solar_constant * (average_distance / distance) ** 2
    # 太阳能板参数设置
    x_neg_config = [
        {'count': 36, 'series': 18, 'area': 0.0604 * 0.0398},
        {'count': 126, 'series': 18, 'area': 0.0403 * 0.0306}
    ]
    y_neg_config = [
        {'count': 108, 'series': 18, 'area': 0.0604 * 0.0398}
    ]
    x_pos_config = [
        {'count': 36, 'series': 18, 'area': 0.0604 * 0.0398},
        {'count': 126, 'series': 18, 'area': 0.0403 * 0.0306}
    ]
    y_pos_config = [
        {'count': 180, 'series': 18, 'area': 0.0403 * 0.0306}
    ]
    # 计算各方向电流
    x_neg_module = PVModule(x_neg_config)
    y_neg_module = PVModule(y_neg_config)
    x_pos_module = PVModule(x_pos_config)
    y_pos_module = PVModule(y_pos_config)

    I_y_neg, P_y_neg = y_neg_module.calculate(
        V=state_precise['28V母线电压-TTC采集'],
        G=irradiance,
        T=state_precise['帆板温度[-Y]'],
        factor=-position_precise['太阳矢量计算Y']
    )
    I_x_neg, P_x_neg = x_neg_module.calculate(
        V=state_precise['28V母线电压-TTC采集'],
        G=irradiance,
        T=state_precise['帆板温度[-X]'],
        factor=-position_precise['太阳矢量计算X']
    )
    I_x_pos, P_x_pos = x_pos_module.calculate(
        V=state_precise['28V母线电压-TTC采集'],
        G=irradiance,
        T=state_precise['帆板温度[+X]'],
        factor=position_precise['太阳矢量计算X']
    )
    I_y_pos, P_y_pos = y_pos_module.calculate(
        V=state_precise['28V母线电压-TTC采集'],
        G=irradiance,
        T=state_precise['帆板温度[+Y]'],
        factor=position_precise['太阳矢量计算Y']
    )
    I_total = (I_x_neg*1.2 + I_y_neg + I_x_pos*0.8+ I_y_pos) * 0.93
    # 计算误差
    error = I_total - state_precise['方阵电流']
    time_data = state_precise['时间'].values

    plt.figure()
    plt.plot(time_data, state_precise['方阵电流'], label='原始电流')
    plt.xlabel('时间')
    plt.ylabel('电流(A)')
    plt.legend(loc='upper right')
    plt.ylim(1.5, 3.5)
    plt.title('电流对比')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(time_data, I_total, label='估计电流')
    plt.xlabel('时间')
    plt.ylabel('电流(A)')
    plt.legend(loc='upper right')
    plt.ylim(1.5, 3.5)
    plt.title('电流对比')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(position_precise['时间'], I_x_neg, label='-x')
    plt.plot(position_precise['时间'], I_y_neg, label='-y')
    plt.plot(position_precise['时间'], I_x_pos, label='+x')
    plt.plot(position_precise['时间'], I_y_pos, label='+y')

    plt.xlabel('时间')
    plt.ylabel('Current (A)')
    plt.legend(loc='upper right')
    plt.show()

    # 电流误差绘制
    plt.figure()
    plt.plot(time_data, error, 'r-', label='估计误差', linewidth=1)
    plt.xlabel('时间')
    plt.ylabel('电流误差 (A)')
    plt.legend(loc='upper right')
    plt.ylim(-0.8, 0.8)
    plt.title('电流估计误差')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
