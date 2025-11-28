# 对原始数据分析，获取光照强度/温度/角度相同时的数据曲线
from matplotlib import pyplot as plt
from utils.PV_array import PVModule
import numpy as np
import pandas as pd
import scienceplots
plt.style.use(['science', 'no-latex', 'cjk-sc-font'])

def main():
    position_precise = pd.read_parquet('../data/position_precise.parquet')
    # 读取数据
    state_precise = pd.read_parquet('../data/state_precise.parquet')
    distance_precise = pd.read_parquet('../data/distance.parquet')
    # # 筛选前面部分数据，基于时间数据筛选
    # position_precise = position_precise[position_precise['时间'] < '20171021']
    # state_precise = state_precise[state_precise['时间'] < '20171021']
    # distance_precise = distance_precise[distance_precise['time'] < '20171021']
    # 获取距离数据
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
        G=1366,
        T=state_precise['帆板温度[-Y]'],
        factor=np.sin(np.radians(position_precise['Beta角(度)']))
    )
    I_x_neg, P_x_neg = x_neg_module.calculate(
        V=state_precise['28V母线电压-TTC采集'],
        G=1366,
        T=state_precise['帆板温度[-X]'],
        factor=np.sin(np.radians(-position_precise['Alpha角(度)']))
    )
    I_x_pos, P_x_pos = x_pos_module.calculate(
        V=state_precise['28V母线电压-TTC采集'],
        G=1366,
        T=state_precise['帆板温度[+X]'],
        factor=np.sin(np.radians(position_precise['Alpha角(度)']))
    )
    I_y_pos, P_y_pos = y_pos_module.calculate(
        V=state_precise['28V母线电压-TTC采集'],
        G=1366,
        T=state_precise['帆板温度[+Y]'],
        factor=np.sin(np.radians(-position_precise['Beta角(度)']))
    )
    I_total = (I_x_neg + I_y_neg*0.8 + I_x_pos + I_y_pos)*1.1
    # 计算误差
    error = I_total - state_precise['方阵电流']
    time_data = state_precise['时间'].values
    # 在计算完 I_total 后，将其添加到 state_precise 中
    state_precise = state_precise.copy()  # 避免修改原始数据
    state_precise['估计电流'] = I_total

    plt.figure(figsize=(12, 6))
    plt.plot(time_data, state_precise['方阵电流'], label='原始电流')
    plt.xlabel('时间')
    plt.ylabel('电流(A)')
    plt.legend(loc='upper right')
    plt.ylim(1.5, 3.5)
    plt.title('电流对比')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(time_data, I_total, label='估计电流')
    plt.xlabel('时间')
    plt.ylabel('电流(A)')
    plt.legend(loc='upper right')
    plt.ylim(1.5, 3.5)
    plt.title('电流对比')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(position_precise['时间'], I_x_neg, label='-x')
    plt.plot(position_precise['时间'], I_y_neg, label='-y')
    plt.plot(position_precise['时间'], I_x_pos, label='+x')
    plt.plot(position_precise['时间'], I_y_pos, label='+y')

    plt.xlabel('时间')
    plt.ylabel('Current (A)')
    plt.legend(loc='upper right')
    plt.show()

    # 电流误差绘制
    plt.figure(figsize=(12, 6))
    plt.plot(time_data, error, 'r-', label='估计误差', linewidth=1)
    plt.xlabel('时间')
    plt.ylabel('电流误差 (A)')
    plt.legend(loc='upper right')
    plt.ylim(-0.8, 0.8)
    plt.title('电流估计误差')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 然后进行分组统计
    segment_analysis = state_precise.groupby('segment').agg({
        '方阵电流': ['count', 'min', 'max', 'mean'],
        '估计电流': ['count', 'min', 'max', 'mean'],
        '时间': ['min'],
    }).round(3)

    # 提取多级列名
    segments = segment_analysis.index
    # 绘制散点图对比估计vs实际电流均值
    plt.figure(figsize=(12, 6))
    plt.plot(segment_analysis[('时间', 'min')], segment_analysis[('估计电流', 'mean')])
    plt.plot(segment_analysis[('时间', 'min')], segment_analysis[('方阵电流', 'mean')])
    plt.xlabel('时间')
    plt.ylabel('电流均值 (A)')
    plt.title('各Segment估计电流 vs 实际电流均值对比')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 添加日期列用于按天分组
    state_precise['日期'] = state_precise['时间'].dt.date

    # 按日期进行分组统计
    daily_analysis = state_precise.groupby('日期').agg({
        '方阵电流': ['count', 'min', 'max', 'mean'],
        '估计电流': ['count', 'min', 'max', 'mean'],
    }).round(3)


    # 绘制每日电流均值对比图
    plt.figure(figsize=(12, 6))
    dates = daily_analysis.index
    plt.plot(dates, daily_analysis[('方阵电流', 'min')], label='实际电流均值', marker='o')
    plt.plot(dates, daily_analysis[('估计电流', 'min')], label='估计电流均值', marker='s')
    plt.xlabel('日期')
    plt.ylabel('电流均值 (A)')
    plt.title('每日电流均值对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 绘制每日电流均值对比图
    plt.figure(figsize=(12, 6))
    dates = daily_analysis.index
    plt.plot(dates, daily_analysis[('方阵电流', 'mean')], label='实际电流均值', marker='o')
    plt.plot(dates, daily_analysis[('估计电流', 'mean')], label='估计电流均值', marker='s')
    plt.xlabel('日期')
    plt.ylabel('电流均值 (A)')
    plt.title('每日电流均值对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 绘制每日电流均值对比图
    plt.figure(figsize=(12, 6))
    dates = daily_analysis.index
    plt.plot(dates, daily_analysis[('方阵电流', 'max')], label='实际电流均值', marker='o')
    plt.plot(dates, daily_analysis[('估计电流', 'max')], label='估计电流均值', marker='s')
    plt.xlabel('日期')
    plt.ylabel('电流均值 (A)')
    plt.title('每日电流最大值对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 绘制每日电流最大对比图
    plt.figure(figsize=(12, 6))
    dates = daily_analysis.index
    plt.plot(dates, daily_analysis[('估计电流', 'max')]-daily_analysis[('方阵电流', 'max')], label='实际电流均值', marker='o')
    plt.xlabel('日期')
    plt.ylabel('电流均值 (A)')
    plt.title('每日电流最大值误差')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 绘制每日电流均值对比图
    plt.figure(figsize=(12, 6))
    dates = daily_analysis.index
    plt.plot(dates, daily_analysis[('估计电流', 'min')]-daily_analysis[('方阵电流', 'min')], label='实际电流均值', marker='o')
    plt.xlabel('日期')
    plt.ylabel('电流均值 (A)')
    plt.title('每日电流均值对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 绘制每日电流均值对比图
    plt.figure(figsize=(12, 6))
    dates = daily_analysis.index
    plt.plot(dates, daily_analysis[('估计电流', 'mean')]-daily_analysis[('方阵电流', 'mean')], label='实际电流均值', marker='o')
    plt.xlabel('日期')
    plt.ylabel('电流均值 (A)')
    plt.title('每日电流均值对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
