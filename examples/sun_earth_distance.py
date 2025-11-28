from utils.utils import sun_earth_distance
import matplotlib.pyplot as plt
import scienceplots
import pandas as pd
import numpy as np
plt.style.use(['science', 'no-latex', 'cjk-sc-font'])

if __name__ == '__main__':
    position_precise = pd.read_parquet('../data/position_precise.parquet')
    time = position_precise['时间'].values
    distance = sun_earth_distance(time)
    # 对数据进行保存
    distance_df = pd.DataFrame({'time': time, 'distance': distance})
    distance_df.to_parquet('../data/distance.parquet')

    # 计算各位置的光照强度
    average_solar_constant = 1366  # W/m² (标准太阳常数)
    average_distance = np.mean(distance)  # 计算平均日地距离
    irradiance = average_solar_constant * (average_distance / distance)**2
    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.plot(position_precise['时间'], irradiance)
    plt.xlabel('时间')
    plt.ylabel('光照强度 (W/m²)')
    plt.title('基于日地距离变化的光照强度')
    plt.show()