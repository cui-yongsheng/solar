import matplotlib.pyplot as plt
from utils.PV_array import PVModule
import numpy as np
import scienceplots
plt.style.use(['science', 'no-latex', 'cjk-sc-font'])


def plot_pv_characteristics_temperature(module, temperature_list=None, irradiance=None):
    """
    绘制不同温度下的太阳能电池伏安特性曲线

    Parameters:
    module: PVModule 实例
    temperature_list: 温度列表，如果为None则使用默认值 [15, 25, 35]
    irradiance: 光照强度，如果为None则使用默认值 1000 W/m²
    """
    if temperature_list is None:
        temperature_list = [15, 25, 35]  # 默认温度列表 (°C)
    if irradiance is None:
        irradiance = 1000  # 默认光照强度 (W/m²)

    # 定义电压范围
    V_range = np.linspace(0, 48.24, 100)

    # 创建I-V特性曲线图
    plt.figure()
    for temp in temperature_list:
        I_values = []
        P_values = []
        for V in V_range:
            I, P = module.calculate(V=V, G=irradiance, T=temp, factor=1)
            I_values.append(I)
            P_values.append(P)

        plt.plot(V_range, I_values, label=f'T={temp}°C, G={irradiance}W/m²', linewidth=2)

    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (A)')
    plt.title('Different Temperatures')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 创建P-V特性曲线图
    plt.figure()
    for temp in temperature_list:
        I_values = []
        P_values = []
        for V in V_range:
            I, P = module.calculate(V=V, G=irradiance, T=temp, factor=1)
            I_values.append(I)
            P_values.append(P)

        plt.plot(V_range, P_values, label=f'T={temp}°C, G={irradiance}W/m²', linewidth=2)

    plt.xlabel('Voltage (V)')
    plt.ylabel('Power (W)')
    plt.title('Different Temperatures')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_pv_characteristics_irradiance(module, irradiance_list=None, temperature=None):
    """
    绘制不同光照强度下的太阳能电池伏安特性曲线

    Parameters:
    module: PVModule 实例
    irradiance_list: 光照强度列表，如果为None则使用默认值 [800, 1000, 1200]
    temperature: 温度，如果为None则使用默认值 25°C
    """
    if irradiance_list is None:
        irradiance_list = [800, 1000, 1200]  # 默认光照强度列表 (W/m²)
    if temperature is None:
        temperature = 25  # 默认温度 (°C)

    # 定义电压范围
    V_range = np.linspace(0, 48.24, 100)

    # 创建I-V特性曲线图
    plt.figure()
    for irr in irradiance_list:
        I_values = []
        P_values = []
        for V in V_range:
            I, P = module.calculate(V=V, G=irr, T=temperature, factor=1)
            I_values.append(I)
            P_values.append(P)

        plt.plot(V_range, I_values, '--', label=f'T={temperature}°C, G={irr}W/m²', linewidth=2)

    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (A)')
    plt.title('I-V Characteristic Curves (Different Irradiances)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 创建P-V特性曲线图
    plt.figure()
    for irr in irradiance_list:
        I_values = []
        P_values = []
        for V in V_range:
            I, P = module.calculate(V=V, G=irr, T=temperature, factor=1)
            I_values.append(I)
            P_values.append(P)

        plt.plot(V_range, P_values, '--', label=f'T={temperature}°C, G={irr}W/m²', linewidth=2)

    plt.xlabel('Voltage (V)')
    plt.ylabel('Power (W)')
    plt.title('P-V Characteristic Curves (Different Irradiances)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


# 在主程序中调用函数
if __name__ == '__main__':
    # 太阳能电池参数
    config = [
        {'count': 108, 'series': 18, 'area': 0.0604 * 0.0398}
    ]
    # 创建实例
    module = PVModule(config)

    # 绘制不同温度下的特性曲线
    plot_pv_characteristics_temperature(module, temperature_list=[10, 25, 50], irradiance=1366)

    # 绘制不同光照强度下的特性曲线
    plot_pv_characteristics_irradiance(module, irradiance_list=[1366*(1-0.04), 1366, 1366*(1+0.04)], temperature=25)
