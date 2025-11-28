import numpy as np
import math

class PVCell:
    """
    单个太阳能电池块模型
    """

    def __init__(self, area=1.0):
        """
        初始化单个太阳能电池参数
        """
        # 基本电气参数
        self.Voc = 2.68  # 单个电池开路电压 (V)
        self.Jsc = 16.8 / 1000 * 10000  # 短路电流密度 (A/m²)
        self.Impp = 16.1 / 1000 * 10000  # 最大功率点电流密度 (A/m²)
        self.Vmpp = 2.35  # 最大功率点电压 (V)

        # 电池物理参数
        self.area = area  # 电池面积 (m²)

        # 参考环境条件
        self.Gref = 1366  # 参考光照强度 (W/m^2)
        self.Tref = 25  # 参考温度 (°C)

        # 温度系数
        self.Kt = -6.45 / 1000  # 温度系数 (V/°C)
        self.Ks = 0.013 / 1000 * 10000  # 温度系数 (A/m²·°C)

    def calculate(self, V, G, T, factor=0):
        """
        计算单个电池在给定条件下的电流和功率

        参数:
        V : float or array-like - 电压 (V)
        G : float or array-like - 光照强度 (W/m^2)
        T : float or array-like - 温度 (°C)
        factor : float or array-like - 光照因子，比例系数cos(beta)

        返回:
        tuple: (电流 (A), 功率 (W))
        """
        # 转换为numpy数组以支持向量化操作
        V = np.asarray(V)
        G = np.asarray(G)
        T = np.asarray(T)
        factor = np.asarray(factor)

        # 广播所有输入数组到相同形状
        shapes = [np.shape(x) for x in [V, G, T, factor]]
        max_shape = np.broadcast_shapes(*shapes)

        V = np.broadcast_to(V, max_shape)
        G = np.broadcast_to(G, max_shape)
        T = np.broadcast_to(T, max_shape)
        factor = np.broadcast_to(factor, max_shape)
        factor = np.maximum(factor, 0)  # 防止负值

        # 实际光照强度
        G_actual = G * factor
        # 计算短路电流
        Isc = self.Jsc * self.area
        # 计算开路电压
        Voc = self.Voc
        # 计算最大功率点电流
        Impp = self.Impp * self.area
        # 计算最大功率点电压
        Vmpp = self.Vmpp
        # 计算电流修正系数
        D_I = ((self.Ks * (T - self.Tref) * self.area) / Isc + 1) * (G_actual / self.Gref)
        # 计算电压修正系数
        D_V = ((self.Kt * (T - self.Tref)) / Voc + 1) * np.log(np.e + 0.0002 * (G_actual - self.Gref))
        # 计算修正电流电压
        Isc = Isc * D_I
        Voc = Voc * D_V
        Impp = Impp * D_I
        Vmpp = Vmpp * D_V

        # 添加保护措施防止除零错误
        # 确保 Isc 不为零或接近零
        Isc = np.maximum(Isc, 1e-12)
        # 确保 Voc 不为零或接近零
        Voc = np.maximum(Voc, 1e-12)
        # 确保 Impp/Isc 不接近 1
        Impp_Isc_ratio = np.minimum(Impp / Isc, 0.999)

        # 更保守地确保 Impp/Isc 不等于 1，避免 log(0) 错误
        Impp_Isc_ratio = np.clip(Impp_Isc_ratio, 0.001, 0.999)

        # 计算中间变量C1，C2
        log_val = np.log(1 - Impp_Isc_ratio)
        # 更保守地防止 log(0) 导致的无穷大
        log_val = np.clip(log_val, -1000, -1e-12)

        numerator = (Vmpp / Voc - 1)
        # 防止分子为零
        numerator = np.where(np.abs(numerator) < 1e-12, 1e-12, numerator)
        C2 = numerator / log_val

        # 防止指数运算溢出
        exp_arg = -(Vmpp / (C2 * Voc))
        # 限制指数参数范围，防止溢出
        exp_arg = np.clip(exp_arg, -500, 500)
        C1 = (1 - Impp_Isc_ratio) * np.exp(exp_arg)

        # 计算输出电流
        # 防止指数运算溢出
        exponent = V / (C2 * Voc)
        # 更保守地设置上限防止溢出
        exponent = np.clip(exponent, -500, 500)
        exp_val = np.exp(exponent)
        # 防止指数值过大
        exp_val = np.clip(exp_val, 0, 1e12)
        I = Isc * (1 - C1 * (exp_val - 1))
        # 限制电流不会为负值且不会过大
        I = np.clip(I, 0, 1e1)
        P = V * I
        # 限制功率不会过大
        P = np.clip(P, 0, 1e1)
        return I, P


class PVModule:
    """
    太阳能电池板模型，支持多种不同面积的电池块组合
    """

    def __init__(self, config=None):
        """
        初始化电池板配置
        """
        if config is None:
            # 默认配置：35个标准电池串联
            config = [{'count': 35, 'series': 35, 'area': 1.0}]

        self.cell_groups = []

        # 根据配置创建电池组
        for group_config in config:
            count = group_config['count']
            series = group_config['series']
            area = group_config['area']

            # 计算并联组数
            parallel = count // series

            # 创建电池组
            cell_group = {
                'cell': PVCell(area=area),  # 使用指定面积创建电池实例
                'series': series,
                'parallel': parallel,
                'area': area
            }

            self.cell_groups.append(cell_group)

    def calculate(self, V, G, T, factor=0):
        """
        计算电池板在给定条件下的电流和功率

        参数:
        V : float or array-like - 电压 (V)
        G : float or array-like - 光照强度 (W/m^2)
        T : float or array-like - 温度 (°C)
        factor : float or array-like - 光照因子，比例系数cos(beta)

        返回:
        tuple: (电流 (A), 功率 (W))
        """
        # 转换为numpy数组以支持向量化操作
        V = np.asarray(V)
        G = np.asarray(G)
        T = np.asarray(T)
        factor = np.asarray(factor)

        # 广播所有输入数组到相同形状
        shapes = [np.shape(x) for x in [V, G, T, factor]]
        max_shape = np.broadcast_shapes(*shapes)

        V = np.broadcast_to(V, max_shape)
        G = np.broadcast_to(G, max_shape)
        T = np.broadcast_to(T, max_shape)
        factor = np.broadcast_to(factor, max_shape)

        # 总电流和功率初始化
        I_total = np.zeros_like(V)
        P_total = np.zeros_like(V)

        # 对每个电池组分别计算
        for cell_group in self.cell_groups:
            # 将输入电压分配给该组的串联电池
            V_group = V / cell_group['series']

            # 直接使用电池实例计算，无需修改面积属性
            I_group, P_group = cell_group['cell'].calculate(V_group, G, T, factor)

            # 累加到总电流和功率（考虑并联效应）
            I_total += I_group * cell_group['parallel']
            P_total += P_group * cell_group['parallel'] * cell_group['series']

        return I_total, P_total
