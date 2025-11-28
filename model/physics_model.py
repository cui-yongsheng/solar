import torch
import torch.nn as nn
from utils.PV_array import PVModule

class BasePhysicsModel:
    """
    物理模型基类，包含通用的物理参数和配置
    """
    
    def __init__(self):
        """
        初始化物理模型参数和配置
        """
        # 固定物理参数
        self.solar_constant = 1366.0  # 太阳常数 (W/m²)
        self.sun_earth_distance = 149.6e6

        # 初始化光伏模块配置
        self._init_pv_configurations()
        
        # 初始化衰减参数
        self._init_degradation_params()
        
    def _init_pv_configurations(self):
        """
        初始化光伏模块配置
        """
        # X方向负配置
        self.x_neg_config = [
            {'count': 36, 'series': 18, 'area': 0.0604 * 0.0398},
            {'count': 126, 'series': 18, 'area': 0.0403 * 0.0306}
        ]
        # Y方向负配置
        self.y_neg_config = [
            {'count': 108, 'series': 18, 'area': 0.0604 * 0.0398}
        ]
        # X方向正配置
        self.x_pos_config = [
            {'count': 36, 'series': 18, 'area': 0.0604 * 0.0398},
            {'count': 126, 'series': 18, 'area': 0.0403 * 0.0306}
        ]
        # Y方向正配置
        self.y_pos_config = [
            {'count': 180, 'series': 18, 'area': 0.0403 * 0.0306}
        ]
        
    def _init_degradation_params(self):
        """
        初始化衰减参数 - 子类需要重写此方法
        """
        raise NotImplementedError("子类必须实现 _init_degradation_params 方法")
        
    def _compute_panel_currents(self, x_tensor, x_neg_factor=1.0, y_neg_factor=1.0,
                               x_pos_factor=1.0, y_pos_factor=1.0):
        """
        计算各方向太阳能板电流值
        
        参数:
        x_tensor: 输入特征tensor
        x_neg_factor: X负方向修正因子
        y_neg_factor: Y负方向修正因子
        x_pos_factor: X正方向修正因子
        y_pos_factor: Y正方向修正因子
        
        返回:
        各方向电流值元组 (I_x_neg, I_y_neg, I_x_pos, I_y_pos)
        """
        # 提取特征
        x_tensor = x_tensor.cpu()
        if len(x_tensor.shape) > 1:
            # 批量处理
            sun_x = x_tensor[:, 0]  # 太阳矢量计算X
            sun_y = x_tensor[:, 1]  # 太阳矢量计算Y
            temp_pos_x = x_tensor[:, 2]  # 帆板温度[+X]
            temp_neg_x = x_tensor[:, 3]  # 帆板温度[-X]
            temp_pos_y = x_tensor[:, 4]  # 帆板温度[+Y]
            temp_neg_y = x_tensor[:, 5]  # 帆板温度[-Y]
            voltage = x_tensor[:, 6]  # 28V母线电压-TTC采集
            distance = x_tensor[:, 7]  # distance
            year = x_tensor[:, 8]  # year
            month = x_tensor[:, 9]  # month
            day = x_tensor[:, 10]  # day
        else:
            # 单个样本处理
            sun_x = x_tensor[0]  # 太阳矢量计算X
            sun_y = x_tensor[1]  # 太阳矢量计算Y
            temp_pos_x = x_tensor[2]  # 帆板温度[+X]
            temp_neg_x = x_tensor[3]  # 帆板温度[-X]
            temp_pos_y = x_tensor[4]  # 帆板温度[+Y]
            temp_neg_y = x_tensor[5]  # 帆板温度[-Y]
            voltage = x_tensor[6]  # 28V母线电压-TTC采集
            distance = x_tensor[7]  # distance
            year = x_tensor[8]  # year
            month = x_tensor[9]  # month
            day = x_tensor[10]  # day

        irradiance = self.solar_constant * (self.sun_earth_distance / distance) ** 2

        # 创建光伏模块实例
        x_neg_module = PVModule(self.x_neg_config)
        y_neg_module = PVModule(self.y_neg_config)
        x_pos_module = PVModule(self.x_pos_config)
        y_pos_module = PVModule(self.y_pos_config)

        # 计算各方向电流
        I_y_neg, _ = y_neg_module.calculate(
            V=voltage,
            G=irradiance,
            T=temp_neg_y,
            factor=-sun_y
        )

        I_x_neg, _ = x_neg_module.calculate(
            V=voltage,
            G=irradiance,
            T=temp_neg_x,
            factor=-sun_x
        )

        I_x_pos, _ = x_pos_module.calculate(
            V=voltage,
            G=irradiance,
            T=temp_pos_x,
            factor=sun_x
        )

        I_y_pos, _ = y_pos_module.calculate(
            V=voltage,
            G=irradiance,
            T=temp_pos_y,
            factor=sun_y
        )

        # 应用方向修正因子
        I_x_neg *= x_neg_factor
        I_y_neg *= y_neg_factor
        I_x_pos *= x_pos_factor
        I_y_pos *= y_pos_factor
        
        return I_x_neg, I_y_neg, I_x_pos, I_y_pos


class PhysicsModel(BasePhysicsModel):
    """
    物理模型计算类，用于计算基于物理规律的太阳能电池板电流
    """

    def _init_degradation_params(self):
        """
        初始化衰减参数
        """
        # 设置初始时间和年化衰减率（可学习参数）
        self.annual_degradation_rate = nn.Parameter(torch.tensor(0.02))  # 年化衰减率(2%/年)
        # 各方向的独立衰减因子（可学习参数）
        self.x_neg_degradation_factor = nn.Parameter(torch.tensor(1.0))
        self.x_pos_degradation_factor = nn.Parameter(torch.tensor(1.0))
        self.y_neg_degradation_factor = nn.Parameter(torch.tensor(1.0))
        self.y_pos_degradation_factor = nn.Parameter(torch.tensor(1.0))

    def compute_degradation_factor(self, year, month, day):
        """
        计算衰减因子

        参数:
        year: 年份
        month: 月份
        day: 日期

        返回:
        衰减因子
        """
        # 计算相对于初始时间的时间差（以年为单位）
        time_diff_years = year - 2014

        # 使用指数衰减模型计算衰减因子
        # 确保时间差非负
        time_diff_years = torch.clamp(time_diff_years, min=0.0)

        # 计算总体衰减因子
        global_degradation = torch.exp(-self.annual_degradation_rate * time_diff_years)

        return global_degradation

    def compute_current(self, x, x_neg_factor=1.0, y_neg_factor=1.0,
                        x_pos_factor=1.0, y_pos_factor=1.0):
        """
        根据物理模型计算电流

        参数:
        x: 输入特征（推荐使用原始特征）
           特征顺序假设为: ['太阳矢量计算X', '太阳矢量计算Y',
                          '帆板温度[+X]', '帆板温度[-X]', '帆板温度[+Y]', '帆板温度[-Y]',
                          '28V母线电压-TTC采集', 'distance', 'year', 'month', 'day']
        x_neg_factor: X负方向修正因子
        y_neg_factor: Y负方向修正因子
        x_pos_factor: X正方向修正因子
        y_pos_factor: Y正方向修正因子

        返回:
        基于物理模型计算的电流值
        """
        # 确保输入是torch.Tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        # 保存设备信息
        device = x.device

        # 计算各方向电流
        I_x_neg, I_y_neg, I_x_pos, I_y_pos = self._compute_panel_currents(
            x, x_neg_factor, y_neg_factor, x_pos_factor, y_pos_factor)

        # 确保电流值是torch.Tensor
        I_x_neg = torch.tensor(I_x_neg, dtype=torch.float32, device=device)
        I_y_neg = torch.tensor(I_y_neg, dtype=torch.float32, device=device)
        I_x_pos = torch.tensor(I_x_pos, dtype=torch.float32, device=device)
        I_y_pos = torch.tensor(I_y_pos, dtype=torch.float32, device=device)

        # 提取特征用于衰减计算
        year = x[:, 8]
        month = x[:, 9]
        day = x[:, 10]

        # 计算衰减因子
        degradation_factor = self.compute_degradation_factor(year, month, day)

        # 确保所有参数在相同的设备上
        x_neg_degradation_factor = self.x_neg_degradation_factor.to(device)
        y_neg_degradation_factor = self.y_neg_degradation_factor.to(device)
        x_pos_degradation_factor = self.x_pos_degradation_factor.to(device)
        y_pos_degradation_factor = self.y_pos_degradation_factor.to(device)

        # 应用全局衰减因子和各方向独立衰减因子
        I_x_neg *= degradation_factor * x_neg_degradation_factor
        I_y_neg *= degradation_factor * y_neg_degradation_factor
        I_x_pos *= degradation_factor * x_pos_degradation_factor
        I_y_pos *= degradation_factor * y_pos_degradation_factor

        # 计算总电流
        I_total = (I_x_neg + I_y_neg + I_x_pos + I_y_pos) * 0.92

        # 确保输出形状正确
        if len(I_total.shape) == 1:
            I_total = I_total.reshape(-1, 1)

        return I_total


class SegmentPhysicsModel(BasePhysicsModel, nn.Module):
    """
    支持序列输入的物理模型计算类，用于计算基于物理规律的太阳能电池板电流
    支持输入格式: [batch, length, feature_size]
    """

    def _init_degradation_params(self):
        """
        初始化衰减参数
        """
        # 各方向的独立衰减因子（可学习参数）
        self.x_neg_degradation_factor = nn.Parameter(torch.tensor(1.0))
        self.x_pos_degradation_factor = nn.Parameter(torch.tensor(1.0))
        self.y_neg_degradation_factor = nn.Parameter(torch.tensor(1.0))
        self.y_pos_degradation_factor = nn.Parameter(torch.tensor(1.0))
        
    def __init__(self):
        nn.Module.__init__(self)
        BasePhysicsModel.__init__(self)

    def compute_current(self, x, x_neg_factor=1.0, y_neg_factor=1.0,
                        x_pos_factor=1.0, y_pos_factor=1.0):
        """
        根据物理模型计算电流，支持序列输入

        参数:
        x: 输入特征（推荐使用原始特征）
           特征顺序假设为: ['太阳矢量计算X', '太阳矢量计算Y',
                          '帆板温度[+X]', '帆板温度[-X]', '帆板温度[+Y]', '帆板温度[-Y]',
                          '28V母线电压-TTC采集', 'distance', 'year', 'month', 'day']
           输入格式：[batch, length, feature_size]

        x_neg_factor: X负方向修正因子
        y_neg_factor: Y负方向修正因子
        x_pos_factor: X正方向修正因子
        y_pos_factor: Y正方向修正因子

        返回:
        基于物理模型计算的电流值
        """
        # 保存设备信息
        device = x.device

        # 序列输入处理 [batch, length, feature]
        batch_size, sequence_length, feature_size = x.shape
        year = x[:, :, 8]  # year
        month = x[:, :, 9]  # month
        day = x[:, :, 10]  # day
        # reshape为二维以便处理: [batch*length, feature]
        year_flat = year.reshape(-1)
        month_flat = month.reshape(-1)
        day_flat = day.reshape(-1)
        x_flat = x.reshape(-1, feature_size)

        # 计算各方向电流
        I_x_neg_flat, I_y_neg_flat, I_x_pos_flat, I_y_pos_flat = self._compute_panel_currents(
            x_flat, x_neg_factor, y_neg_factor, x_pos_factor, y_pos_factor)

        # 将numpy数组转换为torch张量并移动到相同设备
        I_x_neg_flat = torch.tensor(I_x_neg_flat, dtype=torch.float32, device=device)
        I_y_neg_flat = torch.tensor(I_y_neg_flat, dtype=torch.float32, device=device)
        I_x_pos_flat = torch.tensor(I_x_pos_flat, dtype=torch.float32, device=device)
        I_y_pos_flat = torch.tensor(I_y_pos_flat, dtype=torch.float32, device=device)

        # 计算总电流
        I_total_flat = (I_x_neg_flat + I_y_neg_flat + I_x_pos_flat + I_y_pos_flat) * 0.92

        # 序列输入: 重塑为 [batch, length, 1]
        physics_current = I_total_flat.reshape(batch_size, sequence_length, 1)

        return physics_current