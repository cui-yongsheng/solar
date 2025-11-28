from skyfield.api import load
from astropy.time import Time
import numpy as np

def quaternion_rotation_matrix(q):
    """
    将四元数转换为旋转矩阵
    :param q: 四元数 [w, x, y, z]
    :return: 3x3旋转矩阵
    """
    w, x, y, z = q
    # 归一化四元数（确保是单位四元数）
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    # 计算旋转矩阵
    rotation_matrix = np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]
    ])
    return rotation_matrix

def calculate_solar_power(df, ephemeris_file='de421.bsp'):
    """
    计算四块太阳能板的发电量比例因子

    :param df: 输入DataFrame，包含轨道位置、速度、时间和姿态四元数
    :param ephemeris_file: 星历文件路径，默认为'de421.bsp'
    :return: 添加了四块板发电量比例因子的DataFrame
    """
    # 确保输入数据包含必要的列
    required_columns = ["轨道位置X", "轨道位置Y", "轨道位置Z",
                        "轨道速度X", "轨道速度Y", "轨道速度Z",
                        "姿态四元数Q1", "姿态四元数Q2", "姿态四元数Q3", "姿态四元数Q4",
                        "时间"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")
    # 加载星历数据
    try:
        planets = load(ephemeris_file)
        sun = planets['sun']
        earth = planets['earth']
    except Exception as e:
        raise RuntimeError(f"加载星历数据失败: {e}")
    # 创建时间尺度对象
    ts = load.timescale()
    # 转换时间为Skyfield时间对象
    print("正在转换时间格式...")
    astropy_times = Time(df['时间'].values, scale='utc')
    skyfield_times = ts.from_astropy(astropy_times)
    # 提取向量数据
    # 轨道位置向量 (r)
    r = np.array([df["轨道位置X"], df["轨道位置Y"], df["轨道位置Z"]]).T
    # 轨道速度向量 (v)
    v = np.array([df["轨道速度X"], df["轨道速度Y"], df["轨道速度Z"]]).T / 1000.0  # 转换为km/s
    # 计算太阳矢量
    print("计算太阳矢量...")
    try:
        # 一次性计算所有时间点的太阳位置
        sun_positions = earth.at(skyfield_times).observe(sun).apparent()
        s = sun_positions.position.km.T  # 转置为(N, 3)形状
    except Exception as e:
        print(f"计算太阳矢量时出错: {e}")
        # 如果出错，则创建一个NaN数组
        s = np.full((len(df), 3), np.nan)
    # 定义四块太阳能板的法向量
    solar_panel_normals = [
        np.array([1, 0, 0]),  # 正X方向板
        np.array([-1, 0, 0]),  # 负X方向板
        np.array([0, 1, 0]),  # 正Y方向板
        np.array([0, -1, 0])  # 负Y方向板
    ]
    # 为每块板创建结果列
    df["正X板比例因子"] = 0.0
    df["负X板比例因子"] = 0.0
    df["正Y板比例因子"] = 0.0
    df["负Y板比例因子"] = 0.0
    # 逐行计算发电量
    print("计算太阳能板发电量比例因子...")
    for i in range(len(df)):
        # 获取当前行数据
        sat_pos = r[i]
        sat_vel = v[i]
        sun_pos = s[i]
        # 提取姿态四元数 (Q1, Q2, Q3, Q4)
        attitude_quaternion = np.array([
            df.at[i, "姿态四元数Q1"],
            df.at[i, "姿态四元数Q2"],
            df.at[i, "姿态四元数Q3"],
            df.at[i, "姿态四元数Q4"]
        ])
        # 计算卫星指向太阳的矢量 (GCRS坐标系)
        sun_vector_gcrs = sun_pos - sat_pos
        sun_vector_gcrs = sun_vector_gcrs / np.linalg.norm(sun_vector_gcrs)  # 归一化
        # 建立轨道坐标系
        # Z轴（径向）指向地心
        z_axis = -sat_pos / np.linalg.norm(sat_pos)
        # Y轴（轨道法向）垂直于轨道平面
        y_axis = np.cross(sat_vel, sat_pos)
        y_axis = y_axis / np.linalg.norm(y_axis)
        # X轴（切向）沿速度方向
        x_axis = sat_vel / np.linalg.norm(sat_vel)
        # 轨道坐标系到GCRS的旋转矩阵
        rotation_orbit_to_gcrs = np.array([x_axis, y_axis, z_axis]).T
        # 将太阳矢量从GCRS转换到轨道坐标系
        sun_vector_orbit = rotation_orbit_to_gcrs.T @ sun_vector_gcrs
        # 计算姿态矩阵（从轨道坐标系到本体坐标系）
        rotation_orbit_to_body = quaternion_rotation_matrix(attitude_quaternion)
        # 将太阳矢量从轨道坐标系转换到本体坐标系
        sun_vector_body = rotation_orbit_to_body @ sun_vector_orbit
        # 计算每块太阳能板的发电量比例因子
        cos_thetas = []
        for normal in solar_panel_normals:
            # 计算太阳矢量与太阳能板法向量的夹角余弦
            cos_theta = np.dot(sun_vector_body, normal)
            # 确保余弦值在有效范围内
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            cos_thetas.append(cos_theta)
        # 将结果存入DataFrame
        df.at[i, "正X板比例因子"] = cos_thetas[0]
        df.at[i, "负X板比例因子"] = cos_thetas[1]
        df.at[i, "正Y板比例因子"] = cos_thetas[2]
        df.at[i, "负Y板比例因子"] = cos_thetas[3]
    return df


def calculate_angles(df, ephemeris_file='de421.bsp'):
    """
    计算卫星的Alpha角和Beta角
    Alpha: 卫星速度方向垂直面和太阳方向夹角
    Beta: 卫星轨道平面和太阳夹角

    Args:
        df (pandas.DataFrame): 包含轨道位置、速度和时间的DataFrame
        ephemeris_file (str): 星历文件路径，默认为'de421.bsp'

    Returns:
        pandas.DataFrame: 添加了Alpha和Beta角的DataFrame
    """
    # 确保输入数据包含必要的列
    required_columns = ["轨道位置X", "轨道位置Y", "轨道位置Z",
                        "轨道速度X", "轨道速度Y", "轨道速度Z",
                        "时间"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")

    # 加载星历数据
    try:
        planets = load(ephemeris_file)
        sun = planets['sun']
        earth = planets['earth']
    except Exception as e:
        raise RuntimeError(f"加载星历数据失败: {e}")

    # 创建时间尺度对象
    ts = load.timescale()

    # 转换时间为Skyfield时间对象
    print("正在转换时间格式...")
    astropy_times = Time(df['时间'].values, scale='utc')
    skyfield_times = ts.from_astropy(astropy_times)

    # 提取向量数据
    # 轨道位置向量 (r)
    r = np.array([df["轨道位置X"], df["轨道位置Y"], df["轨道位置Z"]]).T

    # 轨道速度向量 (v)
    v = np.array([df["轨道速度X"], df["轨道速度Y"], df["轨道速度Z"]]).T/1000.0

    # 计算太阳矢量（向量化计算）
    print("计算太阳矢量...")
    try:
        # 一次性计算所有时间点的太阳位置
        sun_positions = earth.at(skyfield_times).observe(sun).apparent()
        s = sun_positions.position.km.T  # 转置为(N, 3)形状
    except Exception as e:
        print(f"计算太阳矢量时出错: {e}")
        # 如果出错，则创建一个NaN数组
        s = np.full((len(df), 3), np.nan)

    # 计算Alpha角: 速度方向法平面与太阳方向的夹角
    # cos(alpha) = (v · s) / (|v| * |s|)
    dot_v_s = np.sum(v * s, axis=1)
    norm_v = np.linalg.norm(v, axis=1)
    norm_s = np.linalg.norm(s, axis=1)

    # 避免除零错误
    alpha_cos = np.divide(dot_v_s, norm_v * norm_s,
                          out=np.zeros_like(dot_v_s), where=(norm_v * norm_s) != 0)

    # 限制cos值在[-1, 1]范围内，避免数值误差
    alpha_cos = np.clip(alpha_cos, -1.0, 1.0)
    alpha_rad = np.arcsin(alpha_cos)
    alpha_deg = np.degrees(alpha_rad)

    # 计算Beta角: 轨道平面与太阳矢量的夹角
    # 轨道平面法向量 h = r × v
    h = np.cross(r, v)

    # 归一化法向量
    norm_h = np.linalg.norm(h, axis=1)
    # 避免除零错误
    valid_norm = norm_h > 0
    h_unit = np.zeros_like(h)
    h_unit[valid_norm] = h[valid_norm] / norm_h[valid_norm, None]

    # 归一化太阳矢量
    norm_s = np.linalg.norm(s, axis=1)
    valid_norm_s = norm_s > 0
    s_unit = np.zeros_like(s)
    s_unit[valid_norm_s] = s[valid_norm_s] / norm_s[valid_norm_s, None]

    # 计算点积并限制范围
    dot_h_s = np.sum(h_unit * s_unit, axis=1)
    dot_h_s = np.clip(dot_h_s, -1.0, 1.0)

    # 计算Beta角（使用arcsin取绝对值）
    beta_rad = np.arcsin(dot_h_s)
    beta_deg = np.degrees(beta_rad)

    # 添加计算结果到DataFrame
    result_df = df.copy()
    result_df["Alpha角(度)"] = alpha_deg
    result_df["Beta角(度)"] = beta_deg
    print("计算完成")
    return result_df

# 如果需要单独调用
if __name__ == "__main__":
    pass
