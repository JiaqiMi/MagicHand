import numpy as np
import pandas as pd

def highpass_filter(x, fc, fs):
    """一阶高通滤波"""
    dt = 1.0 / fs
    RC = 1.0 / (2 * np.pi * fc)
    alpha = RC / (RC + dt)
    y = np.zeros_like(x)
    y_prev = 0.0
    x_prev = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * (y_prev + x[i] - x_prev)
        y_prev = y[i]
        x_prev = x[i]
    return y

def compute_e0_trajectory(data, fs=50, fc=0.3, static_samples=250):
    """
    在 e0 坐标系下计算轨迹
    data: 包含 time, ax, ay, az, gx, gy, gz
    fs: 采样率 Hz
    fc: 高通滤波截止频率 Hz
    static_samples: 静止样本数（用于零偏估计）
    """

    # 原始数据
    t = np.array(data['time'])
    acc = np.vstack([data['ax'], data['ay'], data['az']]).T
    gyr = np.vstack([data['gx'], data['gy'], data['gz']]).T * np.pi / 180.0  # deg/s -> rad/s

    # Step1: 静止段估计加速度零偏
    acc_static_mean = np.mean(acc[:static_samples], axis=0)
    g_norm = np.linalg.norm(acc_static_mean)
    scale = 9.81 / g_norm  # 缩放到 1g = 9.81 m/s²
    acc = acc * scale
    acc_bias = np.mean(acc[:static_samples], axis=0)  # 真实偏置
    acc = acc - acc_bias

    # Step2: 对加速度做高通滤波（逐轴）
    acc_hp = np.zeros_like(acc)
    for j in range(3):
        acc_hp[:, j] = highpass_filter(acc[:, j], fc=fc, fs=fs)

    # Step3: 积分得到速度
    vel = np.zeros_like(acc_hp)
    dt = 1.0 / fs
    for i in range(1, len(acc_hp)):
        vel[i] = vel[i - 1] + acc_hp[i] * dt

    # Step4: 积分得到位置
    pos = np.zeros_like(acc_hp)
    for i in range(1, len(vel)):
        pos[i] = pos[i - 1] + vel[i] * dt

    return acc_hp, vel, pos

# ========== 示例用法 ==========
if __name__ == "__main__":
    # 假设 data 是 pandas.DataFrame，包含列 time, ax, ay, az, gx, gy, gz
    data = pd.read_csv('./test_data/magic_hand_static_data_20250627.txt')
    data.columns = ['gx', 'gy', 'gz', 'ax', 'ay', 'az', 'roll', 'pitch', 'yaw']
    data = data.dropna(how="any", axis=0)
    data = data.loc[:50*15, :]
    # add time column
    sample_rate = 50
    num_samples = data.shape[0]
    timestamp = np.linspace(0, (num_samples - 1) / sample_rate, num_samples)
    data['time'] = timestamp

    acc_hp, vel, pos = compute_e0_trajectory(data)

    print("Final position:", pos[-1])
