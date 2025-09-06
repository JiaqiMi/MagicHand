import pandas as pd
import numpy as np


data = pd.read_csv('./test_data/mpu9050_data_nomove_2.txt', header=None)
data.columns = ['gx', 'gy', 'gz', 'ax', 'ay', 'az', 'roll', 'pitch', 'yaw']
print(data[:2])

# gx, gy, gz = data[:, 0], data[:, 1], data[:, 2]
# ax, ay, az = data[:, 3], data[:, 4], data[:, 5]
# roll, pitch, yaw = data[:, 6], data[:, 7], data[:, 8]

def get_norm_mean(data: np.array) -> float:
    """剔除异常值后计算数据的均值

    :param data: The shape is N*1

    """
    mean = data.mean()
    std = data.std()

    threshold = 3
    filtered_data = data[np.abs((data - mean) / std) < threshold]

    filtered_data_mean = filtered_data.mean()

    return filtered_data_mean

gyro_x_bias = get_norm_mean(np.array(data.loc[:, ['gx']]))
gyro_y_bias = get_norm_mean(np.array(data.loc[:, ['gy']]))
gyro_z_bias = get_norm_mean(np.array(data.loc[:, ['gz']]))
print(f"gyro_x_bias: {gyro_x_bias}, gyro_y_bias: {gyro_y_bias}, gyro_z_bias: {gyro_z_bias}")



# 计算加速度计偏移（期望应为 [0, 0, 9.81]，实际值为 [ax, ay, az]*9.81）
accel_mean = np.mean(data.loc[:, ['ax', 'ay', 'az']], axis=0)
accel_expected = np.array([0, 0, 9.81])
accel_actual = accel_mean * 9.81
accel_bias = accel_actual - accel_expected

# print(f"gyro_bias: {gyro_bias}, accel_bias: {accel_bias}")

