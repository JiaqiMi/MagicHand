from dataclasses import dataclass
from scipy.interpolate import interp1d
import imufusion
import numpy as np


def parse_position(data):
    """位移解算

    :param data:
    :return:
    """
    sample_rate = 50
    num_samples = len(data['time'])
    timestamp = np.linspace(0, (num_samples - 1) / sample_rate, num_samples)
    # timestamp = np.array(data['time'])
    gyroscope = np.zeros((num_samples, 3))
    accelerometer = np.zeros((num_samples, 3))

    for index in range(len(data['time'])):
        accelerometer[index] = np.array([data['ax'][index], data['ay'][index], data['az'][index]])
        gyroscope[index] = np.array([data['gx'][index], data['gy'][index], data['gz'][index]])

    # Instantiate AHRS algorithms
    offset = imufusion.Offset(sample_rate)
    ahrs = imufusion.Ahrs()

    ahrs.settings = imufusion.Settings(imufusion.CONVENTION_NWU,
                                       0.5,  # gain
                                       2000,  # gyroscope range
                                       10,  # acceleration rejection
                                       0,  # magnetic rejection
                                       5 * sample_rate)  # rejection timeout = 5 seconds

    # Process sensor data
    delta_time = np.diff(timestamp, prepend=timestamp[0])

    euler = np.empty((len(timestamp), 3))
    internal_states = np.empty((len(timestamp), 3))
    acceleration = np.empty((len(timestamp), 3))

    for index in range(len(timestamp)):
        gyroscope[index] = offset.update(gyroscope[index])

        ahrs.update_no_magnetometer(gyroscope[index], accelerometer[index], delta_time[index])

        euler[index] = ahrs.quaternion.to_euler()

        ahrs_internal_states = ahrs.internal_states
        internal_states[index] = np.array([ahrs_internal_states.acceleration_error,
                                              ahrs_internal_states.accelerometer_ignored,
                                              ahrs_internal_states.acceleration_recovery_trigger])

        acceleration[index] = 9.81 * ahrs.earth_acceleration  # convert g to m/s/s

    # Identify moving periods
    is_moving = np.empty(len(timestamp))

    for index in range(len(timestamp)):
        is_moving[index] = np.sqrt(acceleration[index].dot(acceleration[index])) > 3  # threshold = 3 m/s/s

    margin = int(0.1 * sample_rate)  # 100 ms

    for index in range(len(timestamp) - margin):
        is_moving[index] = any(is_moving[index:(index + margin)])  # add leading margin

    for index in range(len(timestamp) - 1, margin, -1):
        is_moving[index] = any(is_moving[(index - margin):index])  # add trailing margin

    # Calculate velocity (includes integral drift)
    velocity = np.zeros((len(timestamp), 3))

    for index in range(len(timestamp)):
        if is_moving[index]:  # only integrate if moving
            velocity[index] = velocity[index - 1] + delta_time[index] * acceleration[index]

    # Find start and stop indices of each moving period
    is_moving_diff = np.diff(is_moving, append=is_moving[-1])


    @dataclass
    class IsMovingPeriod:
        start_index: int = -1
        stop_index: int = -1


    is_moving_periods = []
    is_moving_period = IsMovingPeriod()

    for index in range(len(timestamp)):
        if is_moving_period.start_index == -1:
            if is_moving_diff[index] == 1:
                is_moving_period.start_index = index

        elif is_moving_period.stop_index == -1:
            if is_moving_diff[index] == -1:
                is_moving_period.stop_index = index
                is_moving_periods.append(is_moving_period)
                is_moving_period = IsMovingPeriod()

    # Remove integral drift from velocity
    velocity_drift = np.zeros((len(timestamp), 3))

    for is_moving_period in is_moving_periods:
        start_index = is_moving_period.start_index
        stop_index = is_moving_period.stop_index

        t = [timestamp[start_index], timestamp[stop_index]]
        x = [velocity[start_index, 0], velocity[stop_index, 0]]
        y = [velocity[start_index, 1], velocity[stop_index, 1]]
        z = [velocity[start_index, 2], velocity[stop_index, 2]]

        t_new = timestamp[start_index:(stop_index + 1)]

        velocity_drift[start_index:(stop_index + 1), 0] = interp1d(t, x)(t_new)
        velocity_drift[start_index:(stop_index + 1), 1] = interp1d(t, y)(t_new)
        velocity_drift[start_index:(stop_index + 1), 2] = interp1d(t, z)(t_new)

    velocity = velocity - velocity_drift

    # Calculate position
    position = np.zeros((len(timestamp), 3))

    for index in range(len(timestamp)):
        position[index] = position[index - 1] + delta_time[index] * velocity[index]

    # Print error as distance between start and final positions
    print("Error: " + "{:.3f}".format(np.sqrt(position[-1].dot(position[-1]))) + " m")

    return position


def interpolate_position_sequence(position: np.ndarray, M: int) -> np.ndarray:
    """
    将形状为 (N, 3) 的position数据展开并插值成长度为 M 的一维序列，输出形状为 (1, M)。

    参数:
        position (np.ndarray): 输入数组，形状为 (N, 3)，每行为 [x, y, z]。
        M (int): 目标序列长度。

    返回:
        np.ndarray: 插值后的序列，形状为 (1, M)。
    """
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("输入必须是形状为 (N, 3) 的数组")
    
    N = position.shape[0]

    # 步骤1：展平为 [x1, x2, ..., xn, y1, y2, ..., yn, z1, z2, ..., zn]
    flat_sequence = position.T.flatten()

    # 步骤2：原始序列索引（0 到 3N-1）
    original_indices = np.linspace(0, len(flat_sequence) - 1, num=len(flat_sequence))

    # 步骤3：目标索引（0 到 3N-1 上等间距的 M 个点）
    target_indices = np.linspace(0, len(flat_sequence) - 1, num=M)

    # 步骤4：插值
    interpolator = interp1d(original_indices, flat_sequence, kind='linear')
    interpolated_sequence = interpolator(target_indices)

    # 步骤5：reshape 为 (1, M)
    return interpolated_sequence.reshape(1, M)


def topN_indices(lst, N):
    """
    返回列表中数值大小前三的元素索引。

    :param lst: 一维数值列表。
    :param N: top N大小的数值对应的位置索引
    :return: list: 按值从大到小排序的前三个索引。
    """
    if len(lst) == 0:
        print("The length of the prediction is zero. Please check the model.")
        return []

    arr = np.array(lst)
    sorted_indices = np.argsort(arr)
    top_indices = sorted_indices[-N:][::-1]
    print(top_indices)

    return top_indices.tolist()
