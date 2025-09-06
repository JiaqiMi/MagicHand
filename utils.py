from dataclasses import dataclass
from scipy.interpolate import interp1d
import imufusion
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def clean_imu_signal(signal, window_size=11, sigma_thresh=3):
    """滑动窗口 3σ 异常值剔除，返回剔除后的信号"""
    cleaned = signal.copy()
    half_window = window_size // 2
    for i in range(half_window, len(signal) - half_window):
        window = signal[i - half_window:i + half_window + 1]
        mean = np.mean(window)
        std = np.std(window)
        if abs(signal[i] - mean) > sigma_thresh * std:
            # cleaned[i] = mean  # 或改为 window.median() / window[i-1] 插值
            cleaned[i] = np.median(window)  # 或改为 window.median() / window[i-1] 插值

    return cleaned


def parse_position(data):
    """位移解算
    :param data:
    :return:
    """
    sample_rate = 50        # 采样率
    num_samples = len(data['time'])
    timestamp = np.linspace(0, (num_samples - 1) / sample_rate, num_samples)        # 构建时间轴
    gyroscope = np.zeros((num_samples, 3))
    accelerometer = np.zeros((num_samples, 3))

    # Apply 3σ filtering to raw data
    ax = clean_imu_signal(np.array(data['ax']))
    ay = clean_imu_signal(np.array(data['ay']))
    az = clean_imu_signal(np.array(data['az']))
    gx = clean_imu_signal(np.array(data['gx']))
    gy = clean_imu_signal(np.array(data['gy']))
    gz = clean_imu_signal(np.array(data['gz']))

    fig = plt.figure(figsize=(18, 12))
    plt.subplot(231)
    plt.plot(ax, 'r--', label='filtered_ax')
    plt.plot(data['ax'], 'g', alpha=0.5, label='raw_ax')

    plt.subplot(232)
    plt.plot(ay, 'r--', label='filtered_ay')
    plt.plot(data['ay'], 'g', alpha=0.5, label='raw_ay')

    plt.subplot(233)
    plt.plot(az, 'r--', label='filtered_az')
    plt.plot(data['az'], 'g', alpha=0.5, label='raw_az')

    plt.subplot(234)
    plt.plot(gx, 'r--', label='filtered_gx')
    plt.plot(data['gx'], 'g', alpha=0.5, label='raw_gx')

    plt.subplot(235)
    plt.plot(gy, 'r--', label='filtered_gy')
    plt.plot(data['gy'], 'g', alpha=0.5, label='raw_gy')

    plt.subplot(236)
    plt.plot(gz, 'r--', label='filtered_gz')
    plt.plot(data['gz'], 'g', alpha=0.5, label='raw_gz')
    plt.legend()

    plt.show()

    for index in range(len(data['time'])):
        accelerometer[index] = np.array([ax[index], ay[index], az[index]])
        gyroscope[index] = np.array([gx[index], gy[index], gz[index]])

    # Instantiate AHRS algorithms
    offset = imufusion.Offset(sample_rate)
    ahrs = imufusion.Ahrs()            # 创建AHRS对象，配置姿态融合参数（如增益、陀螺范围、加速度拒绝阈值等）

    ahrs.settings = imufusion.Settings(imufusion.CONVENTION_NWU,
                                       0.1,  # gain
                                       2000,  # gyroscope range
                                       30,  # acceleration rejection
                                       0,  # magnetic rejection
                                       5 * sample_rate)  # rejection timeout = 5 seconds

    # Process sensor data
    delta_time = np.diff(timestamp, prepend=timestamp[0])

    euler = np.empty((len(timestamp), 3))
    internal_states = np.empty((len(timestamp), 3))
    acceleration = np.empty((len(timestamp), 3))

    # 逐时刻姿态更新，加速度修正
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
        is_moving[index] = np.sqrt(acceleration[index].dot(acceleration[index])) > 0.2  # threshold = 3 m/s/s
        # is_moving[index] = True
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

    # 找到所有开始移动和停止移动的时间段索引
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

    return velocity, position

def zupt_integration(data,
                     sample_rate=None,
                     init_static_duration=3.0,
                     hp_cutoff=0.05,
                     acc_std_window=0.5,
                     acc_std_thresh=0.02,
                     gyro_thresh=0.5,
                     static_margin=0.1,
                     min_static_duration=0.2,
                     accel_in_g_auto=True):
    """
    通用 ZUPT + 高通去偏 + 多段静止/运动 积分模板
    Inputs:
        data: dict with keys 'time','ax','ay','az','gx','gy','gz'
              time: sequence (seconds) OR None (then sample_rate required)
              ax,ay,az: accelerometer readings (either in g or m/s^2)
              gx,gy,gz: gyroscope readings (same units as your data, e.g. deg/s or rad/s)
        sample_rate: if None, will be estimated from data['time']
        init_static_duration: 前 N 秒作为初始静止窗口（s），用于估计重力偏置
        hp_cutoff: 高通滤波截止频率 (Hz) 用于去除低频偏置，常用 0.01-0.5
        acc_std_window: 用于静止检测的滑窗长度 (s)
        acc_std_thresh: 加速度滑窗 std 阈值 (m/s^2)
        gyro_thresh: 陀螺阈值 (same units as gx/gy/gz)
        static_margin: 静止段前后膨胀 (s)，避免断裂
        min_static_duration: 最小被认为是静止段的时长 (s)
        accel_in_g_auto: 如果 True，自动判断加速度是否为以 g 为单位 (<20 => g)
    Returns:
        velocity: (N,3) m/s
        position: (N,3) m
        static_mask: boolean array length N (True => 静止)
    Note: 该函数假设输入时间序列单调且无缺失。
    """

    # --- prepare time / dt / fs ---
    t = np.asarray(data['time'], dtype=float) if ('time' in data and data['time'] is not None) else None
    if t is None and sample_rate is None:
        raise ValueError("必须提供 data['time'] 或 sample_rate")
    if t is None:
        # synthesize timestamps
        N = len(data['ax'])
        dt = 1.0 / sample_rate
        t = np.linspace(0, (N - 1) * dt, N)
    else:
        t = t - t[0]
    dt_arr = np.diff(t, prepend=t[0])
    # avoid zero dt at start: set first dt to median of rest
    if len(dt_arr) > 1:
        med = np.median(dt_arr[1:])
        if med <= 0:
            raise ValueError("时间戳间隔异常")
        dt_arr[0] = med
    fs = 1.0 / np.median(dt_arr) if sample_rate is None else sample_rate

    # --- read sensors ---
    ax = np.asarray(data['ax'], dtype=float)
    ay = np.asarray(data['ay'], dtype=float)
    az = np.asarray(data['az'], dtype=float)
    gx = np.asarray(data['gx'], dtype=float)
    gy = np.asarray(data['gy'], dtype=float)
    gz = np.asarray(data['gz'], dtype=float)
    N = len(ax)
    assert len(ay) == N and len(az) == N, "加速度长度不一致"
    assert len(gx) == N and len(gy) == N and len(gz) == N, "陀螺长度不一致"

    acc = np.vstack([ax, ay, az]).T
    gyr = np.vstack([gx, gy, gz]).T

    # --- units: detect if accel in g ---
    if accel_in_g_auto:
        if np.max(np.abs(acc)) < 20:
            # likely in g
            acc = acc * 9.81

    # --- initial static: estimate gravity vector in sensor frame ---
    N0 = int(max(1, round(init_static_duration * fs)))
    if N0 >= N:
        N0 = int(max(1, N // 4))
    g0 = np.mean(acc[:N0], axis=0)  # gravity vector (sensor-frame)
    # remove gravity (assumes orientation 在 initial window 内基本不变)
    lin_acc = acc - g0  # 近似的线加速度（sensor frame）

    # --- high-pass filter (first-order, sample-varying) per axis ---
    # y[k] = alpha_k * (y[k-1] + x[k] - x[k-1]), alpha_k = rc / (rc + dt)
    fc = float(hp_cutoff)
    rc_arr = 1.0 / (2.0 * np.pi * fc) if fc > 0 else 1e9
    acc_hp = np.zeros_like(lin_acc)
    x_prev = lin_acc[0].copy()
    y_prev = np.zeros(3)
    for i in range(N):
        dt = dt_arr[i]
        alpha = rc_arr / (rc_arr + dt)
        x = lin_acc[i]
        y = alpha * (y_prev + x - x_prev)
        acc_hp[i] = y
        x_prev = x
        y_prev = y

    # --- moving-window std of magnitude for static detection ---
    win_len = int(max(1, round(acc_std_window * fs)))
    mag = np.linalg.norm(acc_hp, axis=1)
    # moving mean and mean of squares -> std
    kernel = np.ones(win_len) / win_len
    # pad to keep same length
    pad = win_len // 2
    mag_pad = np.pad(mag, (pad, pad), mode='edge')
    mean1 = np.convolve(mag_pad, kernel, mode='valid')
    mean2 = np.convolve(mag_pad**2, kernel, mode='valid')
    mag_std = np.sqrt(np.maximum(0.0, mean2 - mean1**2))

    gyro_norm = np.linalg.norm(gyr, axis=1)

    static_mask = (mag_std < acc_std_thresh) & (gyro_norm < gyro_thresh)

    # --- expand static regions by static_margin and remove short islands ---
    m_margin = int(round(static_margin * fs))
    if m_margin > 0:
        # leading/trailing expansion
        from_idx = np.where(static_mask)[0]
        if from_idx.size > 0:
            # dilation-like: for each True set neighbors True
            pad_mask = np.copy(static_mask)
            for i in range(1, m_margin + 1):
                pad_mask[:-i] = pad_mask[:-i] | static_mask[i:]
                pad_mask[i:] = pad_mask[i:] | static_mask[:-i]
            static_mask = pad_mask

    # remove static segments shorter than min_static_duration
    min_len = int(max(1, round(min_static_duration * fs)))
    # find runs
    def runs_from_mask(mask):
        runs = []
        i = 0
        N = len(mask)
        while i < N:
            if mask[i]:
                j = i
                while j < N and mask[j]:
                    j += 1
                runs.append((i, j))  # [i, j)
                i = j
            else:
                i += 1
        return runs
    static_runs = runs_from_mask(static_mask)
    for (s, e) in static_runs:
        if (e - s) < min_len:
            static_mask[s:e] = False

    # recompute static runs after cleaning
    static_runs = runs_from_mask(static_mask)

    # --- motion runs (intervals where static_mask==False) ---
    motion_mask = ~static_mask
    motion_runs = runs_from_mask(motion_mask)

    # --- allocate outputs ---
    velocity = np.zeros((N, 3), dtype=float)
    position = np.zeros((N, 3), dtype=float)

    # If everything static -> return zeros (简单快速路径)
    if len(motion_runs) == 0:
        return velocity, position, static_mask

    # --- For each motion run: integrate acceleration, then remove linear velocity drift so ends at zero ---
    # We'll assume velocity at the sample just before motion start is zero (ZUPT)
    for (ms, me) in motion_runs:
        # define integration indices: ms..me-1 (motion_mask True)
        idx = np.arange(ms, me)
        L = len(idx)
        if L == 0:
            continue

        # start velocity: often sample ms-1 is static -> v0 = 0, else use existing velocity
        if ms - 1 >= 0:
            v0 = velocity[ms - 1].copy()
        else:
            v0 = np.zeros(3)

        # integrate with trapezoidal rule over this motion run
        v_raw = np.zeros((L, 3))
        v_prev = v0.copy()
        for k in range(L):
            i = idx[k]
            if k == 0:
                dt_k = dt_arr[i]
                a_prev = acc_hp[i]  # we approximate a_prev==a for first step
                v_new = v_prev + a_prev * dt_k
            else:
                i_prev = idx[k - 1]
                dt_k = dt_arr[i]
                a_prev = acc_hp[i_prev]
                a_curr = acc_hp[i]
                v_new = v_prev + 0.5 * (a_prev + a_curr) * dt_k
            v_raw[k] = v_new
            v_prev = v_new

        # compute drift (assume velocity should be zero at end if next sample is static)
        if me < N and static_mask[me]:
            vend = v_raw[-1]
            # subtract linear ramp from 0 -> vend
            ramp = np.linspace(0.0, 1.0, L)[:, None] * vend[None, :]
            v_corr = v_raw - ramp
        else:
            # cannot force end to zero (no following static). instead subtract linear trend to nearest static if exists
            # try to find next static after me (rare), else do nothing
            # here simply remove mean drift to make average velocity zero (conservative)
            vend = v_raw[-1]
            ramp = np.linspace(0.0, 1.0, L)[:, None] * vend[None, :]
            v_corr = v_raw - ramp

        # write back into velocity array
        velocity[idx] = v_corr

        # ensure velocities at static boundaries are zero
        if ms - 1 >= 0:
            velocity[ms - 1] = np.zeros(3)
        if me < N:
            velocity[me] = np.zeros(3)

    # --- integrate velocity -> position (trapezoidal), and clamp pos constant on static intervals ---
    p_prev = np.zeros(3)
    for i in range(1, N):
        # trapezoidal integrate v[i-1] and v[i]
        position[i] = position[i - 1] + 0.5 * (velocity[i - 1] + velocity[i]) * dt_arr[i]

        # if this sample is static, keep position identical to previous (no drift accumulation)
        if static_mask[i]:
            position[i] = position[i - 1]

    # final safeguard: set all static-samples' velocities to zero
    velocity[static_mask] = 0.0

    return acc_hp, velocity, position, static_mask

def parse_position_v2(data):
    """位移解算函数
    根据静止数据消除了零偏干扰

    :param data:
    :return:
    """
    sample_rate = 50
    num_samples = len(data['time'])
    timestamp = np.linspace(0, (num_samples - 1) / sample_rate, num_samples)
    # timestamp = np.array(data['time'])
    gyro_bias = np.array([2.343397, -3.554945, 9.185590])  # °/s
    accel_bias = np.array([-0.667775, -0.056149, 0.474415])  # m/s²

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


def compute_displacement(data):
    """
    利用加速度和欧拉角进行位移解算
    :param acc: Nx3 numpy 数组，加速度 [ax, ay, az]（单位: g）
    :param euler_deg: Nx3 numpy 数组，欧拉角 [roll, pitch, yaw]（单位: 度）
    :param sample_rate: 采样率（Hz）
    :return: Nx3 位移数组 [x, y, z]（单位: 米）
    """
    sample_rate = 50  # 采样率
    num_samples = len(data['time'])
    timestamp = np.linspace(0, (num_samples - 1) / sample_rate, num_samples)  # 构建时间轴

    # Apply 3σ filtering to raw data
    ax = clean_imu_signal(np.array(data['ax']))
    ay = clean_imu_signal(np.array(data['ay']))
    az = clean_imu_signal(np.array(data['az']))
    roll = clean_imu_signal(np.array(data['roll']))
    pitch = clean_imu_signal(np.array(data['pitch']))
    yaw = clean_imu_signal(np.array(data['yaw']))

    dt = 1.0 / sample_rate

    # 初始化速度和位移
    velocity = np.zeros((num_samples, 3))
    position = np.zeros((num_samples, 3))

    euler_deg = np.zeros((num_samples, 3))
    acc = np.zeros((num_samples, 3))

    for index in range(len(data['time'])):
        euler_deg[index] = np.array([roll[index], pitch[index], yaw[index]])
        acc[index] = np.array([ax[index], ay[index], az[index]])

    earth_accs = np.zeros((num_samples, 3))
    for i in range(num_samples):
        # 当前欧拉角 → 旋转矩阵
        rot = R.from_euler('xyz', euler_deg[i], degrees=True)
        acc_earth = rot.apply(acc[i]) * 9.81  # g 转 m/s^2

        earth_accs[i] = np.array([acc_earth[0], acc_earth[1], acc_earth[2]])

        # # 减去重力在 z 轴方向（假设z向上）
        acc_earth[2] -= 9.81

        # 速度积分（矩形法）
        if i > 0:
            velocity[i] = velocity[i - 1] + acc_earth * dt
            position[i] = position[i - 1] + velocity[i] * dt

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


def topN_indices(lst: np.ndarray, N: int) -> list:
    """
    返回列表中数值大小前三的元素索引。

    :param lst: 一维数值列表。
    :param N: top N大小的数值对应的位置索引
    :return: list: 按值从大到小排序的前三个索引。
    """
    if not lst:
        return []

    arr = np.array(lst)
    sorted_indices = np.argsort(arr)
    top_indices = sorted_indices[-N:][::-1]

    return top_indices.tolist()
