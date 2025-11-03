import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# 参数设置
fs = 50  # 采样率 Hz
t = np.linspace(0, 20, fs*20)  # 20秒数据

# 构造信号：低频重力泄露 + 高频运动加速度
# gravity_leak = 0.05 * np.sin(0.2 * np.pi * t)   # 假设重力泄露 (低频)
gravity_leak = 0.2
motion = 0.5 * np.sin(2 * np.pi * 1.0 * t)      # 运动 (1 Hz)
signal = gravity_leak + motion

# 设计高通滤波器
cutoff = 0.3  # 截止频率 Hz
b, a = butter(2, cutoff / (fs/2), btype='high')
signal_hp = filtfilt(b, a, signal)

# 速度计算 (积分)
dt = 1/fs
vel_raw = np.cumsum(signal) * dt
vel_hp = np.cumsum(signal_hp) * dt

# 绘制图像
fig, axs = plt.subplots(3, 1, figsize=(10, 8))

# 原始信号
axs[0].plot(t, signal, label="原始加速度 (运动+重力泄露)", color='blue')
axs[0].plot(t, motion, '--', label="真实运动分量", color='orange')
axs[0].set_ylabel("加速度 (g)")
axs[0].legend()
axs[0].set_title("高通滤波对重力泄露的抑制作用")

# 高通滤波后
axs[1].plot(t, signal_hp, label="高通滤波后加速度", color='green')
axs[1].plot(t, motion, '--', label="真实运动分量", color='orange')
axs[1].set_ylabel("加速度 (g)")
axs[1].legend()

# 积分结果 (速度漂移对比)
axs[2].plot(t, vel_raw, label="未滤波积分 (严重漂移)", color='red')
axs[2].plot(t, vel_hp, label="高通滤波后积分", color='green')
axs[2].set_ylabel("速度 (m/s)")
axs[2].set_xlabel("时间 (s)")
axs[2].legend()

plt.tight_layout()
plt.show()
