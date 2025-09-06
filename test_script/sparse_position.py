import sys
sys.path.append('./../')
import numpy as np
import utils
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D绘图支持


data = pd.read_csv('./../test_data/magic_hand_static_data_20250627.txt')
data.columns = ['gx', 'gy', 'gz', 'ax', 'ay', 'az', 'roll', 'pitch', 'yaw']
data = data.dropna(how="any", axis=0)

# add time column
sample_rate = 50
num_samples = data.shape[0]
timestamp = np.linspace(0, (num_samples - 1) / sample_rate, num_samples)
data['time'] = timestamp

velocity, trajectory = utils.parse_position(data)

print(trajectory)

fig = plt.figure(figsize=(12, 8))
plt.subplot(321)
plt.plot(velocity[:, 0], label='vx')
plt.legend()

plt.subplot(323)
plt.plot(velocity[:, 1], label='vy')
plt.legend()

plt.subplot(325)
plt.plot(velocity[:, 2], label='vz')
plt.legend()

plt.subplot(322)
plt.plot(trajectory[:, 0], label='x')
plt.legend()

plt.subplot(324)
plt.plot(trajectory[:, 1], label='y')
plt.legend()

plt.subplot(326)
plt.plot(trajectory[:, 2], label='z')
plt.legend()

plt.show()



def plot_trajectory(position):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    x = position[:, 0]
    y = position[:, 1]
    z = position[:, 2]

    ax.plot(x, y, z, label='3D Trajectory', color='blue', linewidth=2)
    ax.scatter(x[0], y[0], z[0], color='green', s=50, label='Start')  # 起点
    ax.scatter(x[-1], y[-1], z[-1], color='red', s=50, label='End')   # 终点

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('3D Position Trajectory')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()

# 调用函数



# 方法二：借助欧拉角计算位移
#
# position2 = utils.compute_displacement(data)
# print(position2)
# plot_trajectory(position2)


