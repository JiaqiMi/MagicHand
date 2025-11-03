from dataclasses import dataclass
from matplotlib import animation, pyplot as plt
from scipy.interpolate import interp1d
import imufusion
import matplotlib.pyplot as pyplot
import numpy
import pandas as pd

# Import sensor data ("short_walk.csv" or "long_walk.csv")
data = pd.read_csv('./test_data/MRU20251025/20251025_161900.txt', header=None, sep=r'\s+')
data.columns = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'index']

sample_rate = 200  # 400 Hz

# 如果没有 timestamp 列，则根据 sample_rate 自动生成
if data.shape[1] <= 9:
    num_samples = data.shape[0]
    timestamp = numpy.linspace(0, (num_samples - 1) / sample_rate, num_samples)  # 从 0 开始，按照采样率生成时间戳
else:
    timestamp = data[:, 0]

print("data size: ", data.shape[0])
data = data.dropna(how="any", axis=0)

data['ax'] = data['ax'].map(lambda x: float(x))
data['ay'] = data['ay'].map(lambda x: float(x))
data['az'] = data['az'].map(lambda x: float(x))
data['gx'] = data['gx'].map(lambda x: float(x)/3600)
data['gy'] = data['gy'].map(lambda x: float(x)/3600)
data['gz'] = data['gz'].map(lambda x: float(x)/3600)
print(data.info())

gyroscope = data.loc[:, ['gx', 'gy', 'gz']].to_numpy()
accelerometer = data.loc[:, ['ax', 'ay', 'az']].to_numpy()

# plot acceleration and gyroscope data
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(timestamp, accelerometer[:, 0], label='Acc X')
plt.plot(timestamp, accelerometer[:, 1], label='Acc Y')
plt.plot(timestamp, accelerometer[:, 2], label='Acc Z')
plt.title('Accelerometer Data')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(timestamp, gyroscope[:, 0], label='Gyro X')
plt.plot(timestamp, gyroscope[:, 1], label='Gyro Y')
plt.plot(timestamp, gyroscope[:, 2], label='Gyro Z')
plt.title('Gyroscope Data')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (°/s)')
plt.legend()
plt.tight_layout()
plt.show()

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
delta_time = numpy.diff(timestamp, prepend=timestamp[0])

euler = numpy.empty((len(timestamp), 3))
internal_states = numpy.empty((len(timestamp), 3))
acceleration = numpy.empty((len(timestamp), 3))

for index in range(len(timestamp)):
    gyroscope[index] = offset.update(gyroscope[index])

    ahrs.update_no_magnetometer(gyroscope[index], accelerometer[index], delta_time[index])

    euler[index] = ahrs.quaternion.to_euler()

    ahrs_internal_states = ahrs.internal_states
    internal_states[index] = numpy.array([ahrs_internal_states.acceleration_error,
                                          ahrs_internal_states.accelerometer_ignored,
                                          ahrs_internal_states.acceleration_recovery_trigger])

    # acceleration[index] = 9.81 * ahrs.earth_acceleration  # convert g to m/s/s
    acceleration[index] = 1 * ahrs.earth_acceleration  # convert g to m/s/s



# Identify moving periods
is_moving = numpy.empty(len(timestamp))

for index in range(len(timestamp)):
    # print(acceleration[index], acceleration[index].dot(acceleration[index]), numpy.sqrt(acceleration[index].dot(acceleration[index])))
    is_moving[index] = numpy.sqrt(acceleration[index].dot(acceleration[index])) > 0  # threshold = 3 m/s/s
    
plt.figure(figsize=(12, 3))
plt.plot(timestamp, is_moving, label='Is Moving')
plt.title('Is Moving Detection')
plt.xlabel('Time (s)')
plt.ylabel('Is Moving (1=True, 0=False)')
plt.ylim(-0.1, 1.1)
plt.legend()
plt.tight_layout()
plt.show()

margin = int(0.1 * sample_rate)  # 0.1 second margin

for index in range(len(timestamp) - margin):
    is_moving[index] = any(is_moving[index:(index + margin)])  # add leading margin

for index in range(len(timestamp) - 1, margin, -1):
    is_moving[index] = any(is_moving[(index - margin):index])  # add trailing margin

# Plot is_moving
plt.figure(figsize=(12, 3))
plt.plot(timestamp, is_moving, label='Is Moving')
plt.title('Is Moving Detection')
plt.xlabel('Time (s)')
plt.ylabel('Is Moving (1=True, 0=False)')
plt.ylim(-0.1, 1.1)
plt.legend()
plt.tight_layout()
plt.show()

# Calculate velocity (includes integral drift)
velocity = numpy.zeros((len(timestamp), 3))

for index in range(len(timestamp)):
    if is_moving[index]:  # only integrate if moving
        velocity[index] = velocity[index - 1] + delta_time[index] * acceleration[index]

# Find start and stop indices of each moving period
is_moving_diff = numpy.diff(is_moving, append=is_moving[-1])


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
velocity_drift = numpy.zeros((len(timestamp), 3))

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
position = numpy.zeros((len(timestamp), 3))

for index in range(len(timestamp)):
    position[index] = position[index - 1] + delta_time[index] * velocity[index]

# plot 3D position
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(position[:, 0], position[:, 1], position[:, 2], label='3D Position')
ax.set_title('3D Position Plot')
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Z Position (m)')
ax.legend()
plt.show()
# plot 2D position (top-down view)
plt.figure()
plt.scatter(position[:, 0], position[:, 1], c='r')
plt.show()

# Print error as distance between start and final positions
print("Error: " + "{:.3f}".format(numpy.sqrt(position[-1].dot(position[-1]))) + " m")

# Create 3D animation (takes a long time, set to False to skip)
# if False:
#     figure = pyplot.figure(figsize=(10, 10))
#
#     axes = pyplot.axes(projection="3d")
#     axes.set_xlabel("m")
#     axes.set_ylabel("m")
#     axes.set_zlabel("m")
#
#     x = []
#     y = []
#     z = []
#
#     scatter = axes.scatter(x, y, z)
#
#     fps = 30
#     samples_per_frame = int(sample_rate / fps)
#
#     def update(frame):
#         index = frame * samples_per_frame
#
#         axes.set_title("{:.3f}".format(timestamp[index]) + " s")
#
#         x.append(position[index, 0])
#         y.append(position[index, 1])
#         z.append(position[index, 2])
#
#         scatter._offsets3d = (x, y, z)
#
#         if (min(x) != max(x)) and (min(y) != max(y)) and (min(z) != max(z)):
#             axes.set_xlim3d(min(x), max(x))
#             axes.set_ylim3d(min(y), max(y))
#             axes.set_zlim3d(min(z), max(z))
#
#             axes.set_box_aspect((numpy.ptp(x), numpy.ptp(y), numpy.ptp(z)))
#
#         return scatter
#
#     anim = animation.FuncAnimation(figure, update,
#                                    frames=int(len(timestamp) / samples_per_frame),
#                                    interval=1000 / fps,
#                                    repeat=False)
#
#     anim.save("animation.gif", writer=animation.PillowWriter(fps))
#
# pyplot.show()


