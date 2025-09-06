"""
imu_shape_pipeline.py
纯 NumPy 实现：从 MPU6050 原始数据（ax,ay,az,gx,gy,gz,time）生成短时相对轨迹（3D & PCA 投影 2D）。
特点：不依赖初始对准，在线 lowpass 重力估计、小 alpha 互补姿态、hp 去偏、buffer 回溯修正、PCA 后处理。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# ---  Quaternion helpers  ---
# ----------------------------
def quat_normalize(q):
    q = np.array(q, dtype=float)
    n = np.linalg.norm(q)
    if n == 0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n

def quat_mult(q1, q2):
    # q = [w, x, y, z]
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2], dtype=float)

def quat_from_small_ang(omega, dt):
    # omega: vector (rad/s); small-angle quaternion approximation
    theta = 0.5 * np.asarray(omega) * dt
    q = np.concatenate(([1.0], theta))
    return quat_normalize(q)

def quat_from_euler(roll, pitch, yaw):
    # roll,pitch,yaw in radians
    cr = np.cos(roll/2); sr = np.sin(roll/2)
    cp = np.cos(pitch/2); sp = np.sin(pitch/2)
    cy = np.cos(yaw/2); sy = np.sin(yaw/2)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return quat_normalize(np.array([w,x,y,z]))

def quat_rotate(q, v):
    # rotate vector v by quaternion q
    # v' = R(q) v
    w,x,y,z = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1-2*(x*x+z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1-2*(x*x+y*y)]
    ], dtype=float)
    return R.dot(v)

def quat_from_acc(acc):
    # derive roll & pitch from accelerometer vector (assume acc ~ gravity direction)
    ax,ay,az = acc
    # protect against zero
    if np.linalg.norm(acc) == 0:
        return np.array([1.0,0.0,0.0,0.0])
    # roll = atan2(ay, az); pitch = atan2(-ax, sqrt(ay^2+az^2))
    roll = np.arctan2(ay, az)
    pitch = np.arctan2(-ax, np.sqrt(ay*ax + az*az + 1e-12))
    yaw = 0.0
    return quat_from_euler(roll, pitch, yaw)

# ----------------------------
# ---  signal filters      ---
# ----------------------------
def lowpass_step(x, x_prev, y_prev, rc, dt):
    # first-order lowpass (stateful): y = y_prev + (dt/ (rc + dt)) * (x - y_prev)
    alpha = dt / (rc + dt)
    return y_prev + alpha * (x - y_prev)

def highpass_step(x, x_prev, y_prev, rc, dt):
    # first-order highpass (stateful), same formula as earlier:
    alpha = rc / (rc + dt)
    y = alpha * (y_prev + x - x_prev)
    return y

# ----------------------------
# ---  static detection    ---
# ----------------------------
def moving_std(x, win_len):
    # moving std on 1D array x, returns same length (edge using edge padding)
    if win_len <= 1:
        return np.zeros_like(x)
    pad = win_len // 2
    xp = np.pad(x, (pad, pad), mode='edge')
    kernel = np.ones(win_len) / win_len
    m1 = np.convolve(xp, kernel, mode='valid')
    m2 = np.convolve(xp*xp, kernel, mode='valid')
    std = np.sqrt(np.maximum(0.0, m2 - m1*m1))
    return std

# ----------------------------
# ---  main pipeline       ---
# ----------------------------
def imu_to_shape_positions(data,
                           sample_rate=None,
                           gyro_in_deg=True,
                           accel_in_g_auto=True,
                           init_gravity_lp_tau=1.0,   # seconds for lowpass gravity estimate
                           comp_acc_alpha=0.04,      # accel contribution in complementary filter (0..1)
                           hp_cutoff=0.05,           # highpass cutoff freq (Hz) for a_lin
                           static_acc_std_window=0.5,# s window for acc std for static detection
                           static_acc_std_thresh=0.02,# m/s^2 threshold
                           static_gyro_thresh=0.6,   # deg/s or rad/s depending on gyro_in_deg
                           static_margin_s=0.1,
                           min_static_s=0.2):
    """
    输入:
        data: dict with keys 'time','ax','ay','az','gx','gy','gz'
        sample_rate: optional if 'time' missing
        gyro_in_deg: True if gx/gy/gz in deg/s (MPU6050 typical). 如果 False，视为 rad/s.
        accel_in_g_auto: if True, auto-convert accel (|acc|<20 -> multiple of g)
    输出:
        result: dict with keys:
            't', 'pos3' (N,3), 'pos2' (N,2 PCA projected & normalized),
            'vel' (N,3), 'quat' (N,4), 'lin_acc' (N,3), 'static_mask' (N,)
    """
    # ---- read time and dt ----
    if 'time' in data and data['time'] is not None:
        t = np.asarray(data['time'], dtype=float)
        t = t - t[0]
    else:
        if sample_rate is None:
            raise ValueError("data['time'] missing and sample_rate not provided")
        N = len(data['ax'])
        dt = 1.0 / sample_rate
        t = np.linspace(0, (N-1)*dt, N)
    dt_arr = np.diff(t, prepend=t[0])
    # fix first dt
    if len(dt_arr) > 1:
        med = np.median(dt_arr[1:])
        if med <= 0:
            raise ValueError("time stamps invalid")
        dt_arr[0] = med
    fs = 1.0 / np.median(dt_arr)

    # ---- read sensors ----
    ax = np.asarray(data['ax'], dtype=float)
    ay = np.asarray(data['ay'], dtype=float)
    az = np.asarray(data['az'], dtype=float)
    gx = np.asarray(data['gx'], dtype=float)
    gy = np.asarray(data['gy'], dtype=float)
    gz = np.asarray(data['gz'], dtype=float)
    N = len(ax)
    assert len(ay)==N and len(az)==N and len(gx)==N and len(gy)==N and len(gz)==N

    acc = np.vstack([ax,ay,az]).T
    gyr = np.vstack([gx,gy,gz]).T

    # units: accel in g -> convert
    if accel_in_g_auto:
        if np.max(np.abs(acc)) < 20.0:
            acc = acc * 9.81

    # gyro units: if in deg/s, convert to rad/s for quaternion integration
    if gyro_in_deg:
        gyr = np.deg2rad(gyr)

    # ---- online gravity estimate (lowpass on accel) ----
    # stateful lowpass: y[k] = y[k-1] + alpha*(x[k]-y[k-1]) where rc = tau, alpha = dt/(rc+dt)
    gravity_lp = np.zeros(3)
    rc_g = init_gravity_lp_tau
    gravity_est = np.zeros((N,3))
    for i in range(N):
        dt = dt_arr[i]
        gravity_lp = lowpass_step(acc[i], gravity_lp, gravity_lp, rc_g, dt)  # use same var for y_prev
        gravity_est[i] = gravity_lp.copy()

    # ---- complementary-filter attitude (quaternion) ----
    # initialize quaternion as identity (will be slowly corrected by accel lowpass)
    quats = np.zeros((N,4))
    q = np.array([1.0, 0.0, 0.0, 0.0])
    for i in range(N):
        dt = dt_arr[i]
        omega = gyr[i]
        # integrate gyro to get predicted quaternion (small-angle)
        dq = quat_from_small_ang(omega, dt)  # uses rad/s input
        q_pred = quat_mult(q, dq)

        # accel-based quaternion from instantaneous accel (use accel vector but prefer gravity_est)
        a_meas = acc[i]
        # if accel magnitude indicates strong motion, trust gyro more; but we'll use comp_acc_alpha small
        q_acc = quat_from_acc(a_meas)

        # combine via normalized linear interpolation (nlerp)
        alpha = comp_acc_alpha
        q = quat_normalize((1.0 - alpha) * q_pred + alpha * q_acc)
        quats[i] = q.copy()

    # ---- convert body accel -> inertial (earth) and subtract gravity vector (0,0,9.81) ----
    earth_acc = np.zeros((N,3))
    for i in range(N):
        a_b = acc[i]
        q = quats[i]
        a_e = quat_rotate(q, a_b)  # body->earth approx
        # subtract gravity vector (0,0,9.81). Note: if your earth frame has z up/down change sign accordingly.
        a_lin = a_e - np.array([0.0, 0.0, 9.81])
        earth_acc[i] = a_lin

    # ---- highpass to remove low-frequency bias (stateful sample-varying) ----
    fc = float(hp_cutoff)
    if fc <= 0:
        rc_hp = 1e9
    else:
        rc_hp = 1.0 / (2.0 * np.pi * fc)
    acc_hp = np.zeros_like(earth_acc)
    xprev = earth_acc[0].copy()
    yprev = np.zeros(3)
    for i in range(N):
        dt = dt_arr[i]
        x = earth_acc[i]
        y = highpass_step(x, xprev, yprev, rc_hp, dt)
        acc_hp[i] = y
        xprev = x
        yprev = y

    # ---- static detection (acc hp magnitude std + gyro norm) ----
    win_len = int(max(1, round(static_acc_std_window * fs)))
    mag = np.linalg.norm(acc_hp, axis=1)
    mag_std = moving_std(mag, win_len)
    gyro_norm = np.linalg.norm(gyr, axis=1) if not gyro_in_deg else np.linalg.norm(gyr, axis=1)
    static_mask = (mag_std < static_acc_std_thresh) & (gyro_norm < (np.deg2rad(static_gyro_thresh) if gyro_in_deg else static_gyro_thresh))
    # expand static by margin and remove short islands
    margin = int(round(static_margin_s * fs))
    if margin > 0:
        padded = static_mask.copy()
        for i in range(1, margin+1):
            padded[:-i] = padded[:-i] | static_mask[i:]
            padded[i:] = padded[i:] | static_mask[:-i]
        static_mask = padded
    # remove short static runs
    min_len = int(max(1, round(min_static_s * fs)))
    # find runs
    def runs(mask):
        out = []; i=0
        while i < len(mask):
            if mask[i]:
                j=i
                while j<len(mask) and mask[j]:
                    j+=1
                out.append((i,j))
                i=j
            else:
                i+=1
        return out
    for (s,e) in runs(static_mask):
        if (e-s) < min_len:
            static_mask[s:e] = False

    # ---- integrate (buffering + retro correction similar to previous design) ----
    vel = np.zeros((N,3))
    pos = np.zeros((N,3))
    # we'll accumulate segments between static-confirmed samples
    # naive: integrate sequentially; whenever we detect a newly confirmed static run following motion, apply ramp correction on the previous motion run
    last_confirmed_idx = -1
    for i in range(1, N):
        # trapezoidal integrate velocity
        v_prev = vel[i-1]
        a_prev = acc_hp[i-1]
        a_curr = acc_hp[i]
        dt = dt_arr[i]
        vel[i] = v_prev + 0.5*(a_prev + a_curr) * dt
        pos[i] = pos[i-1] + 0.5*(vel[i-1] + vel[i]) * dt

        # check for static-run that ends at i (i is static and previous was moving)
        if static_mask[i] and (i-1 >=0) and (not static_mask[i-1]):
            # find start of last motion run (index mstart)
            j = i-1
            while j >= 0 and not static_mask[j]:
                j -= 1
            mstart = j+1
            mend = i-1
            L = mend - mstart + 1
            if L > 0:
                vend = vel[mend].copy()
                # subtract linear ramp (0->vend) over L samples
                for k in range(L):
                    idx = mstart + k
                    frac = (k+1)/L
                    vel[idx] = vel[idx] - frac * vend
                # re-integrate positions for that segment
                # p at mstart-1 is unchanged
                p0 = pos[mstart-1].copy() if mstart-1 >= 0 else np.zeros(3)
                v_prev_seg = vel[mstart-1].copy() if mstart-1 >= 0 else np.zeros(3)
                for k in range(L):
                    idx = mstart + k
                    dtk = dt_arr[idx]
                    v_curr = vel[idx]
                    pos[idx] = p0 + 0.5*(v_prev_seg + v_curr) * dt_arr[idx] if k==0 else pos[idx-1] + 0.5*(vel[idx-1] + vel[idx]) * dtk
                    v_prev_seg = v_curr
                # ensure velocity at i (first static) is zero and position equal last corrected
                vel[i] = np.zeros(3)
                pos[i] = pos[mend].copy() if mend>=0 else p0.copy()
            last_confirmed_idx = i

    # if entire sequence static -> vel/pos zeros
    if np.all(static_mask):
        vel[:] = 0.0
        pos[:] = 0.0

    # ---- PCA plane projection and normalization (produce 2D shape) ----
    # center
    pos_centered = pos - np.mean(pos, axis=0)
    # SVD for PCA
    U, S, Vt = np.linalg.svd(pos_centered, full_matrices=False)
    # take first two principal components as plane basis
    basis2 = Vt[:2,:]  # shape (2,3)
    pos2 = pos_centered.dot(basis2.T)  # (N,2)
    # normalize scale: divide by max range
    rng = np.max(np.ptp(pos2, axis=0))
    if rng <= 0:
        pos2norm = pos2.copy()
    else:
        pos2norm = pos2 / rng

    # optional: rotate pos2 so that major axis aligns with x (not necessary)
    # we return both pos2 (raw scaled) and pos2norm (normalized to [-0.5..0.5] roughly)

    result = {
        't': t,
        'pos3': pos,
        'pos2': pos2,
        'pos2norm': pos2norm,
        'vel': vel,
        'quat': quats,
        'lin_acc': acc_hp,
        'static_mask': static_mask,
        'pca_basis': basis2,
        'gravity_est': gravity_est
    }
    return result

# ----------------------------
# ---  quick usage example ---
# ----------------------------
if __name__ == "__main__":
    # Minimal demo usage: user should replace with real data loading
    # Example: data dict with 50 Hz, 10 s, simulate a circular hand motion (toy)
    # fs = 50.0
    # dur = 10.0
    # N = int(dur * fs)
    # t = np.linspace(0, dur, N)
    # # toy: a circle in xy plane radius 0.2 m at 0.5 Hz, produce synthetic pos, then diff twice to get acc
    # r = 0.2
    # omega = 2*np.pi*0.5
    # px = r * np.cos(omega * t)
    # py = r * np.sin(omega * t)
    # pz = 0.0 * t
    # # velocities and accelerations (numerical)
    # vx = np.gradient(px, t)
    # vy = np.gradient(py, t)
    # vz = np.gradient(pz, t)
    # ax_s = np.gradient(vx, t)
    # ay_s = np.gradient(vy, t)
    # az_s = np.gradient(vz, t)
    # # build synthetic imu where body frame == earth frame and no gyro rotation (not realistic)
    # # add gravity to accelerometer
    # ax_meas = ax_s
    # ay_meas = ay_s
    # az_meas = az_s + 9.81
    # gx_meas = np.zeros_like(t)
    # gy_meas = np.zeros_like(t)
    # gz_meas = np.zeros_like(t)
    # data = {'time': t, 'ax': ax_meas, 'ay': ay_meas, 'az': az_meas, 'gx': gx_meas, 'gy': gy_meas, 'gz': gz_meas}
    # res = imu_to_shape_positions(data, sample_rate=fs)
    # print("pos3 shape:", res['pos3'].shape)
    # print("pos2norm range:", np.min(res['pos2norm'], axis=0), np.max(res['pos2norm'], axis=0))

    # You would replace the synthetic data with your real data and then:
    data = pd.read_csv('./test_data/magic_hand_static_data_20250627.txt')
    data.columns = ['gx', 'gy', 'gz', 'ax', 'ay', 'az', 'roll', 'pitch', 'yaw']
    data = data.dropna(how="any", axis=0)

    # add time column
    sample_rate = 50
    num_samples = data.shape[0]
    timestamp = np.linspace(0, (num_samples - 1) / sample_rate, num_samples)
    data['time'] = timestamp
    res = imu_to_shape_positions(data, sample_rate=50.0, gyro_in_deg=True)
    final_2d = res['pos3']   # feed this into your recognition model
    print(final_2d)

    fig = plt.figure(figsize=(16, 8))
    plt.subplot(331)
    plt.plot(final_2d[:, 0], label='x')
    plt.legend()

    plt.subplot(334)
    plt.plot(final_2d[:, 1], label='y')
    plt.legend()

    plt.subplot(337)
    plt.plot(final_2d[:, 2], label='z')
    plt.legend()

    plt.show()
