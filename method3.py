"""
imu_e0_pipeline.py

Usage:
    from imu_e0_pipeline import compute_e0_trajectory
    res = compute_e0_trajectory(data, sample_rate=50.0)

Input:
    data: dict with keys ['time','ax','ay','az','gx','gy','gz']
          ax/ay/az in g, gx/gy/gz in deg/s.
    sample_rate: used if data['time'] missing. Default 50.

Output (dict):
    't' : time array
    'pos3' : position in e0 frame (N,3) (meters)
    'vel' : velocity in e0 frame (N,3) (m/s)
    'lin_acc' : highpass(linear accel in e0) (N,3) (m/s^2)
    'quat' : quaternion sequence (N,4) mapping body -> e0
    'pos2' : PCA-projected 2D coordinates (not normalized)
    'pos2norm' : PCA-projected and scale-normalized 2D coords (range normalized)
    'static_mask' : boolean mask static samples
    'g_e0' : estimated gravity vector in e0 (3,)
    'gyro_bias' : estimated gyro bias (rad/s)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Quaternion utilities
# ---------------------------
def quat_normalize(q):
    q = np.array(q, dtype=float)
    n = np.linalg.norm(q)
    return q / n if n > 0 else np.array([1.0,0.0,0.0,0.0])

def quat_mult(q1, q2):
    # Hamilton product q = q1 * q2
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2], dtype=float)

def quat_from_rotvec(rotvec):
    # rotvec: rotation vector (rad) 3-vector; angle = ||rotvec||
    theta = np.linalg.norm(rotvec)
    if theta < 1e-12:
        # small-angle approx
        return quat_normalize(np.concatenate(([1.0], 0.5*rotvec)))
    axis = rotvec / theta
    w = np.cos(theta/2.0)
    s = np.sin(theta/2.0)
    xyz = axis * s
    return np.concatenate(([w], xyz))

def quat_rotate(q, v):
    # rotate vector v by quaternion q (assume q rotates from body -> e0)
    w,x,y,z = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1-2*(x*x+z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1-2*(x*x+y*y)]
    ], dtype=float)
    return R.dot(v)

# ---------------------------
# simple filters (stateful in loops)
# ---------------------------
def lowpass_step(x, x_prev, y_prev, rc, dt):
    alpha = dt / (rc + dt)
    return y_prev + alpha * (x - y_prev)

def highpass_step(x, x_prev, y_prev, rc, dt):
    alpha = rc / (rc + dt)
    return alpha * (y_prev + x - x_prev)

def moving_std(x, win_len):
    if win_len <= 1:
        return np.zeros_like(x)
    pad = win_len // 2
    xp = np.pad(x, (pad,pad), mode='edge')
    kernel = np.ones(win_len) / win_len
    m1 = np.convolve(xp, kernel, mode='valid')
    m2 = np.convolve(xp*xp, kernel, mode='valid')
    std = np.sqrt(np.maximum(0.0, m2 - m1*m1))
    return std

# ---------------------------
# main pipeline
# ---------------------------
def compute_e0_trajectory(data,
                          sample_rate=50.0,
                          init_static_s=3.0,
                          comp_acc_alpha=0.02,
                          hp_cutoff=0.05,
                          gravity_lp_tau=1.0,
                          static_acc_std_window=0.5,
                          static_acc_std_thresh=0.02,
                          static_gyro_thresh_deg=0.6,
                          static_margin_s=0.1,
                          min_static_s=0.2):
    """
    Compute trajectory in e0 frame per your scheme.
    """
    # --- time ---
    if ('time' in data) and (data['time'] is not None):
        t = np.asarray(data['time'], dtype=float)
        t = t - t[0]
    else:
        N = len(data['ax'])
        dt = 1.0 / sample_rate
        t = np.linspace(0, (N-1)*dt, N)
    dt_arr = np.diff(t, prepend=t[0])
    if len(dt_arr)>1:
        med = np.median(dt_arr[1:])
        dt_arr[0] = med
    fs = 1.0 / np.median(dt_arr)

    # --- read sensors and units conversion ---
    ax = np.asarray(data['ax'], dtype=float)
    ay = np.asarray(data['ay'], dtype=float)
    az = np.asarray(data['az'], dtype=float)
    gx = np.asarray(data['gx'], dtype=float)
    gy = np.asarray(data['gy'], dtype=float)
    gz = np.asarray(data['gz'], dtype=float)
    N = len(ax)
    assert len(ay)==N and len(az)==N and len(gx)==N and len(gy)==N and len(gz)==N

    acc = np.vstack([ax,ay,az]).T  # in g
    gyr_deg = np.vstack([gx,gy,gz]).T  # deg/s

    # convert accel to m/s^2
    acc_m = acc * 9.81

    # --- initial static window ---
    N0 = int(max(1, round(init_static_s * fs)))
    if N0 >= N:
        N0 = max(1, N//4)
    gyro_bias_deg = np.mean(gyr_deg[:N0], axis=0)
    gyro_bias_rad = np.deg2rad(gyro_bias_deg)  # rad/s

    g_e0_sensor = np.mean(acc_m[:N0], axis=0)  # estimated gravity vector in sensor coords (m/s^2)
    g_norm = np.linalg.norm(g_e0_sensor)
    if g_norm <= 1e-6:
        g_norm = 9.81
    # scale correction (make magnitude close to 9.81)
    scale = 9.81 / g_norm
    acc_m *= scale
    g_e0 = g_e0_sensor * scale  # final gravity vector in e0 coords

    # --- prepare gyro in rad/s (corrected) ---
    gyr = np.deg2rad(gyr_deg)  # rad/s
    gyr -= np.deg2rad(gyro_bias_deg)  # remove bias (rad/s)

    # --- gravity lowpass (diagnostic) ---
    gravity_lp = np.zeros(3)
    rc_g = gravity_lp_tau
    gravity_est = np.zeros((N,3))
    for i in range(N):
        gravity_lp = lowpass_step(acc_m[i], acc_m[i] if i==0 else acc_m[i-1], gravity_lp, rc_g, dt_arr[i])
        gravity_est[i] = gravity_lp.copy()

    # --- quaternion integration (body -> e0). start from identity (q=I) because e0 == initial body -->
    quats = np.zeros((N,4))
    q = np.array([1.0, 0.0, 0.0, 0.0])  # identity
    for i in range(N):
        dt = dt_arr[i]
        omega = gyr[i]  # rad/s, bias removed
        rotvec = omega * dt  # small rotation vector
        dq = quat_from_rotvec(rotvec)
        q = quat_mult(q, dq)
        q = quat_normalize(q)
        # optional tiny accel-based correction (very small alpha) to roll/pitch could be inserted here,
        # but we rely on g_e0 subtraction strategy, so skip aggressive accel correction.
        quats[i] = q.copy()

    # --- transform acc to e0 and subtract g_e0 to get linear acceleration in e0 ---
    acc_e0 = np.zeros((N,3))
    lin_acc = np.zeros((N,3))
    for i in range(N):
        a_b = acc_m[i]
        q = quats[i]
        a_e = quat_rotate(q, a_b)  # body->e0
        acc_e0[i] = a_e
        lin_acc[i] = a_e - g_e0

    # --- highpass filter lin_acc to remove low-frequency bias ---
    fc = float(hp_cutoff)
    rc_hp = 1.0 / (2.0 * np.pi * fc) if fc > 0 else 1e9
    acc_hp = np.zeros_like(lin_acc)
    xprev = lin_acc[0].copy()
    yprev = np.zeros(3)
    for i in range(N):
        dt = dt_arr[i]
        x = lin_acc[i]
        y = highpass_step(x, xprev, yprev, rc_hp, dt)
        acc_hp[i] = y
        xprev = x
        yprev = y

    # --- static detection based on acc_hp magnitude std + gyro magnitude ---
    win_len = int(max(1, round(static_acc_std_window * fs)))
    mag = np.linalg.norm(acc_hp, axis=1)
    mag_std = moving_std(mag, win_len)
    gyro_norm = np.linalg.norm(np.rad2deg(gyr), axis=1)  # in deg/s for threshold
    static_mask = (mag_std < static_acc_std_thresh) & (gyro_norm < static_gyro_thresh_deg)

    # expand static and remove short islands
    margin = int(round(static_margin_s * fs))
    if margin > 0:
        padded = static_mask.copy()
        for k in range(1, margin+1):
            padded[:-k] = padded[:-k] | static_mask[k:]
            padded[k:] = padded[k:] | static_mask[:-k]
        static_mask = padded
    min_len = int(max(1, round(min_static_s * fs)))
    # remove short runs
    def runs_from_mask(mask):
        runs = []
        i = 0
        Nn = len(mask)
        while i < Nn:
            if mask[i]:
                j = i
                while j < Nn and mask[j]:
                    j += 1
                runs.append((i, j))
                i = j
            else:
                i += 1
        return runs
    for (s,e) in runs_from_mask(static_mask):
        if (e - s) < min_len:
            static_mask[s:e] = False

    # --- integrate velocity & position (trapezoidal) with retro correction when static appears ---
    vel = np.zeros((N,3))
    pos = np.zeros((N,3))

    for i in range(1, N):
        dt = dt_arr[i]
        # a_prev = acc_hp[i-1]
        # a_curr = acc_hp[i]
        a_prev = lin_acc[i-1]
        a_curr = lin_acc[i]
        vel[i] = vel[i-1] + 0.5 * (a_prev + a_curr) * dt
        pos[i] = pos[i-1] + 0.5 * (vel[i-1] + vel[i]) * dt

        # if sample i is static and previous was moving => end of motion segment
        if static_mask[i] and (not static_mask[i-1]):
            # find start of motion
            j = i-1
            while j >= 0 and not static_mask[j]:
                j -= 1
            mstart = j + 1
            mend = i - 1
            L = mend - mstart + 1
            if L > 0:
                vend = vel[mend].copy()
                # subtract linear ramp (0->vend) over L samples
                for k in range(L):
                    idx = mstart + k
                    frac = (k+1) / L
                    vel[idx] = vel[idx] - frac * vend
                # re-integrate positions for that segment
                p0 = pos[mstart-1].copy() if (mstart - 1) >= 0 else np.zeros(3)
                for k in range(L):
                    idx = mstart + k
                    if k == 0:
                        pos[idx] = p0 + 0.5 * ( (vel[mstart-1] if (mstart-1)>=0 else np.zeros(3)) + vel[idx]) * dt_arr[idx]
                    else:
                        pos[idx] = pos[idx-1] + 0.5 * (vel[idx-1] + vel[idx]) * dt_arr[idx]
                # ensure velocity at static sample i is zero and position equals last corrected
                vel[i] = np.zeros(3)
                pos[i] = pos[mend].copy()

    # if all static, leave zeros
    if np.all(static_mask):
        vel[:] = 0.0
        pos[:] = 0.0

    # --- PCA projection and normalization to 2D ---
    pos_centered = pos - np.mean(pos, axis=0)
    # SVD for PCA
    U,S,Vt = np.linalg.svd(pos_centered, full_matrices=False)
    basis2 = Vt[:2,:]  # 2x3
    pos2 = pos_centered.dot(basis2.T)  # (N,2)
    rng = np.max(np.ptp(pos2, axis=0))
    pos2norm = pos2.copy()
    if rng > 0:
        pos2norm = pos2 / rng


    fig = plt.figure(figsize=(16, 8))
    plt.subplot(331)
    plt.plot(lin_acc[:, 0], label='lin_acc x')
    plt.legend()

    plt.subplot(334)
    plt.plot(lin_acc[:, 1], label='lin_acc y')
    plt.legend()

    plt.subplot(337)
    plt.plot(lin_acc[:, 2], label='lin_acc z')
    plt.legend()

    plt.subplot(332)
    plt.plot(acc_hp[:, 0], label='acc_hp x')
    plt.legend()

    plt.subplot(335)
    plt.plot(acc_hp[:, 1], label='acc_hp y')
    plt.legend()

    plt.subplot(338)
    plt.plot(acc_hp[:, 2], label='acc_hp z')
    plt.legend()

    plt.show()

    result = {
        't': t,
        'pos3': pos,
        'vel': vel,
        'lin_acc': acc_hp,
        'quat': quats,
        'pos2': pos2,
        'pos2norm': pos2norm,
        'static_mask': static_mask,
        'g_e0': g_e0,
        'gyro_bias': gyro_bias_deg
    }
    return result

# ---------------------------
# Example usage (if run directly)
# ---------------------------
if __name__ == "__main__":
    # Quick synthetic demo (replace with real data)
    fs = 50.0
    dur = 15.0
    N = int(dur * fs)

    # 大概会飘1m

    data = pd.read_csv('./test_data/magic_hand_static_data_20250627.txt')
    data.columns = ['gx', 'gy', 'gz', 'ax', 'ay', 'az', 'roll', 'pitch', 'yaw']
    data = data.dropna(how="any", axis=0)
    data = data.loc[:N, :]
    # add time column
    sample_rate = 50
    num_samples = data.shape[0]
    timestamp = np.linspace(0, (num_samples - 1) / sample_rate, num_samples)
    data['time'] = timestamp

    res = compute_e0_trajectory(data, sample_rate=fs)
    print("pos3 shape:", res['pos3'].shape)
    print("pos2norm sample range:", np.min(res['pos2norm'], axis=0), np.max(res['pos2norm'], axis=0))

    # optional plotting if you want to visualize (requires matplotlib)
    try:
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(121, projection='3d')
        ax.plot(res['pos3'][:,0], res['pos3'][:,1], res['pos3'][:,2], '-o', markersize=2)
        ax.set_title('pos3 in e0')
        ax2 = fig.add_subplot(122)
        ax2.plot(res['pos2norm'][:,0], res['pos2norm'][:,1], '-o', markersize=2)
        ax2.set_title('pos2norm (PCA projected)')
        plt.show()
    except Exception:
        pass
