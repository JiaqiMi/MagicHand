# -*- coding: utf-8 -*-
"""
IMU e0 轨迹重建（离线单文件 · 漂移捕捉与补偿版）
- 用起始静止窗口(默认5s)拟合速度线性漂移，并回推为等效加速度偏置进行全段补偿
- 静止强约束：静止点速度强制为0，位置保持，抑制纯静止段爬行
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# =========================
# 配置
# =========================
@dataclass
class Config:
    sample_rate: float = 200.0
    init_static_s: float = 5.0           # 起始静止时长(秒) —— 同时作为漂移拟合窗口
    gyro_unit: str = 'deg_per_hour'      # 'deg_per_hour' | 'deg_per_sec' | 'rad_per_sec'
    g0: float = 9.80665

    # 速度/位置漂移抑制
    drift_use_init_win: bool = True      # 是否用起始静止窗口做线性漂移捕捉
    hp_cutoff_hz: float = 0.02            # 速度积分前对线加速度做轻微高通（去残余DC）
    strong_static_clamp: bool = False     # 静止点速度钳0、位置保持

    # 静止检测
    acc_std_win_s: float = 0.5
    acc_std_thresh: float = 0.015         # m/s^2
    gyro_thresh_deg_s: float = 0.3       # deg/s
    static_margin_s: float = 0.05
    min_static_s: float = 0.15

    # 可视化
    visualize: bool = True
    apply_pca: bool = True

    # 读文件
    file_has_header: bool = False
    file_sep_regex: str = r'\s+'
    columns_no_header: Tuple[str, ...] = ('ax','ay','az','gx','gy','gz','index')

# =========================
# 基础工具
# =========================
def butter_highpass(data: np.ndarray, fc: float, fs: float, order: int = 2) -> np.ndarray:
    if fc <= 0 or fc >= fs/2 or data.shape[0] < 3*order+1:
        return data.copy()
    b, a = butter(order, fc/(fs/2), btype='high')
    X = np.atleast_2d(data)
    Y = np.zeros_like(X)
    for j in range(X.shape[1]):
        Y[:, j] = filtfilt(b, a, X[:, j], method="pad")
    return Y if data.ndim > 1 else Y.ravel()

def moving_std_1d(x: np.ndarray, win_len: int) -> np.ndarray:
    if win_len <= 1:
        return np.zeros_like(x)
    pad = win_len // 2
    xp = np.pad(x, (pad, pad), mode='edge')
    ker = np.ones(win_len) / win_len
    m1 = np.convolve(xp, ker, mode='valid')
    m2 = np.convolve(xp*xp, ker, mode='valid')
    return np.sqrt(np.maximum(0.0, m2 - m1*m1))

def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q);  return q/n if n>0 else np.array([1.,0.,0.,0.])

def quat_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1,x1,y1,z1 = q1;  w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2], dtype=float)

def quat_from_rotvec(rv: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(rv)
    if th < 1e-12:
        return quat_normalize(np.array([1., *(0.5*rv)]))
    axis = rv / th
    w = np.cos(th/2.0); s = np.sin(th/2.0)
    return np.array([w, *(axis*s)])

def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w,x,y,z = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1-2*(x*x+z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1-2*(x*x+y*y)]
    ], dtype=float)
    return R @ v

# =========================
# 数据读取
# =========================
def load_imu_data(file_path: str, cfg: Config) -> Dict[str, np.ndarray]:
    if cfg.file_has_header:
        df = pd.read_csv(file_path, sep=cfg.file_sep_regex)
    else:
        df = pd.read_csv(file_path, header=None, sep=cfg.file_sep_regex)
        ncol = df.shape[1]
        cols = list(cfg.columns_no_header[:ncol])
        df.columns = cols

    need = ['ax','ay','az','gx','gy','gz']
    for k in need:
        if k not in df.columns:
            raise ValueError(f"缺少必要列: {k}")
        df[k] = pd.to_numeric(df[k], errors='coerce')

    df = df.dropna(how='any')
    N = len(df)
    t = np.linspace(0, (N-1)/cfg.sample_rate, N)

    gx, gy, gz = df['gx'].to_numpy(), df['gy'].to_numpy(), df['gz'].to_numpy()
    if cfg.gyro_unit == 'deg_per_hour':
        gx = np.deg2rad(gx/3600.0); gy = np.deg2rad(gy/3600.0); gz = np.deg2rad(gz/3600.0)
    elif cfg.gyro_unit == 'deg_per_sec':
        gx = np.deg2rad(gx); gy = np.deg2rad(gy); gz = np.deg2rad(gz)
    elif cfg.gyro_unit == 'rad_per_sec':
        pass
    else:
        raise ValueError("gyro_unit 取值非法")

    return dict(
        time=t,
        ax=df['ax'].to_numpy(), ay=df['ay'].to_numpy(), az=df['az'].to_numpy(),
        gx=gx, gy=gy, gz=gz
    )

# =========================
# 初始静止估计（g 向量 & 陀螺零偏）
# =========================
def estimate_init(acc_m_s2: np.ndarray,
                  gyr_rad_s: np.ndarray,
                  fs: float,
                  init_static_s: float,
                  g0: float) -> Tuple[np.ndarray, np.ndarray, float]:
    N0 = max(1, int(round(init_static_s * fs)))
    g_vec_raw = acc_m_s2[:N0].mean(axis=0)
    g_norm = np.linalg.norm(g_vec_raw) or g0
    scale = g0 / g_norm
    g_e0 = g_vec_raw * scale
    gyro_bias = gyr_rad_s[:N0].mean(axis=0)
    return g_e0, gyro_bias, scale

# =========================
# 姿态积分（仅陀螺）
# =========================
def integrate_orientation(gyr_rad_s: np.ndarray, dt_arr: np.ndarray) -> np.ndarray:
    N = len(gyr_rad_s)
    quats = np.zeros((N, 4))
    q = np.array([1.,0.,0.,0.])
    for i in range(N):
        dq = quat_from_rotvec(gyr_rad_s[i] * dt_arr[i])
        q = quat_mult(q, dq)
        q = quat_normalize(q)
        quats[i] = q
    return quats

# =========================
# 线加速度（旋到 e0 并减 g）
# =========================
def compute_linear_acc(acc_body: np.ndarray,
                       quats: np.ndarray,
                       g_e0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    N = len(acc_body)
    acc_e0 = np.zeros((N,3))
    lin_acc = np.zeros((N,3))
    for i in range(N):
        a_e = quat_rotate(quats[i], acc_body[i])
        acc_e0[i] = a_e
        lin_acc[i] = a_e - g_e0
    return acc_e0, lin_acc

# =========================
# 用起始静止窗口捕捉“速度线性漂移”并回推加速度常偏
# =========================
def estimate_linear_drift_from_init(lin_acc: np.ndarray,
                                    dt_arr: np.ndarray,
                                    fs: float,
                                    win_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回：
      a_hat: (3,)   等效恒定加速度偏置（m/s^2）
      b_hat: (3,)   初始速度偏置（m/s）
    做法：
      - 先对 lin_acc 积分得 v_raw（不高通）
      - 在 [0, win_s] 上拟合 v_raw(t) ≈ b + a*t
    """
    N0 = max(2, int(round(win_s * fs)))
    t = np.cumsum(dt_arr)
    # 原始速度（不高通）
    v = np.zeros_like(lin_acc)
    for i in range(1, len(lin_acc)):
        v[i] = v[i-1] + 0.5*(lin_acc[i-1] + lin_acc[i]) * dt_arr[i]

    # 线性回归：逐轴 v ~ a*t + b
    t0 = t[:N0]
    A = np.vstack([t0, np.ones_like(t0)]).T
    a_hat = np.zeros(3)
    b_hat = np.zeros(3)
    for d in range(3):
        y = v[:N0, d]
        # 正规方程（N0>=2 已保证）
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        a_hat[d], b_hat[d] = coef[0], coef[1]
    return a_hat, b_hat

# =========================
# 静止检测
# =========================
def detect_stationary(lin_acc_e0: np.ndarray,
                      gyr_rad_s: np.ndarray,
                      fs: float,
                      acc_std_win_s: float,
                      acc_std_thresh: float,
                      gyro_thresh_deg_s: float) -> np.ndarray:
    hp_cutoff_det = 0.3
    acc_hp = butter_highpass(lin_acc_e0, hp_cutoff_det, fs, order=2)
    mag = np.linalg.norm(acc_hp, axis=1)

    win_len = max(1, int(round(acc_std_win_s * fs)))
    mag_std = moving_std_1d(mag, win_len)

    gyro_norm_deg_s = np.linalg.norm(np.rad2deg(gyr_rad_s), axis=1)
    # 修复长度截断：不要再用 mag_std[:-1]
    static_mask = (mag_std[:-1] < acc_std_thresh) & (gyro_norm_deg_s < gyro_thresh_deg_s)
    return static_mask

def expand_and_clean_static(static_mask: np.ndarray, fs: float,
                            static_margin_s: float, min_static_s: float) -> np.ndarray:
    mask = static_mask.copy()
    N = len(mask)
    m = int(round(static_margin_s * fs))
    if m > 0:
        padded = mask.copy()
        for k in range(1, m+1):
            padded[:-k] = padded[:-k] | mask[k:]
            padded[k:]  = padded[k:]  | mask[:-k]
        mask = padded
    min_len = max(1, int(round(min_static_s * fs)))
    i = 0
    while i < N:
        if mask[i]:
            j = i
            while j < N and mask[j]:
                j += 1
            if (j - i) < min_len:
                mask[i:j] = False
            i = j
        else:
            i += 1
    return mask

# =========================
# 速度/位置积分 + 静止强约束 + 段尾回溯归零
# =========================
def integrate_vel_pos(lin_acc_hp: np.ndarray,
                      dt_arr: np.ndarray,
                      static_mask: np.ndarray,
                      strong_static_clamp: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    N = len(lin_acc_hp)
    vel = np.zeros((N,3))
    pos = np.zeros((N,3))

    for i in range(1, N):
        dt = dt_arr[i]
        vel[i] = vel[i-1] + 0.5*(lin_acc_hp[i-1] + lin_acc_hp[i]) * dt

        # 强静止约束：直接钳零
        if strong_static_clamp and static_mask[i]:
            vel[i] = np.zeros(3)

        pos[i] = pos[i-1] + 0.5*(vel[i-1] + vel[i]) * dt

        # “由动入静”段尾回溯线性归零（可与强约束并存）
        if static_mask[i] and (not static_mask[i-1]):
            j = i - 1
            while j >= 0 and not static_mask[j]:
                j -= 1
            mstart = j + 1
            mend = i - 1
            Ls = mend - mstart + 1
            if Ls > 0:
                vend = vel[mend].copy()
                for k in range(Ls):
                    idx = mstart + k
                    frac = (k + 1) / Ls
                    vel[idx] = vel[idx] - frac * vend
                # 位置重算该段
                p0 = pos[mstart-1].copy() if (mstart - 1) >= 0 else np.zeros(3)
                for k in range(Ls):
                    idx = mstart + k
                    if k == 0:
                        prev_v = vel[mstart-1] if (mstart - 1) >= 0 else np.zeros(3)
                        pos[idx] = p0 + 0.5*(prev_v + vel[idx]) * dt_arr[idx]
                    else:
                        pos[idx] = pos[idx-1] + 0.5*(vel[idx-1] + vel[idx]) * dt_arr[idx]
                vel[i] = np.zeros(3)
                pos[i] = pos[mend].copy()
    return vel, pos


# =========================
# 回路闭合（可选）
# =========================
def loop_closure_linear(pos: np.ndarray, static_mask: np.ndarray) -> np.ndarray:
    """
    简单回路闭合：若首尾都有静止段，把终点偏移 Δ 分布成沿时长线性分摊的位移修正。
    对“方形回到起点”的任务很有效。
    """
    N = len(pos)
    # 找首静止段中心、尾静止段中心
    def center_of_first_run(mask):
        i = 0
        while i < N and not mask[i]: i += 1
        if i == N: return None
        j = i
        while j < N and mask[j]: j += 1
        return (i + j) // 2

    def center_of_last_run(mask):
        j = N-1
        while j >=0 and not mask[j]: j -= 1
        if j < 0: return None
        i = j
        while i >= 0 and mask[i]: i -= 1
        return (i + j) // 2

    c0 = center_of_first_run(static_mask)
    c1 = center_of_last_run(static_mask)
    if c0 is None or c1 is None or c1 <= c0:
        return pos

    delta = pos[c1] - pos[c0]  # 终点相对起点的偏移
    if np.linalg.norm(delta) < 1e-6:
        return pos

    t = np.linspace(0.0, 1.0, N)
    correction = (t[:, None]) * (-delta[None, :])
    return pos + correction

# =========================
# PCA（可选）
# =========================
def pca_project_2d(pos3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    P = pos3 - pos3.mean(axis=0, keepdims=True)
    if P.shape[0] < 2:
        return np.zeros((P.shape[0],2)), np.zeros((P.shape[0],2))
    U,S,Vt = np.linalg.svd(P, full_matrices=False)
    basis2 = Vt[:2, :]
    pos2 = P @ basis2.T
    rng = np.max(np.ptp(pos2, axis=0)) if pos2.size else 0.0
    pos2norm = pos2 / rng if rng>0 else pos2.copy()
    return pos2, pos2norm

# =========================
# 可视化
# =========================
def visualize_results(t: np.ndarray,
                      lin_acc: np.ndarray,
                      lin_acc_hp: np.ndarray,
                      gyr_rad_s: np.ndarray,
                      static_mask: np.ndarray):
    fig = plt.figure(figsize=(16,8))
    ax1 = plt.subplot(331); ax1.plot(lin_acc[:,0]); ax1.set_title('lin_acc x (e0)')
    ax2 = plt.subplot(334); ax2.plot(lin_acc[:,1]); ax2.set_title('lin_acc y (e0)')
    ax3 = plt.subplot(337); ax3.plot(lin_acc[:,2]); ax3.set_title('lin_acc z (e0)')
    ax4 = plt.subplot(332); ax4.plot(lin_acc_hp[:,0]); ax4.set_title('lin_acc HP x')
    ax5 = plt.subplot(335); ax5.plot(lin_acc_hp[:,1]); ax5.set_title('lin_acc HP y')
    ax6 = plt.subplot(338); ax6.plot(lin_acc_hp[:,2]); ax6.set_title('lin_acc HP z')
    ax7 = plt.subplot(333); ax7.plot(np.rad2deg(gyr_rad_s)[:,0]); ax7.set_title('gyro x (deg/s)')
    ax8 = plt.subplot(336); ax8.plot(np.rad2deg(gyr_rad_s)[:,1]); ax8.set_title('gyro y (deg/s)')
    ax9 = plt.subplot(339); ax9.plot(np.rad2deg(gyr_rad_s)[:,2]); ax9.set_title('gyro z (deg/s)')

    for ax in [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]:
        ymin,ymax = ax.get_ylim()
        in_seg = False; s = 0
        for i in range(len(static_mask)):
            if static_mask[i] and not in_seg:
                in_seg = True; s = i
            if (in_seg and (i==len(static_mask)-1 or not static_mask[i])):
                e = i if not static_mask[i] else i
                ax.axvspan(s, e, color='lightgray', alpha=0.3)
                in_seg = False
        ax.set_ylim(ymin, ymax)
    plt.tight_layout(); plt.show()

# =========================
# 主流程
# =========================
def compute_e0_trajectory_from_arrays(data: Dict[str, np.ndarray],
                                      cfg: Config) -> Dict[str, np.ndarray]:
    # 时间
    if ('time' in data) and (data['time'] is not None):
        t = np.asarray(data['time'], dtype=float); t = t - t[0]
    else:
        N = len(data['ax'])
        t = np.linspace(0, (N-1)/cfg.sample_rate, N)
    dt_arr = np.diff(t, prepend=t[0])
    if len(dt_arr) > 1:
        med = np.median(dt_arr[1:]); dt_arr[0] = med
    fs = 1.0 / np.median(dt_arr)

    # 传感器
    ax = np.asarray(data['ax'], dtype=float)
    ay = np.asarray(data['ay'], dtype=float)
    az = np.asarray(data['az'], dtype=float)
    gx = np.asarray(data['gx'], dtype=float)
    gy = np.asarray(data['gy'], dtype=float)
    gz = np.asarray(data['gz'], dtype=float)
    N = len(ax)
    assert all(len(x)==N for x in [ay,az,gx,gy,gz])

    # 角速度单位换算
    if cfg.gyro_unit == 'deg_per_hour':
        gx = np.deg2rad(gx/3600.0); gy = np.deg2rad(gy/3600.0); gz = np.deg2rad(gz/3600.0)
    elif cfg.gyro_unit == 'deg_per_sec':
        gx = np.deg2rad(gx); gy = np.deg2rad(gy); gz = np.deg2rad(gz)
    elif cfg.gyro_unit == 'rad_per_sec':
        pass
    else:
        raise ValueError("gyro_unit 取值非法")

    acc_m = np.vstack([ax, ay, az]).T
    gyr = np.vstack([gx, gy, gz]).T

    # 起始静止：g 与 gyro_bias
    g_e0, gyro_bias, scale = estimate_init(acc_m, gyr, fs, cfg.init_static_s, cfg.g0)
    acc_m *= scale
    gyr = gyr - gyro_bias

    # 姿态
    quats = integrate_orientation(gyr, dt_arr)
    # 将姿态以横滚、俯仰、偏航角打印出来（可选）
    euler_angles = np.zeros((N, 3))
    for i in range(N):
        q = quats[i]
        w,x,y,z = q
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0))
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        euler_angles[i] = np.array([roll, pitch, yaw])

    # 线加速度（未高通）
    acc_e0, lin_acc = compute_linear_acc(acc_m, quats, g_e0)

    # —— 核心增强：用起始静止窗口捕捉线性漂移，并做全段补偿 ——
    a_hat = np.zeros(3)   # 等效恒加速度偏置
    b_hat = np.zeros(3)   # 初始速度偏置
    if cfg.drift_use_init_win:
        a_hat, b_hat = estimate_linear_drift_from_init(lin_acc, dt_arr, fs, cfg.init_static_s)
        # 在加速度层面先去掉恒定偏置 a_hat
        lin_acc = lin_acc - a_hat

    # 静止检测（用补偿后的 lin_acc）
    static_mask = detect_stationary(lin_acc, gyr, fs,
                                    cfg.acc_std_win_s, cfg.acc_std_thresh, cfg.gyro_thresh_deg_s)
    static_mask = expand_and_clean_static(static_mask, fs, cfg.static_margin_s, cfg.min_static_s)

    # 积分前轻微高通（进一步抑制残余 DC）
    lin_acc_hp = butter_highpass(lin_acc, cfg.hp_cutoff_hz, fs, order=2)

    # 速度/位置积分（强静止约束 + 段尾回溯）
    vel, pos = integrate_vel_pos(lin_acc_hp, dt_arr, static_mask, cfg.strong_static_clamp)

    # 速度层面的常值偏置 b_hat 再整体移除，使起始静止速度均值贴零
    if cfg.drift_use_init_win:
        vel -= b_hat  # 常量项
        # 再做一次起始窗口零均值微调（更稳健）
        N0 = max(1, int(round(cfg.init_static_s * fs)))
        vel -= vel[:N0].mean(axis=0, keepdims=True)
    
    # 回路闭合（可选）
    pos = loop_closure_linear(pos, static_mask) 

    # PCA
    if cfg.apply_pca:
        pos2, pos2norm = pca_project_2d(pos)
    else:
        pos2 = np.zeros((N,2))
        pos2norm = np.zeros((N,2))

    # 可视化
    if cfg.visualize:
        try:
            # 绘制姿态曲线（可选）
            plt.figure(figsize=(10,4))
            plt.plot(t, np.rad2deg(euler_angles[:,0]), label='Roll (deg)')
            plt.plot(t, np.rad2deg(euler_angles[:,1]), label='Pitch (deg)')
            plt.plot(t, np.rad2deg(euler_angles[:,2]), label='Yaw (deg)')
            plt.title('Estimated Orientation (Euler Angles)')
            plt.xlabel('Time (s)')
            plt.ylabel('Angle (degrees)')
            plt.legend()
            plt.grid(True)
            plt.show()

            # 可视化速度
            fig = plt.figure(figsize=(10,4))
            axv = fig.add_subplot(131)
            axv.plot(t, vel[:,0], label='vel x')
            axv.plot(t, vel[:,1], label='vel y')
            axv.plot(t, vel[:,2], label='vel z')
            axv.set_title('Velocity in e0'); axv.set_xlabel('Time (s)'); axv.set_ylabel('Velocity (m/s)')
            axv.legend(); axv.grid(True)

            # 可视化加速度与静止段
            visualize_results(t, lin_acc, lin_acc_hp, gyr, static_mask)
            fig = plt.figure(figsize=(10,5))
            ax3d = fig.add_subplot(121, projection='3d')
            tt = np.arange(N)
            sc = ax3d.scatter(pos[:,0], pos[:,1], pos[:,2], c=tt, cmap='viridis', s=10)
            ax3d.plot(pos[:,0], pos[:,1], pos[:,2], color='gray', alpha=0.4)
            
            # 添加坐标轴标签
            ax3d.set_xlabel('X (m)')
            ax3d.set_ylabel('Y (m)')
            ax3d.set_zlabel('Z (m)')
            ax3d.set_title('pos3 in e0')
            fig.colorbar(sc, ax=ax3d, label='time step')
            ax2 = fig.add_subplot(122)
            sc2 = ax2.scatter(pos2norm[:,0], pos2norm[:,1], c=tt, cmap='viridis', s=10)
            ax2.plot(pos2norm[:,0], pos2norm[:,1], color='gray', alpha=0.4)
            ax2.set_title('pos2norm (PCA)')
            fig.colorbar(sc2, ax=ax2, label='time step')
            plt.tight_layout(); plt.show()
        except Exception as e:
            print("绘图出错：", e)

    return dict(
        t=t, pos3=pos, vel=vel, lin_acc=lin_acc_hp, quat=quats,
        pos2=pos2, pos2norm=pos2norm, static_mask=static_mask,
        g_e0=g_e0, gyro_bias=gyro_bias, a_hat=a_hat, b_hat=b_hat
    )

def compute_e0_trajectory(file_or_data,
                          sample_rate: Optional[float] = None,
                          **kwargs) -> Dict[str, np.ndarray]:
    cfg = Config()
    if sample_rate is not None:
        cfg.sample_rate = float(sample_rate)
    for k,v in kwargs.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    
    # 打印各参数
    print("配置参数：")
    for field in cfg.__dataclass_fields__:
        print(f"  {field}: {getattr(cfg, field)}")

    if isinstance(file_or_data, str):
        data = load_imu_data(file_or_data, cfg)
        return compute_e0_trajectory_from_arrays(data, cfg)
    elif isinstance(file_or_data, dict):
        return compute_e0_trajectory_from_arrays(file_or_data, cfg)
    else:
        raise TypeError("需要文件路径(str)或数据字典(dict)")

# =========================
# 直接运行示例
# =========================
if __name__ == "__main__":
    # 你的静止数据文件路径（可改成你当前的）
    path = "./test_data/MRU20251025/20251025_161900.txt"   # 或 "/mnt/data/20251025_161720.txt"
    fs = 200.0

    res = compute_e0_trajectory(
        path,
        sample_rate=fs,
        gyro_unit='deg_per_hour',    # 如为°/s，改成 'deg_per_sec'
        visualize=True,
        apply_pca=True,

        # --- 关键参数，可按需要微调 ---
        drift_use_init_win=True,     # 开头5s捕捉漂移
        init_static_s=5.0,           # 你的静止窗口长度
        hp_cutoff_hz=0.05,          # 速度积分前高通
        acc_std_thresh=0.015,        # 静止检测阈值
        gyro_thresh_deg_s=0.3,
        strong_static_clamp=True     # 静止强约束
    )

    print("静止占比：", float(res['static_mask'].mean()))
    print("gyro_bias (deg/s)：", np.rad2deg(res['gyro_bias']))
    print("a_hat 等效加速度偏置 (m/s^2)：", res['a_hat'])
    print("b_hat 初始速度偏置   (m/s)：",   res['b_hat'])
    print("pos3 形状：", res['pos3'].shape)
