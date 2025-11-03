# -*- coding: utf-8 -*-
"""
IMU e0 轨迹重建（模块化离线版）
- 单文件脚本，可直接运行
- 主坐标系：初始时刻的载体坐标系 e0（q 初值为单位四元数，表示 body->e0 恒等）
- 前 init_static_s 秒静止用于：重力幅值标定 & 陀螺零偏估计
- 线加速度：acc_e0 - g_e0；再做轻微高通(默认 0.3Hz)后积分，配合静止检测 + 段尾回溯线性归零，抑制漂移
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
    sample_rate: float = 200.0            # 采样频率(Hz)，若文件无时间戳则用它生成
    init_static_s: float = 5.0            # 文件开头静止时长(s)，用于估计g与陀螺零偏
    gyro_unit: str = 'deg_per_hour'       # 'deg_per_hour' | 'deg_per_sec' | 'rad_per_sec'
    g0: float = 9.80665                   # 重力常数
    hp_cutoff_hz: float = 0.3             # 高通截止频率，用于速度积分前的线加速度去漂
    acc_std_win_s: float = 0.5            # 静止检测：加速度高通模值的std窗口(s)
    acc_std_thresh: float = 0.05          # 静止检测：加速度高通模值std阈值 (m/s^2) 建议0.05~0.2
    gyro_thresh_deg_s: float = 1.0        # 静止检测：角速度阈值 (deg/s) 建议1~3
    static_margin_s: float = 0.1          # 静止边界膨胀(s)
    min_static_s: float = 0.2             # 过短静止岛剔除(s)
    visualize: bool = True                # 是否绘图
    apply_pca: bool = True                # 是否输出PCA投影/归一
    # 读文件
    file_has_header: bool = False
    file_sep_regex: str = r'\s+'
    columns_no_header: Tuple[str,...] = ('ax','ay','az','gx','gy','gz','index')  # 无表头时默认列名


# =========================
# 工具函数：滤波、移动统计、四元数
# =========================
def butter_highpass(data: np.ndarray, fc: float, fs: float, order: int = 2) -> np.ndarray:
    """对 N×D 或 N 向量数据逐列进行高通滤波；数据太短时直接返回原值"""
    if fc <= 0 or fc >= fs/2 or data.shape[0] < 3*order+1:
        return data.copy()
    b, a = butter(order, fc/(fs/2), btype='high')
    data = np.atleast_2d(data)
    out = np.zeros_like(data)
    for j in range(data.shape[1]):
        out[:, j] = filtfilt(b, a, data[:, j], method="pad")
    return out if out.shape[0] > 1 else out.ravel()


def moving_std_1d(x: np.ndarray, win_len: int) -> np.ndarray:
    """简单滑动窗口标准差（边缘复制）。win_len<2 时返回全零"""
    if win_len <= 1:
        return np.zeros_like(x)
    pad = win_len // 2
    xp = np.pad(x, (pad, pad), mode='edge')
    ker = np.ones(win_len) / win_len
    m1 = np.convolve(xp, ker, mode='valid')
    m2 = np.convolve(xp*xp, ker, mode='valid')
    std = np.sqrt(np.maximum(0.0, m2 - m1*m1))
    return std


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > 0 else np.array([1.0,0.0,0.0,0.0])


def quat_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2], dtype=float)


def quat_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    """旋转向量->四元数（角度=范数，轴=归一化rotvec）"""
    theta = np.linalg.norm(rotvec)
    if theta < 1e-12:
        return quat_normalize(np.concatenate(([1.0], 0.5*rotvec)))
    axis = rotvec / theta
    w = np.cos(theta/2.0)
    s = np.sin(theta/2.0)
    xyz = axis * s
    return np.array([w, xyz[0], xyz[1], xyz[2]])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """用四元数 q（body->e0）旋转 3D 向量 v"""
    w,x,y,z = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1-2*(x*x+z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1-2*(x*x+y*y)]
    ], dtype=float)
    return R @ v


# =========================
# 模块1：数据读取 / 时间轴
# =========================
def load_imu_data(file_path: str, cfg: Config) -> Dict[str, np.ndarray]:
    """读取 IMU 文本文件，返回 dict: time, ax, ay, az, gx, gy, gz  (角速度单位转换到 rad/s)"""
    if cfg.file_has_header:
        df = pd.read_csv(file_path, sep=cfg.file_sep_regex)
    else:
        df = pd.read_csv(file_path, header=None, sep=cfg.file_sep_regex)
        # 兜底列名
        ncol = df.shape[1]
        cols = list(cfg.columns_no_header[:ncol])
        df.columns = cols

    # 取需要列（至少 ax,ay,az,gx,gy,gz）
    need = ['ax','ay','az','gx','gy','gz']
    for k in need:
        if k not in df.columns:
            raise ValueError(f"缺少必要列: {k}")

    df = df.dropna(how='any')
    # 转为float
    for k in need:
        df[k] = pd.to_numeric(df[k], errors='coerce')
    df = df.dropna(how='any')
    N = df.shape[0]

    # 时间轴：若文件无时间列，则按采样率生成
    t = np.linspace(0, (N-1)/cfg.sample_rate, N)

    # 角速度单位换算 -> rad/s
    gx, gy, gz = df['gx'].to_numpy(), df['gy'].to_numpy(), df['gz'].to_numpy()
    if cfg.gyro_unit == 'deg_per_hour':
        gx = gx / 3600.0
        gy = gy / 3600.0
        gz = gz / 3600.0
        gx = np.deg2rad(gx); gy = np.deg2rad(gy); gz = np.deg2rad(gz)  # -> rad/s
    elif cfg.gyro_unit == 'deg_per_sec':
        gx = np.deg2rad(gx); gy = np.deg2rad(gy); gz = np.deg2rad(gz)  # -> rad/s
    elif cfg.gyro_unit == 'rad_per_sec':
        pass
    else:
        raise ValueError("cfg.gyro_unit 必须是 'deg_per_hour'|'deg_per_sec'|'rad_per_sec'")

    return dict(
        time=t,
        ax=df['ax'].to_numpy(),
        ay=df['ay'].to_numpy(),
        az=df['az'].to_numpy(),
        gx=gx, gy=gy, gz=gz
    )


# =========================
# 模块2：初始静止估计（重力幅值标定 + 陀螺零偏）
# =========================
def estimate_init(acc_m_s2: np.ndarray,
                  gyr_rad_s: np.ndarray,
                  fs: float,
                  init_static_s: float,
                  g0: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    acc_m_s2: (N,3) m/s^2
    gyr_rad_s: (N,3) rad/s
    返回：g_e0 (3,), gyro_bias_rad_s (3,), scale(float)
    - g_e0 定义在 e0（初始body）坐标系下
    - scale 用于将加速度标定到 g0
    """
    N0 = max(1, int(round(init_static_s * fs)))
    g_vec_raw = acc_m_s2[:N0].mean(axis=0)
    g_norm = np.linalg.norm(g_vec_raw)
    if g_norm < 1e-6:
        g_norm = g0
    scale = g0 / g_norm
    g_e0 = g_vec_raw * scale
    gyro_bias = gyr_rad_s[:N0].mean(axis=0)
    return g_e0, gyro_bias, scale


# =========================
# 模块3：姿态积分（仅陀螺）
# =========================
def integrate_orientation(gyr_rad_s: np.ndarray,
                          dt_arr: np.ndarray) -> np.ndarray:
    """
    仅用陀螺积分姿态（四元数），q0=单位四元数，q为 body->e0
    注意：此处 e0=初始body，因此初始时 q=I（恒等）
    """
    N = gyr_rad_s.shape[0]
    quats = np.zeros((N, 4))
    q = np.array([1.0, 0.0, 0.0, 0.0])
    for i in range(N):
        rotvec = gyr_rad_s[i] * dt_arr[i]
        dq = quat_from_rotvec(rotvec)
        q = quat_mult(q, dq)
        q = quat_normalize(q)
        quats[i] = q
    return quats


# =========================
# 模块4：线加速度（旋转至e0并减重力）
# =========================
def compute_linear_acc(acc_body: np.ndarray,
                       quats: np.ndarray,
                       g_e0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    将加速度从 body 旋转到 e0，并减去 g_e0，得到线加速度
    返回：acc_e0(N,3), lin_acc(N,3)
    """
    N = acc_body.shape[0]
    acc_e0 = np.zeros((N,3))
    lin_acc = np.zeros((N,3))
    for i in range(N):
        a_e = quat_rotate(quats[i], acc_body[i])
        acc_e0[i] = a_e
        lin_acc[i] = a_e - g_e0
    return acc_e0, lin_acc


# =========================
# 模块5：静止检测
# =========================
def detect_stationary(lin_acc_e0: np.ndarray,
                      gyr_rad_s: np.ndarray,
                      fs: float,
                      acc_std_win_s: float,
                      acc_std_thresh: float,
                      gyro_thresh_deg_s: float) -> np.ndarray:
    """
    使用 高通后的线加速度模值 的滑动std 与 角速度幅值阈值 进行静止检测
    返回：static_mask (N,) bool
    """
    # 用线加速度做一个轻微高通再求模值
    hp_cutoff_det = 0.3  # 用于检测的小高通，固定即可
    acc_hp = butter_highpass(lin_acc_e0, hp_cutoff_det, fs, order=2)
    mag = np.linalg.norm(acc_hp, axis=1)
    win_len = max(1, int(round(acc_std_win_s * fs)))
    mag_std = moving_std_1d(mag, win_len)

    # 陀螺范数 deg/s
    gyro_norm_deg_s = np.linalg.norm(np.rad2deg(gyr_rad_s), axis=1)
    print(mag_std.shape, gyro_norm_deg_s.shape)
    static_mask = (mag_std[:-1] < acc_std_thresh) & (gyro_norm_deg_s < gyro_thresh_deg_s)
    return static_mask


def expand_and_clean_static(static_mask: np.ndarray, fs: float,
                            static_margin_s: float, min_static_s: float) -> np.ndarray:
    """边界膨胀 + 移除过短静止岛"""
    mask = static_mask.copy()
    N = len(mask)
    # 膨胀
    m = int(round(static_margin_s * fs))
    if m > 0:
        padded = mask.copy()
        for k in range(1, m+1):
            padded[:-k] = padded[:-k] | mask[k:]
            padded[k:]  = padded[k:]  | mask[:-k]
        mask = padded
    # 清除短岛
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
# 模块6：速度/位置积分 + 段尾回溯归零
# =========================
def integrate_vel_pos(lin_acc_hp: np.ndarray,
                      dt_arr: np.ndarray,
                      static_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    梯形积分速度/位置；当检测到“由动入静”的瞬间，回溯到本段运动起点，
    将速度从 0 线性过渡到段末速度，再整体减掉该线性项，实现段末归零（ZUPT风格）
    """
    N = lin_acc_hp.shape[0]
    vel = np.zeros((N,3))
    pos = np.zeros((N,3))

    for i in range(1, N):
        dt = dt_arr[i]
        a_prev, a_curr = lin_acc_hp[i-1], lin_acc_hp[i]
        vel[i] = vel[i-1] + 0.5 * (a_prev + a_curr) * dt
        pos[i] = pos[i-1] + 0.5 * (vel[i-1] + vel[i]) * dt

        # 刚进入静止：回溯修正前一段运动
        if static_mask[i] and (not static_mask[i-1]):
            # 找到这段运动的起点
            j = i - 1
            while j >= 0 and not static_mask[j]:
                j -= 1
            mstart = j + 1
            mend = i - 1
            Ls = mend - mstart + 1
            if Ls > 0:
                vend = vel[mend].copy()
                # 速度线性回零
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
                        pos[idx] = p0 + 0.5 * (prev_v + vel[idx]) * dt_arr[idx]
                    else:
                        pos[idx] = pos[idx-1] + 0.5 * (vel[idx-1] + vel[idx]) * dt_arr[idx]
                # 将当前静止点速度置零，位置对齐段末
                vel[i] = np.zeros(3)
                pos[i] = pos[mend].copy()
    return vel, pos


# =========================
# 模块7：PCA 投影（可选）
# =========================
def pca_project_2d(pos3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """将3D轨迹去均值后做SVD，取前2主成分作为2D投影；并做范围归一化"""
    P = pos3 - pos3.mean(axis=0, keepdims=True)
    if P.shape[0] < 2:
        return np.zeros((P.shape[0],2)), np.zeros((P.shape[0],2))
    # SVD
    U, S, Vt = np.linalg.svd(P, full_matrices=False)
    basis2 = Vt[:2, :]       # 2x3
    pos2 = P @ basis2.T      # N x 2
    rng = np.max(np.ptp(pos2, axis=0)) if pos2.size > 0 else 0.0
    pos2norm = pos2 / rng if rng > 0 else pos2.copy()
    return pos2, pos2norm


# =========================
# 模块8：可视化
# =========================
def visualize_results(t: np.ndarray,
                      lin_acc: np.ndarray,
                      lin_acc_hp: np.ndarray,
                      gyr_rad_s: np.ndarray,
                      static_mask: np.ndarray):
    """简要九宫格可视化"""
    fig = plt.figure(figsize=(16, 8))
    ax1 = plt.subplot(331); ax1.plot(lin_acc[:,0]); ax1.set_title('lin_acc x (e0)')
    ax2 = plt.subplot(334); ax2.plot(lin_acc[:,1]); ax2.set_title('lin_acc y (e0)')
    ax3 = plt.subplot(337); ax3.plot(lin_acc[:,2]); ax3.set_title('lin_acc z (e0)')

    ax4 = plt.subplot(332); ax4.plot(lin_acc_hp[:,0]); ax4.set_title('lin_acc HP x')
    ax5 = plt.subplot(335); ax5.plot(lin_acc_hp[:,1]); ax5.set_title('lin_acc HP y')
    ax6 = plt.subplot(338); ax6.plot(lin_acc_hp[:,2]); ax6.set_title('lin_acc HP z')

    ax7 = plt.subplot(333); ax7.plot(np.rad2deg(gyr_rad_s)[:,0]); ax7.set_title('gyro x (deg/s)')
    ax8 = plt.subplot(336); ax8.plot(np.rad2deg(gyr_rad_s)[:,1]); ax8.set_title('gyro y (deg/s)')
    ax9 = plt.subplot(339); ax9.plot(np.rad2deg(gyr_rad_s)[:,2]); ax9.set_title('gyro z (deg/s)')

    # 叠加静止mask阴影
    for ax in [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]:
        ymin, ymax = ax.get_ylim()
        # 将静止段以淡灰色背景标出
        in_seg = False; s = 0
        N = len(static_mask)
        for i in range(N):
            if static_mask[i] and not in_seg:
                in_seg = True; s = i
            if (in_seg and (i==N-1 or not static_mask[i])):
                e = i if not static_mask[i] else i
                ax.axvspan(s, e, color='lightgray', alpha=0.3)
                in_seg = False
        ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    plt.show()


# =========================
# 主流程：compute_e0_trajectory
# =========================
def compute_e0_trajectory_from_arrays(data: Dict[str, np.ndarray],
                                      cfg: Config) -> Dict[str, np.ndarray]:
    """
    直接从数组字典进入（兼容原接口）
    data: keys ['time','ax','ay','az','gx','gy','gz']
          加速度 m/s^2；角速度单位将按 cfg.gyro_unit 解释并换算为 rad/s（若已是rad/s则设定对应单位）
    """
    # 时间
    if ('time' in data) and (data['time'] is not None):
        t = np.asarray(data['time'], dtype=float)
        t = t - t[0]
    else:
        N = len(data['ax'])
        t = np.linspace(0, (N-1)/cfg.sample_rate, N)

    dt_arr = np.diff(t, prepend=t[0])
    if len(dt_arr) > 1:
        med = np.median(dt_arr[1:])
        dt_arr[0] = med
    fs = 1.0 / np.median(dt_arr)

    # 传感器
    ax = np.asarray(data['ax'], dtype=float)
    ay = np.asarray(data['ay'], dtype=float)
    az = np.asarray(data['az'], dtype=float)
    gx = np.asarray(data['gx'], dtype=float)
    gy = np.asarray(data['gy'], dtype=float)
    gz = np.asarray(data['gz'], dtype=float)
    N = len(ax)
    assert all(len(x)==N for x in [ay,az,gx,gy,gz]), "各通道长度不一致"

    # 若角速度单位非rad/s，按cfg进行换算
    if cfg.gyro_unit == 'deg_per_hour':
        gx = np.deg2rad(gx/3600.0); gy = np.deg2rad(gy/3600.0); gz = np.deg2rad(gz/3600.0)
    elif cfg.gyro_unit == 'deg_per_sec':
        gx = np.deg2rad(gx); gy = np.deg2rad(gy); gz = np.deg2rad(gz)
    elif cfg.gyro_unit == 'rad_per_sec':
        pass
    else:
        raise ValueError("cfg.gyro_unit 设置非法")

    acc_m = np.vstack([ax, ay, az]).T           # m/s^2
    gyr = np.vstack([gx, gy, gz]).T             # rad/s

    # 初始静止估计（重力幅值标定 + 陀螺零偏）
    g_e0, gyro_bias, scale = estimate_init(acc_m, gyr, fs, cfg.init_static_s, cfg.g0)
    acc_m *= scale
    gyr = gyr - gyro_bias

    # 姿态积分（q: body->e0）
    quats = integrate_orientation(gyr, dt_arr)

    # 线加速度（旋到e0并减 g_e0）
    acc_e0, lin_acc = compute_linear_acc(acc_m, quats, g_e0)

    # 静止检测
    static_mask = detect_stationary(lin_acc, gyr, fs,
                                    cfg.acc_std_win_s, cfg.acc_std_thresh, cfg.gyro_thresh_deg_s)
    static_mask = expand_and_clean_static(static_mask, fs, cfg.static_margin_s, cfg.min_static_s)

    # 积分前对线加速度做轻微高通，抑制残余偏置
    lin_acc_hp = butter_highpass(lin_acc, cfg.hp_cutoff_hz, fs, order=2)

    # 速度/位置积分 + 段尾回溯归零
    vel, pos = integrate_vel_pos(lin_acc_hp, dt_arr, static_mask)

    # PCA 投影（可选）
    if cfg.apply_pca:
        pos2, pos2norm = pca_project_2d(pos)
    else:
        pos2 = np.zeros((N,2)); pos2norm = np.zeros((N,2))

    # 可视化（可选）
    if cfg.visualize:
        try:
            # 绘制静止检测结果等
            fig = plt.figure(figsize=(16, 8))
            
            visualize_results(t[:N], lin_acc, lin_acc_hp, gyr, static_mask)
            # 3D 轨迹（彩色表示时间）
            fig = plt.figure(figsize=(10,5))
            ax3d = fig.add_subplot(121, projection='3d')
            tt = np.arange(N)
            sc = ax3d.scatter(pos[:,0], pos[:,1], pos[:,2], c=tt, cmap='viridis', s=10)
            ax3d.plot(pos[:,0], pos[:,1], pos[:,2], color='gray', alpha=0.4)
            ax3d.set_title('pos3 in e0')
            fig.colorbar(sc, ax=ax3d, label='time step')
            # 2D PCA
            ax2 = fig.add_subplot(122)
            sc2 = ax2.scatter(pos2norm[:,0], pos2norm[:,1], c=tt, cmap='viridis', s=10)
            ax2.plot(pos2norm[:,0], pos2norm[:,1], color='gray', alpha=0.4)
            ax2.set_title('pos2norm (PCA)')
            fig.colorbar(sc2, ax=ax2, label='time step')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print("绘图出错：", e)

    return dict(
        t=t[:N],
        pos3=pos,
        vel=vel,
        lin_acc=lin_acc_hp,      # 高通后的线加速度（用于积分/静止检测）
        quat=quats,
        pos2=pos2,
        pos2norm=pos2norm,
        static_mask=static_mask,
        g_e0=g_e0,
        gyro_bias=gyro_bias      # (rad/s)
    )


def compute_e0_trajectory(file_or_data,
                          sample_rate: Optional[float] = None,
                          **kwargs) -> Dict[str, np.ndarray]:
    """
    兼容原始接口：
    - 若传入的是 dict(data)，则直接用数组版本
    - 若传入的是文件路径(str)，则读取后再计算
    可通过 kwargs 覆盖 Config 默认参数，例如：
      compute_e0_trajectory(path, sample_rate=200, gyro_unit='deg_per_hour',
                            acc_std_thresh=0.08, gyro_thresh_deg_s=1.5, visualize=False)
    """
    cfg = Config()
    # 覆盖 sample_rate
    if sample_rate is not None:
        cfg.sample_rate = float(sample_rate)
    # 其它参数覆盖
    for k,v in kwargs.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    if isinstance(file_or_data, str):
        data = load_imu_data(file_or_data, cfg)
        return compute_e0_trajectory_from_arrays(data, cfg)
    elif isinstance(file_or_data, dict):
        return compute_e0_trajectory_from_arrays(file_or_data, cfg)
    else:
        raise TypeError("compute_e0_trajectory 需要文件路径(str)或数据字典(dict)作为输入")


# =========================
# 命令行/直接运行示例
# =========================
if __name__ == "__main__":
    # 根据你的描述，这里用你上传的数据路径。
    # 若你的角速度原始单位是 °/h，请保持 gyro_unit='deg_per_hour'（默认即此）
    # path = "./test_data/MRU20251025/20251025_161900.txt"
    path = "./test_data/MRU20251025/20251025_161720.txt"

    # 采样率：请按你的实际数据设置
    fs = 200.0

    # 运行：若想关闭所有绘图，可以设置 visualize=False
    res = compute_e0_trajectory(
        path,
        sample_rate=fs,
        gyro_unit='deg_per_hour',   # 如果已是°/s：改为 'deg_per_sec'
        acc_std_thresh=0.05,        # 可调：0.05~0.2 之间试验
        gyro_thresh_deg_s=1.0,      # 可调：1~3 之间试验
        visualize=True,
        apply_pca=True
    )

    print("静止占比：", float(res['static_mask'].mean()))
    print("gyro_bias (deg/s)：", np.rad2deg(res['gyro_bias']))
    print("pos3 形状：", res['pos3'].shape)
