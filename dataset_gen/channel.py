# -*- coding: utf-8 -*-
# =====================================================================
# channel.py —— 光信道模块
#
# 实现（对应论文 VPI 仿真中的信道损伤）:
#   add_ase            ASE 噪声加载: 按 OSNR(0.1nm 参考带宽)标定噪声 PSD
#   apply_dgd          一阶 PMD/DGD: 频域 Jones 矩阵, 主态旋转角随机
#   apply_phase_noise  激光线宽: 组合线宽(TX+LO)维纳相噪, 两偏振共模
#   ssfm               SSMF 分步傅里叶传输: 色散+衰减+克尔非线性(Manakov),
#                      每跨(80km)后理想放大回入纤功率(等效 EDFA 16dB 增益)
#
# ⚠ 简化说明(请 review):
#   * 光纤中 ASE 由 EDFA 产生; 这里统一改为"跨间理想放大 + 接收端噪声加载"
#     —— OSNR 完全由噪声加载控制, 与论文实验节"通过 2x2 耦合器注入 ASE"
#     的定标方式一致。
#   * 非线性采用 Manakov 平均(系数 8/9), 单波长, 不含 WDM XPM。
# =====================================================================
import numpy as np

C_LIGHT = 299792458.0


def beta2_from_D(D_ps_nm_km, wavelength=1550e-9):
    """色散系数 D (ps/nm/km) -> 群速度色散 β2 (s^2/m)。"""
    # D [s/m^2] = D[ps/(nm·km)] * 1e-12 / (1e-9 * 1e3)
    D_si = D_ps_nm_km * 1e-12 / (1e-9 * 1e3)
    return -wavelength ** 2 * D_si / (2.0 * np.pi * C_LIGHT)


def add_ase(sig, osnr_db, rng, fs, ref_bw_hz=12.5e9, mode="0.1nm"):
    """ASE 噪声加载（两偏振独立同分布）。

    口径推导: 离散白噪声每采样点方差 σ^2 对应连续复包络 PSD
    N0 = σ^2/fs [W/Hz], 带宽 B 内噪声功率 = N0·B, 故
        OSNR = P_sig / (N0 · B_ref)  =>  σ^2 = P_sig·fs / (lin·B_ref)
    "0.1nm" 模式即上式(B_ref=12.5GHz@1550nm, 光通信标准口径)。
    "inband" 模式直接令 σ^2 = P/lin, 即匹配滤波后符号带内 SNR = OSNR。

    sig: (2, N) 波形, 每偏振功率独立测量。"""
    P = np.mean(np.abs(sig) ** 2, axis=1)            # 每偏振功率 (2,)
    lin = 10.0 ** (osnr_db / 10.0)
    if mode == "0.1nm":
        sigma2 = P * fs / (lin * ref_bw_hz)          # (2,)
    elif mode == "inband":
        sigma2 = P / lin
    else:
        raise ValueError("未知 OSNR 口径: %s" % mode)
    n = (rng.standard_normal(sig.shape)
         + 1j * rng.standard_normal(sig.shape)) * np.sqrt(sigma2 / 2.0)[:, None]
    return sig + n


def apply_dgd(sig, dgd_ps, rng, fs):
    """一阶 PMD(差分群时延 DGD)。

    频域 Jones 模型: E'(ω) = R(θ)·diag(e^{+jωτ/2}, e^{-jωτ/2})·R(θ)^H·E(ω)
    θ 为主态旋转角, 每样本随机(均匀 0~π)。"""
    tau = dgd_ps * 1e-12
    if tau <= 0:
        return sig
    N = sig.shape[1]
    w = 2.0 * np.pi * np.fft.fftfreq(N, 1.0 / fs)
    theta = rng.uniform(0.0, np.pi)
    c, s = np.cos(theta), np.sin(theta)
    F = np.fft.fft(sig, axis=1)
    Ex, Ey = F[0], F[1]
    # u = R^H E
    u1 = c * Ex + s * Ey
    u2 = -s * Ex + c * Ey
    # D
    ph = np.exp(1j * w * tau / 2.0)
    u1 *= ph
    u2 /= ph
    # E' = R u
    return np.fft.ifft(np.vstack([c * u1 - s * u2,
                                  s * u1 + c * u2]), axis=1)


def apply_phase_noise(sig, linewidth_hz, rng, fs):
    """激光线宽(组合线宽 = TX 激光器 + LO, 维纳相噪)。

    离散模型: φ[k+1] = φ[k] + sqrt(2π·Δν/fs)·n_k, n~N(0,1)。
    同一激光器作用于两偏振 -> 共模相位。相位盲的特征提取对相噪
    不敏感, 主要影响 CMA 收敛, 与论文鲁棒性实验目的一致。"""
    if linewidth_hz <= 0:
        return sig
    n = sig.shape[1]
    dphi = rng.normal(0.0, np.sqrt(2.0 * np.pi * linewidth_hz / fs), n)
    ph = np.exp(1j * np.cumsum(dphi))
    return sig * ph[None, :]


def ssfm(sig, length_km, fiber, launch_dbm=0.0, fs=56e9,
         wavelength=1550e-9, rng=None):
    """SSMF 分步傅里叶传输（对称条件: 每跨后理想放大）。

    参数:
        sig       : (2, N) 单位功率/偏振 的 PDM 波形
        length_km : 总传输距离
        fiber     : config.FIBER 参数字典
        launch_dbm: 总入纤功率(dBm) [每偏振 P/2]
    返回:
        (2, N) 传输后波形(每跨后功率回到入纤值, 末跨不再放大 -> 保持入纤功率)
    """
    if length_km <= 0:
        return sig
    beta2 = beta2_from_D(fiber["D_ps_nm_km"], wavelength)          # s^2/m
    gamma = fiber["gamma_per_Wkm"] * 1e-3                          # 1/(W·m)
    alpha_p = fiber["alpha_db_km"] * 1e-3 / 4.34294485             # 功率衰减 1/m
    alpha_f = alpha_p / 2.0                                         # 场衰减 1/m
    manakov = fiber.get("manakov", 8.0 / 9.0)
    span_km = fiber["span_km"]
    dz = fiber["ssfm_step_km"] * 1e3                                # 步长(m)

    # 入纤功率标定: 总功率 P_tot, 每偏振 P_tot/2
    p_launch_pol = 10.0 ** (launch_dbm / 10.0) * 1e-3 / 2.0
    p_in = np.mean(np.abs(sig) ** 2, axis=1)
    sig = sig * np.sqrt(p_launch_pol / p_in)[:, None]              # (2, N)

    N = sig.shape[1]
    w = 2.0 * np.pi * np.fft.fftfreq(N, 1.0 / fs)
    lin_step = np.exp((1j * 0.5 * beta2 * w ** 2 - alpha_f) * dz)  # 线性算子

    n_spans = int(round(length_km / span_km))
    n_steps_per_span = int(round(span_km * 1e3 / dz))
    A = sig.copy()
    for _ in range(n_spans):
        for _ in range(n_steps_per_span):
            # 非线性步(Manakov): 两偏振共模相位, 由总功率决定
            p_tot = np.abs(A[0]) ** 2 + np.abs(A[1]) ** 2
            A = A * np.exp(1j * manakov * gamma * p_tot * dz)[None, :]
            # 线性步(色散 + 衰减), 频域
            A = np.fft.ifft(np.fft.fft(A, axis=1) * lin_step[None, :], axis=1)
        # 跨末: 理想放大回入纤功率(等效 EDFA 增益=跨损, 无 ASE —— ASE 由
        # 接收端噪声加载统一控制)
        p_now = np.mean(np.abs(A) ** 2, axis=1)
        A = A * np.sqrt(p_launch_pol / p_now)[:, None]
    return A
