# -*- coding: utf-8 -*-
# =====================================================================
# dsp.py —— 接收机 DSP（对应论文 Fig.1）
#
#   matched_filter  RRC 匹配滤波(单位能量, 信号功率/噪声方差均不变)
#   cd_compensate   色散补偿(频域精确反演, 与 channel.ssfm 的线性算子互逆)
#   mimo_cma        2x2 MIMO CMA 均衡(常数模: e = y(|y|²−R))
#
# 接收 DSP 链(见 generate_dataset.gen_one_sample):
#   Coherent Rx(MF) → I/Q 失衡补偿 → CD 补偿 → 定时恢复 → CMA
#   (I/Q 失衡与定时恢复的盲补偿实现见 impairments.py)
#
# 说明:
#   * 论文正文明确写的是 "constant modulus algorithm (CMA)-equalized
#     signal"(其 Fig.2 显示的也是被 CMA 压缩后的星座), 故均衡器只用 CMA。
#   * 均衡输出存在相位模糊/极性翻转 —— 45 维特征全部基于幅度统计,
#     对相位不敏感, 故无需 CPR。
#   * 更新式 w ← w − μ·e·conj(x) 为标准形式(Wirtinger 梯度 + GNU Radio
#     同款实现), μ 除以瞬时快照功率做归一化以保收敛稳定。
# =====================================================================
import numpy as np

from channel import beta2_from_D


def matched_filter(sig, h):
    """RRC 匹配滤波。h 为单位能量 RRC(与发射端相同)。"""
    return np.vstack([np.convolve(s, h) for s in sig])


def cd_compensate(sig, length_km, D_ps_nm_km, fs, wavelength=1550e-9):
    """频域精确色散补偿: H(ω) = exp(-j·β2/2·ω²·L), 与 SSFM 线性步互逆。"""
    if length_km <= 0:
        return sig
    beta2 = beta2_from_D(D_ps_nm_km, wavelength)
    N = sig.shape[1]
    w = 2.0 * np.pi * np.fft.fftfreq(N, 1.0 / fs)
    H = np.exp(-1j * 0.5 * beta2 * w ** 2 * length_km * 1e3)
    return np.fft.ifft(np.fft.fft(sig, axis=1) * H[None, :], axis=1)


def _mimo_adaptive(sig, n_taps, mu, err_fn, w0=None):
    """2x2 MIMO 自适应均衡公共内核(T/2 间隔抽头, 输出符号率)。

    err_fn(y) -> complex 误差; CMA 与 MMA 只差这一个误差函数。
    """
    x0 = np.ascontiguousarray(sig[0])
    x1 = np.ascontiguousarray(sig[1])
    N = sig.shape[1]
    L = n_taps
    M = L // 2
    n_out = (N - L) // 2 + 1

    if w0 is None:
        wxx = np.zeros(L, dtype=np.complex128); wxx[M] = 1.0
        wyy = np.zeros(L, dtype=np.complex128); wyy[M] = 1.0
        wxy = np.zeros(L, dtype=np.complex128)
        wyx = np.zeros(L, dtype=np.complex128)
    else:
        wxx, wxy, wyx, wyy = (np.array(w, dtype=np.complex128) for w in w0)

    yx = np.empty(n_out, dtype=np.complex128)
    yy = np.empty(n_out, dtype=np.complex128)
    dot, conj = np.dot, np.conj

    for k in range(n_out):
        n = k + k                                  # 每 2 个输入采样出 1 个符号
        xs = x0[n:n + L]
        ys = x1[n:n + L]
        a = dot(wxx, xs) + dot(wxy, ys)            # X 偏振输出
        b = dot(wyx, xs) + dot(wyy, ys)            # Y 偏振输出
        yx[k] = a
        yy[k] = b
        pn = (dot(xs, conj(xs)).real + dot(ys, conj(ys)).real) + 1e-12
        g = mu / pn                                # 归一化步长
        ca = g * err_fn(a)
        cb = g * err_fn(b)
        wxx -= ca * conj(xs)
        wxy -= ca * conj(ys)
        wyx -= cb * conj(xs)
        wyy -= cb * conj(ys)

    return np.vstack([yx, yy]), (wxx, wxy, wyx, wyy)


def mimo_cma(sig, n_taps=13, mu=1e-3, R=1.0, w0=None):
    """2x2 MIMO CMA(常数模)。R 由 constellation.cma_modulus 给出。"""
    return _mimo_adaptive(
        sig, n_taps, mu,
        lambda y: y * (y.real * y.real + y.imag * y.imag - R), w0)


def equalize(sig, mod_name, n_taps=13, mu=1e-3, w0=None):
    """2x2 MIMO CMA 均衡, 便于 generate_dataset.py 统一调用。
    常数模参考电平 R 由 constellation.cma_modulus 按调制格式给出。"""
    from constellation import cma_modulus
    return mimo_cma(sig, n_taps=n_taps, mu=mu, R=cma_modulus(mod_name), w0=w0)
