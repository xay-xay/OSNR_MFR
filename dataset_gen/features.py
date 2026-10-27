# -*- coding: utf-8 -*-
# =====================================================================
# features.py —— 论文 2.1 节轻量级特征提取（47 维）
#
# 输入: 均衡后的 X 偏振符号序列 x (复数, 论文要求 16384 点)
# 输出: 47 维特征 = 40-bin 幅度直方图 AH + 3 个幅度统计量 + 2 个熵
#                   + 2 个四阶累积量 |C40| / C42
#
# 特征定义逐条对应论文公式(1)~(8):
#   AH(40)     |x_i| 的 40-bin 直方图, 归一化为概率分布 p_k
#              【已修正】使用全局固定幅度范围 [0, config.AH_RANGE_MAX] 分箱,
#              不再每样本自适应 max —— 与论文 Fig.3 把 5 种格式画在同一
#              横轴范围内可比的做法一致, 且 OSNR 变化不再引起横轴漂移。
#   Dms        式(1): 1/N·Σ|x_i − μ|²,  μ = 信号复均值
#   σ_spread   式(2): sqrt(1/(N−1)·Σ(|x_i| − μ_a)²),  μ_a = 幅度均值
#   σ_vel      式(3)~(5)【已修正】: 先取幅度序列 a_i=|x_i|, 再做一阶差分
#              v_i = a_{i+1} − a_i（论文正文: "difference in amplitude between
#              consecutive samples"）, σvel 为 v 的样本标准差(ddof=1)。
#              注意: 论文式(3)的字面写法是 v_i = |x_{i+1} − x_i|（复差分取模）,
#              两种读法都保留为配置项(config.VEL_FROM_AMPLITUDE / VEL_ABS)。
#   H_log      式(6)(7): 能量熵 −Σ E_i·ln(E_i+ε),  E_i = |x_i|²/Σ_j|x_j|²
#              论文式(6)明确定义 E_i 为逐符号归一化能量, 式(7)(8)的熵都
#              建立在该 E_i 上（不是直方图概率）。
#   H_norm     式(8): 范数熵 Σ E_i^p,  p=2
#   |C40|,C42  四阶累积量(扩展特征, 用于区分恒包络 PSK 阶数):
#              M20=E[x²], M21=E[|x|²], M40=E[x⁴], M42=E[|x|⁴]
#              C40 = M40 − 3·M20²
#              C42 = M42 − |M20|² − 2·M21²
#              归一化后取 |C40| 与实值 C42 —— 二者对全局相位旋转不变,
#              故不受 CMA 相位模糊影响。
#
# 归一化: 论文输入为"归一化特征"; 这里保存原始特征 +
# 数据集级 z-score 统计量(mean/std), 训练时套用。
# =====================================================================
import numpy as np

import config as C

N_BINS = 40
EPS = 1e-12

FEATURE_NAMES = (["ah_%02d" % i for i in range(N_BINS)]
                 + ["dms", "sigma_spread", "sigma_vel", "h_log", "h_norm",
                    "c40_abs", "c42"])
N_FEATURES = len(FEATURE_NAMES)


def _cumulants(x):
    """归一化四阶累积量 (|C40|, C42)，对全局相位旋转不变。"""
    xc = x - x.mean()
    m20 = np.mean(xc ** 2)
    m21 = np.mean(np.abs(xc) ** 2)
    m40 = np.mean(xc ** 4)
    m42 = np.mean(np.abs(xc) ** 4)
    den = m21 ** 2 + EPS
    c40 = (m40 - 3.0 * m20 ** 2) / den
    c42 = (m42 - np.abs(m20) ** 2 - 2.0 * m21 ** 2) / den
    return float(np.abs(c40)), float(np.real(c42))


def extract_features(x, ah_range=None):
    """从均衡后符号序列提取 47 维特征。x: (N,) complex 数组。

    ah_range: 直方图分箱范围 (lo, hi); None 用 config.AH_RANGE_MAX
    入口功率归一化(config.FEAT_POWER_NORM, 方案 B): x ← x/√E|x|²,
    把每样本的绝对尺度除掉, 星座环被钉在 1.0 —— AH 只剩形状信息,
    不再携带"CMA 增益随 OSNR 漂移"的尺度线索(对齐论文数据难度)。
    """
    x = np.asarray(x)
    if C.FEAT_POWER_NORM:
        x = x / np.sqrt(np.mean(np.abs(x) ** 2) + 1e-30)
    a = np.abs(x)
    N = len(x)
    if ah_range is None:
        ah_range = (0.0, C.AH_RANGE_MAX)

    # ---- AH: 全局固定范围分箱 -> 概率分布(超过上界的样本落入末箱) ----
    counts, _ = np.histogram(a, bins=N_BINS, range=ah_range)
    p = counts.astype(np.float64) / counts.sum()

    # ---- 式(1) 幅度均方距离 Dms (μ 为复均值) ----
    mu_c = x.mean()
    dms = np.mean(np.abs(x - mu_c) ** 2)

    # ---- 式(2) 幅度离散标准差 σ_spread (μ 为幅度均值) ----
    mu_a = a.mean()
    sigma_spread = np.sqrt(np.sum((a - mu_a) ** 2) / (N - 1))

    # ---- 式(3)~(5) 瞬时速度标准差 σ_vel ----
    if C.VEL_FROM_AMPLITUDE:
        v = np.diff(a)                     # 【修正】幅度序列差分
    else:
        v = np.abs(np.diff(x))             # 论文式(3)字面读法: 复差分取模
    if C.VEL_ABS:
        v = np.abs(v)
    sigma_vel = np.std(v, ddof=1)          # 1/(N−2)·Σ(v−v̄)² 开根

    # ---- 式(6) 逐符号归一化能量 E_i = |x_i|²/Σ_j|x_j|² ----
    E = (a ** 2) / np.sum(a ** 2)

    # ---- 式(7) 能量熵 H_log ----
    h_log = -np.sum(E * np.log(E + EPS))

    # ---- 式(8) 范数熵 H_norm (p=2) ----
    h_norm = np.sum(E ** 2)

    # ---- 扩展: 四阶累积量 |C40| / C42 (区分 PSK 阶数) ----
    c40_abs, c42 = _cumulants(x)

    return np.concatenate([p, [dms, sigma_spread, sigma_vel, h_log, h_norm,
                               c40_abs, c42]])


def ah_clip_fraction(x, ah_range=None):
    """诊断用: 幅度超过固定分箱上界的样本占比(应为极小值)。"""
    if ah_range is None:
        ah_range = (0.0, C.AH_RANGE_MAX)
    a = np.abs(np.asarray(x))
    return float(np.mean(a > ah_range[1]))


def zscore_stats(X):
    """数据集级 z-score 统计量(按特征列)。X: (n, 47)。"""
    return X.mean(axis=0), X.std(axis=0) + 1e-12


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(16384) + 1j * rng.standard_normal(16384))
    f = extract_features(x)
    print("feature dim =", len(f), "(应为 47)")
    print("AH 上界 =", C.AH_RANGE_MAX)
    print("σ_vel 基于幅度序列差分 =", C.VEL_FROM_AMPLITUDE)
    print("超范围样本占比 = %.3f%%" % (100 * ah_clip_fraction(x)))
