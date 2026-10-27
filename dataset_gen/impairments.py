# -*- coding: utf-8 -*-
# =====================================================================
# impairments.py —— 硬件损伤注入与对应盲补偿（RadioML 式做法，路径①）
#
# 注入(发射/接收前端, 每样本参数随机):
#   iq_imbalance   IQ 失衡: y = μx + νx*  (增益误差 ε + 相位误差 φ)
#   timing_offset  定时偏差: 分数时延 τ + 采样率偏移 SRO(线性插值重采样)
#   adc_quantize   ADC 量化(bits 位, 满量程自适应)
#
# 盲补偿(接收 DSP, 顺序: I/Q 失衡补偿 → 定时恢复, 其后接 CMA):
#   iq_comp_gs       盲 Gram-Schmidt IQ 补偿(利用 E[x²]=0 圆对称先验,
#                    κ = E[y²]/2E|y|², 残差为 O(ν²) 二阶小量 —— "洗不完美")
#   gardner_tr       Gardner 定时恢复(两遍法: 先环估计时延轨迹,
#                    再按轨迹统一重采样; 仅用 X 偏振驱动, 双偏振共用时钟)
#
# 注: 频偏注入(CFO)与四次方频偏估计(FOE)已移除 —— DSP 链不含 FOE,
#     残留频偏会使 GS-IQ 补偿失效, 故 config.IMPAIR 关闭 cfo。
# =====================================================================
import numpy as np

FS = None   # 由调用方通过 set_fs 注入, 避免与 config 循环导入


def _interp_lin(x, t):
    """线性插值取 x(t)。x: (N,) complex, t: 标量。"""
    N = len(x)
    t = min(max(t, 0.0), N - 1.0001)
    i0 = int(t)
    fr = t - i0
    return (1 - fr) * x[i0] + fr * x[i0 + 1]


def _interp_cubic(x, t):
    """Catmull-Rom 三次插值(矢量化的 pass-2 用)。x: (N,), t: array。"""
    N = len(x)
    t = np.clip(t, 1.0, N - 2.0001)
    i1 = np.floor(t).astype(np.intp)
    fr = t - i1
    i0, i2, i3 = i1 - 1, i1 + 1, i1 + 2
    f2, f3 = fr * fr, fr * fr * fr
    return 0.5 * ((-f3 + 2 * f2 - fr) * x[i0]
                  + (3 * f3 - 5 * f2 + 2) * x[i1]
                  + (-3 * f3 + 4 * f2 + fr) * x[i2]
                  + (f3 - f2) * x[i3])


# =====================================================================
# 注入
# =====================================================================
def iq_imbalance(x, eps, phi):
    """IQ 失衡: y = μx + νx*。
    eps: 增益误差(如 0.05 = I/Q 增益差 10%); phi: 相位误差(rad)。"""
    mu = np.cos(phi / 2) + 1j * eps * np.sin(phi / 2)
    nu = eps * np.cos(phi / 2) - 1j * np.sin(phi / 2)
    return mu * x + nu * np.conj(x)


def timing_offset(x, tau, sro_ppm):
    """定时偏差注入: y[m] = x[m(1+δ) + τ], δ = SRO。
    tau: 分数时延(采样, 2sps 网格); sro_ppm: 采样率偏移(ppm, 可为负)。"""
    N = x.shape[-1]
    t = np.arange(N) * (1.0 + sro_ppm * 1e-6) + tau
    t = np.clip(t, 0.0, N - 1.0001)
    i0 = np.floor(t).astype(np.intp)
    fr = t - i0
    return (1 - fr) * x[..., i0] + fr * x[..., i0 + 1]


def adc_quantize(x, bits=8):
    """ADC 量化: 满量程取本记录最大幅度的 1.1 倍。"""
    full = 1.1 * np.max(np.abs(x)) + 1e-30
    L = 2 ** (bits - 1)
    return np.round(x / full * L) / L * full


# =====================================================================
# 盲补偿
# =====================================================================
def iq_comp_gs(y):
    """盲 Gram-Schmidt IQ 补偿。
    圆对称信源 E[x²]=0, 失衡后 E[y²]=2μνP ≠ 0 → κ̂=E[y²]/2E|y|²,
    y' = y − κ̂·y* 消除一阶镜像, 残差 O(ν²)。最后恢复原功率。"""
    P = np.mean(np.abs(y) ** 2)
    if P < 1e-30:
        return y
    kappa = np.mean(y ** 2) / (2.0 * P)
    y1 = y - kappa * np.conj(y)
    P1 = np.mean(np.abs(y1) ** 2)
    return y1 * np.sqrt(P / (P1 + 1e-30))


def gardner_tr(sig, kp=0.02, ki=1e-4):
    """Gardner 定时恢复(两遍法, 盲)。

    pass1: Gardner 环(仅 X 偏振驱动)逐符号估计时延轨迹 tau[n]。
           TED: e = Re{(y(t−1) − y(t+1))·conj(y(t))}, t 为符号中心选通。
           误差限幅防积分器饱和; PI 环(ki≈kp²/4 临界阻尼)。
    pass2: 按 tau 轨迹(线性内插到采样栅格)对两偏振统一三次插值重采样。
    收敛后 taû(n) ≈ −τ_true − 2n·δ(SRO), pass2 的 x̂[k]=y(k+taû(k))
    恰好同时去除静态时延与时间翘曲。返回 (2, N-4) 2sps 信号 + 轨迹。"""
    x = sig[0]
    N = sig.shape[1]
    nsym = N // 2 - 2
    tau = 0.0
    integ = 0.0
    taus = np.empty(nsym)
    for n in range(nsym):
        t = 2 * n + 1 + tau                       # 符号中心选通位置
        yA = _interp_lin(x, t - 1)                # 前一符号中心
        yB = _interp_lin(x, t)                    # 半符号点
        yC = _interp_lin(x, t + 1)                # 后一符号中心
        e = (yA - yC).real * yB.real + (yA - yC).imag * yB.imag
        if e > 1.0:
            e = 1.0
        elif e < -1.0:
            e = -1.0
        integ += ki * e
        tau += kp * e + integ
        taus[n] = tau
    # pass2: 轨迹内插到采样栅格并重采样
    sym_pos = 2 * np.arange(nsym) + 1
    m = np.arange(N - 4)
    tau_m = np.interp(m, sym_pos, taus)
    pos = m + tau_m
    out = np.vstack([_interp_cubic(sig[p], pos) for p in range(sig.shape[0])])
    return out, taus


# =====================================================================
# 每样本损伤参数抽取(确定性, 由样本 rng 驱动)
# =====================================================================
def draw_impairments(rng, imp):
    """按 config.IMPAIR 抽取一组本样本的损伤参数。"""
    def u(lo, hi):
        return float(rng.uniform(lo, hi))

    p = dict(do=imp.get("enable", False))
    if not p["do"]:
        return p
    if imp.get("tx_iq", False):
        e, ph = imp["eps"], np.deg2rad(imp["phi_deg"])
        p["tx_eps"] = u(*e) * rng.choice([-1, 1])
        p["tx_phi"] = u(*ph) * rng.choice([-1, 1])
    if imp.get("rx_iq", False):
        e, ph = imp["eps"], np.deg2rad(imp["phi_deg"])
        p["rx_eps"] = u(*e) * rng.choice([-1, 1])
        p["rx_phi"] = u(*ph) * rng.choice([-1, 1])
    if imp.get("timing", False):
        p["tau"] = u(*imp["tau_samples"])
        p["sro"] = u(*imp["sro_ppm"]) * rng.choice([-1, 1])
    if imp.get("adc", False):
        p["adc_bits"] = int(imp["adc_bits"])
    return p


def inject_tx(wav, p):
    """发射端 IQ 失衡(每偏振独立的调制器误差)。"""
    if "tx_eps" not in p:
        return wav
    return np.vstack([iq_imbalance(wav[0], p["tx_eps"], p["tx_phi"]),
                      iq_imbalance(wav[1], p["tx_eps"], p["tx_phi"])])


def inject_rx(wav, p, fs):
    """接收端损伤链: RX IQ → 定时 → ADC。"""
    if "rx_eps" in p:
        wav = np.vstack([iq_imbalance(wav[0], p["rx_eps"], p["rx_phi"]),
                         iq_imbalance(wav[1], p["rx_eps"], p["rx_phi"])])
    if "tau" in p:
        wav = timing_offset(wav, p["tau"], p["sro"])
    if "adc_bits" in p:
        wav = adc_quantize(wav, p["adc_bits"])
    return wav


def iq_compensate(rx, p):
    """I/Q 失衡盲补偿(GS, 双偏振)。仅在存在 IQ 损伤时执行。"""
    if ("tx_eps" in p) or ("rx_eps" in p):
        rx = np.vstack([iq_comp_gs(rx[0]), iq_comp_gs(rx[1])])
    return rx


def timing_recover(rx, p):
    """Gardner 定时恢复。仅在存在定时损伤时执行。"""
    if "tau" in p:
        rx, _ = gardner_tr(rx)
    return rx
