# -*- coding: utf-8 -*-
# =====================================================================
# transmitter.py —— 发射机（基带 PDM-QAM 波形生成）
#
# 链路（对应论文发射机 DSP 链: PRBS 生成 -> QAM 映射 -> 上采样）:
#   随机比特(PRBS 等效, 种子可控) -> 格雷 QAM 映射(单位功率)
#   -> 上采样 sps 倍 -> RRC 脉冲成形 -> 双偏振(X/Y 独立数据)
# 输出: (2, N) complex 波形, 每偏振平均功率归一化到 1
#       （入纤功率缩放在 channel.ssfm 内按 launch_dbm 处理）
# =====================================================================
import numpy as np
from constellation import get_constellation, random_symbols


def rrc_taps(sps, span, beta):
    """根升余弦(RRC)滤波器冲激响应，单位能量。
    span: 滤波器半长度(符号数), 总抽头数 = 2*span*sps + 1。"""
    t = np.arange(-span * sps, span * sps + 1) / float(sps)   # 以符号周期为单位
    h = np.empty_like(t)
    # t = 0 处
    h[t == 0] = 1.0 - beta + 4.0 * beta / np.pi
    # t = ±1/(4β) 处
    t1 = 1.0 / (4.0 * beta)
    idx = np.isclose(np.abs(t), t1) & (t != 0)
    h[idx] = (beta / np.sqrt(2.0)) * (
        (1 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
        + (1 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta)))
    # 其余
    idx2 = ~(idx | (t == 0))
    tt = t[idx2]
    h[idx2] = (np.sin(np.pi * tt * (1 - beta))
               + 4.0 * beta * tt * np.cos(np.pi * tt * (1 + beta))) \
              / (np.pi * tt * (1 - (4.0 * beta * tt) ** 2))
    return h / np.sqrt(np.sum(np.abs(h) ** 2))   # 单位能量


def transmit_pdm(mod_name, rng, n_symbols, sps=2, beta=0.2, span=10):
    """生成 PDM-QAM 基带波形。

    参数:
        mod_name : "QAM4"/"QAM8"/"QAM16"/"QAM32"/"QAM64"
        rng      : np.random.Generator（每样本独立种子 -> 样本间独立）
        n_symbols: 每偏振符号数（含 CMA 前置训练符号）
    返回:
        wav: (2, n_symbols*sps + 2*span*sps) complex128, X/Y 偏振,
             每偏振平均功率归一到 1。
    """
    h = rrc_taps(sps, span, beta)
    pols = []
    for _ in range(2):                       # X/Y 两偏振独立数据
        syms = random_symbols(mod_name, rng, n_symbols)
        up = np.zeros(n_symbols * sps, dtype=np.complex128)
        up[::sps] = syms                     # 上采样
        w = np.convolve(up, h)               # RRC 成形
        w /= np.sqrt(np.mean(np.abs(w) ** 2))  # 每偏振功率归一
        pols.append(w)
    return np.vstack(pols)


if __name__ == "__main__":
    # 自检: 星座回采 + 功率
    rng = np.random.default_rng(0)
    for m in ("QAM4", "QAM64"):
        wav = transmit_pdm(m, rng, 4096)
        print("%-6s wav shape=%s  Px=%.4f  Py=%.4f"
              % (m, wav.shape,
                 np.mean(np.abs(wav[0]) ** 2),
                 np.mean(np.abs(wav[1]) ** 2)))
