# -*- coding: utf-8 -*-
# =====================================================================
# constellation.py —— QAM 星座定义与格雷比特映射（纯 numpy，无 GR 依赖）
#
# 星座几何与 symvals 格雷映射表移植自 dataset-master/mapper.py 和
# transmitters.py（已由 check_gray.py 验证全部满足格雷映射），仅两处改动:
#   1. 归一化由"平均幅度=1"改为"平均符号功率 E|s|^2 = 1"
#      （便于 ASE 噪声加载按 OSNR 精确定标）;
#   2. 去掉 GNU Radio 依赖，bit 打包改为纯 numpy 实现。
#
# 提供调制: QAM8 / QAM16 / QAM32 / QAM64 与 BPSK / QPSK / 8PSK
#   * 8/16/32/64QAM: 方形或十字星座
#   * BPSK/QPSK/8PSK: 单位圆等间隔 PSK
#   * QAM4 仅为兼容旧数据集保留定义(与 QPSK 几何完全相同, 不再作为独立类别)
# =====================================================================
import numpy as np


def _square_grid(side):
    """side×side 方形 QAM（坐标为奇整数，行优先、顶部行在前），
    与原 mapper.py 的 _square_grid 排列一致。"""
    coords = list(range(-(side - 1), side, 2))     # side=4 -> [-3,-1,1,3]
    pts = [complex(i, q) for q in reversed(coords) for i in coords]
    return np.array(pts)


def _cross_qam(side, corner_cut):
    """十字 QAM: side×side 方格去掉 |I|>=corner_cut 且 |Q|>=corner_cut 的四角。"""
    coords = list(range(-(side - 1), side, 2))
    pts = [complex(i, q) for q in reversed(coords) for i in coords
           if not (abs(i) >= corner_cut and abs(q) >= corner_cut)]
    return np.array(pts)


def _psk(m, phase0=0.0):
    """M-PSK: 单位圆上等间隔的 M 个点（已满足 E|s|^2 = 1）。"""
    return np.exp(1j * (2.0 * np.pi * np.arange(m) / m + phase0))


# 原始几何（未归一化）
_RAW = {
    "QAM4":  _square_grid(2),
    "BPSK":  _psk(2, 0.0),
    "QPSK":  _psk(4, np.pi / 4.0),
    "8PSK":  _psk(8, np.pi / 8.0),
    # 8QAM 十字星座: 矩形 4x2 右翼翻折（对照 Broadcom 十字星座映射）
    "QAM8":  np.array([-1 + 3j,
                       -3 + 1j, -1 + 1j, 1 + 1j,
                       -3 - 1j, -1 - 1j, 1 - 1j,
                       -1 - 3j]),
    "QAM16": _square_grid(4),
    "QAM32": _cross_qam(6, 5),     # 6x6 方格去 4 个单点角 -> 32 点
    "QAM64": _square_grid(8),
}

# 格雷映射表: maptable[符号值] = points[symvals[符号值]]
# （移植自 dataset-master/transmitters.py，check_gray.py 验证过全格雷）
_SYMVALS = {
    "QAM4":  [0, 1, 2, 3],
    "BPSK":  [0, 1],
    "QPSK":  [0, 1, 3, 2],
    "8PSK":  [0, 1, 3, 2, 6, 7, 5, 4],
    "QAM8":  [0, 7, 3, 6, 1, 4, 2, 5],
    "QAM16": [2, 6, 14, 10, 3, 7, 15, 11, 1, 5, 13, 9, 0, 4, 12, 8],
    "QAM32": [3, 2, 31, 30, 9, 15, 27, 21, 7, 13, 25, 19, 8, 14, 26, 20,
              0, 1, 28, 29, 4, 10, 22, 16, 6, 12, 24, 18, 5, 11, 23, 17],
    "QAM64": [0, 32, 8, 40, 3, 35, 11, 43,
              48, 16, 56, 24, 51, 19, 59, 27,
              12, 44, 4, 36, 15, 47, 7, 39,
              60, 28, 52, 20, 63, 31, 55, 23,
              2, 34, 10, 42, 1, 33, 9, 41,
              50, 18, 58, 26, 49, 17, 57, 25,
              14, 46, 6, 38, 13, 45, 5, 37,
              62, 30, 54, 22, 61, 29, 53, 21],
}

_CACHE = {}


def get_constellation(name):
    """返回 (points, bps, maptable)。
    points: 归一化到平均符号功率 E|s|^2 = 1 的星座点;
    bps:    每符号比特数;
    maptable[符号值] -> 复星座点（格雷映射）。"""
    if name in _CACHE:
        return _CACHE[name]
    if name not in _RAW:
        raise ValueError("未知调制: %s (支持 %s)" % (name, list(_RAW)))
    pts = _RAW[name].astype(np.complex128)
    pts = pts / np.sqrt(np.mean(np.abs(pts) ** 2))   # 平均功率归一
    table = pts[np.asarray(_SYMVALS[name], dtype=np.intp)]
    bps = int(round(np.log2(len(pts))))
    _CACHE[name] = (pts, bps, table)
    return pts, bps, table


def random_symbols(name, rng, n_symbols):
    """用随机比特流生成 n_symbols 个单位功率 QAM 符号（格雷映射）。"""
    _, bps, table = get_constellation(name)
    bits = rng.integers(0, 2, size=n_symbols * bps, dtype=np.uint8)
    # 每 bps 个比特 MSB-first 打包成符号值
    vals = bits.reshape(n_symbols, bps).dot(
        1 << np.arange(bps - 1, -1, -1, dtype=np.int64))
    return table[vals]


def cma_modulus(name):
    """CMA 的常数模 R = E|s|^4 / E|s|^2（单位功率星座即 E|s|^4）。"""
    pts, _, _ = get_constellation(name)
    return float(np.mean(np.abs(pts) ** 4) / np.mean(np.abs(pts) ** 2))


if __name__ == "__main__":
    # 自检: 点数 / 每符号比特 / 平均功率 / CMA 模常数
    for name in ("QAM4", "QAM8", "QAM16", "QAM32", "QAM64",
                 "BPSK", "QPSK", "8PSK"):
        pts, bps, table = get_constellation(name)
        print("%-6s M=%3d bps=%d E|s|^2=%.4f R_cma=%.4f"
              % (name, len(pts), bps, np.mean(np.abs(pts) ** 2),
                 cma_modulus(name)))
