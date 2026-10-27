#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =====================================================================
# mapper.py —— 纯 Python 星座映射库（替代已下架的 gr-mapper OOT 模块）
#
# 本文件是 gr-vt/gr-mapper 的纯 Python 等价实现，供 transmitters.py 里
# `import mapper` 使用。星座几何与归一化方式逐条对照原库 C++ 源码
# (lib/constellation.cc) 还原，保证与原 RML2016 生成链路一致：
#   * 归一化: scale = N / Σ|point|，即归一到"平均幅度 = 1"
#   * QPSK 在原库基础上额外旋转 π/4
#   * 比特打包 MSB-first: idx |= bit << (bps-j-1)
#   * 映射语义: output = points[ symvals[packed_symbol_value] ]
#
# 提供的星座对象: BPSK / QPSK / PSK8 / PAM4 / QAM16 / QAM64
#   （以上 6 个与原库完全一致）
# 以及 QAM4（几何与 QPSK 等价）和 QAM8 / QAM32 / QAM128（原库 C++ switch 未实现、
#   会抛 "bad mod type"，这里用"十字星座"补齐，属于本项目的新增扩展；
#   QAM8 / QAM32 的几何与位映射对照 Broadcom 十字星座映射幻灯片）。
# =====================================================================
import math
import numpy as np
from gnuradio import gr

_SQRT1_2 = math.sqrt(0.5)


def _normalize(points):
    """归一到平均幅度 = 1（对照原库: scale = N / Σ|point|）。"""
    pts = np.array(points, dtype=np.complex128)
    if pts.ndim != 1:
        # 传入 (a,b) 元组列表会被 numpy 解析成二维数组而非复数，这里显式拦截
        raise ValueError("星座点必须是一维复数序列（检查是否误用了元组）")
    avg_mag = np.abs(pts).sum()
    return pts * (len(pts) / avg_mag)


def _square_grid(side):
    """生成 side×side 方形 QAM 星座（坐标为奇整数，行优先、顶部行在前），
    与原库 sym_qam16 / sym_qam64 的排列一致。"""
    coords = list(range(-(side - 1), side, 2))   # 如 side=4 -> [-3,-1,1,3]
    pts = []
    for q in reversed(coords):                    # 虚部从大到小（顶部行在前）
        for i in coords:                          # 实部从小到大（左到右）
            pts.append(complex(i, q))
    return pts


def _cross_qam(side, corner_cut):
    """生成"十字"QAM 星座: side×side 方格去掉四个角上 |re|>=corner_cut
    且 |im|>=corner_cut 的区域。用于原库没有的 QAM32 / QAM128。"""
    coords = list(range(-(side - 1), side, 2))
    pts = []
    for q in reversed(coords):
        for i in coords:
            if abs(i) >= corner_cut and abs(q) >= corner_cut:
                continue                          # 跳过角落，形成十字
            pts.append(complex(i, q))
    return pts


class constellation(object):
    """星座对象: 保存归一化后的星座点与每符号比特数。"""
    def __init__(self, points):
        self._pts = _normalize(points)
        self._bps = int(round(math.log2(len(points))))

    @property
    def points(self):
        return self._pts

    def bits_per_symbol(self):
        return self._bps


# ---------------------------------------------------------------------
# 各星座定义（未归一化的原始几何，随后统一归一化）
# ---------------------------------------------------------------------
# BPSK: (1,0), (-1,0)
BPSK = constellation([1 + 0j, -1 + 0j])

# QPSK: (1,0),(0,1),(-1,0),(0,-1)，再整体旋转 π/4（与原库一致）
_rot = complex(_SQRT1_2, _SQRT1_2)
QPSK = constellation([p * _rot for p in [1 + 0j, 1j, -1 + 0j, -1j]])

# 8PSK: 单位圆上间隔 π/4 的 8 个点
PSK8 = constellation([complex(math.cos(k * math.pi / 4),
                              math.sin(k * math.pi / 4)) for k in range(8)])

# PAM4: 实轴上 -3,-1,3,1（顺序与原库一致）
PAM4 = constellation([-3 + 0j, -1 + 0j, 3 + 0j, 1 + 0j])

# QAM4: 2x2 方形四点，归一化后与 QPSK 几何完全相同（45°/135°/225°/315° 单位圆四点）。
QAM4 = constellation(_square_grid(2))

# QAM8: 十字星座 (3+3+1+1)，对照 Broadcom 幻灯片 "MAPPING ON CROSS CONSTELLATION
# CASE I: 8-QAM" 公式版：矩形 4x2 的右翼 (I=3) 经 I_cf=-sign(I)(4-|I|)、
# Q_cf=sign(Q)(|Q|+2) 翻折到 (-1,3) 与 (-1,-3)（两点 I_rect 均为 +3，故 I_cf 均为 -1）。
# 点序: 行优先、顶行在前，是 transmitters.py 中 symvals 的"下标"所指的星座点。
QAM8 = constellation([-1 + 3j,
                      -3 + 1j, -1 + 1j, 1 + 1j,
                      -3 - 1j, -1 - 1j, 1 - 1j,
                      -1 - 3j])

# QAM16 / QAM64: 方形星座
QAM16 = constellation(_square_grid(4))
QAM64 = constellation(_square_grid(8))

# QAM32 / QAM128: 十字星座（原库未实现，这里补齐）
#   32  = 6×6 方格去掉 4 个单点角；128 = 12×12 方格去掉 4 个 2×2 角
QAM32 = constellation(_cross_qam(6, 5))
QAM128 = constellation(_cross_qam(12, 9))


# ---------------------------------------------------------------------
# mapper 块: 比特流(char) -> 星座点(complex)
# 每 bits_per_symbol 个输入比特打包成一个符号(MSB-first)，
# 再按 output = points[symvals[symbol]] 输出对应星座点，抽取比 = bps。
# ---------------------------------------------------------------------
class mapper(gr.decim_block):
    def __init__(self, modtype, symvals):
        self._bps = modtype.bits_per_symbol()
        pts = modtype.points
        symvals = list(symvals)
        M = len(pts)
        if len(symvals) != M:
            raise ValueError("symvals 长度(%d)必须等于星座点数(%d)" % (len(symvals), M))
        # 预生成映射表: maptable[符号值] = points[symvals[符号值]]
        self._maptable = np.array([pts[symvals[j]] for j in range(M)],
                                  dtype=np.complex64)
        # MSB-first 打包权重: [2^(bps-1), ..., 2, 1]
        self._weights = 2 ** np.arange(self._bps - 1, -1, -1)
        gr.decim_block.__init__(self,
            name="mapper",
            in_sig=[np.uint8],
            out_sig=[np.complex64],
            decim=self._bps)

    def work(self, input_items, output_items):
        n = len(output_items[0])
        bits = np.array(input_items[0][:n * self._bps], dtype=np.int32)
        bits = bits.reshape(n, self._bps)
        idx = bits.dot(self._weights).astype(np.intp)   # 每 bps 比特 -> 符号值
        output_items[0][:] = self._maptable[idx]
        return n
