# -*- coding: utf-8 -*-
# =====================================================================
# pcs_constellation.py —— 概率整形(PCS)星座库(独立模块, 不改动原有文件)
#
# Maxwell-Boltzmann 整形: 星座点 x_k 的出现概率
#     P(x_k) = exp(-ν·|x_k|²) / Σ_j exp(-ν·|x_j|²)
#   ν=0 -> 均匀(退化为普通格式); ν 越大概率越集中于内圈低能量点,
#   每符号信息熵随之降低(低于几何 log2 M)。
#
# 支持格式:
#   均匀: QAM4 / QAM16 / QAM64 / QAM256            (nu=0, 等概率)
#   整形: QAM4_PCS / QAM16_PCS / QAM64_PCS / QAM256_PCS (nu>0)
#   * QAM4/16/64 几何直接复用 constellation.py 的定义(与主数据集严格
#     同一几何, 只 import 不修改); QAM256 为本模块新增(原库未定义)。
#   * ⚠ QAM4 四个点能量完全相同(|x|² 全相等), MB 整形对它退化为
#     均匀分布 —— 保留仅为格式完整性, 物理上没有任何整形效果。
#
# 说明: 本库不做比特级的分布匹配(DM)。特征识别只取决于符号的统计
# 分布, 按 PMF 直接采样与真实 DM 输出同分布, 对 45 维特征完全等价;
# 比特级 Gray 标签只影响 FEC 度量, 不影响本数据集。
# =====================================================================
import numpy as np

import config as C                          # 只读: AH_RANGE_MAX 等口径
from constellation import get_constellation  # 只读: 复用 4/16/64 几何

# ----------------------------- 格式目录 ------------------------------
UNIFORM_MODS = ["QAM4", "QAM16", "QAM64", "QAM256"]
PCS_MODS = [m + "_PCS" for m in UNIFORM_MODS]
ALL_MODS = UNIFORM_MODS + PCS_MODS

# 各整形格式的默认 ν。
#   ν 的选取受 config.AH_RANGE_MAX=2.5 约束: 整形使星座整体放大,
#   ν 过大时外圈点幅度会超出幅度直方图的统计上界。
#   QAM256 点数多、外圈半径大, 只能用较小的 ν(1.5 时 max|x|≈2.3)。
DEFAULT_NU = {
    "QAM4_PCS":   2.0,   # (退化: 四点能量相同, 任意 ν 都是均匀)
    "QAM16_PCS":  2.0,   # H ≈ 3.30 bit (几何 4 bit)
    "QAM64_PCS":  2.0,   # H ≈ 5.32 bit (几何 6 bit)
    "QAM256_PCS": 1.5,   # H ≈ 7.4 bit  (几何 8 bit)
}

_CACHE = {}


def _square_grid(side):
    """side×side 方形 QAM(坐标为奇整数, 行优先、顶部行在前) —— 与
    constellation.py 的排列完全一致, 用于 256QAM。"""
    coords = list(range(-(side - 1), side, 2))   # side=16 -> [-15..15 奇数]
    return np.array([complex(i, q) for q in reversed(coords) for i in coords])


def base_points(name):
    """返回"均匀口径"归一化(E|x|²=1)的基础几何。
    QAM4/16/64 复用原库; QAM256 本地生成。"""
    if name == "QAM256":
        pts = _square_grid(16).astype(np.complex128)
        return pts / np.sqrt(np.mean(np.abs(pts) ** 2))
    pts, _, _ = get_constellation(name)         # 原库已做功率归一
    return pts


def mb_pmf(pts, nu):
    """Maxwell-Boltzmann 概率分布(定义在均匀口径几何上, Σp=1)。"""
    w = np.exp(-nu * np.abs(pts) ** 2)
    return w / w.sum()


def resolve(name, nu=None):
    """解析格式 -> 完整参数字典(带缓存)。
    dict(base, nu, points, pmf, R_cma, entropy, max_amp):
      points  : 最终星座几何。整形格式按 MB 加权平均功率重归一到 1
                (否则概率内移后平均功率<1, OSNR 噪声定标失真);
      pmf     : 先验分布(均匀格式为 None);
      R_cma   : CMA 常数模, 整形格式按 PMF 加权 E|s|⁴/E|s|²;
      entropy : 每符号信息熵(bit), 整形格式的"实际速率";
      max_amp : 星座最大幅度(对照 AH_RANGE_MAX 检查是否超界)。"""
    key = (name, None if nu is None else float(nu))
    if key in _CACHE:
        return _CACHE[key]
    if name.endswith("_PCS"):
        base = name[:-4]
        if base not in UNIFORM_MODS:
            raise ValueError("未知整形基础格式: %s" % name)
        nu_eff = DEFAULT_NU[name] if nu is None else float(nu)
    else:
        if name not in UNIFORM_MODS:
            raise ValueError("未知格式: %s (支持 %s)" % (name, ALL_MODS))
        base, nu_eff = name, 0.0

    pts0 = base_points(base)                    # 均匀口径几何
    if nu_eff <= 0.0:                           # ---- 均匀路径 ----
        pts = pts0
        pmf = None
        R = float(np.mean(np.abs(pts) ** 4) / np.mean(np.abs(pts) ** 2))
        H = float(np.log2(len(pts)))
    else:                                       # ---- 整形路径 ----
        pmf = mb_pmf(pts0, nu_eff)
        pts = pts0 / np.sqrt(np.sum(pmf * np.abs(pts0) ** 2))  # 重归一
        R = float(np.sum(pmf * np.abs(pts) ** 4) / np.sum(pmf * np.abs(pts) ** 2))
        H = float(-np.sum(pmf * np.log2(pmf)))
    d = dict(base=base, nu=nu_eff, points=pts, pmf=pmf,
             R_cma=R, entropy=H, max_amp=float(np.abs(pts).max()))
    _CACHE[key] = d
    return d


def random_symbols(name, rng, n_symbols, nu=None):
    """按格式先验分布采样 n_symbols 个符号(最终几何, 已功率归一)。
    均匀: 点索引等概率随机; 整形: 点索引按 MB 分布抽取。"""
    fmt = resolve(name, nu)
    M = len(fmt["points"])
    if fmt["pmf"] is None:
        idx = rng.integers(0, M, size=n_symbols)
    else:
        idx = rng.choice(M, size=n_symbols, p=fmt["pmf"])
    return fmt["points"][idx]


def report(mods, nus=None):
    """打印诊断表: ν / 熵 / CMA 模 / 最大幅度 / 超 AH 上界概率。
    生成数据集前先看这张表 —— "超AH上界%" 应为 0(或极小)。"""
    nus = nus or {}
    print("%-12s %-6s %8s %8s %8s %10s  %s"
          % ("格式", "ν", "熵(bit)", "R_cma", "max|x|", "超AH上界%", "备注"))
    for m in mods:
        fmt = resolve(m, nus.get(m))
        p = (fmt["pmf"] if fmt["pmf"] is not None
             else np.full(len(fmt["points"]), 1.0 / len(fmt["points"])))
        over = 100.0 * float(np.sum(p[np.abs(fmt["points"]) > C.AH_RANGE_MAX]))
        note = ""
        if m == "QAM4_PCS":
            note = "⚠ 退化: 四点等能量, 整形无效果"
        elif over > 0.5:
            note = "⚠ 外圈超出 AH 上界, 建议减小 ν"
        print("%-12s %-6.2f %8.3f %8.4f %8.3f %10.3f  %s"
              % (m, fmt["nu"], fmt["entropy"], fmt["R_cma"],
                 fmt["max_amp"], over, note))


if __name__ == "__main__":
    # 自检: 打印全部支持格式的参数表
    report(ALL_MODS)
