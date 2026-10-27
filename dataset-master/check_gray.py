#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# check_gray.py —— 格雷码自检（临时脚本，验证完可删）
# 对每个基于 mapper 的发射机，实例化后直接读取 mapper 最终映射表
# maptable[符号值] = 星座点，枚举所有"空间最近邻"点对，
# 检查相邻两点对应的符号值是否恰好只差 1 个比特。
# 输出为 ASCII，避免 WSL 下中文编码问题。
import numpy as np
import transmitters as T


def check(name, maptable, bps):
    pts = np.array(maptable, dtype=np.complex128)
    M = len(pts)
    # 1) 求最小距离 dmin
    dmin = None
    for i in range(M):
        for j in range(i + 1, M):
            d = abs(pts[i] - pts[j])
            if dmin is None or d < dmin:
                dmin = d
    # 2) 枚举相邻点对（距离 <= dmin*1.05），检查比特差是否为 1
    nadj = 0
    viol = []
    for i in range(M):
        for j in range(i + 1, M):
            if abs(pts[i] - pts[j]) <= dmin * 1.05:
                nadj += 1
                db = bin(i ^ j).count("1")
                if db != 1:
                    viol.append((i, j, pts[i], pts[j], db))
    if viol:
        print("%-8s NOT GRAY: %d/%d neighbor pairs differ by >1 bit"
              % (name, len(viol), nadj))
        for i, j, pi, pj, db in viol[:8]:
            print("    %s @(%+.3f,%+.3f) <-> %s @(%+.3f,%+.3f)  diff %d bits"
                  % (format(i, "0%db" % bps), pi.real, pi.imag,
                     format(j, "0%db" % bps), pj.real, pj.imag, db))
        if len(viol) > 8:
            print("    ... and %d more" % (len(viol) - 8))
    else:
        print("%-8s GRAY OK  (%d neighbor pairs, all differ by 1 bit)"
              % (name, nadj))


mods = [T.transmitter_bpsk, T.transmitter_qpsk, T.transmitter_8psk,
        T.transmitter_pam4, T.transmitter_qam4, T.transmitter_qam8,
        T.transmitter_qam16, T.transmitter_qam32, T.transmitter_qam64,
        T.transmitter_qam128]
for cls in mods:
    tx = cls()
    check(cls.modname, tx.mod._maptable, tx.mod._bps)
