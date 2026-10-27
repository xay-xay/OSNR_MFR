# -*- coding: utf-8 -*-
# =====================================================================
# analyze_stats.py —— 数据集统计分析工具
#
# 提供向量能量计算与按(调制方式, SNR)分组的统计功能（平均能量/均值偏差/标准差），
# 并可加载 pickle 保存的数据集文件(.dat)进行分析。
# 被 timeseries_slicer.py 调用其中的 calc_vec_energy；
# 也可作为独立脚本运行（main 默认读取 X_4_dict.dat 并打印偏差统计）。
# （已移植到 Python 3: pickle、print()。）
# =====================================================================

import numpy as np
import pickle                # Python 3 序列化库（对应 Python 2 的 cPickle）

def calc_vec_energy(vec):
    """计算一个 I/Q 样本的总能量: 对每个采样点取 sqrt(I^2 + Q^2) 后求和。"""
    isquared = np.power(vec[0],2.0)     # I 通道平方
    qsquared = np.power(vec[1], 2.0)    # Q 通道平方
    inst_energy = np.sqrt(isquared+qsquared)  # 瞬时幅度（包络）
    return sum(inst_energy)

def calc_mod_energies(ds):
    """遍历数据集 {(调制, SNR): 样本数组}，打印每种调制在每个 SNR 下的平均能量。"""
    for modulation, snr in ds:
        avg_energy = 0
        nvectors = ds[(modulation,snr)].shape[0]
        for vec in ds[(modulation, snr)]:
            avg_energy += calc_vec_energy(vec)
        avg_energy /= nvectors
        print("%s at %i has %i vectors avg energy of %2.1f" % (modulation, snr, nvectors, avg_energy))

def calc_mod_bias(ds):
    """计算并打印每种调制在每个 SNR 下 I/Q 两通道的均值偏差（直流分量）。"""
    for modulation, snr in ds:
        avg_bias_re = 0
        avg_bias_im = 0
        nvectors = ds[(modulation,snr)].shape[0]
        for vec in ds[(modulation, snr)]:
            avg_bias_re += (np.mean(vec[0]))   # I 通道均值累加
            avg_bias_im += (np.mean(vec[1]))   # Q 通道均值累加
        #avg_bias_re /= nvectors
        #avg_bias_im /= nvectors
        print("%s at %i has %i vectors avg bias of %2.1f + %2.1f j" % (modulation, snr, nvectors, avg_bias_re, avg_bias_im))

def calc_mod_stddev(ds):
    """计算并打印每种调制在每个 SNR 下复包络的标准差。"""
    for modulation, snr in ds:
        avg_stddev = 0
        nvectors = ds[(modulation,snr)].shape[0]
        for vec in ds[(modulation, snr)]:
            avg_stddev += np.abs(np.std(vec[0]+1j*vec[1]))  # 重组复信号后取标准差
        #avg_stddev /= nvectors
        print("%s at %i has %i vectors avg stddev of %2.1f" % (modulation, snr, nvectors, avg_stddev))

def open_ds(location="X_4_dict.dat"):
    """加载 pickle 数据集文件（默认 X_4_dict.dat），返回字典。"""
    f = open(location, "rb")
    ds = pickle.load(f)
    return ds

def main():
    # 独立运行入口: 加载数据集并输出统计（绘图与其他统计项被注释掉）
    ds = open_ds()
    #plt.plot(ds[('BPSK', 12)][25][0][:])
    #plt.plot(ds[('BPSK', 12)][25][1][:])
    #plt.show()
    #calc_mod_energies(ds)
    #calc_mod_stddev(ds)
    calc_mod_bias(ds)

if __name__ == "__main__":
    main()
