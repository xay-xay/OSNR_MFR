#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =====================================================================
# generate_RML2016.04c.py —— RML2016.04c 数据集生成脚本
#
# 与 10a 的区别: 本脚本先为每个 (调制, SNR) 组合跑完整流图得到一条长已调序列，
# 记录最短序列长度并把所有调制裁剪对齐到同一长度（去掉首尾瞬态），
# 再交给 timeseries_slicer 统一切成 128 点窗口（步长 64）样本，保存两种格式:
#   RML2014.04c_dict.dat : 字典形式 { (调制, SNR): [1000, 2, 128] }
#   RML2016.04c.dat      : 所有样本堆叠成一个大数组
# （已移植到 Python 3: print()、pickle。）
# =====================================================================
from transmitters import transmitters        # 11 种调制发射机（见 transmitters.py）
from source_alphabet import source_alphabet  # 信源: 比特流/音频（见 source_alphabet.py）
import timeseries_slicer                     # 切片工具: 长序列 -> 128 点样本（见 timeseries_slicer.py）
import analyze_stats                         # 导入但本脚本未实际调用
from gnuradio import channels, gr, blocks
import numpy as np
import numpy.fft, pickle, gzip

'''
Generate dataset with dynamic channel model across range of SNRs
'''

apply_channel = True   # 是否叠加信道模型
output = {}            # 暂存 { (调制, SNR): 长已调序列 }
min_length = 9e9       # 记录最短输出序列长度（用于对齐所有调制的样本长度）
snr_vals = range(-20,20,2)  # SNR: -20 ~ 18dB，步长 2dB
for snr in snr_vals:
    for alphabet_type in transmitters.keys():   # "discrete" / "continuous"
        print(alphabet_type)
        for i,mod_type in enumerate(transmitters[alphabet_type]):
            print("running test", i,mod_type)

            tx_len = int(10e3)             # 信源符号数
            if mod_type.modname == "QAM64":
                tx_len = int(30e3)         # 高阶调制需要更长序列
            if mod_type.modname == "QAM16":
                tx_len = int(20e3)
            src = source_alphabet(alphabet_type, tx_len, True)  # 信源（加扰）
            mod = mod_type()                                    # 发射机
            fD = 1                       # 归一化多普勒
            delays = [0.0, 0.9, 1.7]     # 多径时延
            mags = [1, 0.8, 0.3]         # 多径幅度
            ntaps = 8                    # 信道滤波器抽头数
            noise_amp = 10**(-snr/10.0)  # 由 SNR 换算噪声幅度
            print(noise_amp)
            #noise_amp = 0.1
            chan = channels.dynamic_channel_model( 200e3, 0.01, 1e2, 0.01, 1e3, 8, fD, True, 4, delays, mags, ntaps, noise_amp, 0x1337 )
            # 动态信道模型: 200kHz 采样率，含多径 + 多普勒 + 噪声

            snk = blocks.vector_sink_c()

            tb = gr.top_block()

            # connect blocks
            # 流图: 信源 -> 调制 -> (信道) -> 接收缓存
            if apply_channel:
                tb.connect(src, mod, chan, snk)
            else:
                tb.connect(src, mod, snk)
            tb.run()

            modulated_vector = np.array(snk.data(), dtype=np.complex64)
            if len(snk.data()) < min_length:   # 更新最短序列长度及其对应调制
                min_length = len(snk.data())
                min_length_mod = mod_type
            output[(mod_type.modname, snr)] = modulated_vector

print("min length mod is %s with %i samples" % (min_length_mod, min_length))
# trim the beginning and ends, and make all mods have equal number of samples
# 裁剪首尾（去掉信道瞬态），并把所有调制对齐到相同长度
start_indx = 100
fin_indx = min_length-100
for mod, snr in output:
 output[(mod,snr)] = output[(mod,snr)][start_indx:fin_indx]  # 循环体缩进 1 空格为原代码风格
X = timeseries_slicer.slice_timeseries_dict(output, 128, 64, 1000)  # 切片: 窗长128、步长64、每组最多1000个
pickle.dump( X, open("RML2014.04c_dict.dat", "wb" ) )  # 保存字典形式数据集
X = np.vstack(list(X.values()))
pickle.dump( X, open("RML2016.04c.dat", "wb" ) )       # 保存堆叠后的大数组数据集
