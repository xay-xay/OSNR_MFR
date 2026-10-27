#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =====================================================================
# generate_RML2016.10a.py —— RML2016.10a 数据集生成脚本（主入口）
#
# 流程: 对每个 SNR(-20~18dB, 步长2) × 每种调制方式(14 种):
#   1. source_alphabet 生成比特/音频信源（加扰）
#   2. 对应 transmitter 完成调制
#   3. 通过 GNU Radio 动态信道模型（多径 + 多普勒 + 高斯噪声）
#   4. 从输出中随机抽取 1000 段 128 点向量，并按能量归一化
# 最终得到 dataset { (调制方式, SNR): [1000, 2, 128] float32 }，
# pickle 保存为 RML2016.10a_dict.dat。
# （已移植到 Python 3: print()、pickle。）
# =====================================================================
from transmitters import transmitters        # 14 种调制发射机（见 transmitters.py）
from source_alphabet import source_alphabet  # 信源: 比特流/音频（见 source_alphabet.py）
import analyze_stats                          # 导入但本脚本未实际调用（能量在下方自行计算）
from gnuradio import channels, gr, blocks
import numpy as np
import numpy.fft, pickle, gzip
import random

'''
Generate dataset with dynamic channel model across range of SNRs
'''

apply_channel = True  # 是否叠加信道模型（False 则生成理想无信道调制样本）

dataset = {}

# The output format looks like this
# {('mod type', SNR): np.array(nvecs_per_key, 2, vec_length), etc}
# 数据集格式: { (调制方式, SNR): 样本数组[样本数, 2(I/Q), 128] }

# CIFAR-10 has 6000 samples/class. CIFAR-100 has 600. Somewhere in there seems like right order of magnitude
nvecs_per_key = 1000   # 每个 (调制, SNR) 组合 1000 个样本
vec_length = 128       # 每个样本 128 个采样点
snr_vals = range(-20,20,2)  # SNR -20dB ~ 18dB，步长 2dB，共 20 档
for snr in snr_vals:
    print("snr is ", snr)
    for alphabet_type in transmitters.keys():      # 遍历 "discrete" / "continuous" 两类
        for i,mod_type in enumerate(transmitters[alphabet_type]):  # 遍历该类下每种调制
          dataset[(mod_type.modname, snr)] = np.zeros([nvecs_per_key, 2, vec_length], dtype=np.float32)
          # moar vectors!
          insufficient_modsnr_vectors = True  # 循环控制标志: 样本数不足 1000 就继续生成
          modvec_indx = 0                     # 已生成样本计数
          while insufficient_modsnr_vectors:
              tx_len = int(10e3)              # 每次生成 1 万个信源符号
              if mod_type.modname == "QAM16":
                  tx_len = int(20e3)          # 高阶调制需要更长序列才能充分覆盖星座点
              if mod_type.modname == "QAM32":
                  tx_len = int(25e3)          # 规律: 输出样点数≈40e3，即 5000 符号 × 每符号比特数
              if mod_type.modname == "QAM64":
                  tx_len = int(30e3)
              if mod_type.modname == "QAM128":
                  tx_len = int(40e3)          # 更高阶星座需要更长序列才能充分覆盖星座点
              src = source_alphabet(alphabet_type, tx_len, True)  # 搭建信源（randomize=True 加扰）
              mod = mod_type()                                    # 搭建发射机
              fD = 1                        # 归一化多普勒频移
              delays = [0.0, 0.9, 1.7]      # 多径时延轮廓（单位: 符号周期）
              mags = [1, 0.8, 0.3]          # 各径幅度
              ntaps = 8                     # 信道滤波器抽头数
              noise_amp = 10**(-snr/10.0)   # 由 SNR 换算噪声幅度
              chan = channels.dynamic_channel_model( 200e3, 0.01, 50, .01, 0.5e3, 8, fD, True, 4, delays, mags, ntaps, noise_amp, 0x1337 )
              # 动态信道模型: 200kHz 采样率，含多径衰落 + 多普勒 + AWGN，随机种子 0x1337

              snk = blocks.vector_sink_c()  # 收集输出复信号

              tb = gr.top_block()

              # connect blocks
              # 连接流图: 信源 -> 调制 -> (信道) -> 接收缓存
              if apply_channel:
                  tb.connect(src, mod, chan, snk)
              else:
                  tb.connect(src, mod, snk)
              tb.run()  # 运行流图

              raw_output_vector = np.array(snk.data(), dtype=np.complex64)
              # start the sampler some random time after channel model transients (arbitrary values here)
              # 随机选择起始采样位置，跳过信道模型的瞬态段
              sampler_indx = random.randint(50, 500)
              while sampler_indx + vec_length < len(raw_output_vector) and modvec_indx < nvecs_per_key:
                  sampled_vector = raw_output_vector[sampler_indx:sampler_indx+vec_length]  # 截取 128 点
                  # Normalize the energy in this vector to be 1
                  energy = np.sum((np.abs(sampled_vector)))
                  sampled_vector = sampled_vector / energy   # 能量归一化
                  dataset[(mod_type.modname, snr)][modvec_indx,0,:] = np.real(sampled_vector)  # 存 I 路
                  dataset[(mod_type.modname, snr)][modvec_indx,1,:] = np.imag(sampled_vector)  # 存 Q 路
                  # bound the upper end very high so it's likely we get multiple passes through
                  # independent channels
                  # 随机大步长跳转采样位置，尽量采到不同信道实现下的样本
                  sampler_indx += random.randint(vec_length, round(len(raw_output_vector)*.05))
                  modvec_indx += 1

              if modvec_indx == nvecs_per_key:
                  # we're all done
                  insufficient_modsnr_vectors = False  # 该 (调制, SNR) 样本已够，退出 while

print("all done. writing to disk")
pickle.dump( dataset, open("RML2016.10a_dict.dat", "wb" ) )  # pickle 序列化保存数据集
