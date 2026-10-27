#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =====================================================================
# generate_alphabet_dataset.py —— 原始字母表（信源）数据集生成脚本
#
# 不做任何调制: 直接从 source_alphabet 读取 100 万个信源符号（比特流或音频），
# 经 timeseries_slicer.slice_timeseries_real_dict 切成 128 点实数样本，
# 保存为 alphabet_dict.dat，可作为"未调制"对照数据集。
# （已移植到 Python 3: print()、pickle）；虽然 import 了 transmitters，
# 但只用它的 keys（"discrete"/"continuous"）来区分两种信源类型。
# =====================================================================
from transmitters import transmitters        # 仅用其 keys 区分 discrete/continuous 两类信源
from source_alphabet import source_alphabet  # 信源: 比特流/音频（见 source_alphabet.py）
import timeseries_slicer                     # 切片工具（见 timeseries_slicer.py）
from gnuradio import channels, gr, blocks
#import matplotlib.pyplot as plt
import numpy as np
import numpy.fft, pickle, gzip


apply_channel = True  # 已声明但本脚本实际未使用（没有接入信道模型）

output = {}
#snr_vals = [-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20]
# 两种信源类型对应的数据类型与接收器 (sink) 配置
blks={"continuous":{ "size":gr.sizeof_float,
                     "sink":blocks.vector_sink_f },
      "discrete"  :{ "size":gr.sizeof_char,
                     "sink":blocks.vector_sink_b }}
for alphabet_type in transmitters.keys():   # 遍历 "discrete" 与 "continuous"


    print("running test", alphabet_type)
    tx_len = int(1000e3)                    # 一次读取 100 万个符号
    src = source_alphabet(alphabet_type, tx_len, True)  # 信源（加扰）
    head = blocks.head(blks[alphabet_type]["size"], tx_len)  # 截取 tx_len 个符号
    snk = blks[alphabet_type]["sink"]()     # 对应类型的接收缓存
    tb = gr.top_block()
    tb.connect(src,head,snk)                # 流图: 信源 -> 截取 -> 缓存（无调制、无信道）
    tb.run()
    print("finished: ", len(snk.data()))
    output[alphabet_type] = np.array(snk.data(), dtype=np.float32)

print(output)
X = timeseries_slicer.slice_timeseries_real_dict(output, 128, 64, 1000)  # 切成 [k, 1, 128] 实数样本
pickle.dump( X, open("alphabet_dict.dat", "wb" ) )  # 保存数据集
