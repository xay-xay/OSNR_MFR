#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =====================================================================
# transmitters.py —— 发射机（调制器）模块
#
# 原本定义 14 种调制方式的发射机类（均为 GNU Radio hier_block2），分两大类
# 现只生成 QAM 类调制，其余非 QAM 发射机类均已注释保留（可随时取消注释恢复），
# `transmitters` 字典中也只登记 QAM 类。
# 离散类发射机输入符号流(char)，连续类输入音频流(float)，
# 统一输出复基带已调信号，供三个 generate_*.py 脚本调用。
# =====================================================================
import time, math
from gnuradio import gr, blocks, digital, analog, filter
from gnuradio.filter import firdes
import mapper  # 本地纯 Python 星座映射库(见 mapper.py)，提供 BPSK/QPSK/PSK8/PAM4/QAM4/QAM16/QAM32/QAM64/QAM128 星座对象与 mapper 块

sps = 8       # 每符号采样数 (samples per symbol)
ebw = 0.35    #  excess bandwidth，RRC 成形滤波器的滚降系数

# ---------------------------------------------------------------------
# 离散调制发射机基类
# 信号流: 符号流(char) -> 星座映射 -> RRC 脉冲成形插值 -> 复基带信号
# ---------------------------------------------------------------------
class transmitter_mapper(gr.hier_block2):
    def __init__(self, modtype, symvals, txname, samples_per_symbol=2, excess_bw=0.35):
        gr.hier_block2.__init__(self, txname,
            gr.io_signature(1, 1, gr.sizeof_char),       # 输入: 1 路符号(字节)流
            gr.io_signature(1, 1, gr.sizeof_gr_complex)) # 输出: 1 路复信号流
        self.mod = mapper.mapper(modtype, symvals)       # 星座映射器: 符号值 -> 星座点 (symvals 定义映射表)
        # pulse shaping filter
        # 脉冲成形滤波器: 多相滤波器组任意速率插值器 + 根升余弦(RRC)抽头
        nfilts = 32
        ntaps = nfilts * 11 * int(samples_per_symbol)    # make nfilts filters of ntaps each
        rrc_taps = filter.firdes.root_raised_cosine(
            nfilts,          # gain
            nfilts,          # sampling rate based on 32 filters in resampler
            1.0,             # symbol rate
            excess_bw, # excess bandwidth (roll-off factor)
            ntaps)
        # 插值重采样: 把符号率信号插值成每符号 samples_per_symbol 个采样点
        self.rrc_filter = filter.pfb_arb_resampler_ccf(samples_per_symbol, rrc_taps)
        self.connect(self, self.mod, self.rrc_filter, self)
        #self.rate = const.bits_per_symbol()

# 下面各子类分别实例化一种具体调制，区别仅在星座类型
# 和 符号->星座点 的映射表 symvals（格雷码映射）

# 只生成 QAM 类调制，以下非 QAM 的离散发射机暂时注释掉
# class transmitter_bpsk(transmitter_mapper):
#     modname = "BPSK"
#     def __init__(self):
#         transmitter_mapper.__init__(self, mapper.BPSK,
#             [0,1], "transmitter_bpsk", sps, ebw)
#
# class transmitter_qpsk(transmitter_mapper):
#     modname = "QPSK"
#     def __init__(self):
#         transmitter_mapper.__init__(self, mapper.QPSK,
#             [0,1,3,2], "transmitter_qpsk", sps, ebw)
#
# class transmitter_8psk(transmitter_mapper):
#     modname = "8PSK"
#     def __init__(self):
#         transmitter_mapper.__init__(self, mapper.PSK8,
#             [0,1,3,2,7,6,4,5], "transmitter_8psk", sps, ebw)
#
# class transmitter_pam4(transmitter_mapper):
#     modname = "PAM4"
#     def __init__(self):
#         transmitter_mapper.__init__(self, mapper.PAM4,
#             [0,1,3,2], "transmitter_pam4", sps, ebw)

# QAM4: 4-QAM 的星座几何与 QPSK 完全相同（2x2 方形四点），
# 使用 mapper.py 中的 QAM4 对象（归一化后与 QPSK 点集一致）
class transmitter_qam4(transmitter_mapper):
    modname = "QAM4"
    def __init__(self):
        transmitter_mapper.__init__(self, mapper.QAM4,
            [0,1,2,3], "transmitter_qam4", sps, ebw)

class transmitter_qam8(transmitter_mapper):
    modname = "QAM8"
    def __init__(self):
        # 8-QAM（十字星座，3 bit/符号）；符号值 d=a0a1b0 按幻灯片格雷码标注:
        # 000->(-1,3), 100->(-3,1), 110->(-1,1), 010->(1,1),
        # 001->(-1,-3), 101->(-3,-1), 111->(-1,-1), 011->(1,-1)
        transmitter_mapper.__init__(self, mapper.QAM8,
            [0,7,3,6,1,4,2,5], "transmitter_qam8", sps, ebw)

class transmitter_qam16(transmitter_mapper):
    modname = "QAM16"
    def __init__(self):
        transmitter_mapper.__init__(self, mapper.QAM16,
            [2,6,14,10,3,7,15,11,1,5,13,9,0,4,12,8], "transmitter_qam16", sps, ebw)

class transmitter_qam32(transmitter_mapper):
    modname = "QAM32"
    def __init__(self):
        # 32-QAM（十字星座，5 bit/符号）；几何与幻灯片一致（4+6+6+6+6+4 十字）。
        # symvals 按幻灯片 "MAPPING ON 32 CROSS CONSTELLATION QAM" 的格雷码标注:
        # d=a0a1a2b0b1，矩形 8x4 的 |I|=7 两翼翻折到 Q=±5 上下两行
        transmitter_mapper.__init__(self, mapper.QAM32,
            [3,2,31,30,9,15,27,21,7,13,25,19,8,14,26,20,
             0,1,28,29,4,10,22,16,6,12,24,18,5,11,23,17],
            "transmitter_qam32", sps, ebw)

class transmitter_qam64(transmitter_mapper):
    modname = "QAM64"
    def __init__(self):
        transmitter_mapper.__init__(self, mapper.QAM64,
            [0,32,8,40,3,35,11,43,
             48,16,56,24,51,19,59,27,
            12,44,4,36,15,47,7,39,
            60,28,52,20,63,31,55,23,
            2,34,10,42,1,33,9,41,
            50,18,58,26,49,17,57,25,
            14,46,6,38,13,45,5,37,
            62,30,54,22,61,29,53,21], "transmitter_qam64", sps, ebw)

#class transmitter_qam128(transmitter_mapper):
#    modname = "QAM128"
#    def __init__(self):
#        # 128-QAM（十字星座，7 bit/符号）；符号值按自然顺序 range(128) 映射
#        transmitter_mapper.__init__(self, mapper.QAM128,
#            range(128), "transmitter_qam128", sps, ebw)

# ---------------------------------------------------------------------
# 以下发射机不经过星座映射，直接由对应 GNU Radio 模块完成调制
# ---------------------------------------------------------------------

# 只生成 QAM 类调制，以下非 QAM 的发射机暂时注释掉

# GFSK: 高斯滤波频移键控
# class transmitter_gfsk(gr.hier_block2):
#     modname = "GFSK"
#     def __init__(self):
#         gr.hier_block2.__init__(self, "transmitter_gfsk",
#             gr.io_signature(1, 1, gr.sizeof_char),
#             gr.io_signature(1, 1, gr.sizeof_gr_complex))
#         self.repack = blocks.unpacked_to_packed_bb(1, gr.GR_MSB_FIRST)  # 把每字节1比特打包成连续比特流
#         self.mod = digital.gfsk_mod(sps, sensitivity=0.1, bt=ebw)       # GFSK 调制器 (bt 为带宽时间积)
#         self.connect( self, self.repack, self.mod, self )

# CPFSK: 连续相位频移键控
# class transmitter_cpfsk(gr.hier_block2):
#     modname = "CPFSK"
#     def __init__(self):
#         gr.hier_block2.__init__(self, "transmitter_cpfsk",
#             gr.io_signature(1, 1, gr.sizeof_char),
#             gr.io_signature(1, 1, gr.sizeof_gr_complex))
#         self.mod = analog.cpfsk_bc(0.5, 1.0, sps)  # 调制指数 0.5
#         self.connect( self, self.mod, self )

# WBFM: 宽带调频广播（输入为音频 float 流）
# class transmitter_fm(gr.hier_block2):
#     modname = "WBFM"
#     def __init__(self):
#         gr.hier_block2.__init__(self, "transmitter_fm",
#             gr.io_signature(1, 1, gr.sizeof_float),
#             gr.io_signature(1, 1, gr.sizeof_gr_complex))
#         self.mod = analog.wfm_tx( audio_rate=44100.0, quad_rate=220.5e3 )  # 音频 44.1kHz, 正交采样率 220.5kHz
#         self.connect( self, self.mod, self )
#         self.rate = 200e3/44.1e3  # 速率换算系数（与信道模型 200kHz 采样率对齐用）

# AM-DSB: 标准双边带调幅（输入为音频 float 流）
# class transmitter_am(gr.hier_block2):
#     modname = "AM-DSB"
#     def __init__(self):
#         gr.hier_block2.__init__(self, "transmitter_am",
#         gr.io_signature(1, 1, gr.sizeof_float),
#         gr.io_signature(1, 1, gr.sizeof_gr_complex))
#         self.rate = 44.1e3/200e3
#         #self.rate = 200e3/44.1e3
#         self.interp = filter.mmse_resampler_ff(0.0, self.rate)  # 小数插值，适配采样率
#         self.cnv = blocks.float_to_complex()      # 实信号转复信号
#         self.mul = blocks.multiply_const_cc(1.0)  # 幅度缩放（调制度）
#         self.add = blocks.add_const_cc(1.0)       # 加直流分量，即构造 s(t) = 1 + m(t)
#         self.src = analog.sig_source_c(200e3, analog.GR_SIN_WAVE, 0e3, 1.0)  # 载波（200kHz 正弦）
#         #self.src = analog.sig_source_c(200e3, analog.GR_SIN_WAVE, 50e3, 1.0)
#         self.mod = blocks.multiply_cc()           # 与载波相乘完成调幅
#         self.connect( self, self.interp, self.cnv, self.mul, self.add, self.mod, self )
#         self.connect( self.src, (self.mod,1) )    # 载波接入乘法器第 2 输入口

# AM-SSB: 单边带调幅（希尔伯特变换法提取单边带）
# class transmitter_amssb(gr.hier_block2):
#     modname = "AM-SSB"
#     def __init__(self):
#         gr.hier_block2.__init__(self, "transmitter_amssb",
#         gr.io_signature(1, 1, gr.sizeof_float),
#         gr.io_signature(1, 1, gr.sizeof_gr_complex))
#         self.rate = 44.1e3/200e3
#         #self.rate = 200e3/44.1e3
#         self.interp = filter.mmse_resampler_ff(0.0, self.rate)
# #        self.cnv = blocks.float_to_complex()
#         self.mul = blocks.multiply_const_ff(1.0)
#         self.add = blocks.add_const_ff(1.0)       # 加直流，构造标准调幅信号
#         self.src = analog.sig_source_f(200e3, analog.GR_SIN_WAVE, 0e3, 1.0)
#         #self.src = analog.sig_source_c(200e3, analog.GR_SIN_WAVE, 50e3, 1.0)
#         self.mod = blocks.multiply_ff()           # 与载波相乘
#         #self.filt = filter.fir_filter_ccf(1, firdes.band_pass(1.0, 200e3, 10e3, 60e3, 0.25e3, firdes.WIN_HAMMING, 6.76))
#         self.filt = filter.hilbert_fc(401)        # 401 抽头希尔伯特变换滤波器，提取单边带
#         self.connect( self, self.interp, self.mul, self.add, self.mod, self.filt, self )
#         self.connect( self.src, (self.mod,1) )


# 对外导出的发射机登记表: generate_*.py 遍历该字典逐种调制搭建流图
#   "discrete"   -> 12 种离散调制（输入符号流）
#   "continuous" -> 3 种连续调制（输入音频流）
transmitters = {
    "discrete":[transmitter_bpsk, transmitter_qpsk, transmitter_8psk, transmitter_pam4, transmitter_qam4, transmitter_qam8, transmitter_qam16, transmitter_qam32, transmitter_qam64, transmitter_qam128, transmitter_gfsk, transmitter_cpfsk],
    "continuous":[transmitter_fm, transmitter_am, transmitter_amssb]
    }
