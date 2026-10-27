#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =====================================================================
# source_alphabet.py —— 信源（字母表源）模块
#
# 提供 GNU Radio hier_block2 信源 `source_alphabet`，为发射机提供"承载信息"，
# 分两种模式:
#   discrete   : 读取 source_material/gutenberg_shakespeare.txt 莎士比亚文本，
#                展开为 0/1 比特流(char)，可选与随机序列异或加扰；
#   continuous : 用 ffmpeg 解码 source_material/serial-s01-e01.mp3 音频，
#                转换为 float 音频采样流（替代已下架的 mediatools 库）。
# 供三个 generate_*.py 脚本调用。
# =====================================================================
from gnuradio import gr, blocks
import subprocess  # 调用 ffmpeg 解码 MP3（替代已下架的 mediatools 库）
import numpy as np

class source_alphabet(gr.hier_block2):
    def __init__(self, dtype="discrete", limit=10000, randomize=False):
        # dtype     : "discrete" 比特流信源 / 其他值 音频信源
        # limit     : 最大输出样点数（None 表示不限制）
        # randomize : 是否用随机比特加扰（仅离散模式有效）
        if(dtype == "discrete"):
            # ---- 离散信源: 文本文件 -> 比特流 ----
            gr.hier_block2.__init__(self, "source_alphabet",
                gr.io_signature(0,0,0),
                gr.io_signature(1,1,gr.sizeof_char))  # 输出: 字节符号流

            self.src = blocks.file_source(gr.sizeof_char, "source_material/gutenberg_shakespeare.txt")  # 按字节读入文本
            self.convert = blocks.packed_to_unpacked_bb(1, gr.GR_LSB_FIRST);  # 每字节展开为 8 个 1bit 符号 (LSB 优先)
            #self.convert = blocks.packed_to_unpacked_bb(8, gr.GR_LSB_FIRST);
            self.limit = blocks.head(gr.sizeof_char, limit)  # 截取前 limit 个符号
            self.connect(self.src,self.convert)
            last = self.convert

            # whiten our sequence with a random block scrambler (optionally)
            # 可选加扰器: 用周期性随机比特序列与信源异或，
            # "白化"数据（文本中 0/1 分布不均匀，加扰后接近等概）
            if(randomize):
                rand_len = 256
                rand_bits = np.random.randint(2, size=rand_len)   # 生成 256 位随机序列
                self.randsrc = blocks.vector_source_b(rand_bits, True)  # 循环输出的随机比特源
                self.xor = blocks.xor_bb()                        # 异或即加扰
                self.connect(self.randsrc,(self.xor,1))
                self.connect(last, self.xor)
                last = self.xor

        else:   # "type_continuous"
            # ---- 连续信源: MP3 音频 -> float 采样流 ----
            gr.hier_block2.__init__(self, "source_alphabet",
                gr.io_signature(0,0,0),
                gr.io_signature(1,1,gr.sizeof_float))  # 输出: 音频 float 流

            # 用 ffmpeg 把 MP3 解码为单声道 44.1kHz float32 原始流（替代 mediatools）
            need = limit if limit else 44100 * 60
            dur = need / 44100.0 + 1.0
            raw = subprocess.check_output([
                "ffmpeg", "-v", "quiet", "-i", "source_material/serial-s01-e01.mp3",
                "-ac", "1", "-ar", "44100", "-f", "f32le", "-t", "%.3f" % dur, "-"])
            audio = np.frombuffer(raw, dtype=np.float32)
            self.src = blocks.vector_source_f(audio, False)  # 音频 float 流（不循环）
            self.limit = blocks.head(gr.sizeof_float, limit)
            last = self.src

        # connect head or not, and connect to output
        # limit 为 None 时直接输出，否则先经 head 截取 limit 个样点再输出
        if(limit==None):
            self.connect(last, self)
        else:
            self.connect(last, self.limit, self)


# 模块自测代码 (QA)，被其他脚本 import 时不会执行
if __name__ == "__main__":
    print("QA...")

    # Test discrete source
    # 测试离散信源
    tb = gr.top_block()
    src = source_alphabet("discrete", 1000)
    snk = blocks.vector_sink_b()
    tb.run()

    # Test continuous source
    # 测试连续信源
    tb = gr.top_block()
    src = source_alphabet("continuous", 1000)
    snk = blocks.vector_sink_f()
    tb.run()
