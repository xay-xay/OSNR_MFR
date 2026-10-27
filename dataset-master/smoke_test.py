#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =====================================================================
# smoke_test.py —— 冒烟测试（临时文件，验证通过后可删除）
#
# 目的: 在正式跑数据集生成前，快速验证 Python3 + GNU Radio 3.10 移植是否可用。
# 做法: 遍历 transmitters 字典里每一种调制，搭建
#       信源 -> 调制 -> 动态信道模型 -> 接收缓存  的最小流图并运行，
#       只要每种调制都能跑通、输出非空，就说明各 GR 模块 API 兼容。
# =====================================================================
from transmitters import transmitters
from source_alphabet import source_alphabet
from gnuradio import channels, gr, blocks

ok = True
for alphabet_type in transmitters.keys():          # "discrete" / "continuous"
    for mod_type in transmitters[alphabet_type]:
        try:
            src = source_alphabet(alphabet_type, 5000, True)
            mod = mod_type()
            chan = channels.dynamic_channel_model(
                200e3, 0.01, 50, .01, 0.5e3, 8, 1, True, 4,
                [0.0, 0.9, 1.7], [1, 0.8, 0.3], 8, 0.5, 0x1337)
            snk = blocks.vector_sink_c()
            tb = gr.top_block()
            tb.connect(src, mod, chan, snk)
            tb.run()
            n = len(snk.data())
            flag = "OK" if n > 0 else "EMPTY!"
            if n == 0:
                ok = False
            print("[%s] %-10s %-10s -> %d samples" % (flag, alphabet_type, mod_type.modname, n))
        except Exception as e:
            ok = False
            print("[FAIL] %-10s %-10s -> %s" % (alphabet_type, mod_type.modname, e))

print("=" * 50)
print("SMOKE TEST", "PASSED" if ok else "FAILED")
