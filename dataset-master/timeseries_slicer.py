# -*- coding: utf-8 -*-
# =====================================================================
# timeseries_slicer.py —— 时间序列切片工具
#
# 把长的一维信号用滑动窗口切成定长的数据集样本（向量）：
#   slice_timeseries      复信号 -> [k, 2, l] 的 I/Q 双通道样本（按能量归一化）
#   slice_timeseries_real 实信号 -> [k, 1, l] 的单通道样本
#   *_dict                对字典 {键: 信号} 批量切片的封装
# 被 generate_RML2016.04c.py（复切片）和 generate_alphabet_dataset.py（实切片）调用。
# （已移植到 Python 3: 整数除法用 //，iteritems 改为 items。）
# =====================================================================
import numpy as np
import analyze_stats          # 借用其 calc_vec_energy() 做能量归一化

def slice_timeseries(x, l=128, d=64, max_k = None):
    """把复信号 x 切成形状 [k, 2, l] 的样本（2 通道 = I/Q）。
    l: 窗口长度(默认128)；d: 滑动步长(默认64，即 50% 重叠)；
    max_k: 样本数上限；每个样本按其能量归一化。"""
    k = (len(x) - l + 1) // d     # 整数除法: 最多可切出的窗口数
    if not max_k == None:
        k = min(k, max_k)
    X = np.zeros([k,2,l], dtype=np.float32)
    for i in range(0,k):
        # Rect Window
        w = np.ones([l])                       # 矩形窗（即不加窗）
        # Sin Window
        #w = np.sin(np.arange(0,np.pi,np.pi/l))
        x_i =      x[i*d:i*d+l] * w            # 取从 i*d 开始的一个窗口
        X[i,0,:] = np.real(x_i)                # 通道 0: I 路（实部）
        X[i,1,:] = np.imag(x_i)                # 通道 1: Q 路（虚部）
        energy = analyze_stats.calc_vec_energy(X[i])
        X[i,0,:] /= energy                     # 按能量归一化，统一不同样本的幅度尺度
        X[i,1,:] /= energy
    return X

def slice_timeseries_dict(td, l=128, d=64, max_k = None):
    """对字典批量调用 slice_timeseries：
    把 {键: 复信号} 中每个信号分别切片，供 RML2016.04c 生成脚本使用。"""
    nd = {}
    for k,v in td.items():
        nd[k] = slice_timeseries(v,l,d,max_k)
    return nd

def slice_timeseries_real(x, l=128, d=64, max_k = None):
    """把实信号 x 切成形状 [k, 1, l] 的单通道样本（不做能量归一化）。"""
    k = (len(x) - l + 1) // d
    if not max_k == None:
        k = max(k, max_k)                      # 注意: 此处为 max()，与 slice_timeseries 的 min() 相反，疑似原作者笔误
    X = np.zeros([k,1,l], dtype=np.float32)
    for i in range(0,k):
        x_i =      x[i*d:i*d+l]
        X[i,0,:] = x_i
    return X


def slice_timeseries_real_dict(td, l=128, d=64, max_k = None):
    """对字典批量调用 slice_timeseries_real，供 alphabet 数据集生成脚本使用。"""
    nd = {}
    for k,v in td.items():
        nd[k] = slice_timeseries_real(v,l,d,max_k)
    return nd
