# -*- coding: utf-8 -*-
# =====================================================================
# config.py —— 全局参数与实验预设
#
# 参数取值逐条对应论文:
#   Zhang et al., "Multi-feature fusion for joint modulation format
#   recognition and OSNR monitoring based on a lightweight adaptive
#   attention network," Optics Express 34(8), 14686 (2026).
#   （北邮 BUPT OE 论文, LA-MT-SEDNN）
#
# 论文第 3 节数据集要求:
#   * 28 GBaud PDM-QAM: 4/8/16/32/64QAM 共 5 种格式
#   * OSNR 网格(1 dB 步进): 4QAM 9~20, 8QAM 10~22, 16QAM 15~27,
#     32QAM 18~30, 64QAM 18~30 (dB)
#   * 每个 (格式, OSNR) 100 组样本, 每组 16384 符号(CMA 均衡后, X 偏振)
#   * 共 6400 组; 70%/20%/10% 训练/测试/验证划分
# 论文鲁棒性实验(第 3 节后半):
#   * 光纤长度 0/80/160/240/320 km (SSMF, 每跨 80 km + EDFA 16 dB)
#   * DGD 0~10 ps; 激光线宽 50~250 kHz; 入纤功率 -2~6 dBm @ 400 km
#
# ⚠ 论文未明确给出的参数在注释中以 [假设] 标出, 请 review 时重点关注。
# =====================================================================

# ----------------------------- 发射机 -------------------------------
BAUD_RATE   = 28e9        # 28 GBaud (论文)
SPS         = 2           # 每符号采样数(2 sps, 供 T/2 间隔 CMA 使用)
ROLLOFF     = 0.2         # RRC 滚降系数 [假设: 论文未给出, VPI 常用 0.2, 可改]
FILTER_SPAN = 10          # RRC 滤波器跨度(符号数)

# ----------------------------- 数据集 -------------------------------
N_SYMBOLS   = 16384       # 每样本参与特征提取的符号数 (论文要求 16384)
N_SAMPLES   = 100         # 每个 (调制, OSNR) 条件的样本组数 (论文要求 100)
CMA_EXTRA   = 4096        # CMA 收敛用前置训练符号数, 提特征前丢弃
                          # (保证送入特征的恰好是收敛后的 16384 符号)

# ----------------------------- OSNR 口径 ----------------------------
# OSNR 定义为 0.1 nm 参考带宽(1550 nm 处 ≈ 12.5 GHz)内信号功率与
# ASE 噪声功率之比 —— 光通信标准口径, 论文使用 VPI 默认该口径。
# "0.1nm": 噪声 PSD 按 12.5 GHz 参考带宽标定 (标准, 默认)
#          在 28 GBaud 符号带内, SNR = OSNR - 10log10(28/12.5) ≈ OSNR - 3.5 dB
# "inband": 直接令匹配滤波后符号带内 SNR = OSNR 值 (部分复现采用的简化口径)
OSNR_REF_BW_HZ = 12.5e9
OSNR_MODE      = "0.1nm"
WAVELENGTH     = 1550e-9

# ----------------------------- 光纤 (论文第 3 节) -------------------
FIBER = dict(
    alpha_db_km  = 0.2,    # 衰减系数 (dB/km)
    D_ps_nm_km   = 17.0,   # 色散系数 (ps/nm/km)
    gamma_per_Wkm= 1.3,    # 非线性系数 (1/W/km, 论文实验节口径)
    span_km      = 80.0,   # 每跨长度
    ssfm_step_km = 0.5,    # SSFM 分步长度 [假设: 论文未给出, 0.5 km 已足够]
    manakov      = 8.0/9.0,# PDM 非线性的 Manakov 系数 [简化: 忽略 WDM XPM]
)

# ----------------------------- 均衡器 -------------------------------
# 论文正文口径: "constant modulus algorithm (CMA)-equalized signal"
# 故统一使用 CMA（论文 Fig.2 展示的也正是 CMA 均衡后的星座）。
CMA_N_TAPS = 13           # T/2 间隔抽头数(奇数), 覆盖 ±3 个符号
CMA_MU     = 1e-3         # 归一化步长
CMA_WARM_START = True     # 同一条件内 100 个样本间热启动传递均衡器权重

# ----------------------------- 特征提取 -----------------------------
# 幅度直方图: 全局固定分箱范围 [0, AH_RANGE_MAX]（禁止每样本自适应 max）
#   2.5 的依据: CMA 输出下各格式最大幅度 ≤2.44, 超界占比 ≤0.04%
AH_RANGE_MAX = 2.5

# σ_vel 口径: True = 先取幅度序列 a_i=|x_i| 再差分(论文正文"difference in
# amplitude between consecutive samples"); False = 论文式(3)字面写法
# v_i=|x_{i+1}−x_i|(复差分取模)
VEL_FROM_AMPLITUDE = True
VEL_ABS    = False        # 差分是否再取绝对值
# 熵特征: 按论文式(6) 逐符号归一化能量 E_i=|x_i|²/Σ|x_j|² 计算(见 features.py)

# 特征入口功率归一化(方案 B): x ← x/√E|x|²。
#   True : 每样本尺度被除掉, 星座环钉在 1.0, AH 只剩形状 —— 推测为论文
#          口径(其 AH-only 消融仅 92.1%, 且 VPI 相干链通常做功率归一化);
#   False: 保留绝对尺度, AH 峰位随 OSNR 漂移(方案 A, 旧数据集口径,
#          AH-only 即达 99.9%, 难度低于论文)。
FEAT_POWER_NORM = True

# ------------------------ 硬件损伤注入(路径①) ------------------------
# 目的: 对齐论文数据难度(其 AH-only 仅 92.1%) + 补 sim-to-real 域差距。
# 每样本参数独立随机抽取(见 impairments.draw_impairments);
# 接收端有对应的盲补偿: I/Q 失衡补偿(GS) → Gardner-TR → CMA,
# 补偿残差(二阶小量 + 环路抖动)正是真实收发机中存在的残余损伤。
#   eps/phi_deg: IQ 增益误差与相位误差幅度范围(符号随机) ——
#                最大 10%/10° 对应约 -18dB 镜像抑制(未校准硬件水平)
#   cfo        : 载波频偏注入。已关闭 —— 接收 DSP 链不含 FOE(见 rx_chain),
#                残留频偏会使 GS-IQ 补偿失效(镜像项被平均成 0)。
#   tau_samples: 分数时延范围(2sps 采样网格); sro_ppm: 采样率偏移(ppm)
#   adc_bits   : ADC 量化位数(8bit 量化噪声 ~50dB, 影响极小, 仅作真实性)
IMPAIR = dict(
    enable     = True,
    tx_iq      = True,  eps=(0.02, 0.10), phi_deg=(2.0, 10.0),
    rx_iq      = True,  # 复用 tx 的 eps/phi_deg 范围
    cfo        = False,
    timing     = True,  tau_samples=(0.2, 1.8), sro_ppm=(50.0, 200.0),
    adc        = True,  adc_bits=8,
)

# ----------------------------- 划分 ---------------------------------
SPLIT       = (0.7, 0.2, 0.1)   # 训练/测试/验证 (论文: 随机划分 70/20/10)
SPLIT_SEED  = 2026
BASE_SEED   = 42           # 数据集全局随机种子

# ----------------------------- 调制与 OSNR 网格 ---------------------
# QAM 为论文原始 5 格式; BPSK/QPSK/8PSK 为扩展的恒包络 PSK 族
# (与 QAM 共用同一套 BTB 信道、损伤与 CMA 链路)。
# 注意: 原 QAM4 与 QPSK 星座完全相同(均为 45°/135°/225°/315° 单位圆四点),
#       物理不可分, 故统一保留 QPSK, 不再单列 QAM4。
MODULATIONS = ["QAM8", "QAM16", "QAM32", "QAM64",
               "BPSK", "QPSK", "8PSK"]

# 全部格式统一 OSNR 网格: 0 ~ 30 dB, 步进 2 dB (16 点)
OSNR_GRID = {
    m: list(range(0, 31, 2)) for m in MODULATIONS
}

# ----------------------------- 实验预设 -----------------------------
# 每个条件是一个 dict: mod / osnr / length_km / dgd_ps / linewidth_hz /
# launch_dbm。btb 为论文主数据集; 其余为鲁棒性实验预设(生成耗时较长,
# 可用 CLI --mods/--osnrs/--lengths/--values 过滤裁剪)。

DGD_VALUES_PS      = [0, 2, 4, 6, 8, 10]                      # [假设: 论文只给 0~10ps 范围]
LINEWIDTH_VALUES_HZ= [50e3, 100e3, 150e3, 200e3, 250e3]       # 组合线宽(TX+LO 合并等效)
LAUNCH_VALUES_DBM  = [-2, 0, 2, 4, 6]                          # @ 400 km
FIBER_LENGTHS_KM   = [0, 80, 160, 240, 320]
LAUNCH_TEST_KM     = 400.0


def _base_cond(mod, osnr, **kw):
    c = dict(mod=mod, osnr=float(osnr), length_km=0.0,
             dgd_ps=0.0, linewidth_hz=0.0, launch_dbm=0.0)
    c.update(kw)
    return c


def preset_btb(mods=None):
    """论文主数据集: BTB, 5 格式 × OSNR 网格, 共 6400 组。"""
    conds = []
    for m in (mods or MODULATIONS):
        for o in OSNR_GRID.get(m, []):
            conds.append(_base_cond(m, o))
    return conds


def preset_fiber(mods=None, lengths=None):
    """光纤长度鲁棒性: 0~320 km (SSFM + 每跨理想 EDFA), 其余同 btb。"""
    conds = []
    for L in (lengths if lengths is not None else FIBER_LENGTHS_KM):
        for m in (mods or MODULATIONS):
            for o in OSNR_GRID.get(m, []):
                conds.append(_base_cond(m, o, length_km=float(L)))
    return conds


def preset_dgd(mods=None, values=None):
    """一阶 PMD/DGD 鲁棒性: DGD 0~10 ps, BTB。"""
    conds = []
    for v in (values if values is not None else DGD_VALUES_PS):
        for m in (mods or MODULATIONS):
            for o in OSNR_GRID.get(m, []):
                conds.append(_base_cond(m, o, dgd_ps=float(v)))
    return conds


def preset_linewidth(mods=None, values=None):
    """激光线宽鲁棒性(论文唯一待复现项): 50~250 kHz 组合线宽, BTB。"""
    conds = []
    vs = values if values is not None else LINEWIDTH_VALUES_HZ
    for v in vs:
        for m in (mods or MODULATIONS):
            for o in OSNR_GRID.get(m, []):
                conds.append(_base_cond(m, o, linewidth_hz=float(v)))
    return conds


def preset_launch_power(mods=None, values=None):
    """光纤非线性鲁棒性: 400 km, 入纤功率 -2~6 dBm。"""
    conds = []
    vs = values if values is not None else LAUNCH_VALUES_DBM
    for v in vs:
        for m in (mods or MODULATIONS):
            for o in OSNR_GRID.get(m, []):
                conds.append(_base_cond(m, o, length_km=LAUNCH_TEST_KM,
                                        launch_dbm=float(v)))
    return conds


PRESETS = {
    "btb":          preset_btb,
    "fiber":        preset_fiber,
    "dgd":          preset_dgd,
    "linewidth":    preset_linewidth,
    "launch_power": preset_launch_power,
}
