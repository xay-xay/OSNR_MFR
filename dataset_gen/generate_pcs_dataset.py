#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =====================================================================
# generate_pcs_dataset.py —— 概率整形(PCS)数据集生成(独立入口)
#
# 与 generate_dataset.py 的唯一区别: 发射符号按 Maxwell-Boltzmann
# 整形分布采样(见 pcs_constellation.py)。其余环节 —— 信道(BTB ASE)、
# 硬件损伤、DSP 链(MF→IQ补偿→定时恢复→CMA)、45 维特征、70/20/10
# 分层划分 —— 全部复用原模块, 输出 npz 格式与主数据集完全兼容
# (model/ 侧从 meta 读类别表, 无需改动)。
#
# 支持格式: 均匀 QAM4/16/64/256 + 整形 QAM4_PCS/16_PCS/64_PCS/256_PCS
#
# 用法:
#   # 默认: 4 种整形格式 × OSNR 0~30dB 步长2 × 100 样本 = 6400 组
#   python generate_pcs_dataset.py --out-dir output/pcs
#
#   # 整形 + 均匀混排(同几何 A/B 对照), 自定义 ν
#   python generate_pcs_dataset.py --out-dir output/pcs_ab \
#       --mods QAM64_PCS QAM64 QAM256_PCS QAM256 --nu QAM256_PCS=1.0
#
#   # 快速冒烟
#   python generate_pcs_dataset.py --out-dir output/_smoke_pcs \
#       --mods QAM16_PCS --osnr-min 12 --osnr-max 12 \
#       --n-samples 2 --n-symbols 4096 --cma-extra 1024
# =====================================================================
import argparse
import json
import os
import time

import numpy as np

import config as C
from transmitter import rrc_taps
import channel as ch
import dsp
import impairments as IM
import features as F
import pcs_constellation as PC


def transmit_pdm_pcs(mod_name, nu, rng, n_symbols, sps, beta, span):
    """PCS 版 PDM 发射机: 与 transmitter.transmit_pdm 相同的上采样 +
    RRC 成形 + 每偏振功率归一, 仅符号生成换成整形采样。"""
    h = rrc_taps(sps, span, beta)
    pols = []
    for _ in range(2):                          # X/Y 两偏振独立数据
        syms = PC.random_symbols(mod_name, rng, n_symbols, nu)
        up = np.zeros(n_symbols * sps, dtype=np.complex128)
        up[::sps] = syms                        # 2 倍上采样
        w = np.convolve(up, h)                  # RRC 成形
        w /= np.sqrt(np.mean(np.abs(w) ** 2))   # 每偏振功率归一到 1
        pols.append(w)
    return np.vstack(pols)


def gen_one_sample(cond, rng, h_tx, args, w0=None):
    """单样本完整链路(BTB): 与主脚本 generate_dataset.gen_one_sample
    逐环节一致, 两处替换 —— 发射用整形采样; CMA 的 R 用整形加权模。"""
    n_sym_total = args.n_symbols + args.cma_extra
    fmt = PC.resolve(cond["mod"], cond["nu"])   # 该格式的星座/R/PMF

    # ---- 发射(整形符号) + TX IQ 失衡 ----
    wav = transmit_pdm_pcs(cond["mod"], cond["nu"], rng, n_sym_total,
                           C.SPS, C.ROLLOFF, C.FILTER_SPAN)
    imp = IM.draw_impairments(rng, C.IMPAIR)
    wav = IM.inject_tx(wav, imp)

    # ---- BTB 光信道: 仅 ASE 噪声加载(按 OSNR 定标, 按实际功率测) ----
    wav = ch.add_ase(wav, cond["osnr"], rng, C.SPS * C.BAUD_RATE,
                     ref_bw_hz=C.OSNR_REF_BW_HZ, mode=args.osnr_mode)

    # ---- 接收前端损伤: RX IQ → 定时(τ+SRO) → ADC ----
    wav = IM.inject_rx(wav, imp, C.SPS * C.BAUD_RATE)

    # ---- DSP 链: MF → I/Q 补偿 → Gardner 定时恢复 → CMA ----
    rx = dsp.matched_filter(wav, h_tx)          # Coherent Rx 匹配滤波
    rx = IM.iq_compensate(rx, imp)              # I/Q 失衡盲补偿(GS)
    rx = IM.timing_recover(rx, imp)             # 定时恢复
    eq, w = dsp.mimo_cma(rx,                    # 2x2 MIMO CMA
                         n_taps=args.cma_taps, mu=args.cma_mu,
                         R=fmt["R_cma"], w0=w0)
    x_pol = eq[0, args.cma_extra: args.cma_extra + args.n_symbols]
    return x_pol, w


def main():
    ap = argparse.ArgumentParser(description="概率整形(PCS)数据集生成(BTB)")
    ap.add_argument("--out-dir", default="output/pcs", help="输出目录")
    ap.add_argument("--mods", nargs="*", default=None,
                    help="格式列表(默认 4 种整形); 可混入均匀 "
                         "QAM4/16/64/256 作同几何对照")
    ap.add_argument("--nu", action="append", default=None, metavar="NAME=VALUE",
                    help="覆盖整形参数 ν, 如 --nu QAM256_PCS=1.0 (可多次)")
    ap.add_argument("--osnr-min", type=float, default=0.0, help="OSNR 下限(dB)")
    ap.add_argument("--osnr-max", type=float, default=30.0, help="OSNR 上限(dB)")
    ap.add_argument("--osnr-step", type=float, default=2.0, help="OSNR 步长(dB)")
    ap.add_argument("--n-samples", type=int, default=C.N_SAMPLES,
                    help="每条件样本组数(论文 100)")
    ap.add_argument("--n-symbols", type=int, default=C.N_SYMBOLS,
                    help="每样本特征符号数(论文 16384)")
    ap.add_argument("--cma-extra", type=int, default=C.CMA_EXTRA,
                    help="CMA 前置收敛符号数(丢弃)")
    ap.add_argument("--cma-taps", type=int, default=C.CMA_N_TAPS)
    ap.add_argument("--cma-mu", type=float, default=5e-3,
                    help="CMA 归一化步长(整形星座收敛慢, 默认 5e-3; "
                         "均匀格式口径 1e-3 见 config.CMA_MU)")
    ap.add_argument("--osnr-mode", default=C.OSNR_MODE,
                    choices=["0.1nm", "inband"], help="OSNR 定标口径")
    ap.add_argument("--seed", type=int, default=C.BASE_SEED)
    ap.add_argument("--no-raw", action="store_true", help="不保存原始均衡符号")
    ap.add_argument("--no-warm-start", action="store_true",
                    help="禁用同条件内 CMA 权重热启动")
    args = ap.parse_args()

    # ---- 格式与 ν 解析 ----
    mods = args.mods if args.mods else PC.PCS_MODS
    for m in mods:
        if m not in PC.ALL_MODS:
            raise SystemExit("不支持格式 %s (支持 %s)" % (m, PC.ALL_MODS))
    nus = {}
    for kv in (args.nu or []):
        k, _, v = kv.partition("=")
        k, v = k.strip(), float(v)
        if k not in PC.ALL_MODS:
            raise SystemExit("--nu 的格式名不认识: %s" % k)
        nus[k] = v

    osnr_grid = [float(o) for o in
                 np.arange(args.osnr_min, args.osnr_max + 1e-9, args.osnr_step)]
    conds = [dict(mod=m, nu=nus.get(m), osnr=o)
             for m in mods for o in osnr_grid]
    n_total = len(conds) * args.n_samples
    print("PCS 数据集: 格式=%d %s" % (len(mods), mods))
    print("OSNR 网格: %.0f~%.0f dB 步长 %g (%d 点)  每条件 %d 样本  总样本 %d"
          % (args.osnr_min, args.osnr_max, args.osnr_step,
             len(osnr_grid), args.n_samples, n_total))
    PC.report(mods, nus)                        # 启动前先打印整形参数表

    os.makedirs(args.out_dir, exist_ok=True)
    h_tx = rrc_taps(C.SPS, C.FILTER_SPAN, C.ROLLOFF)

    X = np.zeros((n_total, F.N_FEATURES), dtype=np.float64)
    raw_path = raw = None
    if not args.no_raw:
        raw_path = os.path.join(args.out_dir, "pcs_cma_raw.npy")
        raw = np.lib.format.open_memmap(raw_path, mode="w+",
                                        dtype=np.complex64,
                                        shape=(n_total, args.n_symbols))
    y_mod = np.zeros(n_total, dtype=np.int8)
    y_osnr = np.zeros(n_total, dtype=np.float32)

    t0 = time.time()
    n_done = 0
    for ci, cond in enumerate(conds):
        cstart = time.time()
        w = None                                # 条件内 CMA 权重热启动
        for si in range(args.n_samples):
            rng = np.random.default_rng([args.seed, ci, si])  # 确定性种子
            row = ci * args.n_samples + si
            for attempt in range(2):            # CMA 发散保护: 重试一次
                x_pol, w = gen_one_sample(
                    cond, rng, h_tx, args,
                    w0=w if (attempt == 0 and not args.no_warm_start) else None)
                if np.all(np.isfinite(x_pol)) and x_pol.size == args.n_symbols:
                    break
                w = None
                print("  [warn] 样本 (%s@%gdB #%d) 非有限值, 重置权重重试"
                      % (cond["mod"], cond["osnr"], si))
            X[row] = F.extract_features(x_pol)
            if raw is not None:
                raw[row] = x_pol.astype(np.complex64)
            y_mod[row] = mods.index(cond["mod"])   # 类别按本地 mods 表编号
            y_osnr[row] = cond["osnr"]
        n_done += args.n_samples
        eta = (time.time() - t0) / n_done * (n_total - n_done)
        print("[%3d/%3d] %-12s OSNR=%5.1fdB | 本条件 %.1fs | 剩余~%.1fmin"
              % (ci + 1, len(conds), cond["mod"], cond["osnr"],
                 time.time() - cstart, eta / 60.0))

    # ---- z-score 归一化统计(训练代码会只用训练集重算, 更严格) ----
    mean, std = F.zscore_stats(X)
    X_norm = ((X - mean) / std).astype(np.float32)

    # ---- 70/20/10 分层随机划分(按条件分层, 固定种子, 同主脚本) ----
    rng_split = np.random.default_rng(C.SPLIT_SEED)
    n_tr = int(round(args.n_samples * C.SPLIT[0]))
    n_te = int(round(args.n_samples * C.SPLIT[1]))
    train_idx, test_idx, val_idx = [], [], []
    for ci in range(len(conds)):
        rows = np.arange(ci * args.n_samples, (ci + 1) * args.n_samples)
        perm = rng_split.permutation(rows)
        train_idx.extend(perm[:n_tr])
        test_idx.extend(perm[n_tr:n_tr + n_te])
        val_idx.extend(perm[n_tr + n_te:])

    meta = dict(
        dataset="PCS probabilistic constellation shaping, BTB",
        shaping="maxwell-boltzmann P(x) ~ exp(-nu*|x|^2)",
        mods=mods,
        fmt_info={m: dict(nu=PC.resolve(m, nus.get(m))["nu"],
                          entropy=PC.resolve(m, nus.get(m))["entropy"],
                          R_cma=PC.resolve(m, nus.get(m))["R_cma"],
                          max_amp=PC.resolve(m, nus.get(m))["max_amp"])
                  for m in mods},
        osnr_grid=osnr_grid, n_samples_per_cond=args.n_samples,
        n_symbols=args.n_symbols, n_total=n_total,
        baud_rate=C.BAUD_RATE, sps=C.SPS, rolloff=C.ROLLOFF,
        osnr_mode=args.osnr_mode, osnr_ref_bw_hz=C.OSNR_REF_BW_HZ,
        equalizer=dict(mode="cma", taps=args.cma_taps, mu=args.cma_mu,
                       extra=args.cma_extra,
                       warm_start=not args.no_warm_start),
        feature_cfg=dict(ah_bins=F.N_BINS, ah_range=(0.0, C.AH_RANGE_MAX),
                         entropy="energy (paper Eq.6)",
                         vel_from_amplitude=C.VEL_FROM_AMPLITUDE,
                         vel_abs=C.VEL_ABS),
        impairments=C.IMPAIR,
        rx_chain=["matched_filter", "iq_comp_gs", "gardner_tr", "cma"],
        split=C.SPLIT, split_seed=C.SPLIT_SEED, seed=args.seed,
    )

    if raw is not None:
        raw.flush()
        del raw
    npz_path = os.path.join(args.out_dir, "pcs_cma_features.npz")
    np.savez_compressed(
        npz_path,
        X=X.astype(np.float32), X_norm=X_norm,
        feat_mean=mean.astype(np.float32), feat_std=std.astype(np.float32),
        feature_names=np.array(F.FEATURE_NAMES),
        y_mod=y_mod, y_osnr=y_osnr,
        y_len_km=np.zeros(n_total, dtype=np.float32),      # BTB: 全 0 占位
        y_dgd_ps=np.zeros(n_total, dtype=np.float32),
        y_lw_khz=np.zeros(n_total, dtype=np.float32),
        y_launch_dbm=np.zeros(n_total, dtype=np.float32),
        train_idx=np.array(sorted(train_idx), dtype=np.int64),
        test_idx=np.array(sorted(test_idx), dtype=np.int64),
        val_idx=np.array(sorted(val_idx), dtype=np.int64),
        meta=json.dumps(meta, ensure_ascii=False),
    )
    print("完成: 总耗时 %.1f min" % ((time.time() - t0) / 60.0))
    print("特征文件: %s" % npz_path)
    if raw_path:
        print("原始符号: %s  (%.1f MB)"
              % (raw_path, os.path.getsize(raw_path) / 1e6))


if __name__ == "__main__":
    main()
