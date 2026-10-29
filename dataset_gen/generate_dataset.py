#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =====================================================================
# generate_dataset.py —— 数据集生成主入口
#
# 用法示例:
#   # 主数据集(BTB, 7 格式 × 21 OSNR 点 × 100 样本, 共 14700 组)
#   python generate_dataset.py --preset btb --out-dir output/btb_v2
#
#   # 鲁棒性预设(光纤长度 / DGD / 线宽 / 入纤功率)
#   python generate_dataset.py --preset linewidth --out-dir output/linewidth
#
#   # 快速冒烟(微规模, 用于验证链路)
#   python generate_dataset.py --preset btb --out-dir _smoke \
#       --mods QPSK QAM16 --osnrs -10 12 30 --n-samples 2 --n-symbols 8192
#
# 每个样本的完整链路:
#   发射(PRBS->格雷QAM->上采样->RRC->双偏振, 功率归一)
#   -> [入纤功率标定] -> [SSFM 光纤(可选)] -> [DGD(可选)]
#   -> [激光线宽相噪(可选)] -> ASE 噪声加载(按 OSNR 定标)
#   -> RRC 匹配滤波(Coherent Rx) -> [I/Q 失衡补偿] -> [色散补偿(有光纤时)]
#   -> [定时恢复(Gardner)] -> 2x2 MIMO CMA
#   -> 丢弃前置收敛符号 -> X 偏振 16384 符号
#   -> 45 维特征(存矩阵) + 原始均衡符号(存 memmap .npy, 每行一样本)
#
# 输出文件(out-dir 下):
#   <preset>_cma_features.npz : X(N,45) 原始特征 / X_norm / z-score 统计 /
#                           标签(调制/OSNR/长度/DGD/线宽/功率) /
#                           70-20-10 划分索引 / meta(JSON)
#   <preset>_cma_raw.npy      : (N, n_symbols) complex64 原始均衡符号(memmap)
# =====================================================================
import argparse
import json
import os
import time

import numpy as np

import config as C
from transmitter import transmit_pdm, rrc_taps
import channel as ch
import dsp
import features as F


def build_conditions(args):
    """按预设 + CLI 过滤器生成条件列表。"""
    mods = args.mods if args.mods else None
    conds = C.PRESETS[args.preset](mods=mods)

    if args.lengths is not None:          # fiber 预设: 过滤长度
        keep = set(float(v) for v in args.lengths)
        conds = [c for c in conds if c["length_km"] in keep]
    if args.values is not None:           # dgd/线宽/功率预设: 过滤损伤值
        keep = set(float(v) for v in args.values)
        conds = [c for c in conds
                 if (c["dgd_ps"] in keep or c["linewidth_hz"] in keep
                     or c["launch_dbm"] in keep)]
    if args.osnrs is not None:            # 所有预设: 过滤 OSNR 点
        keep = set(float(v) for v in args.osnrs)
        conds = [c for c in conds if c["osnr"] in keep]

    if not conds:
        raise SystemExit("过滤后条件数为 0, 请检查 --mods/--osnrs/--lengths/--values")
    return conds


def gen_one_sample(cond, rng, h_tx, args, w0=None):
    """生成单个样本。
    返回 (x_pol, w): X 偏振均衡后符号 (n_symbols,) + CMA 最终权重(热启动用)。"""
    import impairments as IM
    n_sym_total = args.n_symbols + args.cma_extra

    # ---- 发射 ----
    wav = transmit_pdm(cond["mod"], rng, n_sym_total,
                       sps=C.SPS, beta=C.ROLLOFF, span=C.FILTER_SPAN)
    # 发射端 IQ 失衡(每样本参数随机, 见 config.IMPAIR)
    imp = IM.draw_impairments(rng, C.IMPAIR)
    wav = IM.inject_tx(wav, imp)

    # ---- 光信道（严格按真实物理传输顺序：激光器 -> 光纤/PMD -> ASE）----
    if cond["linewidth_hz"] > 0:          # ① 发射端激光器线宽(组合线宽等效)
        wav = ch.apply_phase_noise(wav, cond["linewidth_hz"], rng,
                                   C.SPS * C.BAUD_RATE)
    if cond["length_km"] > 0:             # ② SSMF 传输(含入纤功率标定与克尔非线性)
        wav = ch.ssfm(wav, cond["length_km"], C.FIBER,
                      launch_dbm=cond["launch_dbm"], fs=C.SPS * C.BAUD_RATE,
                      wavelength=C.WAVELENGTH)
    if cond["dgd_ps"] > 0:                # ③ 光纤 PMD(一阶 DGD)
        wav = ch.apply_dgd(wav, cond["dgd_ps"], rng, C.SPS * C.BAUD_RATE)
    # ④ ASE 噪声加载(按 OSNR 定标) —— 与 EDFA 级联积累等效, 放在最后
    wav = ch.add_ase(wav, cond["osnr"], rng, C.SPS * C.BAUD_RATE,
                     ref_bw_hz=C.OSNR_REF_BW_HZ, mode=args.osnr_mode)

    # ---- 接收前端损伤: RX IQ → 定时(τ+SRO) → ADC ----
    wav = IM.inject_rx(wav, imp, C.SPS * C.BAUD_RATE)

    # ---- 接收 DSP: Coherent Rx(MF) → I/Q 失衡补偿 → CD 补偿 → 定时恢复 → CMA ----
    rx = dsp.matched_filter(wav, h_tx)                 # Coherent Rx 前端: RRC 匹配滤波
    rx = IM.iq_compensate(rx, imp)                     # I/Q 失衡补偿(GS)
    if cond["length_km"] > 0:                          # 色散补偿
        rx = dsp.cd_compensate(rx, cond["length_km"],
                               C.FIBER["D_ps_nm_km"],
                               C.SPS * C.BAUD_RATE, C.WAVELENGTH)
    rx = IM.timing_recover(rx, imp)                    # 定时恢复(Gardner)
    eq, w = dsp.equalize(rx, cond["mod"],              # CMA 均衡
                         n_taps=args.cma_taps, mu=args.cma_mu, w0=w0)
    x_pol = eq[0, args.cma_extra: args.cma_extra + args.n_symbols]
    return x_pol, w


def main():
    ap = argparse.ArgumentParser(description="LA-MT-SEDNN 论文数据集生成")
    ap.add_argument("--preset", default="btb",
                    choices=list(C.PRESETS), help="实验预设")
    ap.add_argument("--out-dir", default="output", help="输出目录")
    ap.add_argument("--mods", nargs="*", default=None, help="调制格式过滤")
    ap.add_argument("--osnrs", nargs="*", type=float, default=None,
                    help="OSNR(dB) 过滤")
    ap.add_argument("--lengths", nargs="*", type=float, default=None,
                    help="fiber 预设: 传输距离(km) 过滤")
    ap.add_argument("--values", nargs="*", type=float, default=None,
                    help="dgd/linewidth/launch_power 预设: 损伤值过滤"
                         "(ps / Hz / dBm)")
    ap.add_argument("--n-samples", type=int, default=C.N_SAMPLES,
                    help="每条件样本组数(论文 100)")
    ap.add_argument("--n-symbols", type=int, default=C.N_SYMBOLS,
                    help="每样本特征符号数(论文 16384)")
    ap.add_argument("--cma-extra", type=int, default=C.CMA_EXTRA,
                    help="CMA 前置收敛符号数(丢弃)")
    ap.add_argument("--cma-taps", type=int, default=C.CMA_N_TAPS)
    ap.add_argument("--cma-mu", type=float, default=C.CMA_MU)
    ap.add_argument("--osnr-mode", default=C.OSNR_MODE,
                    choices=["0.1nm", "inband"], help="OSNR 定标口径")
    ap.add_argument("--seed", type=int, default=C.BASE_SEED)
    ap.add_argument("--no-raw", action="store_true",
                    help="不保存原始均衡符号(默认保存)")
    ap.add_argument("--no-warm-start", action="store_true",
                    help="禁用同条件内 CMA 权重热启动")
    args = ap.parse_args()

    conds = build_conditions(args)
    n_total = len(conds) * args.n_samples
    print("预设=%s  条件数=%d  每条件样本=%d  总样本=%d  符号数/样本=%d"
          % (args.preset, len(conds), args.n_samples, n_total,
             args.n_symbols))

    os.makedirs(args.out_dir, exist_ok=True)
    h_tx = rrc_taps(C.SPS, C.FILTER_SPAN, C.ROLLOFF)

    X = np.zeros((n_total, F.N_FEATURES), dtype=np.float64)
    raw_path = None
    raw = None
    if not args.no_raw:
        raw_path = os.path.join(args.out_dir,
                                "%s_cma_raw.npy" % args.preset)
        raw = np.lib.format.open_memmap(raw_path, mode="w+",
                                        dtype=np.complex64,
                                        shape=(n_total, args.n_symbols))

    y_mod = np.zeros(n_total, dtype=np.int8)
    y_osnr = np.zeros(n_total, dtype=np.float32)
    y_len = np.zeros(n_total, dtype=np.float32)
    y_dgd = np.zeros(n_total, dtype=np.float32)
    y_lw = np.zeros(n_total, dtype=np.float32)
    y_pwr = np.zeros(n_total, dtype=np.float32)

    t0 = time.time()
    n_done = 0
    for ci, cond in enumerate(conds):
        cstart = time.time()
        w = None                                    # 条件内 CMA 权重热启动
        for si in range(args.n_samples):
            rng = np.random.default_rng([args.seed, ci, si])  # 确定性种子
            row = ci * args.n_samples + si
            for attempt in range(2):          # CMA 发散保护: 重置权重重试一次
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
            y_mod[row] = C.MODULATIONS.index(cond["mod"])
            y_osnr[row] = cond["osnr"]
            y_len[row] = cond["length_km"]
            y_dgd[row] = cond["dgd_ps"]
            y_lw[row] = cond["linewidth_hz"] / 1e3   # 存 kHz
            y_pwr[row] = cond["launch_dbm"]
        n_done += args.n_samples
        dt = time.time() - t0
        eta = dt / n_done * (n_total - n_done)
        print("[%3d/%3d] %-6s OSNR=%5.1fdB len=%3.0fkm dgd=%4.1fps "
              "lw=%6.1fkHz pwr=%+5.1fdBm | 本条件 %.1fs | 剩余~%.1fmin"
              % (ci + 1, len(conds), cond["mod"], cond["osnr"],
                 cond["length_km"], cond["dgd_ps"],
                 cond["linewidth_hz"] / 1e3, cond["launch_dbm"],
                 time.time() - cstart, eta / 60.0))

    # ---- z-score 归一化统计(全数据集; 严格做法应只用训练集, 训练代码可重算) ----
    mean, std = F.zscore_stats(X)
    X_norm = ((X - mean) / std).astype(np.float32)

    # ---- 70/20/10 分层随机划分(按条件分层, 固定种子) ----
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
        paper="Optics Express 34(8):14686 (2026), LA-MT-SEDNN, BUPT",
        preset=args.preset,
        n_conditions=len(conds), n_samples_per_cond=args.n_samples,
        n_symbols=args.n_symbols, n_total=n_total,
        baud_rate=C.BAUD_RATE, sps=C.SPS, rolloff=C.ROLLOFF,
        osnr_mode=args.osnr_mode, osnr_ref_bw_hz=C.OSNR_REF_BW_HZ,
        equalizer=dict(mode="cma", taps=args.cma_taps, mu=args.cma_mu,
                       extra=args.cma_extra, warm_start=not args.no_warm_start),
        feature_cfg=dict(ah_bins=F.N_BINS, ah_range=(0.0, C.AH_RANGE_MAX),
                         entropy="energy (paper Eq.6)",
                         vel_from_amplitude=C.VEL_FROM_AMPLITUDE,
                         vel_abs=C.VEL_ABS),
        channel_order=["linewidth", "ssfm_fiber", "dgd", "ase"],
        impairments=C.IMPAIR,
        rx_chain=["matched_filter", "iq_comp_gs", "cd_comp",
                  "gardner_tr", "cma"],
        fiber=C.FIBER, split=C.SPLIT, split_seed=C.SPLIT_SEED,
        seed=args.seed,
        mods=C.MODULATIONS,
        conditions=[dict(c) for c in conds],
    )

    if raw is not None:
        raw.flush()
        del raw
    npz_path = os.path.join(args.out_dir,
                            "%s_cma_features.npz" % args.preset)
    np.savez_compressed(
        npz_path,
        X=X.astype(np.float32), X_norm=X_norm,
        feat_mean=mean.astype(np.float32), feat_std=std.astype(np.float32),
        feature_names=np.array(F.FEATURE_NAMES),
        y_mod=y_mod, y_osnr=y_osnr,
        y_len_km=y_len, y_dgd_ps=y_dgd, y_lw_khz=y_lw, y_launch_dbm=y_pwr,
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
