#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =====================================================================
# rebuild_features.py —— 从已保存的原始均衡符号重提特征
#
# 数据集的 839MB 原始符号(*_raw.npy)是仿真结果的完整记录, 特征只是它
# 的函数。改特征定义(如入口功率归一化)时无需重新仿真, 从 raw 重算即可。
#
# 用法:
#   python rebuild_features.py --dir output/btb            # 自动找 npz/raw
#   python rebuild_features.py --npz x.npz --raw x.npy --suffix _pnorm
#
# 输出: <原文件名><suffix>_features.npz, 标签/划分索引原样保留,
#       meta 中 feature_cfg 记录当前 config 的特征口径。
# =====================================================================
import argparse
import glob
import json
import os

import numpy as np

import config as C
import features as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="output/btb")
    ap.add_argument("--npz", default=None)
    ap.add_argument("--raw", default=None)
    ap.add_argument("--suffix", default="_pnorm", help="输出文件名后缀")
    args = ap.parse_args()

    if args.npz:
        npz_path, raw_path = args.npz, args.raw
    else:
        cands = sorted(glob.glob(os.path.join(args.dir, "*_features.npz")))
        if not cands:
            raise SystemExit("目录下没找到 *_features.npz: %s" % args.dir)
        npz_path = cands[0]
        raw_path = npz_path.replace("_features.npz", "_raw.npy")
    if not (raw_path and os.path.exists(raw_path)):
        raise SystemExit("找不到原始符号文件: %s" % raw_path)

    d = np.load(npz_path, allow_pickle=True)
    raw = np.load(raw_path, mmap_mode="r")
    n, n_sym = raw.shape
    print("源: %s  (%d 样本 × %d 符号)" % (npz_path, n, n_sym))
    print("特征口径: power_norm=%s  ah_range=[0,%.1f]  vel_from_amplitude=%s"
          % (C.FEAT_POWER_NORM, C.AH_RANGE_MAX, C.VEL_FROM_AMPLITUDE))

    X = np.zeros((n, F.N_FEATURES), dtype=np.float64)
    clip_tot = 0.0
    clip_n = 0
    for i in range(n):
        x = np.asarray(raw[i])
        X[i] = F.extract_features(x)
        # 诊断: 功率归一化后超出 AH 上界的占比(只统计需关注的样本)
        if i % 100 == 0:
            a = np.abs(x)
            if C.FEAT_POWER_NORM:
                a = a / np.sqrt(np.mean(a ** 2) + 1e-30)
            c = float(np.mean(a > C.AH_RANGE_MAX))
            clip_tot += c
            clip_n += 1
    print("AH 超界占比(抽样均值): %.4f%%" % (100 * clip_tot / max(clip_n, 1)))

    mean, std = F.zscore_stats(X)
    X_norm = ((X - mean) / std).astype(np.float32)

    try:
        meta = json.loads(str(d["meta"]))
    except Exception:
        meta = {}
    meta["feature_cfg"] = dict(
        ah_bins=F.N_BINS, ah_range=(0.0, C.AH_RANGE_MAX),
        entropy="energy (paper Eq.6)",
        power_norm=C.FEAT_POWER_NORM,
        vel_from_amplitude=C.VEL_FROM_AMPLITUDE, vel_abs=C.VEL_ABS,
        cumulants=["|C40|", "C42"])
    meta["rebuilt_from"] = os.path.basename(raw_path)

    out = npz_path.replace("_features.npz",
                           "%s_features.npz" % args.suffix)
    np.savez_compressed(
        out,
        X=X.astype(np.float32), X_norm=X_norm,
        feat_mean=mean.astype(np.float32), feat_std=std.astype(np.float32),
        feature_names=np.array(F.FEATURE_NAMES),
        y_mod=d["y_mod"], y_osnr=d["y_osnr"],
        y_len_km=d["y_len_km"], y_dgd_ps=d["y_dgd_ps"],
        y_lw_khz=d["y_lw_khz"], y_launch_dbm=d["y_launch_dbm"],
        train_idx=d["train_idx"], test_idx=d["test_idx"],
        val_idx=d["val_idx"],
        meta=json.dumps(meta, ensure_ascii=False))
    print("已保存: %s  (%.1f MB)" % (out, os.path.getsize(out) / 1e6))


if __name__ == "__main__":
    main()
