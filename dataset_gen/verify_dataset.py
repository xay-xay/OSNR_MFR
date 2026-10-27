#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =====================================================================
# verify_dataset.py —— 数据集核验（对照论文 Fig.2 / Fig.3 / Fig.4）
#
# 用法:
#   python verify_dataset.py --dir output/btb          # 自动找 npz/npy
#   python verify_dataset.py --npz x.npz --raw x.npy   # 显式指定
#
# 输出到 <dir>/:
#   check_constellation.png  5 种格式的 CMA 均衡后星座(对照论文 Fig.2)
#   check_ah.png             各格式幅度直方图(对照论文 Fig.3)
#   check_boxplot.png        特征箱线图(对照论文 Fig.4 的条件对)
#   终端打印样本计数表与基本一致性检查
# =====================================================================
import argparse
import glob
import json
import os

import numpy as np

import config as C
import features as F

# 论文 Fig.4 的 10 个 (调制, OSNR) 条件对(原图用 QAM4, 此处对应 QPSK)
FIG4_CONDS = [("QPSK", 9), ("QPSK", 14), ("QAM8", 12), ("QAM8", 17),
              ("QAM16", 15), ("QAM16", 19), ("QAM32", 18), ("QAM32", 23),
              ("QAM64", 21), ("QAM64", 26)]
FIG4_FEATS = ["sigma_spread", "h_norm"]


def load(args):
    if args.npz:
        npz_path, raw_path = args.npz, args.raw
    else:
        cands = sorted(glob.glob(os.path.join(args.dir, "*_features.npz")))
        if not cands:
            raise SystemExit("目录下没找到 *_features.npz: %s" % args.dir)
        npz_path = cands[0]
        raw_path = npz_path.replace("_features.npz", "_raw.npy")
    d = np.load(npz_path, allow_pickle=True)
    raw = None
    if raw_path and os.path.exists(raw_path):
        raw = np.load(raw_path, mmap_mode="r")
    return d, raw, npz_path, raw_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="output")
    ap.add_argument("--npz", default=None)
    ap.add_argument("--raw", default=None)
    args = ap.parse_args()

    d, raw, npz_path, raw_path = load(args)
    X = d["X"]
    y_mod, y_osnr = d["y_mod"], d["y_osnr"]
    fnames = list(d["feature_names"])
    mods = C.MODULATIONS
    n, dim = X.shape
    try:
        meta = json.loads(str(d["meta"]))
    except Exception:
        meta = {}
    eq_mode = meta.get("equalizer", {}).get("mode", "CMA")
    fcfg = meta.get("feature_cfg", {})
    print("均衡器=%s  熵来源=%s  AH范围=%s"
          % (eq_mode, fcfg.get("entropy", "?"), fcfg.get("ah_range", "?")))
    print("=" * 64)
    print("数据集: %s" % npz_path)
    print("样本数=%d  特征维度=%d (应为 %d)" % (n, dim, F.N_FEATURES))
    print("原始符号文件: %s" % (raw_path if raw is not None else "(无)"))
    print("划分: train=%d test=%d val=%d"
          % (len(d["train_idx"]), len(d["test_idx"]), len(d["val_idx"])))
    assert dim == F.N_FEATURES, "特征维度不是 %d!" % F.N_FEATURES

    # ---- 每格式计数表 ----
    print("-" * 64)
    print("%-8s %-12s %-10s %-8s" % ("调制", "OSNR点数", "样本数", "OSNR范围(dB)"))
    for mi, m in enumerate(mods):
        sel = y_mod == mi
        if not sel.any():
            continue
        os_ = np.unique(y_osnr[sel])
        print("%-8s %-12d %-10d %g ~ %g"
              % (m, len(os_), sel.sum(), os_.min(), os_.max()))
    total_expect = len(np.unique(np.stack([y_mod, y_osnr], 1), axis=0))
    print("条件数=%d  总样本=%d" % (total_expect, n))
    if raw is not None:
        print("raw shape=%s dtype=%s" % (raw.shape, raw.dtype))

    # ---- 绘图(需要原始符号) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(未安装 matplotlib, 跳过绘图)")
        return

    if raw is None:
        print("(无原始符号文件, 跳过星座/AH 绘图)")
        return

    # Fig.2 式星座图: 每种格式取中间 OSNR 的第 1 个样本
    fig, axes = plt.subplots(1, len(mods), figsize=(3 * len(mods), 3.2))
    for mi, m in enumerate(mods):
        sel = np.where(y_mod == mi)[0]
        if not len(sel):
            continue
        os_mid = np.median(np.unique(y_osnr[sel]))
        row = sel[np.argmin(np.abs(y_osnr[sel] - os_mid))]
        x = np.asarray(raw[row])
        ax = axes[mi]
        ax.scatter(x.real, x.imag, s=1, alpha=0.3, c="tab:blue",
                   rasterized=True)
        ax.set_title("%s @ %.0f dB" % (m, y_osnr[row]))
        ax.set_aspect("equal")
        ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)
    fig.suptitle("Constellations after %s (cf. paper Fig.2)" % eq_mode)
    fig.tight_layout()
    p1 = os.path.join(args.dir, "check_constellation.png")
    fig.savefig(p1, dpi=140); plt.close(fig)

    # Fig.3 式 AH 曲线(横轴为固定全局幅度范围下的真实幅度值)
    bin_w = C.AH_RANGE_MAX / 40.0
    ax_amp = (np.arange(40) + 0.5) * bin_w
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for mi, m in enumerate(mods):
        sel = np.where(y_mod == mi)[0]
        if not len(sel):
            continue
        os_mid = np.median(np.unique(y_osnr[sel]))
        row = sel[np.argmin(np.abs(y_osnr[sel] - os_mid))]
        ax.plot(ax_amp, X[row][:40], label="%s @ %.0f dB" % (m, y_osnr[row]))
        # 诊断: 超过固定上界的样本占比(应为 0 或极小)
        clip = float(np.mean(np.abs(np.asarray(raw[row])) > C.AH_RANGE_MAX))
        if clip > 0:
            print("  [note] %s @ %.0fdB 超出 AH 上界(%.1f)的符号占比 %.4f%%"
                  % (m, y_osnr[row], C.AH_RANGE_MAX, 100 * clip))
    ax.set_xlabel("|x|  (fixed global range 0 ~ %.1f, 40 bins)" % C.AH_RANGE_MAX)
    ax.set_ylabel("probability")
    ax.set_title("Amplitude histograms after %s (cf. paper Fig.3)" % eq_mode)
    ax.legend(fontsize=7)
    fig.tight_layout()
    p2 = os.path.join(args.dir, "check_ah.png")
    fig.savefig(p2, dpi=140); plt.close(fig)

    # Fig.4 式箱线图: sigma_spread / h_norm 在论文 10 个条件下的分布
    fig, axes = plt.subplots(1, len(FIG4_FEATS), figsize=(4.5 * len(FIG4_FEATS), 4))
    labels = []
    have_any = False
    for feat in FIG4_FEATS:
        fi = fnames.index(feat)
        data, labels = [], []
        for (m, o) in FIG4_CONDS:
            mi = mods.index(m)
            sel = np.where((y_mod == mi) & (np.abs(y_osnr - o) < 0.5))[0]
            if len(sel):
                data.append(X[sel, fi])
                labels.append("%s\n%ddB" % (m, o))
        if data:
            have_any = True
            ax = axes[FIG4_FEATS.index(feat)]
            # matplotlib 3.9+ 将 labels 改名为 tick_labels, 兼容两者
            try:
                ax.boxplot(data, tick_labels=labels)
            except TypeError:
                ax.boxplot(data, labels=labels)
            ax.set_title("%s (cf. paper Fig.4)" % feat)
            ax.tick_params(axis="x", labelsize=6)
    if have_any:
        fig.tight_layout()
        p3 = os.path.join(args.dir, "check_boxplot.png")
        fig.savefig(p3, dpi=140)
        plt.close(fig)
        print("核验图已保存: %s, %s, %s" % (p1, p2, p3))
    else:
        plt.close(fig)
        print("核验图已保存: %s, %s (当前子集不含 Fig.4 条件, 跳过箱线图)" % (p1, p2))


if __name__ == "__main__":
    main()
