# -*- coding: utf-8 -*-
# =====================================================================
# evaluate.py —— LA-MT-SEDNN 推理与结果评估（仿论文展示）
#
# 对照论文展示内容:
#   Fig.6  BTB 链路上各格式的 OSNR 监测 MAE 随 OSNR 变化(evaluate 默认)
#   Fig.7  MFR 精度 RA 与 OSNR MAE 随 epoch 变化(BO 自适应 vs 固定学习率)
#   混淆矩阵 / 预测散点
#   Table 3 特征方案消融(AHs / AHs+熵 / AHs+统计 / 全 45 维)  --ablation
#   Table 6 计算复杂度(论文式(16))                     --complexity
#
# 用法:
#   python evaluate.py --npz ../dataset_gen/output/btb/btb_cma_features.npz \
#                      --ckpt results/best_model.pt --out-dir results
# =====================================================================
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data import load_dataset, feature_subset
from model import LA_MT_SEDNN, complexity_mults

MODS = ["QAM4_PCS", "QAM16_PCS", "QAM64_PCS", "QAM256_PCS"]


def _json_default(o):
    """把 numpy 标量/数组转成可 JSON 序列化的类型。"""
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def load_model(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = LA_MT_SEDNN(in_dim=ck["in_dim"], n_class=ck.get("n_class", len(MODS)),
                      dropout=ck["dropout"], use_se=ck.get("use_se", True))
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, ck


@torch.no_grad()
def infer(net, X):
    logits, pred = net(torch.from_numpy(X))
    return logits.argmax(1).numpy(), pred.numpy()


def summarize(y_true_mod, y_pred_mod, y_true_osnr, y_pred_osnr):
    acc = 100.0 * float((y_true_mod == y_pred_mod).mean())
    err = y_pred_osnr - y_true_osnr
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return dict(acc=acc, mae=mae, rmse=rmse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--ckpt", default="results/best_model.pt")
    ap.add_argument("--dir", default="results", help="训练输出目录(含 history.json)")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--history", default=None, help="history.json 路径")
    ap.add_argument("--ablation", action="store_true", help="跑 Table 3 特征消融")
    ap.add_argument("--ablation-schemes", default=None,
                    help="逗号分隔的消融方案子集, 如 'all45,all'; 默认跑全部")
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    d = load_dataset(args.npz)
    mods = d.mod_names if d.mod_names else MODS
    net, ck = load_model(args.ckpt)
    mu, sd = ck["norm_mean"], ck["norm_std"]
    use_se = ck.get("use_se", True)
    scheme = ck.get("scheme", "all")
    Xsel, _ = feature_subset(d.X, d.feature_names, scheme)
    if len(mu) != Xsel.shape[1]:
        raise SystemExit(
            "ckpt 归一化统计量为 %d 维, 与方案 '%s' 的 %d 维不一致"
            % (len(mu), scheme, Xsel.shape[1]))
    Xn = (Xsel - mu) / sd
    print("ckpt 特征方案 %s (%d 维) | SE=%s" % (scheme, Xsel.shape[1], use_se))

    idx_te = d.test_idx
    yp_mod, yp_osnr = infer(net, Xn[idx_te])
    res = summarize(d.y_mod[idx_te], yp_mod, d.y_osnr[idx_te], yp_osnr)

    print("=" * 66)
    print("测试集评估 (样本 %d)" % len(idx_te))
    print("  MFR 精度 RA      : %.2f %%" % res["acc"])
    print("  OSNR MAE         : %.4f dB" % res["mae"])
    print("  OSNR RMSE        : %.4f dB" % res["rmse"])
    print("-" * 66)
    print("%-8s %-10s %-12s %-10s" % ("调制", "样本数", "MFR精度(%)", "MAE(dB)"))
    per_mod = {}
    for mi, m in enumerate(mods):
        sel = idx_te[d.y_mod[idx_te] == mi]
        if not len(sel):
            continue
        pm, po = infer(net, Xn[sel])
        s = summarize(d.y_mod[sel], pm, d.y_osnr[sel], po)
        per_mod[m] = s
        print("%-8s %-10d %-12.2f %-10.4f" % (m, len(sel), s["acc"], s["mae"]))

    fig6 = None

    # ---------------- Fig.6: 各格式 MAE 随 OSNR 变化 ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(mods), figsize=(3.0 * len(mods), 3.2),
                                 sharey=True)
        fig6 = {}
        for mi, m in enumerate(mods):
            ax = axes[mi]
            xs, ys = [], []
            for o in sorted(set(d.y_osnr[idx_te])):
                sel = idx_te[(d.y_mod[idx_te] == mi) & (np.abs(d.y_osnr[idx_te] - o) < 0.5)]
                if not len(sel):
                    continue
                pm, po = infer(net, Xn[sel])
                xs.append(o)
                ys.append(float(np.mean(np.abs(po - d.y_osnr[sel]))))
            fig6[m] = dict(osnr=xs, mae=ys)
            ax.plot(xs, ys, "o-", ms=4)
            ax.set_title(m)
            ax.set_xlabel("OSNR (dB)")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("OSNR MAE (dB)")
        fig.suptitle("OSNR monitoring MAE in BTB link (cf. paper Fig.6)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "fig6_mae_vs_osnr.png"), dpi=140)
        plt.close(fig)
        print("已保存 fig6_mae_vs_osnr.png")

        # 混淆矩阵 + 预测散点
        cm = np.zeros((len(mods), len(mods)), dtype=int)
        for t, p in zip(d.y_mod[idx_te], yp_mod):
            cm[t, p] += 1
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
        im = axs[0].imshow(cm, cmap="Blues")
        for i in range(len(mods)):
            for j in range(len(mods)):
                if cm[i, j]:
                    axs[0].text(j, i, cm[i, j], ha="center", va="center",
                                fontsize=8,
                                color="white" if cm[i, j] > cm.max() / 2 else "black")
        axs[0].set_xticks(range(len(mods)), mods)
        axs[0].set_yticks(range(len(mods)), mods)
        axs[0].set_xlabel("predicted"); axs[0].set_ylabel("true")
        axs[0].set_title("Confusion matrix (test)")
        fig.colorbar(im, ax=axs[0], fraction=0.046)
        axs[1].scatter(d.y_osnr[idx_te], yp_osnr, s=4, alpha=0.3)
        lim = [d.y_osnr[idx_te].min() - 1, d.y_osnr[idx_te].max() + 1]
        axs[1].plot(lim, lim, "r--", lw=1)
        axs[1].set_xlabel("true OSNR (dB)"); axs[1].set_ylabel("estimated OSNR (dB)")
        axs[1].set_title("OSNR estimation (MAE=%.3f dB)" % res["mae"])
        axs[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "confusion_scatter.png"), dpi=140)
        plt.close(fig)
        print("已保存 confusion_scatter.png")

        # ---------------- Fig.7: RA/MAE 随 epoch 变化 ----------------
        hp = args.history or os.path.join(args.dir, "history.json")
        if os.path.exists(hp):
            with open(hp, encoding="utf-8") as f:
                hist = json.load(f)
            curves = hist["curves"]
            fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
            for k, v in curves.items():
                ep = [r["epoch"] for r in v]
                lab = "adaptive (BO)" if k.startswith("adaptive") else k.replace("fixed_lr_", "lr=")
                key_ra = "test_acc" if "test_acc" in v[0] else "val_acc"
                key_mae = "test_mae" if "test_mae" in v[0] else "val_mae"
                axs[0].plot(ep, [r[key_ra] for r in v], lw=1.2, label=lab)
                axs[1].plot(ep, [r[key_mae] for r in v], lw=1.2, label=lab)
            axs[0].set_xlabel("epoch"); axs[0].set_ylabel("MFR accuracy (%)")
            axs[0].set_title("(a) RA vs epoch"); axs[0].grid(alpha=0.3)
            axs[1].set_xlabel("epoch"); axs[1].set_ylabel("OSNR MAE (dB)")
            axs[1].set_title("(b) MAE vs epoch"); axs[1].set_yscale("log")
            axs[1].grid(alpha=0.3)
            axs[0].legend(fontsize=7); axs[1].legend(fontsize=7)
            fig.suptitle("Convergence under fixed and adaptive learning rates (cf. paper Fig.7)")
            fig.tight_layout()
            fig.savefig(os.path.join(args.out_dir, "fig7_convergence.png"), dpi=140)
            plt.close(fig)
            print("已保存 fig7_convergence.png")
    except ImportError:
        print("(未安装 matplotlib, 跳过绘图)")

    # ---------------- Table 3: 特征方案消融 ----------------
    table3 = None
    if args.ablation:
        from train import train_once
        # 消融统一用全特征数据集上、仅训练集统计的 mu/sd, 再按列截取,
        # 避免依赖 ckpt 里的维度(ckpt 可能是子集方案训练的)
        mu_all = d.X[d.train_idx].mean(axis=0)
        sd_all = d.X[d.train_idx].std(axis=0) + 1e-12
        ymtr, ymva = d.y_mod[d.train_idx], d.y_mod[d.val_idx]
        yotr, yova = d.y_osnr[d.train_idx], d.y_osnr[d.val_idx]
        table3 = {}
        scheme_list = [("ah", "AHs features"),
                       ("ah+ent", "AHs & entropy"),
                       ("ah+stat", "AHs & statistical"),
                       ("ent+stat", "Entropy & statistical"),
                       ("all", "Proposed (all 45)")]
        if args.ablation_schemes:
            keep = [s.strip() for s in args.ablation_schemes.split(",")]
            scheme_list = [(s, lb) for s, lb in scheme_list if s in keep]
        print("-" * 66)
        print("%-22s %-14s %-11s %-11s" % ("特征方案", "MFR精度(%)", "MAE(dB)", "RMSE(dB)"))
        for scheme, label in scheme_list:
            Xs, sel = feature_subset(d.X, d.feature_names, scheme)
            Xs = (Xs - mu_all[sel]) / sd_all[sel]   # 归一化统计量按同一列子集截取
            st, _, _, _ = train_once(Xs[d.train_idx], ymtr, yotr,
                                     Xs[d.val_idx], ymva, yova,
                                     ck["lr"], ck["dropout"],
                                     epochs=args.epochs, seed=0, use_se=use_se,
                                     n_class=len(mods))
            n2 = LA_MT_SEDNN(in_dim=Xs.shape[1], n_class=len(mods),
                             dropout=ck["dropout"], use_se=use_se)
            n2.load_state_dict(st); n2.eval()
            pm, po = infer(n2, Xs[d.test_idx])
            s = summarize(d.y_mod[d.test_idx], pm, d.y_osnr[d.test_idx], po)
            table3[scheme] = s
            print("%-22s %-14.2f %-11.4f %-11.4f"
                  % (label, s["acc"], s["mae"], s["rmse"]))
        print("论文 Table 3 参考: AHs 92.1/0.552/0.77 | AHs+ent 97.1/0.491/0.512 |"
              " AHs+stat 98.9/0.271/0.328 | Proposed 99.2/0.169/0.233")

    # ---------------- Table 6: 计算复杂度 ----------------
    print("-" * 66)
    d_in = ck["in_dim"]
    n_cls = len(mods)
    W = complexity_mults(d=d_in, c_mfr=n_cls, use_se=use_se)
    print("计算复杂度(论文式(16)): W_total = %.3f M 次乘法" % (W / 1e6))
    print("  共享层 d·h=%.4fM | SE 2(2h²/r+d)=%.4fM | MFR h·h1+h1·%d=%.4fM |"
          " OSNR h·h1+h1·1=%.4fM"
          % (d_in * 64 / 1e6,
             (2 * (2 * 64 * 64 / 2 + d_in) / 1e6) if use_se else 0.0,
             n_cls, (64 * 32 + 32 * n_cls) / 1e6, (64 * 32 + 32 * 1) / 1e6))

    with open(os.path.join(args.out_dir, "eval_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(dict(overall=res, per_mod=per_mod,
                       per_mod_curve=fig6,
                       table3=table3, complexity_mults=W),
                  f, ensure_ascii=False, indent=1, default=_json_default)
    print("评估结果已保存到 %s/eval_summary.json" % args.out_dir)


if __name__ == "__main__":
    main()
