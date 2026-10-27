# -*- coding: utf-8 -*-
# =====================================================================
# train.py —— LA-MT-SEDNN 训练与超参数搜索
#
# 对照论文 3 节训练设置:
#   * Adam 优化器, 损失 L = 0.8·CCE + 1.5·MSE
#   * 最大 300 epoch; 早停依据"OSNR 任务的验证损失", 连续 200 epoch 不改善
#     则终止
#   * Optuna 贝叶斯优化(BO) 30 次试验, 搜索空间:
#       学习率     log-uniform(5e-4, 1e-2)
#       dropout    uniform(0.1, 0.4)  (仅作用于 MFR 分支)
#   * 另跑固定学习率 {5e-4, 1e-3, 1e-2, 5e-2} 作为论文 Fig.7 的对照
#
# 用法:
#   python train.py --npz ../dataset_gen/output/btb/btb_cma_features.npz
#   python train.py ... --trials 30 --no-bo        # 跳过 BO, 用默认超参
#
# 输出 (--out-dir, 默认 results/):
#   best_model.pt     最优模型权重 + 归一化统计量 + 超参
#   history.json      各 epoch 的 RA / MAE / 损失曲线(BO 最优 + 各固定学习率)
#   bo_trials.json    BO 各次试验的超参与验证指标
# =====================================================================
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data import load_dataset, feature_subset
from model import LA_MT_SEDNN, multitask_loss

torch.set_num_threads(max(1, os.cpu_count() - 1))


def make_loader(X, ym, yo, batch_size, shuffle, seed=0):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(ym),
                       torch.from_numpy(yo))
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g)


@torch.no_grad()
def evaluate(net, loader):
    """返回 acc(%), mae(dB), rmse(dB), osnr_mse, cce。"""
    net.eval()
    n = 0
    correct = 0
    ae = 0.0
    se = 0.0
    mse_sum = 0.0
    for xb, mb, ob in loader:
        logits, pred = net(xb)
        correct += int((logits.argmax(1) == mb).sum())
        d = (pred - ob)
        ae += float(d.abs().sum())
        se += float((d ** 2).sum())
        mse_sum += float((d ** 2).sum())
        n += len(mb)
    return dict(acc=100.0 * correct / n, mae=ae / n, rmse=np.sqrt(se / n),
                osnr_mse=mse_sum / n)


def train_once(Xtr, ymtr, yotr, Xva, ymva, yova, lr, dropout,
               epochs=300, patience=200, batch_size=128, seed=0,
               verbose=False, Xte=None, ymte=None, yote=None, use_se=True,
               n_class=7):
    """训练一次, 返回 (最优权重, 训练历史, 最优验证 MAE)。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = LA_MT_SEDNN(in_dim=Xtr.shape[1], n_class=n_class,
                      dropout=dropout, use_se=use_se)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    tr = make_loader(Xtr, ymtr, yotr, batch_size, True, seed)
    va = make_loader(Xva, ymva, yova, 512, False)
    te = (make_loader(Xte, ymte, yote, 512, False)
          if Xte is not None else None)

    best = dict(mae=1e9, epoch=-1, state=None)
    history = []
    stale = 0
    for ep in range(1, epochs + 1):
        net.train()
        for xb, mb, ob in tr:
            opt.zero_grad()
            logits, pred = net(xb)
            loss, _, _ = multitask_loss(logits, pred, mb, ob)
            loss.backward()
            opt.step()
        vm = evaluate(net, va)
        rec = dict(epoch=ep, val_acc=vm["acc"], val_mae=vm["mae"],
                   val_rmse=vm["rmse"], val_osnr_mse=vm["osnr_mse"])
        if te is not None:
            tm = evaluate(net, te)
            rec.update(test_acc=tm["acc"], test_mae=tm["mae"],
                       test_rmse=tm["rmse"])
        history.append(rec)
        # 早停: 依据 OSNR 任务(回归 MSE)的验证损失, 论文口径
        if vm["osnr_mse"] < best["mae"] - 1e-9:
            best = dict(mae=vm["osnr_mse"], epoch=ep,
                        state={k: v.clone() for k, v in net.state_dict().items()})
            stale = 0
        else:
            stale += 1
        if verbose and ep % 25 == 0:
            print("    ep %3d  val_RA=%.2f%%  val_MAE=%.4f dB"
                  % (ep, vm["acc"], vm["mae"]))
        if stale >= patience:
            if verbose:
                print("    早停于 epoch %d (连续 %d 轮验证损失无改善)"
                      % (ep, patience))
            break

    net.load_state_dict(best["state"])
    final = evaluate(net, va)
    return best["state"], history, final["mae"], best["epoch"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="dataset_gen 产出的特征 npz")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--trials", type=int, default=30, help="BO 试验次数")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-bo", action="store_true", help="跳过 BO, 用默认超参")
    ap.add_argument("--no-fixed-lr", action="store_true",
                    help="跳过固定学习率对照(论文 Fig.7)")
    ap.add_argument("--scheme", default="all",
                    choices=["all", "ah", "ah+ent", "ah+stat",
                             "ent+stat", "cum", "all45"],
                    help="特征方案: all=47维 / all45=45维(去累积量) / ah=40维 / "
                         "ent+stat=5维(去AH) / cum=2维累积量")
    ap.add_argument("--no-se", action="store_true",
                    help="去掉 SE 注意力模块(消融对照)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    d = load_dataset(args.npz)
    use_se = not args.no_se
    n_class = len(d.mod_names)
    Xsel, _ = feature_subset(d.X, d.feature_names, args.scheme)
    mu = Xsel[d.train_idx].mean(axis=0)
    sd = Xsel[d.train_idx].std(axis=0) + 1e-12
    Xn = (Xsel - mu) / sd
    np.savez(os.path.join(args.out_dir, "norm_stats.npz"), mean=mu, std=sd)

    Xtr, Xva, Xte = d.split(Xn)
    ymtr, ymva, ymte = d.split(d.y_mod)
    yotr, yova, yote = d.split(d.y_osnr)
    print("特征方案 %s (%d 维) | 类别 %d | SE=%s | 训练 %d / 验证 %d / 测试 %d"
          % (args.scheme, Xtr.shape[1], n_class, use_se,
             len(Xtr), len(Xva), len(Xte)))

    # ---------------- Optuna BO 超参搜索 ----------------
    bo_records = []
    if not args.no_bo:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            lr = trial.suggest_float("lr", 5e-4, 1e-2, log=True)
            dr = trial.suggest_float("dropout", 0.1, 0.4)
            _, hist, vmae, bep = train_once(
                Xtr, ymtr, yotr, Xva, ymva, yova, lr, dr,
                epochs=args.epochs, patience=args.patience,
                batch_size=args.batch_size, seed=args.seed, use_se=use_se,
                n_class=n_class)
            trial.set_user_attr("best_epoch", bep)
            trial.set_user_attr("val_mae", vmae)
            bo_records.append(dict(trial=trial.number, lr=lr, dropout=dr,
                                   val_mae=vmae, best_epoch=bep))
            print("  [BO %2d] lr=%.2e dropout=%.2f -> val MAE=%.4f dB (ep %d)"
                  % (trial.number, lr, dr, vmae, bep))
            return vmae

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=args.seed))
        study.optimize(objective, n_trials=args.trials)
        best_lr = study.best_params["lr"]
        best_dr = study.best_params["dropout"]
        print("BO 最优: lr=%.3e  dropout=%.3f  验证 MAE=%.4f dB"
              % (best_lr, best_dr, study.best_value))
        with open(os.path.join(args.out_dir, "bo_trials.json"), "w",
                  encoding="utf-8") as f:
            json.dump(bo_records, f, ensure_ascii=False, indent=1)
    else:
        best_lr, best_dr = 1e-3, 0.2
        print("跳过 BO, 使用默认超参 lr=%.1e dropout=%.2f" % (best_lr, best_dr))

    # ---------------- 用最优超参训练最终模型 ----------------
    print("训练最终模型 ...")
    state, history, vmae, bep = train_once(
        Xtr, ymtr, yotr, Xva, ymva, yova, best_lr, best_dr,
        epochs=args.epochs, patience=args.patience,
        batch_size=args.batch_size, seed=args.seed, verbose=True,
        Xte=Xte, ymte=ymte, yote=yote, use_se=use_se, n_class=n_class)
    torch.save(dict(state_dict=state, lr=best_lr, dropout=best_dr,
                    in_dim=Xtr.shape[1], n_class=n_class,
                    norm_mean=mu, norm_std=sd,
                    use_se=use_se, scheme=args.scheme),
               os.path.join(args.out_dir, "best_model.pt"))
    hist = {"adaptive(BO)": history}
    print("最终模型: 最优 epoch=%d, 验证 MAE=%.4f dB" % (bep, vmae))

    # ---------------- 固定学习率对照(论文 Fig.7) ----------------
    if not args.no_fixed_lr:
        for lr in (5e-4, 1e-3, 1e-2, 5e-2):
            print("对照: 固定学习率 %.1e ..." % lr)
            _, h, vm, _ = train_once(
                Xtr, ymtr, yotr, Xva, ymva, yova, lr, best_dr,
                epochs=args.epochs, patience=args.patience,
                batch_size=args.batch_size, seed=args.seed,
                Xte=Xte, ymte=ymte, yote=yote, use_se=use_se,
                n_class=n_class)
            hist["fixed_lr_%g" % lr] = h

    with open(os.path.join(args.out_dir, "history.json"), "w",
              encoding="utf-8") as f:
        json.dump(dict(best=dict(lr=best_lr, dropout=best_dr,
                                 best_epoch=bep, val_mae=vmae),
                       curves=hist), f, ensure_ascii=False)
    print("完成: 权重与曲线已保存到 %s" % args.out_dir)


if __name__ == "__main__":
    main()
