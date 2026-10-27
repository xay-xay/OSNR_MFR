# -*- coding: utf-8 -*-
# =====================================================================
# data.py —— 数据集加载与预处理
#
# 读入 dataset_gen 生成的特征数据集 (X, 标签, 70/20/10 划分索引),
# 做 z-score 归一化。
#
# 注意: 划分索引由生成脚本写入(按 (调制, OSNR) 条件分层随机, 固定种子);
#       z-score 统计量这里**只用训练集**计算(比生成脚本里用全数据集更严格),
#       避免验证/测试集信息泄漏。
# =====================================================================
import numpy as np

MODULATIONS = ["QAM8", "QAM16", "QAM32", "QAM64",
               "BPSK", "QPSK", "8PSK"]


class FeatureData(object):
    """特征数据集容器。"""

    def __init__(self, X, y_mod, y_osnr, train_idx, val_idx, test_idx,
                 mod_names, feature_names, meta=None):
        self.X = X                      # (N, 47) float32 原始特征
        self.y_mod = y_mod              # (N,) int64  调制类别
        self.y_osnr = y_osnr            # (N,) float32 OSNR(dB)
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.mod_names = mod_names
        self.feature_names = feature_names
        self.meta = meta or {}

    def normalized(self):
        """按训练集统计量做 z-score, 返回 (X_norm, mean, std)。"""
        mu = self.X[self.train_idx].mean(axis=0)
        sd = self.X[self.train_idx].std(axis=0) + 1e-12
        return (self.X - mu) / sd, mu, sd

    def split(self, X):
        return X[self.train_idx], X[self.val_idx], X[self.test_idx]


def load_dataset(npz_path, verbose=True):
    """加载 dataset_gen 产出的 *_features.npz。"""
    import json
    d = np.load(npz_path, allow_pickle=True)
    try:
        meta = json.loads(str(d["meta"]))
    except Exception:
        meta = {}
    demo = FeatureData(
        X=d["X"].astype(np.float32),
        y_mod=d["y_mod"].astype(np.int64),
        y_osnr=d["y_osnr"].astype(np.float32),
        train_idx=d["train_idx"], val_idx=d["val_idx"], test_idx=d["test_idx"],
        mod_names=list(meta.get("mods", MODULATIONS)),
        feature_names=[str(s) for s in d["feature_names"]],
        meta=meta,
    )
    if verbose:
        n = len(demo.X)
        print("数据集 %s: 样本 %d, 维度 %d, 类别 %d"
              % (npz_path, n, demo.X.shape[1], len(demo.mod_names)))
        print("划分: 训练 %d / 验证 %d / 测试 %d"
              % (len(demo.train_idx), len(demo.val_idx), len(demo.test_idx)))
    return demo


def feature_subset(X, feature_names, scheme):
    """按特征方案取子集, 对应论文 Table 3 的消融:
        "ah"      : 仅 40 维幅度直方图
        "ah+ent"  : AH + 2 个熵
        "ah+stat" : AH + 3 个统计量
        "ent+stat": 去掉 AH, 仅 3 统计量 + 2 熵 = 5 维
        "cum"     : 仅 2 维四阶累积量 |C40| / C42
        "all45"   : AH + 统计量 + 熵 = 45 维(去掉累积量, 用于 A/B 对照)
        "all"     : 47 维全特征(论文方法 + 累积量)
    """
    idx_ah = [i for i, n in enumerate(feature_names) if n.startswith("ah_")]
    idx_ent = [i for i, n in enumerate(feature_names) if n in ("h_log", "h_norm")]
    idx_stat = [i for i, n in enumerate(feature_names)
                if n in ("dms", "sigma_spread", "sigma_vel")]
    idx_cum = [i for i, n in enumerate(feature_names)
               if n in ("c40_abs", "c42")]
    if scheme == "ah":
        sel = idx_ah
    elif scheme == "ah+ent":
        sel = idx_ah + idx_ent
    elif scheme == "ah+stat":
        sel = idx_ah + idx_stat
    elif scheme == "ent+stat":
        sel = idx_stat + idx_ent
    elif scheme == "cum":
        sel = idx_cum
    elif scheme == "all45":
        sel = idx_ah + idx_stat + idx_ent
    elif scheme == "all":
        sel = idx_ah + idx_stat + idx_ent + idx_cum
    else:
        raise ValueError("未知特征方案: %s" % scheme)
    return X[:, sel], sel
