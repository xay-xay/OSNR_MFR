# -*- coding: utf-8 -*-
# =====================================================================
# model.py —— LA-MT-SEDNN（轻量级自适应多任务 SE 深度神经网络）
#
# 结构严格对照论文 2.2 节 / Fig.5:
#   输入 47 维归一化特征
#     -> 共享层: 64 全连接神经元 + BatchNorm + ELU
#     -> SE 注意力模块: 压缩 FC(64->32)+ReLU, 扩张 FC(32->64)+Sigmoid,
#        h_SE = s ⊙ h（压缩比 r = 2, 对应论文式(9)~(12)）
#     -> MFR 分支: 32 神经元 + BN + ELU + Dropout -> 7 输出(softmax)
#        OSNR 分支: 32 神经元 + BN + ELU（无 dropout）-> 1 输出(回归)
#
# 损失: L = λ1·CCE + λ2·MSE, 论文取 λ1 = 0.8, λ2 = 1.5
#       （OSNR 回归权重更大, 以在保持高 MFR 精度的同时提升回归精度）
#
# 复杂度(论文式(16)): W_total = d·h + 2(2h²/r + d) + h·h1 + h1·C_MFR
#                              + h·h1 + h1·C_OSNR
# =====================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEModule(nn.Module):
    """通道注意力(SE): 压缩 -> 激励 -> 重标定。"""

    def __init__(self, h=64, r=2):
        super(SEModule, self).__init__()
        self.fc1 = nn.Linear(h, h // r)     # 压缩 W1 ∈ R^{32×64}
        self.fc2 = nn.Linear(h // r, h)     # 扩张 W2 ∈ R^{64×32}

    def forward(self, h):
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(h))))   # 式(10)(11)
        return s * h                                        # 式(12) 哈达玛积


class LA_MT_SEDNN(nn.Module):
    def __init__(self, in_dim=47, h=64, h1=32, r=2, n_class=7, dropout=0.2,
                 use_se=True):
        super(LA_MT_SEDNN, self).__init__()
        self.use_se = use_se
        # ---- 共享特征层 ----
        self.shared = nn.Linear(in_dim, h)
        self.shared_bn = nn.BatchNorm1d(h)
        # ---- SE 注意力(use_se=False 退化为恒等映射, 用于消融对照) ----
        self.se = SEModule(h, r) if use_se else nn.Identity()
        # ---- MFR 分支(带 dropout) ----
        self.mfr_fc = nn.Linear(h, h1)
        self.mfr_bn = nn.BatchNorm1d(h1)#归一化稳定输出
        self.mfr_drop = nn.Dropout(dropout)#防止过拟合
        self.mfr_out = nn.Linear(h1, n_class)
        # ---- OSNR 分支(无 dropout) ----
        self.osnr_fc = nn.Linear(h, h1)
        self.osnr_bn = nn.BatchNorm1d(h1)
        self.osnr_out = nn.Linear(h1, 1)

    def forward(self, x):
        h = F.elu(self.shared_bn(self.shared(x)))
        h = self.se(h)
        # MFR
        hm = self.mfr_drop(F.elu(self.mfr_bn(self.mfr_fc(h))))
        logits = self.mfr_out(hm)
        # OSNR
        ho = F.elu(self.osnr_bn(self.osnr_fc(h)))#F.elu用在归一化之后
        osnr = self.osnr_out(ho).squeeze(-1)#.squeeze(-1)减少一维
        return logits, osnr


def multitask_loss(logits, osnr_pred, y_mod, y_osnr, lam1=0.8, lam2=1.5):
    """论文的联合损失: 分类交叉熵 + 回归均方误差。"""
    cce = F.cross_entropy(logits, y_mod)
    mse = F.mse_loss(osnr_pred, y_osnr)
    return lam1 * cce + lam2 * mse, float(cce.detach()), float(mse.detach())


def complexity_mults(d=47, h=64, h1=32, r=2, c_mfr=8, c_osnr=1, use_se=True):
    """论文式(16): 前向传播乘法次数(仅统计线性层)。
    use_se=False 时不计 SE 的两次线性变换。"""
    se = 2 * (2 * h * h / r + d) if use_se else 0.0
    return (d * h                       # 共享层
            + se                        # SE 两次线性变换
            + h * h1 + h1 * c_mfr       # MFR 分支
            + h * h1 + h1 * c_osnr)     # OSNR 分支


if __name__ == "__main__":
    net = LA_MT_SEDNN()
    x = torch.randn(4, 47)
    lg, os_ = net(x)
    print("logits", tuple(lg.shape), " osnr", tuple(os_.shape))
    print("参数量:", sum(p.numel() for p in net.parameters()))
    print("乘法复杂度(式16): %.3f M" % (complexity_mults() / 1e6))
