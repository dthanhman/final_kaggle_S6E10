"""Việc 2 (21/10): thay "teacher" trong file đám đông bằng stack CV của mình có lợi trên OOF không?

File đám đông = h-blend các file dò LB (không OOF), mỗi file trộn w teacher (nina v10/M208: w 0.188 = T188 LR thường;
K206: w 0.30 = TB C176 + D183). Chỉ teacher có OOF -> đo trên OOF: AUC((1-w)X + w*S) - AUC((1-w)X + w*T)
với X là nhiều file đại diện (không có OOF của X thật), trong không gian percentile-rank, theo từng fold của folds.csv.
Lợi ích dương ổn định với mọi X -> có cơ sở CV để thay teacher.

Chạy: python notebooks/A_teacher_swap.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]


def pr(v):
    return (rankdata(np.asarray(v, dtype=np.float64).ravel()) - 0.5) / len(v)


def oof(rel):
    return pr(np.load(ROOT / rel))


y = pd.read_csv(ROOT / "data/train.csv", usecols=["satisfaction"])["satisfaction"].astype(int).values
fold = pd.read_csv(ROOT / "data/folds.csv")["fold"].values

TEACHERS = {  # ứng viên teacher (có OOF)
    "LRflat(~T188)": oof("oof/oof_stack_lr_clean_public.npy"),
    "fw_p247(~E188)": oof("oof/oof_stack_fw_p247.npy"),
    "S14": oof("oof/oof_stack_v14_both_tt_rm.npy"),
    "team_rm_noforge": oof("oof/oof_stack_team_realmlp_fw_noforge.npy"),
}
X_PROXY = {  # X đại diện cho các file đám đông: stack khác pool / khác loại, member mạnh khác loại
    "own181(mayank)": oof("data/public_subs_2010/own_cv_stack_181_oof.npy"),
    "team_tfm": oof("oof/oof_stack_team_tfm.npy"),
    "lr_all": oof("oof/oof_stack_lr_all.npy"),
}
# thêm vài member công khai mạnh, khác loại (TabPFN / RealMLP / CatBoost) nếu có
for name, rel in [("c124_best_cat", None)]:
    pass
c124 = sorted((ROOT / "data/public_oof_c124").glob("oof_*.npy"))
scored = []
for p in c124:
    v = np.load(p).ravel()
    if len(v) == len(y) and np.isfinite(v).all():
        scored.append((roc_auc_score(y, v), p))
for a, p in sorted(scored, reverse=True)[:3]:
    X_PROXY[p.stem] = pr(np.load(p))
# X "giống đám đông": trộn stack + member mạnh khác loại (đám đông lệch stack std dR ~0.02)
X_PROXY["mix(own181,team_tfm)"] = pr(X_PROXY["own181(mayank)"] + X_PROXY["team_tfm"])


def fold_auc(s):
    return np.array([roc_auc_score(y[fold == k], s[fold == k]) for k in range(5)])


print("AUC teacher:", {k: round(roc_auc_score(y, v), 6) for k, v in TEACHERS.items()})
print("AUC X:", {k: round(roc_auc_score(y, v), 6) for k, v in X_PROXY.items()})
base_t = "LRflat(~T188)"
for w in (0.188, 0.30):
    print(f"\n== w = {w}: chênh AUC so với teacher {base_t} (×1e-6), [fold tăng/5]")
    for xn, X in X_PROXY.items():
        ref = fold_auc((1 - w) * X + w * TEACHERS[base_t])
        cells = []
        for tn, T in TEACHERS.items():
            if tn == base_t:
                continue
            d = fold_auc((1 - w) * X + w * T) - ref
            cells.append(f"{tn} {d.mean() * 1e6:+6.1f} [{(d > 0).sum()}/5]")
        print(f"  X={xn:28s} " + " | ".join(cells))
