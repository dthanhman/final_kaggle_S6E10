"""Tìm cách đưa team vào stack P247 (nền stack_v10_drop_forge) mà không làm giảm CV.

Biến thể (LR trên logit chuẩn hoá, nested meta seed 42/7/101, fold-mean AUC, so với LR chỉ trên nền):
  mean15   : nền + TB logit chuẩn hoá của 15 model sạch của team (không fit trên nhãn -> 1 cột sạch)
  mean_nt  : nền + TB của các model không phải cây (lr, knn, fm, ftt, glr, tabm, gptc, kozy, realmlp)
  all_C*   : nền + cả 15 cột, C nhỏ (phạt mạnh)
  realmlp  : nền + B_realmlp_v1
Biến thể tốt nhất được lưu thành stack_team_<tên> (test = TB 15 fold-model của 3 seed).

Chạy: python notebooks/A_team_into_stack.py [--base stack_fw_noforge] [--only realmlp]
(nền khác stack_v10_drop_forge -> tên lưu stack_team_<biến thể>_<nền bỏ "stack_">)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils import save_preds  # noqa: E402

OOF, DATA = ROOT / "oof", ROOT / "data"
ap = argparse.ArgumentParser()
ap.add_argument("--base", default="stack_v10_drop_forge")
ap.add_argument("--only", nargs="*", help="chỉ chạy các biến thể này (luôn có base)")
args = ap.parse_args()
BASE = args.base
TREES = ["A_lgb_v2", "A_lgb_origrow_v1", "quan_xgb_safe_teacher_v4", "quan_xgb_dual_teacher_v4",
         "quan_cat_dual_teacher_v4", "quan_xgb_original_aug_v4"]
NONTREE = ["A_lr_v2", "A_knn_v2", "A_fm_v1", "A_ftt_v1", "A_glr_v1", "A_tabm_v1", "A_gptc_v1", "A_kozy_v1", "B_realmlp_v1"]
SEEDS = [42, 7, 101]

y = pd.read_csv(DATA / "train.csv", usecols=["satisfaction"]).satisfaction.astype(int).to_numpy()
folds = pd.read_csv(DATA / "folds.csv")["fold"].to_numpy()


def logit(v):
    v = np.asarray(v, dtype=np.float64)
    if v.min() < 0 or v.max() > 1:
        return v
    v = np.clip(v, 1e-6, 1 - 1e-6)
    return np.log(v / (1 - v))


def load(n):
    o, t = logit(np.load(OOF / f"oof_{n}.npy")), logit(np.load(OOF / f"test_{n}.npy"))
    mu, sd = o.mean(), o.std()
    return (o - mu) / sd, (t - mu) / sd


B = load(BASE)
M = {n: load(n) for n in TREES + NONTREE}
avg = lambda ns: (np.mean([M[n][0] for n in ns], 0), np.mean([M[n][1] for n in ns], 0))
cols = lambda *cs: (np.column_stack([c[0] for c in cs]), np.column_stack([c[1] for c in cs]))
ALL = [M[n] for n in TREES + NONTREE]
VARIANTS = {"base": (cols(B), 1.0), "mean15": (cols(B, avg(TREES + NONTREE)), 1.0),
            "mean_nt": (cols(B, avg(NONTREE)), 1.0), "realmlp": (cols(B, M["B_realmlp_v1"]), 1.0),
            "all_C1e-3": (cols(B, *ALL), 1e-3), "all_C1e-5": (cols(B, *ALL), 1e-5)}

if args.only:
    VARIANTS = {v: VARIANTS[v] for v in ["base", *args.only]}


def meta_folds(seed):
    if seed == 42:
        return folds
    f = np.zeros(len(y), int)
    for k, (_, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(y, y)):
        f[b] = k
    return f


def nested(X, XT, mf, c):
    o, t = np.zeros(len(y)), np.zeros(len(XT))
    for k in range(5):
        a, b = mf != k, mf == k
        m = LogisticRegression(C=c, tol=1e-7, max_iter=5000).fit(X[a], y[a])
        o[b] = m.decision_function(X[b]); t += m.decision_function(XT) / 5
    return o, t


MF = {s: meta_folds(s) for s in SEEDS}
fa = lambda o, mf: np.array([roc_auc_score(y[mf == k], o[mf == k]) for k in range(5)])
R, F = {}, {}
for v, ((X, XT), c) in VARIANTS.items():
    for s in SEEDS:
        R[v, s] = nested(X, XT, MF[s], c)
        F[v, s] = fa(R[v, s][0], MF[s])
    if v == "base":
        print(f"{v:10s} " + " / ".join(f"{F[v, s].mean():.6f}" for s in SEEDS), flush=True)
        continue
    d = {s: F[v, s] - F["base", s] for s in SEEDS}
    m = LogisticRegression(C=c, tol=1e-7, max_iter=5000).fit(X, y)
    print(f"{v:10s} " + " / ".join(f"{F[v, s].mean():.6f}" for s in SEEDS) + " | chênh "
          + "/".join(f"{d[s].mean() * 1e6:+.1f}" for s in SEEDS) + f"e-6, {sum((d[s] > 0).sum() for s in SEEDS)}/15 fold"
          + f" | w team/nền {np.abs(m.coef_[0][1:]).sum() / abs(m.coef_[0][0]):.3f}", flush=True)

gain = {v: np.mean([(F[v, s] - F["base", s]).mean() for s in SEEDS]) for v in VARIANTS if v != "base"}
best = max(gain, key=gain.get)
print(f"\nbiến thể team tốt nhất: {best} ({gain[best] * 1e6:+.1f}e-6 TB 3 seed)")
sig = lambda v: 1 / (1 + np.exp(-v))
suffix = "" if BASE == "stack_v10_drop_forge" else "_" + BASE.removeprefix("stack_")
save_preds(f"stack_team_{best.replace('-', '')}{suffix}", sig(R[best, 42][0]), sig(np.mean([R[best, s][1] for s in SEEDS], 0)), y)
