"""Bước 4 thí nghiệm LLR: feature LLR của bộ sinh (kernel cs116-llr-gen, notebooks/A_llr_gen.py) có thêm thông tin không.

1. AUC riêng của LLR, tương quan rank với stack nền.
2. Sàng lọc trên stack: LR [logit nền, LLR, ...] trên folds.csv, meta seed 42/7/101, so với nền (như A_probe_new_public.py).
3. Model đơn: LightGBM trên 21 cột gốc có / không có LLR (fold 0, early stopping trên 10% tách từ phần train).
LLR không dùng nhãn cuộc thi nên dùng thẳng làm cột được, OOF vẫn sạch.
Chạy: python notebooks/A_llr_eval.py <thư mục output kernel> [--base stack_fw_p247]
"""
import argparse
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from blend import to_logit  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--base", default="stack_fw_p247")
args = ap.parse_args()

train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
y = train["satisfaction"].astype(int).to_numpy()
folds = pd.read_csv(os.path.join(ROOT, "data", "folds.csv"))
assert (folds["id"].to_numpy() == train["id"].to_numpy()).all()
fold42 = folds["fold"].to_numpy()
SEEDS = [42, 7, 101]


def meta_folds(seed):
    if seed == 42:
        return fold42
    f = np.zeros(len(y), int)
    for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(y, y)):
        f[va] = k
    return f


FOLDS = {s: meta_folds(s) for s in SEEDS}
L = lambda n: np.load(os.path.join(args.out, n)).astype(np.float64)
llr, llr_k = L("llr_train.npy"), L("llr_orders_train.npy")
lpt, lpf = L("lp_true_train.npy"), L("lp_false_train.npy")
assert len(llr) == len(y)
bo = np.load(os.path.join(ROOT, "oof", f"oof_{args.base}.npy")).astype(np.float64)
B = to_logit(bo, bo)[0].astype(np.float64)
print(f"nền {args.base}: AUC {roc_auc_score(y, B):.6f}")
print(f"LLR: AUC {roc_auc_score(y, llr):.6f} | từng thứ tự " + " ".join(f"{roc_auc_score(y, llr_k[:, k]):.6f}" for k in range(llr_k.shape[1]))
      + f" | tương quan rank với nền {np.corrcoef(rankdata(B), rankdata(llr))[0, 1]:.4f}")
print(f"log p(x|True) AUC {roc_auc_score(y, lpt):.6f}, log p(x|False) AUC {roc_auc_score(y, -lpf):.6f}")


def lr_cv(X, f, C):
    o = np.zeros(len(y))
    for k in range(5):
        tr, va = f != k, f == k
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=C, tol=1e-7, max_iter=5000).fit(sc.transform(X[tr]), y[tr])
        o[va] = m.decision_function(sc.transform(X[va]))
    return o


def gain(cols, C=1.0):
    X = np.column_stack([B] + cols)
    res = []
    for s in SEEDS:
        f = FOLDS[s]
        o, o0 = lr_cv(X, f, C), lr_cv(B[:, None], f, C)
        fa = lambda v: np.array([roc_auc_score(y[f == k], v[f == k]) for k in range(5)])
        d = fa(o) - fa(o0)
        res.append(f"seed {s}: {d.mean() * 1e6:+6.1f}e-6 ({(d > 0).sum()}/5 fold)")
    return " | ".join(res)


print("\nThêm vào stack nền (chênh AUC fold-mean so với nền):")
for name, cols in [("LLR", [llr]), ("LLR từng thứ tự", list(llr_k.T)), ("LLR + lp_true + lp_false", [llr, lpt, lpf])]:
    print(f"  {name:26s} {gain(cols)}")

# model đơn: LightGBM trên cột gốc, có/không LLR, fold 0
X = train.drop(columns=["id", "satisfaction"]).copy()
for c in X.columns[X.dtypes == object]:
    X[c] = X[c].astype("category")
P = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=8, seed=0)
tr_idx, va_idx = np.where(fold42 != 0)[0], np.where(fold42 == 0)[0]
r = np.random.default_rng(0).permutation(tr_idx); cut = int(0.9 * len(r))
print("\nLightGBM model đơn, fold 0:")
for name, XX in [("cột gốc", X), ("cột gốc + LLR", X.assign(llr=llr, lpt=lpt, lpf=lpf))]:
    m = lgb.train(P, lgb.Dataset(XX.iloc[r[:cut]], y[r[:cut]]), 5000,
                  valid_sets=[lgb.Dataset(XX.iloc[r[cut:]], y[r[cut:]])], callbacks=[lgb.early_stopping(200, verbose=False)])
    print(f"  {name:16s} AUC {roc_auc_score(y[va_idx], m.predict(XX.iloc[va_idx], num_iteration=m.best_iteration)):.6f} "
          f"({m.best_iteration} cây)")
