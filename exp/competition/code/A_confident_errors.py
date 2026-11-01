"""Dòng stack đoán sai với độ tự tin cao: chỉ là nhiễu nhãn ngẫu nhiên, hay có dấu vết mà feature của dòng nhận ra được?

1. Mô tả: trong vùng tự tin (p > 0.9 hoặc p < 0.1), so dòng sai với dòng đúng theo vài dấu vết của bộ sinh dữ liệu
   (delay > 0, thiếu Arrival Delay, số chữ số Flight Distance, điểm toàn bằng nhau, có điểm 0, Age < 18 ...).
2. Đo: LightGBM CHỈ trên vùng tự tin, init_score = logit stack, học từ 21 cột gốc (theo folds.csv, dừng sớm trên 10%
   tách từ phần train). So AUC trong vùng (và AUC toàn bộ khi thay dự đoán trong vùng) với stack đứng riêng.
   A_residual.py đã đo việc này trên toàn bộ dòng (≤ 0); ở đây tín hiệu nhỏ chỉ có trong vùng tự tin không bị loãng.

Chạy: python notebooks/A_confident_errors.py [--stack stack_team_realmlp_fw_noforge] [--cut 0.9]
"""
import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
T0 = time.time()
ap = argparse.ArgumentParser()
ap.add_argument("--stack", default="stack_team_realmlp_fw_noforge")
ap.add_argument("--cut", type=float, default=0.9)
args = ap.parse_args()

tr = pd.read_csv(ROOT / "data" / "train.csv")
y = tr.satisfaction.astype(int).to_numpy()
fold = pd.read_csv(ROOT / "data" / "folds.csv").set_index("id").fold.loc[tr.id].to_numpy()
FEATS = [c for c in tr.columns if c not in ("id", "satisfaction")]
CATS = ["Gender", "Customer Type", "Type of Travel", "Class"]
RATINGS = [c for c in FEATS if c not in CATS + ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]]
p = np.clip(np.load(ROOT / "oof" / f"oof_{args.stack}.npy").astype(np.float64), 1e-7, 1 - 1e-7)
s = np.log(p / (1 - p))

hi, lo = p > args.cut, p < 1 - args.cut
err = (hi & (y == 0)) | (lo & (y == 1))
conf = hi | lo
print(f"vùng tự tin: {conf.mean():.1%} số dòng; tỉ lệ sai trong vùng {err[conf].mean():.2%} "
      f"(p > {args.cut}: {1 - y[hi].mean():.2%} sai, p < {1 - args.cut:.1f}: {y[lo].mean():.2%} sai)")

R = tr[RATINGS].to_numpy()
traits = {
    "dep delay > 0": tr["Departure Delay in Minutes"].to_numpy() > 0,
    "arr delay thiếu": tr["Arrival Delay in Minutes"].isna().to_numpy(),
    "arr != dep delay": (tr["Arrival Delay in Minutes"].fillna(-1) != tr["Departure Delay in Minutes"]).to_numpy(),
    "FD < 1000": tr["Flight Distance"].to_numpy() < 1000,
    "điểm toàn bằng nhau": (R == R[:, :1]).all(1),
    "có điểm 0": (R == 0).any(1),
    "Age < 18": tr["Age"].to_numpy() < 18,
    "Age >= 60": tr["Age"].to_numpy() >= 60,
    "disloyal": (tr["Customer Type"] == "disloyal Customer").to_numpy(),
}
print(f"\n{'dấu vết':22s} {'trong dòng sai':>15s} {'trong dòng đúng':>16s} {'tỉ lệ':>7s}")
for k, v in traits.items():
    a, b = v[err].mean(), v[conf & ~err].mean()
    print(f"{k:22s} {a:15.3%} {b:16.3%} {a / max(b, 1e-9):7.2f}")

# ---- LightGBM chỉ trên vùng tự tin, bắt đầu từ logit stack
X = tr[FEATS].copy()
for c in CATS:
    X[c] = X[c].astype("category")
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=4, seed=0)
new = s.copy()
rows = []
for k in range(fold.max() + 1):
    a = np.where(conf & (fold != k))[0]
    b = conf & (fold == k)
    ai = np.random.default_rng(k).permutation(a); cut = int(len(ai) * 0.9)
    f_, e_ = ai[:cut], ai[cut:]
    m = lgb.train(PARAMS, lgb.Dataset(X.iloc[f_], y[f_], init_score=s[f_]), 3000,
                  valid_sets=[lgb.Dataset(X.iloc[e_], y[e_], init_score=s[e_])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    new[b] = s[b] + m.predict(X[b], raw_score=True, num_iteration=m.best_iteration)
    fk = fold == k
    rows.append(dict(fold=k, iters=m.best_iteration,
                     d_conf=(roc_auc_score(y[b], new[b]) - roc_auc_score(y[b], s[b])) * 1e6,
                     d_all=(roc_auc_score(y[fk], new[fk]) - roc_auc_score(y[fk], s[fk])) * 1e6))
    print(f"fold {k}: {rows[-1]}   [{time.time() - T0:.0f}s]", flush=True)
T = pd.DataFrame(rows)
print(f"\nTB chênh trong vùng tự tin {T.d_conf.mean():+.1f}e-6 ({(T.d_conf > 0).sum()}/5 fold); "
      f"toàn bộ {T.d_all.mean():+.1f}e-6 ({(T.d_all > 0).sum()}/5 fold)")
