"""Teacher chỉ train trên dữ liệu gốc: thử cách xử lý chỗ lệch delay (gốc 56% delay = 0, cuộc thi 91%).

Biến thể (LightGBM, cùng tham số, chỉ train trên 129k dòng gốc, không thấy nhãn cuộc thi):
  full      : như teacher hiện có
  nodelay   : bỏ hai cột delay
  delay0    : chỉ dòng gốc có cả hai delay = 0
  reweight  : trọng số dòng gốc = p/(1-p), p = xác suất "giống cuộc thi" từ classifier gốc-vs-cuộc thi
              (cross-fit 5 fold trên dữ liệu gốc, chỉ dùng feature, không dùng nhãn)
Vì teacher không thấy nhãn cuộc thi, dự đoán của nó trên mọi dòng train là ngoài mẫu.
Đo: AUC trên train cuộc thi; thêm logit teacher cạnh logit stack công khai (LR nested 5 fold, meta seed 7/101/2026).
Chạy từ thư mục gốc repo: python notebooks/A_orig_teacher.py [--stack stack_lr_clean_public]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import CAT_COLS, DATA_DIR, FEATURES, OOF_DIR, TARGET
from data import load_data

T0 = time.time()
DELAYS = ["Departure Delay in Minutes", "Arrival Delay in Minutes"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=50, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=42, num_threads=12, verbose=-1)
N_ROUNDS = 600


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def prep(df):
    X = df[FEATURES].copy()
    for c in CAT_COLS:
        X[c] = X[c].astype("category")
    return X


def teacher(Xo, yo, Xq, cols, w=None):
    m = lgb.train(PARAMS, lgb.Dataset(Xo[cols], yo, weight=w), N_ROUNDS)
    return m.predict(Xq[cols])


def stack_gain(s, t, y):
    out = []
    for seed in (7, 101, 2026):
        base, new = np.zeros(len(y)), np.zeros(len(y))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(s, y):
            base[b] = s[b]   # stack một mình: chỉ đổi thang đo, AUC giữ nguyên
            X = np.column_stack([s, t])
            new[b] = LogisticRegression(C=1.0, max_iter=2000).fit(X[a], y[a]).decision_function(X[b])
        out.append(roc_auc_score(y, new) - roc_auc_score(y, base))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default="stack_lr_clean_public")
    args = ap.parse_args()
    train = load_data()[0]
    og = pd.read_csv(DATA_DIR / "original" / "data.csv")
    y, yo = train[TARGET].astype(int).to_numpy(), og[TARGET].astype(int).to_numpy()
    Xt, Xo = prep(train), prep(og)
    for c in CAT_COLS:   # cùng bảng category
        cats = sorted(set(Xt[c].cat.categories) | set(Xo[c].cat.categories))
        Xt[c], Xo[c] = Xt[c].cat.set_categories(cats), Xo[c].cat.set_categories(cats)
    s = logit(np.load(Path(OOF_DIR) / f"oof_{args.stack}.npy"))
    log(f"train {len(Xt):,}, gốc {len(Xo):,} | {args.stack} AUC {roc_auc_score(y, s):.6f}")

    # trọng số adversarial: classifier gốc (0) vs cuộc thi (1), cross-fit trên dòng gốc
    Xa = pd.concat([Xo, Xt.sample(len(Xo) * 2, random_state=0)], ignore_index=True)
    za = np.r_[np.zeros(len(Xo)), np.ones(len(Xo) * 2)]
    p_og = np.zeros(len(Xo))
    adv = dict(PARAMS, num_leaves=31)
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(Xa, za):
        m = lgb.train(adv, lgb.Dataset(Xa.iloc[a], za[a]), 300)
        bo = b[b < len(Xo)]
        p_og[bo] = m.predict(Xa.iloc[bo])
    w = p_og / (1 - p_og) / 2   # chia 2: tỉ lệ mẫu cuộc thi:gốc là 2:1
    w = np.clip(w / w.mean(), 0.05, 20)
    log(f"trọng số reweight: trung vị {np.median(w):.2f}, q01/q99 {np.quantile(w, .01):.2f}/{np.quantile(w, .99):.2f}, "
        f"cỡ mẫu hiệu dụng {w.sum() ** 2 / (w ** 2).sum():,.0f} / {len(w):,}")

    z0 = ((og[DELAYS[0]] == 0) & (og[DELAYS[1]] == 0)).to_numpy()
    nod = [c for c in FEATURES if c not in DELAYS]
    variants = {
        "full": teacher(Xo, yo, Xt, FEATURES),
        "nodelay": teacher(Xo, yo, Xt, nod),
        "delay0": teacher(Xo[z0], yo[z0], Xt, FEATURES),
        "reweight": teacher(Xo, yo, Xt, FEATURES, w),
    }
    print(f"\n{'biến thể':10s} {'AUC train':>10s} {'tương quan':>10s}   thêm vào stack (x 1e-6, seed 7/101/2026)")
    for k, p in variants.items():
        t = logit(p)
        g = stack_gain(s, t, y)
        print(f"{k:10s} {roc_auc_score(y, p):10.6f} {np.corrcoef(s, t)[0, 1]:10.4f}   "
              + " ".join(f"{x * 1e6:+6.1f}" for x in g))
    log("xong")
