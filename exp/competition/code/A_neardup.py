"""Tìm tín hiệu giữa các dòng: dòng gần trùng (khớp đúng mọi cột trừ d cột) có nhãn giống nhau hơn mức stack dự đoán không?

Với mỗi cách bỏ d cột (d = 1: 21 cách, d = 2: 210 cách), băm 21 - d cột còn lại; dòng cùng mã băm là "gần trùng".
Feature cho dòng i ở fold k: số dòng gần trùng và tỉ lệ dương của chúng, CHỈ lấy dòng ở fold khác k (OOF sạch).
Đo: LR [logit stack, feature] fit trên 4 fold, chấm fold còn lại; so với stack đứng riêng (AUC theo fold).
Kèm: tỉ lệ dòng test có dòng gần trùng trong train (nếu test khác train nhiều, feature không chuyển được).

Chạy: python notebooks/A_neardup.py [--stack stack_team_realmlp_fw_noforge] [--d 1 2]
"""
import argparse
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
T0 = time.time()
ap = argparse.ArgumentParser()
ap.add_argument("--stack", default="stack_team_realmlp_fw_noforge")
ap.add_argument("--d", type=int, nargs="+", default=[1, 2])
args = ap.parse_args()

tr = pd.read_csv(ROOT / "data" / "train.csv")
te = pd.read_csv(ROOT / "data" / "test.csv")
y = tr.satisfaction.astype(int).to_numpy()
fold = pd.read_csv(ROOT / "data" / "folds.csv").set_index("id").fold.loc[tr.id].to_numpy()
FEATS = [c for c in tr.columns if c not in ("id", "satisfaction")]
NTR, K = len(tr), fold.max() + 1

# mã số nguyên cho từng cột trên train + test chung (NaN thành một giá trị riêng)
allx = pd.concat([tr[FEATS], te[FEATS]], ignore_index=True)
codes = np.column_stack([pd.factorize(allx[c].astype(str))[0] for c in FEATS]).astype(np.int64)
rng = np.random.default_rng(0)
mult = rng.integers(1, 2**61 - 1, size=len(FEATS), dtype=np.int64)
part = codes * mult                    # băm cộng: tổng các cột giữ lại (tràn số int64 vẫn là băm hợp lệ)
full = part.sum(1)

s = np.load(ROOT / "oof" / f"oof_{args.stack}.npy").astype(np.float64)
s = np.log(np.clip(s, 1e-7, 1 - 1e-7) / (1 - np.clip(s, 1e-7, 1 - 1e-7))) if s.min() >= 0 and s.max() <= 1 else s
base = np.array([roc_auc_score(y[fold == k], s[fold == k]) for k in range(K)])
print(f"stack {args.stack}: AUC theo fold {np.round(base, 6)}  TB {base.mean():.6f}")


def neighbor_stats(keys):
    """Số dòng train cùng mã và số dương, tách theo fold; trả về (cnt_oof, pos_oof) cho train và (cnt, pos) cho test."""
    ktr, kte = keys[:NTR], keys[NTR:]
    df = pd.DataFrame({"k": ktr, "f": fold, "y": y})
    g = df.groupby(["k", "f"]).y.agg(["size", "sum"]).reset_index()
    tot = g.groupby("k")[["size", "sum"]].sum()
    own = g.set_index(["k", "f"])
    idx = pd.MultiIndex.from_arrays([ktr, fold])
    c_all = tot["size"].reindex(ktr).to_numpy(); p_all = tot["sum"].reindex(ktr).to_numpy()
    c_own = own["size"].reindex(idx).to_numpy(); p_own = own["sum"].reindex(idx).to_numpy()
    c_te = tot["size"].reindex(kte).fillna(0).to_numpy(); p_te = tot["sum"].reindex(kte).fillna(0).to_numpy()
    return c_all - c_own, p_all - p_own, c_te, p_te


def evaluate(tag, cnt, pos):
    prior = y.mean()
    has = (cnt > 0).astype(float)
    rate = (pos + 2 * prior) / (cnt + 2)                       # làm mượt về tỉ lệ chung
    lr_rate = np.log(rate / (1 - rate)) - np.log(prior / (1 - prior))
    X = np.column_stack([s, lr_rate, has, np.log1p(cnt)])
    gain = []
    for k in range(K):
        a, b = fold != k, fold == k
        m = LogisticRegression(C=1.0, max_iter=2000, tol=1e-8).fit(X[a], y[a])
        gain.append(roc_auc_score(y[b], m.decision_function(X[b])) - base[k])
    gain = np.array(gain) * 1e6
    print(f"{tag:24s} có láng giềng {has.mean():6.1%} | gain x1e-6 theo fold {np.round(gain, 1)} TB {gain.mean():+.1f}"
          f" | {(gain > 0).sum()}/5 fold   [{time.time() - T0:.0f}s]", flush=True)
    return gain.mean()


for d in args.d:
    CNT, POS = np.zeros(NTR), np.zeros(NTR)
    cover_te = np.zeros(len(te), bool)
    best = []
    for drop in combinations(range(len(FEATS)), d):
        keys = full - part[:, list(drop)].sum(1)
        c, p, ct, pt = neighbor_stats(keys)
        CNT += c; POS += p; cover_te |= ct > 0
        if d == 1:
            best.append((evaluate(f"bỏ {FEATS[drop[0]][:20]}", c, p), FEATS[drop[0]]))
    print(f"\n== d = {d}: gộp mọi cách bỏ {d} cột | test có láng giềng trong train: {cover_te.mean():.1%}")
    evaluate(f"gộp d={d}", CNT, POS)
    if best:
        best.sort(reverse=True)
        print("tốt nhất:", [(n, round(g, 1)) for g, n in best[:5]])
