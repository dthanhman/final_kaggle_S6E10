"""Nhiễu public/private của khoảng cách giữa M213 (cụm 0,96213), N210 (Nina), slot 2 mix50 và slot 1 (24/10).

Nhãn giả ~ Bernoulli(p), p = isotonic(OOF S14) áp vào thứ tự của file "sự thật" (N210 hoặc M213);
chia ngẫu nhiên public 59.969 / private. Chỉ đọc sd (độ lớn nhiễu), không đọc trung bình (thiên về file làm sự thật).
Chạy: python notebooks/A_slot2_noise.py [--sims 300]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
N_PUB = 59_969
ids = pd.read_csv(D / "test.csv", usecols=["id"]).id.to_numpy()


def load(path):
    d = pd.read_csv(path)
    col = [c for c in d.columns if c != "id"][0]
    return d.set_index("id")[col].loc[ids].to_numpy(np.float64)


def auc_masked(order, y, m):
    lab, mm = y[order], m[order]
    neg = np.cumsum(mm & ~lab)
    pos = mm & lab
    return neg[pos].sum(dtype=np.float64) / (float(pos.sum()) * float((mm & ~lab).sum()))


ap = argparse.ArgumentParser()
ap.add_argument("--sims", type=int, default=300)
a = ap.parse_args()
F = {"M213": load(D / "public_subs_2410/mehrankazeminia/3-airline-s6e10-ensembling/submission.csv"),
     "N210": load(D / "public_subs_2210/nina2025/submission.csv"),
     "mix50": load(ROOT / "submissions/sub_slot2_mix50_nina210.csv"),
     "slot1": load(ROOT / "submissions/sub_slot1_stack80_kirill20.csv")}
N = len(ids)
R = {k: rankdata(v) / N for k, v in F.items()}
y_tr = pd.read_csv(D / "train.csv", usecols=["satisfaction"]).satisfaction.astype(int).to_numpy()
oof = np.load(ROOT / "oof/oof_stack_v14_both_tt_rm.npy").ravel()
p_sorted = np.sort(IsotonicRegression(out_of_bounds="clip").fit(oof, y_tr).predict(np.load(ROOT / "oof/test_stack_v14_both_tt_rm.npy").ravel()))
orders = {k: np.argsort(v, kind="stable") for k, v in F.items()}
pairs = [("M213", "N210"), ("M213", "mix50"), ("N210", "mix50"), ("M213", "slot1"), ("mix50", "slot1")]
rng = np.random.default_rng(2410)
for tname in ("N210", "M213"):
    p = p_sorted[(R[tname] * N - 1).round().astype(int).clip(0, N - 1)]
    gp, gq = {q: [] for q in pairs}, {q: [] for q in pairs}
    for _ in range(a.sims):
        y = rng.random(N) < p
        m = np.zeros(N, bool)
        m[rng.choice(N, N_PUB, replace=False)] = True
        A = {k: (auc_masked(o, y, m), auc_masked(o, y, ~m)) for k, o in orders.items()}
        for x, z in pairs:
            gp[(x, z)].append(A[x][0] - A[z][0])
            gq[(x, z)].append(A[x][1] - A[z][1])
    print(f"\n== truth = {tname} (×1e-5): sd public | sd private | std dR | k_pub = sd_pub/std dR")
    for x, z in pairs:
        dr = np.std(R[x] - R[z])
        print(f"  {x:5s}-{z:5s} sd pub {np.std(gp[(x, z)]) * 1e5:5.2f} | sd priv {np.std(gq[(x, z)]) * 1e5:5.2f} | "
              f"std dR {dr:.5f} | k_pub {np.std(gp[(x, z)]) / dr:.5f}")
