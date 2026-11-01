"""Phương án dung hoà cho slot 2 (25/10): tiêm TabPFN theo đúng công thức cụm 0,96213, nhưng trên nền của nhóm.

    sub = rank(0.9·rank(slot2 mix50_nina210) + 0.1·p_TabPFN mehran)

Cụm 0,96213 = 0.9·rank(Nina 0,96210) + 0.1·p_TabPFN (xem A_slot2_dir.py). CV trên nền S14 nói bước tiêm thua (−16..−23e-6),
nhưng không đo được trên file đám đông → file này là cược theo public mà không trùng y hệt 114 đội. CHƯA NỘP.
Chạy: python notebooks/A_slot2_hedge.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
ids = pd.read_csv(D / "test.csv", usecols=["id"]).id.to_numpy()
pr = lambda v: (rankdata(v) - 0.5) / len(v)


def load(path):
    d = pd.read_csv(path)
    col = [c for c in d.columns if c != "id"][0]
    return d.set_index("id")[col].loc[ids].to_numpy(np.float64)


P24, P22 = D / "public_subs_2410", D / "public_subs_2210"
mix50 = load(ROOT / "submissions/sub_slot2_mix50_nina210.csv")
pt = load(P24 / "mehrankazeminia/1-airline-s6e10-tabpfn/submission.csv")
hedge = pr(0.9 * pr(mix50) + 0.1 * pt)

R = {"slot2 mix50 (0.96209)": pr(mix50), "M213 cụm (0.96213)": pr(load(P24 / "mehrankazeminia/3-airline-s6e10-ensembling/submission.csv")),
     "N210 nina": pr(load(P22 / "nina2025/submission.csv")), "slot1 (0.96191)": pr(load(ROOT / "submissions/sub_slot1_stack80_kirill20.csv"))}
print("file hedge so với:")
for k, r in R.items():
    print(f"  {k:24s} corr hạng {np.corrcoef(hedge, r)[0, 1]:.6f}   std dR {(hedge - r).std():.5f}")

out = ROOT / "submissions/sub_slot2_mix50_tabpfn.csv"
pd.DataFrame({"id": ids, "satisfaction": hedge}).to_csv(out, index=False)
print(f"đã lưu {out.relative_to(ROOT)} ({len(ids):,} dòng, giá trị trong ({hedge.min():.2e}, {hedge.max():.6f}))")
