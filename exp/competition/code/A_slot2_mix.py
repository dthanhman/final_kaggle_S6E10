"""Slot 2 mới (22/10): trộn 50/50 hạng phần trăm giữa slot 2 cũ (sub_slot2_hblend_teacher, LB 0.96206) và file đám đông
mới nhất (Nina ps-s6e10 22/10, LB 0.96210). Lý do: A_crowd_drift.py, cụm đỉnh trượt ~std dR 0.01 khỏi slot 2;
trộn 50/50 lấy nửa lợi ích nếu bước leo là thật, nửa rủi ro nếu nó chỉ khớp nhiễu public.

Chạy: python notebooks/A_slot2_mix.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
ids = pd.read_csv(ROOT / "data" / "test.csv", usecols=["id"]).id.to_numpy()
pr = lambda v: (rankdata(v) - 0.5) / len(v)


def load(path):
    d = pd.read_csv(path)
    col = [c for c in d.columns if c != "id"][0]
    assert d.id.nunique() == len(ids) and set(d.id) == set(ids)
    return pr(d.set_index("id")[col].loc[ids].to_numpy(np.float64))


A = load(ROOT / "submissions" / "sub_slot2_hblend_teacher.csv")
B = load(ROOT / "data" / "public_subs_2210" / "nina2025" / "submission.csv")
M = 0.5 * A + 0.5 * B
out = ROOT / "submissions" / "sub_slot2_mix50_nina210.csv"
pd.DataFrame({"id": ids, "satisfaction": M}).to_csv(out, index=False)
for k, r in {"slot2 cũ": A, "nina 0.96210": B}.items():
    print(f"corr hạng với {k}: {np.corrcoef(M, r)[0, 1]:.6f}, std dR {np.std(M - r):.5f}")
print(f"ghi {out.name}: {len(M)} dòng, min {M.min():.6f} max {M.max():.6f}")
