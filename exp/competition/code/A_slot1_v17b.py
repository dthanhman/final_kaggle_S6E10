"""Slot 1 bản 2 seed (27/10): 0.8·hạng(stack_v17b_qft15) + 0.2·hạng(0.96206 của Kirill), công thức như A_final_pair.py.

stack_v17b_qft15 = kernel cs116-stack-v17 version 2: S14 + qft2 TB logit seed 42 + seed 7 (LR ×1.5), CV 0.962233,
+6.6/+4.5/+4.9e-6 so với S14 (bản seed 42 riêng +7.8/+5.9/+6.0e-6, một phần do may seed). Mục đích: bớt phụ thuộc 1 seed.
Chỉ ghi slot 1, không đụng slot 2. Kiểm tra: dựng lại đúng sub_slot1_v17qft80_kirill20 từ stack_v17_qft15.

Chạy: python notebooks/A_slot1_v17b.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
ids = pd.read_csv(ROOT / "data" / "test.csv", usecols=["id"]).id.to_numpy()
N = len(ids)
pct = lambda v: (rankdata(np.asarray(v, dtype=np.float64), method="average") - 0.5) / len(v)


def load_sub(path):
    d = pd.read_csv(path)
    d.columns = ["id", "p"]
    v = d.set_index("id").p.reindex(ids).to_numpy(np.float64)
    assert v.shape == (N,) and np.isfinite(v).all(), path
    return v


ref = pct(load_sub(ROOT / "data" / "public_subs_2010" / "0.96206.csv"))
build = lambda name: pct(0.8 * pct(np.load(ROOT / "oof" / f"test_{name}.npy")) + 0.2 * ref)

old = load_sub(ROOT / "submissions" / "sub_slot1_v17qft80_kirill20.csv")
assert np.abs(build("stack_v17_qft15") - old).max() < 1e-8, "không dựng lại được slot 1 cũ"
print("kiểm tra: dựng lại đúng sub_slot1_v17qft80_kirill20")

y = pd.read_csv(ROOT / "data" / "train.csv", usecols=["satisfaction"]).satisfaction.astype(int).to_numpy()
for n in ("stack_v14_both_tt_rm", "stack_v17_qft15", "stack_v17b_qft15"):
    print(f"OOF {n:22s} AUC {roc_auc_score(y, np.load(ROOT / 'oof' / f'oof_{n}.npy')):.6f}")

new = build("stack_v17b_qft15")
print(f"tương quan hạng với slot 1 cũ {np.corrcoef(new, old)[0, 1]:.6f} | std hiệu hạng {np.std(new - old):.5f}")
out = pd.DataFrame({"id": ids, "satisfaction": new})
assert len(out) == N and out.id.is_unique and out.satisfaction.between(0, 1).all()
path = ROOT / "submissions" / "sub_slot1_v17b80_kirill20.csv"
out.to_csv(path, index=False, float_format="%.9f")
print(f"đã ghi {path.relative_to(ROOT)} (chưa nộp)")
