"""Slot 2 bước nhỏ (28/10): từ file cụm M213 (0.96213) đi một bước nhỏ về stack của nhóm (slot1_v17b, hướng duy nhất có CV).

    sub_w = rank((1 - w)·rank(M213) + w·rank(slot1_v17b80_kirill20)),  w = 0.10 / 0.20 / 0.30

Lý do: chép y M213 thì hoà private với cả cụm và đứng cuối (nộp muộn); bước nhỏ giữ public ≈ cụm nhưng tách khỏi hoà điểm.
Cỡ nhiễu public ước theo quy tắc goodpjw: sd_public ≈ 0.003 × std hiệu hạng (sdR).
Chạy: python notebooks/A_slot2_step.py
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
    v = d.set_index("id")[col].reindex(ids).to_numpy(np.float64)
    assert np.isfinite(v).all(), path
    return pr(v)


M213 = load(D / "public_subs_2410/mehrankazeminia/3-airline-s6e10-ensembling/submission.csv")
S1 = load(ROOT / "submissions/sub_slot1_v17b80_kirill20.csv")
MIX = load(ROOT / "submissions/sub_slot2_mix50_tabpfn.csv")
print(f"M213 vs slot1_v17b: sdR {np.std(M213 - S1):.5f} | M213 vs mix50_tabpfn: sdR {np.std(M213 - MIX):.5f}")

# w < 0 (anti): đi NGƯỢC hướng stack, ra xa slot 1. Cụm 0.96213 là một khối gần như trùng (sdR ≤ 0.001 giữa các file
# công khai) và step10 lệch M213 gần như hoàn toàn theo hướng stack → step+ thắng khối chỉ khi hướng stack tốt = cùng cược
# với slot 1. anti thắng khối khi hướng stack xấu → che đúng kịch bản còn lại.
for w in (-0.10, -0.05, 0.10, 0.20, 0.30):
    s = pr((1 - w) * M213 + w * S1)
    d = {k: np.std(s - r) for k, r in {"M213": M213, "mix50_tabpfn": MIX, "slot1_v17b": S1}.items()}
    print(f"w={w:.2f}: sdR " + ", ".join(f"{k} {v:.5f}" for k, v in d.items())
          + f" | sd_public so M213 ≈ {0.003 * d['M213'] * 1e5:.1f}e-5")
    tag = f"step{round(w * 100):02d}" if w > 0 else f"anti{round(-w * 100):02d}"
    out = ROOT / f"submissions/sub_slot2_{tag}_v17b.csv"
    o = pd.DataFrame({"id": ids, "satisfaction": s})
    assert len(o) == len(ids) and o.id.is_unique and o.satisfaction.between(0, 1).all()
    o.to_csv(out, index=False, float_format="%.9f")
    print(f"  đã ghi {out.relative_to(ROOT)}")
