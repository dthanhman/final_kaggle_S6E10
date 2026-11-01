"""Slot 2 tâm dòng probe (31/10): trung bình hạng 7 file công khai 0.96217–0.96225 (goodpjw meta → tkms push → mirror → areebbb).

Các bước probe đi theo hướng khá khác nhau (goodpjw chỉ 24% theo hướng areebbb−M213), nên lấy tâm giữ phần chung của dòng
mà triệt bớt phần dò public riêng của từng file. Kiểm tra trên OOF: bước "đẩy xa file yếu" của tkms (kumolarge, tabpfn35, xgb)
cho stack_v17b ±0,1e-5 → phần lợi public của dòng không có cơ sở CV rõ.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

D = Path(__file__).resolve().parents[1] / "data"
T = D / "public_subs_3110"
ids = pd.read_csv(D / "test.csv", usecols=["id"]).id.to_numpy()


def load(path):
    p = pd.read_csv(path).set_index("id").loc[ids, "satisfaction"].to_numpy()
    return rankdata(p) / len(p)


names = ["goodpjw_meta_96217", "tkms_r1_96217", "areeb_less_96218", "mirror_pair_96218", "mirror3_96219", "tkms_t3_96223", "areeb_endgame_96225"]
C = rankdata(np.mean([load(T / f"{n}.csv") for n in names], 0)) / len(ids)
A = load(T / "areeb_endgame_96225.csv")
S2 = load(D.parent / "submissions/sub_slot2_anti10_v17b.csv")
print(f"centroid: sdR vs areebbb {np.std(C - A):.5f} | vs anti10 {np.std(C - S2):.5f}")
pd.DataFrame({"id": ids, "satisfaction": C}).to_csv(D.parent / "submissions/sub_slot2_lineage_centroid.csv", index=False)
