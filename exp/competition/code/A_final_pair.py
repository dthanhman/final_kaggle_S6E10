"""Cặp bài cuối 20/10: slot 1 nghiêng stack CV, slot 2 = gated h-blend (Nina / Kirill / mayankraaz) với stack của mình làm teacher.

Slot 1: 0.8·hạng(stack) + 0.2·hạng(0.96206 của Kirill). Phần lớn vẫn là stack CV, thêm 20% hướng đang dẫn public LB.
Slot 2: công thức "q9" của mayankraaz Surface Ascent v3 (public 0.96206): h-blend 4 file public 0.96201-0.96204
  + teacher chiếm trung bình 35% (±1% theo độ bất đồng giữa 4 member), teacher = (Chris 0.96176 + Den 0.96183 + 3·stack)/5.
  Thay stack 181 member của họ (CV 0.962166) bằng stack của mình (CV 0.962215); trên OOF trộn thêm stack 181 không tăng ổn định
  (w 0.1: -5.8/+2.8/+3.8/-0.3/+0.2e-6 theo fold).
Các file public không có OOF nên không chấm được bằng CV; trọng số lấy theo công thức công khai, không dò LB.
Hàm hblend / gated_blend chép từ gate_blend.py của mayankraaz (kiểm tra: tái tạo đúng file 0.96206 của Kirill).

Chạy: python notebooks/A_final_pair.py [--base stack_team_realmlp_fw_noforge]
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "data" / "public_subs_2010"     # từ dataset nina2025/ps-s6e10-7 và mayankraaz/s6e10-lb-surface-atlas
ap = argparse.ArgumentParser()
ap.add_argument("--base", default="stack_team_realmlp_fw_noforge")
ap.add_argument("--w1", type=float, default=0.2, help="tỉ lệ file 0.96206 trong slot 1")
args = ap.parse_args()

PINNED = {"0.96201.csv": "8883773ff30ba9a71581657b9daf881ac45652031b6e3ee565844dbcff35797a",
          "0.96202.csv": "4765dbd769768cb9a7cbe030b9ecfd83a12b4fab25529d135cc6fef6fbc6e028",
          "0.96203.csv": "b31119ef653e50b71452187174d6c85d7929550c23b4b020797f19f5f132112d",
          "0.96204.csv": "c7d3b9e4c1c377c74e0f75b475c50319413a264fc4401c0a00fe689a99542873",
          "0.96176.csv": "9b5dc8e667fdecb388e56afd4772ecfb386cfb783f48b4d2270e747ebb371471",
          "0.96183.csv": "9f499e3b36232a0969dc50c26968946935808503a1f6abb562b91452251211db",
          "0.96206.csv": "de8438a811025afe781ff32af09ae0ada7a035741cdb98c1029be06ddba6143f"}
for f, h in PINNED.items():
    assert hashlib.sha256((PUB / f).read_bytes()).hexdigest() == h, f"{f} khác bản đã ghim"

ids = pd.read_csv(ROOT / "data" / "test.csv", usecols=["id"]).id.to_numpy()
N = len(ids)


def pct(v):
    return (rankdata(np.asarray(v, dtype=np.float64), method="average") - 0.5) / len(v)


def load(name):
    d = pd.read_csv(PUB / name)
    d.columns = ["id", "p"]
    v = d.set_index("id").p.reindex(ids).to_numpy(np.float64)
    assert v.shape == (N,) and np.isfinite(v).all(), name
    return v


MAIN_W = np.array([0.1, 0.2, 0.3, 0.4])          # trọng số theo member (h-blend của Nina)
ORDER_C = np.array([0.07, -0.01, -0.02, -0.04])  # chỉnh theo vị trí sau khi sắp xếp từng dòng


def hblend(M, main=MAIN_W, corr=ORDER_C, mix=(0.3, 0.7)):
    n, sides = len(M), []
    for rev in (False, True):
        order = np.argsort(-M if rev else M, axis=1, kind="stable")
        s = np.zeros(n)
        for j in range(M.shape[1]):
            k = order[:, j]
            s += M[np.arange(n), k] * (main[k] + corr[j])
        sides.append(s.astype(np.float32))
    return (np.float32(mix[0]) * sides[0] + np.float32(mix[1]) * sides[1]).astype(np.float32)


def gated_blend(members, teacher, weight=0.30, amp=0.02):
    w = weight + amp * (pct(members.std(axis=1)) - 0.5)
    return pct(hblend((1 - w[:, None]) * members + w[:, None] * teacher[:, None]))


M = np.column_stack([load(f"0.{v}.csv") for v in ("96201", "96202", "96203", "96204")])
tc, td, ref = load("0.96176.csv"), load("0.96183.csv"), pct(load("0.96206.csv"))
assert np.array_equal(gated_blend(M, (tc + td) / 2, 0.30, 0.02), ref), "không tái tạo được file 0.96206"
print("kiểm tra: tái tạo đúng từng bit file 0.96206 của Kirill")

stack = np.load(ROOT / "oof" / f"test_{args.base}.npy").astype(np.float64)   # xác suất, cùng thang với tc, td
assert stack.shape == (N,) and 0 < stack.min() and stack.max() < 1
own = np.load(PUB / "own_cv_stack_181_test.npy").astype(np.float64)

slot1 = pct((1 - args.w1) * pct(stack) + args.w1 * ref)
slot2 = gated_blend(M, (tc + td + 3 * stack) / 5, 0.35, 0.02)
q9 = gated_blend(M, (tc + td + 3 * own) / 5, 0.35, 0.02)          # bản public 0.96206 của mayankraaz, để so

c = lambda a, b: np.corrcoef(a, b)[0, 1]
S = {"stack": pct(stack), "0.96206 Kirill": ref, "q9 mayankraaz": q9, "slot1": slot1, "slot2": slot2}
print(f"\n{'tương quan hạng':16s}" + "".join(f"{k:>16s}" for k in S))
for k, r in S.items():
    print(f"{k:16s}" + "".join(f"{c(r, s):16.6f}" for s in S.values()))

for tag, v in (("slot1_stack80_kirill20", slot1), ("slot2_hblend_teacher", slot2)):
    out = pd.DataFrame({"id": ids, "satisfaction": v})
    assert len(out) == N and out.id.is_unique and out.satisfaction.between(0, 1).all()
    path = ROOT / "submissions" / f"sub_{tag}.csv"
    out.to_csv(path, index=False, float_format="%.9f")
    print(f"đã ghi {path.relative_to(ROOT)} (chưa nộp)")
