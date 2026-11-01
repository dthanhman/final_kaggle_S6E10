"""Phân tích lát cắt: stack công khai dự đoán lệch một cách hệ thống ở nhóm khách nào?

Cần oof/oof_stack_lr_clean_public.npy (OOF của stack ~198 member công khai, tải từ notebook cs116-stack).
Chạy từ thư mục gốc repo:
  python notebooks/A_slices.py

Với mỗi khóa (một cột, một cột biến đổi, hoặc một cặp cột), chia train theo giá trị của khóa. Ở mỗi nhóm:
  quan sát = số khách hài lòng thật, kỳ vọng = tổng xác suất stack dự đoán,
  z = (quan sát - kỳ vọng) / sqrt(sum p(1-p)).
Nếu stack đã nắm hết thông tin của khóa thì z của các nhóm xấp xỉ chuẩn N(0, 1): trung bình z^2 khoảng 1.
Trung bình z^2 lớn hơn 1 rõ ràng = khóa đó còn tín hiệu stack chưa dùng (ứng viên feature).
Thử hàng nghìn nhóm nên sẽ có vài |z| > 4 do ngẫu nhiên; so với số kỳ vọng in ra trước khi tin.
"""
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from config import FEATURES, NUM_COLS, OOF_DIR, SCORE_COLS, TARGET
from data import load_data

AGE, FD, DEP, ARR = NUM_COLS
MIN_N = 200          # chỉ xét nhóm có ít nhất chừng này dòng
STACK = sys.argv[1] if len(sys.argv) > 1 else "stack_lr_clean_public"

train, _ = load_data()
y = train[TARGET].values.astype(float)
p = np.clip(np.load(OOF_DIR / f"oof_{STACK}.npy"), 1e-6, 1 - 1e-6)
print(f"stack {STACK}: AUC {roc_auc_score(y, p):.6f} | tỉ lệ thật {y.mean():.4f}, trung bình dự đoán {p.mean():.4f}")

# khóa đơn: 21 cột gốc + vài biến đổi (dấu vết của bộ sinh dữ liệu, không dùng nhãn)
S = train[SCORE_COLS]
derived = {
    "fd_mod10": train[FD] % 10, "fd_mod100": train[FD] % 100, "fd_div10": train[FD] // 10,
    "age_mod10": train[AGE] % 10, "dep_eq_arr": (train[DEP] == train[ARR]).astype(int),
    "arr_missing": train[ARR].isna().astype(int), "dep_mod5": train[DEP] % 5,
    "zero_count": (S == 0).sum(axis=1), "score_sum": S.sum(axis=1), "score_max_minus_min": S.max(1) - S.min(1),
}
D = pd.concat([train[FEATURES], pd.DataFrame(derived)], axis=1)
keys = {c: [c] for c in D.columns}
keys.update({f"{a} x {b}": [a, b] for a, b in combinations(FEATURES, 2)})

var = p * (1 - p)
rows, groups = [], []
for name, cols in keys.items():
    g = pd.DataFrame({"y": y, "p": p, "v": var}).groupby([D[c].fillna(-1).values for c in cols])
    a = g.agg(n=("y", "size"), obs=("y", "sum"), exp=("p", "sum"), v=("v", "sum"))
    a = a[a["n"] >= MIN_N]
    if a.empty:
        continue
    z = (a["obs"] - a["exp"]) / np.sqrt(a["v"])
    covered = a["n"].sum() / len(y)
    rows.append(dict(key=name, groups=len(a), covered=covered, mean_z2=float((z ** 2).mean()),
                     max_abs_z=float(z.abs().max())))
    top = a.assign(z=z, obs_rate=a["obs"] / a["n"], pred_rate=a["exp"] / a["n"]).sort_values("z", key=abs).tail(3)
    for val, r in top.iterrows():
        groups.append(dict(key=name, value=val, n=int(r["n"]), obs_rate=r["obs_rate"], pred_rate=r["pred_rate"],
                           z=r["z"]))

K = pd.DataFrame(rows).sort_values("mean_z2", ascending=False)
G = pd.DataFrame(groups)
n_groups = int(K["groups"].sum())
print(f"\n{len(K)} khóa, {n_groups:,} nhóm có >= {MIN_N} dòng")
print(f"số nhóm |z| > 4: {int((G['z'].abs() > 4).sum())} (trong top 3 mỗi khóa) | "
      f"kỳ vọng nếu chỉ là ngẫu nhiên trên mọi nhóm: {n_groups * 2 * norm.sf(4):.1f}")

pd.set_option("display.width", 200)
print("\nKhóa có trung bình z^2 cao nhất (~1 = stack đã nắm hết; lớn hơn rõ = còn tín hiệu):")
print(K.head(25).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("\nNhóm lệch nhiều nhất (obs_rate thật vs pred_rate stack):")
print(G.reindex(G["z"].abs().sort_values(ascending=False).index).head(30)
      .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
K.to_csv(OOF_DIR.parent / "reports" / "slices_keys.csv", index=False)
G.to_csv(OOF_DIR.parent / "reports" / "slices_groups.csv", index=False)
print("\nđã lưu reports/slices_keys.csv, reports/slices_groups.csv")
