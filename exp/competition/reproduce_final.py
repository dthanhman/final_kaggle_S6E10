"""Tái tạo file nộp slot 1 (private 0,96239, hạng 66) từ các phần đã lưu, và so với file đã nộp lên Kaggle.

Bài cuối = 0,8 · hạng(stack_v17b_qft15) + 0,2 · hạng(file công khai 0,96206 của Kirill), rồi đổi về hạng phần trăm.
Code gốc: exp/competition/code/A_slot1_v17b.py.

Đầu vào (đặt ở data/artifacts/, không commit; lấy từ repo cũ hoặc Kaggle):
  test_stack_v17b_qft15.npy, oof_stack_v17b_qft15.npy   dự đoán test / OOF của stack v17b (kernel cs116-stack-v17 v2)
  0.96206.csv                                          file công khai của Kirill (public LB 0,96206)
  sub_slot1_v17b80_kirill20.csv                        file đã nộp ngày 27/10, dùng để đối chiếu

Chạy từ gốc repo: python exp/competition/reproduce_final.py
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from config import DATA_DIR, SUB_DIR, TARGET  # noqa: E402

ART = DATA_DIR / "artifacts"
ids = pd.read_csv(DATA_DIR / "test.csv", usecols=["id"])["id"].to_numpy()


def pct(v):
    v = np.asarray(v, dtype=np.float64)
    return (rankdata(v, method="average") - 0.5) / len(v)


def load_sub(path):
    d = pd.read_csv(path)
    d.columns = ["id", "p"]
    v = d.set_index("id")["p"].reindex(ids).to_numpy(np.float64)
    assert np.isfinite(v).all(), path
    return v


y = pd.read_csv(DATA_DIR / "train.csv", usecols=[TARGET])[TARGET].astype(int).to_numpy()
print(f"OOF stack_v17b_qft15: AUC {roc_auc_score(y, np.load(ART / 'oof_stack_v17b_qft15.npy')):.6f} (log khi thi: 0,962224)")

stack = np.load(ART / "test_stack_v17b_qft15.npy")
kirill = load_sub(ART / "0.96206.csv")
final = pct(0.8 * pct(stack) + 0.2 * pct(kirill))

submitted = load_sub(ART / "sub_slot1_v17b80_kirill20.csv")
diff = np.abs(final - submitted).max()
print(f"chênh lệch lớn nhất với file đã nộp: {diff:.2e}")
assert diff < 1e-8, "không tái tạo được file đã nộp"

SUB_DIR.mkdir(exist_ok=True)
out = SUB_DIR / "reproduced_slot1.csv"
pd.DataFrame({"id": ids, "satisfaction": final}).to_csv(out, index=False, float_format="%.9f")
md5 = lambda p: hashlib.md5(Path(p).read_bytes()).hexdigest()  # noqa: E731
print(f"đã ghi {out.relative_to(ROOT)} | md5 {md5(out)} | md5 file đã nộp {md5(ART / 'sub_slot1_v17b80_kirill20.csv')}")
print("TÁI TẠO ĐÚNG file nộp slot 1 (public 0,96190, private 0,96239)")
