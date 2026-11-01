"""Tạo file nộp từ oof/test_<tên>.npy và kiểm tra định dạng.

Chạy: python src/submit.py lgb   -> submissions/sub_lgb.csv
"""
import sys

import numpy as np
import pandas as pd

from config import DATA_DIR, ID, OOF_DIR, SUB_DIR, TARGET

name = sys.argv[1]
pred = np.load(OOF_DIR / f"test_{name}.npy")
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

assert len(pred) == len(sample), f"số dòng {len(pred)} khác sample_submission {len(sample)}"
assert not np.isnan(pred).any(), "có NaN"
assert np.unique(pred).size > 2, "có vẻ là nhãn 0/1, phải nộp xác suất"
if pred.min() < 0 or pred.max() > 1:
    print("Cảnh báo: giá trị ngoài [0, 1] (AUC vẫn tính được nhưng nên kiểm tra lại)")

sub = pd.DataFrame({ID: sample[ID], TARGET: pred})
SUB_DIR.mkdir(exist_ok=True)
out = SUB_DIR / f"sub_{name}.csv"
sub.to_csv(out, index=False)
print(f"Đã lưu {out}")
print(sub.head())
print(f'Nộp: kaggle competitions submit -c playground-series-s6e10 -f "{out}" -m "{name}"')
