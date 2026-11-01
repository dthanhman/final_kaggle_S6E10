"""Tạo data/folds.csv (đã chạy MỘT lần, file có sẵn trong repo). Chạy: python src/make_folds.py"""
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from config import DATA_DIR, ID, N_FOLDS, SEED, TARGET

train = pd.read_csv(DATA_DIR / "train.csv", usecols=[ID, TARGET])
out = DATA_DIR / "folds.csv"
if out.exists():
    raise SystemExit(f"{out} đã tồn tại. Không tạo lại.")

folds = train[[ID]].copy()
folds["fold"] = -1
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
for k, (_, val_idx) in enumerate(skf.split(train, train[TARGET])):
    folds.loc[val_idx, "fold"] = k

assert (folds["fold"] >= 0).all()
print(pd.crosstab(folds["fold"], train[TARGET], normalize="index").round(4))
folds.to_csv(out, index=False)
print(f"Đã lưu {out} ({len(folds)} dòng, {N_FOLDS} fold, seed {SEED})")
