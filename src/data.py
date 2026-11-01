"""Đọc dữ liệu kèm cột fold chung. Mọi model đều nên dùng load_data()."""
import pandas as pd

from config import DATA_DIR, ID, TARGET


def load_data():
    """Trả về (train, test). train có thêm cột 'fold' lấy từ folds.csv, target đã đổi sang 0/1."""
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    train[TARGET] = train[TARGET].astype(int)

    folds = pd.read_csv(DATA_DIR / "folds.csv")
    assert folds[ID].equals(train[ID]), "folds.csv không khớp thứ tự id của train.csv"
    train["fold"] = folds["fold"].values
    return train, test
