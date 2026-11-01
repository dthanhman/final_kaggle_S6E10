"""Lưu / đọc OOF và test preds theo quy ước oof_<tên>.npy, test_<tên>.npy."""
import numpy as np
from sklearn.metrics import roc_auc_score

from config import OOF_DIR

N_TRAIN, N_TEST = 699_635, 299_844


def save_preds(name, oof, test_pred, y=None):
    """Lưu dự đoán của một model. Truyền y để in luôn CV AUC."""
    oof, test_pred = np.asarray(oof, dtype=float), np.asarray(test_pred, dtype=float)
    assert oof.shape == (N_TRAIN,), f"oof phải có shape ({N_TRAIN},), đang là {oof.shape}"
    assert test_pred.shape == (N_TEST,), f"test_pred phải có shape ({N_TEST},), đang là {test_pred.shape}"
    assert not np.isnan(oof).any() and not np.isnan(test_pred).any(), "có NaN trong dự đoán"

    OOF_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OOF_DIR / f"oof_{name}.npy", oof)
    np.save(OOF_DIR / f"test_{name}.npy", test_pred)
    msg = f"Đã lưu oof_{name}.npy, test_{name}.npy"
    if y is not None:
        msg += f" | CV AUC = {roc_auc_score(y, oof):.6f}"
    print(msg)


def load_preds(name):
    return np.load(OOF_DIR / f"oof_{name}.npy"), np.load(OOF_DIR / f"test_{name}.npy")
