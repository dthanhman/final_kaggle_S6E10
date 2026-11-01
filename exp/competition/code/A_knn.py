"""kNN: tỉ lệ hài lòng của k khách giống nhất trong phần train của fold. Member khác hẳn cây / tuyến tính.

Chạy từ thư mục gốc repo:
  python notebooks/A_knn.py --folds 0 --name A_knn_try        (thử fold 0: in AUC mọi k / cách lấy trọng số)
  python notebooks/A_knn.py --k 50 --weight dist --name A_knn_v1

Không gian khoảng cách: knn_space của A_lgb_fe.py (13 điểm, one-hot category x2, tuổi/10, quãng đường/400, log delay).
OOF sạch: hàng xóm của fold k chỉ lấy trong 4 fold còn lại; test = trung bình 5 mô hình fold. Không dùng dữ liệu gốc.
Trọng số "dist" = 1 / (khoảng cách + 0.1), vừa ưu tiên khách gần hơn vừa phá các giá trị trùng nhau (AUC cần thứ hạng).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from A_lgb_fe import knn_space
from config import N_FOLDS, TARGET
from data import load_data
from utils import save_preds

KS = [10, 25, 50, 100, 200]
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def knn_rate(dist, idx, y_tr, k, weight):
    lab = y_tr[idx[:, :k]]
    if weight == "uniform":
        return lab.mean(1)
    w = 1 / (dist[:, :k] + 0.1)
    return (lab * w).sum(1) / w.sum(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--weight", choices=["uniform", "dist"], default="dist")
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(N_FOLDS)), help="chỉ lưu khi chạy đủ 5 fold")
    ap.add_argument("--name", required=True, help="tên dạng A_<model>_<phiên bản>")
    args = ap.parse_args()
    if not args.name.startswith(("A_", "B_", "C_")):
        ap.error("tên phải dạng <người>_<model>_<phiên bản>, ví dụ A_knn_v1")
    full = len(args.folds) == N_FOLDS
    k_max = args.k if full else max(KS)

    train, test = load_data()
    y, fold = train[TARGET].values, train["fold"].values
    Z = knn_space(train)
    Z_test = knn_space(test).reindex(columns=Z.columns, fill_value=0)
    Z, Z_test = Z.values.astype(np.float32), Z_test.values.astype(np.float32)
    log(f"không gian {Z.shape[1]} chiều")

    oof, test_pred = np.zeros(len(y)), np.zeros(len(Z_test))
    for k in args.folds:
        tr, va = np.where(fold != k)[0], np.where(fold == k)[0]
        nn = NearestNeighbors(n_neighbors=k_max, algorithm="brute", n_jobs=-1).fit(Z[tr])
        d_va, i_va = nn.kneighbors(Z[va])
        if not full:  # thử fold: in mọi cấu hình, không dự đoán test
            for kk in KS:
                res = {w: roc_auc_score(y[va], knn_rate(d_va, i_va, y[tr], kk, w)) for w in ("uniform", "dist")}
                log(f"fold {k} k={kk:<4} uniform {res['uniform']:.6f} | dist {res['dist']:.6f}")
            continue
        oof[va] = knn_rate(d_va, i_va, y[tr], args.k, args.weight)
        d_te, i_te = nn.kneighbors(Z_test)
        test_pred += knn_rate(d_te, i_te, y[tr], args.k, args.weight) / N_FOLDS
        log(f"fold {k}: AUC = {roc_auc_score(y[va], oof[va]):.6f}")

    if full:
        log(f"k {args.k} {args.weight}: CV AUC = {roc_auc_score(y, oof):.6f}")
        save_preds(args.name, oof, test_pred, y)
