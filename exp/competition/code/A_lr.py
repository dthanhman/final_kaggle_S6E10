"""Logistic regression trên feature giàu: member tuyến tính, khác hẳn model cây.

Chạy từ thư mục gốc repo:
  python notebooks/A_lr.py --C 0.05 --name A_lr_v1
  python notebooks/A_lr.py --C 0.05 --folds 0 --name A_lr_try   (thử nhanh 1 fold, không lưu)

Feature (dùng lại build_static / add_te của A_lgb_fe.py, --step 4 --extra teachers):
  - one-hot thưa: từng giá trị của 21 cột, các cột chia nhóm, cặp cột, chữ số
    (giá trị hiếm < --min-freq dòng gộp chung)
  - số: teacher / tỉ lệ trên dữ liệu gốc -> logit, tần suất -> log, Age / Flight Distance / log delay
  - target encoding fit trên 4 fold train (cross-fit bên trong) -> logit
OOF sạch: không early stopping; TE chỉ dùng nhãn của 4 fold train; dữ liệu gốc chỉ làm feature.
Feature tĩnh được cache ở data/cache/ (không dùng nhãn cuộc thi nên dùng lại được).
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from A_lgb_fe import DIGIT_COLS, add_te, build_static
from config import CAT_COLS, DATA_DIR, N_FOLDS, NUM_COLS, TARGET
from data import load_data
from utils import save_preds

AGE, FD, DEP, ARR = NUM_COLS
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def static_features(train, test):
    cache = DATA_DIR / "cache" / "static_step4_teachers.pkl"
    if cache.exists():
        log(f"đọc cache {cache.name}")
        return pickle.loads(cache.read_bytes())
    orig = pd.read_csv(DATA_DIR / "original" / "data.csv")
    out = build_static(train, test, orig, 4, ["teachers"])
    cache.parent.mkdir(exist_ok=True)
    cache.write_bytes(pickle.dumps(out))
    return out


def dense_static(X):
    """Feature số không dùng nhãn cuộc thi, đưa về thang hợp với mô hình tuyến tính."""
    D = {AGE: X[AGE], FD: X[FD], "log_dep": np.log1p(X[DEP]),
         "log_arr": np.log1p(X[ARR].fillna(X[DEP])), "arr_missing": X[ARR].isna()}
    for c in X.columns:
        if c.endswith("_freq"):
            D[c] = np.log1p(X[c])
        elif "teacher" in c or c.endswith("_orig_mean") or c.endswith("_rate"):
            D[c] = logit(X[c])
        elif c.endswith("_logcnt"):
            D[c] = X[c]
    return pd.DataFrame(D).astype(np.float64)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=float, default=0.05)
    ap.add_argument("--min-freq", type=int, default=20, help="giá trị one-hot ít hơn số dòng này thì gộp chung")
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(N_FOLDS)), help="chỉ lưu khi chạy đủ 5 fold")
    ap.add_argument("--name", required=True, help="tên dạng A_<model>_<phiên bản>")
    args = ap.parse_args()
    if not args.name.startswith(("A_", "B_", "C_")):
        ap.error("tên phải dạng <người>_<model>_<phiên bản>, ví dụ A_lr_v1")

    train, test = load_data()
    y, fold = train[TARGET].values, train["fold"].values
    X, X_test, K, K_test = static_features(train, test)
    n = len(X)

    # one-hot thưa: fit trên train + test (không dùng nhãn)
    oh_cols = pd.concat([K, X[[f"{c}_d{k}" for c in DIGIT_COLS for k in range(4)]]], axis=1)
    oh_cols_test = pd.concat([K_test, X_test[[f"{c}_d{k}" for c in DIGIT_COLS for k in range(4)]]], axis=1)
    ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=args.min_freq, dtype=np.float32)
    ohe.fit(pd.concat([oh_cols, oh_cols_test], ignore_index=True).values)
    S, S_test = ohe.transform(oh_cols.values).tocsr(), ohe.transform(oh_cols_test.values).tocsr()

    Dn, Dn_test = dense_static(X), dense_static(X_test)
    sc = StandardScaler().fit(pd.concat([Dn, Dn_test], ignore_index=True))
    Dn, Dn_test = sc.transform(Dn), sc.transform(Dn_test)
    log(f"{S.shape[1]} cột one-hot + {Dn.shape[1]} cột số + {2 * K.shape[1]} cột target encoding")

    empty, empty_test = pd.DataFrame(index=range(n)), pd.DataFrame(index=range(len(X_test)))
    oof, test_pred = np.zeros(n), np.zeros(len(X_test))
    for k in args.folds:
        tr, va = np.where(fold != k)[0], np.where(fold == k)[0]
        T_tr, T_va, T_te = add_te(empty.iloc[tr], K.iloc[tr], y[tr],
                                  [(empty.iloc[va], K.iloc[va]), (empty_test, K_test)])
        T_tr, T_va, T_te = (logit(t.values) for t in (T_tr, T_va, T_te))
        tsc = StandardScaler().fit(T_tr)
        mats = [sp.hstack([S[idx], sp.csr_matrix(Dn[idx]), sp.csr_matrix(tsc.transform(T))]).tocsr()
                for idx, T in ((tr, T_tr), (va, T_va))]
        M_te = sp.hstack([S_test, sp.csr_matrix(Dn_test), sp.csr_matrix(tsc.transform(T_te))]).tocsr()
        m = LogisticRegression(C=args.C, max_iter=3000, tol=1e-5).fit(mats[0], y[tr])
        oof[va] = m.decision_function(mats[1])
        test_pred += m.decision_function(M_te) / N_FOLDS
        log(f"fold {k}: AUC = {roc_auc_score(y[va], oof[va]):.6f} | {m.n_iter_[0]} vòng lặp")

    if len(args.folds) == N_FOLDS:
        log(f"C {args.C}: CV AUC = {roc_auc_score(y, oof):.6f}")
        # lưu dạng xác suất
        save_preds(args.name, 1 / (1 + np.exp(-oof)), 1 / (1 + np.exp(-test_pred)), y)
