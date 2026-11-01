"""Phân tích phần sai số của stack công khai: nhóm feature nào còn thông tin mà ~200 member công khai chưa có?

Cần oof/oof_stack_lr_clean_public.npy (tải từ output notebook C_stack_public_oof trên Kaggle).
Chạy từ thư mục gốc repo:
  python notebooks/A_residual.py                    (mọi nhóm)
  python notebooks/A_residual.py --groups orig te   (chỉ vài nhóm)

Cách đo: LightGBM nhỏ bắt đầu từ logit của stack (init_score) rồi học thêm từ MỘT nhóm feature,
chia theo folds.csv, dừng sớm trên 10% tách từ phần train của fold. So AUC với stack đứng một mình.
Tăng rõ (vài x 1e-5, ở phần lớn fold) = nhóm đó có thông tin mới, đáng làm feature / member.
Stack OOF được tính bằng meta fold seed 7 nên mỗi dòng vẫn là dự đoán ngoài mẫu; TE vẫn tính trong fold.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from A_lgb_fe import DIGIT_COLS, add_te
from A_lr import logit, static_features
from config import FEATURES, N_FOLDS, OOF_DIR, TARGET
from data import load_data

T0 = time.time()
PARAMS = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=15, min_child_samples=500,
              lambda_l2=10.0, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, seed=42,
              num_threads=12, verbose=-1)


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def groups(X):
    cols = X.columns
    orig = [c for c in cols if "teacher" in c or c.endswith(("_orig_mean", "_rate", "_logcnt"))]
    return {
        "raw": FEATURES,
        "digits": [f"{c}_d{k}" for c in DIGIT_COLS for k in range(4)],
        "freq": [c for c in cols if c.endswith("_freq")],
        "orig": orig,
        "te": [],      # target encoding: tính trong fold
        "members": [],  # OOF của A_lgb_v1 / A_lr_v1 / A_knn_v1
        "all": list(cols),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default="stack_lr_clean_public", help="tên oof_<tên>.npy của stack công khai")
    ap.add_argument("--groups", nargs="+", default=["raw", "digits", "freq", "orig", "te", "members", "all"])
    args = ap.parse_args()

    train, test = load_data()
    y, fold = train[TARGET].values, train["fold"].values
    s = logit(np.load(OOF_DIR / f"oof_{args.stack}.npy"))
    X, _, K, _ = static_features(train, test)
    G = groups(X)
    members = {n: logit(np.load(OOF_DIR / f"oof_{n}.npy")) for n in ["A_lgb_v1", "A_lr_v1", "A_knn_v1"]}
    base = [roc_auc_score(y[fold == k], s[fold == k]) for k in range(N_FOLDS)]
    log(f"stack {args.stack}: AUC {roc_auc_score(y, s):.6f}")

    rows = []
    for g in args.groups:
        oof = np.zeros(len(y))
        for k in range(N_FOLDS):
            va = np.where(fold == k)[0]
            tr_idx = np.where(fold != k)[0]
            es = np.random.default_rng(42).choice(tr_idx, size=len(tr_idx) // 10, replace=False)
            fit = np.setdiff1d(tr_idx, es)
            if g == "te":
                F_fit, F_va, F_es = add_te(pd.DataFrame(index=fit), K.iloc[fit], y[fit],
                                           [(pd.DataFrame(index=va), K.iloc[va]), (pd.DataFrame(index=es), K.iloc[es])])
            elif g == "members":
                M = pd.DataFrame(members)
                F_fit, F_va, F_es = M.iloc[fit], M.iloc[va], M.iloc[es]
            else:
                F_fit, F_va, F_es = X[G[g]].iloc[fit], X[G[g]].iloc[va], X[G[g]].iloc[es]
            m = lgb.train(PARAMS, lgb.Dataset(F_fit, y[fit], init_score=s[fit]), num_boost_round=3000,
                          valid_sets=[lgb.Dataset(F_es, y[es], init_score=s[es])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            oof[va] = s[va] + m.predict(F_va, num_iteration=m.best_iteration, raw_score=True)
        per = [roc_auc_score(y[fold == k], oof[fold == k]) - base[k] for k in range(N_FOLDS)]
        d = roc_auc_score(y, oof) - roc_auc_score(y, s)
        rows.append((g, d * 1e6, sum(p > 0 for p in per)))
        log(f"{g:<8} {d * 1e6:+8.1f} x 1e-6, tăng {sum(p > 0 for p in per)}/5 fold | "
            + " ".join(f"{p * 1e6:+.1f}" for p in per))

    print("\nTóm tắt (x 1e-6; dưới ~10 coi như nhiễu):")
    print(pd.DataFrame(rows, columns=["nhóm", "chênh AUC", "fold tăng"]).to_string(index=False))
