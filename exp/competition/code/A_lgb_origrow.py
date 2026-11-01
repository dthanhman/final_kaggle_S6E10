"""LightGBM "orig as rows": ghép 129k dòng dữ liệu gốc vào PHẦN TRAIN của mỗi fold (ý của Masaya Kawamata,
writeup S6E5: model "orig row" và "orig col" blend với nhau tốt). Dùng làm member cho stack, không phải model đơn.

Đúng STACKING.md: dòng gốc chỉ vào phần fit của fold, không bao giờ vào fold validation hay phần dừng sớm;
dừng sớm trên 10% tách từ phần train (dòng cuộc thi) của fold. Không có target encoding (tránh trộn nhãn gốc vào TE).

Feature: 21 cột gốc + chữ số + tần suất (đếm trên train + test) + cờ is_orig.
  --orig rows     ghép dòng gốc (member chính)
  --orig none     cùng feature, không ghép (đối chứng: phần tăng của member chính phải hơn bản này)

Chạy: python notebooks/A_lgb_origrow.py --orig rows --name A_lgb_origrow_v1
      python notebooks/A_lgb_origrow.py --orig none --name A_lgb_noorig_v1
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import CAT_COLS, DATA_DIR, FEATURES, N_FOLDS, NUM_COLS, SCORE_COLS, TARGET
from data import load_data
from utils import save_preds

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def build(train, test, orig):
    """Feature cho train + test + gốc (không dùng nhãn). Tần suất đếm trên train + test."""
    A = pd.concat([train[FEATURES], test[FEATURES], orig[FEATURES]], ignore_index=True)
    comp = A.iloc[:len(train) + len(test)]
    X = A.copy()
    for c in CAT_COLS:
        X[c] = X[c].astype("category")
    for c in NUM_COLS:
        v = A[c].fillna(0).astype(np.int64)
        for k in range(4):
            X[f"{c}_d{k}"] = (v // 10 ** k % 10).astype(np.int8)
    for c in NUM_COLS + SCORE_COLS:
        X[f"{c}_freq"] = A[c].map(comp[c].value_counts(dropna=False)).fillna(0).astype(np.float32)
    X["is_orig"] = np.r_[np.zeros(len(comp)), np.ones(len(orig))].astype(np.int8)
    n1, n2 = len(train), len(train) + len(test)
    return X.iloc[:n1].reset_index(drop=True), X.iloc[n1:n2].reset_index(drop=True), X.iloc[n2:].reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", choices=["rows", "none"], required=True)
    ap.add_argument("--orig-weight", type=float, default=1.0, help="trọng số mẫu của dòng gốc")
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--leaves", type=int, default=63)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    if not args.name.startswith(("A_", "B_", "C_")):
        ap.error("tên phải dạng <người>_<model>_<phiên bản>")

    train, test = load_data()
    orig = pd.read_csv(DATA_DIR / "original" / "data.csv")
    y, fold = train[TARGET].values, train["fold"].values
    yo = orig[TARGET].astype(int).values
    X, X_test, X_orig = build(train, test, orig)
    log(f"orig={args.orig}: {X.shape[1]} feature, {len(X_orig)} dòng gốc")

    params = dict(objective="binary", metric="auc", learning_rate=args.lr, num_leaves=args.leaves, max_depth=8,
                  min_child_samples=5, lambda_l2=5.0, lambda_l1=0.05, bagging_fraction=0.9, bagging_freq=1,
                  feature_fraction=0.9, seed=args.seed, num_threads=12, verbose=-1)
    oof, test_pred, n_trees = np.zeros(len(y)), np.zeros(len(X_test)), []
    for k in range(N_FOLDS):
        va = fold == k
        tr_idx = np.where(fold != k)[0]
        es_idx = np.random.default_rng(42).choice(tr_idx, size=len(tr_idx) // 10, replace=False)
        fit_idx = np.setdiff1d(tr_idx, es_idx)
        X_fit, y_fit, w_fit = X.iloc[fit_idx], y[fit_idx], np.ones(len(fit_idx))
        if args.orig == "rows":  # dòng gốc chỉ vào phần fit
            X_fit = pd.concat([X_fit, X_orig], ignore_index=True)
            y_fit = np.r_[y_fit, yo]
            w_fit = np.r_[w_fit, np.full(len(yo), args.orig_weight)]
        m = lgb.train(params, lgb.Dataset(X_fit, y_fit, weight=w_fit), num_boost_round=20000,
                      valid_sets=[lgb.Dataset(X.iloc[es_idx], y[es_idx])],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        best = m.best_iteration
        n_trees.append(best)
        oof[va] = m.predict(X[va], num_iteration=best)
        test_pred += m.predict(X_test, num_iteration=best) / N_FOLDS
        log(f"fold {k}: AUC = {roc_auc_score(y[va], oof[va]):.6f} | {best} cây")

    log(f"CV AUC = {roc_auc_score(y, oof):.6f} | số cây TB {int(np.mean(n_trees))}")
    save_preds(args.name, oof, test_pred, y)
