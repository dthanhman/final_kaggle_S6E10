"""Model cây KHÁC LOẠI với pool công khai (vốn gần như toàn LGB/XGB/CatBoost): RGF và YDF.

  --model rgf        Regularized Greedy Forest (rgf_python): tối ưu lại toàn rừng sau mỗi bước, có L2 trên lá
  --model ydf_obl    YDF GBT với split xiên (SPARSE_OBLIQUE): mỗi split là tổ hợp tuyến tính nhiều cột
  --model ydf_rf     YDF Random Forest (không boosting)
  --model ebm        Explainable Boosting Machine (InterpretML, GA2M): tổng hàm 1 cột + --interactions hàm cặp cột

Feature: 21 cột gốc (category -> mã số cho RGF, giữ dạng category cho YDF). Không dùng dữ liệu gốc, không TE.
OOF sạch: RGF số lá cố định (không early stopping); YDF GBT và EBM tự tách 10% / 15% từ dữ liệu fit để dừng
sớm (chỉ là phần train của fold). Test = trung bình 5 mô hình fold.

Chạy: python notebooks/A_alt_trees.py --model rgf --folds 0 --sub 100000 --name A_rgf_try   (đo thời gian)
      python notebooks/A_alt_trees.py --model rgf --name A_rgf_v1
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import CAT_COLS, FEATURES, N_FOLDS, TARGET
from data import load_data
from utils import save_preds

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def fit_predict(model, X_tr, y_tr, X_list, args):
    if model == "rgf":
        from rgf.sklearn import RGFClassifier
        m = RGFClassifier(max_leaf=args.max_leaf, algorithm="RGF_Sib", loss="Log", l2=args.l2,
                          min_samples_leaf=10, learning_rate=0.5, test_interval=args.max_leaf, verbose=False)
        m.fit(X_tr.values.astype(np.float32), y_tr)
        return [m.predict_proba(X.values.astype(np.float32))[:, 1] for X in X_list]
    if model == "ebm":
        from interpret.glassbox import ExplainableBoostingClassifier
        m = ExplainableBoostingClassifier(interactions=args.interactions, max_bins=4096, outer_bags=8,
                                          n_jobs=args.jobs, random_state=args.seed)
        m.fit(X_tr, y_tr)
        return [m.predict_proba(X)[:, 1] for X in X_list]
    import ydf
    d = X_tr.copy()
    d[TARGET] = y_tr
    if model == "ydf_obl":
        m = ydf.GradientBoostedTreesLearner(label=TARGET, num_trees=5000, shrinkage=0.05, max_depth=8,
                                            split_axis="SPARSE_OBLIQUE", sparse_oblique_normalization="MIN_MAX",
                                            validation_ratio=0.1, early_stopping="LOSS_INCREASE",
                                            early_stopping_num_trees_look_ahead=100, num_threads=12,
                                            random_seed=args.seed).train(d, verbose=0)
    else:
        m = ydf.RandomForestLearner(label=TARGET, num_trees=500, max_depth=20, num_threads=12,
                                    random_seed=args.seed).train(d, verbose=0)
    return [np.asarray(m.predict(X)) for X in X_list]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["rgf", "ydf_obl", "ydf_rf", "ebm"], required=True)
    ap.add_argument("--max-leaf", type=int, default=10000)
    ap.add_argument("--l2", type=float, default=0.1)
    ap.add_argument("--interactions", type=int, default=30, help="EBM: số hàm cặp cột")
    ap.add_argument("--jobs", type=int, default=8, help="EBM: số luồng")
    ap.add_argument("--sub", type=int, default=0, help="chỉ fit trên N dòng (đo thời gian), không lưu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(N_FOLDS)), help="chỉ lưu khi chạy đủ 5 fold")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    if not args.name.startswith(("A_", "B_", "C_")):
        ap.error("tên phải dạng <người>_<model>_<phiên bản>")

    train, test = load_data()
    y, fold = train[TARGET].values, train["fold"].values
    X, X_test = train[FEATURES].copy(), test[FEATURES].copy()
    for c in CAT_COLS:
        if args.model == "rgf":
            codes = {v: i for i, v in enumerate(sorted(pd.concat([X[c], X_test[c]]).dropna().unique()))}
            X[c], X_test[c] = X[c].map(codes), X_test[c].map(codes)
    if args.model == "rgf":  # RGF không nhận NaN
        X, X_test = X.fillna(-1), X_test.fillna(-1)

    oof, test_pred = np.zeros(len(y)), np.zeros(len(X_test))
    for k in args.folds:
        tr, va = np.where(fold != k)[0], np.where(fold == k)[0]
        if args.sub:
            tr = np.random.default_rng(42).choice(tr, args.sub, replace=False)
        p_va, p_te = fit_predict(args.model, X.iloc[tr], y[tr], [X.iloc[va], X_test], args)
        oof[va] = p_va
        test_pred += p_te / N_FOLDS
        log(f"fold {k}: AUC = {roc_auc_score(y[va], p_va):.6f}")

    if len(args.folds) == N_FOLDS and not args.sub:
        log(f"CV AUC = {roc_auc_score(y, oof):.6f}")
        save_preds(args.name, oof, test_pred, y)
