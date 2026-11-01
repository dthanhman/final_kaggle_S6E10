"""E2 (phần LightGBM): bậc thang feature và các nhánh tiền xử lý, chạy trên Kaggle CPU.

Đẩy lên Kaggle: python exp/tools/kaggle_bundle.py exp/kaggle/E2_lgb.py --slug cs116-e2-lightgbm-ladder --push
Thử nhanh trên máy: SMOKE=1 python exp/kaggle/E2_lgb.py   (3% số dòng, 2 fold)

Mọi lần chạy dùng 5 fold của data/folds.csv (dựng lại bằng StratifiedKFold seed 42 và kiểm tra md5),
early stopping trên 10% tách từ phần train của fold, target encoding fit trên phần còn lại (OOF sạch, xem E0).
Kết quả ghi dần vào E2_lgb.csv sau mỗi lần chạy.
"""
import hashlib
import os
import sys
import time
from collections import deque
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from config import CAT_COLS, DATA_DIR, FEATURES, NUM_COLS, SCORE_COLS, TARGET  # noqa: E402
from features import add_te, build_static  # noqa: E402

FOLD_MD5 = "9b4ec081a927d21459a607937e6c562c"
SMOKE = os.environ.get("SMOKE") == "1"
N_JOBS = int(os.environ.get("N_JOBS", os.cpu_count() or 4))
ON_KAGGLE = os.path.isdir("/kaggle/input")
WORK = Path("/kaggle/working") if ON_KAGGLE else Path(os.environ.get("WORK", "results/E2_lgb"))  # local: chạy từ gốc repo
WORK.mkdir(parents=True, exist_ok=True)
T0 = time.time()
AGE, FD, DEP, ARR = NUM_COLS


def log(msg):
    print(f"[{(time.time() - T0) / 60:6.1f} phút] {msg}", flush=True)


def find_dir(name, root="/kaggle/input", max_depth=5):
    q = deque([(root, 0)])
    while q:
        d, depth = q.popleft()
        try:
            entries = sorted(os.scandir(d), key=lambda e: e.name)
        except OSError:
            continue
        for e in entries:
            if e.is_dir():
                if e.name == name:
                    return Path(e.path)
                if depth < max_depth:
                    q.append((e.path, depth + 1))
    return None


def load():
    comp = find_dir("playground-series-s6e10") if ON_KAGGLE else DATA_DIR
    orig_dir = find_dir("binary-aviation-satisfaction-129k") if ON_KAGGLE else DATA_DIR / "original"
    train, test = pd.read_csv(comp / "train.csv"), pd.read_csv(comp / "test.csv")
    orig = pd.read_csv(orig_dir / "data.csv")
    train[TARGET] = train[TARGET].astype(int)
    fold = np.zeros(len(train), np.int8)
    for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(train[["id", TARGET]], train[TARGET])):
        fold[va] = k
    assert hashlib.md5(fold.tobytes()).hexdigest() == FOLD_MD5, "fold không khớp data/folds.csv"
    if SMOKE:
        keep = np.random.default_rng(0).random(len(train)) < 0.03
        train, fold = train[keep].reset_index(drop=True), fold[keep]
        test = test.sample(3000, random_state=0).reset_index(drop=True)
    log(f"train {train.shape}, test {test.shape}, gốc {orig.shape}, {N_JOBS} luồng" + (" [SMOKE]" if SMOKE else ""))
    return train, test, orig, fold


def params(lr):
    # đúng tham số của model thi A_lgb (A_lgb_fe.py), trừ learning rate
    return dict(objective="binary", metric="auc", learning_rate=lr, num_leaves=63, max_depth=8,
                min_child_samples=5, lambda_l2=5.0, lambda_l1=0.05, bagging_fraction=0.9, bagging_freq=1,
                feature_fraction=0.9, seed=42, num_threads=N_JOBS, verbose=-1)


def run(X, X_test, K, K_test, y, fold, lr=0.05):
    folds = [0, 1] if SMOKE else range(5)
    oof, test_pred, aucs, trees = np.full(len(y), np.nan), np.zeros(len(X_test)), [], []
    t0 = time.time()
    for k in folds:
        va = fold == k
        tr_idx = np.where(fold != k)[0]
        es_idx = np.random.default_rng(42).choice(tr_idx, size=len(tr_idx) // 10, replace=False)
        fit_idx = np.setdiff1d(tr_idx, es_idx)
        X_fit, X_va, X_te, X_es = add_te(X.iloc[fit_idx], K.iloc[fit_idx], y[fit_idx],
                                         [(X[va], K[va]), (X_test, K_test), (X.iloc[es_idx], K.iloc[es_idx])])
        m = lgb.train(params(lr), lgb.Dataset(X_fit, y[fit_idx]), num_boost_round=50 if SMOKE else 20000,
                      valid_sets=[lgb.Dataset(X_es, y[es_idx])], callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va] = m.predict(X_va, num_iteration=m.best_iteration)
        test_pred += m.predict(X_te, num_iteration=m.best_iteration) / len(folds)
        aucs.append(roc_auc_score(y[va], oof[va]))
        trees.append(m.best_iteration)
    done = ~np.isnan(oof)
    return {"oof": oof, "test": test_pred, "fold_auc": aucs, "cv_auc": roc_auc_score(y[done], oof[done]),
            "trees": trees, "seconds": time.time() - t0, "n_feat": X_fit.shape[1]}


RESULTS = WORK / "E2_lgb.csv"


def record(group, variant, res, base=None, notes=""):
    row = {"group": group, "variant": variant, "cv_auc": round(res["cv_auc"], 6),
           "fold_mean": round(float(np.mean(res["fold_auc"])), 6), "fold_std": round(float(np.std(res["fold_auc"])), 6),
           **{f"fold_{k}": round(a, 6) for k, a in enumerate(res["fold_auc"])},
           "n_features": res["n_feat"], "trees_mean": int(np.mean(res["trees"])), "minutes": round(res["seconds"] / 60, 1)}
    if base is not None:
        d = np.array(res["fold_auc"]) - np.array(base["fold_auc"])
        row["delta_cv"], row["folds_up"] = round(res["cv_auc"] - base["cv_auc"], 6), int((d > 0).sum())
    row["notes"] = notes
    df = pd.read_csv(RESULTS) if RESULTS.exists() else pd.DataFrame()
    pd.concat([df, pd.DataFrame([row])], ignore_index=True).to_csv(RESULTS, index=False)
    log(f"{group:8s} {variant:28s} CV {res['cv_auc']:.6f}  Δ {row.get('delta_cv', 0):+.6f}  "
        f"{row['n_features']} feature, {row['trees_mean']} cây, {row['minutes']} phút")


def raw_frame(df, mode, ref):
    """Nhánh tiền xử lý trên 21 cột gốc (bậc 0). Thống kê (median, q99, mean, std) lấy từ ref = train."""
    X = df[FEATURES].copy()
    for c in CAT_COLS:
        X[c] = X[c].astype("category")
    if mode == "missing_fill_dep":
        X[ARR] = X[ARR].fillna(X[DEP])
    elif mode == "missing_median":
        X[ARR] = X[ARR].fillna(ref[ARR].median())
    elif mode == "missing_flag":
        X["arr_missing"] = X[ARR].isna().astype(np.int8)
    elif mode == "delay_log1p":
        X[DEP], X[ARR] = np.log1p(X[DEP]), np.log1p(X[ARR])
    elif mode == "delay_clip_q99":
        for c in (DEP, ARR):
            X[c] = X[c].clip(upper=ref[c].quantile(0.99))
    elif mode == "scale_standard":
        for c in NUM_COLS + SCORE_COLS:
            X[c] = (X[c] - ref[c].mean()) / ref[c].std()
    elif mode == "cat_ordinal":
        for c in CAT_COLS:
            X[c] = X[c].cat.codes.astype(np.int8)
    elif mode == "cat_onehot":
        X = pd.concat([X.drop(columns=CAT_COLS), pd.get_dummies(df[CAT_COLS], dtype=np.int8)], axis=1)
    elif mode == "scores_category":
        for c in SCORE_COLS:
            X[c] = X[c].astype("category")
    return X


if __name__ == "__main__":
    train, test, orig, fold = load()
    y = train[TARGET].values
    empty_tr, empty_te = pd.DataFrame(index=range(len(train))), pd.DataFrame(index=range(len(test)))

    # 1. Bậc thang feature (lr 0,05): mỗi bậc cộng thêm một nhóm feature
    ladder = {}
    for step, teachers, name in [(0, False, "0 cột gốc"), (1, False, "1 + chữ số"), (2, False, "2 + tần suất"),
                                 (3, False, "3 + TE trong fold"), (4, False, "4 + dữ liệu gốc"),
                                 (4, True, "4t + 3 teacher gốc"), (5, False, "5 + tra cứu bộ điểm"),
                                 (6, False, "6 + feature nghiệp vụ")]:
        X, X_test, K, K_test = build_static(train, test, orig, step=step, teachers=teachers, n_jobs=N_JOBS)
        res = run(X, X_test, K, K_test, y, fold, lr=0.05)
        prev = ladder.get("4" if step == 5 else str(step - 1)) if step > 0 else None
        if teachers:
            prev = ladder["4"]
        record("ladder", name, res, base=prev, notes="so với bậc liền trước")
        ladder["4t" if teachers else str(step)] = res
        if step == 0:
            np.save(WORK / "oof_E2_lgb_raw.npy", res["oof"])
            np.save(WORK / "test_E2_lgb_raw.npy", res["test"])

    # 2. Bậc cuối với lr 0,02 = công thức model thi A_lgb (CV khi thi 0,960893)
    X, X_test, K, K_test = build_static(train, test, orig, step=4, teachers=True, n_jobs=N_JOBS)
    res = run(X, X_test, K, K_test, y, fold, lr=0.02)
    record("final", "4t, lr 0,02 (= A_lgb khi thi)", res, base=ladder["4t"], notes="so với 4t lr 0,05; khi thi 0,960893")
    np.save(WORK / "oof_E2_lgb_final.npy", res["oof"])
    np.save(WORK / "test_E2_lgb_final.npy", res["test"])
    del X, X_test, K, K_test

    # 3. Nhánh tiền xử lý trên 21 cột gốc: cây có nhạy với các phép biến đổi này không?
    base = ladder["0"]
    for mode in ["missing_fill_dep", "missing_median", "missing_flag", "delay_log1p", "delay_clip_q99",
                 "scale_standard", "cat_ordinal", "cat_onehot", "scores_category"]:
        res = run(raw_frame(train, mode, train), raw_frame(test, mode, train), empty_tr, empty_te, y, fold, lr=0.05)
        record("branch", mode, res, base=base, notes="so với bậc 0 (NaN để nguyên, category native)")

    log("xong")
