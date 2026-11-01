"""E3 (phần chạy Kaggle): so sánh các mô hình phân lớp của Bài 07 trên cùng dữ liệu, cùng 5 fold.

Đẩy lên Kaggle: python exp/tools/kaggle_bundle.py exp/kaggle/E3_models.py --slug cs116-e3-models --push
Thử nhanh trên máy: SMOKE=1 WORK=<thư mục tạm> python exp/kaggle/E3_models.py

Hai dạng dữ liệu đầu vào (rút ra từ E2):
- "cây": 21 cột gốc; category để dạng `category` (LightGBM, CatBoost, HistGB) hoặc mã số (RF, ExtraTrees, XGBoost).
- "tuyến tính": dạng L1 của E2: one-hot 4 category + 13 cột điểm, z-score Age / Flight Distance / log1p delay (MLP).
Không tinh chỉnh tham số (để E4); dùng cấu hình phổ biến. Early stopping (nếu có) dùng 10% tách từ phần train của fold.
Với mỗi mô hình ghi: CV AUC, AUC trên chính dữ liệu train (mẫu 100 nghìn dòng) để đo overfitting, thời gian.
Cuối cùng: learning curve của LightGBM (5% → 100% phần train của fold).
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

FOLD_MD5 = "9b4ec081a927d21459a607937e6c562c"
SMOKE = os.environ.get("SMOKE") == "1"
N_JOBS = int(os.environ.get("N_JOBS", os.cpu_count() or 4))
ON_KAGGLE = os.path.isdir("/kaggle/input")
WORK = Path("/kaggle/working") if ON_KAGGLE else Path(os.environ.get("WORK", "results/E3_models"))
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
    train, test = pd.read_csv(comp / "train.csv"), pd.read_csv(comp / "test.csv")
    train[TARGET] = train[TARGET].astype(int)
    fold = np.zeros(len(train), np.int8)
    for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(train[["id", TARGET]], train[TARGET])):
        fold[va] = k
    assert hashlib.md5(fold.tobytes()).hexdigest() == FOLD_MD5, "fold không khớp data/folds.csv"
    if SMOKE:
        keep = np.random.default_rng(0).random(len(train)) < 0.03
        train, fold = train[keep].reset_index(drop=True), fold[keep]
        test = test.sample(3000, random_state=0).reset_index(drop=True)
    log(f"train {train.shape}, test {test.shape}, {N_JOBS} luồng" + (" [SMOKE]" if SMOKE else ""))
    return train, test, fold


def tree_frame(df, codes):
    X = df[FEATURES].copy()
    for c in CAT_COLS:
        X[c] = X[c].astype("category")
        if codes:
            X[c] = X[c].cat.codes.astype(np.int8)
    if codes:
        X[ARR] = X[ARR].fillna(X[DEP])  # RF / ExtraTrees của sklearn cần không thiếu
    return X


def linear_frame(train, test):
    """Dạng L1 của E2: one-hot category + điểm, z-score cột số. Trả về mảng dense float32."""
    def num(df):
        return pd.DataFrame({AGE: df[AGE], FD: df[FD], "ld": np.log1p(df[DEP]),
                             "la": np.log1p(df[ARR].fillna(df[DEP]))})
    A = pd.concat([train, test], ignore_index=True)
    oh = pd.get_dummies(A[CAT_COLS + SCORE_COLS].astype(str), dtype=np.float32)
    nm = num(A)
    nm = (nm - nm.iloc[:len(train)].mean()) / nm.iloc[:len(train)].std()
    M = np.hstack([oh.values, nm.values.astype(np.float32)])
    return M[:len(train)], M[len(train):]


def split(fold, k):
    tr_idx = np.where(fold != k)[0]
    es_idx = np.random.default_rng(42).choice(tr_idx, size=len(tr_idx) // 10, replace=False)
    return tr_idx, np.setdiff1d(tr_idx, es_idx), es_idx


def take(X, idx):
    return X.iloc[idx] if isinstance(X, pd.DataFrame) else X[idx]


def run(name, family, fit_pred, X, X_test, y, fold, needs_es=False):
    """fit_pred(X_fit, y_fit, X_es, y_es, [X_va, X_te, X_trsample]) -> list dự đoán."""
    folds = [0, 1] if SMOKE else range(5)
    oof, test_pred, aucs, tr_aucs = np.full(len(y), np.nan), np.zeros(len(X_test)), [], []
    t0 = time.time()
    for k in folds:
        va = np.where(fold == k)[0]
        tr_idx, fit_idx, es_idx = split(fold, k)
        if not needs_es:
            fit_idx, es_idx = tr_idx, None
        samp = np.random.default_rng(k).choice(fit_idx, size=min(100_000, len(fit_idx)), replace=False)
        p_va, p_te, p_tr = fit_pred(take(X, fit_idx), y[fit_idx],
                                    None if es_idx is None else take(X, es_idx), None if es_idx is None else y[es_idx],
                                    [take(X, va), X_test, take(X, samp)])
        oof[va] = p_va
        test_pred += p_te / len(folds)
        aucs.append(roc_auc_score(y[va], p_va))
        tr_aucs.append(roc_auc_score(y[samp], p_tr))
    done = ~np.isnan(oof)
    row = {"model": name, "family": family, "cv_auc": round(roc_auc_score(y[done], oof[done]), 6),
           "fold_mean": round(float(np.mean(aucs)), 6), "fold_std": round(float(np.std(aucs)), 6),
           **{f"fold_{k}": round(a, 6) for k, a in enumerate(aucs)},
           "train_auc": round(float(np.mean(tr_aucs)), 6), "minutes": round((time.time() - t0) / 60, 1)}
    row["train_minus_cv"] = round(row["train_auc"] - row["fold_mean"], 6)
    path = WORK / "E3_models.csv"
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    pd.concat([df, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)
    np.save(WORK / f"oof_E3_{name}.npy", oof)
    np.save(WORK / f"test_E3_{name}.npy", test_pred)
    log(f"{name:22s} CV {row['cv_auc']:.6f} | train {row['train_auc']:.6f} | {row['minutes']} phút")


def proba(m, Xs):
    return [m.predict_proba(x)[:, 1] for x in Xs]


# ---------- các mô hình ----------
def fp_rf(Xf, yf, Xe, ye, Xs):
    from sklearn.ensemble import RandomForestClassifier
    m = RandomForestClassifier(n_estimators=40 if SMOKE else 300, min_samples_leaf=5, max_features="sqrt",
                               n_jobs=N_JOBS, random_state=42).fit(Xf, yf)
    return proba(m, Xs)


def fp_et(Xf, yf, Xe, ye, Xs):
    from sklearn.ensemble import ExtraTreesClassifier
    m = ExtraTreesClassifier(n_estimators=40 if SMOKE else 300, min_samples_leaf=5, max_features="sqrt",
                             n_jobs=N_JOBS, random_state=42).fit(Xf, yf)
    return proba(m, Xs)


def fp_hgb(Xf, yf, Xe, ye, Xs):
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(learning_rate=0.1, max_iter=60 if SMOKE else 1000, max_leaf_nodes=63,
                                       categorical_features="from_dtype", early_stopping=True,
                                       validation_fraction=0.1, n_iter_no_change=50, random_state=42).fit(Xf, yf)
    return proba(m, Xs)


def fp_xgb(Xf, yf, Xe, ye, Xs):
    import xgboost as xgb
    m = xgb.XGBClassifier(n_estimators=60 if SMOKE else 5000, learning_rate=0.05, max_depth=7, subsample=0.8,
                          colsample_bytree=0.8, tree_method="hist", eval_metric="auc", early_stopping_rounds=200,
                          n_jobs=N_JOBS, random_state=42).fit(Xf, yf, eval_set=[(Xe, ye)], verbose=False)
    return proba(m, Xs)


def fp_cat(Xf, yf, Xe, ye, Xs):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(iterations=60 if SMOKE else 5000, learning_rate=0.08, depth=7, eval_metric="AUC",
                           od_type="Iter", od_wait=200, cat_features=CAT_COLS, thread_count=N_JOBS,
                           random_seed=42, verbose=0, allow_writing_files=False)
    conv = lambda X: X.astype({c: str for c in CAT_COLS})  # noqa: E731
    m.fit(conv(Xf), yf, eval_set=(conv(Xe), ye), use_best_model=True)
    return [m.predict_proba(conv(x))[:, 1] for x in Xs]


def fp_lgb(Xf, yf, Xe, ye, Xs):
    m = lgb.LGBMClassifier(n_estimators=60 if SMOKE else 20000, learning_rate=0.05, num_leaves=63,
                           min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           n_jobs=N_JOBS, verbose=-1, random_state=42)
    m.fit(Xf, yf, eval_set=[(Xe, ye)], eval_metric="auc", callbacks=[lgb.early_stopping(200, verbose=False)])
    return proba(m, Xs)


def fp_mlp(Xf, yf, Xe, ye, Xs):
    from sklearn.neural_network import MLPClassifier
    m = MLPClassifier(hidden_layer_sizes=(256, 128), batch_size=1024, learning_rate_init=1e-3, alpha=1e-4,
                      max_iter=3 if SMOKE else 40, early_stopping=True, validation_fraction=0.1,
                      n_iter_no_change=5, random_state=42).fit(Xf, yf)
    return proba(m, Xs)


if __name__ == "__main__":
    train, test, fold = load()
    y = train[TARGET].values
    Xc, Xc_te = tree_frame(train, codes=False), tree_frame(test, codes=False)
    Xo, Xo_te = tree_frame(train, codes=True), tree_frame(test, codes=True)
    Xl, Xl_te = linear_frame(train, test)

    run("lightgbm", "boosting", fp_lgb, Xc, Xc_te, y, fold, needs_es=True)
    run("xgboost", "boosting", fp_xgb, Xo, Xo_te, y, fold, needs_es=True)
    run("catboost", "boosting", fp_cat, Xc, Xc_te, y, fold, needs_es=True)
    run("hist_gb", "boosting", fp_hgb, Xc, Xc_te, y, fold)
    run("random_forest", "bagging", fp_rf, Xo, Xo_te, y, fold)
    run("extra_trees", "bagging", fp_et, Xo, Xo_te, y, fold)
    run("mlp", "mạng nơ-ron", fp_mlp, Xl, Xl_te, y, fold)

    # learning curve: LightGBM train trên một phần phần train của fold, chấm trên cả fold validation
    rows = []
    for frac in [0.05, 0.1, 0.25, 0.5, 1.0]:
        for k in ([0, 1] if SMOKE else range(5)):
            va = np.where(fold == k)[0]
            tr_idx, fit_idx, es_idx = split(fold, k)
            sub = fit_idx if frac == 1.0 else np.random.default_rng(k).choice(fit_idx, int(len(fit_idx) * frac), replace=False)
            samp = np.random.default_rng(k).choice(sub, size=min(100_000, len(sub)), replace=False)
            p_va, _, p_tr = fp_lgb(Xc.iloc[sub], y[sub], Xc.iloc[es_idx], y[es_idx], [Xc.iloc[va], Xc_te.head(5), Xc.iloc[samp]])
            rows.append({"frac": frac, "n_train": len(sub), "fold": k, "val_auc": roc_auc_score(y[va], p_va),
                         "train_auc": roc_auc_score(y[samp], p_tr)})
        pd.DataFrame(rows).to_csv(WORK / "E3_learning_curve.csv", index=False)
        log(f"learning curve {frac:.0%}: val {np.mean([r['val_auc'] for r in rows if r['frac'] == frac]):.6f}")
    log("xong")
