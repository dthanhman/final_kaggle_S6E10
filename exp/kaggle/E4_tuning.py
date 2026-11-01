"""E4 (phần chạy Kaggle): Grid Search, Random Search và Bayesian Optimization (Optuna TPE) với cùng ngân sách.

Đẩy lên Kaggle: python exp/tools/kaggle_bundle.py exp/kaggle/E4_tuning.py --slug cs116-e4-tuning --push
Thử nhanh trên máy: SMOKE=1 WORK=<thư mục tạm> python exp/kaggle/E4_tuning.py

Mô hình: LightGBM trên 21 cột gốc (category dạng `category`), như mô hình nền của E3 (CV 0,958595 ở lr 0,05).
- Hàm mục tiêu: AUC trên **fold 0** (validation), early stopping trên 10% tách từ phần train của fold 0.
  learning rate cố định 0,1 khi dò để mỗi lần thử nhanh.
- Ngân sách: BUDGET lần thử cho mỗi phương pháp. Grid: lưới 4 × 3 × 2 trên 3 tham số; Random và Optuna: không gian
  5 tham số, mỗi phương pháp chạy 2 seed để thấy độ dao động.
- Đánh giá cuối: cấu hình tốt nhất của mỗi lần dò + cấu hình mặc định của E3, chạy đủ 5 fold (lr 0,1). Fold 0 đã dùng
  để chọn tham số nên lạc quan; báo riêng trung bình fold 1–4 (chưa từng thấy khi dò).
Ghi: E4_trials.csv (mọi lần thử), E4_final.csv (5 fold), oof_E4_<cấu hình>.npy.
"""
import hashlib
import itertools
import json
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
from config import CAT_COLS, DATA_DIR, FEATURES, TARGET  # noqa: E402

FOLD_MD5 = "9b4ec081a927d21459a607937e6c562c"
SMOKE = os.environ.get("SMOKE") == "1"
N_JOBS = int(os.environ.get("N_JOBS", os.cpu_count() or 4))
ON_KAGGLE = os.path.isdir("/kaggle/input")
WORK = Path("/kaggle/working") if ON_KAGGLE else Path(os.environ.get("WORK", "results/E4_tuning"))
WORK.mkdir(parents=True, exist_ok=True)
T0 = time.time()
BUDGET = 4 if SMOKE else 24
LR = 0.1
DEFAULT = {"num_leaves": 63, "min_child_samples": 20, "colsample_bytree": 0.8, "subsample": 0.8, "reg_lambda": 0.0}
GRID = {"num_leaves": [31, 63, 127, 255], "min_child_samples": [10, 50, 200], "colsample_bytree": [0.5, 0.8]}


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
    train = pd.read_csv(comp / "train.csv")
    train[TARGET] = train[TARGET].astype(int)
    fold = np.zeros(len(train), np.int8)
    for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(train[["id", TARGET]], train[TARGET])):
        fold[va] = k
    assert hashlib.md5(fold.tobytes()).hexdigest() == FOLD_MD5, "fold không khớp data/folds.csv"
    if SMOKE:
        keep = np.random.default_rng(0).random(len(train)) < 0.03
        train, fold = train[keep].reset_index(drop=True), fold[keep]
    X = train[FEATURES].copy()
    for c in CAT_COLS:
        X[c] = X[c].astype("category")
    log(f"train {X.shape}, {N_JOBS} luồng" + (" [SMOKE]" if SMOKE else ""))
    return X, train[TARGET].values, fold


def split(fold, k):
    tr_idx = np.where(fold != k)[0]
    es_idx = np.random.default_rng(42).choice(tr_idx, size=len(tr_idx) // 10, replace=False)
    return np.setdiff1d(tr_idx, es_idx), es_idx


def fit_fold(params, X, y, fold, k):
    fit_idx, es_idx = split(fold, k)
    va = np.where(fold == k)[0]
    m = lgb.LGBMClassifier(n_estimators=60 if SMOKE else 20000, learning_rate=LR, subsample_freq=1,
                           n_jobs=N_JOBS, verbose=-1, random_state=42, **params)
    m.fit(X.iloc[fit_idx], y[fit_idx], eval_set=[(X.iloc[es_idx], y[es_idx])], eval_metric="auc",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    p = m.predict_proba(X.iloc[va])[:, 1]
    return va, p, roc_auc_score(y[va], p), m.best_iteration_


TRIALS = []


def objective(params, method, seed, X, y, fold):
    t0 = time.time()
    _, _, auc, trees = fit_fold(params, X, y, fold, 0)
    TRIALS.append({"method": method, "seed": seed, "trial": sum(1 for t in TRIALS if t["method"] == method and t["seed"] == seed),
                   "val_auc_fold0": auc, "trees": trees, "seconds": round(time.time() - t0, 1), **params})
    pd.DataFrame(TRIALS).to_csv(WORK / "E4_trials.csv", index=False)
    return auc


def sample_random(rng):
    return {"num_leaves": int(np.exp(rng.uniform(np.log(15), np.log(255)))),
            "min_child_samples": int(np.exp(rng.uniform(np.log(5), np.log(300)))),
            "colsample_bytree": round(float(rng.uniform(0.4, 1.0)), 3),
            "subsample": round(float(rng.uniform(0.5, 1.0)), 3),
            "reg_lambda": float(np.exp(rng.uniform(np.log(1e-3), np.log(30))))}


def best_of(method, seed):
    t = pd.DataFrame([r for r in TRIALS if r["method"] == method and r["seed"] == seed])
    r = t.loc[t["val_auc_fold0"].idxmax()]
    return {k: (int(r[k]) if k in ("num_leaves", "min_child_samples") else float(r[k])) for k in DEFAULT}


if __name__ == "__main__":
    import optuna

    X, y, fold = load()
    configs = {"default": DEFAULT}

    # 1. Grid Search: 24 điểm, 2 tham số còn lại giữ mặc định
    grid = list(itertools.product(*GRID.values()))[:BUDGET]
    for vals in grid:
        p = {**DEFAULT, **dict(zip(GRID, vals))}
        objective(p, "grid", 0, X, y, fold)
    configs["grid"] = best_of("grid", 0)
    log(f"grid xong: {max(t['val_auc_fold0'] for t in TRIALS if t['method'] == 'grid'):.6f} {configs['grid']}")

    for seed in [0, 1]:
        # 2. Random Search: BUDGET điểm ngẫu nhiên trong không gian 5 tham số
        rng = np.random.default_rng(seed)
        for _ in range(BUDGET):
            objective(sample_random(rng), "random", seed, X, y, fold)
        configs[f"random_s{seed}"] = best_of("random", seed)
        log(f"random seed {seed}: {max(t['val_auc_fold0'] for t in TRIALS if t['method'] == 'random' and t['seed'] == seed):.6f}")

        # 3. Bayesian Optimization: Optuna TPE, 8 lần đầu ngẫu nhiên rồi theo mô hình xác suất
        def opt_obj(trial):
            p = {"num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
                 "min_child_samples": trial.suggest_int("min_child_samples", 5, 300, log=True),
                 "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                 "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                 "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30, log=True)}
            return objective(p, "optuna", seed, X, y, fold)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=min(8, BUDGET // 2)))
        study.optimize(opt_obj, n_trials=BUDGET)
        configs[f"optuna_s{seed}"] = best_of("optuna", seed)
        log(f"optuna seed {seed}: {study.best_value:.6f}")

    # 4. Đánh giá cuối: 5 fold cho mỗi cấu hình tốt nhất
    rows = []
    for name, p in configs.items():
        oof, aucs, trees, t0 = np.full(len(y), np.nan), [], [], time.time()
        for k in ([0, 1] if SMOKE else range(5)):
            va, pr, auc, nt = fit_fold(p, X, y, fold, k)
            oof[va] = pr
            aucs.append(auc)
            trees.append(nt)
        done = ~np.isnan(oof)
        rows.append({"config": name, "cv_auc": roc_auc_score(y[done], oof[done]), "fold0_tuned": aucs[0],
                     "folds1_4_mean": float(np.mean(aucs[1:])), **{f"fold_{k}": a for k, a in enumerate(aucs)},
                     "trees_mean": float(np.mean(trees)), "minutes": round((time.time() - t0) / 60, 1),
                     "params": json.dumps(p)})
        pd.DataFrame(rows).to_csv(WORK / "E4_final.csv", index=False)
        np.save(WORK / f"oof_E4_{name}.npy", oof)
        log(f"{name:10s} CV {rows[-1]['cv_auc']:.6f} | fold 1–4 {rows[-1]['folds1_4_mean']:.6f}")
    log("xong")
