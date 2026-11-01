"""Chạy 5-fold CV trên data/folds.csv và ghi bảng kết quả. Mọi thí nghiệm E2–E6 dùng chung file này.

Cách dùng trong một script ở exp/:

    from data import load_data
    from cv import run_cv, log_result, compare

    def fit_predict(X_tr, y_tr, X_va, X_te):
        # MỌI bước học từ dữ liệu (imputer, scaler, target encoding, chọn feature...) fit trên X_tr.
        model = make_pipeline(...).fit(X_tr, y_tr)
        return model.predict_proba(X_va)[:, 1], model.predict_proba(X_te)[:, 1]

    train, test = load_data()
    base = run_cv(fit_predict, train, test, FEATURES, name="E2_lr_base")
    log_result("E2_preprocess", "lr_base", base)

fit_predict chỉ thấy phần train của fold, nên OOF sạch đúng nghĩa (không rò rỉ từ fold đang chấm).
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import N_FOLDS, RESULTS_DIR, TARGET
from utils import save_preds


def run_cv(fit_predict, train, test, features, name=None, verbose=True):
    """Trả về dict: oof, test, fold_auc (list), cv_auc (AUC trên toàn bộ OOF), seconds.

    name khác None thì lưu oof/test preds vào OOF_DIR theo quy ước oof_<name>.npy.
    Test preds = trung bình dự đoán của 5 model fold.
    """
    y = train[TARGET].values
    fold = train["fold"].values
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    fold_auc = []
    t0 = time.time()
    for k in range(N_FOLDS):
        tr, va = fold != k, fold == k
        p_va, p_te = fit_predict(
            train.loc[tr, features].reset_index(drop=True),
            y[tr],
            train.loc[va, features].reset_index(drop=True),
            test[features],
        )
        oof[va] = p_va
        test_pred += np.asarray(p_te, dtype=float) / N_FOLDS
        fold_auc.append(roc_auc_score(y[va], p_va))
        if verbose:
            print(f"  fold {k}: AUC {fold_auc[-1]:.6f} ({time.time() - t0:.0f}s)")
    res = {
        "oof": oof,
        "test": test_pred,
        "fold_auc": fold_auc,
        "cv_auc": roc_auc_score(y, oof),
        "seconds": time.time() - t0,
    }
    if verbose:
        print(f"CV AUC {res['cv_auc']:.6f} | fold mean {np.mean(fold_auc):.6f} ± {np.std(fold_auc):.6f}")
    if name is not None:
        save_preds(name, oof, test_pred, y)
    return res


def compare(base, new):
    """Chênh lệch so với baseline: delta CV và số fold tăng (tiêu chí giữ thay đổi)."""
    d = np.array(new["fold_auc"]) - np.array(base["fold_auc"])
    return {"delta_cv": new["cv_auc"] - base["cv_auc"], "folds_up": int((d > 0).sum()), "delta_fold": d}


def log_result(exp, variant, res, base=None, notes=""):
    """Thêm một dòng vào results/<exp>.csv (ghi đè dòng cùng variant nếu chạy lại)."""
    row = {
        "variant": variant,
        "cv_auc": round(res["cv_auc"], 6),
        "fold_mean": round(float(np.mean(res["fold_auc"])), 6),
        "fold_std": round(float(np.std(res["fold_auc"])), 6),
        **{f"fold_{k}": round(a, 6) for k, a in enumerate(res["fold_auc"])},
        "seconds": round(res["seconds"], 1),
    }
    if base is not None:
        c = compare(base, res)
        row["delta_cv"] = round(c["delta_cv"], 6)
        row["folds_up"] = c["folds_up"]
    row["notes"] = notes

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{exp}.csv"
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(df):
        df = df[df["variant"] != variant]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)
    return row
