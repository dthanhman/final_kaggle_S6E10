"""Dựng feature dùng chung (chuyển từ code thi: notebooks/A_lgb_fe.py của repo cũ).

Các bậc cộng dồn (step N gồm mọi bậc <= N), đúng như khi thi:
  0  21 cột gốc, 4 cột category để dạng `category`
  1  + từng chữ số của Age, Flight Distance, 2 cột delay (dấu vết của dữ liệu tổng hợp)
  2  + tần suất mỗi giá trị (đếm trên train + test, không dùng nhãn)
  3  + khoá cho target encoding (cột đơn, cột chia khoảng, cặp cột); TE tính trong fold bằng add_te()
  4  + dữ liệu gốc 129k dòng, chỉ làm feature: teacher LightGBM train trên dữ liệu gốc, tỉ lệ hài lòng theo giá trị / khoá
  5  + tra cứu theo cả bộ điểm dịch vụ trên dữ liệu gốc
  6  + feature nghiệp vụ: tổng / hiệu delay, số điểm 0, thống kê điểm, chỉ số nhóm dịch vụ
teachers=True: thêm 3 teacher khác loại (CatBoost, logistic, LightGBM chỉ dùng điểm + category), đều chỉ thấy nhãn
của dữ liệu gốc. Bộ feature của model thi `A_lgb_v1` = step 4 + teachers.

Mọi feature trong build_static() không dùng nhãn của dữ liệu cuộc thi nên tính một lần trên train + test.
Target encoding dùng nhãn nên chỉ tính trong fold (add_te).
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import TargetEncoder

from config import CAT_COLS, FEATURES, NUM_COLS, SCORE_COLS, TARGET

AGE, FD, DEP, ARR = NUM_COLS
DIGIT_COLS = [AGE, FD, DEP, ARR]
KEY_PAIRS = [  # cặp cột cho target encoding
    ("Class", "Type of Travel"), ("Customer Type", "Type of Travel"), ("Class", "Customer Type"),
    ("Gender", "Class"), ("Inflight wifi service", "Online boarding"),
    ("Inflight wifi service", "Type of Travel"), ("Online boarding", "Type of Travel"),
    ("Seat comfort", "Inflight entertainment"), ("Class", "Inflight wifi service"),
    ("Class", "Online boarding"),
    (FD, "Class"), (FD, "Type of Travel"), (FD, AGE), (AGE, "Class"),
]
ORIG_KEYS = {  # khoá tra cứu tỉ lệ hài lòng trên dữ liệu gốc
    "fd": [FD], "fd_class": [FD, "Class"], "fd_travel": [FD, "Type of Travel"],
    "fd_customer": [FD, "Customer Type"], "fd_age": [FD, AGE],
}
DIGITAL = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
           "Online boarding", "Gate location"]
COMFORT = ["Seat comfort", "Inflight entertainment", "Food and drink", "Cleanliness"]
SERVICE = ["On-board service", "Leg room service", "Baggage handling", "Checkin service"]
TUPLE_KEYS = {"s13": SCORE_COLS, "s13_cat": SCORE_COLS + CAT_COLS,
              "digital": DIGITAL, "comfort": COMFORT, "service": SERVICE}


def codes(df, cols):
    """Mã số nguyên cho tổ hợp giá trị của các cột (NaN là một giá trị riêng)."""
    key = np.zeros(len(df), dtype=np.int64)
    for c in cols:
        cc = pd.factorize(df[c], use_na_sentinel=False)[0]
        key = key * (cc.max() + 1) + cc
    return pd.factorize(key)[0]


def hkey(df, cols):
    return pd.util.hash_pandas_object(df[cols].fillna(-1), index=False).values


def orig_lookup(A, o, cols, prior, smooth=2.0):
    """Tỉ lệ hài lòng (làm mượt về prior) và log số dòng của khoá `cols` trên dữ liệu gốc."""
    g = pd.DataFrame({"k": hkey(o, cols), "y": o[TARGET].values}).groupby("k")["y"].agg(["sum", "count"])
    k = pd.Index(hkey(A, cols))
    s, cnt = g["sum"].reindex(k).fillna(0).values, g["count"].reindex(k).fillna(0).values
    return ((s + smooth * prior) / (cnt + smooth)).astype(np.float32), np.log1p(cnt).astype(np.float32)


def extra_teachers(A, o, X, n_jobs=4):
    """3 teacher khác loại, đều chỉ thấy nhãn của dữ liệu gốc."""
    from catboost import CatBoostClassifier
    from sklearn.compose import make_column_transformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    yo = o[TARGET].values
    cb = CatBoostClassifier(iterations=1500, learning_rate=0.05, depth=6, random_seed=123, verbose=0,
                            thread_count=n_jobs, cat_features=CAT_COLS)
    cb.fit(o[FEATURES], yo)
    X["orig_teacher_cb"] = cb.predict_proba(A[FEATURES])[:, 1].astype(np.float32)

    def filled(df):
        d = df[FEATURES].copy()
        d[ARR] = d[ARR].fillna(d[DEP])
        return d
    lr = make_pipeline(make_column_transformer((OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
                                               remainder=StandardScaler()),
                       LogisticRegression(max_iter=2000))
    lr.fit(filled(o), yo)
    X["orig_teacher_lr"] = lr.predict_proba(filled(A))[:, 1].astype(np.float32)

    cols = SCORE_COLS + CAT_COLS
    Xo = o[cols].copy()
    for c in CAT_COLS:
        Xo[c] = pd.Categorical(Xo[c], categories=X[c].cat.categories)
    sc = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, colsample_bytree=0.7,
                            subsample=0.8, subsample_freq=1, reg_lambda=5.0, random_state=123, verbose=-1,
                            n_jobs=n_jobs)
    sc.fit(Xo, yo)
    X["orig_teacher_scores"] = sc.predict_proba(X[cols])[:, 1].astype(np.float32)


def build_static(train, test, orig=None, step=4, teachers=False, n_jobs=4):
    """Feature không dùng nhãn của cuộc thi. Trả về X_train, X_test, K_train, K_test (K = khoá cho TE)."""
    n = len(train)
    A = pd.concat([train[FEATURES], test[FEATURES]], ignore_index=True)
    X = A.copy()
    for c in CAT_COLS:
        X[c] = X[c].astype("category")
    te_keys = pd.DataFrame(index=A.index)

    if step >= 1:
        for c in DIGIT_COLS:
            v = A[c].fillna(0).astype(np.int64)
            for k in range(4):
                X[f"{c}_d{k}"] = (v // 10 ** k % 10).astype(np.int8)

    if step >= 2:
        for c in NUM_COLS + SCORE_COLS:
            X[f"{c}_freq"] = A[c].map(A[c].value_counts(dropna=False)).astype(np.float32)

    if step >= 3:
        for c in FEATURES:
            te_keys[c] = codes(A, [c])
        bins = {"age_5": A[AGE] // 5, "fd_10": A[FD] // 10, "fd_100": A[FD] // 100,
                "fd_500": A[FD] // 500, "fd_mod1000": A[FD] % 1000, "dep_10": A[DEP] // 10}
        for name, v in bins.items():
            te_keys[name] = v.fillna(-1).astype(np.int64)
        for c1, c2 in KEY_PAIRS:
            te_keys[f"{c1}|{c2}"] = codes(A, [c1, c2])

    if step >= 4:
        o = orig.copy()
        o[TARGET] = o[TARGET].astype(int)
        prior = o[TARGET].mean()
        Xo = o[FEATURES].copy()
        for c in CAT_COLS:
            Xo[c] = pd.Categorical(Xo[c], categories=X[c].cat.categories)
        teacher = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, max_depth=8, min_child_samples=20,
                                     subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0,
                                     reg_alpha=0.05, random_state=123, verbose=-1, n_jobs=n_jobs)
        teacher.fit(Xo, o[TARGET])
        X["orig_teacher"] = teacher.predict_proba(X[FEATURES])[:, 1].astype(np.float32)
        for c in FEATURES:
            X[f"{c}_orig_mean"] = A[c].map(o.groupby(c)[TARGET].mean()).fillna(prior).astype(np.float32)
        for name, cols in ORIG_KEYS.items():
            X[f"orig_{name}_rate"], X[f"orig_{name}_logcnt"] = orig_lookup(A, o, cols, prior)
        if teachers:
            extra_teachers(A, o, X, n_jobs=n_jobs)

    if step >= 5:
        for name, cols in TUPLE_KEYS.items():
            X[f"orig_{name}_rate"], X[f"orig_{name}_logcnt"] = orig_lookup(A, o, cols, prior)
            k = pd.Series(hkey(A, cols))
            X[f"{name}_freq"] = k.map(k.value_counts()).astype(np.float32).values

    if step >= 6:
        S = A[SCORE_COLS]
        zeros = (S == 0).sum(axis=1)
        arr = A[ARR].fillna(A[DEP])
        X["delay_total"] = (A[DEP] + arr).astype(np.float32)
        X["delay_diff"] = (arr - A[DEP]).astype(np.float32)
        X["delay_ratio"] = ((arr + 1) / (A[DEP] + 1)).astype(np.float32)
        X["delay_per_mile"] = (X["delay_total"] / (A[FD] + 1)).astype(np.float32)
        X["zero_count"] = zeros.astype(np.int8)
        X["net_score"] = (((S >= 4).sum(axis=1) - ((S >= 1) & (S <= 2)).sum(axis=1)) / (13 - zeros + 1)).astype(np.float32)
        X["score_mean"] = S.replace(0, np.nan).mean(axis=1).astype(np.float32)
        X["score_std"] = S.replace(0, np.nan).std(axis=1).astype(np.float32)
        X["score_min"] = S.replace(0, np.nan).min(axis=1).astype(np.float32)
        X["score_sum"] = S.sum(axis=1).astype(np.int16)
        for name, cols in (("digital", DIGITAL), ("comfort", COMFORT), ("service", SERVICE)):
            X[f"{name}_mean"] = A[cols].mean(axis=1).astype(np.float32)
        X["wifi_x_boarding"] = ((A["Inflight wifi service"] + 1) * (A["Online boarding"] + 1)).astype(np.int8)
        X["wifi_minus_boarding"] = (A["Inflight wifi service"] - A["Online boarding"]).astype(np.int8)

    X = X.copy()
    return (X.iloc[:n].reset_index(drop=True), X.iloc[n:].reset_index(drop=True),
            te_keys.iloc[:n].reset_index(drop=True), te_keys.iloc[n:].reset_index(drop=True))


def add_te(X_tr, K_tr, y_tr, others):
    """Target encoding fit chỉ trên dòng train của fold (bên trong tự cross-fit 5 phần).

    others: list (X, K) chỉ transform (fold validation, phần dừng sớm, test).
    Trả về [X_tr, *X của others] đã thêm cột te_*.
    """
    out = [X_tr] + [x for x, _ in others]
    if K_tr.shape[1] == 0:
        return out
    for smooth in ("auto", 100.0):
        enc = TargetEncoder(target_type="binary", smooth=smooth, cv=5, shuffle=True, random_state=42)
        parts = [enc.fit_transform(K_tr.values, y_tr)] + [enc.transform(k.values) for _, k in others]
        cols = [f"te_{c}_{smooth}" for c in K_tr.columns]
        for i, p in enumerate(parts):
            out[i] = pd.concat([out[i], pd.DataFrame(p.astype(np.float32), columns=cols, index=out[i].index)], axis=1)
    return out
