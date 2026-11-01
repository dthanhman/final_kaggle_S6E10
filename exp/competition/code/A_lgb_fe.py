"""LightGBM + feature engineering (theo lộ trình Дворкин, references/B_gbdt_features), dùng folds.csv.

Chạy từ thư mục gốc repo:
  python notebooks/A_lgb_fe.py --step 4 --extra teachers --name A_lgb_v1

OOF sạch theo docs/STACKING.md mục 5: early stopping trên 10% tách từ phần train của fold
(--n-rounds N để cố định số cây), target encoding chỉ fit trên phần train đó.
Kết quả cũ (lgb_fe*, early stopping trên fold validation) nằm ở oof/_archive_1310.

Các bước cộng dồn (--step N gồm mọi bước <= N):
  0  21 cột gốc, 4 cột category để LightGBM xử lý native
  1  + từng chữ số của Age, Flight Distance, 2 cột delay (dấu vết của dữ liệu tổng hợp)
  2  + tần suất mỗi giá trị (đếm trên train + test, không dùng nhãn)
  3  + target encoding TÍNH TRONG FOLD: cột đơn, cột chia nhóm, cặp cột
  4  + dữ liệu gốc 129k dòng (data/original/data.csv), chỉ dùng làm feature:
       teacher LightGBM train CHỈ trên dữ liệu gốc, tỉ lệ hài lòng theo từng giá trị / khóa.
       Không ghép dòng gốc vào train, nên fold validation không bị ảnh hưởng.
  5  + tra cứu theo BỘ điểm dịch vụ trên dữ liệu gốc (13 điểm trùng với dữ liệu gốc ở khoảng 41% dòng):
       tỉ lệ hài lòng, số dòng trong dữ liệu gốc, số dòng trong dữ liệu cuộc thi
  6  + feature nghiệp vụ (theo Anhad Mahajan): delay, số điểm bằng 0, chỉ số dịch vụ, thống kê điểm

Kết quả lr 0.05: bước 0-4 đều tăng (0,958854 -> 0,960832); bước 5, 6 không tăng (nhiễu).

Thử riêng trên bước 4 bằng --extra (cần --step >= 4):
  teachers  thêm 3 teacher chỉ train trên dữ liệu gốc: CatBoost, logistic, LightGBM chỉ dùng điểm + category
  route     thống kê theo tuyến (Flight Distance) trên train + test, không dùng nhãn: trung bình của
            điểm / tuổi / delay theo tuyến, độ lệch của khách so với trung bình đó, tỉ lệ category trong tuyến
  arith     tự sinh a+b, a-b, a*b, a/(b+1) cho mọi cặp trong 10 cột số quan trọng (180 cột)
  knn       teacher kNN: tỉ lệ hài lòng của 10 / 50 khách giống nhất trong dữ liệu gốc
  dist      khoảng cách tới khách gần nhất / trung bình 10 khách gần nhất trong dữ liệu gốc
  shift     log(tần suất trong dữ liệu cuộc thi / tần suất trong dữ liệu gốc) theo từng cột và vài tổ hợp
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
from sklearn.preprocessing import TargetEncoder

from config import CAT_COLS, DATA_DIR, FEATURES, N_FOLDS, NUM_COLS, SCORE_COLS, TARGET
from data import load_data
from utils import save_preds

AGE, FD, DEP, ARR = NUM_COLS   # đúng thứ tự trong config (trước 16/10 bị đảo FD <-> AGE)
DIGIT_COLS = [AGE, FD, DEP, ARR]
KEY_PAIRS = [  # cặp cột cho target encoding (theo Дворкин)
    ("Class", "Type of Travel"), ("Customer Type", "Type of Travel"), ("Class", "Customer Type"),
    ("Gender", "Class"), ("Inflight wifi service", "Online boarding"),
    ("Inflight wifi service", "Type of Travel"), ("Online boarding", "Type of Travel"),
    ("Seat comfort", "Inflight entertainment"), ("Class", "Inflight wifi service"),
    ("Class", "Online boarding"),
    (FD, "Class"), (FD, "Type of Travel"), (FD, AGE), (AGE, "Class"),
]
ORIG_KEYS = {  # khóa tra cứu tỉ lệ hài lòng trên dữ liệu gốc
    "fd": [FD], "fd_class": [FD, "Class"], "fd_travel": [FD, "Type of Travel"],
    "fd_customer": [FD, "Customer Type"], "fd_age": [FD, AGE],
}
DIGITAL = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
           "Online boarding", "Gate location"]
COMFORT = ["Seat comfort", "Inflight entertainment", "Food and drink", "Cleanliness"]
SERVICE = ["On-board service", "Leg room service", "Baggage handling", "Checkin service"]
TUPLE_KEYS = {  # bước 5: khóa là cả bộ điểm
    "s13": SCORE_COLS, "s13_cat": SCORE_COLS + CAT_COLS,
    "digital": DIGITAL, "comfort": COMFORT, "service": SERVICE,
}

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def codes(df_all, cols):
    """Mã số nguyên cho tổ hợp giá trị của các cột (NaN là một giá trị riêng)."""
    key = np.zeros(len(df_all), dtype=np.int64)
    for c in cols:
        cc = pd.factorize(df_all[c], use_na_sentinel=False)[0]
        key = key * (cc.max() + 1) + cc
    return pd.factorize(key)[0]


def hkey(df, cols):
    return pd.util.hash_pandas_object(df[cols].fillna(-1), index=False).values


def orig_lookup(A, o, cols, prior, smooth=2.0):
    """Tỉ lệ hài lòng (làm mượt về prior) và log số dòng của khóa `cols` trên dữ liệu gốc."""
    g = pd.DataFrame({"k": hkey(o, cols), "y": o[TARGET].values}).groupby("k")["y"].agg(["sum", "count"])
    k = pd.Index(hkey(A, cols))
    s, cnt = g["sum"].reindex(k).fillna(0).values, g["count"].reindex(k).fillna(0).values
    return ((s + smooth * prior) / (cnt + smooth)).astype(np.float32), np.log1p(cnt).astype(np.float32)


def extra_teachers(A, o, X):
    """Teacher khác loại, đều chỉ thấy nhãn của dữ liệu gốc."""
    from catboost import CatBoostClassifier
    from sklearn.compose import make_column_transformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    yo = o[TARGET].values
    cb = CatBoostClassifier(iterations=1500, learning_rate=0.05, depth=6, random_seed=123, verbose=0,
                            thread_count=12, cat_features=CAT_COLS)
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
                            subsample=0.8, subsample_freq=1, reg_lambda=5.0, random_state=123, verbose=-1)
    sc.fit(Xo, yo)
    X["orig_teacher_scores"] = sc.predict_proba(X[cols])[:, 1].astype(np.float32)


def route_profile(A, X):
    """Thống kê theo tuyến (Flight Distance), chỉ dùng feature, không dùng nhãn."""
    g = A.groupby(FD)
    for c in SCORE_COLS + [AGE, DEP]:
        m = g[c].transform("mean")
        X[f"route_{c}_mean"] = m.astype(np.float32)
        X[f"route_{c}_resid"] = (A[c] - m).astype(np.float32)
    n_route = A[FD].map(A[FD].value_counts())
    for c in CAT_COLS:
        k = pd.Series(hkey(A, [FD, c]))
        X[f"route_{c}_share"] = (k.map(k.value_counts()).values / n_route.values).astype(np.float32)


ARITH_COLS = [FD, AGE, DEP, "Inflight wifi service", "Online boarding", "Inflight entertainment",
              "Seat comfort", "Checkin service", "Leg room service", "Cleanliness"]


def arith(A, X):
    new = {}
    for i, a in enumerate(ARITH_COLS):
        for b in ARITH_COLS[i + 1:]:
            va, vb = A[a].astype(np.float32), A[b].astype(np.float32)
            new[f"{a}+{b}"], new[f"{a}-{b}"] = va + vb, va - vb
            new[f"{a}*{b}"], new[f"{a}/{b}"] = va * vb, va / (vb + 1)
    return pd.concat([X, pd.DataFrame(new, index=X.index)], axis=1)


def knn_space(df):
    """Không gian để đo độ giống nhau giữa hai khách (thang đo chọn tay cho các cột cùng tầm ảnh hưởng)."""
    d = df[SCORE_COLS].astype(np.float32).copy()
    for c in CAT_COLS:
        d = d.join(pd.get_dummies(df[c], prefix=c, dtype=np.float32) * 2)
    d["age"] = df[AGE] / 10
    d["fd"] = df[FD] / 400
    d["dep"] = np.log1p(df[DEP])
    d["arr"] = np.log1p(df[ARR].fillna(df[DEP]))
    return d.astype(np.float32)


def knn_neighbors(A, o, k=50):
    from sklearn.neighbors import NearestNeighbors
    Zo, Za = knn_space(o), knn_space(A)
    Za = Za.reindex(columns=Zo.columns, fill_value=0)
    nn = NearestNeighbors(n_neighbors=k, algorithm="brute", n_jobs=12).fit(Zo.values)
    dist, idx = nn.kneighbors(Za.values)
    return dist.astype(np.float32), idx


def shift(A, o, X):
    """Bộ sinh dữ liệu sinh nhiều / ít một giá trị hơn so với dữ liệu gốc bao nhiêu."""
    keys = {c: [c] for c in FEATURES} | {"s13": SCORE_COLS, "fd_class": [FD, "Class"],
                                         "segment": ["Type of Travel", "Class", "Customer Type"]}
    for name, cols in keys.items():
        ka, ko = pd.Series(hkey(A, cols)), pd.Series(hkey(o, cols))
        fa = ka.map(ka.value_counts(normalize=True)).values
        fo = ka.map(ko.value_counts(normalize=True)).fillna(0).values
        X[f"shift_{name}"] = np.log((fa + 1e-6) / (fo + 1e-6)).astype(np.float32)


def build_static(train, test, orig, step, extras=()):
    """Feature không dùng nhãn của cuộc thi: tính một lần trên train + test."""
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
        # teacher: chỉ thấy nhãn của dữ liệu gốc
        Xo = o[FEATURES].copy()
        for c in CAT_COLS:
            Xo[c] = pd.Categorical(Xo[c], categories=X[c].cat.categories)
        teacher = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, max_depth=8, min_child_samples=20,
                                     subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0,
                                     reg_alpha=0.05, random_state=123, verbose=-1)
        teacher.fit(Xo, o[TARGET])
        X["orig_teacher"] = teacher.predict_proba(X[FEATURES])[:, 1].astype(np.float32)
        # tỉ lệ hài lòng theo từng cột trên dữ liệu gốc
        for c in FEATURES:
            X[f"{c}_orig_mean"] = A[c].map(o.groupby(c)[TARGET].mean()).fillna(prior).astype(np.float32)
        # tỉ lệ (làm mượt) và số lượng theo khóa chính xác trên dữ liệu gốc
        for name, cols in ORIG_KEYS.items():
            X[f"orig_{name}_rate"], X[f"orig_{name}_logcnt"] = orig_lookup(A, o, cols, prior)

    if "teachers" in extras:
        extra_teachers(A, o, X)
    if "route" in extras:
        route_profile(A, X)
    if "arith" in extras:
        X = arith(A, X)
    if "knn" in extras or "dist" in extras:
        dist, idx = knn_neighbors(A, o)
        if "knn" in extras:
            yo = o[TARGET].values[idx]
            X["knn10_rate"] = yo[:, :10].mean(1).astype(np.float32)
            X["knn50_rate"] = yo.mean(1).astype(np.float32)
        if "dist" in extras:
            X["knn_dist1"] = dist[:, 0]
            X["knn_dist10"] = dist[:, :10].mean(1)
    if "shift" in extras:
        shift(A, o, X)

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

    X = X.copy()  # gom các cột vừa thêm lại cho liền bộ nhớ
    return X.iloc[:n].reset_index(drop=True), X.iloc[n:].reset_index(drop=True), \
        te_keys.iloc[:n].reset_index(drop=True), te_keys.iloc[n:].reset_index(drop=True)


def add_te(X_tr, K_tr, y_tr, others):
    """Target encoding fit chỉ trên dòng train của fold (bên trong tự cross-fit 5 phần).
    others: list (X, K) chỉ transform (fold validation, phần dừng sớm, test). Trả về [X_tr, *X của others]."""
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=4, choices=range(7))
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--leaves", type=int, default=63)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--extra", nargs="*", default=[], choices=["teachers", "route", "arith", "knn", "dist", "shift"])
    ap.add_argument("--n-rounds", type=int, default=0,
                    help="cố định số cây, không early stopping (cách b, STACKING.md điều 3); 0 = tách 10%% để dừng sớm")
    ap.add_argument("--w-personal", type=float, default=1.0,
                    help="trọng số dòng Type of Travel = Personal Travel khi fit và dừng sớm (1 = không đổi)")
    ap.add_argument("--name", required=True, help="tên dạng A_<model>_<phiên bản>, lưu oof_<tên>.npy / test_<tên>.npy")
    args = ap.parse_args()
    if args.extra and args.step < 4:
        ap.error("--extra cần --step >= 4")
    if not args.name.startswith(("A_", "B_", "C_")):
        ap.error("tên phải dạng <người>_<model>_<phiên bản>, ví dụ A_lgb_v1 (không dùng blend_/stack_)")

    train, test = load_data()
    y, fold = train[TARGET].values, train["fold"].values
    # Personal Travel: 6% số dòng hài lòng nhưng ~34% phần AUC stack_fw_p247 còn mất
    w = np.where(train["Type of Travel"].values == "Personal Travel", args.w_personal, 1.0)
    orig = pd.read_csv(DATA_DIR / "original" / "data.csv") if args.step >= 4 else None
    X, X_test, K, K_test = build_static(train, test, orig, args.step, args.extra)
    log(f"step {args.step} extra {args.extra}: {X.shape[1]} feature tĩnh + {2 * K.shape[1]} cột target encoding")

    params = dict(objective="binary", metric="auc", learning_rate=args.lr, num_leaves=args.leaves, max_depth=8,
                  min_child_samples=5, lambda_l2=5.0, lambda_l1=0.05, bagging_fraction=0.9, bagging_freq=1,
                  feature_fraction=0.9, seed=args.seed, num_threads=12, verbose=-1)
    oof, test_pred, imp, n_trees = np.zeros(len(y)), np.zeros(len(X_test)), 0, []
    for k in range(N_FOLDS):
        va = fold == k
        tr_idx = np.where(fold != k)[0]
        if args.n_rounds:  # cách b: số cây cố định, train trên toàn bộ 4 fold
            fit_idx = tr_idx
        else:  # cách a: 10% phần train của fold để dừng sớm, fold k không tham gia chọn số cây
            es_idx = np.random.default_rng(42).choice(tr_idx, size=len(tr_idx) // 10, replace=False)
            fit_idx = np.setdiff1d(tr_idx, es_idx)
        # TE fit chỉ trên fit_idx, nên phần dừng sớm cũng "chưa thấy" như fold k
        others = [(X[va], K[va]), (X_test, K_test)] + ([] if args.n_rounds else [(X.iloc[es_idx], K.iloc[es_idx])])
        X_fit, X_va, X_te, *X_es = add_te(X.iloc[fit_idx], K.iloc[fit_idx], y[fit_idx], others)
        if args.n_rounds:
            m = lgb.train(params, lgb.Dataset(X_fit, y[fit_idx], weight=w[fit_idx]), num_boost_round=args.n_rounds)
            best = args.n_rounds
        else:
            m = lgb.train(params, lgb.Dataset(X_fit, y[fit_idx], weight=w[fit_idx]), num_boost_round=20000,
                          valid_sets=[lgb.Dataset(X_es[0], y[es_idx], weight=w[es_idx])],
                          callbacks=[lgb.early_stopping(200, verbose=False)])
            best = m.best_iteration
        n_trees.append(best)
        oof[va] = m.predict(X_va, num_iteration=best)
        test_pred += m.predict(X_te, num_iteration=best) / N_FOLDS
        imp = imp + pd.Series(m.feature_importance("gain"), index=m.feature_name())
        log(f"fold {k}: AUC = {roc_auc_score(y[va], oof[va]):.6f} | {best} cây")

    log(f"step {args.step} lr {args.lr}: CV AUC = {roc_auc_score(y, oof):.6f} | "
        f"số cây trung bình {int(np.mean(n_trees))} (dùng cho --n-rounds)")
    print("Top 15 feature (gain):")
    print((imp / imp.sum()).sort_values(ascending=False).head(15).round(4).to_string())
    save_preds(args.name, oof, test_pred, y)
