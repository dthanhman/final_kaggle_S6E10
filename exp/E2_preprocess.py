# %% [markdown]
# # E2. Tiền xử lý, tạo và chọn đặc trưng
#
# Bước **Data preprocessing** của ML pipeline (Bài 04): xử lý giá trị thiếu, ngoại lệ, biến đổi và mã hoá đặc trưng,
# tạo đặc trưng mới, chọn đặc trưng. Mỗi lựa chọn được kiểm chứng bằng CV 5 fold trên `data/folds.csv` (quy ước ở E0),
# xuất phát từ các phát hiện ở E1.
#
# > **Trong cuộc thi.** Model LightGBM đầu tiên của nhóm, `A_lgb` (14/10, CV 0,96089), được xây theo kiểu bậc thang:
# > mỗi bước thêm một nhóm feature, giữ bước nào làm CV tăng (`src/features.py`, chuyển từ code thi). Model tuyến
# > tính `A_lr` (15/10, CV 0,95974) dùng lại các feature đó dưới dạng one-hot thưa. Ngày 18/10 nhóm thử ghép thẳng
# > 129 nghìn dòng dữ liệu gốc vào phần train của fold: chỉ +1e-4 (`A_lgb_origrow` 0,959729 so với 0,959630).
# > Notebook này dựng lại hai bậc thang đó một cách có hệ thống và thêm các so sánh theo Bài 04.
#
# Nội dung:
# 1. Logistic Regression: bậc thang từ baseline E0 tới công thức `A_lr`
# 2. Các nhánh tiền xử lý trên LR: giá trị thiếu, biến đổi delay, chuẩn hoá
# 3. Tạo đặc trưng theo Bài 04 / 06: tổng hợp điểm, cụm KMeans, PCA
# 4. Chọn đặc trưng: filter, wrapper, embedded
# 5. LightGBM (chạy trên Kaggle): cùng bậc thang và cùng các nhánh, so với LR
# 6. Tóm tắt

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))

import pickle
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (FunctionTransformer, MinMaxScaler, OneHotEncoder, RobustScaler,
                                   StandardScaler)

from config import CAT_COLS, DATA_DIR, FEATURES, N_FOLDS, NUM_COLS, OOF_DIR, RESULTS_DIR, SCORE_COLS, TARGET
from cv import compare, log_result, run_cv
from data import load_data
from features import DIGIT_COLS, KEY_PAIRS, add_te, build_static
import plot as P

pd.set_option("display.width", 200)
warnings.filterwarnings("ignore", category=ConvergenceWarning)  # saga 100 vòng (mục 4.3)
train, test = load_data()
y, fold = train[TARGET].values, train["fold"].values
AGE, FD, DEP, ARR = NUM_COLS
EXP = "E2_lr"
OOF_DIR.mkdir(exist_ok=True)


def cached(variant, fn, base=None, notes=""):
    """Chạy fn() -> res (dict của run_cv) rồi ghi results/E2_lr.csv; nếu đã có kết quả và OOF thì đọc lại."""
    path, oof_path = RESULTS_DIR / f"{EXP}.csv", OOF_DIR / f"oof_E2_lr_{variant}.npy"
    if path.exists() and oof_path.exists() and variant in set(pd.read_csv(path)["variant"]):
        oof = np.load(oof_path)
        row = pd.read_csv(path).set_index("variant").loc[variant]
        res = {"oof": oof, "fold_auc": [roc_auc_score(y[fold == k], oof[fold == k]) for k in range(N_FOLDS)],
               "cv_auc": roc_auc_score(y, oof), "seconds": row["seconds"]}
    else:
        res = fn()
        np.save(oof_path, res["oof"])
        if penalty_note := res.get("nonzero"):
            notes = f"{notes} hệ số khác 0 (fold cuối): {penalty_note}/{res['n_features']}".strip()
    if not notes and path.exists() and variant in set(pd.read_csv(path)["variant"]):
        old = pd.read_csv(path).set_index("variant").loc[variant, "notes"]
        notes = old if isinstance(old, str) else ""
    log_result(EXP, variant, res, base=base, notes=notes)
    c = f"  Δ {compare(base, res)['delta_cv']:+.6f} ({compare(base, res)['folds_up']}/5 fold tăng)" if base else ""
    print(f"{variant:32s} CV {res['cv_auc']:.6f}{c}")
    return res


# %% [markdown]
# ## 1. Logistic Regression: bậc thang đặc trưng
# LR là mô hình tuyến tính: mỗi đặc trưng chỉ đóng góp theo một hệ số, nên mọi quan hệ phi tuyến, mọi tương tác và
# "điểm 0 khác biệt" (E1) đều phải được **mã hoá sẵn** thành đặc trưng. Đây là chỗ tiền xử lý quan trọng nhất.
#
# Các khối đặc trưng (khối không dùng nhãn tính một lần trên train + test; khối dùng nhãn tính trong fold):
#
# | Khối | Nội dung | Dùng nhãn? |
# | --- | --- | --- |
# | `cat` | one-hot 4 cột category | không |
# | `num` | Age, Flight Distance, log1p hai cột delay (delay đến thiếu điền bằng delay đi), cờ thiếu; z-score | không |
# | `score_num` | 13 cột điểm để dạng số, z-score | không |
# | `score_oh` | 13 cột điểm one-hot (mỗi mức điểm một cột, kể cả điểm 0) | không |
# | `values` | one-hot từng giá trị Age, Flight Distance, delay và các khoảng chia (tuổi 5 năm, quãng đường 10/100/500…) | không |
# | `pairs` | one-hot 14 cặp cột (ví dụ `Class` × `Type of Travel`) | không |
# | `digits` | one-hot từng chữ số của Age, Flight Distance, delay; log tần suất mỗi giá trị | không |
# | `te` | target encoding của mọi khoá trên (2 mức làm trơn), fit trong fold rồi lấy logit | **có, trong fold** |
# | `orig` | 4 teacher train trên dữ liệu gốc + tỉ lệ hài lòng theo giá trị / khoá trên dữ liệu gốc (logit) | chỉ nhãn dữ liệu gốc |
#
# Giá trị one-hot xuất hiện dưới 20 lần gộp chung một cột. Mọi bậc dùng LR với C = 0,05 (như `A_lr`).

# %%
cache = DATA_DIR / "cache" / "static_step4_teachers.pkl"
if cache.exists():
    X, X_test, K, K_test = pickle.loads(cache.read_bytes())
else:
    orig = pd.read_csv(DATA_DIR / "original" / "data.csv")
    X, X_test, K, K_test = build_static(train, test, orig, step=4, teachers=True, n_jobs=6)
    cache.parent.mkdir(exist_ok=True)
    cache.write_bytes(pickle.dumps((X, X_test, K, K_test)))
n = len(X)
print(f"{X.shape[1]} cột tĩnh, {K.shape[1]} khoá cho target encoding")


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def onehot(cols_tr, cols_te):
    enc = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, dtype=np.float32)
    enc.fit(pd.concat([cols_tr, cols_te], ignore_index=True).values)
    return enc.transform(cols_tr.values).tocsr(), enc.transform(cols_te.values).tocsr()


def dense(D_tr, D_te):
    sc = StandardScaler().fit(pd.concat([D_tr, D_te], ignore_index=True))
    return sp.csr_matrix(sc.transform(D_tr)), sp.csr_matrix(sc.transform(D_te))


def num_frame(Z):
    return pd.DataFrame({AGE: Z[AGE], FD: Z[FD], "log_dep": np.log1p(Z[DEP]),
                         "log_arr": np.log1p(Z[ARR].fillna(Z[DEP])), "arr_missing": Z[ARR].isna()}).astype(float)


value_keys = [AGE, FD, DEP, ARR, "age_5", "fd_10", "fd_100", "fd_500", "fd_mod1000", "dep_10"]
pair_keys = [f"{a}|{b}" for a, b in KEY_PAIRS]
digit_cols = [f"{c}_d{k}" for c in DIGIT_COLS for k in range(4)]
freq_cols = [c for c in X.columns if c.endswith("_freq")]
orig_cols = [c for c in X.columns if "teacher" in c or c.endswith("_orig_mean") or c.endswith("_rate")]
logcnt_cols = [c for c in X.columns if c.endswith("_logcnt")]

B = {
    "cat": onehot(K[CAT_COLS], K_test[CAT_COLS]),
    "num": dense(num_frame(X), num_frame(X_test)),
    "score_num": dense(X[SCORE_COLS].astype(float), X_test[SCORE_COLS].astype(float)),
    "score_oh": onehot(K[SCORE_COLS], K_test[SCORE_COLS]),
    "values": onehot(K[value_keys], K_test[value_keys]),
    "pairs": onehot(K[pair_keys], K_test[pair_keys]),
}
d_tr, d_te = onehot(X[digit_cols], X_test[digit_cols])
f_tr, f_te = dense(np.log1p(X[freq_cols]), np.log1p(X_test[freq_cols]))
B["digits"] = (sp.hstack([d_tr, f_tr]).tocsr(), sp.hstack([d_te, f_te]).tocsr())
o_tr = pd.concat([X[orig_cols].apply(logit), X[logcnt_cols]], axis=1)
o_te = pd.concat([X_test[orig_cols].apply(logit), X_test[logcnt_cols]], axis=1)
B["orig"] = dense(o_tr, o_te)
print(pd.Series({k: v[0].shape[1] for k, v in B.items()}, name="số cột").to_frame().T)

# %%
EMPTY = pd.DataFrame(index=range(n)), pd.DataFrame(index=range(len(X_test)))


def lr_cv(blocks, te=False, C=0.05, extra=None, penalty="l2"):
    """CV 5 fold cho LR trên các khối; te=True thêm target encoding tính trong fold; extra(tr, va) -> khối dense
    tính trong fold (KMeans, PCA...)."""
    oof, t0, aucs = np.zeros(n), time.time(), []
    for k in range(N_FOLDS):
        tr, va = np.where(fold != k)[0], np.where(fold == k)[0]
        mats = [[B[b][0][idx] for b in blocks] for idx in (tr, va)]
        if te:
            T_tr, T_va = add_te(EMPTY[0].iloc[tr], K.iloc[tr], y[tr], [(EMPTY[0].iloc[va], K.iloc[va])])
            T_tr, T_va = logit(T_tr.values), logit(T_va.values)
            sc = StandardScaler().fit(T_tr)
            mats[0].append(sp.csr_matrix(sc.transform(T_tr)))
            mats[1].append(sp.csr_matrix(sc.transform(T_va)))
        if extra is not None:
            E_tr, E_va = extra(tr, va)
            mats[0].append(sp.csr_matrix(E_tr))
            mats[1].append(sp.csr_matrix(E_va))
        M_tr, M_va = sp.hstack(mats[0]).tocsr(), sp.hstack(mats[1]).tocsr()
        if penalty == "l1":  # LASSO: scikit-learn >= 1.8 dùng l1_ratio = 1
            m = LogisticRegression(C=C, l1_ratio=1.0, solver="saga", max_iter=100, tol=1e-3).fit(M_tr, y[tr])
        else:
            m = LogisticRegression(C=C, max_iter=3000, tol=1e-5).fit(M_tr, y[tr])
        oof[va] = m.decision_function(M_va)
        aucs.append(roc_auc_score(y[va], oof[va]))
    res = {"oof": oof, "fold_auc": aucs, "cv_auc": roc_auc_score(y, oof), "seconds": time.time() - t0,
           "n_features": M_tr.shape[1], "nonzero": int((np.abs(m.coef_) > 0).sum()) if penalty == "l1" else None}
    return res


ladder = [
    ("L0 baseline (điểm dạng số)", ["cat", "num", "score_num"], False),
    ("L1 điểm one-hot", ["cat", "num", "score_oh"], False),
    ("L2 + one-hot giá trị, khoảng", ["cat", "num", "score_oh", "values"], False),
    ("L3 + cặp cột", ["cat", "num", "score_oh", "values", "pairs"], False),
    ("L4 + chữ số, tần suất", ["cat", "num", "score_oh", "values", "pairs", "digits"], False),
    ("L5 + target encoding", ["cat", "num", "score_oh", "values", "pairs", "digits"], True),
    ("L6 + dữ liệu gốc (= A_lr khi thi)", ["cat", "num", "score_oh", "values", "pairs", "digits", "orig"], True),
]
LR = {}
prev = None
for name, blocks, te in ladder:
    LR[name] = cached(name, lambda b=blocks, t=te: lr_cv(b, te=t), base=prev, notes="so với bậc liền trước")
    prev = LR[name]

# %% [markdown]
# **Nhận xét mục 1**
# - Tổng cộng LR đi từ 0,9217 lên **0,9597 (+0,038)**, mọi bậc giữ lại đều tăng ở cả 5 fold.
# - **Bậc lớn nhất là L1 (+0,0270): mã hoá điểm dịch vụ thành one-hot.** Đúng như E1: điểm 0 ("không áp dụng") và
#   quan hệ không đơn điệu chỉ học được khi mỗi mức điểm có hệ số riêng. Một thay đổi mã hoá, không thêm thông tin
#   nào, mà tăng gần 6 lần tổng các bậc L2–L5 cộng lại (0,0046).
# - L2 one-hot giá trị và khoảng (+0,0021) bắt được quan hệ hình chữ ∩ của tuổi; L3 cặp cột (+0,0023) bắt được tương
#   tác `Type of Travel` × `Class` (E1 mục 5). Mô hình tuyến tính cần feature ghép để học tương tác.
# - L4 chữ số, tần suất: +3e-6, chỉ 3/5 fold, **không có tác dụng với LR** (ngược với LightGBM, mục 5).
# - L5 target encoding trong fold: +0,00025. L6 dữ liệu gốc: **+0,0064**, bậc lớn thứ hai: các teacher học trên dữ liệu
#   gốc mang sẵn quan hệ phi tuyến mà LR không tự học được.
# - **Tái lập:** L6 đạt 0,959727, so với `A_lr` khi thi 0,959737 (chênh 1e-5, tương quan thứ hạng giữa hai bộ dự đoán
#   0,9998). Sai khác còn lại do phiên bản scikit-learn.

# %% [markdown]
# ## 2. Các nhánh tiền xử lý trên LR
# Giữ nguyên baseline L0, mỗi lần chỉ đổi **một** bước. Mọi bước học từ dữ liệu (imputer, scaler) nằm trong pipeline
# nên chỉ fit trên phần train của fold.
#
# ### 2.1 Giá trị thiếu (`Arrival Delay`, 0,04% số dòng)

# %%
def l0_pipeline(impute="fill_dep", delay="log1p", scaler=StandardScaler):
    def prep(Z):
        Z = Z.copy()
        if impute == "fill_dep":
            Z[ARR] = Z[ARR].fillna(Z[DEP])
        if impute == "flag":
            Z["arr_missing"] = Z[ARR].isna().astype(float)
            Z[ARR] = Z[ARR].fillna(Z[DEP])
        for c in (DEP, ARR):
            if delay == "log1p":
                Z[c] = np.log1p(Z[c])
            elif delay == "has_delay":
                Z[c + "_has"] = (Z[c] > 0).astype(float)
                Z[c] = np.log1p(Z[c])
        return Z

    num = NUM_COLS + SCORE_COLS + (["arr_missing"] if impute == "flag" else []) + \
        ([DEP + "_has", ARR + "_has"] if delay == "has_delay" else [])
    steps = []
    if impute == "median":
        steps.append(SimpleImputer(strategy="median"))
    if impute == "knn":
        steps += [StandardScaler(), KNNImputer(n_neighbors=5)]
    if scaler is not None:
        steps.append(scaler())
    num_pipe = make_pipeline(*steps) if steps else "passthrough"
    pre = ColumnTransformer([("num", num_pipe, num), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)])
    return make_pipeline(FunctionTransformer(prep), pre, LogisticRegression(C=0.05, max_iter=3000, tol=1e-5))


class ClipQ99(BaseEstimator, TransformerMixin):
    """Cắt hai cột delay ở phân vị 99 (ngưỡng học từ phần train của fold), sau đó log1p."""

    def fit(self, Z, y=None):
        self.q = {c: Z[c].quantile(0.99) for c in (DEP, ARR)}
        return self

    def transform(self, Z):
        Z = Z.copy()
        for c in (DEP, ARR):
            Z[c] = np.log1p(Z[c].clip(upper=self.q[c]))
        return Z


def fp_from(make):
    def fp(X_tr, y_tr, X_va, X_te):
        m = make().fit(X_tr, y_tr)
        return m.decision_function(X_va), m.decision_function(X_te)
    return fp


def raw_cv(make):
    res = run_cv(fp_from(make), train, test.head(10), FEATURES, verbose=False)
    return res


base0 = cached("nhánh: gốc (điền bằng delay đi, log1p, z-score)", lambda: raw_cv(lambda: l0_pipeline()))
for name, kw in [("thiếu: điền median", dict(impute="median")),
                 ("thiếu: điền delay đi + cờ thiếu", dict(impute="flag")),
                 ("thiếu: KNNImputer (k=5)", dict(impute="knn"))]:
    cached(name, lambda kw=kw: raw_cv(lambda: l0_pipeline(**kw)), base=base0)

# %% [markdown]
# ### 2.2 Ngoại lệ và biến đổi hai cột delay

# %%
for name, make in [
    ("delay: để nguyên phút", lambda: l0_pipeline(delay="raw")),
    ("delay: cắt q99 rồi log1p", lambda: make_pipeline(
        FunctionTransformer(lambda Z: Z.assign(**{ARR: Z[ARR].fillna(Z[DEP])})), ClipQ99(),
        ColumnTransformer([("num", StandardScaler(), NUM_COLS + SCORE_COLS),
                           ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)]),
        LogisticRegression(C=0.05, max_iter=3000, tol=1e-5))),
    ("delay: log1p + cờ có trễ", lambda: l0_pipeline(delay="has_delay")),
]:
    cached(name, lambda make=make: raw_cv(make), base=base0)

# %% [markdown]
# ### 2.3 Chuẩn hoá cột số (Bài 04: Min-Max, Standard, Robust)

# %%
for name, sc in [("scale: không chuẩn hoá", None), ("scale: Min-Max", MinMaxScaler), ("scale: Robust", RobustScaler)]:
    cached(name, lambda sc=sc: raw_cv(lambda: l0_pipeline(scaler=sc)), base=base0)

# %% [markdown]
# **Nhận xét mục 2**
# - **Giá trị thiếu:** điền median, điền bằng delay đi, thêm cờ thiếu hay KNNImputer đều cho cùng kết quả (±1e-6):
#   292 dòng (0,04%) quá ít để ảnh hưởng. Giữ cách đơn giản nhất có cơ sở từ E1 (điền bằng delay đi).
# - **Biến đổi delay:** để nguyên phút làm giảm 0,00017 (0/5 fold tăng), log1p giảm ảnh hưởng của đuôi dài lên hệ số
#   tuyến tính. Sau khi đã log1p, cắt ở phân vị 99 (+4e-6) hay thêm cờ "có trễ" (+8e-6) không thêm gì.
# - **Chuẩn hoá:** z-score, Min-Max, Robust hay không chuẩn hoá cho cùng AUC (±3e-6) vì mô hình hội tụ về cùng nghiệm,
#   nhưng **không chuẩn hoá thì thời gian huấn luyện tăng từ khoảng 21 giây lên 898 giây (×43)**: thuật toán tối ưu
#   (L-BFGS) hội tụ chậm khi các cột có thang đo chênh lệch. Đúng ý Bài 04: chuẩn hoá giúp "học nhanh hơn".
# - Kết luận: với dữ liệu này, các bước tiền xử lý "kinh điển" chỉ có tác dụng khi chúng đổi được dạng quan hệ
#   (log1p), còn điền thiếu và cắt ngoại lệ không đáng kể.

# %% [markdown]
# ## 3. Tạo đặc trưng theo Bài 04 / 06: tổng hợp điểm, cụm KMeans, PCA
# Ở E1, riêng PC1 đạt AUC 0,880 và mã cụm KMeans đạt 0,854, cao hơn mọi cột đơn. Thử thêm vào L1 (LR còn đơn giản)
# và L6 (công thức đầy đủ). KMeans và PCA fit trên phần train của fold.

# %%
S_all = X[SCORE_COLS].astype(float)
Zs_all = np.column_stack([StandardScaler().fit_transform(S_all), StandardScaler().fit_transform(num_frame(X))])


def aggregates(idx):
    S = S_all.iloc[idx]
    nz = S.replace(0, np.nan)
    return np.column_stack([nz.mean(1).fillna(0), nz.std(1).fillna(0), nz.min(1).fillna(0), (S == 0).sum(1), S.sum(1)])


def extra_agg(tr, va):
    sc = StandardScaler().fit(aggregates(tr))
    return sc.transform(aggregates(tr)), sc.transform(aggregates(va))


def extra_kmeans(tr, va, k=6):
    sub = np.random.default_rng(42).choice(tr, 60_000, replace=False)
    km = KMeans(k, n_init=3, random_state=42).fit(Zs_all[sub, :13])
    oh = lambda idx: np.eye(k)[km.predict(Zs_all[idx, :13])]  # noqa: E731
    return oh(tr), oh(va)


def extra_pca(tr, va, k=5):
    pca = PCA(k, random_state=42).fit(Zs_all[tr])
    sc = StandardScaler().fit(pca.transform(Zs_all[tr]))
    return sc.transform(pca.transform(Zs_all[tr])), sc.transform(pca.transform(Zs_all[va]))


def extra_all(tr, va):
    parts = [f(tr, va) for f in (extra_agg, extra_kmeans, extra_pca)]
    return np.column_stack([p[0] for p in parts]), np.column_stack([p[1] for p in parts])


L1_blocks, L6_blocks = ladder[1][1], ladder[-1][1]
for name, ex in [("L1 + tổng hợp điểm", extra_agg), ("L1 + cụm KMeans (k=6)", extra_kmeans),
                 ("L1 + PCA 5 thành phần", extra_pca), ("L1 + cả ba", extra_all)]:
    cached(name, lambda ex=ex: lr_cv(L1_blocks, extra=ex), base=LR[ladder[1][0]])
cached("L6 + cả ba", lambda: lr_cv(L6_blocks, te=True, extra=extra_all), base=LR[ladder[-1][0]])

# %% [markdown]
# **Nhận xét mục 3**
# - Trên L1: tổng hợp điểm +0,00038, cụm KMeans +0,00031, gộp cả ba +0,00051 (đều 5/5 fold). **PCA không thêm gì**
#   (≈ 0, 3/5 fold): PCA chỉ là phép xoay tuyến tính, LR vốn đã tự tổ hợp tuyến tính các cột. Trung bình điểm
#   (bỏ điểm 0), số điểm 0 hay mã cụm là biến đổi **phi tuyến** nên mới mang thông tin mới.
# - Trên L6: cả ba chỉ thêm +0,00003. Khi đã có one-hot từng mức điểm, cặp cột và teacher, thông tin của chúng đã có
#   sẵn, nên **không giữ**. Đặc trưng tổng hợp chỉ có ích khi mô hình còn nghèo đặc trưng.

# %% [markdown]
# ## 4. Chọn đặc trưng (Bài 04)
# ### 4.1 Filter: bỏ các cột yếu nhất khi đứng riêng (E1: AUC một cột ≤ 0,53)

# %%
weak = ["Gender", "Gate location", "Departure/Arrival time convenient", DEP, ARR]
print("Bỏ:", weak)


def fp_l1_drop(X_tr, y_tr, X_va, X_te):
    keep_num = [c for c in NUM_COLS + SCORE_COLS if c not in weak]
    keep_cat = [c for c in CAT_COLS if c not in weak]
    keep_oh = [c for c in SCORE_COLS if c not in weak]
    pre = ColumnTransformer([("num", StandardScaler(), [c for c in keep_num if c not in SCORE_COLS]),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), keep_cat + keep_oh)])
    m = make_pipeline(pre, LogisticRegression(C=0.05, max_iter=3000, tol=1e-5)).fit(X_tr, y_tr)
    return m.decision_function(X_va), m.decision_function(X_te)


cached("filter: L1 bỏ 5 cột yếu", lambda: run_cv(fp_l1_drop, train, test.head(10), FEATURES, verbose=False),
       base=LR[ladder[1][0]])

# %% [markdown]
# ### 4.2 Wrapper: loại ngược từng khối khỏi công thức đầy đủ L6
# Bỏ lần lượt từng khối, xem CV giảm bao nhiêu: khối nào bỏ mà không giảm là khối thừa.

# %%
drop_rows = []
for blk in ["values", "pairs", "digits", "orig"]:
    r = cached(f"wrapper: L6 bỏ {blk}", lambda blk=blk: lr_cv([b for b in L6_blocks if b != blk], te=True),
               base=LR[ladder[-1][0]])
r = cached("wrapper: L6 bỏ target encoding", lambda: lr_cv(L6_blocks, te=False), base=LR[ladder[-1][0]])

# %% [markdown]
# ### 4.3 Embedded: LR chính quy hoá L1 (LASSO) trên L1
# Phạt L1 đẩy hệ số của đặc trưng không quan trọng về 0. Giảm C thì ít đặc trưng được giữ hơn.

# %%
emb = []
for C in [0.0005, 0.002, 0.01, 0.05]:
    r = cached(f"embedded: L1-penalty C={C}", lambda C=C: lr_cv(L1_blocks, C=C, penalty="l1"), base=LR[ladder[1][0]])
    note = pd.read_csv(RESULTS_DIR / f"{EXP}.csv").set_index("variant").loc[f"embedded: L1-penalty C={C}", "notes"]
    emb.append({"C": C, "CV AUC": round(r["cv_auc"], 6), "Δ so với L1": round(r["cv_auc"] - LR[ladder[1][0]]["cv_auc"], 6),
                "hệ số khác 0 (fold cuối)": note.split(": ")[-1] if isinstance(note, str) else ""})
pd.DataFrame(emb)

# %% [markdown]
# **Nhận xét mục 4**
# - **Filter theo AUC một cột là sai hướng:** bỏ 5 cột yếu nhất ở E1 (AUC ≤ 0,53) làm CV giảm 0,0014, 0/5 fold tăng.
#   Cột yếu khi đứng riêng vẫn có ích khi kết hợp, đúng như cảnh báo ở E1.
# - **Wrapper (bỏ từng khối khỏi L6):** dữ liệu gốc −0,0064, cặp cột −0,00056, one-hot giá trị −0,00009,
#   target encoding −0,00006 (đều 0/5 fold), riêng **khối chữ số −0,000001 (2/5 fold) là thừa** và bỏ được.
# - **Embedded (LASSO trên L1, 91 hệ số):** C = 0,05 giữ 77 hệ số và gần như không mất gì (−0,000015); C = 0,01 giữ 67
#   (−0,00011); C = 0,002 giữ 57 (−0,00085); C = 0,0005 giữ 36 (−0,0035). Có thể bỏ khoảng 15% hệ số mà không mất độ
#   chính xác, nhưng bỏ nhiều hơn thì giảm nhanh. (saga chạy 100 vòng, chưa hội tụ hoàn toàn ở C lớn.)
# - Với 700 nghìn dòng, LR không bị overfit vì thừa feature, nên chọn feature ở đây chủ yếu
#   để hiểu khối nào quan trọng, không để tăng điểm.

# %% [markdown]
# ## 5. LightGBM: cùng bậc thang và cùng các nhánh (chạy trên Kaggle)
# Script `exp/kaggle/E2_lgb.py`, kernel `cs116-e2-lightgbm-ladder` (CPU). Tham số giống `A_lgb`, learning rate 0,05
# cho bậc thang và các nhánh; bậc cuối chạy lại với 0,02 đúng như khi thi.

# %%
lgb_path = RESULTS_DIR / "E2_lgb" / "E2_lgb.csv"
lgb_res = pd.read_csv(lgb_path) if lgb_path.exists() else None
lgb_res[["group", "variant", "cv_auc", "delta_cv", "folds_up", "n_features", "trees_mean", "minutes"]]

# %% [markdown]
# ### Cùng một bước tiền xử lý, hai mô hình
# Chênh lệch CV khi áp dụng từng bước (so với nền của mỗi mô hình): LR dùng nền L0 / L1, LightGBM dùng bậc 0 (21 cột gốc).

# %%
lr_t = pd.read_csv(RESULTS_DIR / f"{EXP}.csv").set_index("variant")
g_t = lgb_res.set_index("variant")
base_lr = lr_t.loc["nhánh: gốc (điền bằng delay đi, log1p, z-score)", "cv_auc"]
base_g = g_t.loc["0 cột gốc", "cv_auc"]
rows = [
    ("Điền thiếu bằng median (thay vì để trống / điền delay đi)", lr_t.loc["thiếu: điền median", "cv_auc"] - base_lr,
     g_t.loc["missing_median", "cv_auc"] - base_g),
    ("Thêm cờ thiếu", lr_t.loc["thiếu: điền delay đi + cờ thiếu", "cv_auc"] - base_lr,
     g_t.loc["missing_flag", "cv_auc"] - base_g),
    ("log1p hai cột delay (so với để nguyên phút)", base_lr - lr_t.loc["delay: để nguyên phút", "cv_auc"],
     g_t.loc["delay_log1p", "cv_auc"] - base_g),
    ("Cắt delay ở phân vị 99", lr_t.loc["delay: cắt q99 rồi log1p", "cv_auc"] - base_lr,
     g_t.loc["delay_clip_q99", "cv_auc"] - base_g),
    ("Chuẩn hoá z-score (so với không chuẩn hoá)", base_lr - lr_t.loc["scale: không chuẩn hoá", "cv_auc"],
     g_t.loc["scale_standard", "cv_auc"] - base_g),
    ("Điểm 0–5 thành biến phân loại (one-hot / category)", lr_t.loc["L1 điểm one-hot", "cv_auc"] - lr_t.loc["L0 baseline (điểm dạng số)", "cv_auc"],
     g_t.loc["scores_category", "cv_auc"] - base_g),
    ("Category one-hot (so với mã số / native)", float("nan"), g_t.loc["cat_onehot", "cv_auc"] - base_g),
]
same_step = pd.DataFrame(rows, columns=["bước", "Δ CV với LR", "Δ CV với LightGBM"]).set_index("bước")
same_step.to_csv(RESULTS_DIR / "E2_same_step.csv")
same_step.map(lambda v: f"{v:+.6f}" if pd.notna(v) else "–")

# %%
lr_tab = pd.read_csv(RESULTS_DIR / f"{EXP}.csv").set_index("variant")
lr_ladder = lr_tab.loc[[name for name, _, _ in ladder], "cv_auc"]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), sharey=False)
ax = axes[0]
ax.plot(range(len(lr_ladder)), lr_ladder.values, color=P.ORANGE, marker="o", label="Logistic Regression")
for i, v in enumerate(lr_ladder.values):
    ax.annotate(f"{v:.4f}", (i, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color=P.INK_2)
ax.set_xticks(range(len(lr_ladder)), [s.split(" ", 1)[0] for s in lr_ladder.index])
ax.set_ylabel("CV AUC")
ax.set_title("Logistic Regression (trục tung 0,92–0,96)")
ax.legend(loc="lower right")
if lgb_res is not None:
    g = lgb_res.set_index("variant")["cv_auc"]
    kept = ["0 cột gốc", "1 + chữ số", "2 + tần suất", "3 + TE trong fold", "4 + dữ liệu gốc", "4t + 3 teacher gốc",
            "4t, lr 0,02 (= A_lgb khi thi)"]
    labels = ["0", "1", "2", "3", "4", "4 + teacher", "lr 0,02"]
    ax2 = axes[1]
    ax2.plot(range(len(kept)), g[kept].values, color=P.BLUE, marker="o", label="bậc được giữ")
    for i, v in enumerate(g[kept].values):
        ax2.annotate(f"{v:.4f}", (i, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color=P.INK_2)
    for x, name, lab in [(4.35, "5 + tra cứu bộ điểm", "5"), (4.65, "6 + feature nghiệp vụ", "6")]:
        ax2.plot([4, x], [g["4 + dữ liệu gốc"], g[name]], color=P.INK_3, lw=1, ls="--")
        ax2.scatter([x], [g[name]], s=40, facecolors=P.SURFACE, edgecolors=P.INK_3, zorder=3)
    ax2.annotate(f"bậc 5: {g['5 + tra cứu bộ điểm']:.5f}\nbậc 6: {g['6 + feature nghiệp vụ']:.5f}",
                 (4.5, g["5 + tra cứu bộ điểm"]), textcoords="offset points", xytext=(0, -30), ha="center",
                 fontsize=7, color=P.INK_3)
    ax2.scatter([], [], s=40, facecolors=P.SURFACE, edgecolors=P.INK_3, label="thêm vào bậc 4, không giữ")
    ax2.set_xticks(range(len(kept)), labels)
    ax2.set_title("LightGBM (trục tung 0,958–0,961, hẹp hơn 13 lần)")
    ax2.legend(loc="lower right")
fig.suptitle("Bậc thang đặc trưng: CV AUC sau mỗi bậc", y=1.02)
fig.tight_layout()
P.save(fig, "E2_01_ladder")

# %% [markdown]
# **Nhận xét mục 5**
# - **Bậc thang LightGBM:** 0,95873 → 0,96084 (+0,0021) qua 5 bậc, mọi bậc giữ lại đều 5/5 fold. Bậc lớn nhất là
#   target encoding trong fold (+0,00070), rồi chữ số (+0,00058) và dữ liệu gốc (+0,00048).
# - Bậc 5 (tra cứu cả bộ điểm trên dữ liệu gốc, 2/5 fold) và bậc 6 (feature nghiệp vụ: tổng delay, số điểm 0, trung bình
#   điểm…, 3/5 fold) **không tăng**: khớp với kết luận khi thi. Feature nghiệp vụ cây tự tách được từ các cột gốc.
# - **Tái lập:** bậc cuối (lr 0,02) đạt 0,960889, so với `A_lgb` khi thi 0,960893 (chênh 4e-6; chênh từng fold có cả âm
#   lẫn dương ±1e-4, nhỏ hơn nhiễu giữa hai seed). lr 0,02 hơn lr 0,05 khoảng +0,00005 nhưng cần gấp đôi số cây.
# - **Các nhánh tiền xử lý đều trong ±0,00005** và không nhất quán theo fold: cây chia nhánh theo thứ tự giá trị nên
#   không nhạy với log1p, chuẩn hoá hay cắt ngưỡng; giá trị thiếu LightGBM tự xử lý; mã hoá category cũng gần như không
#   đổi gì.
# - **Bảng "cùng một bước, hai mô hình"** cho thấy rõ: mã hoá điểm thành biến phân loại +0,027 với LR nhưng −0,00004 với
#   LightGBM; log1p +0,00017 với LR nhưng −0,00001 với LightGBM. Ngược lại, chữ số và tần suất +0,0008 với LightGBM nhưng
#   ≈ 0 với LR.

# %% [markdown]
# ## 6. Tóm tắt và quyết định
#
# | | Logistic Regression | LightGBM |
# | --- | --- | --- |
# | Baseline → bộ đầy đủ | 0,9217 → 0,9597 (+0,038) | 0,9587 → 0,9609 (+0,002) |
# | Bước quan trọng nhất | mã hoá điểm one-hot (+0,027), dữ liệu gốc (+0,006) | TE trong fold, chữ số, dữ liệu gốc (+0,0005–0,0007 mỗi bước) |
# | Điền thiếu, log1p, chuẩn hoá, cắt ngoại lệ | log1p có ích; chuẩn hoá chỉ làm nhanh hơn ×43 | không tác dụng (±0,00005) |
# | Tái lập model khi thi | 0,959727 so với 0,959737 | 0,960889 so với 0,960893 |
#
# **Quyết định:**
# 1. Tiền xử lý **theo mô hình**: mô hình tuyến tính / mạng nơ-ron dùng dạng L1 (one-hot điểm và category, log1p delay,
#    z-score); mô hình cây dùng 21 cột gốc, category để dạng `category`. Dùng ở E3.
# 2. Bộ feature đầy đủ của LightGBM = bậc 4 + teacher (`src/features.py`, step 4, teachers=True).
# 3. Không xoá dòng ngoại lệ, không cần imputer phức tạp; chọn feature bằng CV, không bằng AUC một cột.
#
# **Đã thử nhưng không giữ**
#
# | Ý tưởng | Thời điểm | Δ CV | Số fold tăng | Lý do không giữ |
# | --- | --- | --- | --- | --- |
# | Ghép 129 nghìn dòng dữ liệu gốc vào phần train của fold (LightGBM) | 18/10 | +0,0001 (model đơn), 0 vào stack | – | lợi ích nhỏ, không thêm thông tin cho stack |
# | Ghép dòng gốc với trọng số 0,35 (XGBoost của nhóm) | 17/10 | 0,95853, thấp hơn bản dùng teacher (0,96094) | – | teacher tốt hơn ghép dòng |
# | LightGBM bậc 5: tra cứu bộ điểm trên dữ liệu gốc | 14/10, kiểm chứng | +0,000001 | 2/5 | dưới nhiễu |
# | LightGBM bậc 6: feature nghiệp vụ | 14/10, kiểm chứng | +0,000018 | 3/5 | dưới nhiễu, cây tự tách được |
# | LR: chữ số, tần suất (L4) | kiểm chứng | +0,000003 | 3/5 | không có tác dụng với mô hình tuyến tính |
# | LR: tổng hợp điểm, KMeans, PCA trên bộ đầy đủ | kiểm chứng | +0,000028 | 5/5 | quá nhỏ, thông tin đã có sẵn |
# | LR: PCA 5 thành phần | kiểm chứng | ≈ 0 | 3/5 | biến đổi tuyến tính, LR tự làm được |
# | Mọi nhánh điền thiếu, cắt ngoại lệ, chuẩn hoá (cả hai mô hình) | kiểm chứng | ±0,00005 | không nhất quán | dữ liệu thiếu rất ít; cây không nhạy |
# | Bỏ 5 cột yếu nhất theo AUC một cột | kiểm chứng | −0,0014 | 0/5 | cột yếu riêng lẻ vẫn có ích khi kết hợp |
