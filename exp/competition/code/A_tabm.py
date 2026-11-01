"""TabM (Gorishniy et al., ICLR 2025): MLP với batch-ensemble k đầu, member mạng nơ-ron khác RealMLP.

Chạy trên Kaggle GPU (Internet ON để pip install), gắn các input:
  - competition playground-series-s6e10
  - dataset arseniyshutko/binary-aviation-satisfaction-129k   (dữ liệu gốc, data.csv)
  - dataset thanhmn123/cs116-s6e10-oof                        (folds.csv của nhóm)
Output: /kaggle/working/oof_A_tabm_v1.npy, test_A_tabm_v1.npy (xác suất, thứ tự như train.csv / test.csv).
Thử nhanh: đặt FAST = True (fold 0, 100k dòng train).
Kiểm tra code trên máy (CPU, dữ liệu trong repo, mẫu nhỏ, 1 epoch, không lưu): SMOKE=1 python notebooks/A_tabm.py

Feature (khác bộ công khai ở phần dữ liệu gốc):
  - số: Age, Flight Distance, log delay, cờ thiếu arrival, 13 điểm, logit teacher LightGBM chỉ train trên dữ liệu gốc,
        logit tỉ lệ hài lòng theo từng giá trị trên dữ liệu gốc, target encoding trong fold (route, route x class, ...)
  - category (one-hot trong TabM): 4 cột chữ, Flight Distance như mã tuyến, Age, 13 điểm
OOF sạch (docs/STACKING.md mục 5): fold theo folds.csv; target encoding, quantile transform, bin của embedding
chỉ fit trên 90% phần train của fold; dừng sớm trên 10% còn lại (fold đang chấm không tham gia).
"""
import glob
import os
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore", message=".*has just two bin edges.*")   # cột nhị phân: 1 bin là bình thường
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tabm", "rtdl_num_embeddings"], check=True)

import lightgbm as lgb
import numpy as np
import pandas as pd
import rtdl_num_embeddings
import tabm
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer, TargetEncoder

SMOKE = os.environ.get("SMOKE") == "1"
FAST = SMOKE or False
NAME = "A_tabm_v1"
SEED = 42
BATCH, EVAL_BATCH = 512, 8192
MAX_EPOCHS, PATIENCE = (1, 1) if SMOKE else (12, 2)   # mỗi epoch ~1.000 bước; dừng khi AUC phần dừng sớm không tăng 2 epoch
LR, WD = 2e-3, 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T0 = time.time()

TARGET, ID = "satisfaction", "id"
CATS = ["Gender", "Customer Type", "Type of Travel", "Class"]
SCORES = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking", "Gate location",
          "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment", "On-board service",
          "Leg room service", "Baggage handling", "Checkin service", "Cleanliness"]
FD, AGE, DEP, ARR = "Flight Distance", "Age", "Departure Delay in Minutes", "Arrival Delay in Minutes"
FEATURES = CATS + SCORES + [AGE, FD, DEP, ARR]
TE_KEYS = {"te_fd": [FD], "te_fd_class": [FD, "Class"], "te_fd_travel": [FD, "Type of Travel"],
           "te_age_class": [AGE, "Class"], "te_age_travel": [AGE, "Type of Travel"]}


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


REPO_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
LOCAL = {"train.csv": "train.csv", "test.csv": "test.csv", "data.csv": "original/data.csv", "folds.csv": "folds.csv"}


def find(name, hint):
    if not os.path.isdir("/kaggle/input"):   # chạy trên máy: dùng data/ của repo
        return os.path.join(REPO_DATA, LOCAL[name])
    hits = [p for p in glob.glob(f"/kaggle/input/**/{name}", recursive=True) if hint in p]
    assert len(hits) >= 1, f"không thấy {name} ({hint}): gắn input"
    return sorted(hits, key=len)[0]


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p)).astype(np.float32)


def codes(df, cols):
    key = np.zeros(len(df), dtype=np.int64)
    for c in cols:
        cc = pd.factorize(df[c], use_na_sentinel=False)[0]
        key = key * (cc.max() + 1) + cc
    return pd.factorize(key)[0]


# ---------------- dữ liệu và fold chung ----------------
train = pd.read_csv(find("train.csv", "playground-series-s6e10"))
test = pd.read_csv(find("test.csv", "playground-series-s6e10"))
orig = pd.read_csv(find("data.csv", "aviation"))
folds = pd.read_csv(find("folds.csv", "cs116-s6e10-oof"))
assert folds[ID].equals(train[ID]), "folds.csv không khớp thứ tự id của train.csv"
y = train[TARGET].astype(int).values
fold = folds["fold"].values
orig[TARGET] = orig[TARGET].astype(int)
if SMOKE:   # mẫu nhỏ để kiểm tra code, giữ nguyên fold của từng dòng
    keep = np.random.default_rng(0).choice(len(train), 30_000, replace=False)
    train, y, fold = train.iloc[keep].reset_index(drop=True), y[keep], fold[keep]
    test = test.head(5_000)
n = len(train)
A = pd.concat([train[FEATURES], test[FEATURES]], ignore_index=True)
log(f"train {train.shape}, test {test.shape}, gốc {orig.shape}, thiết bị {DEVICE}")

# ---------------- feature không dùng nhãn cuộc thi (tính một lần) ----------------
num = pd.DataFrame({AGE: A[AGE], FD: A[FD], "log_dep": np.log1p(A[DEP]),
                    "log_arr": np.log1p(A[ARR].fillna(A[DEP])), "arr_missing": A[ARR].isna().astype(float)})
for c in SCORES:
    num[c] = A[c]
# teacher: LightGBM chỉ thấy nhãn của dữ liệu gốc
Xa, Xo = A.copy(), orig[FEATURES].copy()
for c in CATS:
    cats = pd.Categorical(pd.concat([A[c], orig[c]])).categories
    Xa[c], Xo[c] = pd.Categorical(Xa[c], categories=cats), pd.Categorical(Xo[c], categories=cats)
teacher = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, max_depth=8, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.5, reg_lambda=5.0, random_state=123, verbose=-1)
teacher.fit(Xo, orig[TARGET])
num["teacher"] = logit(teacher.predict_proba(Xa)[:, 1])
prior = orig[TARGET].mean()
for c in FEATURES:   # tỉ lệ hài lòng theo từng giá trị trên dữ liệu gốc (làm mượt)
    g = orig.groupby(c)[TARGET].agg(["sum", "count"])
    s, cnt = A[c].map(g["sum"]).fillna(0), A[c].map(g["count"]).fillna(0)
    num[f"og_{c}"] = logit((s + 20 * prior) / (cnt + 20))
num = num.astype(np.float32)
log(f"teacher trên train cuộc thi: AUC {roc_auc_score(y, num['teacher'].values[:n]):.4f}")

CAT_COLS = CATS + [FD, AGE] + SCORES
cat = np.column_stack([pd.factorize(A[c], use_na_sentinel=False)[0] for c in CAT_COLS]).astype(np.int64)
cat_card = [int(cat[:, j].max()) + 1 for j in range(cat.shape[1])]
te_codes = pd.DataFrame({k: codes(A, v) for k, v in TE_KEYS.items()})
log(f"{num.shape[1]} cột số + {len(TE_KEYS)} target encoding | {len(CAT_COLS)} cột category, tổng {sum(cat_card)} giá trị")


def batches(x_num, x_cat, bs):
    for i in range(0, len(x_num), bs):
        yield x_num[i:i + bs], x_cat[i:i + bs]


@torch.no_grad()
def predict(model, x_num, x_cat):
    model.eval()
    out = []
    for xn, xc in batches(x_num, x_cat, EVAL_BATCH):
        p = torch.sigmoid(model(xn, xc).squeeze(-1).float())   # (batch, k)
        out.append(p.mean(1).cpu().numpy())                    # trung bình xác suất của k đầu
    return np.concatenate(out)


def to_dev(a, dtype):
    return torch.as_tensor(a, dtype=dtype, device=DEVICE)


oof, test_pred = np.zeros(n), np.zeros(len(test))
run_folds = [0] if FAST else range(5)
for k in run_folds:
    torch.manual_seed(SEED + k)
    va = np.where(fold == k)[0]
    tr_idx = np.where(fold != k)[0]
    if FAST and not SMOKE:
        tr_idx = np.random.default_rng(0).choice(tr_idx, 100_000, replace=False)
    es = np.random.default_rng(42).choice(tr_idx, size=len(tr_idx) // 10, replace=False)
    fit = np.setdiff1d(tr_idx, es)
    te_idx = np.arange(n, len(A))

    # target encoding: chỉ nhãn của phần fit (cross-fit 5 phần bên trong)
    enc = TargetEncoder(target_type="binary", cv=5, shuffle=True, random_state=42)
    T = np.zeros((len(A), len(TE_KEYS)), dtype=np.float32)
    T[fit] = enc.fit_transform(te_codes.values[fit], y[fit])
    rest = np.setdiff1d(np.arange(len(A)), fit)
    T[rest] = enc.transform(te_codes.values[rest])
    X = np.hstack([num.values, logit(T)])

    # quantile transform + bin của embedding: fit trên phần fit
    noise = np.random.default_rng(0).normal(0.0, 1e-5, X[fit].shape)
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=10**9,
                             random_state=0).fit(X[fit] + noise)
    X = qt.transform(X).astype(np.float32)
    Xn, Xc = to_dev(X, torch.float32), to_dev(cat, torch.long)

    model = tabm.TabM.make(
        n_num_features=X.shape[1], cat_cardinalities=cat_card, d_out=1,
        num_embeddings=rtdl_num_embeddings.PiecewiseLinearEmbeddings(
            rtdl_num_embeddings.compute_bins(torch.as_tensor(X[fit]), n_bins=48),
            d_embedding=16, activation=False, version="B"),
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    kk = model.backbone.k
    yt = to_dev(y, torch.float32)
    best, best_state, wait = -1.0, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.as_tensor(np.random.default_rng(SEED * 100 + k * 10 + epoch).permutation(fit), device=DEVICE)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            logits = model(Xn[b], Xc[b]).squeeze(-1)            # (batch, k)
            # train k đầu độc lập: trung bình loss của từng đầu, không phải loss của trung bình
            loss = F.binary_cross_entropy_with_logits(logits.flatten(), yt[b].repeat_interleave(kk))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        auc = roc_auc_score(y[es], predict(model, Xn[es], Xc[es]))
        log(f"fold {k} epoch {epoch}: AUC phần dừng sớm {auc:.6f}")
        if auc > best:
            best, wait = auc, 0
            best_state = {kname: v.detach().clone() for kname, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    oof[va] = predict(model, Xn[va], Xc[va])
    test_pred += predict(model, Xn[te_idx], Xc[te_idx]) / len(run_folds)
    log(f"fold {k}: AUC = {roc_auc_score(y[va], oof[va]):.6f}")
    del model, opt, Xn, Xc
    torch.cuda.empty_cache()

if FAST:
    log("FAST: chỉ thử fold 0, không lưu")
else:
    assert not np.isnan(oof).any() and not np.isnan(test_pred).any()
    log(f"{NAME}: CV AUC = {roc_auc_score(y, oof):.6f}")
    np.save(f"/kaggle/working/oof_{NAME}.npy", oof)
    np.save(f"/kaggle/working/test_{NAME}.npy", test_pred)
    log("đã lưu oof / test")
