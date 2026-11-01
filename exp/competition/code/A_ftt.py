"""FT-Transformer với token tra cứu giá trị đúng + target encoding giá trị đúng (ý từ S6E7 4th / S6E8 14th "Lookup-Transformer"):
mỗi cột là một token, attention giữa các token; số lặp lại theo giá trị đúng nên mỗi giá trị có embedding riêng.
Member mạng nơ-ron khác TabM / RealMLP trong pool.

Chạy trên Kaggle GPU (Internet ON để pip install), gắn các input:
  - competition playground-series-s6e10
  - dataset arseniyshutko/binary-aviation-satisfaction-129k   (dữ liệu gốc, data.csv)
  - dataset thanhmn123/cs116-s6e10-oof                        (folds.csv của nhóm)
Output: /kaggle/working/oof_A_ftt_v1.npy, test_A_ftt_v1.npy (xác suất, thứ tự như train.csv / test.csv).
Thử nhanh: biến môi trường FAST=1 (fold 0, 100k dòng train). Kiểm tra code trên máy: SMOKE=1 python notebooks/A_ftt.py

Token (mỗi token d chiều):
  - 21 token tra cứu: embedding theo giá trị đúng của từng cột (từ điển trên train + test, không dùng nhãn)
  - token số qua PLR (periodic embedding): Age, Flight Distance, log delay, teacher LightGBM chỉ train trên dữ liệu gốc,
    target encoding giá trị đúng của 21 cột + 4 cặp (Flight Distance / Age × Class / Type of Travel)
OOF sạch (docs/STACKING.md mục 5): fold theo folds.csv; target encoding, quantile transform chỉ fit trên 90% phần train
của fold; dừng sớm trên 10% còn lại (fold đang chấm không tham gia).
"""
import glob
import os
import subprocess
import sys
import time

if os.path.isdir("/kaggle/input"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rtdl_num_embeddings"], check=True)

import lightgbm as lgb
import numpy as np
import pandas as pd
import rtdl_num_embeddings
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer, TargetEncoder

SMOKE = os.environ.get("SMOKE") == "1"
FAST = SMOKE or os.environ.get("FAST") == "1"
NAME = "A_ftt_v1"
SEED = 42
D, HEADS, LAYERS, FFN_MULT, DROP = 64, 8, 3, 2, 0.1
BATCH, EVAL_BATCH = 1024, 8192
MAX_EPOCHS, PATIENCE = (1, 1) if SMOKE else (12, 2)
LR, WD = 1e-3, 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T0 = time.time()

TARGET, ID = "satisfaction", "id"
CATS = ["Gender", "Customer Type", "Type of Travel", "Class"]
SCORES = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking", "Gate location",
          "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment", "On-board service",
          "Leg room service", "Baggage handling", "Checkin service", "Cleanliness"]
FD, AGE, DEP, ARR = "Flight Distance", "Age", "Departure Delay in Minutes", "Arrival Delay in Minutes"
FEATURES = CATS + SCORES + [AGE, FD, DEP, ARR]
TE_KEYS = {f"te_{c}": [c] for c in FEATURES}
TE_KEYS.update({"te_fd_class": [FD, "Class"], "te_fd_travel": [FD, "Type of Travel"],
                "te_age_class": [AGE, "Class"], "te_age_travel": [AGE, "Type of Travel"]})


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
                    "log_arr": np.log1p(A[ARR].fillna(A[DEP]))})
Xa, Xo = A.copy(), orig[FEATURES].copy()
for c in CATS:
    cats = pd.Categorical(pd.concat([A[c], orig[c]])).categories
    Xa[c], Xo[c] = pd.Categorical(Xa[c], categories=cats), pd.Categorical(Xo[c], categories=cats)
teacher = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, max_depth=8, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.5, reg_lambda=5.0, random_state=123, verbose=-1)
teacher.fit(Xo, orig[TARGET])
num["teacher"] = logit(teacher.predict_proba(Xa)[:, 1])
num = num.astype(np.float32)

# token tra cứu: mã giá trị đúng của từng cột (0 = giá trị hiếm < 5 dòng, dùng chung một embedding)
lookup = []
for c in FEATURES:
    s = A[c].astype(str)
    vc = s.value_counts()
    code = pd.Series(pd.factorize(s.where(s.map(vc) >= 5, "__rare__"))[0] + 1, dtype=np.int64)
    lookup.append(code.values)
lookup = np.column_stack(lookup)
vocab = [int(lookup[:, j].max()) + 1 for j in range(lookup.shape[1])]
te_codes = pd.DataFrame({k: codes(A, v) for k, v in TE_KEYS.items()})
log(f"{len(FEATURES)} token tra cứu (tổng {sum(vocab)} giá trị) + {num.shape[1] + len(TE_KEYS)} token số")


class FTT(nn.Module):
    def __init__(self, n_num, vocab):
        super().__init__()
        self.num_emb = rtdl_num_embeddings.PeriodicEmbeddings(n_num, D, lite=False)
        self.cat_emb = nn.ModuleList([nn.Embedding(v, D) for v in vocab])
        self.col_bias = nn.Parameter(torch.zeros(len(vocab), D))
        self.cls = nn.Parameter(torch.randn(1, 1, D) * 0.02)
        layer = nn.TransformerEncoderLayer(D, HEADS, D * FFN_MULT, DROP, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, LAYERS)
        self.head = nn.Sequential(nn.LayerNorm(D), nn.ReLU(), nn.Linear(D, 1))

    def forward(self, x_num, x_cat):
        tok_num = self.num_emb(x_num)                                                   # (b, n_num, D)
        tok_cat = torch.stack([e(x_cat[:, j]) for j, e in enumerate(self.cat_emb)], 1) + self.col_bias
        x = torch.cat([self.cls.expand(len(x_num), -1, -1), tok_num, tok_cat], 1)
        return self.head(self.encoder(x)[:, 0]).squeeze(-1)


def to_dev(a, dtype):
    return torch.as_tensor(a, dtype=dtype, device=DEVICE)


@torch.no_grad()
def predict(model, x_num, x_cat):
    model.eval()
    out = []
    for i in range(0, len(x_num), EVAL_BATCH):
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            out.append(torch.sigmoid(model(x_num[i:i + EVAL_BATCH], x_cat[i:i + EVAL_BATCH]).float()).cpu().numpy())
    return np.concatenate(out)


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

    # target encoding giá trị đúng: chỉ nhãn của phần fit (cross-fit 5 phần bên trong)
    enc = TargetEncoder(target_type="binary", cv=5, shuffle=True, random_state=42)
    T = np.zeros((len(A), len(TE_KEYS)), dtype=np.float32)
    T[fit] = enc.fit_transform(te_codes.values[fit], y[fit])
    rest = np.setdiff1d(np.arange(len(A)), fit)
    T[rest] = enc.transform(te_codes.values[rest])
    X = np.hstack([num.values, logit(T)])
    noise = np.random.default_rng(0).normal(0.0, 1e-5, X[fit].shape)
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=10**9,
                             random_state=0).fit(X[fit] + noise)
    X = qt.transform(X).astype(np.float32)
    Xn, Xc = to_dev(X, torch.float32), to_dev(lookup, torch.long)

    model = FTT(X.shape[1], vocab).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps = MAX_EPOCHS * ((len(fit) + BATCH - 1) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE == "cuda")
    yt = to_dev(y, torch.float32)
    best, best_state, wait = -1.0, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.as_tensor(np.random.default_rng(SEED * 100 + k * 10 + epoch).permutation(fit), device=DEVICE)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
                loss = F.binary_cross_entropy_with_logits(model(Xn[b], Xc[b]).float(), yt[b])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
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
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

if FAST:
    log("FAST: chỉ thử fold 0, không lưu")
else:
    assert not np.isnan(oof).any() and not np.isnan(test_pred).any()
    log(f"{NAME}: CV AUC = {roc_auc_score(y, oof):.6f}")
    out = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
    np.save(f"{out}/oof_{NAME}.npy", oof)
    np.save(f"{out}/test_{NAME}.npy", test_pred)
    log("đã lưu oof / test")
