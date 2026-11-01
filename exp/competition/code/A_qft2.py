"""Q-FT v2 (theo Chris Deotte, discussion 751334): transformer đọc mỗi dòng như một TẬP token (không vị trí),
side features cho từng token, 2 token tóm tắt, pretrain replaced-token detection (RTD) không dùng nhãn.
Member khác hẳn khối cây trong pool công khai; Deotte báo OOF 0.9608 nhưng chưa công bố OOF.

Khác bản gốc: encoder tự viết cỡ ModernBERT-base (22 lớp, 768, 12 head, GeGLU 1152, pre-norm, không bias), khởi tạo
ngẫu nhiên — Deotte: ModernBERT khởi tạo ngẫu nhiên điểm bằng bản pretrained, nên không cần tải trọng số HF.
Không có position embedding nào → thứ tự cột không ảnh hưởng (tương đương position_ids = 0).
v2 (sau v1 CV 0.829, backbone sau RTD gần như còn nguyên khởi tạo): khởi tạo std 0.02 + thu nhỏ lớp ra theo độ sâu,
autocast trong forward (DataParallel), fp16 thật trên T4 thay bf16 giả lập, log grad norm / AUC theo dõi, dừng nếu fold < 0.93.
v3: fine-tune 4 epoch thay 2; dùng lại qft2_rtd.pt của v2 (gắn output cs116-qft2-v2); file fold đặt tên theo NAME
để không dùng nhầm qft2_fold*.npz của v2.
v4 "thuần" (v3 CV 0.9593 < v2: 4 epoch overfit; v2 rho 0.977 với stack → lợi nhỏ): PURE bỏ mọi thứ do LGB tạo ra —
teacher logit trong token tóm tắt 2 và side[6,7] (dự đoán cột bằng LGB) — để khác khối cây hơn; RTD lại từ đầu
(lưu qft2p_rtd.pt, không dùng lại trọng số v2), fine-tune 2 epoch như v2.

Chạy trên Kaggle GPU (T4), gắn input:
  - competition playground-series-s6e10
  - dataset arseniyshutko/binary-aviation-satisfaction-129k   (dữ liệu gốc, data.csv)
  - dataset thanhmn123/cs116-s6e10-oof                        (folds.csv của nhóm)
  - (tuỳ chọn) output kernel có qft2_rtd.pt → bỏ qua pretrain (chạy tiếp khi hết giờ)
Output: /kaggle/working/oof_<NAME>.npy, test_<NAME>.npy (xác suất, thứ tự như train.csv / test.csv),
  qft2_rtd.pt (trọng số sau RTD), oof/test từng fold (<NAME>_fold{k}.npz) để không mất khi hết giờ.
FAST=1: đo tốc độ (RTD 200 bước + fine-tune fold 0 300 bước), không lưu. SMOKE=1: model nhỏ, 30k dòng, chạy trên máy.

OOF sạch (docs/STACKING.md mục 5): fold theo folds.csv; số epoch cố định (2), không dừng sớm, fold đang chấm chỉ được dự đoán.
Mọi side feature không dùng nhãn cuộc thi (chỉ đặc trưng train + test + gốc, nhãn của dữ liệu gốc); RTD không dùng nhãn,
chạy một lần cho mọi fold (như Deotte). Dữ liệu gốc chỉ dùng làm feature và trong pretrain, không vào fold validation.
"""
import glob
import math
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

SMOKE = os.environ.get("SMOKE") == "1"
FAST = (SMOKE and os.environ.get("SMOKE_FULL") != "1") or os.environ.get("FAST") == "1"
NAME = "A_qft2_v4"
SEED = 42
PURE = True                                   # v4: không dùng feature do LGB tạo (teacher logit, side[6,7])
RTD_FILE = "qft2p_rtd.pt" if PURE else "qft2_rtd.pt"
if SMOKE:
    D, HEADS, LAYERS, INTER = 64, 4, 2, 96
else:
    D, HEADS, LAYERS, INTER = 768, 12, 22, 1152
BATCH, EVAL_BATCH = 512, 4096
RTD_EPOCHS, RTD_P, FT_EPOCHS = 3, 0.15, 2          # v4: 2 epoch (v3 4 epoch: AUC đỉnh epoch 2–3 rồi giảm, CV 0.9593)
LR_BB_RTD, LR_TOK_RTD = 2e-4, 1e-3          # RTD từ khởi tạo ngẫu nhiên: lr backbone cao hơn fine-tune
LR_BB, LR_TOK, WD, WARMUP = 5e-5, 1e-3, 0.01, 0.03
FAST_RTD_STEPS, FAST_FT_STEPS = (20, 30) if SMOKE else (200, 300)
N_TREES = 20 if FAST else 150
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# v2: chỉ dùng bf16 khi GPU hỗ trợ thật (Ampere trở lên); T4 (7.5) chỉ giả lập bf16 → dùng fp16 + GradScaler
BF16_OK = DEVICE == "cuda" and torch.cuda.get_device_capability()[0] >= 8
AMP_DTYPE = torch.bfloat16 if BF16_OK else torch.float16
MON_ROWS = 20_000                              # số dòng fold đang chấm dùng để IN AUC trong lúc train (chỉ log, không quyết gì)
T0 = time.time()

TARGET, ID = "satisfaction", "id"
CATS = ["Gender", "Customer Type", "Type of Travel", "Class"]
SCORES = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking", "Gate location",
          "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment", "On-board service",
          "Leg room service", "Baggage handling", "Checkin service", "Cleanliness"]
FD, AGE, DEP, ARR = "Flight Distance", "Age", "Departure Delay in Minutes", "Arrival Delay in Minutes"
FEATURES = CATS + SCORES + [AGE, FD, DEP, ARR]
NUM_COLS = [AGE, FD, DEP, ARR]
NUM_IDX = [FEATURES.index(c) for c in NUM_COLS]
NF = len(FEATURES)
ORIG_KEYS = [[FD], [FD, "Class"], [FD, "Type of Travel"], [FD, AGE], [FD, AGE, "Class"], [AGE, "Class"],
             [AGE, "Type of Travel"], ["Class", "Type of Travel"]]


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
    return np.log(p / (1 - p))


def zscore(a, ref):
    """Chuẩn hoá theo cột với mean/std của các dòng ref (train + test)."""
    mu, sd = np.nanmean(a[ref], 0), np.nanstd(a[ref], 0)
    return np.nan_to_num((a - mu) / np.where(sd > 1e-9, sd, 1.0)).astype(np.float32)


def group_key(df, cols):
    key = np.zeros(len(df), dtype=np.int64)
    for c in cols:
        cc = pd.factorize(df[c], use_na_sentinel=False)[0]
        key = key * (cc.max() + 1) + cc
    return key


# ---------------- dữ liệu và fold chung ----------------
train = pd.read_csv(find("train.csv", "playground-series-s6e10"))
test = pd.read_csv(find("test.csv", "playground-series-s6e10"))
orig = pd.read_csv(find("data.csv", "aviation"))
folds = pd.read_csv(find("folds.csv", "cs116-s6e10-oof"))
assert folds[ID].equals(train[ID]), "folds.csv không khớp thứ tự id của train.csv"
y = train[TARGET].astype(int).values
fold = folds["fold"].values
y_or = orig[TARGET].astype(int).values
if SMOKE:   # mẫu nhỏ để kiểm tra code, giữ nguyên fold của từng dòng
    keep = np.random.default_rng(0).choice(len(train), 30_000, replace=False)
    train, y, fold = train.iloc[keep].reset_index(drop=True), y[keep], fold[keep]
    test = test.head(5_000)
    orig, y_or = orig.head(10_000), y_or[:10_000]
n_tr, n_te, n_or = len(train), len(test), len(orig)
A = pd.concat([train[FEATURES], test[FEATURES], orig[FEATURES]], ignore_index=True)
N = len(A)
comp = np.arange(n_tr + n_te)                  # dòng cuộc thi (train + test)
is_or = np.zeros(N, bool)
is_or[n_tr + n_te:] = True
log(f"train {n_tr}, test {n_te}, gốc {n_or}, thiết bị {DEVICE}, amp {AMP_DTYPE}")

# giá trị số của từng cột (category → mã thứ tự) dùng cho thống kê tuyến / mô hình dự đoán cột
V = np.zeros((N, NF), dtype=np.float64)
for j, c in enumerate(FEATURES):
    if c in CATS:
        V[:, j] = pd.Categorical(A[c], categories=sorted(A[c].dropna().unique())).codes
    elif c == FD:
        V[:, j] = np.log1p(A[c].values)
    elif c in (DEP, ARR):
        V[:, j] = np.log1p(A[c].fillna(A[DEP]).values)
    else:
        V[:, j] = A[c].values

# ---------------- token: mã giá trị ----------------
codes = np.zeros((N, NF), dtype=np.int64)
for j, c in enumerate(FEATURES):
    s = A[c].astype(str)
    cnt = s.iloc[comp].value_counts()
    ok = s.map(cnt).fillna(0).values >= 3        # giá trị gặp ≥ 3 lần trong train + test, còn lại là "hiếm" (mã 0)
    codes[:, j] = np.where(ok, pd.factorize(s.where(ok, "__rare__"))[0] + 1, 0)
vocab = [int(codes[:, j].max()) + 1 for j in range(NF)]

# số liên tục cho embedding tuần hoàn (chỉ 4 cột số)
XNUM = np.zeros((N, NF), dtype=np.float32)
XNUM[:, NUM_IDX] = zscore(V[:, NUM_IDX], comp)

# ---------------- side features (9 số / token) ----------------
SIDE = np.zeros((N, NF, 9), dtype=np.float32)
p0 = y_or.mean()
fd_key = A[FD].values
for j, c in enumerate(FEATURES):
    s = A[c].astype(str).values
    s_comp, s_or = pd.Series(s[comp]), pd.Series(s[is_or])
    cc = pd.Series(s).map(s_comp.value_counts()).fillna(0).values
    co = pd.Series(s).map(s_or.value_counts()).fillna(0).values
    co = co - is_or                              # dòng gốc: không tính chính nó
    SIDE[:, j, 0] = np.log1p(cc)
    SIDE[:, j, 1] = np.log1p(co)
    SIDE[:, j, 2] = np.log((cc + 1) / len(comp)) - np.log((co + 1) / n_or)
    # hồ sơ tuyến: trung bình / độ lệch của cột này trên các dòng cùng Flight Distance (train + test), và trên dữ liệu gốc
    g = pd.DataFrame({"fd": fd_key[comp], "v": V[comp, j]}).groupby("fd")["v"]
    SIDE[:, j, 3] = pd.Series(fd_key).map(g.mean()).fillna(V[comp, j].mean()).values
    SIDE[:, j, 4] = pd.Series(fd_key).map(g.std()).fillna(0).values
    go = pd.DataFrame({"fd": fd_key[is_or], "v": V[is_or, j]}).groupby("fd")["v"].mean()
    SIDE[:, j, 5] = pd.Series(fd_key).map(go).fillna(V[is_or, j].mean()).values
    # ý kiến dữ liệu gốc: log-odds làm trơn của "hài lòng" theo đúng giá trị này (dòng gốc: bỏ chính nó)
    st = pd.DataFrame({"v": s_or.values, "y": y_or}).groupby("v")["y"].agg(["sum", "count"])
    pos = pd.Series(s).map(st["sum"]).fillna(0).values - np.where(is_or, np.r_[np.zeros(len(comp)), y_or], 0)
    cnt = pd.Series(s).map(st["count"]).fillna(0).values - is_or
    SIDE[:, j, 8] = logit((pos + 20 * p0) / (cnt + 20)) - logit(p0)

# giá trị kỳ vọng của mỗi cột từ 20 cột còn lại (không nhãn; cross-fit 2 nửa train + test, dòng gốc lấy TB 2 model)
half = np.random.default_rng(SEED).random(len(comp)) < 0.5
for j in range(0 if PURE else NF):                 # PURE: side[6,7] để 0 (sau zscore vẫn 0)
    others = [i for i in range(NF) if i != j]
    pred = np.zeros(N)
    for h in (True, False):
        fit_rows = comp[half == h]
        fit_rows = np.random.default_rng(j).choice(fit_rows, min(len(fit_rows), 300_000), replace=False)
        m = lgb.LGBMRegressor(n_estimators=N_TREES, learning_rate=0.1, num_leaves=63, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, random_state=j, verbose=-1)
        m.fit(V[fit_rows][:, others], V[fit_rows, j])
        tgt = comp[half != h]
        pred[tgt] = m.predict(V[tgt][:, others])
        pred[is_or] += m.predict(V[is_or][:, others]) / 2
    SIDE[:, j, 6] = pred
    SIDE[:, j, 7] = V[:, j] - pred
log("bỏ mô hình dự đoán cột (PURE)" if PURE else "xong mô hình dự đoán cột")
for j in range(NF):
    SIDE[:, j, :] = zscore(SIDE[:, j, :].astype(np.float64), comp)

# ---------------- token tóm tắt 1: thống kê dòng (28 số) ----------------
R = A[SCORES].values.astype(np.float64)
dep, arr = A[DEP].values.astype(np.float64), A[ARR].values.astype(np.float64)
arr_f = np.where(np.isnan(arr), dep, arr)
rows = [R.sum(1), R.std(1), R.min(1), R.max(1), np.median(R, 1)]
rows += [(R == v).sum(1) for v in range(6)]
rows += [R[:, :4].mean(1), R[:, 4:].mean(1), (R >= 4).sum(1)]
rows += [np.log1p(dep), np.log1p(arr_f), np.isnan(arr), arr_f - dep, np.sign(arr_f - dep), dep == 0, arr_f == 0,
         np.log1p(np.abs(arr_f - dep))]
rows += [A[AGE].values, np.log1p(A[FD].values), A[AGE].values < 18, A[AGE].values >= 40,
         A["Online boarding"].values >= 4, A["Inflight wifi service"].values == 0,
         A["Inflight wifi service"].values == 5, A["Inflight entertainment"].values >= 4]
ROWSTAT = zscore(np.column_stack(rows).astype(np.float64), comp)

# ---------------- token tóm tắt 2: khối dữ liệu gốc (17 số) ----------------
Xo_all = A.copy()
for c in CATS:
    Xo_all[c] = pd.Categorical(Xo_all[c], categories=sorted(A[c].dropna().unique()))
teacher_logit = np.zeros(N)
orig_block = []
params = dict(n_estimators=600 if not FAST else 60, learning_rate=0.05, max_depth=8, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.5, reg_lambda=5.0, verbose=-1)
if not PURE:                                                                    # PURE: bỏ teacher LGB
    t = lgb.LGBMClassifier(random_state=123, **params).fit(Xo_all[is_or], y_or)  # teacher chỉ train trên dữ liệu gốc
    teacher_logit[~is_or] = logit(t.predict_proba(Xo_all[~is_or])[:, 1])
    or_idx = np.where(is_or)[0]
    for a, b in KFold(5, shuffle=True, random_state=0).split(or_idx):            # dòng gốc: cross-fit, không thấy chính nó
        t = lgb.LGBMClassifier(random_state=123, **params).fit(Xo_all.iloc[or_idx[a]], y_or[a])
        teacher_logit[or_idx[b]] = logit(t.predict_proba(Xo_all.iloc[or_idx[b]])[:, 1])
    orig_block = [teacher_logit]
for keys in ORIG_KEYS:
    k = pd.Series(group_key(A, keys))
    st = pd.DataFrame({"k": k[is_or].values, "y": y_or}).groupby("k")["y"].agg(["sum", "count"])
    pos = k.map(st["sum"]).fillna(0).values - np.where(is_or, np.r_[np.zeros(len(comp)), y_or], 0)
    cnt = k.map(st["count"]).fillna(0).values - is_or
    orig_block += [np.log1p(cnt), logit((pos + 20 * p0) / (cnt + 20)) - logit(p0)]
ORIGB = zscore(np.column_stack(orig_block), comp)
log(f"feature xong: {NF} token (vocab {sum(vocab)}), side {SIDE.shape[2]}, tóm tắt {ROWSTAT.shape[1]} + {ORIGB.shape[1]}")


# ---------------- model ----------------
class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype) * self.w


class Block(nn.Module):
    """Lớp kiểu ModernBERT: pre-norm, attention toàn cục hai chiều, GeGLU, không bias."""

    def __init__(self):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(D, bias=False), nn.LayerNorm(D, bias=False)
        self.qkv, self.out = nn.Linear(D, 3 * D, bias=False), nn.Linear(D, D, bias=False)
        self.wi, self.wo = nn.Linear(D, 2 * INTER, bias=False), nn.Linear(INTER, D, bias=False)

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, t, 3, HEADS, D // HEADS).permute(2, 0, 3, 1, 4)
        x = x + self.out(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, t, D))
        g, u = self.wi(self.n2(x)).chunk(2, -1)
        return x + self.wo(F.gelu(g) * u)


class QFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.col = nn.Parameter(torch.randn(NF, D) * 0.02)
        self.val = nn.ModuleList([nn.Embedding(v, D) for v in vocab])
        self.freq = nn.Parameter(torch.randn(len(NUM_IDX), 16) * 1.0)        # embedding tuần hoàn sin/cos → MLP nhỏ
        self.num_mlp = nn.ModuleList([nn.Sequential(nn.Linear(32, D), nn.GELU(), nn.Linear(D, D)) for _ in NUM_IDX])
        self.side = nn.Parameter(torch.randn(NF, 9, D) * 0.02)               # ma trận 9 × D riêng từng cột
        self.sum_row, self.sum_orig = nn.Linear(ROWSTAT.shape[1], D), nn.Linear(ORIGB.shape[1], D)
        self.sum_type = nn.Parameter(torch.randn(2, D) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, D) * 0.02)
        self.tok_norm = RMSNorm(D)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.norm = nn.LayerNorm(D, bias=False)
        self.head = nn.Linear(D, 1)
        self.rtd_head = nn.Linear(D, 1)
        # v2: khởi tạo kiểu ModernBERT/GPT-2 (v1 để mặc định PyTorch, backbone sau RTD gần như không rời điểm khởi tạo)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for blk in self.blocks:                                               # lớp ra của mỗi nhánh dư: thu nhỏ theo độ sâu
            nn.init.normal_(blk.out.weight, std=0.02 / math.sqrt(2 * LAYERS))
            nn.init.normal_(blk.wo.weight, std=0.02 / math.sqrt(2 * LAYERS))
        for e in self.val:
            nn.init.normal_(e.weight, std=0.02)

    def tokens(self, code, xnum, side, rs, ob):
        tok = torch.stack([e(code[:, j]) for j, e in enumerate(self.val)], 1) + self.col
        xs = xnum[:, NUM_IDX].float().unsqueeze(-1) * self.freq * 2 * math.pi
        pe = torch.cat([torch.sin(xs), torch.cos(xs)], -1)
        num_tok = torch.stack([m(pe[:, i]) for i, m in enumerate(self.num_mlp)], 1)
        tok = tok.index_add(1, torch.tensor(NUM_IDX, device=tok.device), num_tok.to(tok.dtype))
        tok = tok + torch.einsum("bjs,jsd->bjd", side.float(), self.side)
        summ = torch.stack([self.sum_row(rs.float()), self.sum_orig(ob.float())], 1) + self.sum_type
        x = self.tok_norm(torch.cat([tok, summ], 1))
        return torch.cat([self.cls.expand(len(x), -1, -1), x], 1)            # CLS + 21 + 2 = 24 token

    def forward(self, code, xnum, side, rs, ob, rtd=False):
        # autocast đặt trong forward: DataParallel chạy mỗi bản sao trong luồng riêng, autocast bên ngoài không có tác dụng
        with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=DEVICE == "cuda"):
            return self._forward(code, xnum, side, rs, ob, rtd)

    def _forward(self, code, xnum, side, rs, ob, rtd):
        x = self.tokens(code, xnum, side, rs, ob)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if rtd:
            return self.rtd_head(x[:, 1:1 + NF]).squeeze(-1)                 # (b, 21): giá trị bị tráo hay thật
        return self.head(x[:, 1:].mean(1)).squeeze(-1)                       # TB 23 token không phải CLS


def param_groups(model, lr_bb, lr_tok):
    bb = [p for n_, p in model.named_parameters() if n_.startswith(("blocks.", "norm."))]
    tok = [p for n_, p in model.named_parameters() if not n_.startswith(("blocks.", "norm."))]
    return [{"params": bb, "lr": lr_bb}, {"params": tok, "lr": lr_tok}]


def schedule(opt, total):
    warm = max(1, int(WARMUP * total))
    f = lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm)) if s >= warm else (s + 1) / warm
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


# toàn bộ dữ liệu lên GPU (fp16 cho số thực) để khỏi tốn thời gian chép mỗi batch
dt = torch.float16 if DEVICE == "cuda" else torch.float32
G = {"code": torch.as_tensor(codes, device=DEVICE), "xnum": torch.as_tensor(XNUM, dtype=dt, device=DEVICE),
     "side": torch.as_tensor(SIDE, dtype=dt, device=DEVICE), "rs": torch.as_tensor(ROWSTAT, dtype=dt, device=DEVICE),
     "ob": torch.as_tensor(ORIGB, dtype=dt, device=DEVICE)}
del SIDE
yt = torch.as_tensor(np.r_[y, np.zeros(N - n_tr)], dtype=torch.float32, device=DEVICE)


def batch(idx):
    return G["code"][idx], G["xnum"][idx], G["side"][idx], G["rs"][idx], G["ob"][idx]


def wrap(model):
    return nn.DataParallel(model) if DEVICE == "cuda" and torch.cuda.device_count() > 1 else model


def train_steps(net, opt, sched, scaler, loss_fn, order, n_steps, tag, monitor=None):
    net.train()
    t0, done, run_loss, run_gn, n_run = time.time(), 0, 0.0, 0.0, 0
    for i in range(n_steps):
        b = order[i * BATCH:(i + 1) * BATCH]
        loss = loss_fn(net, b)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        done += len(b)
        run_loss, run_gn, n_run = run_loss + loss.item(), run_gn + gn.item(), n_run + 1
        if (i + 1) % 500 == 0 or i + 1 == n_steps:
            rate = done / (time.time() - t0)
            msg = f"{tag} bước {i + 1}/{n_steps} loss TB {run_loss / n_run:.4f} grad norm TB {run_gn / n_run:.3g}"
            if monitor is not None:
                msg += f" AUC theo dõi {monitor():.5f}"
                net.train()
            log(msg + f" ({rate:,.0f} dòng/s)")
            run_loss, run_gn, n_run = 0.0, 0.0, 0
    return done / (time.time() - t0)


def rtd_loss(net, b):
    code, xnum, side, rs, ob = batch(b)
    m = torch.rand(len(b), NF, device=DEVICE) < RTD_P
    donor = torch.as_tensor(np.random.randint(0, N, len(b)), device=DEVICE)
    dc, dx, ds, _, _ = batch(donor)
    code2 = torch.where(m, dc, code)
    xnum2 = torch.where(m, dx, xnum)
    side2 = torch.where(m.unsqueeze(-1), ds, side)
    label = (m & (code2 != code)).float()                                     # tráo trúng cùng giá trị thì vẫn là "thật"
    out = net(code2, xnum2, side2, rs, ob, rtd=True)
    return F.binary_cross_entropy_with_logits(out.float(), label)


def ft_loss(net, b):
    return F.binary_cross_entropy_with_logits(net(*batch(b)).float(), yt[b])


@torch.no_grad()
def predict(net, idx):
    net.eval()
    out = []
    for i in range(0, len(idx), EVAL_BATCH):
        b = torch.as_tensor(idx[i:i + EVAL_BATCH], device=DEVICE)
        out.append(torch.sigmoid(net(*batch(b)).float()).cpu().numpy())
    return np.concatenate(out)


def new_scaler():
    return torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda" and AMP_DTYPE == torch.float16)


OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------- pretrain RTD (một lần, không nhãn: train + test + gốc) ----------------
prev = glob.glob(f"/kaggle/input/**/{RTD_FILE}", recursive=True) if os.path.isdir("/kaggle/input") else []
model = QFT().to(DEVICE)
log(f"tham số: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M, GPU: {torch.cuda.device_count()}")
if prev and not FAST:
    model.load_state_dict(torch.load(prev[0], map_location=DEVICE))
    log(f"dùng lại trọng số RTD {prev[0]}")
else:
    net = wrap(model)
    spe = N // BATCH
    total = FAST_RTD_STEPS if FAST else RTD_EPOCHS * spe
    opt = torch.optim.AdamW(param_groups(model, LR_BB_RTD, LR_TOK_RTD), weight_decay=WD)
    sched, scaler = schedule(opt, total), new_scaler()
    order = torch.cat([torch.randperm(N, device=DEVICE) for _ in range(math.ceil(total / spe) + 1)])
    rate = train_steps(net, opt, sched, scaler, rtd_loss, order, total, "RTD")
    if FAST:
        log(f"ƯỚC: RTD {RTD_EPOCHS} epoch × {N:,} dòng ≈ {RTD_EPOCHS * N / rate / 60:.0f} phút")
    else:
        torch.save(model.state_dict(), f"{OUT}/{RTD_FILE}")
        log(f"đã lưu {RTD_FILE}")
rtd_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
del model

# ---------------- fine-tune theo fold ----------------
oof, test_pred = np.zeros(n_tr), np.zeros(n_te)
te_idx = np.arange(n_tr, n_tr + n_te)
run_folds = [0] if FAST else range(5)
done_folds = []
for k in run_folds:
    part = f"{OUT}/{NAME}_fold{k}.npz"                                     # v3: tên theo NAME, không lẫn fold của bản khác
    prev_k = glob.glob(f"/kaggle/input/**/{NAME}_fold{k}.npz", recursive=True) if os.path.isdir("/kaggle/input") else []
    va = np.where(fold == k)[0]
    if prev_k and not FAST:                                                   # fold đã xong ở lần chạy trước
        z = np.load(prev_k[0])
        oof[va], test_pred = z["oof"], test_pred + z["test"] / 5
        np.savez(part, oof=z["oof"], test=z["test"])
        done_folds.append(k)
        log(f"fold {k}: dùng lại {prev_k[0]}, AUC = {roc_auc_score(y[va], oof[va]):.6f}")
        continue
    torch.manual_seed(SEED + k)
    tr_idx = np.where(fold != k)[0]
    model = QFT().to(DEVICE)
    model.load_state_dict(rtd_state)
    nn.init.normal_(model.head.weight, std=0.02)
    nn.init.zeros_(model.head.bias)
    net = wrap(model)
    spe = len(tr_idx) // BATCH
    total = FAST_FT_STEPS if FAST else FT_EPOCHS * spe
    opt = torch.optim.AdamW(param_groups(model, LR_BB, LR_TOK), weight_decay=WD)
    sched, scaler = schedule(opt, total), new_scaler()
    tr_t = torch.as_tensor(tr_idx, device=DEVICE)
    order = torch.cat([tr_t[torch.randperm(len(tr_t), device=DEVICE)] for _ in range(FT_EPOCHS + 1)])
    mon = np.random.default_rng(k).choice(va, min(MON_ROWS, len(va)), replace=False)
    rate = train_steps(net, opt, sched, scaler, ft_loss, order, total, f"fold {k}",
                       monitor=lambda: roc_auc_score(y[mon], predict(net, mon)))
    p_va = predict(net, va)
    auc_k = roc_auc_score(y[va], p_va)
    log(f"fold {k}: AUC = {auc_k:.6f}" + (" (FAST, ít bước)" if FAST else ""))
    # v1 hỏng (0.75–0.87) mà vẫn chạy hết 5 fold: dừng sớm để khỏi phí quota GPU (không ảnh hưởng OOF vì không lưu gì)
    assert FAST or SMOKE or auc_k > 0.93, f"fold {k} AUC {auc_k:.4f} quá thấp → dừng, xem log"
    if FAST:
        log(f"ƯỚC: fine-tune {FT_EPOCHS} epoch × {len(tr_idx):,} dòng × 5 fold ≈ "
            f"{5 * FT_EPOCHS * len(tr_idx) / rate / 60:.0f} phút (chưa tính dự đoán)")
        break
    p_te = predict(net, te_idx)
    oof[va], test_pred = p_va, test_pred + p_te / 5
    np.savez(part, oof=p_va, test=p_te)
    done_folds.append(k)
    del model, net, opt
    torch.cuda.empty_cache() if DEVICE == "cuda" else None

if FAST:
    log("FAST: chỉ đo tốc độ, không lưu")
else:
    assert len(done_folds) == 5 and not np.isnan(oof).any() and not np.isnan(test_pred).any()
    log(f"{NAME}: CV AUC = {roc_auc_score(y, oof):.6f}")
    np.save(f"{OUT}/oof_{NAME}.npy", oof)
    np.save(f"{OUT}/test_{NAME}.npy", test_pred)
    log("đã lưu oof / test")
