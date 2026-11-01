"""Thử nhanh TabPFN-3.5 với thiết lập chưa có trong pool công khai: chỉ fold 0.

Mục đích: (1) đo bộ nhớ GPU đỉnh theo số dòng context để biết 16 GB chứa được bao nhiêu, ngoại suy cho 560k;
(2) lấy dự đoán fold 0 của từng thiết lập để đo trên máy xem có làm tăng stack công khai không
(notebooks/A_tfm_probe_eval.py). Không phải member: chỉ có fold 0, không dự đoán test.

Chạy trên Kaggle GPU (Internet ON: pip install tabpfn, tải trọng số Prior-Labs/tabpfn_3_5 từ Hugging Face), gắn:
  - competition playground-series-s6e10
  - dataset arseniyshutko/binary-aviation-satisfaction-129k   (dữ liệu gốc, data.csv)
  - dataset thanhmn123/cs116-s6e10-oof                        (folds.csv)
Trọng số TabPFN-3.5: license tabpfn-3-5-license-v1.0, chỉ dùng cho nghiên cứu / phi thương mại.
Output: probe_<run>.npy (xác suất trên các dòng fold 0, thứ tự như train.csv), probe_results.json.

Thiết lập (theo các tùy chọn trong script công khai của goodpjw2008, dataset s6e10-tabpfn-member):
  raw50k / raw100k : feature thô, context 50k / 100k (đo bộ nhớ + đối chứng)
  catrat100k       : 13 cột điểm khai báo là category + thống kê gộp điểm (ragg)
  ogctx100k        : 100k dòng cuộc thi + toàn bộ 129k dòng dữ liệu gốc trong context (gốc chỉ ở phía train)
Vòng 1 (15/10, T4): mọi bản trên đều không tăng stack trên fold 0; ogctx ít tương quan nhất (-0..-4.5e-6).
Vòng 2 (PROBE_ROUND=2, mặc định): context đầy đủ của fold (ctx=0, ~560k dòng)
  rawfull     : đối chứng (tương đương tabpfn_full_raw công khai), đo bộ nhớ T4 với full context
  ogctx0full  : + các dòng gốc có cả hai delay = 0 (delay gốc lệch mạnh: 56% bằng 0 so với 91% ở cuộc thi)
  ogctxfull   : + toàn bộ dòng gốc
  -> cả 3 hết bộ nhớ T4 (14,6 GB) ngay lúc fit.
Vòng 3 (PROBE_ROUND=3, mặc định): KV cache để ở RAM CPU (keep_cache_on_device=False), thêm bản context 300k dự phòng
"""
import glob
import inspect
import json
import os
import subprocess
import sys
import time

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tabpfn"], check=True)

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier

T0 = time.time()
TARGET = "satisfaction"
CATS = ["Gender", "Customer Type", "Type of Travel", "Class"]
SCORES = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking", "Gate location",
          "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment", "On-board service",
          "Leg room service", "Baggage handling", "Checkin service", "Cleanliness"]
NUMS = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
FEATS = CATS + SCORES + NUMS
INFLIGHT = ["Inflight wifi service", "Food and drink", "Seat comfort", "Inflight entertainment", "On-board service"]
PREFLIGHT = ["Departure/Arrival time convenient", "Ease of Online booking", "Gate location", "Online boarding",
             "Checkin service"]
RUNS = {1: [("raw50k", "raw", 50_000), ("raw100k", "raw", 100_000),
            ("catrat100k", "catrat", 100_000), ("ogctx100k", "ogctx", 100_000)],
        2: [("rawfull", "raw", 0), ("ogctx0full", "ogctx0", 0), ("ogctxfull", "ogctx", 0)],
        3: [("rawfull", "raw", 0), ("ogctx0full", "ogctx0", 0), ("ogctx0_300k", "ogctx0", 300_000)],
        }[int(os.environ.get("PROBE_ROUND", 3))]
CACHE_ON_GPU = os.environ.get("PROBE_ROUND", "3") in ("1", "2")


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def find(name, hint):
    hits = [p for p in glob.glob(f"/kaggle/input/**/{name}", recursive=True) if hint in p]
    assert hits, f"không thấy {name} ({hint}): gắn input"
    return sorted(hits, key=len)[0]


tr = pd.read_csv(find("train.csv", "playground-series-s6e10"))
og = pd.read_csv(find("data.csv", "aviation"))
folds = pd.read_csv(find("folds.csv", "cs116-s6e10-oof")).set_index("id").loc[tr["id"], "fold"].to_numpy()
y = tr[TARGET].astype(int).to_numpy()
yog = og[TARGET].astype(str).str.lower().map({"true": 1, "false": 0})
assert yog.notna().all(), "nhãn dữ liệu gốc lạ"
yog = yog.astype(int).to_numpy()
A_ALL, B = np.where(folds != 0)[0], np.where(folds == 0)[0]
log(f"train {len(tr):,}, gốc {len(og):,} (tỉ lệ dương {yog.mean():.3f}), fold 0: {len(B):,} dòng chấm, "
    f"GPU {torch.cuda.get_device_name(0)} {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")


def encode(df, vocab):
    X = df[FEATS].copy()
    for c in CATS:
        X[c] = pd.Categorical(X[c], categories=vocab[c]).codes
    X["Arrival Delay in Minutes"] = X["Arrival Delay in Minutes"].fillna(-1)
    return X


VOCAB = {c: sorted(pd.concat([tr[c], og[c]]).dropna().unique()) for c in CATS}
XTR, XOG = encode(tr, VOCAB), encode(og, VOCAB)


def add_ragg(X):
    X = X.copy()
    r = X[SCORES]
    X["r_mean"], X["r_std"], X["r_min"], X["r_max"] = r.mean(1), r.std(1), r.min(1), r.max(1)
    X["r_zero"] = (r == 0).sum(1)
    X["r_inflight"], X["r_preflight"] = X[INFLIGHT].mean(1), X[PREFLIGHT].mean(1)
    return X


CKPT = hf_hub_download("Prior-Labs/tabpfn_3_5", "tabpfn-v3.5-20260909.safetensors", local_dir="/tmp/ckpt")   # không để trong working: output sẽ kèm file trọng số
SIG = inspect.signature(TabPFNClassifier.__init__).parameters
log(f"tabpfn {__import__('tabpfn').__version__}; tham số có: {sorted(SIG)}")


def make_clf(cat_idx):
    kw = dict(model_path=CKPT, n_estimators=1, random_state=0, device="cuda", ignore_pretraining_limits=True,
              fit_mode="fit_with_cache", kv_cache_precision="int8", inference_precision="autocast",
              categorical_features_indices=cat_idx, memory_saving_mode="auto",
              keep_cache_on_device=CACHE_ON_GPU)
    dropped = [k for k in kw if k not in SIG]
    if dropped:
        log(f"  bản tabpfn này không có tham số {dropped}: bỏ qua")
    return TabPFNClassifier(**{k: v for k, v in kw.items() if k in SIG})


def predict(clf, Xq, batch):
    out = np.empty(len(Xq), np.float32)
    i = 0
    while i < len(Xq):
        try:
            out[i:i + batch] = clf.predict_proba(Xq[i:i + batch])[:, 1]
            i += batch
            if i // batch % 5 == 0:
                log(f"    đã dự đoán {min(i, len(Xq)):,} / {len(Xq):,} dòng")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            assert batch > 256, "hết bộ nhớ cả với batch 256"
            batch //= 2
            log(f"    OOM khi dự đoán: giảm batch còn {batch}")
    return out, batch


RESULTS = {}
for run, mode, ctx in RUNS:
    t, clf = time.time(), None
    try:
        a = np.sort(np.random.RandomState(0).choice(A_ALL, ctx, replace=False)) if ctx else A_ALL
        Xa, Xb, ya = XTR.iloc[a], XTR.iloc[B], y[a]
        cat_idx = list(range(len(CATS)))
        if mode == "catrat":
            Xa, Xb = add_ragg(Xa), add_ragg(Xb)
            cat_idx += [Xa.columns.get_loc(c) for c in SCORES]
        if mode == "ogctx":
            Xa, ya = pd.concat([Xa, XOG]), np.r_[ya, yog]
        if mode == "ogctx0":
            z = ((XOG["Departure Delay in Minutes"] == 0) & (XOG["Arrival Delay in Minutes"] == 0)).to_numpy()
            Xa, ya = pd.concat([Xa, XOG[z]]), np.r_[ya, yog[z]]
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        clf = make_clf(cat_idx)
        clf.fit(Xa.to_numpy(np.float32), ya)
        fit_peak = torch.cuda.max_memory_allocated() / 2**30
        log(f"{run}: context {len(Xa):,} dòng, {Xa.shape[1]} cột, fit {time.time() - t:.0f}s, đỉnh {fit_peak:.2f} GB")
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        p, batch = predict(clf, Xb.to_numpy(np.float32), 8192)
        pred_peak = torch.cuda.max_memory_allocated() / 2**30
        auc = roc_auc_score(y[B], p)
        np.save(f"/kaggle/working/probe_{run}.npy", p)
        RESULTS[run] = dict(ctx_rows=len(Xa), n_cols=Xa.shape[1], fold0_auc=auc, fit_peak_gb=fit_peak,
                            pred_peak_gb=pred_peak, pred_batch=batch, fit_s=t1 - t, pred_s=time.time() - t1)
        log(f"{run}: fold 0 AUC {auc:.6f} | dự đoán {len(B):,} dòng {time.time() - t1:.0f}s (batch {batch}), "
            f"đỉnh {pred_peak:.2f} GB")
    except Exception as e:
        RESULTS[run] = dict(error=f"{type(e).__name__}: {e}"[:300])
        log(f"{run}: LỖI {RESULTS[run]['error']}")
    del clf
    torch.cuda.empty_cache()
    json.dump(RESULTS, open("/kaggle/working/probe_results.json", "w"), indent=1)

ok = {k: v for k, v in RESULTS.items() if "error" not in v and k.startswith("raw")}
if len(ok) == 2:
    (n1, r1), (n2, r2) = sorted(((v["ctx_rows"], v) for v in ok.values()), key=lambda x: x[0])
    for key in ("fit_peak_gb", "pred_peak_gb"):
        slope = (r2[key] - r1[key]) / (n2 - n1)
        log(f"ngoại suy tuyến tính {key}: 560k dòng ≈ {r2[key] + slope * (560_000 - n2):.1f} GB "
            f"({slope * 1e5:.2f} GB / 100k dòng)")
log("xong")
