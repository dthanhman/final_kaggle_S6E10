"""Thăm dò TabDPT trên fold 0 (họ foundation model duy nhất chưa có trong pool; context lấy bằng retrieval kNN nên
dùng được toàn bộ ~560k dòng train của fold, khác TabPFN bị giới hạn bộ nhớ).

Không phải member: chỉ fold 0, không dự đoán test. Đo trên máy bằng
  python notebooks/A_tfm_probe_eval.py <thư mục output> --stack stack_fw_noforge
Chỉ đáng chạy đủ 5 fold nếu tương quan với stack thấp (< ~0.95) mà AUC riêng >= ~0.959 và tăng stack cả 3 seed.

Chạy trên Kaggle GPU (Internet ON: pip install tabdpt, trọng số tự tải từ Hugging Face), gắn:
  - competition playground-series-s6e10
  - dataset thanhmn123/cs116-s6e10-oof (folds.csv)
Output: probe_<run>.npy (xác suất trên các dòng fold 0, thứ tự như train.csv), probe_results.json.
"""
import glob
import inspect
import json
import subprocess
import sys
import time

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tabdpt"], check=True)

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tabdpt import TabDPTClassifier

T0, BUDGET = time.time(), 10.5 * 3600
TARGET = "satisfaction"
CATS = ["Gender", "Customer Type", "Type of Travel", "Class"]
SCORES = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking", "Gate location",
          "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment", "On-board service",
          "Leg room service", "Baggage handling", "Checkin service", "Cleanliness"]
NUMS = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
FEATS = CATS + SCORES + NUMS
# (tên, context_size, n_ensembles); bản sau chỉ chạy nếu còn thời gian theo tốc độ đo được
RUNS = [("ctx2k_e1", 2048, 1), ("ctx2k_e4", 2048, 4)]


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def find(name, hint):
    hits = [p for p in glob.glob(f"/kaggle/input/**/{name}", recursive=True) if hint in p]
    assert hits, f"không thấy {name} ({hint}): gắn input"
    return sorted(hits, key=len)[0]


tr = pd.read_csv(find("train.csv", "playground-series-s6e10"))
folds = pd.read_csv(find("folds.csv", "cs116-s6e10-oof")).set_index("id").loc[tr["id"], "fold"].to_numpy()
y = tr[TARGET].astype(int).to_numpy()
X = tr[FEATS].copy()
for c in CATS:
    X[c] = pd.Categorical(X[c]).codes
X["Arrival Delay in Minutes"] = X["Arrival Delay in Minutes"].fillna(-1)
X = X.to_numpy(np.float32)
A, B = np.where(folds != 0)[0], np.where(folds == 0)[0]
log(f"train {len(tr):,}, fold 0: {len(B):,} dòng chấm, context {len(A):,} dòng | "
    f"GPU {torch.cuda.get_device_name(0)} {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")

INIT = inspect.signature(TabDPTClassifier.__init__).parameters
PRED = inspect.signature(TabDPTClassifier.predict_proba).parameters
log(f"tabdpt: __init__ {sorted(INIT)} | predict_proba {sorted(PRED)}")
kw = {k: v for k, v in dict(device="cuda", use_flash=False, compile=False).items() if k in INIT}
clf = TabDPTClassifier(**kw)
t = time.time()
torch.cuda.reset_peak_memory_stats()
clf.fit(X[A], y[A])
fit_s, fit_peak = time.time() - t, torch.cuda.max_memory_allocated() / 2**30
log(f"fit {fit_s:.0f}s, đỉnh {fit_peak:.2f} GB")


def predict(Xq, ctx, ens, batch=4096):
    pk = {k: v for k, v in dict(context_size=ctx, n_ensembles=ens, seed=0, temperature=0.8).items() if k in PRED}
    out, i = np.empty(len(Xq), np.float32), 0
    while i < len(Xq):
        try:
            out[i:i + batch] = clf.predict_proba(Xq[i:i + batch], **pk)[:, 1]
            i += batch
            if i // batch % 5 == 0:
                log(f"    {min(i, len(Xq)):,} / {len(Xq):,} dòng")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            assert batch > 64, "hết bộ nhớ cả với batch 64"
            batch //= 2
            log(f"    OOM: giảm batch còn {batch}")
    return out, batch


RESULTS = {}
for run, ctx, ens in RUNS:
    # đo tốc độ trên 4096 dòng rồi ước lượng thời gian cho cả fold 0; bỏ bản nếu không kịp
    t = time.time()
    predict(X[B[:4096]], ctx, ens)
    eta = (time.time() - t) / 4096 * len(B)
    left = BUDGET - (time.time() - T0)
    log(f"{run}: ước tính {eta / 60:.0f} phút cho fold 0, còn {left / 60:.0f} phút")
    if eta > left:
        RESULTS[run] = dict(error=f"không kịp: cần {eta / 60:.0f} phút, còn {left / 60:.0f}")
        log(f"{run}: bỏ qua"); continue
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    p, batch = predict(X[B], ctx, ens)
    auc = roc_auc_score(y[B], p)
    np.save(f"/kaggle/working/probe_{run}.npy", p)
    RESULTS[run] = dict(ctx_rows=len(A), context_size=ctx, n_ensembles=ens, fold0_auc=auc, fit_peak_gb=fit_peak,
                        pred_peak_gb=torch.cuda.max_memory_allocated() / 2**30, pred_batch=batch, pred_s=time.time() - t)
    log(f"{run}: fold 0 AUC {auc:.6f} | {time.time() - t:.0f}s")
    json.dump(RESULTS, open("/kaggle/working/probe_results.json", "w"), indent=1)
json.dump(RESULTS, open("/kaggle/working/probe_results.json", "w"), indent=1)
log("xong")
