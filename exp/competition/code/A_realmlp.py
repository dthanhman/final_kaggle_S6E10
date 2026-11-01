"""RealMLP-TD (pytabkit) chỉnh cho AUC, dựa trên notebooks/B_realmlp_pytabkit.ipynb của B (B_realmlp_v1, CV 0,957695).

Khác bản của B:
  - use_ls=False (B để mặc định: label smoothing 0,1) và chọn epoch tốt nhất theo cross_entropy (mặc định: class_error);
    tài liệu pytabkit khuyên đúng hai chỉnh này khi đo bằng AUC
  - batch 256 (mặc định pytabkit, lr đã chỉnh cho batch này), 64 epoch (B: 2048, 16), trung bình 3 seed
    (bản thử batch 1024 / 32 epoch: fold 0 AUC 0,957810, không hơn B 0,957868)
Giữ nguyên bộ feature của B (Mean/Min/Max/Zero_Count điểm dịch vụ, log delay, log distance) vì đó là phần làm nó
khác các RealMLP công khai (rho 0,94 với bản gần nhất).

OOF sạch (docs/STACKING.md mục 5): fold theo folds.csv; epoch tốt nhất chọn trên 10% tách từ phần train của fold.
Chạy trên Kaggle GPU (Internet ON để pip install pytabkit), input: competition playground-series-s6e10,
dataset thanhmn123/cs116-s6e10-oof (folds.csv). FAST=1: chỉ fold 0, seed 0 (đo AUC và thời gian).
Output: /kaggle/working/oof_A_realmlp_v1.npy, test_A_realmlp_v1.npy (xác suất); lưu lại sau mỗi seed.
"""
import glob
import os
import subprocess
import sys
import time

if os.path.isdir("/kaggle/input"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytabkit"], check=True)

import numpy as np
import pandas as pd
import torch
from pytabkit import RealMLP_TD_Classifier
from sklearn.metrics import roc_auc_score

FAST = os.environ.get("FAST") == "1"
NAME = "A_realmlp_v1"
SEEDS = [0] if FAST else [0, 1, 2]
FOLDS = [0] if FAST else [0, 1, 2, 3, 4]
BATCH, EPOCHS, TIME_BUDGET = 256, 64, 10.5 * 3600
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
T0 = time.time()

TARGET, ID = "satisfaction", "id"
CATS = ["Gender", "Customer Type", "Type of Travel", "Class"]
SCORES = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking", "Gate location",
          "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment", "On-board service",
          "Leg room service", "Baggage handling", "Checkin service", "Cleanliness"]


def find(name):
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True), key=len)
    assert hits, f"không thấy {name}"
    return hits[0]


train, test = pd.read_csv(find("train.csv")), pd.read_csv(find("test.csv"))
y = train[TARGET].astype(str).str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0}).to_numpy().astype(int)
fo = pd.read_csv(find("folds.csv"))
assert (fo[ID].to_numpy() == train[ID].to_numpy()).all(), "folds.csv không khớp thứ tự id"
folds = fo["fold"].to_numpy()
print(f"train {len(train):,} | test {len(test):,} | {DEVICE} {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else ''}",
      flush=True)


def prep(df):      # giống preprocess_for_realmlp của B
    df = df.copy()
    df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(df["Departure Delay in Minutes"]).fillna(0)
    arr, dep = df["Arrival Delay in Minutes"], df["Departure Delay in Minutes"]
    df["Total_Delay"] = np.log1p(np.maximum(arr + dep, 0))
    df["Log_Distance"] = np.log1p(df["Flight Distance"])
    df["Mean_Score"] = df[SCORES].mean(axis=1)
    df["Min_Score"] = df[SCORES].min(axis=1)
    df["Max_Score"] = df[SCORES].max(axis=1)
    df["Zero_Count"] = (df[SCORES] == 0).sum(axis=1).astype(float)
    for c in CATS:
        df[c] = pd.Categorical(df[c], categories=sorted(train[c].unique()))
    return df[[c for c in df.columns if c not in (ID, TARGET)]]


X, XT = prep(train), prep(test)
print("feature:", list(X.columns), flush=True)
oof_sum, test_sum, n_done = np.zeros(len(X)), np.zeros(len(XT)), 0
for seed in SEEDS:
    if time.time() - T0 > TIME_BUDGET:
        print(f"hết giờ, dừng trước seed {seed}", flush=True)
        break
    oof, tst = np.zeros(len(X)), np.zeros(len(XT))
    for k in FOLDS:
        t0 = time.time()
        va, tr = np.where(folds == k)[0], np.where(folds != k)[0]
        perm = np.random.default_rng(100 * seed + k).permutation(tr)
        n_es = len(tr) // 10
        fit_idx, es_idx = perm[n_es:], perm[:n_es]       # epoch tốt nhất chọn trên es_idx, fold k không tham gia
        Xf = pd.concat([X.iloc[fit_idx], X.iloc[es_idx]], axis=0)
        yf = np.concatenate([y[fit_idx], y[es_idx]])
        clf = RealMLP_TD_Classifier(device=DEVICE, random_state=100 * seed + k, batch_size=BATCH, n_epochs=EPOCHS,
                                    use_ls=False, val_metric_name="cross_entropy", verbosity=0)
        clf.fit(Xf, yf, val_idxs=np.arange(len(fit_idx), len(Xf)))
        oof[va] = clf.predict_proba(X.iloc[va])[:, 1]
        tst += clf.predict_proba(XT)[:, 1] / len(FOLDS)
        print(f"seed {seed} fold {k}: AUC {roc_auc_score(y[va], oof[va]):.6f} | {time.time() - t0:.0f}s "
              f"(tổng {(time.time() - T0) / 60:.0f} phút)", flush=True)
        del clf
        torch.cuda.empty_cache()
    if FAST:
        break
    oof_sum += oof; test_sum += tst; n_done += 1
    print(f"seed {seed}: CV {roc_auc_score(y, oof):.6f} | TB {n_done} seed: CV {roc_auc_score(y, oof_sum / n_done):.6f}",
          flush=True)
    np.save(f"{WORK}/oof_{NAME}.npy", oof_sum / n_done)
    np.save(f"{WORK}/test_{NAME}.npy", test_sum / n_done)
print(f"xong, {(time.time() - T0) / 60:.1f} phút", flush=True)
