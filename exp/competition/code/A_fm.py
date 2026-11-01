"""Factorization Machine (Rendle 2010): member khác hẳn cây. Mỗi giá trị của mỗi cột là một token có
trọng số tuyến tính + vector ẩn; tương tác cặp giữa mọi token = tích vô hướng vector ẩn.
  logit = b + sum w_i + 1/2 * sum_f [(sum_i v_if)^2 - sum_i v_if^2]

Token: giá trị thô của 21 cột (giá trị < --min-freq dòng trên train + test gộp thành "hiếm" theo cột)
       + 4 chữ số của Age / Flight Distance. Không dùng dữ liệu gốc, không TE.
OOF sạch: dừng sớm trên 10% tách từ phần train của fold; test = trung bình 5 mô hình fold. Chạy CPU.

Chạy: python notebooks/A_fm.py --folds 0 --name A_fm_try
      python notebooks/A_fm.py --name A_fm_v1
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from config import FEATURES, N_FOLDS, NUM_COLS, TARGET
from data import load_data
from utils import save_preds

AGE, FD = NUM_COLS[:2]
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def tokenize(train, test, min_freq):
    """Ma trận chỉ số token (n_dòng x n_trường), đánh số chung cho mọi trường."""
    A = pd.concat([train[FEATURES], test[FEATURES]], ignore_index=True)
    fields = {c: A[c].astype(str) for c in FEATURES}
    for c in (AGE, FD):
        v = A[c].astype(np.int64)
        for k in range(4):
            fields[f"{c}_d{k}"] = (v // 10 ** k % 10).astype(str)
    cols, offset = [], 0
    for c, s in fields.items():
        vc = s.value_counts()
        s = s.where(s.map(vc) >= min_freq, "__rare__")
        codes, uniq = pd.factorize(s)
        cols.append(codes + offset)
        offset += len(uniq)
    T = np.stack(cols, 1).astype(np.int64)
    return T[:len(train)], T[len(train):], offset


class FM(torch.nn.Module):
    def __init__(self, n_tok, dim):
        super().__init__()
        self.w = torch.nn.Embedding(n_tok, 1)
        self.v = torch.nn.Embedding(n_tok, dim)
        self.b = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.zeros_(self.w.weight)
        torch.nn.init.normal_(self.v.weight, std=0.01)

    def forward(self, x):
        v = self.v(x)
        inter = 0.5 * (v.sum(1).pow(2) - v.pow(2).sum(1)).sum(1)
        return self.b + self.w(x).squeeze(-1).sum(1) + inter


def predict(m, X, bs=65536):
    m.eval()
    with torch.no_grad():
        return np.concatenate([m(torch.from_numpy(X[i:i + bs])).numpy() for i in range(0, len(X), bs)])


def train_fold(X_fit, y_fit, X_es, y_es, n_tok, args):
    torch.manual_seed(args.seed)
    m = FM(n_tok, args.dim)
    opt = torch.optim.Adam(m.parameters(), lr=args.lr)
    Xt, yt = torch.from_numpy(X_fit), torch.from_numpy(y_fit.astype(np.float32))
    best, best_state, bad = -1, None, 0
    for ep in range(args.epochs):
        m.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), args.bs):
            idx = perm[i:i + args.bs]
            xb = Xt[idx]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(m(xb), yt[idx])
            loss = loss + args.wd * (m.v(xb).pow(2).sum(-1).mean() + m.w(xb).pow(2).mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
        auc = roc_auc_score(y_es, predict(m, X_es))
        if auc > best:
            best, bad, best_state = auc, 0, {k: t.clone() for k, t in m.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    m.load_state_dict(best_state)
    return m, best, ep + 1 - bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-5, help="L2 trên embedding của token trong batch")
    ap.add_argument("--bs", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--min-freq", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(N_FOLDS)), help="chỉ lưu khi chạy đủ 5 fold")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    if not args.name.startswith(("A_", "B_", "C_")):
        ap.error("tên phải dạng <người>_<model>_<phiên bản>")
    torch.set_num_threads(4)

    train, test = load_data()
    y, fold = train[TARGET].values, train["fold"].values
    X, X_test, n_tok = tokenize(train, test, args.min_freq)
    log(f"{X.shape[1]} trường, {n_tok} token")

    oof, test_pred = np.zeros(len(y)), np.zeros(len(X_test))
    for k in args.folds:
        va = fold == k
        tr_idx = np.where(fold != k)[0]
        es_idx = np.random.default_rng(42).choice(tr_idx, size=len(tr_idx) // 10, replace=False)
        fit_idx = np.setdiff1d(tr_idx, es_idx)
        m, es_auc, n_ep = train_fold(X[fit_idx], y[fit_idx], X[es_idx], y[es_idx], n_tok, args)
        oof[va] = predict(m, X[va])
        test_pred += predict(m, X_test) / N_FOLDS
        log(f"fold {k}: AUC = {roc_auc_score(y[va], oof[va]):.6f} | dừng sớm {es_auc:.6f} | {n_ep} epoch")

    if len(args.folds) == N_FOLDS:
        log(f"CV AUC = {roc_auc_score(y, oof):.6f}")
        save_preds(args.name, 1 / (1 + np.exp(-oof)), 1 / (1 + np.exp(-test_pred)), y)
