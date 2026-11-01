"""Generator-aware LR (ý heuljax, S6E9: bước nhảy +1e-4 vào stack của mọi đội top). Dữ liệu do LM sinh, số được viết
theo token GPT-2, nên tỉ lệ hài lòng đổi theo ranh giới token chứ không theo ranh giới thập phân.

Chạy từ thư mục gốc repo:
  python notebooks/A_glr.py --folds 0 --name A_glr_try        (thử fold 0, không lưu)
  python notebooks/A_glr.py --name A_glr_v1

Feature = toàn bộ feature của A_lr.py (one-hot, số, target encoding) cộng thêm phần token:
  - chuỗi số đúng như trong CSV (" 694", " 18.0") tách bằng tokenizer GPT-2 (tiktoken) -> khóa token đầu, 2 token đầu,
    token cuối + số token, số token; one-hot (giá trị hiếm gộp chung) và target encoding
  - ô token × ngữ cảnh (Type of Travel × Class × Customer Type) cho token đầu và 2 token đầu -> target encoding
  - chuỗi beta phân cấp: token đầu -> 2 token đầu -> giá trị đúng, mỗi cấp co về cấp trên (prior 5 / 20 / 80) -> logit
OOF sạch: mọi thống kê dùng nhãn chỉ fit trên 4 fold train (dòng train lấy giá trị cross-fit 5 phần bên trong);
không early stopping; không dùng dữ liệu gốc ngoài phần A_lr đã có (teacher / tỉ lệ trên gốc).
"""
import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import scipy.sparse as sp
import tiktoken
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from A_lgb_fe import DIGIT_COLS, add_te
from A_lr import dense_static, logit, static_features
from config import DATA_DIR, N_FOLDS, NUM_COLS, TARGET
from data import load_data
from utils import save_preds

CTX = ["Type of Travel", "Class", "Customer Type"]
CHAIN_PRIOR = (5.0, 20.0, 80.0)
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def dense_csr(D):
    """Ma trận dày -> CSR không qua nonzero() (đỡ ~1 GB RAM với 560k dòng)."""
    D = np.ascontiguousarray(D, dtype=np.float32)
    r, c = D.shape
    return sp.csr_matrix((D.ravel(), np.tile(np.arange(c, dtype=np.int32), r),
                          np.arange(0, r * c + 1, c, dtype=np.int64)), shape=(r, c))


def token_keys(raw, ctx):
    """raw: DataFrame chuỗi số đúng như CSV. Trả về (khóa one-hot, khóa TE, 3 cấp chuỗi beta cho mỗi cột số)."""
    enc = tiktoken.get_encoding("gpt2")
    oh, te, chains = {}, {}, {}
    for c in NUM_COLS:
        s = raw[c].fillna("nan")
        toks = {v: [enc.decode([t]) for t in enc.encode(" " + v)] for v in s.unique()}
        t1 = s.map(lambda v: toks[v][0])
        t12 = s.map(lambda v: "|".join(toks[v][:2]))
        n = s.map(lambda v: str(len(toks[v])))
        last = s.map(lambda v: toks[v][-1]) + "#" + n
        oh.update({f"{c}_t1": t1, f"{c}_t12": t12, f"{c}_last": last, f"{c}_ntok": n})
        te.update({f"{c}_t1": t1, f"{c}_t12": t12, f"{c}_last": last,
                   f"{c}_t1_ctx": t1 + "@" + ctx, f"{c}_t12_ctx": t12 + "@" + ctx})
        chains[c] = (t1.values, t12.values, s.values)
    return pd.DataFrame(oh), pd.DataFrame(te), chains


def chain_rate(levels, y, fit_idx, apply_idx, prior):
    """Tỉ lệ co phân cấp: mỗi cấp co về tỉ lệ cấp trên với độ mạnh CHAIN_PRIOR. Thống kê chỉ từ fit_idx."""
    out, parent = [], np.full(len(apply_idx), prior)
    for key, a in zip(levels, CHAIN_PRIOR):
        g = pd.DataFrame({"k": key[fit_idx], "y": y[fit_idx]}).groupby("k").y.agg(["sum", "count"])
        s = pd.Series(key[apply_idx]).map(g["sum"]).fillna(0).values
        n = pd.Series(key[apply_idx]).map(g["count"]).fillna(0).values
        parent = (s + a * parent) / (n + a)
        out.append(parent)
    return np.column_stack(out)


def chain_features(chains, y, tr, others):
    """Cross-fit 5 phần cho dòng train của fold; dòng khác dùng thống kê toàn bộ tr. Trả về [F_tr, *F_others] (logit)."""
    prior = y[tr].mean()
    F_tr = np.zeros((len(tr), 3 * len(chains)))
    for itr, iva in StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y[tr]):
        F_tr[iva] = np.hstack([chain_rate(lv, y, tr[itr], tr[iva], prior) for lv in chains.values()])
    F_oth = [np.hstack([chain_rate(lv, y, tr, idx, prior) for lv in chains.values()]) for idx in others]
    return [logit(F) for F in [F_tr] + F_oth]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=float, default=0.05)
    ap.add_argument("--min-freq", type=int, default=20, help="giá trị one-hot ít hơn số dòng này thì gộp chung")
    ap.add_argument("--no-chain", action="store_true", help="bỏ chuỗi beta (đối chứng)")
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(N_FOLDS)), help="chỉ lưu khi chạy đủ 5 fold")
    ap.add_argument("--name", required=True, help="tên dạng A_<model>_<phiên bản>")
    args = ap.parse_args()
    if not args.name.startswith(("A_", "B_", "C_")):
        ap.error("tên phải dạng <người>_<model>_<phiên bản>, ví dụ A_glr_v1")

    train, test = load_data()
    y, fold = train[TARGET].values, train["fold"].values
    n, n_test = len(train), len(test)
    X, X_test, K, K_test = static_features(train, test)

    # chuỗi số đúng như trong CSV (Arrival Delay là float "18.0")
    raw = pd.concat([pd.read_csv(DATA_DIR / f, usecols=NUM_COLS, dtype=str) for f in ("train.csv", "test.csv")],
                    ignore_index=True)
    allc = pd.concat([train[CTX], test[CTX]], ignore_index=True).astype(str)
    ctx = allc[CTX[0]] + "|" + allc[CTX[1]] + "|" + allc[CTX[2]]
    OH_tok, TE_tok, chains = token_keys(raw, ctx)
    log(f"token: {OH_tok.shape[1]} khóa one-hot, {TE_tok.shape[1]} khóa TE, chuỗi beta {len(chains)} cột")

    digits = [f"{c}_d{k}" for c in DIGIT_COLS for k in range(4)]
    oh_all = pd.concat([pd.concat([K, X[digits]], axis=1), pd.concat([K_test, X_test[digits]], axis=1)],
                       ignore_index=True)
    oh_all = pd.concat([oh_all.astype(str), OH_tok], axis=1)
    ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=args.min_freq, dtype=np.float32)
    S_all = ohe.fit_transform(oh_all.values).tocsr()
    S, S_test = S_all[:n], S_all[n:]
    del S_all, oh_all, OH_tok

    Dn, Dn_test = dense_static(X), dense_static(X_test)
    sc = StandardScaler().fit(pd.concat([Dn, Dn_test], ignore_index=True))
    Dn, Dn_test = sc.transform(Dn), sc.transform(Dn_test)
    KT = pd.concat([K.reset_index(drop=True).astype(str), TE_tok.iloc[:n].reset_index(drop=True)], axis=1)
    KT_test = pd.concat([K_test.reset_index(drop=True).astype(str), TE_tok.iloc[n:].reset_index(drop=True)], axis=1)
    chains_tr = {c: tuple(lv[:n] for lv in v) for c, v in chains.items()}
    chains_te = {c: tuple(lv[n:] for lv in v) for c, v in chains.items()}
    log(f"{S.shape[1]} cột one-hot + {Dn.shape[1]} cột số + {2 * KT.shape[1]} cột TE"
        + ("" if args.no_chain else f" + {3 * len(chains)} cột chuỗi beta"))

    empty, empty_test = pd.DataFrame(index=range(n)), pd.DataFrame(index=range(n_test))
    oof, test_pred = np.zeros(n), np.zeros(n_test)
    for k in args.folds:
        tr, va = np.where(fold != k)[0], np.where(fold == k)[0]
        T_tr, T_va, T_te = add_te(empty.iloc[tr], KT.iloc[tr], y[tr], [(empty.iloc[va], KT.iloc[va]), (empty_test, KT_test)])
        T_tr, T_va, T_te = (logit(t.values) for t in (T_tr, T_va, T_te))
        if not args.no_chain:
            # chuỗi beta: test dùng khóa của test, nên tính riêng (thống kê vẫn chỉ từ tr)
            C_tr, C_va = chain_features(chains_tr, y, tr, [va])
            prior = y[tr].mean()
            C_te = logit(np.hstack([chain_rate([np.r_[lv_tr[tr], lv_te] for lv_tr, lv_te in zip(chains_tr[c], chains_te[c])],
                                               np.r_[y[tr], np.zeros(n_test)], np.arange(len(tr)),
                                               np.arange(len(tr), len(tr) + n_test), prior) for c in chains_tr]))
            T_tr, T_va, T_te = np.hstack([T_tr, C_tr]), np.hstack([T_va, C_va]), np.hstack([T_te, C_te])
        tsc = StandardScaler().fit(T_tr)
        # dựng từng ma trận khi cần rồi xóa ngay để giữ đỉnh RAM thấp (máy 12 GB)
        M_tr = sp.hstack([S[tr], dense_csr(Dn[tr]), dense_csr(tsc.transform(T_tr))], format="csr")
        m = LogisticRegression(C=args.C, max_iter=3000, tol=1e-5).fit(M_tr, y[tr])
        del M_tr
        gc.collect()
        M_va = sp.hstack([S[va], dense_csr(Dn[va]), dense_csr(tsc.transform(T_va))], format="csr")
        oof[va] = m.decision_function(M_va)
        del M_va
        M_te = sp.hstack([S_test, dense_csr(Dn_test), dense_csr(tsc.transform(T_te))], format="csr")
        test_pred += m.decision_function(M_te) / N_FOLDS
        del M_te, T_tr, T_va, T_te
        gc.collect()
        log(f"fold {k}: AUC = {roc_auc_score(y[va], oof[va]):.6f} | {m.n_iter_[0]} vòng lặp")

    if len(args.folds) == N_FOLDS:
        log(f"C {args.C}: CV AUC = {roc_auc_score(y, oof):.6f}")
        save_preds(args.name, 1 / (1 + np.exp(-oof)), 1 / (1 + np.exp(-test_pred)), y)
