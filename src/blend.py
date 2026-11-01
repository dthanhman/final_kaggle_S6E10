"""Blend / stack nhiều model trên OOF.

Chạy: python src/blend.py lgb mlp                     -> AUC từng model, tương quan, rank-blend đều
      python src/blend.py lgb mlp --weights 2 1       -> rank-blend có trọng số
      python src/blend.py --method lr --C 0.003       -> stack LR trên mọi oof_*.npy trong OOF_DIR
      python src/blend.py --method lr --meta-seed 7 101  -> đối chiếu stack trên meta fold seed khác
      thêm --save <tên> để lưu oof_<tên>.npy, test_<tên>.npy

Không ghi tên model thì lấy mọi oof_*.npy trong OOF_DIR, trừ file bắt đầu bằng blend_ / stack_
(để không đưa bản blend cũ vào lại). Nên đặt --save là blend_<...> hoặc stack_<...>.

Trước khi blend, tự loại member (tắt bằng --no-filter):
- OOF AUC < 0,90: nhiều khả năng ghép sai thứ tự dòng hoặc sai nhãn.
- OOF AUC > 0,962: cao hơn mọi model đơn công khai (khoảng 0,9617), nhiều khả năng là một stack/blend
  đã fit trên toàn bộ train. Đưa vào làm member thì nested CV cao giả (xem notebook
  references/s6e10-0-96188-lb-l2-family-wise-actually-helps.ipynb).
- Gần trùng (tương quan rank > 0,9999) với một member AUC cao hơn: không thêm thông tin.

Stack LR đưa mỗi member về thang logit (xác suất -> logit, logit giữ nguyên, rank -> logit của rank),
nên member lưu dạng nào cũng dùng được. Mặc định meta fold = folds.csv; --meta-seed chỉ để
kiểm tra độ ổn định, chọn cấu hình theo kết quả ổn định qua nhiều seed.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import DATA_DIR, ID, N_FOLDS, OOF_DIR, TARGET
from utils import load_preds, save_preds

EXCLUDE_PREFIXES = ("blend_", "stack_")
MIN_AUC, MAX_AUC, DUP_RHO = 0.90, 0.962, 0.9999


def to_rank(p):
    return rankdata(p) / len(p)


def to_logit_rank(p):
    r = rankdata(p) / (len(p) + 1)
    return np.log(r / (1 - r)).astype(np.float32)


def is_rank(p):
    """Giá trị trải đều trên (0, 1] như rankdata(p) / len(p)."""
    q = np.quantile(p, [0.1, 0.25, 0.5, 0.75, 0.9])
    return bool(p.min() >= 0 and p.max() <= 1 and np.abs(q - [0.1, 0.25, 0.5, 0.75, 0.9]).max() < 0.01)


def to_logit(oof, test):
    """Đưa một member về thang logit: giữ nguyên nếu đã là logit, logit(xác suất) nếu là xác suất,
    logit(rank) nếu member lưu dưới dạng rank. Dùng logit(rank) cho xác suất làm stack kém đi."""
    if min(oof.min(), test.min()) < 0 or max(oof.max(), test.max()) > 1:
        return np.clip(oof, -30, 30).astype(np.float32), np.clip(test, -30, 30).astype(np.float32)
    if is_rank(oof):
        return to_logit_rank(oof), to_logit_rank(test)
    lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7))).astype(np.float32)
    return lg(oof), lg(test)


def rank_blend(preds, weights=None):
    weights = np.ones(len(preds)) if weights is None else np.asarray(weights, dtype=float)
    return sum(w * to_rank(p) for w, p in zip(weights, preds)) / weights.sum()


def lr_stack(X, y, X_test, folds, C=0.003):
    """Stack LR: mỗi fold k train trên các fold còn lại, dự đoán fold k; test = trung bình các fold."""
    oof, test_pred, coefs = np.zeros(len(y)), np.zeros(len(X_test)), []
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        # solver hội tụ hẳn: chênh lệch giữa các cấu hình stack chỉ cỡ 1e-6
        m = LogisticRegression(C=C, tol=1e-6, max_iter=20000).fit(X[tr], y[tr])
        oof[va] = m.decision_function(X[va])
        test_pred += m.decision_function(X_test) / len(np.unique(folds))
        coefs.append(m.coef_[0])
    return oof, test_pred, np.mean(coefs, axis=0)


def load_y_folds():
    train = pd.read_csv(DATA_DIR / "train.csv", usecols=[ID, TARGET])
    folds = pd.read_csv(DATA_DIR / "folds.csv")
    assert folds[ID].equals(train[ID]), "folds.csv không khớp thứ tự id của train.csv"
    return train[TARGET].astype(int).values, folds["fold"].values


def meta_folds(y, seed):
    folds = np.zeros(len(y), dtype=int)
    for k, (_, va) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(y, y)):
        folds[va] = k
    return folds


def filter_members(names, aucs, oofs):
    """Trả về chỉ số member giữ lại và lý do loại từng member bị bỏ."""
    dropped = {}
    for i, a in enumerate(aucs):
        if a < MIN_AUC:
            dropped[i] = f"AUC {a:.4f} < {MIN_AUC}: nghi ghép sai thứ tự dòng / nhãn"
        elif a > MAX_AUC:
            dropped[i] = f"AUC {a:.6f} > {MAX_AUC}: nghi là stack/blend, làm CV cao giả"
    order = [i for i in np.argsort(-np.asarray(aucs)) if i not in dropped]
    if len(order) > 1:
        R = np.column_stack([to_rank(oofs[i]).astype(np.float32) for i in order])
        corr = np.corrcoef(R, rowvar=False)
        for a in range(len(order)):
            if order[a] in dropped:
                continue
            for b in range(a + 1, len(order)):
                if order[b] not in dropped and corr[a, b] > DUP_RHO:
                    dropped[order[b]] = f"gần trùng {names[order[a]]} (tương quan rank {corr[a, b]:.6f})"
    return [i for i in range(len(names)) if i not in dropped], dropped


def all_names():
    names = sorted(p.stem[len("oof_"):] for p in OOF_DIR.glob("oof_*.npy"))
    return [n for n in names if not n.startswith(EXCLUDE_PREFIXES)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="bỏ trống = mọi oof_*.npy trong OOF_DIR")
    ap.add_argument("--method", choices=["rank", "lr"], default="rank")
    ap.add_argument("--weights", nargs="+", type=float, help="chỉ dùng với --method rank")
    ap.add_argument("--C", type=float, default=0.003, help="regularization của stack LR")
    ap.add_argument("--meta-seed", nargs="+", type=int, help="đối chiếu stack LR trên meta fold seed khác")
    ap.add_argument("--no-filter", action="store_true", help="không tự loại member")
    ap.add_argument("--save")
    args = ap.parse_args()

    names = args.names or all_names()
    assert names, f"không tìm thấy oof_*.npy nào trong {OOF_DIR}"
    y, folds = load_y_folds()
    oofs, tests = zip(*(load_preds(n) for n in names))
    aucs = [roc_auc_score(y, o) for o in oofs]

    print(f"{len(names)} member")
    for n, a in sorted(zip(names, aucs), key=lambda t: -t[1]):
        print(f"  {n:<30} CV AUC = {a:.6f}")

    if not args.no_filter:
        keep, dropped = filter_members(names, aucs, oofs)
        if dropped:
            print(f"\nLoại {len(dropped)} member (dùng --no-filter để giữ):")
            for i, why in dropped.items():
                print(f"  {names[i]:<30} {why}")
        assert keep, "không còn member nào sau khi lọc"
        if args.weights and len(keep) < len(names):
            ap.error("có member bị loại nên --weights không còn khớp; bỏ member đó khỏi danh sách hoặc dùng --no-filter")
        names, oofs, tests = [names[i] for i in keep], [oofs[i] for i in keep], [tests[i] for i in keep]
        print(f"\nCòn {len(names)} member")
    if 1 < len(names) <= 15:
        ranks = np.column_stack([to_rank(o) for o in oofs])
        print("\nTương quan rank giữa các OOF:")
        print(pd.DataFrame(np.corrcoef(ranks, rowvar=False), index=names, columns=names).round(4))

    if args.method == "rank":
        oof_b, test_b = rank_blend(oofs, args.weights), rank_blend(tests, args.weights)
        print(f"\n{'RANK BLEND':<32} CV AUC = {roc_auc_score(y, oof_b):.6f}")
    else:
        Z = [to_logit(o, t) for o, t in zip(oofs, tests)]
        X, X_test = np.column_stack([z[0] for z in Z]), np.column_stack([z[1] for z in Z])
        oof_b, test_b, coef = lr_stack(X, y, X_test, folds, args.C)
        print(f"\n{f'LR STACK C={args.C}':<32} CV AUC = {roc_auc_score(y, oof_b):.6f} (meta fold = folds.csv)")
        for s in args.meta_seed or []:
            o, _, _ = lr_stack(X, y, X_test, meta_folds(y, s), args.C)
            print(f"{'':<32} CV AUC = {roc_auc_score(y, o):.6f} (meta fold seed {s})")
        oof_b, test_b = 1 / (1 + np.exp(-oof_b)), 1 / (1 + np.exp(-test_b))  # lưu dạng xác suất
        top = np.argsort(-np.abs(coef))[:15]
        print("\nHệ số lớn nhất:")
        for i in top:
            print(f"  {names[i]:<30} {coef[i]:+.4f}")

    if args.save:
        save_preds(args.save, oof_b, test_b, y)
