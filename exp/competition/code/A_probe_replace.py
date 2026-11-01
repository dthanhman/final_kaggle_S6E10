"""Thay member cũ trong pool bằng bản mới có giúp không (17/10).

Họ goodpjw2008: bản cũ = gp_* trong c124 (pool nền của P247); bản mới = oof_members.csv của 2 notebook chạy lại
(data/public_oof_1710/goodpjw2008__*) và các biến thể _k/_k10 trong gp_extra (bản 17/10).
  1. So từng cặp cùng tên: AUC cũ/mới, tương quan rank.
  2. Mô phỏng thay thế trong họ: LR (C=0.003, chuẩn hoá) trên logit họ cũ / họ đã thay / cũ + mới, meta seed 42/7/101;
     và trên [logit stack_fw_p247] + họ đó (đo phần thêm so với nền).
Chạy: python notebooks/A_probe_replace.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from blend import to_logit  # noqa: E402

PUB = os.path.join(ROOT, "data", "public_oof_1710")
train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=["id", "satisfaction"])
test_ids = pd.read_csv(os.path.join(ROOT, "data", "test.csv"), usecols=["id"])["id"].to_numpy()
y = train["satisfaction"].astype(int).to_numpy()
folds = pd.read_csv(os.path.join(ROOT, "data", "folds.csv"))
SEEDS = [42, 7, 101]
FOLDS = {42: folds["fold"].to_numpy()}
for s in SEEDS[1:]:
    f = np.zeros(len(y), int)
    for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=s).split(y, y)):
        f[va] = k
    FOLDS[s] = f


def logit(o, t):
    lo, _ = to_logit(np.asarray(o, np.float64).ravel(), np.asarray(t, np.float64).ravel())
    return lo.astype(np.float64)


# ---- bản cũ (c124)
old = {}
for f in sorted(glob.glob(os.path.join(PUB, "c124_gp", "oof_gp_*.npy"))):
    n = os.path.basename(f)[7:-4]
    old[n] = logit(np.load(f), np.load(os.path.join(os.path.dirname(f), "test_" + os.path.basename(f)[4:])))
# ---- bản mới: notebook chạy lại (file catboost-ctr chứa đủ cột) + gp_extra
new = {}
d = os.path.join(PUB, "goodpjw2008__s6e10-catboost-ctr-glm-margin-xgb-lb-0-96127")
o, t = pd.read_csv(f"{d}/oof_members.csv"), pd.read_csv(f"{d}/test_members.csv")
for c in [c for c in o.columns if c != "id"]:
    new[c] = logit(o.set_index("id")[c].reindex(train["id"]), t.set_index("id")[c].reindex(test_ids))
for f in sorted(glob.glob(os.path.join(PUB, "gp_extra", "oof_*.npy"))):
    n = os.path.basename(f)[4:-4]
    if n != "stack245":
        # chỉ đổi tên file: thư mục public_oof_1710 cũng chứa "oof_"
        new["x_" + n] = logit(np.load(f), np.load(os.path.join(os.path.dirname(f), "test_" + os.path.basename(f)[4:])))
print(f"cũ {len(old)} | mới {len(new)}")

# cặp: cùng tên; gp_extra ghép theo tên gốc trước hậu tố _k/_k10/_aux*
PAIR = {"x_xgb_k10": "xgb_v3", "x_xgb_d10_k10": "xgb_v3_d10", "x_lgb_k10": "lgb_v3", "x_lgb_v3_k": "lgb_v3",
        "x_cat_k10": "cat_v3", "x_cat_ctr_k": "cat_ctr", "x_glm6_xgb_k": "glm6_xgb", "x_realmlp_fa_s7_k": "realmlp_fa_s7",
        "x_realmlp_fa_aux3_k": "realmlp_fa_aux3", "x_realmlp_pub_aux_k": "realmlp_pub_aux",
        "x_realmlp_v4fa_auxfull": "realmlp_v4fa", "x_realmlp_vb": "realmlp_pub"}
pairs = [(n, n) for n in new if n in old] + [(n, m) for n, m in PAIR.items() if n in new and m in old]

rows = []
for nn, on in pairs:
    a_new, a_old = roc_auc_score(y, new[nn]), roc_auc_score(y, old[on])
    rows.append(dict(new=nn, old=on, auc_old=a_old, auc_new=a_new, d=(a_new - a_old) * 1e6,
                     rho=np.corrcoef(rankdata(new[nn]), rankdata(old[on]))[0, 1]))
R = pd.DataFrame(rows).sort_values("d", ascending=False)
pd.set_option("display.width", 200)
print("\n1. Từng cặp (d = AUC mới − cũ, ×1e-6)")
print(R.to_string(index=False, formatters={"auc_old": "{:.6f}".format, "auc_new": "{:.6f}".format,
                                            "d": "{:+.1f}".format, "rho": "{:.5f}".format}))
print("\nmember mới không có cặp:", ", ".join(n for n in new if n not in {p[0] for p in pairs}))


def lr_cv(X, f, C=0.003):
    o = np.zeros(len(y))
    for k in range(5):
        tr, va = f != k, f == k
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=C, tol=1e-7, max_iter=5000).fit(sc.transform(X[tr]), y[tr])
        o[va] = m.decision_function(sc.transform(X[va]))
    return o


def score(cols, extra=None):
    X = np.column_stack(([extra] if extra is not None else []) + cols)
    return [roc_auc_score(y, lr_cv(X, FOLDS[s])) for s in SEEDS]


repl = dict(old)
for nn, on in pairs:   # mỗi tên cũ lấy bản mới có AUC cao nhất trong các cặp của nó
    best = R[R.old == on].iloc[0]
    repl[on] = new[best.new]
sets = {"cũ": list(old.values()), "đã thay": list(repl.values()), "cũ + mới": list(old.values()) + list(new.values())}
bo = np.load(os.path.join(ROOT, "oof", "oof_stack_fw_p247.npy"))
bt = np.load(os.path.join(ROOT, "oof", "test_stack_fw_p247.npy"))
B = logit(bo, bt)

print("\n2. LR C=0.003 trên họ goodpjw2008 (AUC theo meta seed 42/7/101; d = so với 'cũ', ×1e-6)")
for base_name, extra in [("chỉ họ", None), ("nền stack_fw_p247 + họ", B)]:
    ref = None
    for nm, cols in sets.items():
        a = score(cols, extra)
        ref = ref or a
        print(f"  {base_name:24s} {nm:9s} ({len(cols):2d} cột): " + " ".join(f"{v:.6f}" for v in a)
              + " | d " + " ".join(f"{(v - r) * 1e6:+.1f}" for v, r in zip(a, ref)))
