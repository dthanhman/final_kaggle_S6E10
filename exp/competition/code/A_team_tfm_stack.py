"""Stack của team, có và không có foundation model công khai (cho bảng ablation trong báo cáo).

So bằng nested LR (meta seed 7/42/101) trên logit chuẩn hoá:
  team        : model sạch của team (oof/oof_<tên>.npy; thêm B_*, quan_* khi có file)
  tfm         : foundation model công khai: 9 TFM của Chris Deotte (data/public_oof_tfm) + TabPFN/TabICL của goodpjw
  team+tfm    : cả hai
  +mlp        : thêm 42 MLP công khai (RealMLP, DeepFM) trong bộ 124 OOF của Chris Deotte (data/public_oof_c124)
  +c124       : team + cả bộ 124 (gồm ~70 cây) + 9 TFM của Chris (gần với pool đầy đủ)
Lưu stack team+tfm (test = TB 5 fold-model của meta seed 42) thành stack_team_tfm.

Chạy: python notebooks/A_team_tfm_stack.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils import save_preds  # noqa: E402

OOF, DATA = ROOT / "oof", ROOT / "data"
TEAM = ["A_lgb_v2", "A_lr_v2", "A_knn_v2", "A_fm_v1", "A_ftt_v1", "A_glr_v1", "A_tabm_v1", "A_gptc_v1", "A_kozy_v1",
        "A_lgb_origrow_v1", "B_realmlp_v1", "quan_xgb_safe_teacher_v4", "quan_xgb_dual_teacher_v4",
        "quan_cat_dual_teacher_v4", "quan_xgb_original_aug_v4",
        "B_personal_spec_v1"]   # KHÔNG sạch: dừng sớm trên fold đang chấm (như phần lớn pool công khai)
TFM_CHRIS = ["causilo", "exaone", "kumolarge", "kumosmall", "limix2", "mitra2", "tabfm", "tabicl2", "tabpfn35"]
TFM_GP = ["gp_tabicl_full_raw", "gp_tabpfn_catfd", "gp_tabpfn_catfd10", "gp_tabpfn_full_raw", "gp_tabpfn_te",
          "gp_tabpfn_te4"]

y = pd.read_csv(DATA / "train.csv", usecols=["satisfaction"]).satisfaction.astype(int).to_numpy()


def logit(v):
    v = np.asarray(v, dtype=np.float64)
    if v.min() < 0 or v.max() > 1:
        return v
    v = np.clip(v, 1e-6, 1 - 1e-6)
    return np.log(v / (1 - v))


def load(pairs):
    cols, tcols, names = [], [], []
    for name, o, t in pairs:
        if o.exists() and t.exists():
            cols.append(logit(np.load(o))); tcols.append(logit(np.load(t))); names.append(name)
        else:
            print(f"  chưa có file: {name}")
    return names, np.column_stack(cols), np.column_stack(tcols)


team = load([(n, OOF / f"oof_{n}.npy", OOF / f"test_{n}.npy") for n in TEAM])
tfm = load([(f"tfm_{n}", DATA / "public_oof_tfm" / f"oof_{n}.npy", DATA / "public_oof_tfm" / f"test_{n}.npy")
            for n in TFM_CHRIS] +
           [(n, DATA / "public_oof_1710" / "c124_gp" / f"oof_{n}.npy", DATA / "public_oof_1710" / "c124_gp" / f"test_{n}.npy")
            for n in TFM_GP])
C124 = DATA / "public_oof_c124"
c124_names = sorted(p.name[4:-4] for p in C124.glob("oof_*.npy"))
mlp = load([(n, C124 / f"oof_{n}.npy", C124 / f"test_{n}.npy") for n in c124_names
            if any(k in n for k in ("realmlp", "rmlp", "deepfm"))])
c124 = load([(n, C124 / f"oof_{n}.npy", C124 / f"test_{n}.npy") for n in c124_names])
chris = load([(f"tfm_{n}", DATA / "public_oof_tfm" / f"oof_{n}.npy", DATA / "public_oof_tfm" / f"test_{n}.npy")
              for n in TFM_CHRIS])
print(f"team: {len(team[0])} model | foundation: {len(tfm[0])} | mlp: {len(mlp[0])} | c124: {len(c124[0])}")


def nested(X, XT, seed):
    mu, sd = X.mean(0), X.std(0)
    X, XT = (X - mu) / sd, (XT - mu) / sd
    o, t = np.zeros(len(y)), np.zeros(len(XT))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m = LogisticRegression(C=1.0, max_iter=3000).fit(X[a], y[a])
        o[b] = m.decision_function(X[b]); t += m.decision_function(XT) / 5
    return o, t


cat = lambda *gs: (np.hstack([g[1] for g in gs]), np.hstack([g[2] for g in gs]))
SETS = {"team": team[1:], "tfm": tfm[1:], "team+tfm": cat(team, tfm), "tfm+mlp": cat(tfm, mlp),
        "team+tfm+mlp": cat(team, tfm, mlp), "c124+tfm9": cat(c124, chris), "team+c124+tfm9": cat(team, c124, chris)}
best = {}
for name, (X, XT) in SETS.items():
    aucs = []
    for seed in (7, 42, 101):
        o, t = nested(X, XT, seed)
        aucs.append(roc_auc_score(y, o))
        if seed == 42:
            best[name] = (o, t)
    print(f"{name:9s} ({X.shape[1]:2d} model): " + " / ".join(f"{a:.6f}" for a in aucs) + f"  (seed 7/42/101)")
fw = np.load(OOF / "oof_stack_fw_noforge.npy")
print(f"tham chiếu stack công khai fw_noforge (~240 model): {roc_auc_score(y, fw):.6f}")
sig = lambda v: 1 / (1 + np.exp(-v))
save_preds("stack_team_tfm", sig(best["team+tfm"][0]), sig(best["team+tfm"][1]), y)
