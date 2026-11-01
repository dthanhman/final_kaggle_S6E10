"""CS116 | stack v4: pool sạch 247 member + nguồn riêng của nhóm + OOF nhóm; LR family-wise và blend 0.8 LR + 0.2 LightGBM.

Phần nạp dữ liệu và kiểm tra member (mục 1-2) lấy nguyên từ notebook
"S6E10 | 0.96188 LB | L2 family-wise actually helps" (ewtherka, dựa trên loader của Dariush Afshar),
bản tải về: references/s6e10-0-96188-lb-l2-family-wise-actually-helps.ipynb. Thay đổi: bỏ gpx_stack245,
thêm các nguồn có trong stack v3 của nhóm (EXTRA_KEYS) và OOF của nhóm (cs116-s6e10-oof), phần thí nghiệm ở cuối.
File này được ghép tự động; sửa phần của nhóm rồi ghép lại.
"""

# ==== 1. Setup (bản gốc) ====

import os, glob, json, time, hashlib, warnings
from collections import deque
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")

T0 = time.time()
INPUT = os.environ.get("SMOKE_ROOT", "/kaggle/input")
WORK = os.environ.get("SMOKE_WORK", "/kaggle/working")
N_JOBS = min(5, os.cpu_count() or 1)


def find_dir(slug, root=INPUT, max_depth=6):
    # the mounted folder called `slug` (a dataset, a notebook's output or the competition), or None
    queue, hits = deque([(root, 0)]), []
    while queue:
        d, depth = queue.popleft()
        try:
            entries = sorted(os.scandir(d), key=lambda e: e.name)
        except OSError:
            continue
        for e in entries:
            if e.is_dir():
                if e.name == slug:
                    hits.append(e.path)
                elif depth < max_depth:
                    queue.append((e.path, depth + 1))
    return sorted(hits, key=len)[0] if hits else None


COMP = find_dir("playground-series-s6e10")
train = pd.read_csv(f"{COMP}/train.csv", usecols=["id", "satisfaction"])
train_ids = train["id"].to_numpy()
test_ids = pd.read_csv(f"{COMP}/test.csv", usecols=["id"])["id"].to_numpy()
y = train["satisfaction"].astype(str).str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0})
assert y.notna().all(), "unexpected label values"
y = y.to_numpy().astype(int)

# the folds every comparison below is scored on; most public pairs were built on this same split
fold = np.zeros(len(y), int)
for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y)):
    fold[va] = k
LIB = find_dir("s6e10-golem-oof-library")
assert (np.load(f"{LIB}/folds_seed42.npy") == fold).all(), "not the frozen folds"
print(f"train {len(y):,} rows, test {len(test_ids):,} rows, positive rate {y.mean():.4f}")
print(f"folds: StratifiedKFold(5, shuffle=True, random_state=42), {np.bincount(fold).tolist()} rows per fold, "
      "identical to folds_seed42.npy in my OOF library")


# ==== 2. Nguồn OOF (bản gốc) ====

SOURCES = []    # every public source in loading order: its key, the mounted folder's slug, and how to read it


def source(key, slug):
    def register(fn):
        SOURCES.append({"key": key, "slug": slug, "load": fn})
        return fn
    return register


def first(d, name):
    hits = sorted(glob.glob(os.path.join(d, "**", name), recursive=True), key=len)
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


def by_id(df, ids, col):
    # column `col` of `df` in the order of `ids`; every id has to be there exactly once
    s = df.set_index("id")[col]
    assert s.index.is_unique, "duplicate ids"
    s = s.reindex(ids)
    assert s.notna().all(), f"{int(s.isna().sum()):,} ids missing"
    return s.to_numpy(np.float64)


def npy_pairs(d, prefix, pattern="oof_*.npy"):
    # oof_<name>.npy with test_<name>.npy next to it, rows in train.csv / test.csv order
    for f in sorted(glob.glob(os.path.join(d, "**", pattern), recursive=True)):
        name = os.path.basename(f)[4:-4]
        yield prefix + name, np.load(f), np.load(os.path.join(os.path.dirname(f), f"test_{name}.npy")), "row order"


def npz_pairs(d, prefix, pattern="*.npz", strip=0):
    # one .npz per member holding `oof` and `test`
    for f in sorted(glob.glob(os.path.join(d, "**", pattern), recursive=True)):
        with np.load(f) as z:
            if "oof" in z.files and "test" in z.files:
                yield prefix + os.path.basename(f)[strip:-4], z["oof"], z["test"], "row order"


# ---- the base pool: Chris Deotte's two packaged datasets
@source("c124", "s6e10-124-public-oof-and-test-predictions")
def load_c124(d):        # 124 pairs Kratosyan collected from every public library of the first week
    yield from npy_pairs(d, "")


@source("tfm", "s6e10-tfm-oof-and-test-predictions")
def load_tfm(d):         # 9 tabular foundation models, a tuned XGBoost and a RealMLP
    yield from npy_pairs(d, "tfm_")


# ---- published since, or not in the 124
@source("golem", "s6e10-golem-oof-library")
def load_golem(d):       # my library: 14 of its 17 are already in the 124, so the near-copy check drops them
    man = pd.read_csv(f"{d}/manifest.csv")
    for r in man.itertuples():
        yield "gl_" + r.wide_column, np.load(f"{d}/{r.oof_file}"), np.load(f"{d}/{r.test_file}"), "row order"


@source("abdullah", "oof-dataset")
def load_abdullah(d):    # three TabPFN-3.5 feature sets, CatBoost, XGBoost (15 folds each); csv files with ids
    for m in ("tabpfn_catfd10", "tabpfn_raw", "tabpfn_te_cnt", "cat_v3", "xgb_v3"):
        o, t = pd.read_csv(f"{d}/{m}_oof.csv"), pd.read_csv(f"{d}/{m}_test.csv")
        yield "ab_" + m, by_id(o, train_ids, "pred"), by_id(t, test_ids, "pred"), "id"


@source("sadam", "ps-s6e10-airline-satisfaction-artifacts")
def load_sadam(d):       # 38 experiments, plus 2 models trained only on the original data (they never saw a train label)
    yield from npz_pairs(d, "st_", "preds_*.npz", strip=6)
    for n in ("orig_cat_pred", "orig_model_pred"):
        with np.load(first(d, f"{n}.npz")) as z:
            yield "st_" + n, z["train"], z["test"], "row order"


@source("forge", "s6e10-forge-own-pairs")
def load_forge(d):       # The Consensus Forge: 6 pairs on its own seed-33 folds
    yield from npy_pairs(d, "am_", "oof_forge_*.npy")


@source("sachith", "s6e10-stack-oof-predictions")
def load_sachith(d):     # 24 members; 22 are already in the 124
    yield from npz_pairs(d, "s7_")


@source("koumeimaki", "s6e10-route-tfm-nn-oof")
def load_koumeimaki(d):  # TabPFN, TabICL, TabM, RealMLP and three GBDTs on shared folds; ids in two side files
    assert (pd.read_csv(f"{d}/train_ids_folds.csv")["id"].to_numpy() == train_ids).all()
    assert (pd.read_csv(f"{d}/test_ids.csv")["id"].to_numpy() == test_ids).all()
    for name, o, t, _ in npy_pairs(d, "km_"):
        yield name, o, t, "id"


@source("gp_tabpfn", "s6e10-tabpfn-member")
def load_gp_tabpfn(d):   # TabPFN-3.5 and TabICL members; csv files with ids (all 8 should already be in the 124)
    for f in sorted(glob.glob(f"{d}/oof_*.csv")):
        n = os.path.basename(f)[4:-4]
        yield ("gpt_" + n, by_id(pd.read_csv(f), train_ids, "pred"),
               by_id(pd.read_csv(f"{d}/test_{n}.csv"), test_ids, "pred"), "id")


@source("gp_extra", "s6e10-goodpjw2008-extra-members-oof")
def load_gp_extra(d):    # 17 members of the goodpjw2008 series that were never published before
    for name, o, t, how in npy_pairs(d, "gpx_"):
        if "stack" not in name:    # CS116: bỏ gpx_stack245 (stack của stack, nested CV lạc quan giả)
            yield name, o, t, how


@source("dmitriy", "s6e10-catboost-all-categorical-oof-and-test")
def load_dmitriy(d):     # CatBoost with every column categorical
    yield "dk_cb_cat", np.load(first(d, "oof_cb_cat.npy")), np.load(first(d, "test_cb_cat.npy")), "row order"


@source("kagan_abl", "s6e10-ablation-members-oof")
def load_kagan_abl(d):   # 17 ablation members: LightGBM, XGBoost, CatBoost and RealMLP on several feature sets
    yield from npy_pairs(d, "kga_")


# ---- notebook outputs
@source("yekenot", "ps-s6-e10-realmlp-pytabkit")
def load_yekenot(d):     # RealMLP (PyTabKit); csv files with ids
    o, t = pd.read_csv(first(d, "oof_preds.csv")), pd.read_csv(first(d, "submission.csv"))
    yield "yk_realmlp", by_id(o, train_ids, "satisfaction"), by_id(t, test_ids, "satisfaction"), "id"


@source("hermengardo", "single-catboost-eda-ps6e10")
def load_hermengardo(d):  # a single CatBoost
    yield "hm_cat", np.load(first(d, "oof_train.npy")), np.load(first(d, "oof_test.npy")), "row order"


@source("kratosyan", "s6e10-xgboost-realmlp-route-id-teacher")
def load_kratosyan(d):   # 10-fold XGBoost and RealMLP with a route-ID teacher
    yield from npz_pairs(d, "kr_", "*_route_ogaux.npz")


@source("kagan_enc", "s6e10-encodings-catboost-realmlp-lb-0-961")
def load_kagan_enc(d):   # LightGBM, XGBoost, two CatBoosts, RealMLP; csv files with ids
    o, t = pd.read_csv(first(d, "oof_members.csv")), pd.read_csv(first(d, "test_members.csv"))
    for c in [c for c in o.columns if c != "id"]:
        yield "kg_" + c, by_id(o, train_ids, c), by_id(t, test_ids, c), "id"


@source("aman_v5", "s6e10-the-original-prior-stack-v5")
def load_aman_v5(d):     # the five base members of The Original-Prior Stack v5 (its blends and stack are left out)
    with np.load(first(d, "oof_artifacts.npz")) as z:
        assert (z["y"] == y).all(), "labels in a different row order"
        for m in ("lgbm_champion", "lgbm_te", "lgbm_te_rows", "xgb_gpu", "catboost_gpu"):
            yield "av5_" + m, z["oof_" + m], z["test_" + m], "row order"


@source("kozy", "s6e10-source-transfer-foundation-model-oof")
def load_kozy(d):        # a LightGBM on engineered features (its output's model_oof_report.csv names the model)
    yield ("kz_source_transfer", np.load(first(d, "source_transfer_oof.npy")),
           np.load(first(d, "source_transfer_test.npy")), "row order")


print(f"{len(SOURCES)} sources registered: {', '.join(s['key'] for s in SOURCES)}")


# ---- CS116: các nguồn có trong stack v3 của nhóm nhưng không có trong pool 247 của bản gốc.
# Nạp SAU mọi nguồn của bản gốc nên pool 247 giữ nguyên; chúng qua cùng bộ kiểm tra (AUC, drift, bản gần trùng).
EXTRA_KEYS = ["megayak", "arhan", "mitudru", "pfn_out", "wangx", "thisray", "busyaprime"]


@source("megayak", "s6e10-oof-library")
def load_megayak(d):     # các seed RealMLP gộp thành một member
    mo, mt = pd.read_parquet(f"{d}/oof_predictions.parquet"), pd.read_parquet(f"{d}/test_predictions.parquet")
    assert (mo["id"].to_numpy() == train_ids).all() and (mt["id"].to_numpy() == test_ids).all()
    groups = {}
    for c in mo.columns:
        if c in ("id", "fold", "satisfaction"):
            continue
        groups.setdefault(c.rsplit("_s", 1)[0] if c.startswith("realmlp") else c, []).append(c)
    for g, cols in groups.items():
        yield "mg_" + g, mo[cols].mean(1).to_numpy(), mt[cols].mean(1).to_numpy(), "row order"


@source("arhan", "s6e10-oof-test-predictions")
def load_arhan(d):
    ao, at = pd.read_parquet(f"{d}/oof_predictions.parquet"), pd.read_parquet(f"{d}/test_predictions.parquet")
    assert (ao["id"].to_numpy() == train_ids).all() and (at["id"].to_numpy() == test_ids).all()
    for c in ["lgb_5f", "xgb_5f", "cat_5f", "realmlp_5f"]:
        yield "ar_" + c, ao[c].to_numpy(), at[c].to_numpy(), "row order"


@source("mitudru", "s6e10-analysis-first-equality-blocks-stack")
def load_mitudru(d):
    o = pd.read_parquet(f"{d}/oof_predictions.parquet")
    t = pd.read_parquet(f"{d}/test_predictions.parquet")
    for c in [c for c in o.columns if c != "id"]:
        yield "mi_" + c, by_id(o, train_ids, c), by_id(t, test_ids, c), "id"


@source("pfn_out", "s6e10-tabpfn-seeds")
def load_pfn_out(d):     # ba output TabPFN; thư mục của source này chỉ để đánh dấu đã gắn, hai cái còn lại tìm riêng
    for nm, slug in [("pfn_samanyu_route_og", "s6e10-tabpfn-route-og"), ("pfn_samanyu_seeds", "s6e10-tabpfn-seeds"),
                     ("pfn_hemingweb_raw", "s6e10-exp01-tabpfn35-full-context")]:
        dd = find_dir(slug)
        if dd is None:
            continue
        yield (nm, by_id(pd.read_csv(first(dd, "oof_exp01_tabpfn35.csv")), train_ids, "pred"),
               by_id(pd.read_csv(first(dd, "test_exp01_tabpfn35.csv")), test_ids, "satisfaction"), "id")


@source("wangx", "s6e10-catboost-rating-crosses-cv-0-96080")
def load_wangx(d):
    for slug, c in [("s6e10-catboost-rating-crosses-cv-0-96080", "rating_segments"),
                    ("s6e10-lightgbm-context-stats-cv-0-96065", "context_numeric")]:
        dd = find_dir(slug)
        if dd is None:
            continue
        yield ("wx_" + c, by_id(pd.read_parquet(first(dd, "oof_predictions.parquet")), train_ids, c),
               by_id(pd.read_parquet(first(dd, "test_predictions.parquet")), test_ids, c), "id")


@source("thisray", "s6e10-xgb-realmlp-catboost-lb-0-96123")
def load_thisray(d):
    for slug, pre in [("s6e10-xgb-realmlp-catboost-lb-0-96123", "th_"), ("s6e10-honest-oof-stack", "mgh_")]:
        dd = find_dir(slug)
        if dd is None:
            continue
        o, t = pd.read_csv(first(dd, "oof_members.csv")), pd.read_csv(first(dd, "test_members.csv"))
        for c in [c for c in t.columns if c != "id"]:
            yield pre + c, by_id(o, train_ids, c), by_id(t, test_ids, c), "id"


@source("busyaprime", "s6e10-busyaprime-five-members-oof")
def load_busyaprime(d):
    yield from npz_pairs(d, "", "bp_*.npz")


# ---- CS116: OOF của nhóm (folds.csv = fold seed 42), nạp cuối cùng
@source("team", "cs116-s6e10-oof")
def load_team(d):
    f = pd.read_csv(f"{d}/folds.csv").set_index("id").loc[train_ids, "fold"].to_numpy()
    assert (f == fold).all(), "folds.csv của nhóm khác fold seed 42"
    yield from npy_pairs(d, "team_")



# ==== 2b. Kiểm tra member và bỏ bản gần trùng (bản gốc) ====

MIN_AUC, MAX_AUC, AGREE_GAP, DRIFT, DUP_RHO = 0.90, 0.9625, 0.03, 0.03, 0.9999


def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-15, 1 - 1e-15)
    return np.log(p / (1 - p))


def is_rank(v):
    # values spread evenly over (0, 1], like rankdata(v) / len(v)
    q = np.quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9])
    return bool(v.min() >= 0 and v.max() <= 1 and np.abs(q - [0.1, 0.25, 0.5, 0.75, 0.9]).max() < 0.01)


def corr_cols(Z, r):
    Zc, rc = Z - Z.mean(0), r - r.mean()
    return (Zc * rc[:, None]).sum(0) / np.sqrt((Zc ** 2).sum(0) * (rc ** 2).sum())


rows, OOFL, TSTL = [], [], []
for s in SOURCES:
    d = find_dir(s["slug"])
    if d is None:
        rows.append(dict(source=s["key"], member="(all)", status="not attached")); continue
    try:
        got = list(s["load"](d))
    except Exception as e:
        rows.append(dict(source=s["key"], member="(all)", status=f"skipped: {type(e).__name__}: {e}"[:110])); continue
    for name, o, t, how in got:
        o, t = np.asarray(o, np.float64).ravel(), np.asarray(t, np.float64).ravel()
        r = dict(source=s["key"], member=name, aligned=how, status="")
        if o.shape != y.shape or t.shape != test_ids.shape:
            r["status"] = f"wrong length {o.shape[0]:,} / {t.shape[0]:,}"
        elif not (np.isfinite(o).all() and np.isfinite(t).all()):
            r["status"] = "NaN or inf"
        else:
            already = bool(o.min() < 0 or o.max() > 1 or t.min() < 0 or t.max() > 1)
            r["scale"] = "logits" if already else ("ranks" if is_rank(o) else "probabilities")
            if not already and is_rank(o) != is_rank(t):
                r["status"] = "OOF and test on different scales"
            p_o, p_t = (1 / (1 + np.exp(-o)), 1 / (1 + np.exp(-t))) if already else (o, t)
            r["drift"] = abs(p_t.mean() - p_o.mean())
            zo, zt = (o, t) if already else (logit(o), logit(t))
            OOFL.append(np.clip(zo, -30, 30).astype(np.float32)); TSTL.append(np.clip(zt, -30, 30).astype(np.float32))
            r["col"] = len(OOFL) - 1
        rows.append(r)
CHK = pd.DataFrame(rows)
assert CHK["col"].notna().any(), "nothing loaded"
CHK["col"] = CHK["col"].fillna(-1).astype(int)
live = CHK["col"] >= 0
Zall, Tall = np.column_stack(OOFL), np.column_stack(TSTL)
del OOFL, TSTL
CHK.loc[live, "oof_auc"] = [roc_auc_score(y, Zall[:, c]) for c in CHK.loc[live, "col"]]

# the base pool: Chris Deotte's two packaged datasets, all 135 pairs, kept exactly as published
BASE = CHK[live & CHK["source"].isin(["c124", "tfm"])]
assert len(BASE) == 135, f"the base pool should have 135 pairs, found {len(BASE)}"
ref_o = np.median(Zall[:, BASE["col"].to_numpy()], 1)
ref_t = np.median(Tall[:, BASE["col"].to_numpy()], 1)
CHK.loc[live, "agree_oof"] = corr_cols(Zall[:, CHK.loc[live, "col"]], ref_o)
CHK.loc[live, "agree_test"] = corr_cols(Tall[:, CHK.loc[live, "col"]], ref_t)

new = live & ~CHK.index.isin(BASE.index) & (CHK["status"] == "")
for i in CHK.index[new]:
    r = CHK.loc[i]
    if r["oof_auc"] < MIN_AUC:
        CHK.loc[i, "status"] = f"OOF AUC {r['oof_auc']:.3f}: row order or label problem"
    elif r["oof_auc"] > MAX_AUC:
        CHK.loc[i, "status"] = f"OOF AUC {r['oof_auc']:.4f}: too high to be out of fold"
    elif r["agree_oof"] - r["agree_test"] > AGREE_GAP:
        CHK.loc[i, "status"] = f"test half agrees less than OOF half ({r['agree_test']:.3f} vs {r['agree_oof']:.3f})"
    elif r["drift"] > DRIFT:
        CHK.loc[i, "status"] = f"mean moves {r['drift']:.3f} between train and test"

# near-copies: rank correlation of the OOF halves, the base first, then the new pairs in loading order
order = list(BASE["col"]) + [int(c) for c in CHK.loc[new & (CHK["status"] == ""), "col"]]
Rk = np.column_stack([rankdata(Zall[:, c]) for c in order])        # float64: the threshold sits in the 5th decimal
Rk -= Rk.mean(0)
Rk /= np.sqrt((Rk ** 2).mean(0))
RHO = np.clip((Rk.T @ Rk) / len(Rk), -1, 1)
del Rk
kept = list(range(len(BASE)))
pos = {c: j for j, c in enumerate(order)}
for j in range(len(BASE), len(order)):
    best = max(kept, key=lambda m: RHO[j, m])
    i = CHK.index[CHK["col"] == order[j]][0]
    CHK.loc[i, "max_rho"] = RHO[j, best]
    if RHO[j, best] > DUP_RHO:
        CHK.loc[i, "status"] = f"near-copy of {CHK.loc[CHK['col'] == order[best], 'member'].iloc[0]} (rho {RHO[j, best]:.6f})"
    else:
        kept.append(j)
FULL_COLS = [order[j] for j in kept]
CHK["in_pool"] = CHK["col"].isin(FULL_COLS)
NAMES = [CHK.loc[CHK["col"] == c, "member"].iloc[0] for c in FULL_COLS]
SRCS = [CHK.loc[CHK["col"] == c, "source"].iloc[0] for c in FULL_COLS]
ZB, ZBT = Zall[:, BASE["col"].to_numpy()], Tall[:, BASE["col"].to_numpy()]
ZF, ZFT = Zall[:, FULL_COLS], Tall[:, FULL_COLS]
N_BASE, N_FULL = ZB.shape[1], ZF.shape[1]
assert len(set(NAMES)) == N_FULL, "duplicate member names"
del Zall, Tall

tab = CHK[CHK["member"] != "(all)"].assign(kept=lambda d: d["in_pool"]).groupby("source", sort=False).agg(
    loaded=("member", "size"), kept=("kept", "sum"), best_oof_auc=("oof_auc", "max"))
tab["dropped"] = tab["loaded"] - tab["kept"]
print(tab[["loaded", "dropped", "kept", "best_oof_auc"]].to_string(formatters={"best_oof_auc": "{:.6f}".format}))
skipped = CHK[CHK["member"] == "(all)"]
if len(skipped):
    print("\nsources not loaded:\n" + skipped[["source", "status"]].to_string(index=False))
drop = CHK[(CHK["member"] != "(all)") & ~CHK["in_pool"] & ~CHK.index.isin(BASE.index)]
reason = drop["status"].str.replace(r" \(.*|:.*| of .*", "", regex=True)
print(f"\nthresholds: OOF AUC from {MIN_AUC} to {MAX_AUC}, agreement gap {AGREE_GAP}, mean drift {DRIFT}, "
      f"near-copy rank correlation above {DUP_RHO}")
print(f"loaded {int((CHK['member'] != '(all)').sum())} pairs from {CHK['source'].nunique()} sources; "
      f"base pool {N_BASE}; new pairs kept {N_FULL - N_BASE}; full pool {N_FULL}")
print("dropped, by reason:", reason.value_counts().to_dict())
print("\nnew pairs that failed a check (near-copies are counted separately below):")
print(drop[~drop["status"].str.startswith("near-copy")][["source", "member", "status"]].to_string(index=False)
      if (~drop["status"].str.startswith("near-copy")).any() else "  none")
print(f"\nnear-copies dropped: {int(drop['status'].str.startswith('near-copy').sum())}")
print(f"\nrank correlation of each kept new pair with its closest pool member: median "
      f"{CHK.loc[CHK['in_pool'] & CHK['max_rho'].notna(), 'max_rho'].median():.5f}, "
      f"lowest {CHK.loc[CHK['in_pool'] & CHK['max_rho'].notna(), 'max_rho'].min():.5f}")
CHK.drop(columns=["col"]).to_csv(f"{WORK}/member_check.csv", index=False)
print(f"every pair's checks and verdict: member_check.csv   [{time.time() - T0:.0f}s]")


# =====================================================================================================
# CS116 v4: so 4 pool x 2 stacker trên 3 meta seed
#   P247      : pool sạch của bản gốc (không có member là stack)
#   P247+team : thêm OOF của nhóm (A_lgb_v1, A_lr_v1, A_tabm_v1)
#   PU        : P247 + các nguồn riêng của stack v3 nhóm (đã qua kiểm tra bản gần trùng)
#   PU+team
# Stacker: 'lr' = LR family-wise C=0.003; 'blend' = 0.8 LR family + 0.2 LightGBM (cả hai chuẩn hoá).
# Nền: LR flat C=0.003 trên P247 (công thức của Dariush Afshar).
# Chênh dưới ~1e-5 xem như nhiễu; một cấu hình đáng dùng khi hơn ở cả 3 seed và phần lớn fold.
# =====================================================================================================
import gc
import lightgbm as lgb
from scipy.stats import spearmanr

Z, ZT = ZF.astype(np.float64), ZFT.astype(np.float64)
del ZF, ZFT, ZB, ZBT
gc.collect()
NTR = len(y)


def family(n):
    if 'tabpfn' in n or n.startswith('pfn_'): return 'tabpfn'
    if n.startswith('tfm_') or 'tabicl' in n or 'tabm' in n: return 'tfm'
    if 'realmlp' in n or 'rmlp' in n or 'mlp' in n: return 'mlp'
    return 'gbdt'


FAM = np.array([family(n) for n in NAMES])
SCALE = np.array([{'tabpfn': 3.0, 'tfm': 1.5}.get(f, 1.0) for f in FAM])
SRCS = np.array(SRCS)
REF_KEYS = [s["key"] for s in SOURCES if s["key"] not in EXTRA_KEYS + ["team"]]
I247 = [i for i, s in enumerate(SRCS) if s in REF_KEYS]
IU = [i for i, s in enumerate(SRCS) if s != 'team']
ITEAM = [i for i, s in enumerate(SRCS) if s == 'team']
print('member nhóm trong pool:', [NAMES[i] for i in ITEAM])
print('nguồn thêm giữ được:', pd.Series(SRCS[[i for i in IU if i not in I247]]).value_counts().to_dict())
POOLS = {'P247': I247, 'PU': IU}
if ITEAM:
    POOLS.update({'P247+team': I247 + ITEAM, 'PU+team': IU + ITEAM})
print({k: len(v) for k, v in POOLS.items()}, '| family:', {f: int((FAM == f).sum()) for f in ('gbdt', 'mlp', 'tabpfn', 'tfm')})

# LightGBM stacker: cùng các nguồn như Trust the CV (bỏ TFM của Chris Deotte, ...); không có member nhóm
GB_EXCLUDE = {'tfm', 'golem', 'sadam', 'koumeimaki', 'kagan_enc', 'aman_v5', 'forge', 'kozy', 'team'}
GB_POOLS = {'P247': [i for i in I247 if SRCS[i] not in GB_EXCLUDE], 'PU': [i for i in IU if SRCS[i] not in GB_EXCLUDE]}
gb_of = lambda pool: 'PU' if pool.startswith('PU') else 'P247'

TOL, MAX_ITER, C, W = 1e-6, 20000, 0.003, 0.2
P = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=2000, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=4, seed=0)


def meta_folds(seed):
    mf = np.zeros(NTR, int)
    for k, (_, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(NTR), y)):
        mf[b] = k
    return mf


def design(idx, scheme, test=False):
    M = (ZT if test else Z)[:, idx]
    return M * SCALE[idx] if scheme == 'family' else M


def _lr(Zm, a, b):
    m = LogisticRegression(C=C, tol=TOL, max_iter=MAX_ITER).fit(Zm[a], y[a])
    return b, m.decision_function(Zm[b])


def nested_lr(idx, scheme, mf):
    Zm, o = design(idx, scheme), np.zeros(NTR)
    for b, v in Parallel(n_jobs=N_JOBS)(delayed(_lr)(Zm, mf != k, mf == k) for k in range(5)):
        o[b] = v
    del Zm; gc.collect()
    return o


def fit_gb(A, ya):
    ai = np.random.default_rng(0).permutation(len(A)); cut = int(len(ai) * 0.9)
    return lgb.train(P, lgb.Dataset(A[ai[:cut]], ya[ai[:cut]]), 3000, valid_sets=[lgb.Dataset(A[ai[cut:]], ya[ai[cut:]])],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def nested_gb(idx, mf):
    o, A = np.zeros(NTR), Z[:, idx]
    for k in range(5):
        a, b = mf != k, mf == k
        m = fit_gb(A[a], y[a]); o[b] = m.predict(A[b], raw_score=True, num_iteration=m.best_iteration)
    return o


zs = lambda v: (v - v.mean()) / v.std()
fa = lambda o, mf: np.array([roc_auc_score(y[mf == k], o[mf == k]) for k in range(5)])

SEEDS = [42, 7, 101]
RES, OOF42, LR42, GB42, ROWS = {}, {}, {}, {}, []
for seed in SEEDS:
    t0 = time.time(); mf = meta_folds(seed)
    gb = {g: nested_gb(idx, mf) for g, idx in GB_POOLS.items()}
    print(f'\nmeta seed {seed}: LightGBM stacker ' + ', '.join(f'{g} {fa(o, mf).mean():.6f}' for g, o in gb.items())
          + f'   ({time.time() - t0:.0f}s)', flush=True)
    out = {('lr', 'flat', 'P247'): nested_lr(I247, 'flat', mf)}
    for pool, idx in POOLS.items():
        t = time.time()
        lr = nested_lr(idx, 'family', mf)
        out['lr', 'family', pool] = lr
        out['blend', 'family', pool] = (1 - W) * zs(lr) + W * zs(gb[gb_of(pool)])
        if seed == 42:
            LR42[pool] = lr
        print(f'  {pool:10s} LR family {fa(lr, mf).mean():.6f} | blend {fa(out["blend", "family", pool], mf).mean():.6f}'
              f'   ({time.time() - t:.0f}s)', flush=True)
    base = fa(out['lr', 'flat', 'P247'], mf)
    print(f'  nền LR flat P247: fold-mean {base.mean():.6f}')
    for k, o in out.items():
        f = fa(o, mf)
        RES[seed, k] = f
        ROWS.append(dict(seed=seed, stacker=k[0], scheme=k[1], pool=k[2], fold_mean=f.mean(), pooled=roc_auc_score(y, o),
                         gain_x1e6=(f - base).mean() * 1e6, folds_up=int((f > base).sum()),
                         **{f'fold{j + 1}': v for j, v in enumerate(f)}))
    if seed == 42:
        GB42, OOF42 = gb, out
    del out, gb; gc.collect()
    print(f'  (seed {seed}: {time.time() - t0:.0f}s)', flush=True)

TAB = pd.DataFrame(ROWS)
TAB.to_csv(f'{WORK}/v4_grid.csv', index=False)
piv = TAB.pivot_table(index=['stacker', 'scheme', 'pool'], columns='seed', values='gain_x1e6')
piv['min'] = piv[SEEDS].min(1)
print('\nChênh so với nền (LR flat P247), x 1e-6, fold-mean:')
print(piv.sort_values('min', ascending=False).round(1).to_string())


def pair(a, b):   # a vs b, từng seed
    s = []
    for seed in SEEDS:
        d = RES[seed, a] - RES[seed, b]
        s.append(f'seed {seed}: {d.mean() * 1e6:+6.1f} ({(d > 0).sum()}/5)')
    return ' | '.join(s)


print('\nSo sánh trực tiếp (x 1e-6, số fold tăng):')
for st in ('lr', 'blend'):
    print(f'  [{st}] PU vs P247:           {pair((st, "family", "PU"), (st, "family", "P247"))}')
    if ITEAM:
        print(f'  [{st}] P247+team vs P247: {pair((st, "family", "P247+team"), (st, "family", "P247"))}')
        print(f'  [{st}] PU+team vs PU:     {pair((st, "family", "PU+team"), (st, "family", "PU"))}')

# ---- refit trên toàn bộ train, lưu submission + OOF (meta seed 42) cho mọi cấu hình blend và nền
sig = lambda v: 1 / (1 + np.exp(-v))
t = time.time()
GBT = {}
for g, idx in GB_POOLS.items():
    m = fit_gb(Z[:, idx], y)
    GBT[g] = m.predict(ZT[:, idx], raw_score=True, num_iteration=m.best_iteration)
LOG = []
for pool, idx in POOLS.items():
    lr_full = LogisticRegression(C=C, tol=TOL, max_iter=MAX_ITER).fit(design(idx, 'family'), y).decision_function(
        design(idx, 'family', test=True))
    lo, go, g = LR42[pool], GB42[gb_of(pool)], gb_of(pool)
    test_b = (1 - W) * (lr_full - lo.mean()) / lo.std() + W * (GBT[g] - go.mean()) / go.std()
    oof_b = OOF42['blend', 'family', pool]
    tag = 'fw_' + pool.lower().replace('+', '_')
    np.save(f'{WORK}/oof_stack_{tag}.npy', sig(oof_b)); np.save(f'{WORK}/test_stack_{tag}.npy', sig(test_b))
    pd.DataFrame({'id': test_ids, 'satisfaction': sig(test_b)}).to_csv(f'{WORK}/sub_stack_{tag}.csv', index=False)
    sel = TAB[(TAB.stacker == 'blend') & (TAB.pool == pool)].set_index('seed').fold_mean
    g3 = '/'.join(f'{sel[s]:.6f}' for s in SEEDS)
    LOG.append(f"{time.strftime('%Y-%m-%d')},A,stack_{tag},{roc_auc_score(y, oof_b):.6f},,"
               f"kernel cs116-stack-v4; 0.8 LR family C=0.003 + 0.2 LGB tren {len(idx)} OOF; fold-mean seed 42/7/101: {g3}")
    gc.collect()
lr_b = LogisticRegression(C=C, tol=TOL, max_iter=MAX_ITER).fit(design(I247, 'flat'), y).decision_function(design(I247, 'flat', test=True))
pd.DataFrame({'id': test_ids, 'satisfaction': sig(lr_b)}).to_csv(f'{WORK}/sub_stack_lr_flat_p247.csv', index=False)
pd.DataFrame({'id': test_ids, 'satisfaction': sig(test_b)}).to_csv(f'{WORK}/submission.csv', index=False)
print(f'\nrefit xong ({time.time() - t:.0f}s); submission.csv = blend {pool}')
print('\nDòng cho experiments.csv (điền LB sau khi nộp):\n' + '\n'.join(LOG))
print(f'\ntổng {(time.time() - T0) / 60:.1f} phút')
