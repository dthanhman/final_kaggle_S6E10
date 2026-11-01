# =====================================================================================================
# CS116 v15 (21/10): tách hai phần của stack S14 (= v14 both_tt_rm) để đo trục "LightGBM stacker" trên CV.
#   Lưu riêng phần LR family (FWLS tt) và phần LGB stacker (logit thô, chưa chuẩn hoá) cho meta seed 42/7/101,
#   và chấm blend (1-W)·LR + W·LGB với W = -0.1..0.5 (W = 0.2 là S14). Dùng ở máy: notebooks/A_crowd_axis.py
#   (vị trí các file đám đông trên trục này; lợi ích CV khi kéo chúng về W tốt nhất).
# =====================================================================================================
import gc
import lightgbm as lgb

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
SRCS = np.array(SRCS)
NAMES = np.array(NAMES)
REF_KEYS = [s["key"] for s in SOURCES if s["key"] not in EXTRA_KEYS + ["team"]]
IDX = [i for i, s in enumerate(SRCS) if s in REF_KEYS and s != 'forge']
GB_EXCLUDE = {'tfm', 'golem', 'sadam', 'koumeimaki', 'kagan_enc', 'aman_v5', 'forge', 'kozy', 'team'}

# bản mới gpx_<mới> thay cho bản cũ gp_<cũ> (như v10)
PAIR = {"xgb_k10": "xgb_v3", "xgb_d10_k10": "xgb_v3_d10", "lgb_k10": "lgb_v3", "cat_k10": "cat_v3", "cat_ctr_k": "cat_ctr",
        "glm6_xgb_k": "glm6_xgb", "realmlp_fa_s7_k": "realmlp_fa_s7", "realmlp_fa_aux3_k": "realmlp_fa_aux3",
        "realmlp_pub_aux_k": "realmlp_pub_aux", "realmlp_v4fa_auxfull": "realmlp_v4fa", "realmlp_vb": "realmlp_pub"}
in_pool = set(NAMES[IDX])
gpx = {n[4:] for n in in_pool if n.startswith('gpx_')}
OLD = sorted({f'gp_{PAIR.get(n, n)}' for n in gpx} & in_pool)
RM = [i for i, n in enumerate(NAMES) if n == 'team_B_realmlp_v1']
assert len(RM) == 1, 'không thấy team_B_realmlp_v1 trong dataset cs116-s6e10-oof'
print(f'P247 bỏ Forge: {len(IDX)} member | bản cũ gp_* bỏ ở both: {len(OLD)} ({", ".join(OLD)})', flush=True)

IDX_BOTH = [i for i in IDX if NAMES[i] not in set(OLD)]
VARIANTS = {'base': (IDX, False), 'both': (IDX_BOTH, False), 'both_tt': (IDX_BOTH, True), 'both_tt_rm': (IDX_BOTH + RM, True)}

C, W, TOL, MAX_ITER, SEEDS, S_FW = 0.003, 0.2, 1e-6, 20000, [42, 7, 101], 0.3
GB_PARAMS = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=2000, feature_fraction=0.5,
                 bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=4, seed=0)
LR_JOBS = 2

_tr = pd.read_csv(f"{COMP}/train.csv", usecols=["id", "Type of Travel"])
_te = pd.read_csv(f"{COMP}/test.csv", usecols=["id", "Type of Travel"])
assert (_tr["id"].to_numpy() == train_ids).all() and (_te["id"].to_numpy() == test_ids).all()
G, GT = (_tr["Type of Travel"] == "Personal Travel").to_numpy(), (_te["Type of Travel"] == "Personal Travel").to_numpy()


def meta_folds(seed):
    mf = np.zeros(NTR, int)
    for k, (_, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(NTR), y)):
        mf[b] = k
    return mf


def fw_design(X, g, s):
    # [X, s·X·g, g] dựng tại chỗ bằng float32 (như v13 v2)
    n, p = X.shape
    D = np.empty((n, 2 * p + 1), np.float32)
    D[:, :p] = X
    np.multiply(D[:, :p], (s * g).astype(np.float32)[:, None], out=D[:, p:2 * p])
    D[:, 2 * p] = g
    return D


def _lr(k, mf, XL, XLT, fw):
    a, b = mf != k, mf == k
    if not fw:
        m = LogisticRegression(C=C, tol=TOL, max_iter=MAX_ITER).fit(XL[a], y[a])
        return k, m.decision_function(XL[b]), m.decision_function(XLT)
    D = fw_design(XL[a], G[a], S_FW)
    m = LogisticRegression(C=C, tol=TOL, max_iter=MAX_ITER).fit(D, y[a])
    del D; gc.collect()
    return k, m.decision_function(fw_design(XL[b], G[b], S_FW)), m.decision_function(fw_design(XLT, GT, S_FW))


def nested_lr(idx, fw, mf):
    s = np.array([{'tabpfn': 3.0, 'tfm': 1.5}.get(f, 1.0) for f in FAM[idx]])
    XL, XLT = Z[:, idx] * s, ZT[:, idx] * s
    o, t = np.zeros(NTR), np.zeros(len(test_ids))
    for k, v, vt in Parallel(n_jobs=LR_JOBS)(delayed(_lr)(k, mf, XL, XLT, fw) for k in range(5)):
        o[mf == k] = v; t += vt / 5
    del XL, XLT; gc.collect()
    return o, t


def nested_gb(gidx, mf):
    o, t, A, At = np.zeros(NTR), np.zeros(len(test_ids)), Z[:, gidx], ZT[:, gidx]
    for k in range(5):
        a, b = np.where(mf != k)[0], mf == k
        ai = np.random.default_rng(k).permutation(a); cut = int(len(ai) * 0.9)     # dừng sớm trên 10% của phần train
        m = lgb.train(GB_PARAMS, lgb.Dataset(A[ai[:cut]], y[ai[:cut]]), 4000,
                      valid_sets=[lgb.Dataset(A[ai[cut:]], y[ai[cut:]])], callbacks=[lgb.early_stopping(100, verbose=False)])
        o[b] = m.predict(A[b], raw_score=True, num_iteration=m.best_iteration)
        t += m.predict(At, raw_score=True, num_iteration=m.best_iteration) / 5
    return o, t


def blend(lr, gb):
    (lo, lt), (go, gt) = lr, gb
    oof = (1 - W) * (lo - lo.mean()) / lo.std() + W * (go - go.mean()) / go.std()
    tst = (1 - W) * (lt - lo.mean()) / lo.std() + W * (gt - go.mean()) / go.std()
    return oof, tst


WS = [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
idx = IDX_BOTH + RM
gidx = [i for i in idx if SRCS[i] not in GB_EXCLUDE]
print(f'S14: LR {len(idx)} / LGB {len(gidx)}', flush=True)
fa = lambda o, mf: np.array([roc_auc_score(y[mf == k], o[mf == k]) for k in range(5)])
rows = []
for seed in SEEDS:
    t0 = time.time()
    mf = meta_folds(seed)
    lr = nested_lr(idx, True, mf)
    gb = nested_gb(gidx, mf)
    for part, (o, t) in (('lr', lr), ('gb', gb)):
        np.save(f'{WORK}/oof_v15_{part}_s{seed}.npy', o); np.save(f'{WORK}/test_v15_{part}_s{seed}.npy', t)
    np.save(f'{WORK}/mf_v15_s{seed}.npy', mf)
    ref = None
    for W in WS:
        lo, go = lr[0], gb[0]
        f = fa((1 - W) * (lo - lo.mean()) / lo.std() + W * (go - go.mean()) / go.std(), mf)
        rows.append(dict(seed=seed, W=W, auc=f.mean(), **{f'f{k}': f[k] for k in range(5)}))
        print(f'  seed {seed} W {W:+.1f}: fold-mean {f.mean():.6f}   [{time.time() - t0:.0f}s]', flush=True)
TAB = pd.DataFrame(rows)
TAB.to_csv(f'{WORK}/v15_grid.csv', index=False)
for seed in SEEDS:
    b = TAB[(TAB.seed == seed) & (TAB.W == 0.2)].iloc[0]
    print(f'seed {seed}: ' + ' | '.join(f'W {r.W:+.1f} {(r.auc - b.auc) * 1e6:+.1f} [{sum(r[f"f{k}"] > b[f"f{k}"] for k in range(5))}/5]'
                                       for r in TAB[TAB.seed == seed].itertuples()))
print(f'\ntổng {(time.time() - T0) / 60:.1f} phút')
