# =====================================================================================================
# CS116 v17 (25/10): thêm Q-FT v2 của nhóm (notebooks/A_qft2.py, CV 0.961091) vào S14 (= v14 both_tt_rm).
#   Probe nhanh (LR trên OOF stack) chỉ +2.6..+3.5e-6 → đo lại khi qft2 là member thật của stack đầy đủ.
#   qft2 nạp từ output kernel cs116-qft2-v2 (seed 42) và cs116-qft2-v2s7 (seed 7, nếu đã gắn): có cả hai thì
#   lấy trung bình logit thành MỘT cột. Biến thể: qft (LR ×1), qft15 (LR ×1.5 như họ tfm), qft_gb (LR + LGB).
#   Cổng như v14: hơn base cả 3 seed, >= 10/15 fold, >= 5e-6.
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
REF_KEYS = [s["key"] for s in SOURCES if s["key"] not in EXTRA_KEYS + ["team"] + QFT_KEYS]
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
IDX_BOTH = [i for i in IDX if NAMES[i] not in set(OLD)]
QI = [i for i, n in enumerate(NAMES) if n.startswith('qft_')]
assert len(QI) >= 1, 'không thấy qft2 (gắn output kernel thanhmn123/cs116-qft2-v2)'
print('qft2 nạp được:', list(NAMES[QI]), 'AUC', [round(roc_auc_score(y, Z[:, i]), 6) for i in QI], flush=True)
if len(QI) > 1:                                                               # trung bình logit các seed vào cột đầu
    Z[:, QI[0]], ZT[:, QI[0]] = Z[:, QI].mean(1), ZT[:, QI].mean(1)
    print(f'qft2 trung bình {len(QI)} seed: AUC {roc_auc_score(y, Z[:, QI[0]]):.6f}', flush=True)
QI = QI[:1]
FAM[QI[0]] = 'qft'

C, W, TOL, MAX_ITER, SEEDS, S_FW = 0.003, 0.2, 1e-6, 20000, [42, 7, 101], 0.3
GB_PARAMS = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=2000, feature_fraction=0.5,
                 bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=4, seed=0)
LR_JOBS = 2

# (hệ số tabpfn, tfm, mlp, qft; có qft trong LR; có qft trong LGB). base = S14
GRID = {'base': ((3.0, 1.5, 1.0, 1.0), False, False),
        'qft': ((3.0, 1.5, 1.0, 1.0), True, False),
        'qft15': ((3.0, 1.5, 1.0, 1.5), True, False),
        'qft_gb': ((3.0, 1.5, 1.0, 1.0), True, True)}

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


def _lr(k, mf, XL, XLT):
    a, b = mf != k, mf == k
    D = fw_design(XL[a], G[a], S_FW)
    m = LogisticRegression(C=C, tol=TOL, max_iter=MAX_ITER).fit(D, y[a])
    del D; gc.collect()
    return k, m.decision_function(fw_design(XL[b], G[b], S_FW)), m.decision_function(fw_design(XLT, GT, S_FW))


def nested_lr(idx, sc, mf):
    pfn, tfm, mlp, qft = sc
    s = np.array([{'tabpfn': pfn, 'tfm': tfm, 'mlp': mlp, 'qft': qft}.get(f, 1.0) for f in FAM[idx]])
    XL, XLT = Z[:, idx] * s, ZT[:, idx] * s
    o, t = np.zeros(NTR), np.zeros(len(test_ids))
    for k, v, vt in Parallel(n_jobs=LR_JOBS)(delayed(_lr)(k, mf, XL, XLT) for k in range(5)):
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


idx = IDX_BOTH + RM
gidx = [i for i in idx if SRCS[i] not in GB_EXCLUDE]
fam_n = pd.Series(FAM[idx]).value_counts().to_dict()
print(f'S14: LR {len(idx)} / LGB {len(gidx)} | họ trong LR: {fam_n}', flush=True)
fa = lambda o, mf: np.array([roc_auc_score(y[mf == k], o[mf == k]) for k in range(5)])
OUT, FOLDS = {}, {}
for seed in SEEDS:
    t0 = time.time()
    mf = meta_folds(seed)
    gbs = {False: nested_gb(gidx, mf), True: nested_gb(gidx + QI, mf)}
    for name, (sc, q_lr, q_gb) in GRID.items():
        OUT[name, seed] = blend(nested_lr(idx + QI if q_lr else idx, sc, mf), gbs[q_gb])
        FOLDS[name, seed] = fa(OUT[name, seed][0], mf)
        print(f'  seed {seed} {name:7s}: fold-mean {FOLDS[name, seed].mean():.6f}   [{time.time() - t0:.0f}s]', flush=True)
    del gbs; gc.collect()

rows = []
for v, sc in GRID.items():
    d = {s: FOLDS[v, s] - FOLDS['base', s] for s in SEEDS}
    rows.append(dict(variant=v, qft=sc[0][3] if sc[1] else 0.0, qft_gb=sc[2], **{f'auc{s}': FOLDS[v, s].mean() for s in SEEDS},
                     **{f'd{s}': d[s].mean() * 1e6 for s in SEEDS},
                     dmin=min(d[s].mean() for s in SEEDS) * 1e6, folds_up=int(sum((d[s] > 0).sum() for s in SEEDS)),
                     **{f'f{s}_{k}': FOLDS[v, s][k] for s in SEEDS for k in range(5)}))
TAB = pd.DataFrame(rows)
TAB.to_csv(f'{WORK}/v17_grid.csv', index=False)
pd.set_option('display.width', 250)
show = ['variant', 'qft', 'qft_gb'] + [f'auc{s}' for s in SEEDS] + [f'd{s}' for s in SEEDS] + ['dmin', 'folds_up']
print('\nChênh so với base = S14 (x 1e-6, fold-mean; folds_up trên 15):')
print(TAB[show].to_string(index=False, float_format=lambda v: f'{v:.6f}' if abs(v) < 1 else f'{v:+.1f}'))
win = TAB[(TAB.variant != 'base') & (TAB.dmin >= 5) & (TAB.folds_up >= 10)].sort_values('dmin', ascending=False)
best = win.variant.iloc[0] if len(win) else 'base'
print(f'\nBiến thể chọn: {best}' + ('' if len(win) else ' (không biến thể nào qua cổng)'))

# ---- lưu mọi biến thể: test = trung bình 3 seed (mỗi seed đã là trung bình 5 fold); OOF = meta seed 42
sig = lambda v: 1 / (1 + np.exp(-v))
LOG = []
for tag in GRID:
    oof42 = OUT[tag, 42][0]
    tst = np.mean([OUT[tag, s][1] for s in SEEDS], axis=0)
    np.save(f'{WORK}/oof_stack_v17_{tag}.npy', sig(oof42)); np.save(f'{WORK}/test_stack_v17_{tag}.npy', sig(tst))
    pd.DataFrame({'id': test_ids, 'satisfaction': sig(tst)}).to_csv(f'{WORK}/sub_stack_v17_{tag}.csv', index=False)
    g3 = '/'.join(f'{FOLDS[tag, s].mean():.6f}' for s in SEEDS)
    LOG.append(f"{time.strftime('%Y-%m-%d')},A,stack_v17_{tag},{roc_auc_score(y, oof42):.6f},,"
               f"kernel cs116-stack-v17; S14 + qft2 {GRID[tag]} ((pfn,tfm,mlp,qft), qft LR, qft LGB) + 0.2 LGB; fold-mean seed 42/7/101: {g3}")
print('\nDòng cho experiments.csv (điền LB sau khi nộp):\n' + '\n'.join(LOG))
print(f'\ntổng {(time.time() - T0) / 60:.1f} phút')
