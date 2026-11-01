# =====================================================================================================
# CS116 v11: thử nốt 2 cách tối ưu phần LR của stack (pool P247 bỏ Forge = stack_v10_drop_forge), LightGBM 0.2 giữ nguyên.
#   base     : LR family C=0.003 (sklearn: 0.5||w||² + C·Σ logloss)
#   enet_r   : elastic-net cùng C: C·Σ logloss + (1-r)/2·||w||² + r·||w||_1 (L1 làm trơn sqrt(w²+1e-8), L-BFGS từ w base),
#              r = 0.1 / 0.5 -> tự lọc bớt member gần trùng
#   pair_l0 / pair_l001: loss logistic theo cặp (dương, âm) tối ưu tự do từ w base, kéo về w base với lam 0 / 0.01
#              (v9 đã thử lam 1 / 0.1: −0.2 / −1.0e-6)
# Chấm nested trên meta seed 42/7/101; chọn: hơn base cả 3 seed, >= 10/15 fold, >= 5e-6.
# =====================================================================================================
import gc
import lightgbm as lgb
from scipy.optimize import minimize

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
REF_KEYS = [s["key"] for s in SOURCES if s["key"] not in EXTRA_KEYS + ["team"]]
IDX = [i for i, s in enumerate(SRCS) if s in REF_KEYS and s != 'forge']
GB_EXCLUDE = {'tfm', 'golem', 'sadam', 'koumeimaki', 'kagan_enc', 'aman_v5', 'forge', 'kozy', 'team'}
GIDX = [i for i in IDX if SRCS[i] not in GB_EXCLUDE]
print(f'P247 bỏ Forge: LR {len(IDX)} member, LGB {len(GIDX)} member', flush=True)

C, W, TOL, MAX_ITER, SEEDS = 0.003, 0.2, 1e-6, 20000, [42, 7, 101]
ENET_R, PAIR_LAMS, N_PAIRS = {'enet_01': 0.1, 'enet_05': 0.5}, {'pair_l0': 0.0, 'pair_l001': 0.01}, 4_000_000
GB_PARAMS = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=2000, feature_fraction=0.5,
                 bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=4, seed=0)
s_fam = np.array([{'tabpfn': 3.0, 'tfm': 1.5}.get(f, 1.0) for f in FAM[IDX]])
XL, XLT = Z[:, IDX] * s_fam, ZT[:, IDX] * s_fam
VARIANTS = ['base', *ENET_R, *PAIR_LAMS]


def meta_folds(seed):
    mf = np.zeros(NTR, int)
    for k, (_, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(NTR), y)):
        mf[b] = k
    return mf


def enet_lr(Xa, ya, w0, b0, r):
    ya = ya.astype(np.float64)

    def f(p):
        w, b = p[:-1], p[-1]
        z = Xa @ w + b
        g = 0.5 * (1 + np.tanh(0.5 * z)) - ya
        sq = np.sqrt(w * w + 1e-8)
        val = C * (np.logaddexp(0, z) - ya * z).sum() + 0.5 * (1 - r) * w @ w + r * sq.sum()
        return val, np.r_[C * (Xa.T @ g) + (1 - r) * w + r * w / sq, C * g.sum()]
    res = minimize(f, np.r_[w0, b0], jac=True, method='L-BFGS-B', options=dict(maxiter=3000, ftol=1e-12, gtol=1e-7))
    return res.x[:-1], res.x[-1]


def pair_opt(Xa, ya, w0, lam, seed):
    pos, neg = np.where(ya == 1)[0], np.where(ya == 0)[0]
    rng = np.random.default_rng(seed)
    pi, ni = pos[rng.integers(0, len(pos), N_PAIRS)], neg[rng.integers(0, len(neg), N_PAIRS)]
    nw = w0 @ w0

    def f(w):
        s = Xa @ w
        d = s[pi] - s[ni]
        g = -0.5 * (1 + np.tanh(-0.5 * d)) / N_PAIRS
        rr = np.bincount(pi, g, len(Xa)) - np.bincount(ni, g, len(Xa))
        return np.logaddexp(0, -d).mean() + 0.5 * lam * (w - w0) @ (w - w0) / nw, Xa.T @ rr + lam * (w - w0) / nw
    return minimize(f, w0.copy(), jac=True, method='L-BFGS-B', options=dict(maxiter=500, ftol=1e-12, gtol=1e-9)).x


def _fold(k, mf, XL, XLT, seed):
    a, b = mf != k, mf == k
    Xa, ya = XL[a], y[a]
    m = LogisticRegression(C=C, tol=TOL, max_iter=MAX_ITER).fit(Xa, ya)
    w0, b0 = m.coef_[0].copy(), float(m.intercept_[0])
    out = {'base': (XL[b] @ w0 + b0, XLT @ w0 + b0)}
    for name, r in ENET_R.items():
        w, bb = enet_lr(Xa, ya, w0, b0, r)
        out[name] = (XL[b] @ w + bb, XLT @ w + bb)
        out[name + '_nz'] = int((np.abs(w) > 1e-3).sum())
    for name, lam in PAIR_LAMS.items():
        w = pair_opt(Xa, ya, w0, lam, seed * 10 + k)
        out[name] = (XL[b] @ w, XLT @ w)
    return k, out


def nested_gb(mf):
    o, t, A, At = np.zeros(NTR), np.zeros(len(test_ids)), Z[:, GIDX], ZT[:, GIDX]
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


fa = lambda o, mf: np.array([roc_auc_score(y[mf == k], o[mf == k]) for k in range(5)])
OUT, FOLDS = {}, {}
for seed in SEEDS:
    t0 = time.time()
    mf = meta_folds(seed)
    gb = nested_gb(mf)
    lr = {v: [np.zeros(NTR), np.zeros(len(test_ids))] for v in VARIANTS}
    for k, out in Parallel(n_jobs=N_JOBS)(delayed(_fold)(k, mf, XL, XLT, seed) for k in range(5)):
        b = mf == k
        for v in VARIANTS:
            lr[v][0][b] = out[v][0]; lr[v][1] += out[v][1] / 5
        print(f'  seed {seed} fold {k}: member khác 0 enet_01 {out["enet_01_nz"]}, enet_05 {out["enet_05_nz"]}', flush=True)
    for v in VARIANTS:
        OUT[v, seed] = blend(lr[v], gb)
        FOLDS[v, seed] = fa(OUT[v, seed][0], mf)
    print(f'seed {seed}: ' + ' | '.join(f'{v} {FOLDS[v, seed].mean():.6f}' for v in VARIANTS) + f'   [{time.time() - t0:.0f}s]',
          flush=True)
    gc.collect()

rows = []
for v in VARIANTS:
    d = {s: FOLDS[v, s] - FOLDS['base', s] for s in SEEDS}
    rows.append(dict(variant=v, **{f'auc{s}': FOLDS[v, s].mean() for s in SEEDS}, **{f'd{s}': d[s].mean() * 1e6 for s in SEEDS},
                     dmin=min(d[s].mean() for s in SEEDS) * 1e6, folds_up=int(sum((d[s] > 0).sum() for s in SEEDS))))
TAB = pd.DataFrame(rows)
TAB.to_csv(f'{WORK}/v11_grid.csv', index=False)
pd.set_option('display.width', 250)
print('\nChênh so với base (x 1e-6, fold-mean; folds_up trên 15):')
print(TAB.to_string(index=False, float_format=lambda v: f'{v:.6f}' if abs(v) < 1 else f'{v:+.1f}'))
win = TAB[(TAB.variant != 'base') & (TAB.dmin >= 5) & (TAB.folds_up >= 10)].sort_values('dmin', ascending=False)
best = win.variant.iloc[0] if len(win) else 'base'
print(f'\nBiến thể chọn: {best}' + ('' if len(win) else ' (không biến thể nào qua cổng)'))

sig = lambda v: 1 / (1 + np.exp(-v))
LOG = []
for tag in sorted({'base', best}):
    oof42 = OUT[tag, 42][0]
    tst = np.mean([OUT[tag, s][1] for s in SEEDS], axis=0)
    np.save(f'{WORK}/oof_stack_v11_{tag}.npy', sig(oof42)); np.save(f'{WORK}/test_stack_v11_{tag}.npy', sig(tst))
    pd.DataFrame({'id': test_ids, 'satisfaction': sig(tst)}).to_csv(f'{WORK}/sub_stack_v11_{tag}.csv', index=False)
    g3 = '/'.join(f'{FOLDS[tag, s].mean():.6f}' for s in SEEDS)
    LOG.append(f"{time.strftime('%Y-%m-%d')},A,stack_v11_{tag},{roc_auc_score(y, oof42):.6f},,"
               f"kernel cs116-stack-v11; P247 bỏ Forge; {tag} + 0.2 gb_v4; test TB 15 fold-model; fold-mean seed 42/7/101: {g3}")
print('\nDòng cho experiments.csv (điền LB sau khi nộp):\n' + '\n'.join(LOG))
print(f'\ntổng {(time.time() - T0) / 60:.1f} phút')
