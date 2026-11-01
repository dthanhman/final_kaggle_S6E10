# =====================================================================================================
# CS116 v9: thay phần LR của stack_fw_p247 (fam_C003 + 0.2 gb_v4), giữ nguyên phần LightGBM.
#   base      : LR family C=0.003 (= stack_fw_p247)
#   rank_logit: LR trên [logit family, probit(rank)] của mỗi member (502 cột)
#   nonneg    : cùng mục tiêu với sklearn (0.5||w||² + C·Σ logloss) nhưng ràng buộc w >= 0 (L-BFGS-B)
#   auc_l1/auc_l01: từ trọng số LR của fold, tối ưu loss logistic theo cặp (dương, âm) trên phần train của meta fold
#               + lam/2 · ||w - w0||² / ||w0||²  (lam 1 / 0.1); mục tiêu sát AUC hơn log-loss
# Mọi biến thể trộn cùng một phần LightGBM (W 0.2) và chấm nested trên meta seed 42/7/101; chọn: hơn base cả 3 seed,
# >= 10/15 fold, >= 5e-6. Test của bản chọn = TB 5 meta seed (42/7/101/2026/314) x 5 fold-model.
# CLIP_L: ngưỡng cắt logit chọn từ v8 (None = như v4: 1e-15, ±30).
# =====================================================================================================
import gc
import lightgbm as lgb
from scipy.optimize import minimize
from scipy.special import ndtri

CLIP_L, CLIP_PROB_ONLY = None, False
Z0, ZT0 = ZF.astype(np.float64), ZFT.astype(np.float64)
del ZF, ZFT, ZB, ZBT
gc.collect()
NTR = len(y)
SCALE_OF = CHK.set_index('col')['scale']
PROB = np.array([SCALE_OF.loc[c] != 'logits' for c in FULL_COLS])
if CLIP_L is not None:
    cols = np.where(PROB)[0] if CLIP_PROB_ONLY else np.arange(Z0.shape[1])
    Z0[:, cols] = np.clip(Z0[:, cols], -CLIP_L, CLIP_L); ZT0[:, cols] = np.clip(ZT0[:, cols], -CLIP_L, CLIP_L)


def family(n):
    if 'tabpfn' in n or n.startswith('pfn_'): return 'tabpfn'
    if n.startswith('tfm_') or 'tabicl' in n or 'tabm' in n: return 'tfm'
    if 'realmlp' in n or 'rmlp' in n or 'mlp' in n: return 'mlp'
    return 'gbdt'


FAM = np.array([family(n) for n in NAMES])
SRCS = np.array(SRCS)
REF_KEYS = [s["key"] for s in SOURCES if s["key"] not in EXTRA_KEYS + ["team"]]
I247 = [i for i, s in enumerate(SRCS) if s in REF_KEYS]
GB_EXCLUDE = {'tfm', 'golem', 'sadam', 'koumeimaki', 'kagan_enc', 'aman_v5', 'forge', 'kozy', 'team'}
GB247 = [i for i in I247 if SRCS[i] not in GB_EXCLUDE]
print(f'P247: {len(I247)} member | LGB stacker: {len(GB247)} member | CLIP_L {CLIP_L}')

C, W, TOL, MAX_ITER = 0.003, 0.2, 1e-6, 20000
GATE_SEEDS, ALL_SEEDS = [42, 7, 101], [42, 7, 101, 2026, 314]
N_PAIRS, AUC_LAMS = 4_000_000, {'auc_l1': 1.0, 'auc_l01': 0.1}
GB_PARAMS = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=2000, feature_fraction=0.5,
                 bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=4, seed=0)
s_fam = np.array([{'tabpfn': 3.0, 'tfm': 1.5}.get(f, 1.0) for f in FAM[I247]])
XL, XLT = Z0[:, I247] * s_fam, ZT0[:, I247] * s_fam                           # thiết kế của base


def probit_rank(M):
    out = np.empty_like(M)
    for j in range(M.shape[1]):
        out[:, j] = ndtri((rankdata(M[:, j]) - 0.5) / len(M))
    return out


def meta_folds(seed):
    mf = np.zeros(NTR, int)
    for k, (_, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(NTR), y)):
        mf[b] = k
    return mf


def sk_lr(X, a):
    m = LogisticRegression(C=C, tol=TOL, max_iter=MAX_ITER).fit(X[a], y[a])
    return m.coef_[0].copy(), float(m.intercept_[0])


def nonneg_lr(X, a, w0, b0):
    Xa, ya = X[a], y[a].astype(np.float64)

    def f(p):
        w, b = p[:-1], p[-1]
        z = Xa @ w + b
        loss = np.logaddexp(0, z) - ya * z
        g = 0.5 * (1 + np.tanh(0.5 * z)) - ya        # sigmoid; không dùng scipy expit (worker loky không unpickle được)
        return 0.5 * w @ w + C * loss.sum(), np.r_[w + C * (Xa.T @ g), C * g.sum()]
    p0 = np.r_[np.maximum(w0, 0), b0]
    r = minimize(f, p0, jac=True, method='L-BFGS-B', bounds=[(0, None)] * len(w0) + [(None, None)],
                 options=dict(maxiter=3000, ftol=1e-12, gtol=1e-7))
    return r.x[:-1], r.x[-1]


def auc_opt(X, a, w0, lam, seed):
    Xa, ya = X[a], y[a]
    pos, neg = np.where(ya == 1)[0], np.where(ya == 0)[0]
    rng = np.random.default_rng(seed)
    pi, ni = pos[rng.integers(0, len(pos), N_PAIRS)], neg[rng.integers(0, len(neg), N_PAIRS)]
    nw = w0 @ w0

    def f(w):
        s = Xa @ w
        d = s[pi] - s[ni]
        loss = np.logaddexp(0, -d).mean()
        g = -0.5 * (1 + np.tanh(-0.5 * d)) / N_PAIRS
        r = np.bincount(pi, g, len(Xa)) - np.bincount(ni, g, len(Xa))
        return loss + 0.5 * lam * (w - w0) @ (w - w0) / nw, Xa.T @ r + lam * (w - w0) / nw
    r = minimize(f, w0.copy(), jac=True, method='L-BFGS-B', options=dict(maxiter=500, ftol=1e-12, gtol=1e-9))
    return r.x


def _fold(k, mf, XL, XLT, X2, X2T, seed):
    # ma trận truyền qua tham số để joblib/loky dùng memmap thay vì chép cả ma trận vào mỗi tác vụ
    a, b = mf != k, mf == k
    out = {}
    w0, b0 = sk_lr(XL, a)
    out['base'] = (XL[b] @ w0 + b0, XLT @ w0 + b0)
    if X2 is not None:
        w2, b2 = sk_lr(X2, a)
        out['rank_logit'] = (X2[b] @ w2 + b2, X2T @ w2 + b2)
        wn, bn = nonneg_lr(XL, a, w0, b0)
        out['nonneg'] = (XL[b] @ wn + bn, XLT @ wn + bn)
        for name, lam in AUC_LAMS.items():
            wa = auc_opt(XL, a, w0, lam, seed * 10 + k)
            out[name] = (XL[b] @ wa, XLT @ wa)
    return k, out


def nested_gb(mf):
    o, t, A, At = np.zeros(NTR), np.zeros(len(test_ids)), Z0[:, GB247], ZT0[:, GB247]
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


t0 = time.time()
# [logit family, probit(rank)] dựng một lần, float32 (502 cột x 700k = 1.4 GB) để 3 worker không tràn RAM 30 GB
X2 = np.hstack([XL, probit_rank(Z0[:, I247])]).astype(np.float32)
X2T = np.hstack([XLT, probit_rank(ZT0[:, I247])]).astype(np.float32)
NJ = min(N_JOBS, 3)
print(f'probit(rank) xong [{time.time() - t0:.0f}s]', flush=True)
fa = lambda o, mf: np.array([roc_auc_score(y[mf == k], o[mf == k]) for k in range(5)])
VARS = ['base', 'rank_logit', 'nonneg'] + list(AUC_LAMS)
LRO, GBO, MF, FOLDS = {}, {}, {}, {}
for seed in ALL_SEEDS:
    t0 = time.time(); mf = meta_folds(seed); MF[seed] = mf
    full = seed in GATE_SEEDS
    res = Parallel(n_jobs=NJ)(delayed(_fold)(k, mf, XL, XLT, X2 if full else None, X2T if full else None, seed)
                              for k in range(5))
    for v in (VARS if full else ['base']):
        o, t = np.zeros(NTR), np.zeros(len(test_ids))
        for k, out in res:
            o[mf == k] = out[v][0]; t += out[v][1] / 5
        LRO[v, seed] = (o, t)
    GBO[seed] = nested_gb(mf)
    for v in (VARS if full else ['base']):
        FOLDS[v, seed] = fa(blend(LRO[v, seed], GBO[seed])[0], mf)
    print(f'seed {seed}: ' + ' | '.join(f'{v} {FOLDS[v, seed].mean():.6f}' for v in (VARS if full else ['base']))
          + f'   [{time.time() - t0:.0f}s]', flush=True)
    del res; gc.collect()

rows = []
for v in VARS:
    d = {s: FOLDS[v, s] - FOLDS['base', s] for s in GATE_SEEDS}
    rows.append(dict(variant=v, **{f'auc{s}': FOLDS[v, s].mean() for s in GATE_SEEDS},
                     **{f'd{s}': d[s].mean() * 1e6 for s in GATE_SEEDS},
                     dmin=min(d[s].mean() for s in GATE_SEEDS) * 1e6, folds_up=int(sum((d[s] > 0).sum() for s in GATE_SEEDS))))
TAB = pd.DataFrame(rows)
TAB.to_csv(f'{WORK}/v9_grid.csv', index=False)
pd.set_option('display.width', 250)
print('\nChênh so với base (x 1e-6, fold-mean; folds_up trên 15; LR thay đổi, LightGBM 0.2 giữ nguyên):')
print(TAB.to_string(index=False, float_format=lambda v: f'{v:.6f}' if abs(v) < 1 else f'{v:+.1f}'))
ok = TAB[(TAB.variant != 'base') & (TAB[[f'd{s}' for s in GATE_SEEDS]] > 0).all(1) & (TAB.folds_up >= 10) & (TAB.dmin >= 5)]
BEST = ok.sort_values('dmin', ascending=False).iloc[0].variant if len(ok) else 'base'
print(f'\nBiến thể chọn: {BEST}' + ('' if len(ok) else ' (không biến thể nào qua cổng)'))

# ---- test của bản chọn: TB 5 meta seed (biến thể không phải base chỉ có 3 seed cổng -> tính thêm 2 seed còn lại)
if BEST != 'base':
    for seed in ALL_SEEDS:
        if (BEST, seed) in LRO:
            continue
        mf = MF[seed]
        res = Parallel(n_jobs=NJ)(delayed(_fold)(k, mf, XL, XLT, X2, X2T, seed) for k in range(5))
        o, t = np.zeros(NTR), np.zeros(len(test_ids))
        for k, out in res:
            o[mf == k] = out[BEST][0]; t += out[BEST][1] / 5
        LRO[BEST, seed] = (o, t)
sig = lambda v: 1 / (1 + np.exp(-v))
LOG = []
for tag in dict.fromkeys(['base', BEST]):
    oof42 = blend(LRO[tag, 42], GBO[42])[0]
    tst = np.mean([blend(LRO[tag, s], GBO[s])[1] for s in ALL_SEEDS], axis=0)
    np.save(f'{WORK}/oof_stack_v9_{tag}.npy', sig(oof42)); np.save(f'{WORK}/test_stack_v9_{tag}.npy', sig(tst))
    pd.DataFrame({'id': test_ids, 'satisfaction': sig(tst)}).to_csv(f'{WORK}/sub_stack_v9_{tag}.csv', index=False)
    g3 = '/'.join(f'{FOLDS[tag, s].mean():.6f}' for s in GATE_SEEDS)
    LOG.append(f"{time.strftime('%Y-%m-%d')},A,stack_v9_{tag},{roc_auc_score(y, oof42):.6f},,"
               f"kernel cs116-stack-v9; P247 {len(I247)} OOF; LR {tag} + 0.2 gb_v4; CLIP_L {CLIP_L}; test TB 5 meta seed x 5 fold; "
               f"fold-mean seed 42/7/101: {g3}")
print('\nDòng cho experiments.csv (điền LB sau khi nộp):\n' + '\n'.join(LOG))
print(f'\ntổng {(time.time() - T0) / 60:.1f} phút')
