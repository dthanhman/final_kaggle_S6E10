"""Chạy lại sạch member kozy (kozykappa "Source-Transfer + Foundation-Model OOF", notebook gốc đã bị xoá; bản sao ở
references/s6e10-source-transfer-foundation-model-oof.ipynb).

Giữ nguyên feature, prior từ dữ liệu gốc, tham số LightGBM và trọng số 0.4 raw + 0.6 source-prior.
Khác bản gốc: dùng folds.csv; early stopping trên 10% tách từ phần train của fold (bản gốc dừng sớm trên fold đang chấm).
Lưu: A_kozy_raw_v1, A_kozy_prior_v1, A_kozy_v1 (= 0.4 raw + 0.6 prior, giống kz_source_transfer).

Chạy: python notebooks/A_kozy_clean.py [--es-on-val]
  --es-on-val: dừng sớm trên fold đang chấm như bản gốc (KHÔNG sạch, chỉ để đo mức thổi phồng), lưu A_kozy_esval_v1.
"""
import argparse
import gc
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils import save_preds  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--es-on-val", action="store_true")
ES_ON_VAL = ap.parse_args().es_on_val

SEED = 42
SURVEY = ['Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
          'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment',
          'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Cleanliness']
CATEGORICAL = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
PRIOR_GROUPS = [
    ['Type of Travel', 'Class', 'Online boarding', 'Inflight entertainment'],
    ['Type of Travel', 'Class', 'Online boarding', 'Inflight entertainment', 'Inflight wifi service'],
    ['Type of Travel', 'Class', 'Online boarding', 'Inflight entertainment', 'Seat comfort'],
    ['Type of Travel', 'Class', 'Online boarding', 'Inflight entertainment', 'On-board service'],
    ['Type of Travel', 'Class', 'Online boarding', 'Inflight entertainment', 'Baggage handling'],
    ['Type of Travel', 'Class', 'Online boarding', 'Inflight entertainment', 'Food and drink'],
    ['Type of Travel', 'Class', 'Online boarding', 'Inflight entertainment', 'Customer Type'],
    ['Type of Travel', 'Class', 'Online boarding', 'Inflight wifi service'],
    ['Customer Type', 'Type of Travel', 'Online boarding', 'Inflight entertainment'],
    ['Type of Travel', 'Class', 'Online boarding', 'Inflight entertainment', 'Inflight wifi service', 'Seat comfort',
     'On-board service'],
]
PARAMS = dict(objective='binary', learning_rate=0.05, num_leaves=63, min_child_samples=50, colsample_bytree=0.80,
              subsample=0.85, subsample_freq=1, n_estimators=1300, random_state=SEED, n_jobs=4, verbosity=-1)


def features(df):
    out = df.drop(columns=['id', 'satisfaction'], errors='ignore').copy()
    a = out['Arrival Delay in Minutes'].fillna(0)
    d = out['Departure Delay in Minutes'].fillna(0)
    out['Total_Delay'] = a + d
    out['Delay_Delta'] = a - d
    out['Delay_Ratio'] = a / (d + 1)
    out['Is_Delayed'] = ((a > 0) | (d > 0)).astype('int8')
    out['Severe_Delay_Flag'] = ((a > 45) | (d > 45)).astype('int8')
    for name, group in [
            ('Inflight_Comfort_Index', ['Seat comfort', 'Leg room service', 'Food and drink', 'Cleanliness']),
            ('Ground_Service_Index', ['Departure/Arrival time convenient', 'Gate location', 'Baggage handling',
                                      'Checkin service']),
            ('Digital_Boarding_Index', ['Ease of Online booking', 'Online boarding', 'Inflight wifi service'])]:
        out[name] = out[group].mean(axis=1)
    out['Service_Mean'] = out[SURVEY].mean(axis=1)
    out['Service_Std'] = out[SURVEY].std(axis=1).fillna(0)
    out['Service_Min'] = out[SURVEY].min(axis=1)
    out['Service_Max'] = out[SURVEY].max(axis=1)
    out['Service_Range'] = out['Service_Max'] - out['Service_Min']
    out['Zero_Rating_Count'] = (out[SURVEY] == 0).sum(axis=1)
    out['Wifi_Booking_Diff'] = out['Inflight wifi service'] - out['Ease of Online booking']
    out['Space_Comfort_Sum'] = out['Seat comfort'] + out['Leg room service']
    out['Distance_Log'] = np.log1p(out['Flight Distance'])
    out['Is_Long_Haul'] = (out['Flight Distance'] > 2000).astype('int8')
    out['Is_Short_Haul'] = (out['Flight Distance'] < 500).astype('int8')
    for c in CATEGORICAL:
        out[c] = out[c].astype('category')
    return out


def with_source_priors(df, raw, original):
    output = df.copy()
    prior = float(original['satisfaction'].mean())
    for i, keys in enumerate(PRIOR_GROUPS):
        table = original.groupby(keys, dropna=False)['satisfaction'].agg(['sum', 'count']).reset_index()
        matching = raw[keys].merge(table, on=keys, how='left', sort=False)
        assert len(matching) == len(raw)
        n = matching['count'].fillna(0).to_numpy(dtype=np.float32)
        s = matching['sum'].fillna(0).to_numpy(dtype=np.float32)
        output[f'source_count_{i}'] = np.log1p(n)
        for strength in (5., 30.):
            output[f'source_prior_{i}_{int(strength)}'] = (s + strength * prior) / (n + strength)
    return output


train = pd.read_csv(ROOT / "data" / "train.csv")
test = pd.read_csv(ROOT / "data" / "test.csv")
source = pd.read_csv(ROOT / "data" / "original" / "data.csv")
source['satisfaction'] = source['satisfaction'].astype(int)
y = train['satisfaction'].astype(int).to_numpy()
folds = pd.read_csv(ROOT / "data" / "folds.csv")
fold = folds[[c for c in folds.columns if "fold" in c.lower()][0]].to_numpy()

# kiểm tra: fold của bản gốc (StratifiedKFold seed 42) có trùng folds.csv không
ref = np.zeros(len(y), int)
for k, (_, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(train, y)):
    ref[b] = k
print(f"fold bản gốc trùng folds.csv: {(ref == fold).mean():.4%} dòng", flush=True)

X_base, T_base = features(train), features(test)
for col in CATEGORICAL:
    vocab = pd.Index(X_base[col].cat.categories).union(T_base[col].cat.categories)
    vocab = vocab.union(pd.Index(source[col].dropna().unique()))
    for df in (X_base, T_base):
        df[col] = pd.Categorical(df[col], categories=vocab)
X_aug, T_aug = with_source_priors(X_base, train, source), with_source_priors(T_base, test, source)
print('feature:', X_base.shape[1], '->', X_aug.shape[1], flush=True)


def fit_oof(name, X, Xt):
    oof, pred = np.zeros(len(y)), np.zeros(len(Xt))
    for k in range(5):
        t0 = time.time()
        tr = np.where(fold != k)[0]
        va = fold == k
        m = lgb.LGBMClassifier(**PARAMS)
        if ES_ON_VAL:                                # như bản gốc: dừng sớm trên fold đang chấm
            m.fit(X.iloc[tr], y[tr], eval_set=[(X[va], y[va])], callbacks=[lgb.early_stopping(65, verbose=False)])
        else:
            tr = np.random.default_rng(k).permutation(tr)
            cut = int(len(tr) * 0.9)                 # dừng sớm trên 10% của phần train, không dùng fold đang chấm
            m.fit(X.iloc[tr[:cut]], y[tr[:cut]], eval_set=[(X.iloc[tr[cut:]], y[tr[cut:]])],
                  callbacks=[lgb.early_stopping(65, verbose=False)])
        oof[va] = m.predict_proba(X[va])[:, 1]
        pred += m.predict_proba(Xt)[:, 1] / 5
        print(f'{name} fold {k}: AUC {roc_auc_score(y[va], oof[va]):.6f}, cây {m.best_iteration_}, '
              f'{time.time() - t0:.0f}s', flush=True)
        del m
        gc.collect()
    return oof, pred


raw = fit_oof('raw', X_base, T_base)
pri = fit_oof('prior', X_aug, T_aug)
blend = (0.4 * raw[0] + 0.6 * pri[0], 0.4 * raw[1] + 0.6 * pri[1])
if ES_ON_VAL:
    save_preds("A_kozy_esval_v1", *blend, y)
else:
    save_preds("A_kozy_raw_v1", *raw, y)
    save_preds("A_kozy_prior_v1", *pri, y)
    save_preds("A_kozy_v1", *blend, y)
