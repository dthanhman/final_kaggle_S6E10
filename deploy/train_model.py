"""Train model triển khai (E7): LightGBM trên 21 cột gốc, cấu hình tốt nhất của E4 (`random_s1`), trên toàn bộ train.

Chạy từ thư mục gốc repo:  python deploy/train_model.py
Ghi ra: deploy/model_lgb.txt (model LightGBM) và deploy/meta.json (tên cột, giá trị category, khoảng giá trị, CV).
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import CAT_COLS, FEATURES, NUM_COLS, RESULTS_DIR, SCORE_COLS, TARGET  # noqa: E402
from data import load_data  # noqa: E402

OUT = ROOT / "deploy"
LR = 0.1  # learning rate dùng khi dò ở E4


def to_frame(df, categories):
    """21 cột gốc; category đặt đúng danh sách giá trị lúc train để mã hoá khớp khi dự đoán."""
    X = df[FEATURES].copy()
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=categories[c])
    return X


def main():
    e4 = pd.read_csv(RESULTS_DIR / "E4_tuning" / "E4_final.csv").set_index("config")
    best = e4.loc["random_s1"]
    params = json.loads(best["params"])
    # E4 dừng sớm trên 10% phần train của fold (TB 177 cây); train trên toàn bộ dữ liệu thì tăng 10% số cây.
    n_trees = int(round(best["trees_mean"] * 1.1))

    train, _ = load_data()
    categories = {c: sorted(train[c].dropna().unique().tolist()) for c in CAT_COLS}
    X, y = to_frame(train, categories), train[TARGET].values
    t0 = time.time()
    model = lgb.LGBMClassifier(n_estimators=n_trees, learning_rate=LR, subsample_freq=1, random_state=42,
                               verbose=-1, **params).fit(X, y)
    model.booster_.save_model(str(OUT / "model_lgb.txt"))

    ranges = {c: [float(train[c].min()), float(train[c].max()), float(train[c].median())] for c in NUM_COLS}
    meta = {
        "features": FEATURES, "cat_cols": CAT_COLS, "score_cols": SCORE_COLS, "num_cols": NUM_COLS,
        "categories": categories, "num_ranges": ranges, "params": params, "n_trees": n_trees, "learning_rate": LR,
        "cv_auc": round(float(best["cv_auc"]), 6), "cv_auc_folds1_4": round(float(best["folds1_4_mean"]), 6),
        "train_rows": int(len(train)), "positive_rate": round(float(y.mean()), 4),
        "note": "LightGBM 21 cột gốc, cấu hình random_s1 của E4; CV đo ở E4 (learning rate 0,1, 5 fold).",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    size = (OUT / "model_lgb.txt").stat().st_size / 1e6
    print(f"Đã train {n_trees} cây trong {time.time() - t0:.0f}s; model {size:.1f} MB -> {OUT / 'model_lgb.txt'}")


if __name__ == "__main__":
    main()
