"""Shake-up public -> private ở các tập Playground S3-S6 (dataset yangkuangou/playground-leaderboard-shakeup-s3-s6).

Câu hỏi: đội đứng ở phần trăm public X thì xác suất vào top 5% private là bao nhiêu?
Lọc: cuộc thi metric AUC, đã kết thúc, >= 1000 đội.

Chạy: python notebooks/A_shakeup.py
"""
import os

import numpy as np
import pandas as pd

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "shakeup")
comp = pd.read_csv(os.path.join(D, "competitions.csv"))
pairs = pd.read_csv(os.path.join(D, "team_rank_and_score_pairs.csv"))
pairs = pairs[pairs["paired_ranks_available"] == True]  # noqa: E712

auc = comp[comp["EvaluationAlgorithmName"].astype(str).str.contains("auc|Area Under", case=False)
           & ~comp["EvaluationAlgorithmName"].astype(str).str.contains("Columnwise")
           & (comp["eligible_final"] == True)]  # noqa: E712
p = pairs[pairs["competition_id"].isin(auc["competition_id"])].copy()
n = p.groupby("competition_id")["PublicLeaderboardRank"].transform("count")
p = p[n >= 1000]
n = p.groupby("competition_id")["PublicLeaderboardRank"].transform("count")
p["pub_pct"] = p["PublicLeaderboardRank"] / n * 100
p["priv_top5"] = p["PrivateLeaderboardRank"] <= 0.05 * n
p = p.merge(auc[["competition_id", "competition_slug" if "competition_slug" in auc else "Title", "season"]],
            on="competition_id", how="left", suffixes=("", "_c"))

print(f"{p['competition_id'].nunique()} cuộc thi AUC >= 1000 đội: "
      + ", ".join(sorted(p["competition_slug"].str.replace("playground-series-", "").unique())))
bins = [0, 2, 4, 5, 6, 8, 10, 12, 14, 16, 20, 30]
p["band"] = pd.cut(p["pub_pct"], bins)
for name, q in [("S3-S6", p), ("chỉ S5-S6", p[p["season"] >= 5])]:
    t = q.groupby("band", observed=True).agg(teams=("priv_top5", "size"), p_top5=("priv_top5", "mean"))
    print(f"\n{name}: phần trăm public -> tỉ lệ vào top 5% private")
    print(t.to_string(formatters={"p_top5": "{:.1%}".format}))
    per = q[(q["pub_pct"] > 11) & (q["pub_pct"] <= 14)].groupby("competition_slug")["priv_top5"].mean()
    print(f"  public 11-14% theo từng cuộc thi: min {per.min():.1%}, trung vị {per.median():.1%}, max {per.max():.1%}")

# Độ "chật" của bảng: chênh điểm public giữa hạng 5% và hạng 13%. S6E10 hiện tại: 0.96188 - 0.96184 = 4e-5.
print("\nTừng cuộc thi: chênh public (hạng 5% - hạng 13%) và tỉ lệ đội public 11-14% vào top 5% private")
rows = []
for slug, q in p.groupby("competition_slug"):
    q = q.sort_values("PublicLeaderboardRank")
    s = q["matched_public_score"].to_numpy()
    k5, k13 = int(0.05 * len(q)), int(0.13 * len(q))
    gap = abs(np.nanmedian(s[max(k5 - 3, 0):k5 + 3]) - np.nanmedian(s[k13 - 3:k13 + 3]))
    band = q[(q["pub_pct"] > 11) & (q["pub_pct"] <= 14)]
    rows.append(dict(comp=slug.replace("playground-series-", ""), teams=len(q), gap_5_13=gap,
                     p_top5_from_11_14=band["priv_top5"].mean()))
print(pd.DataFrame(rows).sort_values("gap_5_13").to_string(index=False,
      formatters={"gap_5_13": "{:.6f}".format, "p_top5_from_11_14": "{:.1%}".format}))
