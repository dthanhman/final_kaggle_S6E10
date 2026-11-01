"""Bước 1 thí nghiệm LLR: dữ liệu S6E10 có dấu vân tay của bộ sinh LLM (kiểu GReaT, số viết theo token GPT-2) không?

Ý từ writeup S6E9 (2nd/3rd place): nếu dòng được sinh bởi một LM fine-tune trên dữ liệu gốc thì
  (1) giá trị số gần như chỉ lấy lại từ dữ liệu gốc,
  (2) tần suất một giá trị trong dữ liệu cuộc thi so với dữ liệu gốc phụ thuộc vào cách GPT-2 tách token
      (kể cả khi so trong cùng số chữ số), còn bộ sinh kiểu GAN/diffusion thì không.
Chạy: python notebooks/A_llm_fingerprint.py
"""
import sys
import numpy as np
import pandas as pd
import tiktoken
from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding="utf-8")
enc = tiktoken.get_encoding("gpt2")
tr = pd.read_csv("data/train.csv")
te = pd.read_csv("data/test.csv")
og = pd.read_csv("data/original/data.csv")
comp = pd.concat([tr.drop(columns="satisfaction"), te], ignore_index=True)
NUM = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]


def text(v):
    # cách giá trị xuất hiện trong CSV gốc (Arrival Delay là float '18.0')
    return " " + (f"{v:.1f}" if isinstance(v, float) and not float(v).is_integer() is False and False else str(v))


print("== 1. Giá trị trong cuộc thi có nằm trong dữ liệu gốc không")
for c in NUM:
    a, b = comp[c].dropna(), og[c].dropna()
    sb = set(b.unique())
    print(f"{c:28s} khác nhau: cuộc thi {a.nunique():5d}, gốc {b.nunique():5d} | "
          f"tỉ lệ dòng cuộc thi có giá trị nằm trong gốc {a.isin(sb).mean():.4f} | bằng 0: cuộc thi {(a == 0).mean():.3f}, gốc {(b == 0).mean():.3f}")

print("\n== 2. Tần suất (cuộc thi / gốc) có phụ thuộc cách tách token GPT-2 không (cùng số chữ số)")
for c in ["Flight Distance", "Age", "Departure Delay in Minutes"]:
    fc = comp[c].dropna().astype(int).value_counts(normalize=True)
    fo = og[c].dropna().astype(int).value_counts(normalize=True)
    v = fc.index.intersection(fo.index)
    df = pd.DataFrame({"v": v, "r": np.log(fc[v].values / fo[v].values), "n_orig": og[c].dropna().astype(int).value_counts()[v].values})
    df = df[df.n_orig >= 20]                                   # bỏ giá trị quá hiếm để tỉ lệ ổn định
    df["digits"] = df.v.astype(str).str.len()
    df["ntok"] = [len(enc.encode(" " + str(x))) for x in df.v]
    df["tok1"] = [enc.decode(enc.encode(" " + str(x))[:1]) for x in df.v]
    df["last_digit"] = df.v % 10
    print(f"\n{c}: {len(df)} giá trị (>=20 dòng gốc)")
    g = df.groupby(["digits", "ntok"]).r.agg(["count", "mean", "std"])
    print(g.round(3).to_string())
    # trong cùng số chữ số, token count giải thích bao nhiêu phương sai của r, so với 1 biến đối chứng cùng số mức (chữ số cuối chẵn/lẻ)
    for d, q in df.groupby("digits"):
        if q.ntok.nunique() < 2 or len(q) < 30:
            continue
        eta = lambda key: q.groupby(key).r.transform("mean").var() / q.r.var()
        print(f"  {d} chữ số: R² theo số token {eta('ntok'):.3f} | đối chứng chẵn/lẻ chữ số cuối {eta(q.last_digit % 2):.3f} "
              f"| theo token đầu ({q.tok1.nunique()} nhóm) {eta('tok1'):.3f}")

print("\n== 3. Tỉ lệ hài lòng theo từng giá trị: cuộc thi có bám dữ liệu gốc không")
y = tr.satisfaction.astype(int)
yo = og.satisfaction.astype(int) if og.satisfaction.dtype != object else (og.satisfaction == "satisfied").astype(int)
for c in ["Flight Distance", "Age"]:
    a = pd.DataFrame({"v": tr[c], "y": y}).groupby("v").y.agg(["mean", "count"])
    b = pd.DataFrame({"v": og[c], "y": yo}).groupby("v").y.agg(["mean", "count"])
    j = a.join(b, lsuffix="_c", rsuffix="_o", how="inner")
    j = j[(j.count_c >= 50) & (j.count_o >= 20)]
    print(f"{c}: {len(j)} giá trị | spearman tỉ lệ hài lòng cuộc thi vs gốc {spearmanr(j.mean_c, j.mean_o)[0]:.3f}")
