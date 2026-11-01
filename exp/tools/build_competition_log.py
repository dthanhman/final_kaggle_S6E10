"""Dựng nguồn số liệu thật của cuộc thi cho báo cáo.

1. results/competition_log.csv: bản sạch của experiments.csv ở repo cũ (cột notes có dấu phẩy không bọc nháy).
2. results/competition_submissions.csv: mọi bài đã nộp, kèm điểm public và private (Kaggle CLI).

Chạy: python exp/tools/build_competition_log.py --old-repo D:/CS116/DoAn
"""
import argparse
import ast
import csv
import re
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results"
COMP = "playground-series-s6e10"
ROW = re.compile(
    r"^(\d+)\s+(\S+)\s+(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(?:\.\d+)?)\s+(.*?)\s*(SubmissionStatus\.\w+)\s+(\S+)\s+(\S+)\s*$"
)


def kind_of(name):
    if name.startswith(("stack_", "ablation_")):
        return "stack"
    if name.startswith(("slot", "sub_", "hedge", "track3", "goodpjw")):
        return "bài nộp / trộn file công khai"
    if name.startswith(("probe", "pub_", "tabdpt")):
        return "thăm dò"
    return "mô hình đơn"


def build_log(old_repo):
    rows = []
    with open(Path(old_repo) / "experiments.csv", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(",", 5)
            parts += [""] * (6 - len(parts))
            rec = dict(zip(header, parts))
            rec["notes"] = rec["notes"].strip().strip('"').replace('""', '"')
            rec["notes"] = rec["notes"].replace("submission.csv do bạn trong nhóm nộp", "bài nộp của nhóm lúc")
            rows.append(rec)
    df = pd.DataFrame(rows)
    for c in ("cv_auc", "lb_auc"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.insert(3, "loại", df["model"].map(kind_of))
    df = df.drop(columns="author")  # báo cáo ghi công cho cả nhóm, không ghi theo người
    df.to_csv(OUT / "competition_log.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"competition_log.csv: {len(df)} dòng")


def build_submissions():
    out = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "--page-size", "200"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    rows = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith(("b'", 'b"')):  # CLI in dòng có ký tự đặc biệt dạng bytes
            line = ast.literal_eval(line).decode("utf-8").strip()
        m = ROW.match(line)
        if m:
            rows.append(m.groups())
    df = pd.DataFrame(rows, columns=["ref", "file", "date", "description", "status", "public", "private"])
    df["date"] = pd.to_datetime(df["date"], format="ISO8601").dt.floor("s")
    for c in ("public", "private"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop(columns="status").sort_values("date").reset_index(drop=True)
    df.to_csv(OUT / "competition_submissions.csv", index=False)
    print(f"competition_submissions.csv: {len(df)} bài nộp")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-repo", default="D:/CS116/DoAn")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    build_log(args.old_repo)
    build_submissions()
