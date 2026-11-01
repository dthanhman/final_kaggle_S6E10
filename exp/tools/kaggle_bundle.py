"""Gộp src/config.py + src/features.py + một script thành MỘT file, kèm kernel-metadata.json, để đẩy lên Kaggle.

Chạy: python exp/tools/kaggle_bundle.py exp/kaggle/E2_lgb.py --slug cs116-e2-lgb [--gpu] [--push]
      -> exp/kaggle/build/<slug>/  (thư mục này không commit)

Code chạy trên Kaggle vì vậy đúng y code trong repo; file gộp chỉ bỏ các dòng import nội bộ và sys.path.
Kết quả kernel tải về bằng: kaggle kernels output <user>/<slug> -p results/kaggle_<slug>
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULES = ["config", "features"]  # thứ tự gộp
INTERNAL = re.compile(r"^\s*from (config|features|cv|data|utils|plot) import .*$")
SYSPATH = re.compile(r"^\s*sys\.path\.insert\(0, .*src.*\)\s*$")
DATASETS = ["arseniyshutko/binary-aviation-satisfaction-129k"]
COMPETITION = "playground-series-s6e10"


def strip(text):
    lines = [ln for ln in text.splitlines() if not INTERNAL.match(ln) and not SYSPATH.match(ln)]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("--slug", required=True)
    ap.add_argument("--title", help="mặc định = slug; tiêu đề khác slug thì Kaggle đặt id khác")
    ap.add_argument("--user", default="thanhmn123")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    script = Path(args.script).resolve()
    parts = [f'"""Bản gộp tự động từ repo final_kaggle_S6E10: src/{", src/".join(m + ".py" for m in MODULES)}, '
             f'{script.relative_to(ROOT).as_posix()}. Sửa ở repo rồi gộp lại."""']
    for m in MODULES:
        parts.append(f"\n# ==== src/{m}.py ====\n" + strip((ROOT / "src" / f"{m}.py").read_text(encoding="utf-8")))
    parts.append(f"\n# ==== {script.relative_to(ROOT).as_posix()} ====\n" + strip(script.read_text(encoding="utf-8")))
    code = "\n".join(parts) + "\n"
    compile(code, args.slug, "exec")  # bắt lỗi cú pháp trước khi đẩy

    out = ROOT / "exp" / "kaggle" / "build" / args.slug
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.slug}.py").write_text(code, encoding="utf-8")
    meta = {
        "id": f"{args.user}/{args.slug}", "title": args.title or args.slug, "code_file": f"{args.slug}.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": args.gpu, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": DATASETS, "kernel_sources": [], "competition_sources": [COMPETITION],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Đã gộp {len(code.splitlines())} dòng -> {out}")
    if args.push:
        subprocess.run(["kaggle", "kernels", "push", "-p", str(out)], check=True)


if __name__ == "__main__":
    main()
