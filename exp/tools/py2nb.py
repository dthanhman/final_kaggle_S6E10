"""Chuyển script dạng ô `# %%` thành notebook và chạy để notebook có sẵn output.

Chạy: python exp/tools/py2nb.py exp/E1_eda.py            -> exp/E1_eda.ipynb (đã chạy)
      python exp/tools/py2nb.py exp/E1_eda.py --no-run   -> chỉ chuyển, không chạy

Quy ước trong file .py:
    # %% [markdown]
    # # Tiêu đề
    # Đoạn văn...
    # %%
    code...
"""
import argparse
import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

CELL = re.compile(r"^# %%( \[markdown\])?\s*$")


def parse(text):
    cells, kind, buf = [], None, []

    def flush():
        src = "\n".join(buf).strip("\n")
        if kind is None or not src.strip():
            return
        if kind == "md":
            lines = [ln[2:] if ln.startswith("# ") else ln.lstrip("#") for ln in src.split("\n")]
            cells.append(nbformat.v4.new_markdown_cell("\n".join(lines)))
        else:
            cells.append(nbformat.v4.new_code_cell(src))

    for line in text.split("\n"):
        m = CELL.match(line)
        if m:
            flush()
            kind, buf = ("md" if m.group(1) else "code"), []
        else:
            buf.append(line)
    flush()
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("--no-run", action="store_true")
    ap.add_argument("--timeout", type=int, default=3600, help="giây tối đa cho mỗi ô")
    args = ap.parse_args()

    src = Path(args.script).resolve()
    nb = nbformat.v4.new_notebook()
    nb.cells = parse(src.read_text(encoding="utf-8"))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    out = src.with_suffix(".ipynb")

    if not args.no_run:
        t0 = time.time()
        NotebookClient(nb, timeout=args.timeout, kernel_name="python3",
                       resources={"metadata": {"path": str(src.parent)}}).execute()
        print(f"Đã chạy {len(nb.cells)} ô trong {time.time() - t0:.0f}s")
    nbformat.write(nb, out)
    print(f"Đã lưu {out}")


if __name__ == "__main__":
    sys.exit(main())
