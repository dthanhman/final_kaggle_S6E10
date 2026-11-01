"""Style hình dùng chung cho mọi thí nghiệm (báo cáo in, nền sáng).

Màu theo vai trò:
- NEG / POS: hai lớp target (không hài lòng / hài lòng), cũng là màu 2 series đầu.
- SEQ: ramp một màu xanh (nhạt -> đậm) cho độ lớn (heatmap tỉ lệ).
- DIV: xanh <-> xám <-> đỏ cho đại lượng có dấu (tương quan, chênh lệch).
- REF: đường tham chiếu (tỉ lệ chung, baseline).
"""
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.ticker import FixedFormatter, FixedLocator  # noqa: E402

from config import FIG_DIR  # noqa: E402

SURFACE = "#fcfcfb"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#898781"
GRID = "#e1e0d9"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE, AQUA = SERIES[0], SERIES[1], SERIES[2]
BLUE_LIGHT = "#9ec5f4"
NEG, POS = ORANGE, BLUE  # 0 = không hài lòng, 1 = hài lòng
REF = INK_3

SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_red_blue", ["#e34948", "#f0efec", "#2a78d6"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 5,
    "figure.dpi": 100, "savefig.bbox": "tight",
})


_DECIMAL = re.compile(r"(?<=\d)\.(?=\d)")
_MINUS = re.compile(r"(?<![\w.])-(?=\d)")


def vn_number(s):
    """'0.9217' -> '0,9217', '-0.52' -> '−0,52'. Không đổi chuỗi mathtext ($...$) hay khoảng như '6-15'."""
    if not s or s.startswith("$"):
        return s
    return _MINUS.sub("−", _DECIMAL.sub(",", s))


def _vn_figure(fig):
    """Đổi mọi số trên hình sang dấu phẩy thập phân, như phần chữ của báo cáo."""
    fig.canvas.draw()  # để vạch trục có nhãn
    for ax in fig.findobj(lambda o: isinstance(o, Axes)):  # gồm cả trục lồng (inset)
        for axis in (ax.xaxis, ax.yaxis):
            for minor in (False, True):
                ticks = axis.get_minorticklocs() if minor else axis.get_majorticklocs()
                labels = axis.get_minorticklabels() if minor else axis.get_majorticklabels()
                texts = [vn_number(t.get_text()) for t in labels]
                if len(texts) != len(ticks):
                    continue
                if minor:
                    axis.set_minor_locator(FixedLocator(ticks))
                    axis.set_minor_formatter(FixedFormatter(texts))
                else:
                    axis.set_major_locator(FixedLocator(ticks))
                    axis.set_major_formatter(FixedFormatter(texts))
    for t in fig.findobj(Text):
        t.set_text(vn_number(t.get_text()))


def save(fig, name, show=True):
    """Lưu reports/figures/<name>.png (150 dpi). show=True thì hiện luôn trong notebook."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _vn_figure(fig)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150)
    if show:
        try:
            from IPython.display import Image, display

            display(Image(filename=str(FIG_DIR / f"{name}.png")))
        except ImportError:
            pass
    plt.close(fig)
