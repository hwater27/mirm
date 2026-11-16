from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler

# Black-and-white with the Psychometrika sky blue; a few accents only where emphasis is needed.
INK = "#1a1a1a"
SKY = "#4b9cd3"
DEEP = "#1f5f8b"
GREY = "#8c8c8c"
LIGHT = "#c8c8c8"
RED = "#da2f3d"       # near-primary red matched to the sky blue (non-monotone items and dips)
ACCENT = "#d55e00"    # emphasis
ACCENT2 = "#e69f00"   # secondary emphasis
PALETTE = [INK, SKY, GREY, DEEP, ACCENT, LIGHT]
LINESTYLES = ["-", "--", ":", "-.", (0, (5, 1, 1, 1)), (0, (1, 1))]
MARKERS = ["o", "s", "^", "D", "v", "P"]
DPI = 200

plt.rcParams["axes.prop_cycle"] = cycler(color=PALETTE)
plt.rcParams["font.size"] = 9
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["legend.frameon"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

from mirm.run.utils.viz.icc import curves, strip, shade
from mirm.run.utils.viz.grid import heatmap
from mirm.run.utils.viz.barplot import bars, hist, bars_and_lines, grouped_row
from mirm.run.utils.viz.scatter import scatter
from mirm.run.utils.viz.ndim import theta_cmaps, clean_3d, cloud3d, path3d_gradient, theta_iccs, fold_inset
from mirm.run.utils.viz.figures import (icc_overview, dip_panel, characteristics_row, confusion, binary_summary,
                                        multiclass_summary, parameter_scatter, monotonicity_figure)
from mirm.run.utils.viz.consistency import cd_color, consistency_figure, consistency_key

__all__ = ["curves", "strip", "shade", "heatmap", "bars", "hist", "bars_and_lines", "grouped_row", "scatter",
           "theta_cmaps", "clean_3d", "cloud3d", "path3d_gradient", "theta_iccs", "fold_inset",
           "icc_overview", "dip_panel", "characteristics_row", "confusion", "binary_summary",
           "multiclass_summary", "parameter_scatter", "monotonicity_figure", "cd_color", "consistency_figure",
           "consistency_key", "figure", "save"]


def figure(rows: int = 1, columns: int = 1, size: tuple[float, float] = (4.2, 3.4), **kwargs):
    fig, axes = plt.subplots(rows, columns, figsize=(size[0] * columns, size[1] * rows), squeeze=False, **kwargs)
    return fig, axes


def save(fig, path, pdf: bool = False, tight: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    if pdf:
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path
