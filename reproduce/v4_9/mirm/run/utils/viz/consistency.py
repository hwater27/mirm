"""Item x split (c, d) consistency tiles with a perceptually uniform bivariate color key (c: blue, d: red)."""
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

CMAX = DMAX = 5

# Perceptually uniform bivariate key in OKLab. Every up/down/left/right step has the same perceived color
# difference (OKLab dE ~= 0.087). (5,1) (many correct states) is blue, (1,5) (many incorrect states) is red, and
# (1,1) is a very light grey with a faint purple tint, so the edges (c,1) and (1,d) and the diagonal (c,c) all start
# from the same hue family and grow monotonically in chroma and darkness (diagonal stays purple). From the flat
# lattice through these three corners, the interior was bent outward in chroma (optimized with the corners fixed,
# equal neighbor steps, sRGB gamut and smoothness penalties) so mixed configurations are less muddy.
# Neighbor steps stay distinguishable under simulated red-green color-vision deficiency.
_LAB = np.array([          # [c-1][d-1] -> OKLab (L, a, b)
    [[0.97500, 0.01061, -0.01061], [0.89899, 0.05406, -0.00353], [0.82077, 0.08984, 0.01303], [0.74539, 0.12625, 0.03819], [0.67395, 0.16686, 0.06823]],
    [[0.89686, 0.00311, -0.04980], [0.82311, 0.04976, -0.04641], [0.74869, 0.09383, -0.03586], [0.67434, 0.13206, -0.01210], [0.60388, 0.17397, 0.01685]],
    [[0.82010, -0.01218, -0.08884], [0.74881, 0.03772, -0.09023], [0.67412, 0.08100, -0.07934], [0.60207, 0.12597, -0.06050], [0.53309, 0.17204, -0.03397]],
    [[0.74640, -0.03842, -0.12793], [0.67505, 0.01173, -0.12878], [0.60275, 0.06020, -0.12419], [0.53078, 0.10590, -0.10659], [0.46466, 0.15855, -0.08610]],
    [[0.67395, -0.06823, -0.16686], [0.60193, -0.01915, -0.16486], [0.53147, 0.03244, -0.16558], [0.46278, 0.08529, -0.15694], [0.39581, 0.13628, -0.13445]]
])


def _oklab_to_srgb(lab):
    L, a, b = lab
    l, m, s = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3, (L - 0.1055613458 * a - 0.0638541728 * b) ** 3, \
              (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    lin = np.clip([4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
                   -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
                   -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s], 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)


def cd_color(c: int, d: int) -> np.ndarray:
    return _oklab_to_srgb(_LAB[min(c, CMAX) - 1, min(d, DMAX) - 1])


def _order(cfg: dict, splits: Sequence[str], items: Sequence[int], sort: str) -> list[int]:
    """Rows by the full-data configuration, then by agreement with it (more splits first), then by the total Manhattan
    distance of the splits from it, then by item index; or simply by item index (sort="index").

    The configurations are ordered by c + d; within a diagonal, sort="zigzag" (Cantor order) runs c ascending when
    c + d is odd and descending when even, so consecutive configurations are always lattice neighbours ((2,1) ->
    (3,1), (1,3) -> (1,4)) and the colors change without jumps; sort="config" runs c ascending on every diagonal."""
    if sort == "index":
        return sorted(items)
    full = cfg["full"]
    agree = {j: sum(cfg[s][j] == full[j] for s in splits) for j in items}
    dist = {j: sum(abs(cfg[s][j][0] - full[j][0]) + abs(cfg[s][j][1] - full[j][1]) for s in splits) for j in items}

    def within(j):
        c, d = full[j]
        return -c if sort == "zigzag" and (c + d) % 2 == 0 else c

    return sorted(items, key=lambda j: (full[j][0] + full[j][1], within(j), -agree[j], dist[j], j))


def consistency_tile(axis, cfg: Optional[dict], splits: Sequence[str], title: str, *, n_blank: int = 0,
                     items: Optional[Sequence[int]] = None, sort: str = "zigzag") -> None:
    """cfg[split][item] = (c, d) with split "full" first; cfg=None draws an empty (white) placeholder."""
    blank = cfg is None
    if blank:
        order, agree_rate = list(range(n_blank)), None
        img = np.ones((n_blank, len(splits), 3))
    else:
        items = sorted(cfg["full"]) if items is None else [j for j in items if j in cfg["full"]]
        order = _order(cfg, splits, items, sort)
        agree_rate = np.mean([all(cfg[s][j] == cfg["full"][j] for s in splits) for j in items])
        img = np.array([[cd_color(*cfg[s][j]) for s in splits] for j in order])
    axis.imshow(img, aspect="auto", interpolation="nearest")
    axis.set_xticks(range(len(splits)), splits, rotation=90, fontsize=7)
    axis.set_yticks(range(len(order)), [str(j) for j in order], fontsize=5 if len(order) > 25 else 7)
    axis.set_xticks(np.arange(-.5, len(splits)), minor=True)
    axis.set_yticks(np.arange(-.5, len(order)), minor=True)
    axis.grid(which="minor", color="0.8", lw=0.3)
    axis.tick_params(which="minor", length=0)
    axis.axvline(0.5, color="k", lw=1)
    for side in ("top", "right"):
        axis.spines[side].set_visible(True)
    summary = "results pending" if blank else f"all splits agree: {agree_rate:.0%} of {len(order)} items"
    axis.set_title(f"{title}\n({summary})", fontsize=9)
    axis.set_ylabel("item" + (" (by item index)" if sort == "index" else " (sorted by full-data config)"), fontsize=7)


def consistency_key(axis) -> None:
    """Color key with c on the vertical axis and d on the horizontal axis."""
    img = np.array([[cd_color(c, d) for d in range(1, DMAX + 1)] for c in range(1, CMAX + 1)])
    axis.imshow(img, origin="lower", interpolation="nearest")
    for c in range(1, CMAX + 1):
        for d in range(1, DMAX + 1):
            axis.text(d - 1, c - 1, f"{c},{d}", ha="center", va="center", fontsize=6,
                      color="white" if np.dot(cd_color(c, d), [0.3, 0.59, 0.11]) < 0.5 else "black")
    axis.set_xticks(range(DMAX), range(1, DMAX + 1), fontsize=7)
    axis.set_yticks(range(CMAX), range(1, CMAX + 1), fontsize=7)
    axis.set_xlabel("d (incorrect states)", fontsize=7)
    axis.set_ylabel("c (correct states)", fontsize=7)
    axis.set_title("color key", fontsize=8)


def consistency_figure(panels: Sequence[tuple], path, *, sort: str = "zigzag", cell: float = 0.075,
                       equal_height: bool = True) -> Path:
    """panels: (title, cfg, splits) or (title, None, splits, n_blank) in display order; every panel has the same
    total height (equal_height) or the same row height."""
    import matplotlib.pyplot as plt
    from mirm.run.utils.viz import DPI

    rows = [len(p[1]["full"]) if p[1] is not None else p[3] for p in panels]
    pw, gap, top, bottom, left, key = 2.2, 0.85, 0.75, 0.75, 0.55, 1.9  # inches
    W = left + len(panels) * (pw + gap) + key + 0.3
    fig = plt.figure(figsize=(W, max(top + max(rows) * cell + bottom, key + top + bottom)))
    H = fig.get_figheight()
    for k, (panel, n) in enumerate(zip(panels, rows)):
        title, cfg, splits = panel[:3]
        h = max(rows) * cell if equal_height else n * cell
        axis = fig.add_axes([(left + k * (pw + gap)) / W, (H - top - h) / H, pw / W, h / H])
        consistency_tile(axis, cfg, splits, title, n_blank=n if cfg is None else 0, sort=sort)
    consistency_key(fig.add_axes([(left + len(panels) * (pw + gap)) / W, (H - top - key) / H, key / W, key / H]))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path
