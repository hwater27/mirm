from typing import Optional, Sequence

import numpy as np
from matplotlib.patches import Rectangle


def heatmap(
    axis,
    M,
    *,
    xticks: Optional[Sequence] = None,
    yticks: Optional[Sequence] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    annotate: Optional[str] = "{:.1f}",
    text: Optional[np.ndarray] = None,
    winner: Optional[str | tuple[int, int]] = None,
    cmap: str = "Blues",
    center: Optional[float] = None,
    colorbar: bool = True,
    origin: str = "lower",
):
    M = np.asarray(M, dtype=float)
    finite = np.isfinite(M)

    kwargs = {}
    if center is not None:
        limit = float(np.abs(M[finite] - center).max()) if finite.any() else 1.0
        kwargs.update(vmin=center - limit, vmax=center + limit)
        cmap = "RdBu_r"

    image = axis.imshow(np.ma.masked_invalid(M), origin=origin, aspect="auto", cmap=cmap, **kwargs)

    if xticks is not None:
        axis.set_xticks(range(M.shape[1]), xticks, rotation=25 if any(len(str(t)) > 3 for t in xticks) else 0, ha="right" if any(len(str(t)) > 3 for t in xticks) else "center")
    if yticks is not None:
        axis.set_yticks(range(M.shape[0]), yticks)
    if xlabel is not None:
        axis.set_xlabel(xlabel)
    if ylabel is not None:
        axis.set_ylabel(ylabel)
    if title is not None:
        axis.set_title(title)

    def ink(i, j):  # white text on dark cells
        rgba = image.cmap(image.norm(M[i, j])) if np.isfinite(M[i, j]) else (1, 1, 1, 1)
        return "white" if 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2] < 0.5 else "black"

    if text is not None:
        for (i, j), value in np.ndenumerate(np.asarray(text, dtype=object)):
            if value:
                axis.text(j, i, str(value), ha="center", va="center", fontsize=7, color=ink(i, j))
    elif annotate:
        for (i, j), value in np.ndenumerate(M):
            if np.isfinite(value):
                axis.text(j, i, annotate.format(value), ha="center", va="center", fontsize=7, color=ink(i, j))

    if winner is not None and finite.any():
        if winner == "min":
            index = np.unravel_index(np.nanargmin(M), M.shape)
        elif winner == "max":
            index = np.unravel_index(np.nanargmax(M), M.shape)
        else:
            index = tuple(winner)
        axis.add_patch(Rectangle((index[1] - 0.5, index[0] - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2.0))

    if colorbar:
        axis.figure.colorbar(image, ax=axis, shrink=0.8)
    return image
