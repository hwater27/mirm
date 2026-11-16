from typing import Optional, Sequence

import numpy as np


def scatter(
    axis,
    x,
    y,
    *,
    labels: Optional[Sequence[str]] = None,
    diagonal: bool = False,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    size: float = 14.0,
) -> None:
    xs = x if isinstance(x, (list, tuple)) else [x]
    ys = y if isinstance(y, (list, tuple)) else [y]

    from mirm.run.utils.viz.icc import _style

    for i, (xi, yi) in enumerate(zip(xs, ys)):
        color, _, marker = _style(i)
        axis.scatter(np.asarray(xi, dtype=float), np.asarray(yi, dtype=float), s=size, alpha=0.75,
                     color=color, marker=marker, label=None if labels is None else labels[i])

    if diagonal:
        low = min(float(np.nanmin(np.asarray(v, dtype=float))) for v in (*xs, *ys))
        high = max(float(np.nanmax(np.asarray(v, dtype=float))) for v in (*xs, *ys))
        axis.plot([low, high], [low, high], color="black", linewidth=1.0, linestyle="--")

    if labels is not None:
        axis.legend(fontsize=7)
    if title is not None:
        axis.set_title(title)
    if xlabel is not None:
        axis.set_xlabel(xlabel)
    if ylabel is not None:
        axis.set_ylabel(ylabel)
    axis.grid(alpha=0.15)
