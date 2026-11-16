from typing import Optional, Sequence

import numpy as np


def _style(i: int):
    from mirm.run.utils.viz import PALETTE, LINESTYLES, MARKERS
    return PALETTE[i % len(PALETTE)], LINESTYLES[i % len(LINESTYLES)], MARKERS[i % len(MARKERS)]


def curves(
    axis,
    x,
    Y,
    *,
    labels: Optional[Sequence[str]] = None,
    highlight: Optional[Sequence[int]] = None,
    mean: bool = False,
    ylim: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    alpha: Optional[float] = None,
    linewidth: Optional[float] = None,
    marker: Optional[str] = None,
    color: Optional[str] = None,
) -> None:
    """Few labeled curves are told apart by color, dash and (marker="auto") marker; many unlabeled curves are drawn
    as a faint bundle in which highlighted curves keep the same width and only change color."""
    from mirm.run.utils.viz import RED, INK, SKY

    x = np.asarray(x, dtype=float)
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    n = len(Y)

    highlighted = set() if highlight is None else {int(i) for i in highlight}
    base_alpha = alpha if alpha is not None else min(0.9, max(0.05, 8.0 / max(1, n)))
    base_width = linewidth if linewidth is not None else (1.4 if n <= 8 else 0.8)
    every = max(1, len(x) // 12)

    for i, y in enumerate(Y):
        label = labels[i] if labels is not None else None
        if i in highlighted:
            axis.plot(x, y, color=RED, linewidth=base_width, alpha=min(1.0, base_alpha * 2.5 + 0.2), label=label)
        elif labels is not None:
            c, ls, mk = _style(i)
            axis.plot(x, y, color=color or c, linestyle=ls, linewidth=base_width, label=label,
                      marker=mk if marker == "auto" else marker, markevery=every if marker == "auto" else None,
                      markersize=3.5)
        else:
            axis.plot(x, y, color=color or SKY, linewidth=base_width, alpha=base_alpha, marker=marker)

    if mean and n:
        axis.plot(x, Y.mean(axis=0), color=INK, linewidth=1.8, label="mean")

    if ylim is not None:
        axis.set_ylim(*ylim)
    if title is not None:
        axis.set_title(title)
    if xlabel is not None:
        axis.set_xlabel(xlabel)
    if ylabel is not None:
        axis.set_ylabel(ylabel)
    if labels is not None or mean:
        axis.legend(fontsize=7)
    axis.grid(alpha=0.15)


def strip(axis, x, density, height: float = 0.12, color: str = "0.55", alpha: float = 0.35) -> None:
    """Small grey density (e.g., the students' posterior distribution) drawn just above the x-axis."""
    x = np.asarray(x, dtype=float)
    density = np.asarray(density, dtype=float)
    if not np.isfinite(density).any() or np.nanmax(density) <= 0:
        return
    low, high = axis.get_ylim()
    top = low + height * (high - low) * density / np.nanmax(density)
    axis.fill_between(x, low, top, color=color, alpha=alpha, linewidth=0, zorder=0)
    axis.set_ylim(low, high)


def shade(axis, x, weight, strength: float = 0.22) -> None:
    """Background that is white where weight is high and grey where it is low (weight scaled to [0, 1])."""
    x = np.asarray(x, dtype=float)
    weight = np.asarray(weight, dtype=float)
    if not np.isfinite(weight).any() or np.nanmax(weight) <= 0:
        return
    level = 1.0 - np.clip(weight / np.nanmax(weight), 0, 1)
    low, high = axis.get_ylim()
    axis.imshow(level[None, :], extent=(x[0], x[-1], low, high), aspect="auto", cmap="Greys", vmin=0, vmax=1,
                alpha=strength, zorder=0, interpolation="bilinear")
    axis.set_ylim(low, high)
    axis.set_xlim(x[0], x[-1])
