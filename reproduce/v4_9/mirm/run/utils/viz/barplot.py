from typing import Optional, Sequence

import numpy as np


def bars(
    axis,
    values,
    *,
    errors=None,
    labels: Optional[Sequence[str]] = None,
    groups: Optional[Sequence[str]] = None,
    best: Optional[str] = None,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    annotate: Optional[str] = "{:.3g}",
) -> None:
    from mirm.run.utils.viz import SKY, GREY, DEEP, LIGHT, INK

    values = np.atleast_2d(np.asarray(values, dtype=float))
    errors = None if errors is None else np.atleast_2d(np.asarray(errors, dtype=float))
    n_groups, n_categories = values.shape
    group_colors = [SKY, GREY, DEEP, LIGHT, INK]
    hatches = [None, "//", "..", "xx", "\\\\"]

    width = 0.8 / n_groups
    positions = np.arange(n_categories)

    for g in range(n_groups):
        offsets = positions + (g - (n_groups - 1) / 2) * width
        colors = group_colors[g % len(group_colors)]
        if best is not None and n_groups == 1:
            finite = np.isfinite(values[g])
            target = np.nanmin(values[g]) if best == "min" else np.nanmax(values[g])
            colors = [DEEP if (finite[i] and values[g, i] == target) else SKY for i in range(n_categories)]
        rects = axis.bar(
            offsets, values[g], width,
            yerr=None if errors is None else errors[g],
            capsize=3,
            color=colors,
            hatch=hatches[g % len(hatches)] if n_groups > 1 else None,
            edgecolor="white" if n_groups > 1 else None,
            error_kw={"elinewidth": 0.8, "ecolor": INK},
            label=None if groups is None else groups[g],
        )
        if annotate:
            for rect, value in zip(rects, values[g]):
                if np.isfinite(value):
                    axis.annotate(
                        annotate.format(value),
                        (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7,
                    )

    if labels is not None:
        axis.set_xticks(positions, labels, rotation=25, ha="right")
    if groups is not None:
        axis.legend(fontsize=7)
    if title is not None:
        axis.set_title(title)
    if ylabel is not None:
        axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.15)


def bars_and_lines(
    axis,
    x,
    bar_values,
    line_values: dict,
    *,
    bar_label: str = "MiRM",
    bar_errors=None,
    line_errors: Optional[dict] = None,
    dodge: float = 0.07,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    log2_ticks: bool = True,
) -> None:
    """One model as light bars, the others as dashed/marked lines over the same categorical x. Error bars (e.g. sd
    over replicates) are drawn in each model's color; with line errors the lines are dodged in steps of `dodge` to
    slots -1, +1, -2, +2, ... (never 0, which holds the bar's error bar) so that no two error bars overlap."""
    from mirm.run.utils.viz import SKY, INK
    from mirm.run.utils.viz.icc import _style

    x = list(x)
    positions = np.arange(len(x))
    error_kw = {"elinewidth": 0.7, "capthick": 0.7}
    axis.bar(positions, np.asarray(bar_values, dtype=float), 0.6, color=SKY, alpha=0.35, label=bar_label,
             yerr=None if bar_errors is None else np.asarray(bar_errors, dtype=float), capsize=2,
             error_kw={**error_kw, "ecolor": SKY})
    slots = sorted([sign * k for k in range(1, len(line_values) + 1) for sign in (-1, 1)][:len(line_values)])
    for i, (name, values) in enumerate(line_values.items()):
        color, linestyle, marker = _style(i if i == 0 else i + 1)  # skip the sky slot used by the bars
        color = color if color != SKY else INK
        errors = None if line_errors is None else line_errors.get(name)
        if errors is None:
            axis.plot(positions, np.asarray(values, dtype=float), color=color, linestyle=linestyle, marker=marker,
                      markersize=4, linewidth=1.2, label=name)
        else:
            offset = slots[i] * dodge
            axis.errorbar(positions + offset, np.asarray(values, dtype=float), yerr=np.asarray(errors, dtype=float),
                          color=color, linestyle=linestyle, marker=marker, markersize=4, linewidth=1.2, capsize=2,
                          label=name, **error_kw)
    axis.set_xticks(positions, [str(v) for v in x])
    if title is not None:
        axis.set_title(title)
    if xlabel is not None:
        axis.set_xlabel(xlabel)
    if ylabel is not None:
        axis.set_ylabel(ylabel)
    axis.legend(fontsize=7)
    axis.grid(axis="y", alpha=0.15)


def hist(
    axis,
    samples,
    *,
    labels: Optional[Sequence[str]] = None,
    bins: int = 30,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    density: bool = True,
) -> None:
    from mirm.run.utils.viz.icc import _style

    samples = [np.asarray(s, dtype=float) for s in (samples if isinstance(samples, (list, tuple)) else [samples])]
    for i, sample in enumerate(samples):
        sample = sample[np.isfinite(sample)]
        color, linestyle, _ = _style(i)
        axis.hist(sample, bins=bins, histtype="step", linewidth=1.4, density=density, color=color, linestyle=linestyle,
                  label=None if labels is None else labels[i])
    if labels is not None:
        axis.legend(fontsize=7)
    if title is not None:
        axis.set_title(title)
    if xlabel is not None:
        axis.set_xlabel(xlabel)
    axis.grid(alpha=0.15)


def grouped_row(groups, path, *, panel_size: tuple[float, float] = (3.6, 3.3), group_gap: Optional[float] = None,
                errors: bool = True):
    """One row of bars-and-lines panels arranged in metric groups, e.g. [("ICC RMSE", [panel, panel]),
    ("ability RMSE", [panel, panel])]; panel = dict(x=..., bars=..., lines={name: values}, bar_label=..., title=...,
    xlabel=..., bar_errors=..., line_errors={name: errors}). Panels of a group share the y-axis and a header; groups
    are separated by a rule and by group_gap inches between the axes boxes. group_gap=None spaces all panels evenly:
    the blank space between groups (after the y tick labels and label of the next group's first panel) equals the
    space between panels within a group. errors=False drops every error bar (and with it the dodging of the lines)."""
    import matplotlib.pyplot as plt
    from mirm.run.utils.viz import DPI

    counts = [len(panels) for _, panels in groups]
    n_groups = len(groups)
    left, right, top, bottom, inner_space = 0.05, 0.99, 0.80, 0.16, 0.28
    width = panel_size[0] * (sum(counts) + 0.175 * (n_groups - 1))
    fig = plt.figure(figsize=(width, panel_size[1] + 0.5))
    outer = fig.add_gridspec(1, n_groups, width_ratios=counts, left=left, right=right, top=top, bottom=bottom)

    def set_gap(gap):
        cell = (width * (right - left) - (n_groups - 1) * gap) / n_groups   # mean group width (gridspec units)
        outer.update(wspace=gap / cell if n_groups > 1 else 0.0)

    spans = []
    for k, (metric, panels) in enumerate(groups):
        inner = outer[0, k].subgridspec(1, len(panels), wspace=inner_space)
        first = None
        for j, panel in enumerate(panels):
            axis = fig.add_subplot(inner[0, j], sharey=first)
            first = first or axis
            bars_and_lines(axis, panel["x"], panel["bars"], panel["lines"], bar_label=panel.get("bar_label", "MiRM"),
                           bar_errors=panel.get("bar_errors") if errors else None,
                           line_errors=panel.get("line_errors") if errors else None,
                           title=panel.get("title"), xlabel=panel.get("xlabel"), ylabel=metric if j == 0 else None)
            if j > 0:
                axis.tick_params(labelleft=False)
                axis.get_legend().remove()
        spans.append((metric, first, axis))

    if group_gap is not None or n_groups == 1:
        set_gap(group_gap or 0.0)
    else:
        set_gap(0.3 * panel_size[0])
        for _ in range(4):   # the inner spacing scales with the axes width, so settle the gap in a few passes
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            a0, a1 = spans[0][1], spans[0][2]
            inner_gap = (a1.get_position().x0 - a0.get_position().x1) * width
            labels = max((first.get_position().x0 * width - first.yaxis.get_tightbbox(renderer).x0 / fig.dpi)
                         for _, first, _ in spans[1:])
            set_gap(inner_gap + labels)

    # headers after the layout is final, so that each rule spans exactly its panels
    for metric, first, last in spans:
        x0, x1 = first.get_position().x0, last.get_position().x1
        fig.text((x0 + x1) / 2, 0.97, metric, ha="center", va="top", fontsize=11, fontweight="bold")
        fig.add_artist(plt.Line2D([x0, x1], [0.925, 0.925], transform=fig.transFigure, color="0.3", linewidth=0.8))
    path = __import__("pathlib").Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path
