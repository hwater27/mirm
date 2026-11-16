"""Composite panels built only from NumPy arrays (the analysis computes the arrays, these functions draw them)."""
from typing import Optional, Sequence

import numpy as np

from mirm.run.utils.viz.icc import curves, strip, shade
from mirm.run.utils.viz.grid import heatmap

LIGHT_GREY = "#b5b5b5"  # reference ICC of the dip panel
CHARACTERISTIC_LIMITS = {"probability": (-0.02, 1.02), "difficulty": (-4.0, 4.0)}


def icc_overview(axis, theta, P, *, highlight=None, density=None, title: Optional[str] = None) -> None:
    """All ICCs of a model as a faint bundle with the mean; flagged items change color only."""
    curves(axis, theta, P, highlight=highlight, mean=True, ylim=(-0.02, 1.02), title=title, xlabel="θ",
           ylabel="P(correct)")
    if density is not None:
        strip(axis, theta, density)


def dip_panel(
    axis,
    theta,
    icc,
    edges,
    binned,
    *,
    z: Optional[float] = None,
    density=None,
    reference_icc=None,
    reference_label: str = "2PL",
    compared: Optional[tuple[int, int]] = None,
    title: Optional[str] = None,
    note: Optional[str] = None,
    points_label: str = "rest-score bins",
) -> None:
    """ICC of a flagged item with binned observed proportions (mean ± 1 SE) and the students' posterior strip.

    binned = (p, sd, n) per bin; compared = (peak bin, dip bin) used by the z test. The ICC is grey, the empirical
    bins sky blue and the reference ICC a light grey dashed line; the dip of the ICC (from the shoulder of the local
    peak until the curve regains that level), the compared bins and their points are red."""
    from mirm.run.utils.viz import SKY, GREY, RED

    theta = np.asarray(theta, dtype=float)
    edges = np.asarray(edges, dtype=float)
    centers = (edges[:-1] + edges[1:]) / 2
    p, sd, n = (np.asarray(v, dtype=float) for v in binned)

    axis.set_ylim(-0.02, 1.02)
    axis.set_xlim(theta[0], theta[-1])
    compared = [b for b in (compared or ()) if 0 <= b < len(centers)]
    for b in compared:
        axis.axvspan(edges[b], edges[b + 1], color=RED, alpha=0.08, linewidth=0, zorder=0)
    if density is not None:
        strip(axis, theta, density)
    if reference_icc is not None:
        axis.plot(theta, reference_icc, color=LIGHT_GREY, linewidth=1.4, linestyle="--", label=reference_label)
    axis.plot(theta, icc, color=GREY, linewidth=1.5, label="MiRM")
    icc = np.asarray(icc, dtype=float)
    running = np.maximum.accumulate(icc)
    dip = int(np.argmax(running - icc))
    depth = running[dip] - icc[dip]
    if depth > 0:
        peak = int(np.flatnonzero(icc[:dip + 1] >= running[dip] - 0.1 * depth).max())
        back = np.flatnonzero(icc[dip:] >= icc[peak])
        stop = dip + (int(back[0]) if back.size else len(icc) - 1 - dip)
        axis.plot(theta[peak:stop + 1], icc[peak:stop + 1], color=RED, linewidth=1.5, label="dip")
    se = sd / np.sqrt(np.maximum(n, 1))
    keep = np.isfinite(p) & (n > 0)
    axis.errorbar(centers[keep], p[keep], yerr=se[keep], fmt="o", color=SKY, markersize=3, capsize=2,
                  elinewidth=0.8, label=points_label, zorder=3)
    marked = [b for b in compared if keep[b]]
    if marked:
        axis.errorbar(centers[marked], p[marked], yerr=se[marked], fmt="o", color=RED, markersize=4, capsize=2,
                      elinewidth=0.9, zorder=4)

    if z is not None and np.isfinite(z):
        axis.text(0.03, 0.97, f"z = {z:.2f}", transform=axis.transAxes, ha="left", va="top", fontsize=8,
                  color=RED, fontweight="bold")
    if note:
        axis.text(0.03, 0.88, note, transform=axis.transAxes, ha="left", va="top", fontsize=6.5, color=GREY)
    if title is not None:
        axis.set_title(title, fontsize=9)
    axis.set_xlabel("θ")
    axis.set_ylabel("P(correct)")
    axis.legend(fontsize=6.5, loc="lower right")
    axis.grid(alpha=0.15)


def characteristics_row(
    axes: Sequence,
    theta,
    char: dict,
    *,
    limits: Optional[dict] = None,
    names: Sequence[str] = ("probability", "discrimination", "difficulty", "acceleration", "information"),
    row_label: Optional[str] = None,
    first_row: bool = False,
    last_row: bool = False,
) -> None:
    """One item: characteristic curves with fixed y-limits. The discrimination and difficulty panels are shaded by
    the item information (white = informative, grey = uninformative) and marked at the most informative theta."""
    from mirm.run.utils.viz import INK, ACCENT

    theta = np.asarray(theta, dtype=float)
    limits = {**CHARACTERISTIC_LIMITS, **(limits or {})}
    info = np.asarray(char["information"], dtype=float)
    peak = int(np.nanargmax(info)) if np.isfinite(info).any() else None

    for c, (axis, name) in enumerate(zip(axes, names)):
        y = np.asarray(char[name], dtype=float)
        axis.plot(theta, y, color=INK, linewidth=1.2)
        if name in limits:
            axis.set_ylim(*limits[name])
        axis.set_xlim(theta[0], theta[-1])
        if name in ("discrimination", "difficulty"):
            shade(axis, theta, info)
            if peak is not None and np.isfinite(y[peak]):
                axis.plot(theta[peak], y[peak], "o", color=ACCENT, markersize=3.5, zorder=3)
        if first_row:
            axis.set_title(name)
        if c == 0 and row_label is not None:
            axis.set_ylabel(row_label)
        if last_row:
            axis.set_xlabel("θ")
        axis.grid(alpha=0.15)


def confusion(axis, counts, *, row_labels, col_labels, title: Optional[str] = None, normalize: bool = True,
              text=None, ylabel: bool = True, summary: Optional[str] = None) -> None:
    """Confusion matrix (rows: truth, columns: estimate) with square cells, row-normalized by default, with counts in
    the cells; summary (e.g. F1, precision, recall) is printed to the right of the matrix."""
    counts = np.asarray(counts, dtype=float)
    rates = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1) if normalize else counts
    if text is None:
        text = np.array([[f"{rates[i, j]:.2f}\n({counts[i, j]:.0f})" for j in range(counts.shape[1])]
                         for i in range(counts.shape[0])], dtype=object)
    heatmap(axis, rates, xticks=col_labels, yticks=row_labels, text=text, title=title, cmap="Blues", origin="upper",
            colorbar=False)
    axis.set_aspect("equal", adjustable="box")  # square cells: a 2x2 matrix is a square, not squeezed by the panel
    if summary is not None:  # metrics to the right of the matrix
        axis.text(1.12, 0.5, summary, transform=axis.transAxes, ha="left", va="center", fontsize=8,
                  family="monospace", linespacing=1.6,
                  bbox={"boxstyle": "round,pad=0.6", "facecolor": "0.96", "edgecolor": "0.80", "linewidth": 0.8})
    axis.set_xlabel("estimate")
    axis.set_ylabel("truth" if ylabel else "")


def binary_summary(counts) -> str:
    """counts = [[TP, FN], [FP, TN]] -> 'F1 · precision · recall · specificity'."""
    (tp, fn), (fp, tn) = np.asarray(counts, dtype=float)
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return "\n".join(f"{name:<11s} {value:.3f}" for name, value in
                     (("F1", f1), ("precision", precision), ("recall", recall), ("specificity", specificity)))


def multiclass_summary(counts, row_labels, col_labels) -> str:
    """Accuracy and macro F1 over the true classes (columns matched to rows by label)."""
    counts = np.asarray(counts, dtype=float)
    hits, f1 = 0.0, []
    for i, label in enumerate(row_labels):
        j = list(col_labels).index(label) if label in col_labels else None
        tp = counts[i, j] if j is not None else 0.0
        hits += tp
        precision = tp / counts[:, j].sum() if j is not None and counts[:, j].sum() else 0.0
        recall = tp / counts[i].sum() if counts[i].sum() else 0.0
        f1.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return f"{'accuracy':<8s} {hits / max(counts.sum(), 1):.3f}\n{'macro F1':<8s} {np.mean(f1):.3f}"


def parameter_scatter(axis, true_alpha, fit_alpha, true_delta, fit_delta, *, title: Optional[str] = None) -> None:
    """Recovered vs true microstate parameters (alpha, delta) of the items whose (c, d) was recovered."""
    from mirm.run.utils.viz import INK, SKY

    values = []
    for (truth, fit, color, marker, name) in ((true_alpha, fit_alpha, INK, "o", "α"),
                                               (true_delta, fit_delta, SKY, "s", "δ")):
        truth, fit = np.asarray(truth, dtype=float), np.asarray(fit, dtype=float)
        keep = np.isfinite(truth) & np.isfinite(fit)
        truth, fit = truth[keep], fit[keep]
        if not truth.size:
            continue
        rmse = float(np.sqrt(np.mean((fit - truth) ** 2)))
        r = float(np.corrcoef(truth, fit)[0, 1]) if truth.size > 1 else float("nan")
        axis.scatter(truth, fit, s=9, alpha=0.45, color=color, marker=marker, linewidths=0,
                     label=f"{name}: RMSE {rmse:.3f}, r {r:.3f} (n={truth.size})")
        values += [truth, fit]
    if values:
        low, high = float(min(v.min() for v in values)), float(max(v.max() for v in values))
        pad = 0.05 * (high - low + 1e-12)
        axis.plot([low - pad, high + pad], [low - pad, high + pad], color="0.6", linewidth=0.8, zorder=0)
        axis.set_xlim(low - pad, high + pad)
        axis.set_ylim(low - pad, high + pad)
    axis.set_xlabel("true")
    axis.set_ylabel("estimate")
    if title is not None:
        axis.set_title(title)
    axis.legend(fontsize=6.5, loc="upper left")
    axis.grid(alpha=0.15)


def monotonicity_figure(matrices, factors, lines, path, *, reference: Optional[dict] = None,
                        title: Optional[str] = None, errors: bool = True):
    """Monotonicity study in one row: confusion matrices (rows truth, columns fit; monotone first; each cell the
    per-fit count mean +- sd over the fits and, in parentheses, the row percentage) next to a 2 x len(factors) grid of
    main-effect lines (false non-monotone rate = fit dip | true monotone, false monotone rate = fit monotone | true
    dip; mean +- sd over the fits at each level of one factor, averaged over the others).

    matrices: {arm: per-fit counts, array (fits, 2, 2) in the order [[mono->mono, mono->dip], [dip->mono, dip->dip]]};
    factors: {factor: [levels]}; lines: {arm: {metric: {factor: (means, sds)}}} with metric "false non-monotone" or
    "false monotone"; reference: {name: {metric: value}} drawn as flat lines over every factor (e.g. 2PL: 0 and 1).
    errors=False draws the lines without error bars and without the horizontal offsets between arms."""
    import matplotlib.pyplot as plt
    from mirm.run.utils.viz import DPI, INK, SKY, GREY, DEEP, ACCENT

    arms = list(matrices)
    styles = [(INK, "--", "s"), (SKY, "-", "o"), (DEEP, "-.", "D"), (ACCENT, (0, (5, 1, 1, 1)), "v")]
    style = {arm: styles[k % len(styles)] for k, arm in enumerate(arms)}
    metrics = [("false non-monotone", "false non-monotone rate\n(fit: dip | true: monotone)"),
               ("false monotone", "false monotone rate\n(fit: monotone | true: dip)")]
    n_factors = len(factors)

    width = 2.6 * len(arms) + 2.4 * n_factors
    fig = plt.figure(figsize=(width, 5.4))
    outer = fig.add_gridspec(1, 2, width_ratios=[len(arms), n_factors], wspace=0.16,
                             left=1.0 / width, right=0.99, top=0.86, bottom=0.14)   # 1 inch for the row labels
    left = outer[0, 0].subgridspec(1, len(arms), wspace=0.12)
    labels = ["monotone", "dip"]
    for k, arm in enumerate(arms):
        axis = fig.add_subplot(left[0, k])
        per_fit = np.asarray(matrices[arm], dtype=float)
        total = per_fit.sum(axis=0)
        rates = total / np.maximum(total.sum(axis=1, keepdims=True), 1)
        mean = per_fit.mean(axis=0)
        sd = per_fit.std(axis=0, ddof=1) if len(per_fit) > 1 else np.zeros((2, 2))
        text = np.array([[f"{mean[i, j]:.1f} ± {sd[i, j]:.1f}\n({100 * rates[i, j]:.1f}%)" for j in range(2)]
                         for i in range(2)], dtype=object)
        heatmap(axis, rates, xticks=labels, yticks=labels if k == 0 else ["", ""], text=text, cmap="Blues",
                origin="upper", colorbar=False)
        axis.set_xticks([0, 1], labels, rotation=0, ha="center")
        for t in axis.texts:
            t.set_fontsize(9)
            t.set_linespacing(1.5)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"{arm}\n(n = {int(total.sum()):,} item fits)", fontsize=10)
        axis.set_xlabel("fit")
        axis.set_ylabel("truth" if k == 0 else "")

    right = outer[0, 1].subgridspec(2, n_factors, hspace=0.45, wspace=0.12)
    offsets = {arm: (k - (len(arms) - 1) / 2) * 0.12 if errors else 0.0 for k, arm in enumerate(arms)}
    for row, (key, ylabel) in enumerate(metrics):
        first = None
        for col, (factor, levels) in enumerate(factors.items()):
            axis = fig.add_subplot(right[row, col], sharey=first)
            first = first or axis
            x = np.arange(len(levels))
            for arm in arms:
                means, sds = lines[arm][key][factor]
                color, linestyle, marker = style[arm]
                if errors:
                    axis.errorbar(x + offsets[arm], means, yerr=sds, color=color, linestyle=linestyle, marker=marker,
                                  markersize=4, linewidth=1.2, capsize=2, elinewidth=0.7, capthick=0.7, label=arm)
                else:
                    axis.plot(x, means, color=color, linestyle=linestyle, marker=marker, markersize=4,
                              linewidth=1.2, label=arm)
            for name, values in (reference or {}).items():
                axis.plot(x, np.full(len(levels), values[key]), color=GREY, linestyle=":", marker="^", markersize=4,
                          linewidth=1.2, label=name)
            axis.set_xticks(x, [str(v) for v in levels])
            axis.set_xlim(-0.4, len(levels) - 0.6)
            axis.grid(axis="y", alpha=0.15)
            if row == len(metrics) - 1:
                axis.set_xlabel(factor)
            if col == 0:
                axis.set_ylabel(ylabel, fontsize=8)
            else:
                axis.tick_params(labelleft=False)
            if row == 0 and col == 0:
                axis.legend(fontsize=7)
        first.set_ylim(0, 1.05)
    if title is not None:
        fig.text(0.01, 0.96, title, fontsize=11, fontweight="bold", ha="left", va="top")
    path = __import__("pathlib").Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path
