from typing import Optional

import numpy as np


def theta_cmaps() -> dict:
    """theta color families of the PRM ablation: PRM in the sky-blue family, no PRM in the red family."""
    from matplotlib.colors import LinearSegmentedColormap
    from mirm.run.utils.viz import SKY, DEEP, RED
    return {
        "PRM": LinearSegmentedColormap.from_list("theta_prm", ["#d3e7f5", SKY, DEEP, "#0f3552"]),
        "no PRM": LinearSegmentedColormap.from_list("theta_noprm", ["#f7d3d6", RED, "#8c1d26", "#4a0d14"]),
    }


def clean_3d(axis) -> None:
    """White panes and a light grid so that a faint grey cloud stays visible; paths drawn above the cloud."""
    axis.computed_zorder = False
    for ax in (axis.xaxis, axis.yaxis, axis.zaxis):
        ax.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.pane.set_edgecolor("0.8")
        ax._axinfo["grid"].update({"color": (0.78, 0.78, 0.78, 1.0), "linewidth": 0.5})


def cloud3d(axis, points, *, max_points: int = 20000, size: float = 2.2, color: str = "0.35", seed: int = 0):
    """Faint cloud that looks the same for 10^4 or 10^6 points: at most max_points are drawn (random subsample with a
    fixed seed) and the opacity falls with the number drawn; rasterized so that vector outputs stay small."""
    points = np.asarray(points, dtype=float)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) > max_points:
        points = points[np.random.default_rng(seed).choice(len(points), max_points, replace=False)]
    if len(points):
        alpha = float(np.clip(1000.0 / len(points), 0.01, 0.3))
        axis.scatter(*points.T, s=size, color=color, alpha=alpha, linewidths=0, depthshade=False, rasterized=True,
                     zorder=1)
    return points


def path3d_gradient(axis, X, t, cmap, *, label: Optional[str] = None, t_range: Optional[tuple] = None,
                    ticks: int = 9, linewidth: float = 2.8, arrow: float = 0.5):
    """3D path colored by t (light -> dark), interior ticks, a dot at the first point and an arrowhead at the last
    point. The arrowhead has no shaft of its own: its base is the point of the path one head length (arrow, in data
    units of arc length) before the end, so the head follows the path's final chord even where the path turns sharply.
    Returns the finite points drawn."""
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    X = np.asarray(X, dtype=float)
    t = np.asarray(t, dtype=float)
    keep = np.isfinite(X).all(axis=1)
    pts, tt = X[keep], t[keep]
    if len(pts) < 2:
        return pts
    norm = Normalize(*(t_range or (np.nanmin(t), np.nanmax(t))))

    segments = np.stack([pts[:-1], pts[1:]], axis=1)
    axis.add_collection3d(Line3DCollection(segments, colors=cmap(norm((tt[:-1] + tt[1:]) / 2)),
                                           linewidths=linewidth, zorder=3))
    marks = np.linspace(0, len(pts) - 1, ticks, dtype=int)[1:-1]
    axis.scatter(*pts[marks].T, color=cmap(norm(tt[marks])), s=22, depthshade=False, zorder=4)
    axis.scatter(*pts[0], color=cmap(norm(tt[0])), marker="o", s=70, depthshade=False, edgecolor="0.3",
                 linewidth=0.5, zorder=4)

    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    base = np.array([np.interp(max(arc[-1] - arrow, 0.0), arc, pts[:, k]) for k in range(3)])
    axis.quiver(*base, *(pts[-1] - base), color=cmap(norm(tt[-1])), arrow_length_ratio=1.0,
                linewidth=linewidth, zorder=4)
    if label is not None:
        axis.plot([], [], color=cmap(0.75), linewidth=linewidth, label=label)
    return pts


def theta_iccs(axis, theta, P, cmap, *, title: Optional[str] = None, mean: bool = True, curve=None,
               curve_label: str = "mean", certified=None, uncertified_label: str = "uncertified direction",
               clip: bool = True) -> None:
    """ICCs colored along theta with the path colormap (the theta color key of the 3D panel). The black curve is the
    mean ICC, or the given curve (e.g. the scaled generalized-score curve), solid where certified (default: all) and
    dashed elsewhere. The y-range is [-0.02, 1.02] and a curve leaving [0, 1] is cut at the frame (clip=False: the
    y-range widens to the curve instead)."""
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    from mirm.run.utils.viz import INK

    theta = np.asarray(theta, dtype=float)
    P = np.atleast_2d(np.asarray(P, dtype=float))
    colors = cmap(Normalize(theta.min(), theta.max())((theta[:-1] + theta[1:]) / 2))
    for y in P:
        segments = np.stack([np.column_stack([theta[:-1], y[:-1]]), np.column_stack([theta[1:], y[1:]])], axis=1)
        axis.add_collection(LineCollection(segments, colors=colors, linewidths=0.7, alpha=0.22))
    low, high = -0.02, 1.02
    if curve is not None:
        curve = np.asarray(curve, dtype=float)
        good = np.ones_like(curve, dtype=bool) if certified is None else np.asarray(certified, dtype=bool)
        solid = np.where(good, curve, np.nan)
        dashed = np.where(~good | np.r_[~good[1:], False] | np.r_[False, ~good[:-1]], curve, np.nan)  # joins the parts
        axis.plot(theta, solid, color=INK, linewidth=2.0, label=curve_label)
        if (~good).any():
            axis.plot(theta, dashed, color=INK, linewidth=1.6, linestyle="--", label=uncertified_label)
        if not clip and np.isfinite(curve).any():
            low, high = min(low, np.nanmin(curve) - 0.02), max(high, np.nanmax(curve) + 0.02)
        axis.legend(fontsize=7, loc="lower right")
    elif mean:
        axis.plot(theta, P.mean(axis=0), color=INK, linewidth=2.0, label="mean")
        axis.legend(fontsize=7, loc="lower right")
    axis.set_xlim(theta[0], theta[-1])
    axis.set_ylim(low, high)
    if title is not None:
        axis.set_title(title)
    axis.set_xlabel("θ")
    axis.set_ylabel("P(correct)")
    axis.grid(alpha=0.15)


def first_near_crossing(X: np.ndarray, Y: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """Index on path X where it first comes back closest to path Y after leaving their common start, with the common
    start and the near-crossing point (midpoints of the two paths)."""
    ok_x, ok_y = np.where(np.isfinite(X).all(1))[0], np.where(np.isfinite(Y).all(1))[0]
    D = np.linalg.norm(X[ok_x, None, :] - Y[None, ok_y, :], axis=2)
    d, j = D.min(1), D.argmin(1)
    away = np.where(d > max(3 * d[0], 0.15))[0]
    if not len(away):
        r = int(np.argmin(d[1:]) + 1)
    else:
        r0 = away[0]
        mins = [r for r in range(r0 + 1, len(d) - 1) if d[r] <= d[r - 1] and d[r] <= d[r + 1]]
        r = mins[0] if mins else int(r0 + np.argmin(d[r0:]))
    start = (X[ok_x[0]] + Y[ok_y[0]]) / 2
    cross = (X[ok_x[r]] + Y[ok_y[j[r]]]) / 2
    return int(ok_x[r]), start, cross


def _figure_point(fig, axis3d, point) -> tuple[float, float]:
    from mpl_toolkits.mplot3d import proj3d
    fig.canvas.draw()
    x2, y2, _ = proj3d.proj_transform(*point, axis3d.get_proj())
    fx, fy = fig.transFigure.inverted().transform(axis3d.transData.transform((x2, y2)))
    return float(fx), float(fy)


def inset_beside(fig, axis3d, point, *, radius: float, circle: float, side: str = "below",
                 pad: float = 0.012) -> tuple[tuple[float, float], str]:
    """Centre (figure coordinates) of a circular inset right next to the dashed circle around point: on the given side
    (below | above | right | left), or the first of these that fits in the figure (side="auto")."""
    W, H = fig.get_size_inches()
    fx, fy = _figure_point(fig, axis3d, point)
    rx, cxw = radius * H / W, circle * H / W
    dy, dx = radius + circle + pad, rx + cxw + pad * H / W
    candidates = {"below": (fx, fy - dy), "above": (fx, fy + dy), "right": (fx + dx, fy), "left": (fx - dx, fy)}
    fits = lambda c: rx <= c[0] <= 1 - rx and radius + 0.03 <= c[1] <= 1 - radius
    order = [side] if side != "auto" else ["below", "above", "right", "left"]
    side = next((s for s in order if fits(candidates[s])), order[0])
    cx = float(np.clip(candidates[side][0], rx, 1 - rx))
    cy = float(np.clip(candidates[side][1], radius + 0.03, 1 - radius))
    return (cx, cy), side


def zoom_inset(fig, axis3d, paths: dict, theta, students, cmaps, *, focus, index: int, fold_key: str = "no PRM",
               half: float = 1.1, circle: float = 0.055, radius: float = 0.17, center: tuple = (0.14, 0.70),
               max_points: int = 6000, note: str = "viewed from above its plane"):
    """Circle focus (a point in domain coordinates) on the 3D panel and show it in a circular inset, rendered
    off-screen in an equal-aspect cube of half-width half around it and viewed along the normal of the plane of
    paths[fold_key] near it (so that the branches open as widely as possible). Call after the main layout is final;
    save the figure without tight_layout. Returns (theta[index], elev, azim)."""
    import io
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Ellipse

    theta = np.asarray(theta, dtype=float)
    X = np.asarray(paths[fold_key], dtype=float)
    fold = np.asarray(focus, dtype=float)
    near = np.isfinite(X).all(axis=1) & (np.linalg.norm(X - fold, axis=1) < half * 1.1)
    pts = X[near] - fold
    normal = np.linalg.svd(pts - pts.mean(0), full_matrices=False)[2][-1] if len(pts) >= 3 else np.array([0, 0, 1.0])
    if normal[2] < 0:
        normal = -normal
    elev = float(np.degrees(np.arcsin(np.clip(normal[2], -1, 1))))
    azim = float(np.degrees(np.arctan2(normal[1], normal[0])))

    # inset image: zoomed cube, equal aspect, top-down onto the local plane
    students = np.asarray(students, dtype=float)
    box = np.all(np.abs(students - fold) < half, axis=1) if len(students) else np.zeros(0, dtype=bool)
    off = plt.figure(figsize=(4.4, 4.4))
    ai = off.add_subplot(111, projection="3d")
    clean_3d(ai)
    cloud3d(ai, students[box] if len(students) else students, max_points=max_points)
    for name in paths:
        path3d_gradient(ai, paths[name], theta, cmaps[name], t_range=(theta[0], theta[-1]), linewidth=3.2,
                        arrow=0.14, ticks=25)
    for setter, c in zip((ai.set_xlim, ai.set_ylim, ai.set_zlim), fold):
        setter(c - half, c + half)
    ai.set_box_aspect((1, 1, 1))
    ai.view_init(elev=elev, azim=azim)
    ai.set_axis_off()
    off.subplots_adjust(0, 0, 1, 1)
    buffer = io.BytesIO()
    off.savefig(buffer, dpi=200, format="png")
    plt.close(off)
    buffer.seek(0)
    image = plt.imread(buffer)

    # circle on the main panel around the projected focus
    W, H = fig.get_size_inches()
    fx, fy = _figure_point(fig, axis3d, fold)
    fig.add_artist(Ellipse((fx, fy), 2 * circle * H / W, 2 * circle, transform=fig.transFigure, fill=False,
                           edgecolor="0.15", linewidth=1.3, linestyle="--", zorder=10))

    # circular inset, linked to the circle
    cx, cy = center
    inset = fig.add_axes([cx - radius * H / W, cy - radius, 2 * radius * H / W, 2 * radius])
    shown = inset.imshow(image)
    shown.set_clip_path(Circle((0.5, 0.5), 0.5, transform=inset.transAxes))
    inset.add_patch(Circle((0.5, 0.5), 0.497, transform=inset.transAxes, fill=False, edgecolor="0.15",
                           linewidth=1.3))
    inset.set_axis_off()
    inset.text(0.5, -0.04, note, transform=inset.transAxes, ha="center", va="top", fontsize=8)
    v = np.array([fx - cx, fy - cy]) * np.array([W, H])
    v /= np.linalg.norm(v) + 1e-12
    start = (cx + radius * v[0] * H / W, cy + radius * v[1])
    end = (fx - circle * v[0] * H / W, fy - circle * v[1])
    fig.add_artist(plt.Line2D([start[0], end[0]], [start[1], end[1]], transform=fig.transFigure, color="0.15",
                              linewidth=1.0, linestyle="--", zorder=10))
    return float(theta[index]), elev, azim


def fold_inset(fig, axis3d, paths: dict, theta, students, cmaps, *, fold_key: str = "no PRM",
               circle: float = 0.055, max_points: int = 6000,
               low_half: float = 2.2, low_shift: float = 0.75, low_radius: float = 0.14,
               low_center: tuple = (0.14, 0.70),
               high: bool = True, high_half: float = 1.1, high_radius: float = 0.10, high_circle: float = 0.055,
               high_side: str = "below", high_center: Optional[tuple] = None) -> dict:
    """Zoom insets of the 3D panel of the PRM ablation (call after the main layout is final; save without
    tight_layout).

    low:  a fixed zoom of half-width low_half, centred between the common start of the two paths and the first
          near-crossing of paths[fold_key] with the other path (first_near_crossing), at most low_shift * low_half
          from the start so that the start stays well inside; the inset sits at low_center.
    high: the turn of paths[fold_key] at high theta (its highest mean domain coordinate), zoom half-width high_half,
          the inset right next to its dashed circle (inset_beside on high_side; high_center = (x, y) overrides).
    Returns {"low": {...}, "high": {...}} with theta, elev, azim and placement details."""
    theta = np.asarray(theta, dtype=float)
    X = np.asarray(paths[fold_key], dtype=float)
    Y = np.asarray(paths["PRM" if fold_key != "PRM" else "no PRM"], dtype=float)
    shown = {}
    if not np.isfinite(X).all(1).any() or not np.isfinite(Y).all(1).any():
        return shown

    k, start, cross = first_near_crossing(X, Y)
    step = (cross - start) / 2
    reach = low_shift * low_half
    if np.linalg.norm(step) > reach:
        step = step * reach / np.linalg.norm(step)
    t, elev, azim = zoom_inset(fig, axis3d, paths, theta, students, cmaps, focus=start + step, index=k,
                               fold_key=fold_key, half=low_half, circle=circle, radius=low_radius, center=low_center,
                               max_points=max_points, note="low-θ fold, viewed from above its plane")
    shown["low"] = {"theta": t, "elev": elev, "azim": azim, "half": low_half,
                    "start_to_crossing": float(np.linalg.norm(cross - start)), "shift": float(np.linalg.norm(step))}

    if high:
        level = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=1)
        kh = int(np.nanargmax(level))
        if high_center is None:
            center, side = inset_beside(fig, axis3d, X[kh], radius=high_radius, circle=high_circle, side=high_side)
        else:
            center, side = tuple(high_center), "manual"
        t, elev, azim = zoom_inset(fig, axis3d, paths, theta, students, cmaps, focus=X[kh], index=kh,
                                   fold_key=fold_key, half=high_half, circle=high_circle, radius=high_radius,
                                   center=center, max_points=max_points,
                                   note="high-θ turn\nviewed from above its plane")
        shown["high"] = {"theta": t, "elev": elev, "azim": azim, "half": high_half, "side": side, "center": center}
    return shown
