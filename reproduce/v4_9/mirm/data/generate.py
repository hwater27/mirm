from typing import Optional

import numpy as np
import torch

from mirm.data.datasets import ScoredIRTDataset
from mirm.model import build_model, MODEL_NAMES, BinaryIRTModel, BinaryMiRM, PRM


def _recipe(model: BinaryIRTModel, rng, **kwargs) -> BinaryIRTModel:
    name = model.name
    n_items = model.n_items

    def _inv_softplus(x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.expm1(x.clamp_min(1e-12)))

    with torch.no_grad():
        if name == "2PL":
            a = _modes(rng, _SLOPES, n_items)
            b = _locations(rng, n_items)
            model.a_raw.copy_(_inv_softplus(torch.as_tensor(a, dtype=torch.float64)))
            model.b.copy_(torch.as_tensor(b, dtype=torch.float64))

        elif name == "3PL":
            # Moderately steep slopes and a low guessing range, so that the guessing floor does not cost much
            # ability precision while the ICCs stay ordinary.
            a = 1.2 * _modes(rng, _SLOPES, n_items)                # a little steeper: the floor costs information
            b = _locations(rng, n_items)
            c = _modes(rng, _GUESSING, n_items)
            model.a_raw.copy_(_inv_softplus(torch.as_tensor(a, dtype=torch.float64)))
            model.b.copy_(torch.as_tensor(b, dtype=torch.float64))
            model.c_raw.copy_(torch.logit(torch.as_tensor(c, dtype=torch.float64)))

        elif name == "LPE":
            a = _modes(rng, _SLOPES, n_items)
            b = _locations(rng, n_items)
            xi = _modes(rng, _EXPONENTS, n_items)
            model.a_raw.copy_(_inv_softplus(torch.as_tensor(a, dtype=torch.float64)))
            model.b.copy_(torch.as_tensor(b, dtype=torch.float64))
            model.xi_raw.copy_(_inv_softplus(torch.as_tensor(xi, dtype=torch.float64)))

        elif name == "Spline":
            grid = np.linspace(model.lower, model.upper, 801)
            basis = lambda x: model._ispline_basis_numpy(np.atleast_1d(x), model.n_basis, model.degree, model.lower,
                                                         model.upper)
            intercept, gamma = _spline_items(rng, n_items, grid, basis)
            model.intercept.copy_(torch.as_tensor(intercept, dtype=torch.float64))
            model.gamma_raw.copy_(_inv_softplus(torch.as_tensor(gamma, dtype=torch.float64)))

        elif name == "MiRM":
            alpha = np.zeros((n_items, model.n_states))
            delta = np.zeros((n_items, model.n_states))
            for j in range(n_items):
                c, d = int(model.c[j]), int(model.d[j])
                a_j, d_j = _mirm_item(rng, c, d)
                alpha[j, :c], delta[j, :c] = a_j[:c], d_j[:c]
                alpha[j, model.c_max:model.c_max + d], delta[j, model.c_max:model.c_max + d] = a_j[c:], d_j[c:]
            model.alpha_raw.copy_(torch.as_tensor(alpha, dtype=torch.float64))
            model.delta_raw.copy_(torch.as_tensor(delta, dtype=torch.float64))

    return model


def _locations(rng, size=None):
    """Item locations (P = 0.5 or the 2PL location): U(-2, 2), spread so that every ability region is measured."""
    return rng.uniform(-2.0, 2.0, size=size)


def _modes(rng, components, size=None):
    """Draw from a mixture of shape regimes: components = [(weight, sampler(rng, n)), ...]."""
    n = 1 if size is None else size
    weights = np.asarray([w for w, _ in components], dtype=np.float64)
    pick = rng.choice(len(components), size=n, p=weights / weights.sum())
    out = np.empty(n)
    for k, (_, sampler) in enumerate(components):
        idx = np.flatnonzero(pick == k)
        if idx.size:
            out[idx] = sampler(rng, idx.size)
    return out[0] if size is None else out


# multimodal shape regimes shared by the parametric generators
_SLOPES = [(0.3, lambda r, n: r.lognormal(np.log(0.6), 0.15, n)),     # flat items
           (0.4, lambda r, n: r.lognormal(np.log(1.1), 0.15, n)),     # ordinary items
           (0.3, lambda r, n: r.lognormal(np.log(2.0), 0.15, n))]     # sharp items
_GUESSING = [(0.35, lambda r, n: r.uniform(0.0, 0.05, n)),            # constructed response: no floor
             (0.40, lambda r, n: r.uniform(0.12, 0.22, n)),           # four-option multiple choice
             (0.25, lambda r, n: r.uniform(0.28, 0.40, n))]           # few options / attractive guessing
_EXPONENTS = [(0.35, lambda r, n: np.exp(r.uniform(np.log(0.25), np.log(0.5), n))),   # early rise, long upper tail
              (0.30, lambda r, n: np.exp(r.uniform(np.log(0.8), np.log(1.25), n))),   # nearly symmetric
              (0.35, lambda r, n: np.exp(r.uniform(np.log(2.0), np.log(4.0), n)))]    # complex: long lower tail


_WINDOW = np.linspace(-4, 4, 801)
_WIDE = np.linspace(-8, 8, 1601)
_DIP_TOLERANCE = 0.01          # largest drop of the ICC below its running maximum on [-4, 4]


def _mirm_icc(alpha_c, delta_c, alpha_d, delta_d, grid):
    log_c = np.logaddexp.reduce(-alpha_c[:, None] * grid + delta_c[:, None], axis=0)
    log_d = np.logaddexp.reduce(-alpha_d[:, None] * grid + delta_d[:, None], axis=0)
    return 1.0 / (1.0 + np.exp(-(log_c - log_d)))


def _mirm_item(rng, c: int, d: int) -> tuple[np.ndarray, np.ndarray]:
    """Free-form MiRM item of configuration (c, d); returns alpha and delta (C states, then D states).

    Slope gaps are exponential (scale 0.5, 1 or 2) and the macrostates are interleaved at random along the slope
    order between a correct ground state (smallest alpha) and a distractor top state (largest alpha), so that P -> 1
    as theta -> +inf and P -> 0 as theta -> -inf on the whole line; consecutive states cross at points over [-3.5, 3.5] and 60% of the
    states get a N(0, 1.2) log-weight perturbation, so floors, ceilings, shoulders, plateaus and multi-step rises all
    occur. The item is then moved to a location b ~ U(-2, 2) (the theta where the curve passes the middle of its
    range) and stretched horizontally around b so that its steepest slope max dP/dtheta equals a / 4 with a from the
    2PL slope regimes (`_SLOPES`): the steepest slopes of the MiRM items follow the 2PL generator in mean and spread.
    Kept only if the item rises over [-4, 4] (P(4) - P(-4) > 0.4, range > 0.5) and any local dip is negligible
    (< 0.01): flagged non-monotone items are not used.
    """
    while True:
        gaps = rng.choice([0.5, 1.0, 2.0]) * rng.exponential(1.0, c + d - 1) + 0.05
        alpha = np.concatenate([[0.0], np.cumsum(gaps)])
        order = np.array(["C"] + list(rng.permutation(["C"] * (c - 1) + ["D"] * (d - 1))) + ["D"])
        crossings = np.sort(rng.uniform(-3.5, 3.5, c + d - 1))[::-1]
        delta = np.concatenate([[0.0], np.cumsum(gaps * crossings)])
        delta += rng.normal(0.0, 1.2, c + d) * (rng.random(c + d) < 0.6)
        is_c = order == "C"
        alpha_c, delta_c, alpha_d, delta_d = alpha[is_c], delta[is_c], alpha[~is_c], delta[~is_c]

        # location: the middle of the curve's range on a wide grid -> b
        p = _mirm_icc(alpha_c, delta_c, alpha_d, delta_d, _WIDE)
        middle = _WIDE[int(np.argmin(np.abs(p - (p.min() + p.max()) / 2)))]
        b = _locations(rng)
        delta_c, delta_d = delta_c + alpha_c * (b - middle), delta_d + alpha_d * (b - middle)

        # stretch around b: steepest slope -> a / 4 with a from the 2PL slope regimes
        p = _mirm_icc(alpha_c, delta_c, alpha_d, delta_d, _WIDE)
        steepest = np.max(np.gradient(p, _WIDE))
        if steepest <= 0:
            continue
        stretch = (_modes(rng, _SLOPES) / 4) / steepest
        delta_c, delta_d = delta_c - (1 - stretch) * alpha_c * b, delta_d - (1 - stretch) * alpha_d * b
        alpha_c, alpha_d = stretch * alpha_c, stretch * alpha_d

        p = _mirm_icc(alpha_c, delta_c, alpha_d, delta_d, _WINDOW)
        if p[-1] - p[0] > 0.4 and p.max() - p.min() > 0.5 and np.max(np.maximum.accumulate(p) - p) < _DIP_TOLERANCE:
            return np.concatenate([alpha_c, alpha_d]), np.concatenate([delta_c, delta_d])


def _spline_items(rng, n_items: int, grid: np.ndarray, basis) -> tuple[np.ndarray, np.ndarray]:
    """Monotone I-spline ICCs from a continuous family around the item location b ~ U(-2, 2): the slope is spread as
    a bump of width log-uniform on [0.4, 2.2] (narrower bumps rise further: total logit rise from about 8 down to 3,
    with noise), plus with probability 0.4 a second bump of weight U(0.2, 1.2) at a distance U(0.8, 2.2) on a random
    side (two-step rises of every degree). Random weights multiply the profile; the intercept puts P = 0.5 at b.
    Overly steep items are softened without changing their shape: if the steepest slope s exceeds 0.45, all weights
    are scaled down so that it becomes 0.45 + 0.3 (s - 0.45) (`_soften`). Returns (intercept, gamma)."""
    B = basis(grid)
    centers = grid[np.gradient(B, grid, axis=0).argmax(axis=0)]              # where each basis function rises
    intercept = np.zeros(n_items)
    gamma = np.zeros((n_items, B.shape[1]))
    bump = lambda m, sd: np.exp(-0.5 * ((centers - m) / sd) ** 2)
    for j in range(n_items):
        b = _locations(rng)
        width = np.exp(rng.uniform(np.log(0.4), np.log(2.2)))
        share = (np.log(2.2) - np.log(width)) / (np.log(2.2) - np.log(0.4))  # 1 for the narrowest, 0 for the widest
        rise = 3.0 + 5.0 * share + rng.normal(0.0, 0.7)
        profile = bump(b, width)
        if rng.random() < 0.4:
            profile = profile + rng.uniform(0.2, 1.2) * bump(b + rng.choice([-1, 1]) * rng.uniform(0.8, 2.2), min(width, 0.8))
        shares = rng.exponential(1.0, size=B.shape[1]) * profile + 1e-3
        g = max(rise, 2.0) * shares / shares.sum()
        at_b = basis(b)[0]
        g = g * _soften(lambda k: np.max(np.gradient(1 / (1 + np.exp(-(B @ (k * g) - at_b @ (k * g)))), grid)))
        gamma[j] = g
        intercept[j] = -gamma[j] @ at_b
    return intercept, gamma


_STEEP_KNEE, _STEEP_KEEP = 0.45, 0.3   # steepest slopes above the knee are pulled in: knee + keep * excess


def _soften(steepest, iterations: int = 40) -> float:
    """Scale factor k <= 1 for the basis weights so that an overly steep item keeps its shape but its steepest slope
    s becomes knee + keep * (s - knee); items at or below the knee are left as they are (k = 1)."""
    s = steepest(1.0)
    if s <= _STEEP_KNEE:
        return 1.0
    target = _STEEP_KNEE + _STEEP_KEEP * (s - _STEEP_KNEE)
    low, high = 0.05, 1.0
    for _ in range(iterations):
        mid = np.sqrt(low * high)
        low, high = (mid, high) if steepest(mid) < target else (low, mid)
    return float(np.sqrt(low * high))


def _mixture_config(rng, n_items: int, mixture: dict) -> dict:
    """Per-item (c, d) drawn in fixed proportions, e.g. {"1,1": 1/3, "2,3": 1/3, "3,1": 1/3}."""
    specs = [tuple(int(v) for v in key.split(",")) for key in mixture]
    weights = np.asarray(list(mixture.values()), dtype=np.float64)
    counts = np.floor(weights / weights.sum() * n_items).astype(int)
    counts[np.argsort(-(weights / weights.sum() * n_items - counts))[: n_items - counts.sum()]] += 1
    assigned = np.repeat(np.arange(len(specs)), counts)
    rng.shuffle(assigned)
    return {"c": [specs[k][0] for k in assigned], "d": [specs[k][1] for k in assigned]}


def _observe(student, item, credit, density, rng, n_students, n_items):
    if density >= 1.0:
        return student, item, credit
    if not 0.0 < density < 1.0:
        raise ValueError("density must be in (0, 1]")

    keep = rng.random(len(credit)) < density
    for s in range(n_students):
        idx = np.flatnonzero(student == s)
        if idx.size and not keep[idx].any():
            keep[rng.choice(idx)] = True
    for j in range(n_items):
        idx = np.flatnonzero(item == j)
        if idx.size and not keep[idx].any():
            keep[rng.choice(idx)] = True
    return student[keep], item[keep], credit[keep]


def generate(
    generator: str,
    n_students: int,
    n_items: int,
    density: float = 1.0,
    generator_config: Optional[dict] = None,
    seed: int = 27,
    n_quad: int = 41,
) -> tuple[ScoredIRTDataset, dict]:
    rng = np.random.default_rng(seed)

    if generator not in MODEL_NAMES.keys():
        raise ValueError(f"Invalid generator: {generator}")

    _DEFAULTS = {"Spline": {"n_basis": 10}, "MiRM": {"c": 2, "d": 3}}
    config = {**_DEFAULTS.get(generator, {}), **(generator_config or {})}
    if "mixture" in config:
        config = _mixture_config(rng, n_items, config.pop("mixture"))

    model = build_model(generator, n_items, n_quad, **config)
    model = _recipe(model, rng, **config)

    theta = torch.as_tensor(rng.standard_normal(n_students), dtype=torch.float64)
    p = model.p_correct_grid(theta).detach().T.numpy()
    credit = rng.random(p.shape) < p

    student, item = np.indices((n_students, n_items))
    student, item, credit = student.ravel(), item.ravel(), credit.ravel()

    student, item, credit = _observe(student, item, credit, density, rng, n_students, n_items)

    data = torch.stack(
        [
            torch.from_numpy(student.astype(np.int32)),
            torch.from_numpy(item.astype(np.int32)),
            torch.from_numpy(credit.astype(np.int32)),
        ],
        dim=1,
    )

    dataset = ScoredIRTDataset(data)

    metadata = {
        "generator": generator,
        "config": config,
        "theta_true": theta,
        "params_true": {k: v.detach().clone() for k, v in model.readable_params.items()},
        "state_dict_true": model.state_dict(),
    }

    return dataset, metadata


# ------------------------------------------------------------
# Nonmonotonic bank generator
# ------------------------------------------------------------

_NOTCH_GRID = np.linspace(-4, 4, 801)


def _set_params(model: BinaryMiRM, alpha: np.ndarray, delta: np.ndarray) -> None:
    with torch.no_grad():
        model.alpha_raw.copy_(torch.as_tensor(alpha, dtype=torch.float64))
        model.delta_raw.copy_(torch.as_tensor(delta, dtype=torch.float64))


def _dip_stats(p: np.ndarray, grid: np.ndarray = _NOTCH_GRID) -> tuple[float, float, float, int]:
    """Largest drop below the running maximum, its theta (trough), the theta of the preceding maximum (peak), and
    the number of separate decreasing stretches."""
    run = np.maximum.accumulate(p)
    drop = run - p
    k = int(drop.argmax())
    peak = int(np.flatnonzero(p[:k + 1] == run[k])[-1])
    dec = np.diff(p) < -1e-7
    n = int(np.sum(dec[1:] & ~dec[:-1]) + dec[0])
    return float(drop[k]), float(grid[k]), float(grid[peak]), n


def _notch_params(s_low, s_high, r, u, width, level):
    """MiRM (2,2) with slope order C0 < D1 < C2 < D3. Its logit is, up to the smoothing of the two log-sum-exps,
    piecewise linear: slope s_low = a3 - a2 below the distractor bend t_D, slope s_high = a1 above the correct
    bend t_C, and slope -(a2 - a1) = -r in between when t_D < t_C: a rising curve with a notch of width
    t_C - t_D centred at u. level is the logit at u. Returns alpha, delta in the C0, D1, C2, D3 order."""
    a1, a2 = s_high, s_high + r
    a3 = a2 + s_low
    alpha = np.array([0.0, a1, a2, a3])
    t_c, t_d = u + width / 2, u - width / 2
    delta = np.array([0.0, 0.0, a2 * t_c, (a3 - a1) * t_d])
    lc = np.logaddexp(delta[0], -alpha[2] * u + delta[2])
    ld = np.logaddexp(-alpha[1] * u + delta[1], -alpha[3] * u + delta[3])
    delta[[1, 3]] += (lc - ld) - level
    return alpha, delta


def _notch_icc(alpha: np.ndarray, delta: np.ndarray, grid: np.ndarray = _NOTCH_GRID) -> np.ndarray:
    lw = -np.outer(alpha, grid) + delta[:, None]
    return 1 / (1 + np.exp(-(np.logaddexp(lw[0], lw[2]) - np.logaddexp(lw[1], lw[3]))))


NOTCH_DEPTH_MIN, NOTCH_DEPTH_MAX = 0.05, 0.20   # target depth range of the generated notch (dip) items


def _notch_item(rng, depth_min: float = NOTCH_DEPTH_MIN,
                depth_max: float = NOTCH_DEPTH_MAX) -> tuple[np.ndarray, np.ndarray, float, float]:
    """A dip item: an ordinary rising ICC (lower and upper slopes from the 2PL slope regimes) with one notch.
    Notch centre u ~ U(-3.0, 1.6), logit at the notch U(-1.5, 2) (P about 0.18-0.88), reversal slope U(0.6, 3);
    a target depth ~ U(depth_min, depth_max) (default U(0.05, 0.2)) is drawn and the notch width is set by bisection
    so that the dip depth equals it.
    Kept when there is a single dip, its trough lies in [-2.5, 2.5], P(-4) <= 0.5, P(4) >= 0.6 and the dip is at
    most 2 wide. Returns alpha, delta in the MiRM (2,2) layout (C0, C2, D1, D3), the depth and the trough theta."""
    while True:
        s_low, s_high = _modes(rng, _SLOPES), _modes(rng, _SLOPES)
        r = rng.uniform(0.6, 3.0)
        u = rng.uniform(-3.0, 1.6)
        level = rng.uniform(-1.5, 2.0)
        target = rng.uniform(depth_min, depth_max)
        if _dip_stats(_notch_icc(*_notch_params(s_low, s_high, r, u, 4.0, level)))[0] < target:
            continue
        lo, hi = 0.0, 4.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if _dip_stats(_notch_icc(*_notch_params(s_low, s_high, r, u, mid, level)))[0] < target:
                lo = mid
            else:
                hi = mid
        alpha, delta = _notch_params(s_low, s_high, r, u, hi, level)
        p = _notch_icc(alpha, delta)
        depth, trough, peak, n = _dip_stats(p)
        if n == 1 and -2.5 <= trough <= 2.5 and peak > -3.9 and p[0] <= 0.5 and p[-1] >= 0.6 and trough - peak <= 2.0:
            order = [0, 2, 1, 3]
            return alpha[order], delta[order], depth, trough


def _layout(alpha_c, delta_c, alpha_d, delta_d, c_max: int, d_max: int) -> tuple[np.ndarray, np.ndarray]:
    """One item's states in the BinaryMiRM layout: c correct states, padding to c_max, d distractor states."""
    alpha, delta = np.zeros(c_max + d_max), np.zeros(c_max + d_max)
    alpha[:len(alpha_c)], delta[:len(delta_c)] = alpha_c, delta_c
    alpha[c_max:c_max + len(alpha_d)], delta[c_max:c_max + len(delta_d)] = alpha_d, delta_d
    return alpha, delta


def _feasible(model: BinaryMiRM, theta: torch.Tensor) -> bool:
    margins = PRM(model.n_items).margins(model, theta)
    return bool(margins["c_margin"].min() >= -1e-10 and margins["d_margin"].min() >= -1e-10)


def _dip_bank(rng, model: BinaryMiRM, theta: torch.Tensor, n_dips: int,
              depth_min: float = NOTCH_DEPTH_MIN) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MiRM (2,2) bank: n_dips notch items (C D C D, depth at least depth_min) and free-form monotone (2,2) items
    (_mirm_item), redrawn until the full bank satisfies PRM on theta."""
    for _ in range(500):
        flags = np.zeros(model.n_items, dtype=bool)
        flags[rng.choice(model.n_items, size=n_dips, replace=False)] = True
        alpha = np.zeros((model.n_items, model.n_states)); delta = np.zeros_like(alpha)
        depths, troughs = np.zeros(model.n_items), np.full(model.n_items, np.nan)
        for j, flag in enumerate(flags):
            if flag:
                a, d_, depths[j], troughs[j] = _notch_item(rng, depth_min)
            else:
                a, d_ = _mirm_item(rng, 2, 2)
                depths[j] = _dip_stats(_mirm_icc(a[:2], d_[:2], a[2:], d_[2:], _NOTCH_GRID))[0]
            alpha[j], delta[j] = _layout(a[:2], d_[:2], a[2:], d_[2:], model.c_max, model.d_max)
        _set_params(model, alpha, delta)
        if _feasible(model, theta):
            return flags, depths, troughs
    raise RuntimeError("Could not generate a PRM-feasible bank")


def _dip_population(rng, model: BinaryMiRM, n_students: int, density: float, theta_dim: int, treat_spread: float,
                    class_fracs, class_spread: float, loading_frac: float, loading_scale: float):
    """Students of the monotonicity study: a latent class mixture on ability and on theta_dim - 1 nuisance traits that
    load on some items (multidimensional deviations the unidimensional fit has to absorb)."""
    n_items = model.n_items
    class_fracs = np.asarray(class_fracs, dtype=np.float64)
    class_fracs /= class_fracs.sum()
    n_classes = len(class_fracs)
    class_ability_mean = rng.normal(0, class_spread, size=n_classes)
    class_ability_mean -= class_fracs @ class_ability_mean
    class_treat_mean = rng.normal(0, treat_spread, size=(n_classes, theta_dim - 1))
    class_treat_mean -= class_fracs @ class_treat_mean
    classes = rng.choice(n_classes, size=n_students, p=class_fracs)
    ability = class_ability_mean[classes] + rng.standard_normal(n_students)
    ability = (ability - ability.mean()) / ability.std()
    treats = class_treat_mean[classes] + rng.standard_normal((n_students, theta_dim - 1))
    loadings = np.zeros((n_items, theta_dim - 1))
    loaded = np.flatnonzero(rng.random(n_items) < loading_frac)
    loadings[loaded, rng.integers(theta_dim - 1, size=loaded.size)] = rng.normal(0, loading_scale, size=loaded.size)
    theta_eff = ability[:, None] + treats @ loadings.T

    student, item = np.indices((n_students, n_items))
    student, item = student.ravel(), item.ravel()
    with torch.no_grad():
        p = model.p_correct_indexed(torch.from_numpy(item), torch.as_tensor(theta_eff.ravel(), dtype=torch.float64)).numpy()
    credit = rng.random(p.shape) < p
    student, item, credit = _observe(student, item, credit, density, rng, n_students, n_items)
    data = torch.stack([torch.from_numpy(student.astype(np.int32)), torch.from_numpy(item.astype(np.int32)),
                        torch.from_numpy(credit.astype(np.int32))], dim=1)
    info = {"theta_true": torch.as_tensor(ability, dtype=torch.float64), "classes": classes, "class_fracs": class_fracs,
            "class_ability_mean": class_ability_mean, "class_treat_mean": class_treat_mean, "treats": treats,
            "loadings": loadings}
    return ScoredIRTDataset(data), info


def generate_dip(
    n_students: int,
    n_items: int,
    density: float = 0.5,
    true_dip_frac: float = 0.1,
    min_dip_depth: float = NOTCH_DEPTH_MIN,
    theta_dim: int = 3,
    treat_spread: float = 1,
    class_fracs: list[float] = [0.1, 0.3, 0.6],
    class_spread: float = 0.75,
    loading_frac: float = 0.25,
    loading_scale: float = 0.5,
    seed: int = 27,
    n_quad: int = 41,
) -> tuple[ScoredIRTDataset, dict]:
    """Monotonicity-study data: a PRM-feasible MiRM (2,2) bank with true_dip_frac notch items whose depths are drawn
    from U(min_dip_depth, 0.2)."""
    rng = np.random.default_rng(seed)
    config = {"c": [2] * n_items, "d": [2] * n_items}
    model = build_model("MiRM", n_items, n_quad, **config)
    theta_dense = torch.linspace(float(model.theta_q.min()), float(model.theta_q.max()), 1601, dtype=torch.float64)
    n_dips = int(round(true_dip_frac * n_items))
    dip_flags, dip_depths, dip_theta = _dip_bank(rng, model, theta_dense, n_dips, min_dip_depth)
    dataset, info = _dip_population(rng, model, n_students, density, theta_dim, treat_spread, class_fracs,
                                    class_spread, loading_frac, loading_scale)
    metadata = {"generator": "dip", "config": config,
                "params_true": {k: v.detach().clone() for k, v in model.readable_params.items()},
                "state_dict_true": model.state_dict(), "dip_flags": dip_flags, "dip_depths": dip_depths,
                "dip_theta": dip_theta, **info}
    return dataset, metadata


# ------------------------------------------------------------
# Items taken from fitted empirical models ('from' profile)
# ------------------------------------------------------------

def _bank_model(bank: dict, idx, n_quad: int) -> tuple[BinaryMiRM, dict]:
    """A per-item MiRM holding the bank items idx (their own configurations and parameters)."""
    c = [int(bank["c"][i]) for i in idx]
    d = [int(bank["d"][i]) for i in idx]
    config = {"c": c, "d": d}
    model = build_model("MiRM", len(idx), n_quad, **config)
    alpha = np.zeros((len(idx), model.n_states)); delta = np.zeros_like(alpha)
    for k, i in enumerate(idx):
        alpha[k], delta[k] = _layout(bank["alpha_c"][i], bank["delta_c"][i], bank["alpha_d"][i], bank["delta_d"][i],
                                     model.c_max, model.d_max)
    _set_params(model, alpha, delta)
    return model, config


def _sample_bank_model(rng, model, n_students: int, n_items: int, density: float) -> tuple[ScoredIRTDataset, torch.Tensor]:
    """Responses of n_students with theta ~ N(0, 1) to the items of model (the 'from' generators)."""
    theta = torch.as_tensor(rng.standard_normal(n_students), dtype=torch.float64)
    with torch.no_grad():
        p = model.p_correct_grid(theta).T.numpy()
    credit = rng.random(p.shape) < p
    student, item = np.indices((n_students, n_items))
    student, item, credit = _observe(student.ravel(), item.ravel(), credit.ravel(), density, rng, n_students, n_items)
    data = torch.stack([torch.from_numpy(student.astype(np.int32)), torch.from_numpy(item.astype(np.int32)),
                        torch.from_numpy(credit.astype(np.int32))], dim=1)
    return ScoredIRTDataset(data), theta


def _pick(rng, pool: np.ndarray, n: int) -> np.ndarray:
    """n bank indices from pool: without replacement while the pool lasts, then with replacement."""
    if n <= len(pool):
        return rng.choice(pool, size=n, replace=False)
    return np.concatenate([rng.permutation(pool), rng.choice(pool, size=n - len(pool), replace=True)])


def generate_from(
    bank: dict,
    kind: str,
    n_students: int,
    n_items: int,
    density: float = 1.0,
    seed: int = 27,
    n_quad: int = 41,
    dip_options: Optional[dict] = None,
) -> tuple[ScoredIRTDataset, dict]:
    """Synthetic data whose items are fitted empirical items (bank from mirm.run.utils.from_bank).

    kind '2PL' / '3PL' / 'LPE' / 'Spline': n_items items of the full-data fits of that model in the bank's runs
    (bank["models"][kind]), drawn at random, theta ~ N(0, 1); 'mono': n_items monotone per-item MiRM bank items drawn
    at random; 'mix' (recovery): every monotone (1,1) and (2,1) bank item plus monotone (1,2) and (2,2) items in
    proportion to their pools up to n_items (the same composition in every replicate; fewer items if the bank runs short); 'dip': the monotonicity-study design (dip_options: true_dip_frac, theta_dim,
    treat_spread, class_fracs, class_spread, loading_frac, loading_scale) with the bank's nonmonotone items as the
    dip items, topped up with generated notch items (depth from the bank's dip_min, i.e. --from-dip-min, to 0.2) when
    the bank has too few, and monotone bank items for the rest.
    The bank is drawn once and repaired until PRM holds on the quadrature range (raw margins >= -1e-10, as _feasible):
    the item that lowers the most negative margin the most is replaced by an unused item of the same kind (another
    monotone bank item; another bank dip item, or a generated notch item when none is left), one at a time; the
    replacements are listed in metadata["replaced"]. All items keep their own (c, d)."""
    rng = np.random.default_rng(seed)
    if kind in bank.get("models", {}):
        source = bank["models"][kind]
        idx = _pick(rng, np.arange(len(source["keys"])), n_items)
        config = dict(source["config"])
        model = build_model(kind, n_items, n_quad, **config)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in source["params"]:
                    param.copy_(source["params"][name][torch.as_tensor(idx)].to(param.dtype))
        dataset, theta = _sample_bank_model(rng, model, n_students, n_items, density)
        metadata = {"generator": kind, "source": f"from:{kind}", "config": config, "theta_true": theta,
                    "bank_items": [source["keys"][i] for i in idx],
                    "params_true": {k: v.detach().clone() for k, v in model.readable_params.items()},
                    "state_dict_true": model.state_dict()}
        return dataset, metadata
    if kind in ("2PL", "3PL", "LPE", "Spline"):
        raise ValueError(f"the 'from' bank has no {kind} items (no full-data {kind} fit in its runs)")

    mono = np.flatnonzero(bank["monotone"])
    if kind == "mono":
        idx = _pick(rng, mono, n_items)
    elif kind == "mix":
        # recovery: every monotone (1,1) and (2,1) bank item, topped up to n_items with monotone (1,2) and (2,2) items
        # drawn without replacement in proportion to their pools (fixed counts, so every replicate has the same
        # composition and size; fewer than n_items when the bank runs short)
        of = lambda cfg: np.array([i for i in mono if (int(bank["c"][i]), int(bank["d"][i])) == cfg], dtype=int)
        always = np.concatenate([of((1, 1)), of((2, 1))])
        pools = [of((1, 2)), of((2, 2))]
        rest = max(0, min(n_items - len(always), sum(len(pool) for pool in pools)))
        sizes = np.array([len(pool) for pool in pools], dtype=float)
        counts = np.floor(rest * sizes / max(sizes.sum(), 1)).astype(int)
        counts[np.argsort(-(rest * sizes / max(sizes.sum(), 1) - counts))[:rest - counts.sum()]] += 1
        idx = np.concatenate([always] + [rng.choice(pool, size=k, replace=False) for pool, k in zip(pools, counts)])
        idx = rng.permutation(idx)
        n_items = len(idx)
    elif kind != "dip":
        raise ValueError(f"unknown kind {kind}")

    if kind in ("mono", "mix"):
        model, config = _bank_model(bank, idx, n_quad)
        dataset, theta = _sample_bank_model(rng, model, n_students, n_items, density)
        metadata = {"generator": "MiRM", "source": f"from:{kind}", "config": config, "theta_true": theta,
                    "bank_items": [bank["key"][i] for i in idx],
                    "params_true": {k: v.detach().clone() for k, v in model.readable_params.items()},
                    "state_dict_true": model.state_dict()}
        return dataset, metadata

    opts = {"true_dip_frac": 0.1, "theta_dim": 3, "treat_spread": 1.0, "class_fracs": [0.1, 0.3, 0.6],
            "class_spread": 0.75, "loading_frac": 0.25, "loading_scale": 0.5, **(dip_options or {})}
    dips = np.flatnonzero(bank["dip"])
    n_dips = int(round(opts["true_dip_frac"] * n_items))
    notch_min = float(bank.get("dip_min", NOTCH_DEPTH_MIN))   # generated notch items as deep as the bank's dips

    # draw once: bank dip items (topped up with generated notch items), monotone bank items, a random item order
    n_bank_dips = min(n_dips, len(dips))
    dip_idx = rng.choice(dips, size=n_bank_dips, replace=False) if n_bank_dips else np.array([], dtype=int)
    mono_idx = _pick(rng, mono, n_items - n_dips)
    slots = ([{"source": "bank", "bank": int(i), "dip": True} for i in dip_idx]
             + [{"source": "notch", "made": _notch_item(rng, notch_min), "dip": True}
                for _ in range(n_dips - n_bank_dips)]
             + [{"source": "bank", "bank": int(i), "dip": False} for i in mono_idx])
    order = rng.permutation(n_items)                                       # item k of the dataset is slot order[k]

    # PRM repair: while the bank violates PRM, replace the item that lowers the most negative margin the most by an
    # unused item of the same kind (a monotone bank item, or a bank dip item / generated notch item for a dip item)
    removed, replaced = set(), []
    for _ in range(500):
        c = [int(bank["c"][sl["bank"]]) if sl["source"] == "bank" else 2 for sl in slots]
        d = [int(bank["d"][sl["bank"]]) if sl["source"] == "bank" else 2 for sl in slots]
        config = {"c": [c[j] for j in order], "d": [d[j] for j in order]}
        model = build_model("MiRM", n_items, n_quad, **config)
        alpha = np.zeros((n_items, model.n_states)); delta = np.zeros_like(alpha)
        for k, j in enumerate(order):
            sl = slots[j]
            if sl["source"] == "bank":
                i = sl["bank"]
                alpha[k], delta[k] = _layout(bank["alpha_c"][i], bank["delta_c"][i], bank["alpha_d"][i],
                                             bank["delta_d"][i], model.c_max, model.d_max)
            else:
                a_, d_ = sl["made"][0], sl["made"][1]
                alpha[k], delta[k] = _layout(a_[:2], d_[:2], a_[2:], d_[2:], model.c_max, model.d_max)
        _set_params(model, alpha, delta)
        theta_dense = torch.linspace(float(model.theta_q.min()), float(model.theta_q.max()), 1601, dtype=torch.float64)
        with torch.no_grad():
            ch = model.characteristics(theta_dense)
            p, disc = ch["probability"], ch["discrimination"]
            c_margin, d_margin = ((1 - p) * disc).sum(0), (p * disc).sum(0)
        if c_margin.min() >= -1e-10 and d_margin.min() >= -1e-10:         # the criterion of _feasible
            break
        side, margin, weight = ("c", c_margin, 1 - p) if c_margin.min() <= d_margin.min() else ("d", d_margin, p)
        at = int(margin.argmin())
        k = int((weight[:, at] * disc[:, at]).argmin())
        j = int(order[k])
        old = slots[j]
        removed.add((old["source"], old.get("bank")))
        in_use = {sl["bank"] for sl in slots if sl["source"] == "bank"}
        pool = dips if old["dip"] else mono
        free = [int(i) for i in pool if int(i) not in in_use and ("bank", int(i)) not in removed]
        if not free and not old["dip"]:  # small monotone pool (it was drawn with replacement): reuse a kept item
            free = [int(i) for i in pool if ("bank", int(i)) not in removed]
        if free:
            slots[j] = {"source": "bank", "bank": int(rng.choice(free)), "dip": old["dip"]}
        elif old["dip"]:
            slots[j] = {"source": "notch", "made": _notch_item(rng, notch_min), "dip": True}
        else:
            raise RuntimeError(f"PRM repair ran out of monotone bank items (item {k}, {side}_margin "
                               f"{float(margin[at]):.3e} at theta {float(theta_dense[at]):+.2f})")
        name = lambda sl: bank["key"][sl["bank"]] if sl["source"] == "bank" else "notch"
        replaced.append({"item": k, "removed": name(old), "added": name(slots[j]), "dip": old["dip"], "side": side,
                         "theta": float(theta_dense[at]), "margin": float(margin[at])})
    else:
        raise RuntimeError("Could not draw a PRM-feasible bank from the empirical items "
                           f"({len(replaced)} replacements; last at theta {replaced[-1]['theta']:+.2f})")

    depth = [float(bank["depth"][sl["bank"]]) if sl["source"] == "bank" else sl["made"][2] for sl in slots]
    trough = [(float(bank["trough"][sl["bank"]]) if sl["dip"] else np.nan) if sl["source"] == "bank" else sl["made"][3]
              for sl in slots]
    keys = [bank["key"][sl["bank"]] if sl["source"] == "bank" else "notch" for sl in slots]
    sources = [sl["source"] for sl in slots]
    flags = np.array([sl["dip"] for sl in slots])
    dataset, info = _dip_population(rng, model, n_students, density, opts["theta_dim"], opts["treat_spread"],
                                    opts["class_fracs"], opts["class_spread"], opts["loading_frac"], opts["loading_scale"])
    metadata = {"generator": "MiRM", "source": "from:dip", "config": config,
                "params_true": {k: v.detach().clone() for k, v in model.readable_params.items()},
                "state_dict_true": model.state_dict(), "dip_flags": flags[order], "dip_depths": np.array(depth)[order],
                "dip_theta": np.array(trough)[order], "dip_source": [sources[j] for j in order],
                "bank_items": [keys[j] for j in order], "replaced": replaced, **info}
    return dataset, metadata
