from typing import Callable, Sequence

import numpy as np
from scipy.stats import binned_statistic, norm, rankdata
import torch

from mirm.data.datasets import KeyedIRTDataset, ScoredIRTDataset
from mirm.model import MarginalBinaryIRTModel, BinaryMiRM


# ------------------------------------------------------------
# Scoring & Monotonicity & Intercept
# ------------------------------------------------------------

Ability = MarginalBinaryIRTModel | Callable[..., torch.Tensor]


def _normspace(theta_min, theta_max, n_bins):
    return norm.ppf(np.linspace(norm.cdf(theta_min), norm.cdf(theta_max), n_bins))


def ability_bins(theta_min: float = -4, theta_max: float = +4, n_bins: int = 20, tail_bins: int = 1) -> np.ndarray:
    """Equal-population bin edges on N(0,1); each outermost bin is further split in theta (tail_bins extra edges per
    side), because the equal-population tail bins are wide and their single point hides the tail of the ICC."""
    edges = _normspace(theta_min, theta_max, n_bins + 1)
    if tail_bins > 0:
        low = np.linspace(edges[0], edges[1], tail_bins + 2)[1:-1]
        high = np.linspace(edges[-2], edges[-1], tail_bins + 2)[1:-1]
        edges = np.sort(np.concatenate([edges, low, high]))
    return edges


def normal_scores(score: torch.Tensor) -> torch.Tensor:
    """Rank-based normal scores Phi^-1((rank - 1/2) / n) of a sum-type score (average ranks for ties), so that it can
    be binned like a latent ability. Deterministic: the same scores always give the same normal scores."""
    values = score.numpy().astype(np.float64).ravel()
    return torch.from_numpy(norm.ppf((rankdata(values) - 0.5) / len(values)))


@torch.no_grad()
def dip_significance(
    model: BinaryMiRM,
    dataset: ScoredIRTDataset,
    item: int,
    abilities: dict[str, torch.Tensor],
    theta_min: float = -4,
    theta_max: float = +4,
    n_bins: int = 20,
    tail_bins: int = 1,
    shoulder: float = 0.1,
) -> dict:
    """Binned test of an ICC dip against precomputed abilities (one value per student, keyed by reference name).

    The dip of the fitted ICC runs from its local peak to its local dip (the deepest drop below the running maximum).
    The local peak is the largest theta before the dip at which the ICC is still within shoulder * depth of the
    running maximum, so that an elevated plateau (e.g., a lower asymptote that sags) is compared at its edge rather
    than at the boundary of the grid. The test compares the two ability bins whose centers are nearest to the peak
    and the dip:
        z = (p_peak_bin - p_dip_bin) / se,   reference value 0.
    When no student of a reference falls in the peak bin, that reference compares the dip bin with its lowest
    non-empty bin below the dip bin instead; the bins compared per reference are in "compared" ("bins" keeps the
    nominal pair).
    """
    data = dataset.data

    dense_theta = torch.linspace(theta_min, theta_max, 1601)
    p = model.p_correct_indexed(torch.tensor([item]), dense_theta)
    running = p.cummax(dim=0).values
    drop = running - p

    dip_index = int(drop.argmax())
    depth = float(drop[dip_index])
    level = float(running[dip_index]) - shoulder * depth
    peak_index = int(torch.nonzero(p[:dip_index + 1] >= level).flatten().max())
    theta_peak, theta_dip = float(dense_theta[peak_index]), float(dense_theta[dip_index])

    edges = ability_bins(theta_min, theta_max, n_bins, tail_bins)
    centers = (edges[:-1] + edges[1:]) / 2
    low = int(np.abs(centers - theta_peak).argmin())
    high = int(np.abs(centers - theta_dip).argmin())
    if high <= low:
        high = low + 1

    found = {
        "significance": depth,
        "theta_peak": theta_peak,
        "theta_dip": theta_dip,
        "bins": (low, high),
        "edges": edges,
        "referenced_significance": None,
        "binned_probs": None,
        "compared": None,
    }
    if depth <= 0 or high >= len(centers) or not abilities:
        return found

    took = data[:, 1].long() == item
    sid = data[:, 0].long()[took]
    solved = data[:, 2][took].numpy()

    referenced_significance, binned_probs, compared = {}, {}, {}
    for tag, ability in abilities.items():
        ability = ability[sid].numpy()
        p_hat, sd, num = (
            binned_statistic(ability, solved, bins=edges, statistic=st).statistic
            for st in ("mean", "std", "count")
        )
        binned_probs[tag] = torch.from_numpy(np.stack([p_hat, sd, num]))

        peak = low
        if not num[low] > 0:  # empty peak bin: the lowest non-empty bin below the dip bin
            nonempty = np.flatnonzero(num[:high] > 0)
            if len(nonempty):
                peak = int(nonempty[0])
        compared[tag] = (peak, high)
        with np.errstate(divide="ignore", invalid="ignore"):  # empty bin or p in {0, 1}: z is inf / nan
            std_error = np.sqrt(p_hat[peak] * (1 - p_hat[peak]) / num[peak] + p_hat[high] * (1 - p_hat[high]) / num[high])
            referenced_significance[tag] = (p_hat[peak] - p_hat[high]) / std_error

    found.update(referenced_significance=referenced_significance, binned_probs=binned_probs, compared=compared)
    return found


@torch.no_grad()
def population_density(model: MarginalBinaryIRTModel, dataset: ScoredIRTDataset, theta: torch.Tensor,
                       chunk_students: int = 20000) -> torch.Tensor:
    """Mean posterior density of the students on theta (N(0,1) prior), for the grey strip under the ICC plots."""
    log_prior = -0.5 * theta.square()
    order = dataset.student_ids.argsort(stable=True)
    data = dataset.data[order]
    students = data[:, 0].long().contiguous()
    total = torch.zeros_like(theta)
    for start in range(0, dataset.n_students, chunk_students):
        lo, hi = torch.searchsorted(students, torch.tensor([start, start + chunk_students], dtype=students.dtype)).tolist()
        if hi == lo:
            continue
        part = data[lo:hi].clone()
        part[:, 0] -= start
        ll = model.ll_person_grid(part, int(part[:, 0].max()) + 1, theta)
        total += torch.softmax(ll + log_prior[None, :], dim=1).sum(dim=0)
    step = float(theta[1] - theta[0])
    return total / (total.sum() * step)


def total_score(dataset: ScoredIRTDataset, **_) -> torch.Tensor:
    data = dataset.data

    students = data[:, 0].long()
    credits = data[:, 2].to(torch.float64)

    total = torch.zeros(int(students.max()) + 1)
    counts = torch.zeros_like(total)

    total.scatter_add_(0, students, credits)
    counts.scatter_add_(0, students, torch.ones_like(credits))

    return total / counts


def rest_score(dataset: ScoredIRTDataset, item: int, **_) -> torch.Tensor:
    data = dataset.data
    if item < 0 or item >= dataset.n_items:
        raise ValueError(f"Invalid item index: {item}")

    students = data[:, 0].long()
    items = data[:, 1].long()
    credits = data[:, 2].to(torch.float64)

    total = torch.zeros(int(students.max()) + 1)
    counts = torch.zeros_like(total)

    total.scatter_add_(0, students, credits)
    counts.scatter_add_(0, students, torch.ones_like(credits))

    mask = items == item

    sid = students[mask]
    other = (counts[sid] - 1).clamp_min(0)
    rest = torch.where(other > 0, (total[sid] - credits[mask]) / other, 0)

    out = total / counts
    out[sid] = rest
    return out


@torch.no_grad()
def generalized_score(dataset: ScoredIRTDataset, model: BinaryMiRM, theta: torch.Tensor = None, **_) -> torch.Tensor:
    data = dataset.data

    students = data[:, 0].long()
    items = data[:, 1].long()
    credits = data[:, 2].to(torch.float64)

    if theta is None:
        theta = model.map_theta(data)
    theta_unique, inverse = torch.unique(theta, return_inverse=True)
    a = model.characteristics(theta_unique)["discrimination"]

    weights = a[items, inverse[students]]

    total = torch.zeros(int(students.max()) + 1)
    scale = torch.zeros_like(total)

    total.scatter_add_(0, students, weights * credits)
    scale.scatter_add_(0, students, weights)

    return total / scale


@torch.no_grad()
def intercept(
    dataset: KeyedIRTDataset, 
    ability_model: Ability, 
    item: int,
    theta_min: float = -4,
    theta_max: float = +4,
    n_bins: int = 20,
    ability: torch.Tensor = None,
) -> tuple[np.ndarray, np.ndarray]:
    data = dataset.data

    took = data[:, 1].long() == item
    sid = data[:, 0].long()[took]
    chosen = data[:, 2][took].int().numpy()

    K = dataset.n_choices
    overall = np.bincount(chosen, minlength=K + 1)[1:] / len(chosen)

    ability_bins = _normspace(theta_min, theta_max, n_bins + 1)

    scored_dataset = dataset.to_scored()
    scored_data = scored_dataset.data

    if ability is not None:
        pass
    elif isinstance(ability_model, MarginalBinaryIRTModel):
        ability = ability_model.map_theta(scored_data)
    else:
        ability = normal_scores(ability_model(scored_dataset, item=item))

    ability = ability[sid].numpy()
    rates = np.stack([
        binned_statistic(ability, (chosen == k).astype(float), bins=ability_bins, statistic="mean").statistic
        for k in range(1, K + 1)
    ])

    return overall, rates


# ------------------------------------------------------------
# Dimensionality
# ------------------------------------------------------------

@torch.no_grad()
def domain_margins(model: BinaryMiRM, domain_masks: torch.Tensor, theta: torch.Tensor) -> dict[str, torch.Tensor]:
    char = model.characteristics(theta)
    p, a = char["probability"], char["discrimination"]
    masks = domain_masks.to(p.dtype)

    c_term, d_term = (1 - p) * a, p * a
    return {
        "c_margin": masks @ c_term,
        "d_margin": masks @ d_term,
        "c_norm": (masks @ c_term) / (masks @ c_term.abs()).clamp_min(torch.finfo(p.dtype).eps),
        "d_norm": (masks @ d_term) / (masks @ d_term.abs()).clamp_min(torch.finfo(p.dtype).eps),
    }


@torch.no_grad()
def prm_coordinate(anchor: BinaryMiRM, theta: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = anchor.p_correct_grid(theta).clamp_min(torch.finfo(theta.dtype).tiny)
    source = torch.log(p).sum(dim=0)

    order = source.argsort()
    source, grid = source[order].numpy(), theta[order].numpy()

    coordinate = np.interp(target.numpy(), source, grid, left=np.nan, right=np.nan)
    return torch.from_numpy(coordinate)


@torch.no_grad()
def trajectory(model: BinaryMiRM, anchors: dict[str, tuple[BinaryMiRM, torch.Tensor]], theta: torch.Tensor) -> dict[str, torch.Tensor]:
    p = model.p_correct_grid(theta).clamp_min(torch.finfo(theta.dtype).tiny)

    coordinates = {}
    for name, (anchor, item_ids) in anchors.items():
        target = torch.log(p[item_ids]).sum(dim=0)
        coordinates[name] = prm_coordinate(anchor, theta, target)

    return coordinates


@torch.no_grad()
def map_abilities(model, data: torch.Tensor, n_students: int, theta: torch.Tensor,
                  chunk_students: int = 20000) -> torch.Tensor:
    """MAP ability (N(0,1) prior) on the grid for every student id in [0, n_students), over whole-student chunks;
    students without a response in data get NaN."""
    log_prior = -0.5 * theta.square()
    data = data[data[:, 0].long().argsort(stable=True)]
    students = data[:, 0].long().contiguous()
    out = torch.full((n_students,), float("nan"), dtype=theta.dtype)
    for start in range(0, n_students, chunk_students):
        lo, hi = torch.searchsorted(students, torch.tensor([start, start + chunk_students], dtype=students.dtype)).tolist()
        if hi == lo:
            continue
        part = data[lo:hi].clone()
        part[:, 0] -= start
        width = int(part[:, 0].max()) + 1
        ll = model.ll_person_grid(part, width, theta) + log_prior[None, :]
        answered = torch.bincount(part[:, 0].long(), minlength=width) > 0
        values = theta[ll.argmax(dim=1)]
        out[start:start + width] = torch.where(answered, values, torch.full_like(values, float("nan")))
    return out


@torch.no_grad()
def generalized_score_curve(model, theta: torch.Tensor, n_samples: int = 400, seed: int = 0,
                            chunk: int = 16, singular: float = 0.05, smooth: float = 0.08) -> tuple[np.ndarray, np.ndarray]:
    """Monte Carlo generalized-score curve G_R(theta) / L for a binary test (all items of the model).

    At each theta, response patterns are drawn from the model and each is scored at its own posterior mean,
    R(x)/L = sum_j a_j(theta_hat) x_j / a_J(theta_hat). The formula is also evaluated where the score direction is not
    certified (a_J <= 0, outside the scoring domain), so that the curve shows what the score does there; patterns with
    a nearly singular denominator (|a_J| < singular * sum_j |a_j|) are left out. Returns (curve, certified share), the
    share being the fraction of patterns with a_J > 0. The curve is lightly smoothed over theta (Gaussian kernel with
    standard deviation `smooth`, NaN-aware) to remove Monte Carlo noise."""
    generator = torch.Generator().manual_seed(seed)
    p = model.p_correct_grid(theta)                                   # (J, T)
    J, T = p.shape
    curve = np.full(T, np.nan)
    share = np.zeros(T)
    for start in range(0, T, chunk):
        block = p[:, start:start + chunk].T                          # (t, J)
        t = block.shape[0]
        x = torch.bernoulli(block[:, None, :].expand(t, n_samples, J).contiguous(), generator=generator).reshape(-1, J)
        n = x.shape[0]
        students = torch.arange(n).repeat_interleave(J)
        items = torch.arange(J).repeat(n)
        data = torch.stack([students.to(torch.float64), items.to(torch.float64), x.reshape(-1)], dim=1)
        theta_hat = model.eap_theta(data, n)
        a = model.characteristics(theta_hat)["discrimination"]       # (J, n)
        a_total = a.sum(dim=0)
        regular = a_total.abs() > singular * a.abs().sum(dim=0)
        score = (a * x.T).sum(dim=0) / torch.where(regular, a_total, torch.ones_like(a_total))
        certified = (a_total > 0).reshape(t, n_samples)
        regular = regular.reshape(t, n_samples)
        score = score.reshape(t, n_samples)
        for i in range(t):
            share[start + i] = float(certified[i].double().mean())
            if regular[i].any():
                curve[start + i] = float(score[i][regular[i]].mean())
    if smooth and T > 1:
        grid = theta.numpy()
        kernel = np.exp(-0.5 * ((grid[:, None] - grid[None, :]) / smooth) ** 2)
        valid = np.isfinite(curve)
        weights = kernel * valid[None, :]
        smoothed = (weights * np.where(valid, curve, 0.0)[None, :]).sum(1) / np.maximum(weights.sum(1), 1e-300)
        curve = np.where(valid, smoothed, np.nan)
    return curve, share
