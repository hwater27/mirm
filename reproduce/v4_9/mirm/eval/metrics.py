from typing import Optional

import numpy as np
import torch
from scipy import sparse
from scipy.stats import chi2, kendalltau, spearmanr
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score

from mirm.data.datasets import ScoredIRTDataset
from mirm.model import BinaryIRTModel, MarginalBinaryIRTModel, BinaryMiRM
from mirm.model.core import response_design


# ------------------------------------------------------------
# General helper utility functions
# ------------------------------------------------------------

def _as_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)

def _pair_as_numpy(truth, estimate) -> tuple[np.ndarray, np.ndarray]:
    truth, estimate = np.broadcast_arrays(_as_numpy(truth), _as_numpy(estimate))
    valid = np.isfinite(truth) & np.isfinite(estimate)

    return truth[valid], estimate[valid]

def _clip_prob(prob: torch.Tensor | np.ndarray, eps: Optional[float] = None) -> torch.Tensor | np.ndarray:
    if isinstance(prob, torch.Tensor):
        eps = torch.finfo(prob.dtype).eps if eps is None else eps
        return torch.clamp(prob, eps, 1 - eps)
    elif isinstance(prob, np.ndarray):
        eps = np.finfo(prob.dtype).eps if eps is None else eps
        return np.clip(prob, eps, 1 - eps)
    else:
        raise TypeError(f"Unsupported type: {type(prob)!r}")

# ------------------------------------------------------------
# General helper metric functions
# ------------------------------------------------------------

def _rmse(truth, estimate) -> Optional[float]:
    if truth is None or estimate is None:
        return None

    truth, estimate = _pair_as_numpy(truth, estimate)
    if truth.size == 0:
        return float("nan")

    return float(np.sqrt(np.mean((truth - estimate) ** 2)))

def _bias(truth, estimate) -> Optional[float]:
    if truth is None or estimate is None:
        return None

    truth, estimate = _pair_as_numpy(truth, estimate)
    if truth.size == 0:
        return float("nan")
    
    return float(np.mean(estimate - truth))

def _mae(truth, estimate) -> Optional[float]:
    if truth is None or estimate is None:
        return None

    truth, estimate = _pair_as_numpy(truth, estimate)
    if truth.size == 0:
        return float("nan")
    
    return float(np.mean(np.abs(estimate - truth)))

def _pearson(x, y) -> Optional[float]:
    if x is None or y is None:
        return None

    x, y = _pair_as_numpy(x, y)
    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    
    return float(np.corrcoef(x, y)[0, 1])

def _spearman(x, y) -> Optional[float]:
    if x is None or y is None:
        return None

    x, y = _pair_as_numpy(x, y)
    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    
    return float(spearmanr(x, y)[0])

def _kendall(x, y) -> Optional[float]:
    if x is None or y is None:
        return None

    x, y = _pair_as_numpy(x, y)
    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    
    return float(kendalltau(x, y)[0])

def _auc(labels, prob) -> Optional[float]:
    if labels is None or prob is None:
        return None

    labels, prob = _pair_as_numpy(labels, prob)
    classes = np.unique(labels)
    if classes.size != 2 or not np.all(np.isin(classes, (0.0, 1.0))):
        return float("nan")

    return float(roc_auc_score(labels, prob))

def _inversion_rate(x, y) -> Optional[float]:
    if x is None or y is None:
        return None

    x, y = _pair_as_numpy(x, y)
    if x.size < 2:
        return float("nan")

    _, x_counts = np.unique(x, return_counts=True)
    _, y_counts = np.unique(y, return_counts=True)
    _, joint_counts = np.unique(np.column_stack((x, y)), axis=0, return_counts=True)
    choose_two = lambda counts: np.sum(counts * (counts - 1) // 2)
    comparable = (
        x.size * (x.size - 1) // 2
        - choose_two(x_counts)
        - choose_two(y_counts)
        + choose_two(joint_counts)
    )
    if comparable == 0:
        return float("nan")

    y_rank = np.unique(y, return_inverse=True)[1]
    order = np.lexsort((y_rank, x))
    ranks = y_rank[order]
    sorted_x = x[order]

    tree = np.zeros(int(ranks.max()) + 2, dtype=np.int64)

    def prefix_sum(index: int) -> int:
        total = 0
        while index > 0:
            total += int(tree[index])
            index -= index & -index
        return total

    def add(index: int) -> None:
        while index < len(tree):
            tree[index] += 1
            index += index & -index

    discordant = 0
    previous = 0
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and sorted_x[stop] == sorted_x[start]:
            stop += 1
        for rank in ranks[start:stop]:
            discordant += previous - prefix_sum(int(rank) + 1)
        for rank in ranks[start:stop]:
            add(int(rank) + 1)
            previous += 1
        start = stop

    return float(discordant / comparable)

# ------------------------------------------------------------
# Helper functions for some calculations
# ------------------------------------------------------------

@torch.no_grad()
def _held_out_predictions(
    model: MarginalBinaryIRTModel,
    dataset: ScoredIRTDataset,
    n_folds: int = 5,
    seed: int = 27,
) -> torch.Tensor:
    if n_folds < 1:
        raise ValueError("n_folds must be positive.")

    p_grid = model.p_correct_grid(model.theta_q)
    p_grid = _clip_prob(p_grid)

    data = dataset.data.to(p_grid.device)
    student_ids = data[:, 0].long()
    item_ids = data[:, 1].long()
    credits = data[:, 2].to(p_grid.dtype)

    log_p = torch.log(p_grid)
    log_1mp = torch.log1p(-p_grid)
    generator = torch.Generator().manual_seed(seed)
    fold_ids = torch.empty_like(student_ids)

    for student_id in range(dataset.n_students):
        indices = torch.nonzero(student_ids == student_id, as_tuple=False).flatten()

        n_responses = len(indices)
        if n_responses == 0:
            continue

        permutation = torch.randperm(n_responses, generator=generator).to(indices.device)
        indices = indices[permutation]

        actual_folds = min(n_folds, n_responses)
        fold_ids[indices] = torch.arange(n_responses, device=indices.device) % actual_folds

    contributions = (credits[:, None] * log_p[item_ids] + (1.0 - credits[:, None]) * log_1mp[item_ids])
    totals = p_grid.new_zeros((dataset.n_students, model.n_quad))
    totals.index_add_(0, student_ids, contributions)

    fold_totals = p_grid.new_zeros((dataset.n_students * n_folds, model.n_quad))
    buckets = student_ids * n_folds + fold_ids
    fold_totals.index_add_(0, buckets, contributions)
    fold_totals = fold_totals.view(dataset.n_students, n_folds, model.n_quad)

    log_posterior = model.log_quad_weights[None, None, :] + totals[:, None, :] - fold_totals
    posterior = torch.softmax(log_posterior, dim=-1)
    observation_posterior = posterior[student_ids, fold_ids]
    predictions = (p_grid[item_ids] * observation_posterior).sum(dim=1)

    return predictions.cpu()    

def _response_matrix(dataset: ScoredIRTDataset, n_items: int) -> np.ndarray:
    student_ids = _as_numpy(dataset.student_ids).astype(int)
    item_ids = _as_numpy(dataset.item_ids).astype(int)
    credits = _as_numpy(dataset.credits)

    if np.any(item_ids >= n_items):
        raise ValueError("Dataset contains item IDs not present in the model.")

    cells = student_ids * n_items + item_ids
    if np.unique(cells).size != cells.size:
        raise ValueError("Each student-item pair must have at most one response.")

    responses = np.full((dataset.n_students, n_items), np.nan)
    responses[student_ids, item_ids] = credits

    return responses

@torch.no_grad()
def _sx2(
    model: MarginalBinaryIRTModel,
    dataset: ScoredIRTDataset,
) -> tuple[np.ndarray, np.ndarray]:
    responses = _response_matrix(dataset, model.n_items)

    if np.isnan(responses).any():
        raise ValueError("S-X2 requires a complete student-by-item response matrix.")
    if not np.all(np.isin(responses, (0.0, 1.0))):
        raise ValueError("S-X2 requires binary responses.")

    total_scores = responses.sum(axis=1).astype(int)
    p_grid = model.p_correct_grid(model.theta_q).transpose(0, 1)
    weights = model.tilde_q

    statistics = np.full(model.n_items, np.nan)
    pvalues = np.full(model.n_items, np.nan)

    for item in range(model.n_items):
        rest_pmf = p_grid.new_zeros((model.n_quad, model.n_items))
        rest_pmf[:, 0] = 1.0
        max_score = 0

        for other in range(model.n_items):
            if other == item:
                continue

            probability = p_grid[:, other, None]
            updated = rest_pmf.clone()
            updated[:, :max_score + 2] = (
                rest_pmf[:, :max_score + 2] * (1.0 - probability)
            )
            updated[:, 1:max_score + 2] += (
                rest_pmf[:, :max_score + 1] * probability
            )
            rest_pmf = updated
            max_score += 1

        item_probability = p_grid[:, item]
        statistic = 0.0
        n_groups = 0

        for score in range(model.n_items + 1):
            selected = total_scores == score
            group_size = int(selected.sum())

            if group_size == 0:
                continue

            correct = (
                (weights * item_probability * rest_pmf[:, score - 1]).sum()
                if score > 0 else weights.new_zeros(())
            )
            incorrect = (
                (weights * (1.0 - item_probability) * rest_pmf[:, score]).sum()
                if score < model.n_items else weights.new_zeros(())
            )
            denominator = correct + incorrect

            if denominator <= 0.0:
                continue

            expected_correct = group_size * float((correct / denominator).item())
            expected_incorrect = group_size - expected_correct

            if expected_correct < 1.0 or expected_incorrect < 1.0:
                continue

            observed_correct = float(responses[selected, item].sum())
            difference = observed_correct - expected_correct
            statistic += difference ** 2 * (
                1.0 / expected_correct + 1.0 / expected_incorrect
            )
            n_groups += 1

        if isinstance(model, BinaryMiRM):
            n_item_params = int(2 * (model.c[item] + model.d[item] - 1).item())
        else:
            n_item_params = max(1, round(model.n_effective_params / model.n_items))

        degrees_of_freedom = n_groups - n_item_params
        if degrees_of_freedom > 0:
            statistics[item] = statistic
            pvalues[item] = chi2.sf(statistic, degrees_of_freedom)

    return statistics, pvalues

@torch.no_grad()
def _q3(
    model: MarginalBinaryIRTModel,
    dataset: ScoredIRTDataset,
    min_pairs: int = 3,
) -> np.ndarray:
    data = dataset.data.to(model.theta_q.device)
    student_ids = data[:, 0].long()
    item_ids = data[:, 1].long()

    theta_hat = model.eap_theta(data)
    expected = model.p_correct_indexed(item_ids, theta_hat[student_ids])
    residual = _as_numpy(data[:, 2] - expected)

    rows = _as_numpy(student_ids).astype(np.int64)
    cols = _as_numpy(item_ids).astype(np.int64)
    if np.any(cols >= model.n_items):
        raise ValueError("Dataset contains item IDs not present in the model.")
    if np.unique(rows * model.n_items + cols).size != rows.size:
        raise ValueError("Each student-item pair must have at most one response.")

    shape = (dataset.n_students, model.n_items)
    R = sparse.csr_matrix((residual, (rows, cols)), shape=shape)
    M = sparse.csr_matrix((np.ones_like(residual), (rows, cols)), shape=shape)

    n = (M.T @ M).toarray()
    sx = (R.T @ M).toarray()
    sxx = (R.multiply(R).T @ M).toarray()
    sxy = (R.T @ R).toarray()

    with np.errstate(divide="ignore", invalid="ignore"):
        cov = sxy - sx * sx.T / n
        var = sxx - sx ** 2 / n
        q3 = cov / np.sqrt(var * var.T)

    q3[(n < min_pairs) | ~(var > 1e-12) | ~(var.T > 1e-12)] = np.nan
    np.fill_diagonal(q3, np.nan)
    return q3

# ------------------------------------------------------------
# Main metric functions
# ------------------------------------------------------------

@torch.no_grad()
def fit_metrics(
    model: MarginalBinaryIRTModel,
    data_train: ScoredIRTDataset,
) -> dict:
    nll = float(model(data_train.data).item())
    n_students = data_train.n_students
    n_obs = len(data_train)
    n_params = model.n_effective_params

    # Information criteria
    aic  = 2.0 * nll + n_params * 2.0
    bic  = 2.0 * nll + n_params * np.log(max(2, n_students))
    sbic = 2.0 * nll + n_params * np.log((n_students + 2.0) / 24.0)

    return {
        "nll_total": nll,
        "nll_per_student": nll / n_students,
        "nll_per_obs": nll / n_obs,
        "n_params": n_params,
        "aic": aic,
        "bic": bic,
        "sbic": sbic,
    }


@torch.no_grad()
def rmsd_metrics(
    model: MarginalBinaryIRTModel,
    data_train: ScoredIRTDataset,
    rmsd_threshold: float = 0.12,
) -> dict:
    """Item fit RMSD (as in PISA): the root of the posterior-density-weighted mean squared difference between the
    pseudo-observed proportion correct and the model ICC on the quadrature nodes,
        RMSD_j = sqrt( sum_q f_jq (P_obs,jq - P_jq)^2 ),  f_jq = n_jq / sum_q n_jq,  P_obs,jq = r_jq / n_jq,
    where n_jq and r_jq are the posterior-weighted counts of respondents and of correct responses of item j."""
    data = data_train.data
    posterior = model.posterior_grid(data, data_train.n_students)
    design = response_design(data, model.n_items, data_train.n_students)
    counts = torch.sparse.mm(design.t(), posterior).reshape(model.n_items, 2, -1)  # [incorrect, correct]

    n_jq = counts.sum(dim=1)
    r_jq = counts[:, 1]
    p_model = model.p_correct_grid(model.theta_q)
    p_obs = torch.where(n_jq > 0, r_jq / n_jq.clamp_min(1e-300), p_model)
    weights = n_jq / n_jq.sum(dim=1, keepdim=True).clamp_min(1e-300)

    rmsd = _as_numpy(((p_obs - p_model).square() * weights).sum(dim=1).sqrt())
    return {
        "item_rmsd": rmsd,
        "rmsd_mean": float(np.mean(rmsd)),
        "rmsd_max": float(np.max(rmsd)),
        "n_rmsd_misfit": int((rmsd > rmsd_threshold).sum()),
    }

def SX2_metrics(
    model: MarginalBinaryIRTModel,
    data_train: ScoredIRTDataset,
    sx2_alpha: float = 0.05,
) -> dict:
    if not 0.0 < sx2_alpha < 1.0:
        raise ValueError("sx2_alpha must be between zero and one.")

    if len(data_train) != data_train.n_students * data_train.n_items:
        return {
            "sx2_per_item": None,
            "sx2_pvalue_per_item": None,
            "n_item_misfit": None,
            "item_misfit_fraction": None,
        }

    sx2_per_item, sx2_pvalue_per_item = _sx2(model, data_train)
    valid = np.isfinite(sx2_pvalue_per_item)
    misfit = valid & (sx2_pvalue_per_item < sx2_alpha)

    n_item_misfit = int(misfit.sum())
    item_misfit_fraction = (
        float(misfit.sum() / valid.sum())
        if valid.any() else float("nan")
    )

    return {
        "sx2_per_item": sx2_per_item,
        "sx2_pvalue_per_item": sx2_pvalue_per_item,
        "n_item_misfit": n_item_misfit,
        "item_misfit_fraction": item_misfit_fraction,
    }

def dependence_metrics(
    model: MarginalBinaryIRTModel,
    data_train: ScoredIRTDataset,
    q3_threshold: float = 0.20,
) -> dict:
    q3_matrix = _q3(model, data_train)
    upper = q3_matrix[np.triu_indices(model.n_items, k=1)]
    valid = np.isfinite(upper)

    if valid.any():
        q3_mean = float(upper[valid].mean())
        q3_max = float(upper[valid].max())

        adjusted = upper[valid] - q3_mean
        adjusted_q3_max = float(adjusted.max())
        n_high_q3_pairs = int((adjusted > q3_threshold).sum())
    else:
        q3_mean = float("nan")
        q3_max = float("nan")
        adjusted_q3_max = float("nan")
        n_high_q3_pairs = 0

    return {
        "q3_matrix": q3_matrix,
        "q3_mean": q3_mean,
        "q3_max": q3_max,
        "adjusted_q3_max": adjusted_q3_max,
        "n_high_q3_pairs": n_high_q3_pairs,
    }

@torch.no_grad()
def predictive_metrics(
    model: MarginalBinaryIRTModel, 
    data_test: ScoredIRTDataset,
    n_folds: int = 5,
    seed: int = 27,
    per_item: bool = False,
) -> dict:
    data = data_test.data
    student_ids = data_test.student_ids
    item_ids = data_test.item_ids
    credits = data_test.credits

    n_students = data_test.n_students
    n_obs = len(data_test)
    n_items = model.n_items

    # Point-estimate predictions
    theta_hat = model.map_theta(data)
    point_prob = model.p_correct_indexed(item_ids, theta_hat[student_ids]).detach().cpu()

    # Response-fold posterior-marginal predictions
    marginal_prob = _held_out_predictions(model, data_test, n_folds=n_folds, seed=seed)

    credits_np = _as_numpy(credits)
    marginal_np = _as_numpy(marginal_prob)

    marginal_np = _clip_prob(marginal_np)

    losses = - credits_np * np.log(marginal_np) - (1 - credits_np) * np.log1p(-marginal_np)

    nll = float(losses.sum())
    auc = _auc(credits, point_prob)
    mauc = _auc(credits, marginal_prob)
    brier_score = float(np.mean((marginal_np - credits_np) ** 2))


    if per_item:
        item_nll = np.full(n_items, np.nan)
        item_auc = np.full(n_items, np.nan)

        item_ids_np = _as_numpy(item_ids).astype(int)

        for item in range(n_items):
            selected = item_ids_np == item

            if not np.any(selected):
                continue

            item_nll[item] = float(losses[selected].mean())
            item_auc[item] = _auc(credits_np[selected], marginal_np[selected])
    else:
        item_nll = None
        item_auc = None

    return {
        "ho_nll_total": nll,
        "ho_nll_per_student": nll / n_students,
        "ho_nll_per_obs": nll / n_obs,
        "ho_brier_score": brier_score,
        "ho_auc": auc,
        "ho_mauc": mauc,
        "item_nll": item_nll,
        "item_auc": item_auc,
    }

@torch.no_grad()
def characteristic_metrics(
    model: BinaryIRTModel,
    theta_grid: torch.Tensor,
    true_model: Optional[BinaryIRTModel] = None,
) -> dict:
    """
    Includes metrics for both ICC and macrostate characteristics.
    """
    # Information-related metrics
    item_info = model.item_information_grid(theta_grid)
    test_info = item_info.sum(dim=0)
    cond_sem = test_info.rsqrt()

    # ICC-related metrics
    icc_true  = None if true_model is None else true_model.p_correct_grid(theta_grid)
    icc_model = model.p_correct_grid(theta_grid)

    icc_rmse  = _rmse(icc_true, icc_model)
    item_icc_rmse = None if icc_true is None else _as_numpy((icc_true - icc_model).square().mean(dim=1).sqrt())
    icc_bias  = _bias(icc_true, icc_model)
    icc_mae   = _mae(icc_true, icc_model)
    icc_pcorr = _pearson(icc_true, icc_model)
    icc_scorr = _spearman(icc_true, icc_model)

    # MiRM-specific metrics
    if isinstance(model, BinaryMiRM) and isinstance(true_model, BinaryMiRM):
        characteristics = model.characteristics(theta_grid)
        true_characteristics = true_model.characteristics(theta_grid)
        a_model, b_model = characteristics["discrimination"], characteristics["difficulty"]
        a_true, b_true = true_characteristics["discrimination"], true_characteristics["difficulty"]
    else:
        a_model, b_model = None, None
        a_true, b_true = None, None

    a_rmse = _rmse(a_true, a_model)
    b_rmse = _rmse(b_true, b_model)
    a_bias = _bias(a_true, a_model)
    b_bias = _bias(b_true, b_model)

    return {
        "icc_rmse": icc_rmse,
        "item_icc_rmse": item_icc_rmse,
        "icc_bias": icc_bias,
        "icc_mae": icc_mae,
        "icc_pearson": icc_pcorr,
        "icc_spearman": icc_scorr,
        "a_rmse": a_rmse,
        "a_bias": a_bias,
        "b_rmse": b_rmse,
        "b_bias": b_bias,
        "item_information": item_info,     
        "test_information": test_info,     
        "conditional_sem": cond_sem,     
    }

@torch.no_grad()
def ability_metrics(
    model: MarginalBinaryIRTModel,
    dataset: ScoredIRTDataset,
    theta_true: Optional[torch.Tensor] = None,
    theta_reference: Optional[torch.Tensor] = None,
) -> dict:
    """
    Includes both intra-model reliability and comparison to external data.
    """
    posterior = model.posterior_grid(dataset.data)
    theta_grid = model.theta_q[None, :]
    theta_hat = (posterior * theta_grid).sum(dim=1)

    posterior_var = (posterior * (theta_grid - theta_hat[:, None]).square()).sum(dim=1)
    posterior_var = posterior_var.clamp_min(0.0)
    posterior_sd = posterior_var.sqrt()

    posterior_sd_mean = float(posterior_sd.mean().item())

    between_variance = theta_hat.var(unbiased=False)
    error_variance = posterior_var.mean()
    total_variance = between_variance + error_variance

    if total_variance > 0.0:
        marginal_reliability = float((between_variance / total_variance).item())
    else:
        marginal_reliability = float("nan")

    # Reference selection
    if theta_true is not None:
        reference = "true"
        theta_compare = theta_true
    elif theta_reference is not None:
        reference = "model"
        theta_compare = theta_reference
    else:
        reference = "self"
        theta_compare = None

    # Ability-related metrics
    rmse = _rmse(theta_compare, theta_hat)
    bias = _bias(theta_compare, theta_hat)
    pearson = _pearson(theta_compare, theta_hat)
    spearman = _spearman(theta_compare, theta_hat)
    kendall = _kendall(theta_compare, theta_hat)
    inv_rate = _inversion_rate(theta_compare, theta_hat)

    return {
        "reference": reference,
        "rmse": rmse,
        "bias": bias,
        "pearson": pearson,
        "spearman": spearman,
        "kendall": kendall,
        "inversion_rate": inv_rate,
        "posterior_sd_mean": posterior_sd_mean,
        "marginal_reliability": marginal_reliability,
    }

def parameter_metrics(model: BinaryIRTModel, true_model: BinaryIRTModel) -> dict:
    if model.n_items != true_model.n_items:
        raise ValueError("Model and truth must have the same number of items.")

    params_model = model.readable_params
    params_truth = true_model.readable_params

    if params_model.keys() != params_truth.keys():
        raise ValueError("Model and truth parameters must have the same keys.")

    # Microstate matching
    if isinstance(model, BinaryMiRM) and isinstance(true_model, BinaryMiRM):
        fit_alpha = _as_numpy(model.alpha_ordered)
        fit_delta = _as_numpy(model.delta_ordered)
        fit_macro = _as_numpy(model.macrostate_ordered).astype(int)

        true_alpha = _as_numpy(true_model.alpha_ordered)
        true_delta = _as_numpy(true_model.delta_ordered)
        true_macro = _as_numpy(true_model.macrostate_ordered).astype(int)

        theta = _as_numpy(true_model.theta_q)
        weights = _as_numpy(true_model.tilde_q)
        weights = weights / weights.sum()

        matched_true_alpha = []
        matched_fit_alpha = []
        matched_true_delta = []
        matched_fit_delta = []

        logweight_rmse = []
        skipped_items = []
        matched_records = []

        n_regauged = 0
        for item in range(true_model.n_items):
            true_counts = [np.sum(true_macro[item] == macrostate) for macrostate in (1,2)]
            fit_counts = [np.sum(fit_macro[item] == macrostate) for macrostate in (1,2)]

            if true_counts != fit_counts:
                skipped_items.append(item)
                continue

            # Gauge after matching: the fitted model is re-gauged so that the state matched to the true
            # reference state (alpha = delta = 0) becomes (0, 0). The matching cost depends on the gauge, so every
            # fitted state of the reference's macrostate is tried as that anchor; the anchor with the smallest total
            # cost (both macrostates) is kept.
            true_active = true_macro[item] > 0
            ground = np.flatnonzero(true_active & (true_alpha[item] == 0) & (true_delta[item] == 0))[0]
            ground_macro = true_macro[item, ground]
            true_idx = {m: np.flatnonzero(true_macro[item] == m) for m in (1, 2)}
            fit_idx = {m: np.flatnonzero(fit_macro[item] == m) for m in (1, 2)}
            own_anchor = np.flatnonzero((fit_macro[item] > 0) & (fit_alpha[item] == 0) & (fit_delta[item] == 0))[0]

            best = None
            for anchor in fit_idx[ground_macro]:
                fa_all = fit_alpha[item] - fit_alpha[item, anchor]
                fd_all = fit_delta[item] - fit_delta[item, anchor]
                total, pairs = 0.0, {}
                for macrostate in (1, 2):
                    ti, fi = true_idx[macrostate], fit_idx[macrostate]
                    ta, td = true_alpha[item, ti], true_delta[item, ti]
                    fa, fd = fa_all[fi], fd_all[fi]
                    true_logweights = -ta[:, None] * theta[None, :] + td[:, None]
                    fit_logweights = -fa[:, None] * theta[None, :] + fd[:, None]
                    differences = true_logweights[:, None, :] - fit_logweights[None, :, :]
                    cost = np.sum(np.square(differences) * weights[None, None, :], axis=2)
                    if macrostate == ground_macro:      # the anchor is matched to the true reference state
                        g, a = int(np.flatnonzero(ti == ground)[0]), int(np.flatnonzero(fi == anchor)[0])
                        rows = np.array([r for r in range(len(ti)) if r != g], dtype=int)
                        cols = np.array([q for q in range(len(fi)) if q != a], dtype=int)
                        r_order, c_order = linear_sum_assignment(cost[np.ix_(rows, cols)]) if len(rows) else ([], [])
                        true_order = np.concatenate([[g], rows[r_order]]).astype(int)
                        fit_order = np.concatenate([[a], cols[c_order]]).astype(int)
                    else:
                        true_order, fit_order = linear_sum_assignment(cost)
                    total += cost[true_order, fit_order].sum()
                    pairs[macrostate] = (ta, td, fa, fd, true_order, fit_order)
                if best is None or total < best[0] - 1e-12:
                    best = (total, anchor, pairs)

            item_cost, anchor, pairs = best
            n_regauged += int(anchor != own_anchor)
            item_n_states = 0
            for macrostate in (1, 2):
                ta, td, fa, fd, true_order, fit_order = pairs[macrostate]
                item_n_states += len(true_order)

                # the reference state is (0, 0) in both models after the re-gauge: not estimated
                estimated = ~((ta[true_order] == 0) & (td[true_order] == 0) & (fa[fit_order] == 0) & (fd[fit_order] == 0))
                true_order, fit_order = true_order[estimated], fit_order[estimated]

                matched_true_alpha.extend(ta[true_order].tolist())
                matched_fit_alpha.extend(fa[fit_order].tolist())
                matched_true_delta.extend(td[true_order].tolist())
                matched_fit_delta.extend(fd[fit_order].tolist())
                matched_records.extend(
                    {"item": item, "macrostate": "C" if macrostate == 1 else "D",
                     "true_c": true_counts[0], "true_d": true_counts[1],
                     "true_alpha": float(ta[t]), "fit_alpha": float(fa[f]),
                     "true_delta": float(td[t]), "fit_delta": float(fd[f]),
                     "regauged": bool(anchor != own_anchor)}
                    for t, f in zip(true_order, fit_order)
                )

            logweight_rmse.append(np.sqrt(item_cost / item_n_states))

        true_alpha_values = np.asarray(matched_true_alpha, dtype=np.float64)
        fit_alpha_values = np.asarray(matched_fit_alpha, dtype=np.float64)
        true_delta_values = np.asarray(matched_true_delta, dtype=np.float64)
        fit_delta_values = np.asarray(matched_fit_delta, dtype=np.float64)

        params_rmse = {
            "alpha": _rmse(true_alpha_values, fit_alpha_values),
            "delta": _rmse(true_delta_values, fit_delta_values),
        }
        params_bias = {
            "alpha": _bias(true_alpha_values, fit_alpha_values),
            "delta": _bias(true_delta_values, fit_delta_values),
        }
        params_mae = {
            "alpha": _mae(true_alpha_values, fit_alpha_values),
            "delta": _mae(true_delta_values, fit_delta_values),
        }

        return {
            "params_rmse": params_rmse,
            "params_bias": params_bias,
            "params_mae": params_mae,
            "alpha_rmse": params_rmse["alpha"],
            "delta_rmse": params_rmse["delta"],
            "alpha_bias": params_bias["alpha"],
            "delta_bias": params_bias["delta"],
            "n_matched_items": int(true_model.n_items - len(skipped_items)),
            "micro_logweight_rmse": np.mean(logweight_rmse) if logweight_rmse else float("nan"),
            "skipped_items": skipped_items,
            "n_regauged_items": n_regauged,
            "matched_records": matched_records,
        }
    
    # Other models
    else:
        params_rmse = {
            name: _rmse(params_truth[name], params_model[name])
            for name in params_truth.keys()
        }
        params_bias = {
            name: _bias(params_truth[name], params_model[name])
            for name in params_truth.keys()
        }
        params_mae = {
            name: _mae(params_truth[name], params_model[name])
            for name in params_truth.keys()
        }

        return {
            "params_rmse": params_rmse,
            "params_bias": params_bias,
            "params_mae": params_mae,
        }
