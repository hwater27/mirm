from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def response_design(x: torch.Tensor, n_items: int, n_students: Optional[int] = None,
                    dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Sparse (n_students, 2 * n_items) design: column 2j+1 holds the credit of item j, column 2j its complement."""
    students = x[:, 0].long()
    items = x[:, 1].long()
    credits = x[:, 2].to(dtype)
    n_students = int(students.max().item()) + 1 if n_students is None else n_students

    rows = torch.cat([students, students])
    cols = torch.cat([2 * items + 1, 2 * items])
    values = torch.cat([credits, 1 - credits])
    keep = values != 0

    return torch.sparse_coo_tensor(
        torch.stack([rows[keep], cols[keep]]), values[keep], (n_students, 2 * n_items), check_invariants=True
    ).coalesce()


def plan_designs(dataset, n_items: int, width: int, chunk_students: int = 0, chunk_mem_gb: float = 12.0) -> list:
    """Whole-student chunks of the response design (no response minibatches; the MML objective is unchanged).

    chunk_students > 0 fixes the chunk size, 0 chunks only when the (N, width) working set exceeds chunk_mem_gb,
    and < 0 never chunks. Returns [(design, n_observations), ...] and caches the plan on the dataset."""
    key = (n_items, width, chunk_students, chunk_mem_gb)
    cache = dataset.__dict__.setdefault("_design_cache", {})
    if key in cache:
        return cache[key]

    n_students = dataset.n_students
    if chunk_students == 0:
        per_student = width * torch.finfo(torch.float64).bits // 8 * 6
        chunk_students = max(1, int(chunk_mem_gb * 1e9 / per_student))
    if chunk_students < 0 or chunk_students >= n_students:
        plan = [(response_design(dataset.data, n_items, n_students), len(dataset))]
    else:
        order = dataset.student_ids.argsort(stable=True)
        data = dataset.data[order]
        students = data[:, 0].long().contiguous()
        plan = []
        for start in range(0, n_students, chunk_students):
            lo, hi = torch.searchsorted(students, torch.tensor([start, start + chunk_students], dtype=students.dtype)).tolist()
            if hi == lo:
                continue
            part = data[lo:hi].clone()
            part[:, 0] -= start
            plan.append((response_design(part, n_items, min(chunk_students, n_students - start)), hi - lo))
    cache[key] = plan
    return plan


class IRTModel(nn.Module):
    def __init__(self, n_items:int) -> None:
        super().__init__()
        self.n_items = n_items

    @property
    def n_effective_params(self) -> int:
        raise NotImplementedError("Subclasses must implement n_effective_params().")

    @property
    def readable_params(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError("Subclasses must implement readable_params().")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward().")


class BinaryIRTModel(IRTModel):
    def __init__(self, n_items:int) -> None:
        super().__init__(n_items)

    def p_correct_grid(self, theta_grid: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement p_correct_grid().")

    def p_correct_indexed(self, item_ids: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement p_correct_indexed().")

    def ll_per_obs(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        student_ids = x[:, 0].long()
        item_ids = x[:, 1].long()
        credits = x[:, 2].to(theta.dtype)

        p = self.p_correct_indexed(item_ids, theta[student_ids])

        eps = torch.finfo(theta.dtype).eps
        p = p.clamp(eps, 1 - eps)

        ll = credits * torch.log(p) + (1 - credits) * torch.log1p(-p)
        return ll

    @torch.no_grad()
    def item_information_grid(self, theta_grid: torch.Tensor, step: Optional[float] = None) -> torch.Tensor:
        if step is None:
            eps = torch.finfo(theta_grid.dtype).eps
            step = float(eps ** (1/3))

        p = self.p_correct_grid(theta_grid)
        p_plus = self.p_correct_grid(theta_grid + step)
        p_minus = self.p_correct_grid(theta_grid - step)

        derivative = (p_plus - p_minus) / (2 * step)

        denominator = p * (1 - p)
        denominator = denominator.clamp_min(torch.finfo(p.dtype).eps)

        return derivative.square() / denominator


class MarginalBinaryIRTModel(BinaryIRTModel):
    def __init__(self, n_items: int, n_quad: int = 41) -> None:
        super().__init__(n_items)

        self.n_quad = n_quad
        x_q, w_q = np.polynomial.hermite.hermgauss(n_quad)
        self.register_buffer("theta_q", torch.tensor(np.sqrt(2) * x_q, dtype=torch.float64))
        self.register_buffer("tilde_q", torch.tensor(w_q / np.sqrt(np.pi), dtype=torch.float64))
        self.register_buffer("log_quad_weights", torch.log(self.tilde_q))

    def _cnt_n_students(self, x: torch.Tensor, n_students: Optional[int] = None) -> int:
        return int(x[:, 0].max().item()) + 1 if n_students is None else n_students

    def log_table(self, theta_grid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(2 * n_items, G): row 2j = log(1 - p_j), row 2j+1 = log p_j on the grid (quadrature by default)."""
        p = self.p_correct_grid(self.theta_q if theta_grid is None else theta_grid)
        eps = torch.finfo(p.dtype).eps
        p = p.clamp(eps, 1 - eps)
        return torch.stack([torch.log1p(-p), torch.log(p)], dim=1).reshape(2 * self.n_items, -1)

    def ll_person_grid(self, x: torch.Tensor, n_students: Optional[int] = None, theta_grid: Optional[torch.Tensor] = None) -> torch.Tensor:
        n_students = self._cnt_n_students(x, n_students)
        design = response_design(x, self.n_items, n_students)
        return torch.sparse.mm(design, self.log_table(theta_grid))

    def sparse_nll(self, design: torch.Tensor) -> torch.Tensor:
        ll = torch.sparse.mm(design, self.log_table())
        return -torch.logsumexp(ll + self.log_quad_weights[None, :], dim=1).sum()

    def marginal_nll(self, x: torch.Tensor, n_students: Optional[int] = None) -> torch.Tensor:
        return self.sparse_nll(response_design(x, self.n_items, self._cnt_n_students(x, n_students)))

    @torch.no_grad()
    def posterior_grid(self, x: torch.Tensor, n_students: Optional[int] = None) -> torch.Tensor:
        ll = self.ll_person_grid(x, n_students)
        return torch.softmax(ll + self.log_quad_weights[None, :], dim=1)

    @torch.no_grad()
    def item_expected_nll(self, x: torch.Tensor, posterior=None) -> torch.Tensor:
        if posterior is None:
            posterior = self.posterior_grid(x)
        design = response_design(x, self.n_items, posterior.shape[0])
        counts = torch.sparse.mm(design.t(), posterior)
        return -(counts * self.log_table()).reshape(self.n_items, -1).sum(dim=1)

    def eap_theta(self, x: torch.Tensor, n_students: Optional[int] = None) -> torch.Tensor:
        posterior = self.posterior_grid(x, n_students)
        return (posterior * self.theta_q[None, :]).sum(dim=1)

    @torch.no_grad()
    def map_theta(self, x: torch.Tensor, n_students: Optional[int] = None,
        theta_min: float = -4,
        theta_max: float = +4,
        n_grid: int = 401,
    ) -> torch.Tensor:
        theta_grid = torch.linspace(theta_min, theta_max, n_grid, dtype=self.theta_q.dtype)

        ll = self.ll_person_grid(x, n_students, theta_grid)
        log_prior = -0.5 * theta_grid.square()

        map_indices = (ll + log_prior[None, :]).argmax(dim=1)

        return theta_grid[map_indices]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.marginal_nll(x)
            