import math

import numpy as np
from scipy.interpolate import BSpline
import torch
import torch.nn as nn

from mirm.model.core import MarginalBinaryIRTModel


class RaschModel(MarginalBinaryIRTModel):
    name = "Rasch"

    def __init__(self, n_items: int, n_quad: int = 41) -> None:
        super().__init__(n_items, n_quad)
        self.b = nn.Parameter(torch.zeros(n_items, dtype=torch.float64))

    @property
    def n_effective_params(self) -> int:
        return self.n_items

    @property
    def readable_params(self) -> dict[str, torch.Tensor]:
        return {
            "b": self.b,
        }

    def __str__(self) -> str:
        return self.name

    def p_correct_grid(self, theta_grid: torch.Tensor) -> torch.Tensor:
        z = theta_grid[None, :] - self.b[:, None]
        return torch.sigmoid(z)

    def p_correct_indexed(self, item_ids: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(theta - self.b[item_ids])

    
class TwoPLModel(MarginalBinaryIRTModel):
    name = "2PL"

    def __init__(self, n_items: int, n_quad: int = 41) -> None:
        super().__init__(n_items, n_quad)
        self.a_raw = nn.Parameter(torch.zeros(n_items, dtype=torch.float64))
        self.b = nn.Parameter(torch.zeros(n_items, dtype=torch.float64))

    @property
    def a(self) -> torch.Tensor:
        return nn.functional.softplus(self.a_raw) + torch.finfo(self.a_raw.dtype).eps

    @property
    def n_effective_params(self) -> int:
        return 2 * self.n_items

    @property
    def readable_params(self) -> dict[str, torch.Tensor]:
        return {
            "a": self.a,
            "b": self.b,
        }

    def __str__(self) -> str:
        return self.name

    def p_correct_grid(self, theta_grid: torch.Tensor) -> torch.Tensor:
        z = self.a[:, None] * (theta_grid[None, :] - self.b[:, None])
        return torch.sigmoid(z)

    def p_correct_indexed(self, item_ids: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        z = self.a[item_ids] * (theta - self.b[item_ids])
        return torch.sigmoid(z)


class ThreePLModel(MarginalBinaryIRTModel):
    name = "3PL"
    
    def __init__(self, n_items: int, n_quad: int = 41) -> None:
        super().__init__(n_items, n_quad)
        self.a_raw = nn.Parameter(torch.zeros(n_items, dtype=torch.float64))
        self.b = nn.Parameter(torch.zeros(n_items, dtype=torch.float64))
        self.c_raw = nn.Parameter(torch.full((n_items,), -2, dtype=torch.float64))

    @property
    def a(self) -> torch.Tensor:
        return nn.functional.softplus(self.a_raw) + torch.finfo(self.a_raw.dtype).eps

    @property
    def c(self) -> torch.Tensor:
        return torch.sigmoid(self.c_raw)

    @property
    def n_effective_params(self) -> int:
        return 3 * self.n_items

    @property
    def readable_params(self) -> dict[str, torch.Tensor]:
        return {
            "a": self.a,
            "b": self.b,
            "c": self.c,
        }

    def __str__(self) -> str:
        return self.name

    def p_correct_grid(self, theta_grid: torch.Tensor) -> torch.Tensor:
        core = torch.sigmoid(self.a[:, None] * (theta_grid[None, :] - self.b[:, None]))
        return self.c[:, None] + (1 - self.c[:, None]) * core

    def p_correct_indexed(self, item_ids: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        core = torch.sigmoid(self.a[item_ids] * (theta - self.b[item_ids]))
        return self.c[item_ids] + (1 - self.c[item_ids]) * core


class LPEModel(MarginalBinaryIRTModel):
    name = "LPE"
    
    def __init__(self, n_items: int, n_quad: int = 41) -> None:
        super().__init__(n_items, n_quad)
        self.a_raw = nn.Parameter(torch.zeros(n_items, dtype=torch.float64))
        self.b = nn.Parameter(torch.zeros(n_items, dtype=torch.float64))
        self.xi_raw = nn.Parameter(torch.zeros(n_items, dtype=torch.float64))

    @property
    def a(self) -> torch.Tensor:
        return nn.functional.softplus(self.a_raw) + torch.finfo(self.a_raw.dtype).eps

    @property
    def xi(self) -> torch.Tensor:
        return nn.functional.softplus(self.xi_raw) + torch.finfo(self.xi_raw.dtype).eps

    @property
    def n_effective_params(self) -> int:
        return 3 * self.n_items

    @property
    def readable_params(self) -> dict[str, torch.Tensor]:
        return {
            "a": self.a,
            "b": self.b,
            "xi": self.xi,
        }

    def __str__(self) -> str:
        return self.name

    def p_correct_grid(self, theta_grid: torch.Tensor) -> torch.Tensor:
        core = nn.functional.logsigmoid(self.a[:, None] * (theta_grid[None, :] - self.b[:, None]))
        return torch.exp(self.xi[:, None] * core)

    def p_correct_indexed(self, item_ids: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        core = nn.functional.logsigmoid(self.a[item_ids] * (theta - self.b[item_ids]))
        return torch.exp(self.xi[item_ids] * core)


class RHModel(MarginalBinaryIRTModel):
    name = "RH"
    
    def __init__(self, n_items: int, n_quad: int = 41) -> None:
        super().__init__(n_items, n_quad)
        self.a_raw = nn.Parameter(torch.zeros(n_items, dtype=torch.float64))
        self.b = nn.Parameter(torch.zeros(n_items, dtype=torch.float64))
        self.h1 = nn.Parameter(torch.zeros(n_items, dtype=torch.float64))

    @property
    def a(self) -> torch.Tensor:
        return nn.functional.softplus(self.a_raw) + torch.finfo(self.a_raw.dtype).eps

    @property
    def n_effective_params(self) -> int:
        return 3 * self.n_items

    @property
    def readable_params(self) -> dict[str, torch.Tensor]:
        return {
            "a": self.a,
            "b": self.b,
            "h1": self.h1,
        }

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def _normal_cdf(z: torch.Tensor) -> torch.Tensor:
        return torch.special.ndtr(z)

    def p_correct_grid(self, theta_grid: torch.Tensor) -> torch.Tensor:
        h_squared = 2 * torch.sigmoid(self.h1[:, None] * theta_grid[None, :])
        h_squared = h_squared.clamp_min(torch.finfo(theta_grid[None, :].dtype).tiny)

        h = torch.sqrt(h_squared)

        z = self.a[:, None] * (theta_grid[None, :] - self.b[:, None]) / h
        return self._normal_cdf(z)

    def p_correct_indexed(self, item_ids: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        h_squared = 2 * torch.sigmoid(self.h1[item_ids] * theta)
        h_squared = h_squared.clamp_min(torch.finfo(theta.dtype).tiny)

        h = torch.sqrt(h_squared)

        z = self.a[item_ids] * (theta - self.b[item_ids]) / h
        return self._normal_cdf(z)


class QuadModel(MarginalBinaryIRTModel):
    name = "3PLQ"
    
    def __init__(self, n_items: int, n_quad: int = 41) -> None:
        super().__init__(n_items, n_quad)
        self.q0 = nn.Parameter(torch.zeros(n_items, dtype=torch.float64))
        self.q1 = nn.Parameter(torch.ones(n_items, dtype=torch.float64))
        self.q2 = nn.Parameter(torch.zeros(n_items, dtype=torch.float64))

    @property
    def n_effective_params(self) -> int:
        return 3 * self.n_items

    @property
    def readable_params(self) -> dict[str, torch.Tensor]:
        return {
            "q0": self.q0,
            "q1": self.q1,
            "q2": self.q2,
        }

    def __str__(self) -> str:
        return self.name

    def p_correct_grid(self, theta_grid: torch.Tensor) -> torch.Tensor:
        logit = self.q0[:, None] + self.q1[:, None] * theta_grid[None, :] + self.q2[:, None] * theta_grid[None, :].square()
        return torch.sigmoid(logit)

    def p_correct_indexed(self, item_ids: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        logit = self.q0[item_ids] + self.q1[item_ids] * theta + self.q2[item_ids] * theta.square()
        return torch.sigmoid(logit)


class SplineModel(MarginalBinaryIRTModel):
    name = "Spline"
    
    def __init__(
        self,
        n_items: int,
        n_basis: int,
        n_quad: int = 41,
        degree: int = 3,
        lower: float = -4,
        upper: float = +4,
    ) -> None:
        super().__init__(n_items, n_quad)

        self.n_basis = n_basis
        self.degree = degree
        self.lower = lower
        self.upper = upper

        self.intercept = nn.Parameter(torch.zeros(n_items, dtype=torch.float64))

        initial_increment = 6 / n_basis
        initial_raw = math.log(math.expm1(initial_increment))
        self.gamma_raw = nn.Parameter(torch.full((n_items, n_basis), initial_raw, dtype=torch.float64))

        quad_basis = self._ispline_basis_numpy(self.theta_q.numpy(), n_basis, degree, lower, upper)
        self.register_buffer("quad_ispline_basis", torch.tensor(quad_basis, dtype=self.theta_q.dtype))

    @property
    def gamma(self) -> torch.Tensor:
        return nn.functional.softplus(self.gamma_raw)

    @property
    def n_effective_params(self) -> int:
        return self.n_items * (1 + self.n_basis)

    @property
    def readable_params(self) -> dict[str, torch.Tensor]:
        return {
            "intercept": self.intercept,
            "gamma": self.gamma,
        }

    def __str__(self) -> str:
        return f"{self.name}_{self.n_basis}"

    def _basis(self, theta: torch.Tensor) -> torch.Tensor:
        if theta is self.theta_q:
            return self.quad_ispline_basis

        values = self._ispline_basis_numpy(theta.detach().cpu().numpy(), self.n_basis, self.degree, self.lower, self.upper)
        return torch.tensor(values, dtype=theta.dtype, device=theta.device)

    @staticmethod
    def _ispline_basis_numpy(x: np.ndarray, n_basis: int, degree: int, lower: float, upper: float) -> np.ndarray:
        if n_basis < degree + 1:
            raise ValueError(f"degree={degree} requires at least {degree + 1} basis functions.")

        x = np.asarray(x, dtype=np.float64).reshape(-1)
        n_internal = n_basis - degree - 1

        if n_internal > 0:
            interior = np.linspace(lower, upper, n_internal + 2, dtype=np.float64)[1:-1]
        else:
            interior = np.empty(0, dtype=np.float64)

        knots = np.concatenate([np.repeat(lower, degree + 1), interior, np.repeat(upper, degree + 1)])

        clipped = np.clip(x, lower, upper)
        basis = np.empty((len(x), n_basis), dtype=np.float64)

        for index in range(n_basis):
            coefficients = np.zeros(n_basis)
            coefficients[index] = 1

            b_spline = BSpline(knots, coefficients, degree, extrapolate=False)
            antiderivative = b_spline.antiderivative()

            denominator = knots[index + degree + 1] - knots[index]
            values = (degree + 1) / denominator * (antiderivative(clipped) - antiderivative(lower))

            values = np.where(x <= lower, 0, values)
            values = np.where(x >= upper, 1, values)
            basis[:, index] = np.clip(values, 0, 1)

        return basis

    @torch.no_grad()
    def initialize_from_data(self, data: torch.Tensor) -> None:
        item_ids = data[:, 1].long()
        credits = data[:, 2].to(dtype=self.intercept.dtype)

        counts = torch.zeros_like(self.intercept)
        correct = torch.zeros_like(self.intercept)

        counts.scatter_add_(0, item_ids, torch.ones_like(credits))
        correct.scatter_add_(0, item_ids, credits)

        rates = ((correct + 0.5) / (counts + 1)).clamp(min=1e-4, max=1 - 1e-4)

        zero = torch.zeros(1, dtype=self.intercept.dtype)
        contribution = (self.gamma * self._basis(zero)[0]).sum(dim=1)

        self.intercept.copy_(torch.logit(rates) - contribution)

    def p_correct_grid(self, theta_grid: torch.Tensor) -> torch.Tensor:
        basis = self._basis(theta_grid)
        logit = self.intercept[:, None] + self.gamma @ basis.transpose(0, 1)
        return torch.sigmoid(logit)

    def p_correct_indexed(self, item_ids: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        basis = self._basis(theta)
        logit = self.intercept[item_ids] + (self.gamma[item_ids] * basis).sum(dim=1)
        return torch.sigmoid(logit)