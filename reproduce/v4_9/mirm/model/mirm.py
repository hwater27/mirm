from typing import Sequence, Optional

import torch
import torch.nn as nn

from mirm.model.core import MarginalBinaryIRTModel


class BinaryMiRM(MarginalBinaryIRTModel):
    name = "MiRM"

    def __init__(
        self,
        n_items: int,
        c: int | Sequence[int],
        d: int | Sequence[int],
        n_quad: int = 41,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__(n_items, n_quad)

        c = torch.as_tensor(c, dtype=torch.long)
        d = torch.as_tensor(d, dtype=torch.long)

        if c.ndim == 0 and d.ndim == 0:
            self._tag = f" ({c},{d})"
        else:
            self._tag = "_per_item"

        if c.ndim == 0:
            c = c.repeat(n_items)
        if d.ndim == 0:
            d = d.repeat(n_items)

        if len(c) != n_items or len(d) != n_items:
            raise ValueError("c and d must contain one count per item.")
        if (c < 1).any() or (d < 1).any():
            raise ValueError("All c and d values must be positive.")

        self.register_buffer("c", c)
        self.register_buffer("d", d)

        self.c_max = int(self.c.max().item())
        self.d_max = int(self.d.max().item())
        self.n_states = self.c_max + self.d_max

        self.alpha_raw = nn.Parameter(torch.randn(n_items, self.n_states, dtype=torch.float64) * init_scale)
        self.delta_raw = nn.Parameter(torch.randn(n_items, self.n_states, dtype=torch.float64) * init_scale)

        state_mask = torch.cat(
            [
                torch.arange(self.c_max)[None, :] < self.c[:, None],
                torch.arange(self.d_max)[None, :] < self.d[:, None],
            ],
            dim=1
        ).bool()
        self.register_buffer("state_mask", state_mask)

        macrostate = torch.cat(
            [
                torch.full((self.n_items, self.c_max), 1, dtype=torch.long),
                torch.full((self.n_items, self.d_max), 2, dtype=torch.long),
            ],
            dim=1
        )
        macrostate = macrostate.masked_fill(~self.state_mask, 0)
        self.register_buffer("macrostate", macrostate)

    @property
    def ground_idx(self) -> torch.Tensor:
        active_alpha = self.alpha_raw.masked_fill(~self.state_mask, torch.inf)
        min_alpha = active_alpha.amin(dim=1, keepdim=True)

        candidates = self.state_mask & (self.alpha_raw == min_alpha)
        candidate_delta = self.delta_raw.masked_fill(~candidates, -torch.inf)

        return candidate_delta.argmax(dim=1, keepdim=True)

    @property
    def alpha(self) -> torch.Tensor:
        reference = self.alpha_raw.gather(dim=1, index=self.ground_idx)
        return self.alpha_raw - reference

    @property
    def delta(self) -> torch.Tensor:
        reference = self.delta_raw.gather(dim=1, index=self.ground_idx)
        return self.delta_raw - reference

    @property
    def state_order(self) -> torch.Tensor:
        delta_key = self.delta.masked_fill(~self.state_mask, -torch.inf)
        delta_order = delta_key.argsort(dim=1, descending=True, stable=True)

        alpha_key = self.alpha.gather(dim=1, index=delta_order)
        mask_after_delta = self.state_mask.gather(dim=1, index=delta_order)
        alpha_key = alpha_key.masked_fill(~mask_after_delta, torch.inf)

        alpha_order = alpha_key.argsort(dim=1, stable=True)
        return delta_order.gather(dim=1, index=alpha_order)

    @property
    def macrostate_ordered(self) -> torch.Tensor:
        return self.macrostate.gather(dim=1, index=self.state_order)

    @property
    def alpha_ordered(self) -> torch.Tensor:
        ordered = self.alpha.gather(dim=1, index=self.state_order)
        return ordered.masked_fill(self.macrostate_ordered == 0, torch.nan)

    @property
    def delta_ordered(self) -> torch.Tensor:
        ordered = self.delta.gather(dim=1, index=self.state_order)
        return ordered.masked_fill(self.macrostate_ordered == 0, torch.nan)

    @property
    def n_effective_params(self) -> int:
        return int(2 * (self.c + self.d - 1).sum().item())

    @property
    def readable_params(self) -> dict[str, torch.Tensor]:
        return {
            "alpha": self.alpha_ordered,
            "delta": self.delta_ordered,
        }

    def __str__(self) -> str:
        return f"{self.name}{self._tag}"

    def _logits_grid(self, theta: torch.Tensor) -> torch.Tensor:
        logits = -self.alpha_raw[:, :, None] * theta[None, None, :] + self.delta_raw[:, :, None]
        return logits.masked_fill(~self.state_mask[:, :, None], -torch.inf)

    def p_correct_grid(self, theta_grid: torch.Tensor) -> torch.Tensor:
        logits = self._logits_grid(theta_grid)

        log_z_c = torch.logsumexp(logits[:, :self.c_max], dim=1)
        log_z_d = torch.logsumexp(logits[:, self.c_max:], dim=1)

        return torch.sigmoid(log_z_c - log_z_d)

    def p_correct_indexed(self, item_ids: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha[item_ids]
        delta = self.delta[item_ids]
        mask = self.state_mask[item_ids]

        logits = -alpha * theta[:, None] + delta
        logits = logits.masked_fill(~mask, -torch.inf)

        log_z_c = torch.logsumexp(logits[:, :self.c_max], dim=1)
        log_z_d = torch.logsumexp(logits[:, self.c_max:], dim=1)

        return torch.sigmoid(log_z_c - log_z_d)

    def item_information_grid(self, theta_grid: torch.Tensor, step: Optional[float] = None) -> torch.Tensor:
        return self.characteristics(theta_grid)["information"]

    def _components(self, theta: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self._logits_grid(theta)

        logits_c = logits[:, :self.c_max]
        logits_d = logits[:, self.c_max:]

        log_z_c = torch.logsumexp(logits_c, dim=1)
        log_z_d = torch.logsumexp(logits_d, dim=1)

        rho_c = torch.softmax(logits_c, dim=1)
        rho_d = torch.softmax(logits_d, dim=1)

        alpha_c = self.alpha_raw[:, :self.c_max, None]
        alpha_d = self.alpha_raw[:, self.c_max:, None]

        mean_c = (rho_c * alpha_c).sum(dim=1)
        mean_d = (rho_d * alpha_d).sum(dim=1)

        var_c = (rho_c * (alpha_c - mean_c[:, None]).square()).sum(dim=1)
        var_d = (rho_d * (alpha_d - mean_d[:, None]).square()).sum(dim=1)

        return {
            "log_z_c": log_z_c,
            "log_z_d": log_z_d,
            "rho_c": rho_c,
            "rho_d": rho_d,
            "mean_alpha_c": mean_c,
            "mean_alpha_d": mean_d,
            "var_alpha_c": var_c,
            "var_alpha_d": var_d,
        }

    def characteristics(self, theta: torch.Tensor, disc_eps: float = 1e-3) -> dict[str, torch.Tensor]:
        components = self._components(theta)

        logit = components["log_z_c"] - components["log_z_d"]
        prob = torch.sigmoid(logit)

        disc = components["mean_alpha_d"] - components["mean_alpha_c"]
        acc = components["var_alpha_c"] - components["var_alpha_d"]

        valid = disc.abs() > disc_eps
        safe_disc = torch.where(valid, disc, torch.ones_like(disc))
        diff = torch.where(valid, theta[None, :] - logit / safe_disc, torch.full_like(logit, torch.nan))

        info = prob * (1 - prob) * disc.square()

        return {
            "logit": logit,
            "probability": prob,
            "discrimination": disc,
            "acceleration": acc,
            "difficulty": diff,
            "information": info,
            **components,
        }

class CellMiRM(BinaryMiRM):
    """Per-item 2x2 cell of MiRM configurations used by the (c,d) selection.

    Item j carries a cell with lower-left corner (c_j, d_j); its four corners LL = (c, d), UL = (c+1, d),
    LU = (c, d+1) and UU = (c+1, d+1) are virtual configurations that share the states of the item (slots up to
    cap_c / cap_d). The training objective is the equal-weight mean of the four corner marginal NLLs.
    """
    CORNERS = ("LL", "UL", "LU", "UU")
    DIRECTIONS = ((0, 0), (1, 0), (0, 1), (1, 1))

    def __init__(
        self,
        n_items: int,
        cap_c: int,
        cap_d: int,
        c0: int = 1,
        d0: int = 1,
        n_quad: int = 41,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__(n_items, cap_c, cap_d, n_quad, init_scale)

        self.cap_c, self.cap_d = cap_c, cap_d
        self.init_scale = init_scale
        self._tag = "_cell"
        self.register_buffer("cell_c", torch.full((n_items,), c0, dtype=torch.long))
        self.register_buffer("cell_d", torch.full((n_items,), d0, dtype=torch.long))

        with torch.no_grad():  # states beyond the lower-left corner start light
            self.delta_raw[:, :cap_c] -= 4.0 * (torch.arange(cap_c)[None, :] >= self.cell_c[:, None])
            self.delta_raw[:, cap_c:] -= 4.0 * (torch.arange(cap_d)[None, :] >= self.cell_d[:, None])

    def virtual(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(4, n_items) state counts of the four corners."""
        c = torch.stack([(self.cell_c + dc).clamp(1, self.cap_c) for dc, _ in self.DIRECTIONS])
        d = torch.stack([(self.cell_d + dd).clamp(1, self.cap_d) for _, dd in self.DIRECTIONS])
        return c, d

    def corner_p_grid(self, theta_grid: torch.Tensor) -> torch.Tensor:
        """(4, n_items, G) correct-response probability of every corner."""
        logits = self._logits_grid(theta_grid)
        cum_c = torch.logcumsumexp(logits[:, :self.cap_c], dim=1)
        cum_d = torch.logcumsumexp(logits[:, self.cap_c:], dim=1)

        vc, vd = self.virtual()
        G = theta_grid.numel()
        log_z_c = cum_c.gather(1, (vc.T - 1)[:, :, None].expand(-1, -1, G))
        log_z_d = cum_d.gather(1, (vd.T - 1)[:, :, None].expand(-1, -1, G))
        return torch.sigmoid(log_z_c - log_z_d).permute(1, 0, 2)

    def corner_log_tables(self) -> torch.Tensor:
        """(4, 2 * n_items, Q) log-probability tables of the corners on the quadrature (see core.log_table)."""
        p = self.corner_p_grid(self.theta_q)
        eps = torch.finfo(p.dtype).eps
        p = p.clamp(eps, 1 - eps)
        return torch.stack([torch.log1p(-p), torch.log(p)], dim=2).reshape(4, 2 * self.n_items, -1)

    def sparse_nll(self, design: torch.Tensor) -> torch.Tensor:
        tables = self.corner_log_tables()
        nll = [
            -torch.logsumexp(torch.sparse.mm(design, tables[k]) + self.log_quad_weights[None, :], dim=1).sum()
            for k in range(4)
        ]
        return torch.stack(nll).mean()

    def corner_characteristics(self, theta: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Probability and local discrimination of corner k on theta (for the corner PRM)."""
        vc, vd = self.virtual()
        mask_c = torch.arange(self.cap_c, device=vc.device)[None, :] < vc[k][:, None]
        mask_d = torch.arange(self.cap_d, device=vd.device)[None, :] < vd[k][:, None]

        logits = self._logits_grid(theta)
        logits_c = logits[:, :self.cap_c].masked_fill(~mask_c[:, :, None], -torch.inf)
        logits_d = logits[:, self.cap_c:].masked_fill(~mask_d[:, :, None], -torch.inf)

        mean_c = (torch.softmax(logits_c, dim=1) * self.alpha_raw[:, :self.cap_c, None]).sum(dim=1)
        mean_d = (torch.softmax(logits_d, dim=1) * self.alpha_raw[:, self.cap_c:, None]).sum(dim=1)
        logit = torch.logsumexp(logits_c, dim=1) - torch.logsumexp(logits_d, dim=1)
        return torch.sigmoid(logit), mean_d - mean_c

    @torch.no_grad()
    def reinit_slot(self, side: str, item: int, slot: int, generator: torch.Generator) -> None:
        """Light re-initialisation of a slot the cell exposes for the first time."""
        offset, width = (0, self.cap_c) if side == "C" else (self.cap_c, self.cap_d)
        if slot >= width:
            return
        active = slice(offset, offset + slot)
        self.alpha_raw[item, offset + slot] = (
            self.alpha_raw[item, active].mean() + 0.3 * float(torch.randn(1, generator=generator))
        )
        self.delta_raw[item, offset + slot] = self.delta_raw[item, active].max() - 4.0

    @torch.no_grad()
    def trimmed(self, c: torch.Tensor, d: torch.Tensor) -> BinaryMiRM:
        """Per-item BinaryMiRM with the selected (c_j, d_j), initialised from the shared states."""
        model = BinaryMiRM(self.n_items, c.clone(), d.clone(), self.n_quad, self.init_scale)
        model.alpha_raw[:, :model.c_max] = self.alpha_raw[:, :model.c_max]
        model.delta_raw[:, :model.c_max] = self.delta_raw[:, :model.c_max]
        model.alpha_raw[:, model.c_max:] = self.alpha_raw[:, self.cap_c:self.cap_c + model.d_max]
        model.delta_raw[:, model.c_max:] = self.delta_raw[:, self.cap_c:self.cap_c + model.d_max]
        return model
