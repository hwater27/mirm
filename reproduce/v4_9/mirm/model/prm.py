from typing import Callable, Optional, Sequence

import numpy as np
import torch

from mirm.model.mirm import BinaryMiRM, CellMiRM


class PRM:
    def __init__(
        self,
        n_items: int,
        worst_m: Optional[int] = None,
        form_masks: Optional[Sequence[torch.Tensor]] = None,
        n_nodes: Optional[int] = None,
        n_scan: int = 32,
        tol: float = 0.0,
    ) -> None:
        """Training constraint points: the Gauss-Hermite nodes of order n_nodes (None: the model's n_quad, i.e. the
        likelihood nodes) plus one movable cut per gap between neighbouring nodes, separately for the C and the D
        margin and for every constrained characteristic (a C cut carries only the C constraint, a D cut only the D
        constraint). The cuts start at the gap midpoints; update(model), called after every epoch, moves each cut to
        the lowest normalized margin of its own side in its gap, searched over n_scan interior points per gap, every
        crossing of two state lines of one item inside the gap and the cut's current position. The constraint vector
        has a fixed length, so a cut keeps its dual when it moves.
        tol is the training feasibility tolerance on the normalized margins."""
        self.n_items = n_items
        self.worst_m = worst_m
        self.n_nodes = n_nodes
        self.n_scan = n_scan
        self.tol = tol
        self.form_masks = None if form_masks is None else torch.as_tensor(form_masks, dtype=torch.float64)
        self.nodes = None        # (n_nodes,)
        self.cut_c = None        # (n_characteristics, n_nodes - 1)
        self.cut_d = None
        self.moves = 0

        if worst_m is not None and form_masks is not None:
            raise ValueError("Cannot combine worst_m and form_masks")
        if worst_m is not None:
            if worst_m > n_items:
                raise ValueError("worst_m cannot be greater than the number of items")
            if worst_m < 1:
                raise ValueError("worst_m must be a positive integer")
        if self.form_masks is not None:
            if self.form_masks.ndim != 2 or self.form_masks.shape[1] != n_items:
                raise ValueError("form_masks must have shape (n_forms, n_items)")

    @staticmethod
    def _characteristics(model: BinaryMiRM) -> list[Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]]:
        """(probability, discrimination) on theta of every constrained characteristic: the model itself, or every
        corner of the cell."""
        if isinstance(model, CellMiRM):
            return [lambda theta, k=k: model.corner_characteristics(theta, k) for k in range(4)]

        def own(theta):
            out = model.characteristics(theta)
            return out["probability"], out["discrimination"]
        return [own]

    def _setup(self, model: BinaryMiRM, n_chars: int) -> None:
        theta_q = model.theta_q
        if self.nodes is not None and self.nodes.device == theta_q.device and self.cut_c.shape[0] == n_chars:
            return
        x, _ = np.polynomial.hermite.hermgauss(self.n_nodes or model.n_quad)
        self.nodes = torch.tensor(np.sort(np.sqrt(2) * x), dtype=theta_q.dtype, device=theta_q.device)
        lo, hi = self.nodes[:-1], self.nodes[1:]
        mid = ((lo + hi) / 2)[None, :].repeat(n_chars, 1)
        self.cut_c, self.cut_d = mid.clone(), mid.clone()
        frac = torch.arange(1, self.n_scan + 1, dtype=theta_q.dtype, device=theta_q.device) / (self.n_scan + 1)
        self.scan = lo[:, None] + (hi - lo)[:, None] * frac[None, :]          # (n_gaps, n_scan) interior points

    def values(self, model: BinaryMiRM) -> torch.Tensor:
        if model.n_items != self.n_items:
            raise ValueError("Model must have the same number of items as the PRM")

        chars = self._characteristics(model)
        self._setup(model, len(chars))
        n_gaps = self.cut_c.shape[1]
        node_parts, cut_parts = [], []
        for k, char in enumerate(chars):
            margins = self._margins_from(*char(self.nodes))
            node_parts += [margins["c_norm"].reshape(-1), margins["d_norm"].reshape(-1)]
            at_cuts = self._margins_from(*char(torch.cat([self.cut_c[k], self.cut_d[k]])))
            cut_parts += [at_cuts["c_norm"][:n_gaps], at_cuts["d_norm"][n_gaps:]]
        return -torch.cat(node_parts + cut_parts)

    @torch.no_grad()
    def _crossings(self, model: BinaryMiRM) -> torch.Tensor:
        """Every crossing of two active state lines (-alpha theta + delta) of one item inside the node range."""
        a, d, mask = model.alpha_raw, model.delta_raw, model.state_mask
        da = a[:, :, None] - a[:, None, :]
        dd = d[:, :, None] - d[:, None, :]
        upper = torch.ones(mask.shape[1], mask.shape[1], dtype=torch.bool, device=mask.device).triu(diagonal=1)
        pair = mask[:, :, None] & mask[:, None, :] & (da.abs() > 1e-10) & upper[None]
        theta = dd[pair] / da[pair]
        return theta[torch.isfinite(theta) & (theta > self.nodes[0]) & (theta < self.nodes[-1])]

    @torch.no_grad()
    def update(self, model: BinaryMiRM) -> int:
        """Move every cut to the lowest normalized margin of its side in its gap; returns the number of moved cuts."""
        chars = self._characteristics(model)
        self._setup(model, len(chars))
        n_gaps = self.cut_c.shape[1]
        cross = self._crossings(model)
        gap_cross = (torch.searchsorted(self.nodes, cross, right=True) - 1).clamp(0, n_gaps - 1)
        gaps = torch.arange(n_gaps, device=self.nodes.device)
        moved = 0
        for k, char in enumerate(chars):
            cand = torch.cat([self.scan.reshape(-1), cross, self.cut_c[k], self.cut_d[k]])
            gap = torch.cat([gaps.repeat_interleave(self.scan.shape[1]), gap_cross, gaps, gaps])
            margins = self._margins_from(*char(cand))
            for values, cuts in ((margins["c_norm"], self.cut_c[k]), (margins["d_norm"], self.cut_d[k])):
                best = torch.full((n_gaps,), torch.inf, dtype=values.dtype, device=values.device)
                best = best.scatter_reduce(0, gap, values, "amin")
                hit = values == best[gap]
                first = torch.full((n_gaps,), len(cand), dtype=torch.long, device=values.device)
                first = first.scatter_reduce(0, gap[hit], torch.nonzero(hit).flatten(), "amin")
                new = cand[first]
                moved += int((new - cuts).abs().gt(1e-12).sum())
                cuts.copy_(new)
        self.moves += moved
        return moved

    def _aggregate(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Full-bank PRM
        if self.worst_m is None and self.form_masks is None:
            return values.sum(0), values.abs().sum(0)

        # Worst-m PRM
        elif self.form_masks is None:
            selected = torch.topk(values, k=self.worst_m, dim=0, largest=False).values
            return selected.sum(0), selected.abs().sum(0)

        # Form-specific PRM
        elif self.worst_m is None:
            masks = self.form_masks.to(values.device)
            form_margins = masks @ values
            form_scales = masks @ values.abs()

            idx = form_margins.amin(dim=1).argmin()
            return form_margins[idx], form_scales[idx]

    @staticmethod
    def _normalize(margin: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))

        return torch.where(scale > 0, margin / safe_scale, torch.zeros_like(margin))

    def margins(self, model: BinaryMiRM, theta: torch.Tensor) -> dict[str, torch.Tensor]:
        if model.n_items != self.n_items:
            raise ValueError("Model must have the same number of items as the PRM")

        characteristics = model.characteristics(theta)
        return self._margins_from(characteristics["probability"], characteristics["discrimination"])

    def _margins_from(self, p: torch.Tensor, a: torch.Tensor) -> dict[str, torch.Tensor]:
        c_margin, c_scale = self._aggregate((1 - p) * a)
        d_margin, d_scale = self._aggregate(p * a)

        return {
            "c_margin": c_margin,
            "d_margin": d_margin,
            "c_norm": self._normalize(c_margin, c_scale),
            "d_norm": self._normalize(d_margin, d_scale),
        }

    @staticmethod
    def _item_lipschitz_bounds(model: BinaryMiRM) -> torch.Tensor:
        alpha_max = model.alpha_ordered.nan_to_num(nan=-torch.inf).max(dim=1).values

        return alpha_max.square() / 4

    def _lipschitz_bound(self, model: BinaryMiRM) -> float:
        item_bounds = self._item_lipschitz_bounds(model)

        if self.worst_m is None and self.form_masks is None:
            bound = item_bounds.sum()
        elif self.form_masks is None:
            bound = torch.topk(item_bounds, k=self.worst_m, largest=True).values.sum()
        elif self.worst_m is None:
            bound = (self.form_masks @ item_bounds).amax()

        return float(bound.detach().item())

    @torch.no_grad()
    def certify(
        self,
        model: BinaryMiRM,
        initial_nodes: int = 1601,
        tolerance: float = 1e-3,
        max_depth: int = 12,
    ) -> dict:
        if model.n_items != self.n_items:
            raise ValueError("Model must have the same number of items as the PRM")

        certification_grid = torch.linspace(
            float(model.theta_q.min()),
            float(model.theta_q.max()),
            initial_nodes,
            dtype=model.theta_q.dtype,
        )
        lipschitz_bound = self._lipschitz_bound(model)
        start = self.margins(model, certification_grid)

        def _side(y0: torch.Tensor, key: str) -> tuple:
            xs, ys = certification_grid.tolist(), y0.tolist()
            cells = [(xs[i], xs[i + 1], ys[i], ys[i + 1], 0) for i in range(len(xs) - 1)]
            evaluations, viol, unres = len(xs), 0, 0
            min_m = float(y0.min())

            while cells:
                split = []
                for xa, xb, ya, yb, depth in cells:
                    thr = (xb - xa) * lipschitz_bound / 2 - tolerance
                    if ya >= thr and yb >= thr:
                        continue
                    if ya < -tolerance or yb < -tolerance:
                        viol += 1
                    elif depth >= max_depth:
                        unres += 1
                    else:
                        split.append((xa, xb, ya, yb, depth))
                if not split:
                    break

                mid = torch.tensor([(xa + xb) / 2 for xa, xb, _, _, _ in split], dtype=certification_grid.dtype)
                ym = torch.cat([self.margins(model, chunk)[key] for chunk in mid.split(4096)])
                evaluations += len(mid)
                min_m = min(min_m, float(ym.min()))
                cells = [
                    cell
                    for (xa, xb, ya, yb, depth), xm, y in zip(split, mid.tolist(), ym.tolist())
                    for cell in ((xa, xm, ya, y, depth + 1), (xm, xb, y, yb, depth + 1))
                ]

            status = "violated" if viol else "inconclusive" if unres else "certified"
            return status, min_m, evaluations, viol, unres

        c_status, min_c, e_c, v_c, u_c = _side(start["c_margin"], "c_margin")
        d_status, min_d, e_d, v_d, u_d = _side(start["d_margin"], "d_margin")
        required = (c_status, d_status)
        status = (
            "violated" if "violated" in required
            else "inconclusive" if "inconclusive" in required
            else "certified"
        )

        return {
            "status": status,
            "c_status": c_status,
            "d_status": d_status,
            "min_c_margin": min_c,
            "min_d_margin": min_d,
            "evaluations": e_c + e_d,
            "violation_intervals": v_c + v_d,
            "unresolved_intervals": u_c + u_d,
            "lipschitz_bound": lipschitz_bound,
        }
