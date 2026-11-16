"""Per-item MiRM configuration (c, d) selection with a moving 2x2 cell whose lower-left corner is the selection.

Each item has a 2x2 cell [a, a+1] x [b, b+1]; its selection is always the lower-left corner (a, b), and the cell and
the selection move together. Start (1, 1). The four corners LL = (a, b), UL = (a+1, b), LU = (a, b+1) and
UU = (a+1, b+1) are fitted together (CellMiRM: shared states, equal weights) and PRM is imposed on every corner. Cells
never move down, and the lower-left corner stays at most cap - 1, so all four corners lie within the cap.

After every round, each item's table under one pooled posterior (geometric by default) gives c = UL - LL, d = LU - LL
and x = LL - UU, and the background-removed table c~ = c - mean(c), d~ = d - mean(d), x~ = x - mean(x) (means over
all items, items at the cap included). A move needs the same rule to hold on both tables:
  C addition   -c > 0 and d + x > 0   (the upper C state gains in both contexts, d and d + 1)    -> a + 1
  D addition   -d > 0 and c + x > 0   (the upper D state gains in both contexts, c and c + 1)    -> b + 1
  jump         0 < min(c, d) < x, with no C and no D addition                                 -> (a + 1, b + 1)
An addition needs a < cap - 1 (b < cap - 1 for D) and a jump needs both. Both axes may grow in the same round; a jump
item adds on neither axis on either table, so it never also grows an axis. States shown by the cell for the first time
start light. Every round uses one early-stopping tolerance (train_options["early_stop_tol"], from
--select-early-stop-tol).

Stopping: stable_rounds consecutive rounds without a change -> converged; the search also stops when the epoch budget
is spent (converged = False).
"""
from typing import Any, Optional

import torch

from mirm.data.datasets import ScoredIRTDataset
from mirm.model import CellMiRM
from mirm.model.core import plan_designs
from mirm.eval.train import trainer

LL, UL, LU, UU = 0, 1, 2, 3
FLAGS = ("grow_c", "grow_d", "jump", "raw_add_c", "raw_add_d", "raw_jump", "cen_add_c", "cen_add_d", "cen_jump")


@torch.no_grad()
def corner_table(model: CellMiRM, designs: list, pool: str = "geo") -> torch.Tensor:
    """(J, 4) expected NLL of every item at every corner under one pooled posterior.

    pool "arith": mean of the four corner posteriors; "geo": softmax of the mean node log-likelihood.
    Posterior pseudo-counts are accumulated as design^T @ posterior (rows 2j: incorrect, 2j+1: correct)."""
    tables = model.corner_log_tables()                                              # (4, 2J, Q)
    counts = torch.zeros(tables.shape[1:], dtype=torch.float64)
    for design, _ in designs:
        ll = torch.stack([torch.sparse.mm(design, tables[k]) for k in range(4)]) + model.log_quad_weights
        pooled = torch.softmax(ll, dim=-1).mean(0) if pool == "arith" else torch.softmax(ll.mean(0), dim=-1)
        counts += torch.sparse.mm(design.t(), pooled)
    return -(counts[None] * tables).reshape(4, model.n_items, -1).sum(-1).T         # (J, corner)


def _rule(c: torch.Tensor, d: torch.Tensor, x: torch.Tensor) -> dict:
    """Additions and jump on one table (c = UL - LL, d = LU - LL, x = LL - UU), no cap condition."""
    add_c = (-c > 0) & (d + x > 0)
    add_d = (-d > 0) & (c + x > 0)
    lower = torch.minimum(c, d)
    return {"add_c": add_c, "add_d": add_d, "jump": (lower > 0) & (lower < x) & ~add_c & ~add_d}


def decide(model: CellMiRM, Tm: torch.Tensor) -> dict:
    """New lower-left corners by the rule of the module docstring: the pooled and the background-removed table agree."""
    cell_c, cell_d = model.cell_c, model.cell_d
    c, d, x = Tm[:, UL] - Tm[:, LL], Tm[:, LU] - Tm[:, LL], Tm[:, LL] - Tm[:, UU]
    raw = _rule(c, d, x)
    cen = _rule(c - c.mean(), d - d.mean(), x - x.mean())
    room_c, room_d = cell_c < model.cap_c - 1, cell_d < model.cap_d - 1
    grow_c = raw["add_c"] & cen["add_c"] & room_c
    grow_d = raw["add_d"] & cen["add_d"] & room_d
    jump = raw["jump"] & cen["jump"] & room_c & room_d
    return {
        "new_c": cell_c + (grow_c | jump).long(),
        "new_d": cell_d + (grow_d | jump).long(),
        "grow_c": grow_c, "grow_d": grow_d, "jump": jump,
        **{f"raw_{k}": v for k, v in raw.items()},
        **{f"cen_{k}": v for k, v in cen.items()},
        "mean": (float(c.mean()), float(d.mean()), float(x.mean())),
    }


def histogram(c: torch.Tensor, d: torch.Tensor) -> str:
    counts: dict[tuple[int, int], int] = {}
    for a, b in zip(c.tolist(), d.tolist()):
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return " ".join(f"({a},{b}):{n}" for (a, b), n in sorted(counts.items(), key=lambda kv: -kv[1]))


def select_configuration(
    dataset: ScoredIRTDataset,
    n_quad: int = 41,
    cap: int = 5,
    max_epochs: int = 3000,
    round_epochs: int = 0,
    pool: str = "geo",
    stable_rounds: int = 2,
    train_options: Optional[dict[str, Any]] = None,
    prm_options: Optional[dict[str, Any]] = None,
    label: str = "",
    verbose: bool = False,
    progress=None,
    seed: int = 27,
) -> dict[str, Any]:
    """Run the cell rounds and return the selected per-item (c, d), the round log and the final cell model.

    progress: optional callable taking a short status string (e.g., the parallel runner's report), called at the
    start of every round; verbose prints one line per round instead (for single-process use)."""
    if cap < 2:
        raise ValueError("cap must be at least 2 (the cell spans two configurations per axis)")
    train_options = dict(train_options or {})
    model = CellMiRM(dataset.n_items, cap, cap, 1, 1, n_quad)
    designs = plan_designs(dataset, model.n_items, 4 * n_quad, train_options.get("chunk_students", 0),
                           train_options.get("chunk_mem_gb", 12.0))
    generator = torch.Generator().manual_seed(seed)

    seen_c, seen_d = model.cell_c.clone(), model.cell_d.clone()
    used, rounds, stable, converged = 0, [], 0, False

    while used < max_epochs:
        if progress is not None:
            progress(f"round {len(rounds) + 1} | {used}/{max_epochs} ep")
        budget = max_epochs - used if not round_epochs else min(max_epochs - used, round_epochs)
        fit = trainer(model, dataset, budget, prm_options=prm_options, verbose=False, seed=seed, **train_options)
        used += fit["elapsed_epoch"]

        Tm = corner_table(model, designs, pool)
        dec = decide(model, Tm)
        cell_c, cell_d = model.cell_c.clone(), model.cell_d.clone()
        new_c, new_d = dec["new_c"], dec["new_d"]
        changed = (new_c != cell_c) | (new_d != cell_d)
        rounds.append({
            "round": len(rounds) + 1,
            "epochs": fit["elapsed_epoch"],
            "nll": fit["nll"],
            "feasible": fit["prm_profile"]["feasible"],
            "c": cell_c.numpy().copy(),
            "d": cell_d.numpy().copy(),
            "n_changes": int(changed.sum()),
            "table": Tm.numpy().copy(),
            "mean": dec["mean"],
            "decisions": {k: dec[k].numpy() for k in FLAGS},
        })
        if verbose:
            n = lambda k: int(dec[k].sum())
            print(f"{label} round {len(rounds)}: {fit['elapsed_epoch']} epochs (total {used}/{max_epochs}) | "
                  f"add C raw {n('raw_add_c')} bg-removed {n('cen_add_c')} -> {n('grow_c')}, "
                  f"add D raw {n('raw_add_d')} bg-removed {n('cen_add_d')} -> {n('grow_d')}, "
                  f"jump raw {n('raw_jump')} bg-removed {n('cen_jump')} -> {n('jump')} | "
                  f"{int(changed.sum())} change(s) | {histogram(new_c, new_d)}", flush=True)

        stable = stable + 1 if not bool(changed.any()) else 0
        if stable >= stable_rounds:
            converged = True
            break

        model.cell_c.copy_(new_c)
        model.cell_d.copy_(new_d)
        for j in torch.nonzero(new_c > seen_c).flatten().tolist():  # first exposure of a slot -> light init
            model.reinit_slot("C", j, int(new_c[j]), generator)
        for j in torch.nonzero(new_d > seen_d).flatten().tolist():
            model.reinit_slot("D", j, int(new_d[j]), generator)
        seen_c, seen_d = torch.maximum(seen_c, new_c), torch.maximum(seen_d, new_d)

    return {"c": model.cell_c.clone(), "d": model.cell_d.clone(), "converged": converged, "epochs": used,
            "rounds": rounds, "cell": model}
