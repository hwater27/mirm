import time
import math
from typing import Optional, Any

import torch

from mirm.data.datasets import ScoredIRTDataset
from mirm.model import BinaryIRTModel, BinaryMiRM, CellMiRM, PRM
from mirm.model.core import plan_designs


PHR_GROWTH_EVERY = 5  # additional epochs per x1.1 step of the PHR strength while the PRM margins are infeasible
_DEVICE_DESIGNS: dict = {}  # (id of the CPU plan, device) -> (CPU plan kept alive, device copies)


def trainer(
    model: BinaryIRTModel,
    dataset: ScoredIRTDataset,
    max_epochs: int,
    lr: float = 1.0,
    max_iter: int = 20,
    history_size: int = 100,
    prm_options: Optional[dict[str, Any]] = None,
    early_stop_tol: Optional[float] = 1e-9,
    early_stop_patience: int = 5,
    chunk_students: int = 0,
    chunk_mem_gb: float = 12.0,
    verbose: bool = True,
    seed: int = 27,
    device: str = "cpu",
) -> dict[str, Any]:
    """Direct MML with L-BFGS on the sparse response design (torch.sparse.mm over whole-student chunks).

    The L-BFGS curvature history is kept across epochs while the PHR objective is unchanged and is reset when the
    PRM dual or strength changes or the PRM cuts move. After every epoch the PRM cuts move first (PRM.update), then
    the dual is updated and feasibility is checked, so feasibility is taken at the lowest margin of every gap.

    With PRM, the returned model is the feasible epoch (training PRM margins within prm_options["tol"]) with the lowest
    NLL, evaluated exactly at the end of that epoch; when no epoch was feasible, the model of the last epoch is
    returned. "returned_epoch" and "nll" (per response) describe the returned model.
    """
    torch.manual_seed(seed)
    history = []

    width = model.n_quad * (4 if isinstance(model, CellMiRM) else 1)
    designs = plan_designs(dataset, model.n_items, width, chunk_students, chunk_mem_gb)
    # Training may run on a GPU: the model and the response designs move there for the fit, and the model returns to
    # the CPU before certification, so every downstream metric and plot runs on the CPU.
    device = torch.device(device)
    if device.type != "cpu":
        model.to(device)
        key = (id(designs), str(device))
        if key not in _DEVICE_DESIGNS:          # one dataset's designs kept on the device per process
            _DEVICE_DESIGNS.clear()
            _DEVICE_DESIGNS[key] = (designs, [(design.to(device), n) for design, n in designs])
        designs = _DEVICE_DESIGNS[key][1]
    n_obs = len(dataset)

    prm = None
    dual = None
    additional_epochs = 0
    force_feasible = False
    PHR_strength = 1.0

    if prm_options is not None:
        if not isinstance(model, BinaryMiRM):
            raise ValueError("PRM is only supported for BinaryMiRM model")

        prm = PRM(
            n_items=model.n_items,
            worst_m=prm_options.get("worst_m", None),
            form_masks=prm_options.get("form_masks", None),
            n_nodes=prm_options.get("n_nodes", None),
            n_scan=prm_options.get("cut_scan", 32),
            tol=prm_options.get("tol", 0.0),
        )

        additional_epochs = prm_options.get("additional_epochs", 20)
        force_feasible = prm_options.get("force_feasible", True)
        PHR_strength = prm_options.get("PHR_strength", 1.0)

        with torch.no_grad():
            values = prm.values(model)
        dual = torch.zeros_like(values)

    started = time.time()
    early_stop_count = 0
    elapsed_epoch = 0
    num_violations = None
    feasible = False
    optimizer = None
    best = None  # (nll, epoch, parameters) of the best feasible epoch

    for epoch in range(max_epochs + additional_epochs):
        if epoch >= max_epochs:
            if not force_feasible or feasible:
                break
            if (epoch - max_epochs) % PHR_GROWTH_EVERY == 0:
                PHR_strength *= 1.1
                optimizer = None

        elapsed_epoch = epoch + 1

        if optimizer is None:
            optimizer = torch.optim.LBFGS(
                model.parameters(), lr=lr, max_iter=max_iter, history_size=history_size, line_search_fn="strong_wolfe"
            )
        loss, nll, penalty = _lbfgs_epoch(model, designs, n_obs, prm, dual, PHR_strength, optimizer)

        if prm is not None:
            with torch.no_grad():
                if prm.update(model):
                    optimizer = None  # the cuts moved: the PHR objective changed, do not reuse curvature pairs
                values = prm.values(model)
                updated = torch.relu(dual + PHR_strength * values)
                if not torch.equal(updated, dual):
                    optimizer = None  # the PHR objective changed: do not reuse curvature pairs
                dual.copy_(updated)

                num_violations = int((values > prm.tol).sum().item())
                feasible = num_violations == 0
                if feasible:
                    exact = sum(float(model.sparse_nll(design)) for design, _ in designs)
                    if best is None or exact < best[0]:
                        best = (exact, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()})
        else:
            feasible = True

        history.append({
            "epoch": epoch,
            "phase": "main" if epoch < max_epochs else "additional",
            "loss": float(loss / n_obs),
            "nll": float(nll / n_obs),
            "penalty": float(penalty / n_obs),
            "feasible": feasible,
            "PHR_strength": PHR_strength,
            "num_violations": num_violations,
        })

        if verbose and epoch % max(1, max_epochs // 10) == 0:
            print(
                f"Epoch {epoch:>4d} | "
                f"Loss: {history[-1]['loss']:.9f} | "
                f"NLL: {history[-1]['nll']:.9f} | "
                f"Feasible: {feasible}"
            )

        if early_stop_tol is not None and epoch > 0:
            nll_change = abs(history[-1]["nll"] - history[-2]["nll"])
            if nll_change < early_stop_tol:
                early_stop_count += 1
            else:
                early_stop_count = 0
            if early_stop_count >= early_stop_patience:
                if not force_feasible or feasible:
                    break
                else:
                    continue

    elapsed_time = time.time() - started
    returned_epoch, returned_nll = elapsed_epoch - 1, history[-1]["nll"] if history else None
    if best is not None:
        model.load_state_dict(best[2])
        returned_epoch, returned_nll = best[1], best[0] / n_obs
        feasible, num_violations = True, 0
    if device.type != "cpu":
        model.cpu()
        if dual is not None:
            dual = dual.cpu()
    prm_profile = {
        "feasible": feasible,
        "dual": None if dual is None else dual.numpy(),
        "PHR_strength": PHR_strength,
        "num_violations": num_violations,
        "cut_moves": None if prm is None else prm.moves,
        "certified": None,
    }

    if prm is not None and not isinstance(model, CellMiRM) and prm_options.get("certify", True):
        prm_profile["certified"] = prm.certify(
            model,
            initial_nodes=prm_options.get("certify_nodes", 1601),
            tolerance=prm_options.get("certify_tolerance", 1e-3),
            max_depth=prm_options.get("certify_depth", 12),
        )

    return {
        "model": model,
        "history": history,
        "elapsed_time": elapsed_time,
        "elapsed_epoch": elapsed_epoch,
        "returned_epoch": returned_epoch,
        "nll": returned_nll,
        "prm_profile": prm_profile,
    }


def _lbfgs_epoch(
    model: BinaryIRTModel,
    designs: list,
    n_obs: int,
    prm: Optional[PRM],
    dual: Optional[torch.Tensor],
    PHR_strength: float,
    optimizer: torch.optim.LBFGS,
) -> tuple[float, float, float]:
    state = {"nll": None, "loss": None, "penalty": None}

    def closure():
        optimizer.zero_grad()
        nll = 0.0
        for design, _ in designs:
            part = model.sparse_nll(design)
            part.backward()
            nll += float(part.detach().item())
        penalty = _prm_loss(model, prm, n_obs, dual, PHR_strength)
        if penalty.requires_grad:
            penalty.backward()

        state["nll"] = nll
        state["penalty"] = float(penalty.detach().item())
        state["loss"] = nll + state["penalty"]

        return torch.tensor(state["loss"], dtype=torch.float64)

    optimizer.step(closure)
    return state["loss"], state["nll"], state["penalty"]


def _prm_loss(
    model: BinaryIRTModel,
    prm: Optional[PRM],
    n_obs: int,
    dual: Optional[torch.Tensor],
    PHR_strength: float = 1.0,
) -> torch.Tensor:
    if prm is None:
        return next(model.parameters()).new_zeros(())

    values = prm.values(model)
    shifted = dual + PHR_strength * values

    PHR = (torch.relu(shifted).square() - dual.square()) / (2 * PHR_strength)
    information_scale = float(n_obs) * math.log(2)

    return information_scale * PHR.mean()
