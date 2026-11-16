"""Item bank of fitted empirical items for the 'from' simulation profile.

From one or more finished application runs, every item of the fitted model --from-model (default the per-item MiRM
of the full data) is read with its own configuration and parameters. The parametric generators of the profile
(2PL, 3PL, LPE, Spline) take their items from the full-data fit of the same arm in the same runs: every item's
parameters (one row of each parameter tensor) go into bank["models"][name], pooled over the runs. Every per-item
MiRM item is classified on theta in [-4, 4] by the largest drop of its ICC below the running maximum: monotone when
the drop is below mono_tol and its configuration is one of FROM_CONFIGS = (1,1), (1,2), (2,2), (2,1); a dip item
when the drop lies in [dip_min, dip_max] with the trough in [-2.5, 2.5], whatever its configuration (other items are
not used). The bank is saved once by the parent process; the jobs only load it.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

FROM_CONFIGS = [(1, 1), (1, 2), (2, 2), (2, 1)]  # configurations of the monotone bank items (dip items: any)
GRID = np.linspace(-4, 4, 801)


def _icc(alpha_c, delta_c, alpha_d, delta_d) -> np.ndarray:
    lc = np.logaddexp.reduce(-np.outer(alpha_c, GRID) + np.asarray(delta_c)[:, None], axis=0)
    ld = np.logaddexp.reduce(-np.outer(alpha_d, GRID) + np.asarray(delta_d)[:, None], axis=0)
    return 1 / (1 + np.exp(-(lc - ld)))


def _drop(p: np.ndarray) -> tuple[float, float]:
    drop = np.maximum.accumulate(p) - p
    k = int(drop.argmax())
    return float(drop[k]), float(GRID[k])


def _run_rows(run: Path) -> tuple[list[dict], SimpleNamespace]:
    from mirm.run.utils.artifacts import load_results
    meta = json.loads((run / "run_meta.json").read_text())
    args = SimpleNamespace(**meta)
    args.data_dir = Path(args.data_dir)
    return [r for r in load_results(args, run) if not r.get("prediction") and not r.get("explore")], args


def _item_names(args) -> dict | None:
    try:
        from mirm.run.utils.jobs import load_empirical
        _, data_meta = load_empirical(args)
        return data_meta.get("idx_to_item")
    except Exception:
        return None


def _run_parametric(run: Path, rows: list[dict], names: dict | None, name: str) -> dict | None:
    """Per-item parameter rows of the full-data fit of arm name in one run (None if the run has no such fit)."""
    row = next((r for r in rows if r["arm"] == name), None)
    if row is None:
        return None
    model = row["result"]["model"]
    keys = [f"{run.name}:{names.get(j, names.get(str(j))) if names else f'item{j}'}" for j in range(model.n_items)]
    params = {k: v.detach().clone() for k, v in model.named_parameters() if v.dim() >= 1 and v.shape[0] == model.n_items}
    config = {"n_basis": int(model.n_basis)} if name == "Spline" else {}
    return {"keys": keys, "params": params, "config": config}


def _run_items(run: Path, model_name: str, rows: list[dict] | None = None, names: dict | None = None) -> list[dict]:
    if rows is None:
        rows, args = _run_rows(run)
        names = _item_names(args)
    models = [r["result"]["model"] for r in rows if str(r["result"]["model"]) == model_name]
    if not models:
        raise ValueError(f"{run}: no full-data model named {model_name!r}")
    model = models[0]
    with torch.no_grad():
        alpha = model.readable_params["alpha"].numpy()
        delta = model.readable_params["delta"].numpy()
        macro = model.macrostate_ordered.numpy()
    items = []
    for j in range(alpha.shape[0]):
        isc, isd = macro[j] == 1, macro[j] == 2
        key = str(names.get(j, names.get(str(j)))) if names else f"item{j}"
        items.append({"key": f"{run.name}:{key}", "alpha_c": alpha[j, isc], "delta_c": delta[j, isc],
                      "alpha_d": alpha[j, isd], "delta_d": delta[j, isd], "c": int(isc.sum()), "d": int(isd.sum())})
    return items


def build_bank(runs: list[Path], model_name: str = "MiRM_per_item", mono_tol: float = 0.01,
               dip_min: float = 0.05, dip_max: float = 0.3, out_path: Path | None = None, verbose: bool = True,
               parametric: tuple[str, ...] = ()) -> dict:
    """parametric: arms (e.g. 2PL, 3PL, LPE, Spline) whose full-data fits in the runs supply the items of the
    parametric generators; each must be found in at least one run (a run without it is skipped with a note)."""
    loaded = []
    for run in runs:
        rows, args = _run_rows(Path(run))
        loaded.append((Path(run), rows, _item_names(args)))
    items = [it for run, rows, names in loaded for it in _run_items(run, model_name, rows, names)]
    allowed = list(FROM_CONFIGS)
    bank = {k: [] for k in ("key", "alpha_c", "delta_c", "alpha_d", "delta_d", "c", "d", "depth", "trough")}
    for it in items:
        depth, trough = _drop(_icc(it["alpha_c"], it["delta_c"], it["alpha_d"], it["delta_d"]))
        for k in ("key", "alpha_c", "delta_c", "alpha_d", "delta_d", "c", "d"):
            bank[k].append(it[k])
        bank["depth"].append(depth)
        bank["trough"].append(trough)
    depth, trough = np.array(bank["depth"]), np.array(bank["trough"])
    bank["c"], bank["d"] = np.array(bank["c"]), np.array(bank["d"])
    bank["depth"], bank["trough"] = depth, trough
    in_allowed = np.array([(int(c), int(d)) in allowed for c, d in zip(bank["c"], bank["d"])], dtype=bool)
    bank["monotone"] = (depth < mono_tol) & in_allowed
    bank["dip"] = (depth >= dip_min) & (depth <= dip_max) & (np.abs(trough) <= 2.5)
    bank["allowed"] = allowed
    bank["dip_min"], bank["dip_max"] = float(dip_min), float(dip_max)
    bank["runs"] = [str(r) for r in runs]

    bank["models"] = {}
    for name in parametric:
        parts = []
        for run, rows, names in loaded:
            part = _run_parametric(run, rows, names, name)
            if part is None:
                if verbose:
                    print(f"'from' bank: {run} has no full-data {name} fit (skipped for {name})", flush=True)
                continue
            parts.append(part)
        if not parts:
            raise ValueError(f"no run in --from-runs has a full-data {name} fit (needed by the {name}_from generator)")
        configs = {tuple(sorted(part["config"].items())) for part in parts}
        if len(configs) > 1:
            raise ValueError(f"{name} fits differ in configuration across the runs: {sorted(configs)}")
        bank["models"][name] = {
            "keys": [k for part in parts for k in part["keys"]],
            "params": {k: torch.cat([part["params"][k] for part in parts]) for k in parts[0]["params"]},
            "config": parts[0]["config"],
        }
        if verbose:
            print(f"'from' bank: {name} {len(bank['models'][name]['keys'])} items from {len(parts)} run(s)"
                  + (f" {parts[0]['config']}" if parts[0]["config"] else ""), flush=True)
    if verbose:
        mono = Counter(zip(bank["c"][bank["monotone"]].tolist(), bank["d"][bank["monotone"]].tolist()))
        dip = Counter(zip(bank["c"][bank["dip"]].tolist(), bank["d"][bank["dip"]].tolist()))
        print(f"'from' bank: {len(items)} items in {len(runs)} run(s); monotone items from {allowed}: "
              f"{int(bank['monotone'].sum())} {dict(mono)}; dip items (any configuration) {int(bank['dip'].sum())} "
              f"{dict(dip)}; unused {int((~bank['monotone'] & ~bank['dip']).sum())}", flush=True)
    if out_path is not None:
        torch.save(bank, out_path)
    return bank


_LOADED: dict[str, dict] = {}


def load_bank(path) -> dict:
    path = str(path)
    if path not in _LOADED:
        _LOADED[path] = torch.load(path, map_location="cpu", weights_only=False)
    return _LOADED[path]
