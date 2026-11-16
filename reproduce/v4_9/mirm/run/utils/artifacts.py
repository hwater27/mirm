from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import csv
import json
import re

import numpy as np
import torch
import matplotlib.pyplot as plt

from mirm.data.datasets import KeyedIRTDataset
from mirm.data.io import load_dataset
from mirm.model import build_model, BinaryMiRM
import mirm.eval.metrics as metrics
import mirm.eval.audit as audit
import mirm.run.utils.viz as viz
from mirm.run.utils.jobs import load_job_dataset, load_empirical
from mirm.run.utils.parallel import map_parallel
from functools import partial

NOT_IDENTIFIERS = {
    "result", "fit_metrics", "dataset", "test_dataset", "metadata",
    "arm", "c", "d", "n_basis", "prm", "explore", "model", "prediction", "select",
}
CHARACTERISTICS = ("probability", "discrimination", "difficulty", "acceleration", "information")
MAX_DIPS = 50
INTERCEPT_EXAMPLES = 4


def _label(row) -> str:
    return row["model"] + ("" if row.get("prm", True) else " noPRM")


def _condition(condition: dict) -> str:
    return "_".join(f"{k}{v}" for k, v in sorted(condition.items()) if k != "seed") or "all"


def _slug(*parts) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "_", "_".join(str(p) for p in parts)).strip("_")


def _dip_depth(p: np.ndarray) -> np.ndarray:
    return (np.maximum.accumulate(p, axis=-1) - p).max(axis=-1)


def _domains(keys: list[str]) -> tuple[list[str], torch.Tensor]:
    letters = [key[1] if key[:1] in ("C", "P") and len(key) > 1 else key[:1] for key in keys]
    names = sorted(set(letters))
    masks = torch.tensor([[letter == name for letter in letters] for name in names])
    return names, masks


def write_csv(path, records: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in records for k in r))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({k: ("" if v is None else v) for k, v in r.items()})


def run_meta(args):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.tag or ""
    out_path = args.out_dir / f"{stamp}_{args.data_name}{tag}"
    out_path.mkdir(parents=True, exist_ok=True)
    for name in ("plots", "tables", "checkpoints", "items", "nonmono_audit", "ndim_audit"):
        (out_path / name).mkdir(parents=True, exist_ok=True)

    with open(out_path / "run_meta.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=4, default=str)
    return out_path


# ------------------------------------------------------------
# Analysis
# ------------------------------------------------------------

_WORKER_CACHE: dict = {}  # per analysis process: keyed data, reference 2PL abilities


def _dip_abilities(dataset, model, reference_2pl) -> dict:
    """Abilities that do not depend on the item, computed once per row (the 2PL MAP once per dataset per process)."""
    data = dataset.data
    theta_self = model.map_theta(data)
    abilities = {}
    if reference_2pl is not None:
        key = ("theta_2pl", id(dataset))
        if key not in _WORKER_CACHE:
            _WORKER_CACHE[key] = reference_2pl.map_theta(data)
        abilities["2PL"] = _WORKER_CACHE[key]
    abilities["total"] = audit.normal_scores(audit.total_score(dataset))
    abilities["generalized"] = audit.normal_scores(audit.generalized_score(dataset, model, theta=theta_self))
    abilities["self"] = theta_self
    return abilities


def _keyed(args):
    if "keyed" not in _WORKER_CACHE:
        original, _ = load_empirical(args)
        _WORKER_CACHE["keyed"] = original if isinstance(original, KeyedIRTDataset) else None
    return _WORKER_CACHE["keyed"]


def _analyze_row(args, reference_2pl, row) -> dict:
    """Everything computed for one fitted job; runs in an analysis worker (datasets are rebuilt there and cached)."""
    dataset, test_dataset, meta = load_job_dataset(args, row)
    half = args.plot_half_bound
    plot_theta_grid = torch.linspace(-half, half, args.plot_nodes)
    synthetic = args.data_name == "synthetic"

    label = _label(row)
    condition = {k: v for k, v in row.items() if k not in NOT_IDENTIFIERS}
    identifier = {"model": label, "condition": _condition(condition), **condition}
    model = row["result"]["model"]
    out = {"category": "grid" if row["explore"] else ("prediction" if row["prediction"] else "full"),
           "items": [], "matrices": [], "dips": [], "bins": [], "intercepts": [], "selection": [],
           "parameters": [], "density": None}

    if row.get("select"):
        split = f"seed{row['seed']}" if row["prediction"] else ("full" if not synthetic else f"rep{row['seed']}")
        counts = torch.bincount(dataset.item_ids, minlength=model.n_items)
        out["selection"] = [
            {**identifier, "split": split, "item": j, "c": int(model.c[j]), "d": int(model.d[j]), "n_j": int(counts[j])}
            for j in range(model.n_items)
        ]

    if row["explore"]:
        sections = {"fit": row["fit_metrics"]}
    elif row["prediction"]:
        sections = {
            "predictive": metrics.predictive_metrics(model, test_dataset, n_folds=args.k_fold, seed=args.seed)
        }
    else:
        true_model = None
        theta_true = None
        config = None
        if synthetic:
            name = "MiRM" if meta["generator"] == "dip" else meta["generator"]
            config = meta.get("config") or {k: meta[k] for k in ("c", "d") if k in meta}
            true_model = build_model(name, model.n_items, args.n_quad, **config)
            true_model.load_state_dict(meta["state_dict_true"])
            theta_true = meta["theta_true"]

        prm_profile = row["result"].get("prm_profile") or {}
        certified = prm_profile.get("certified") or {}
        sections = {"fit": row["fit_metrics"]}
        # Q3 and S-X2 are not part of the monotonicity study (and are its most expensive steps)
        if not (synthetic and row.get("study") == "monotonicity"):
            dependence = metrics.dependence_metrics(model, dataset)
            out["matrices"].append({**identifier, "q3": dependence.pop("q3_matrix")})
            sections["sx2"] = metrics.SX2_metrics(model, dataset)
            sections["dependence"] = dependence
        sections["characteristic"] = metrics.characteristic_metrics(model, plot_theta_grid, true_model)
        sections["ability"] = metrics.ability_metrics(model, dataset, theta_true)
        parameter = metrics.parameter_metrics(model, true_model) if str(true_model) == str(model) else {}
        if synthetic and row.get("study") == "recovery":
            out["parameters"] = [{**identifier, **record} for record in parameter.get("matched_records", [])]
        sections["parameter"] = {k: v for k, v in parameter.items() if k != "matched_records"}
        sections["prm"] = {**prm_profile, **certified}
        if not synthetic:
            sections["rmsd"] = metrics.rmsd_metrics(model, dataset)
        if row.get("select"):
            sections["selection"] = {k: v for k, v in row["result"]["selection"].items()
                                     if k in ("converged", "epochs")}
        if synthetic and row.get("study") == "recovery" and isinstance(model, BinaryMiRM):
            true_c, true_d = np.asarray(config["c"]), np.asarray(config["d"])
            fit_c, fit_d = model.c.numpy(), model.d.numpy()
            sections["config"] = {
                "item_true_c": true_c, "item_true_d": true_d, "item_fit_c": fit_c, "item_fit_d": fit_d,
                "accuracy": float(np.mean((true_c == fit_c) & (true_d == fit_d))),
            }
        if not synthetic or row.get("seed") == 0:
            out["density"] = ((label, _condition(condition), condition.get("seed")),
                              audit.population_density(model, dataset, plot_theta_grid).numpy())

        if isinstance(model, BinaryMiRM):
            with torch.no_grad():
                depth = _dip_depth(model.p_correct_grid(plot_theta_grid).numpy())
            flagged = np.flatnonzero(depth >= args.mono_tol)
            ranked = flagged[np.argsort(-depth[flagged])]

            dip = {"n_flagged": int(flagged.size), "item_depth": depth}
            if synthetic:
                true = np.asarray(meta.get("dip_flags", np.zeros(model.n_items)), dtype=bool)
                fitted = depth >= args.mono_tol
                dip.update({
                    "tp": int((fitted & true).sum()),
                    "fp": int((fitted & ~true).sum()),
                    "fn": int((~fitted & true).sum()),
                    "tn": int((~fitted & ~true).sum()),
                    "item_true": true.astype(int),
                })
                if "dip_depths" in meta:
                    dip["item_true_depth"] = np.asarray(meta["dip_depths"])
                    dip["item_true_theta"] = np.asarray(meta["dip_theta"])
            sections["dip"] = dip

            if not synthetic and ranked.size:
                fixed = _dip_abilities(dataset, model, reference_2pl)
                for j in ranked[:MAX_DIPS]:
                    abilities = {**fixed, "rest": audit.normal_scores(audit.rest_score(dataset, int(j)))}
                    found = audit.dip_significance(
                        model, dataset, int(j), abilities, theta_min=-half, theta_max=half,
                        n_bins=getattr(args, "dip_bins", 20), tail_bins=getattr(args, "dip_tail_bins", 1),
                    )
                    for reference, z in (found["referenced_significance"] or {}).items():
                        out["dips"].append({**identifier, "item": int(j), "depth": float(depth[j]),
                                            "theta_peak": found["theta_peak"], "theta_dip": found["theta_dip"],
                                            "reference": reference, "z": float(z)})
                        probs = found["binned_probs"][reference].numpy()
                        out["bins"].append({**identifier, "item": int(j), "reference": reference,
                                            "p": probs[0], "sd": probs[1], "n": probs[2],
                                            "edges": found["edges"], "compared": found["compared"][reference]})

            keyed = _keyed(args) if (not synthetic and getattr(args, "check_intercept", False)) else None
            if keyed is not None:
                data = keyed.data
                counts = torch.zeros(model.n_items, keyed.n_choices + 1)
                counts.index_put_((data[:, 1].long(), data[:, 2].long()), torch.ones(len(data)), accumulate=True)
                wrong = counts.clone()
                wrong[torch.arange(model.n_items), keyed.answers] = 0
                ratio = (wrong.max(dim=1).values / wrong.sum(dim=1).clamp_min(1)).numpy()
                out["intercepts"] = [
                    {**identifier, "item": j, "dip": bool(depth[j] >= args.mono_tol),
                     "depth": float(depth[j]), "intercept_ratio": float(ratio[j])}
                    for j in range(model.n_items)
                ]
                for j in ranked[:INTERCEPT_EXAMPLES]:
                    overall, rates = audit.intercept(keyed, model, int(j), theta_min=-half, theta_max=half)
                    out["bins"].append({**identifier, "item": int(j), "reference": "intercept",
                                        "rates": rates, "overall": overall, "answer": int(keyed.answers[j])})

    flat = {
        f"{name}_{k}": v
        for name, values in sections.items() for k, v in values.items()
        if v is None or np.isscalar(v)
    }
    out["record"] = {**identifier, **flat}
    if row["explore"]:
        out["record"].update({k: row[k] for k in ("c", "d", "n_basis") if k in row})
    if out["category"] == "full":
        columns = {
            f"{name}_{k}": np.asarray(v)
            for name, values in sections.items() for k, v in values.items()
            if isinstance(v, (np.ndarray, torch.Tensor)) and np.ndim(v) == 1 and len(v) == model.n_items
        }
        out["items"] = [{**identifier, "item": j, **{k: v[j] for k, v in columns.items()}}
                        for j in range(model.n_items)]
    out["identifier_keys"] = list(identifier)
    return out


def analyze(args, results):
    """Per-job metrics in parallel analysis workers (--analysis-workers), merged here in job order."""
    synthetic = args.data_name == "synthetic"
    reference_2pl = None
    if not synthetic:
        reference_2pl = next(
            (row["result"]["model"] for row in results
             if row["arm"] == "2PL" and not row["explore"] and not row["prediction"]),
            None,
        )

    rows = [{k: v for k, v in row.items() if k not in ("dataset", "test_dataset", "metadata")} for row in results]
    workers = getattr(args, "analysis_workers", None) or args.num_workers   # older run_meta.json: --num-workers
    if args.verbose:
        print(f"============ [ analysis: {len(rows)} fits, {workers} workers ] ============")
    outcomes = map_parallel(
        partial(_analyze_row, args, reference_2pl),
        rows,
        max_workers=workers,
        threads_per_worker=args.threads_per_worker,
        verbose=args.verbose,
    )

    if not synthetic:  # the plots read the item keys of the empirical data
        for row in results:
            if not row["explore"] and not row["prediction"]:
                _, _, row["metadata"] = load_job_dataset(args, row)

    tables = {name: [] for name in ("grid", "full", "prediction", "items", "dips", "intercept", "selection",
                                    "parameters")}
    matrices, bins, densities = [], [], {}
    identifier_keys = set()
    for out in outcomes:
        tables[out["category"]].append(out["record"])
        tables["items"].extend(out["items"])
        tables["dips"].extend(out["dips"])
        tables["intercept"].extend(out["intercepts"])
        tables["selection"].extend(out["selection"])
        tables["parameters"].extend(out["parameters"])
        matrices.extend(out["matrices"])
        bins.extend(out["bins"])
        identifier_keys.update(out["identifier_keys"])
        if out["density"] is not None:
            densities[out["density"][0]] = out["density"][1]

    groups = {}
    for record in tables["full"] + tables["prediction"]:
        key = tuple(sorted((k, v) for k, v in record.items() if k in identifier_keys and k != "seed"))
        groups.setdefault(key, []).append(record)

    main = []
    for key, records in groups.items():
        summary = dict(key)
        for name in {k for r in records for k in r if k not in identifier_keys}:
            values = [r[name] for r in records if r.get(name) is not None]
            if values and isinstance(values[0], str):
                summary[name] = values[0] if len(set(values)) == 1 else None
                continue
            summary[f"{name}_mean"] = float(np.mean(values)) if values else None
            summary[f"{name}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        main.append(summary)

    return {
        "tables": {"main": main, **tables},
        "matrices": matrices,
        "bins": bins,
        "densities": densities,
    }


MODEL_ORDER = ["2PL", "3PL", "LPE", "RH", "3PLQ", "Spline", "MiRM (", "MiRM_per_item"]


def _model_rank(name: str) -> tuple:
    """Position of the longest matching prefix ("3PLQ" ranks as 3PLQ, not as 3PL)."""
    matches = [(len(prefix), k) for k, prefix in enumerate(MODEL_ORDER) if name.startswith(prefix)]
    return (max(matches)[1] if matches else len(MODEL_ORDER)), name


def summary_table(main: list[dict]) -> str:
    """Model comparison printed after the analysis: number of free parameters k (full data), AIC/BIC differences
    from the per-item MiRM (signed), held-out NLL and mAUC (mean (sd) over the splits, five decimals), marginal
    reliability and mean item RMSD."""
    rows = sorted((r for r in main if r.get("fit_aic_mean") is not None), key=lambda r: _model_rank(r["model"]))
    if not rows:
        return ""
    reference = next((r for r in rows if r["model"].endswith("_per_item")), min(rows, key=lambda r: r["fit_aic_mean"]))

    def number(value, fmt):
        return "" if value is None or not np.isfinite(value) else format(value, fmt)

    def with_sd(r, name):
        mean, sd = r.get(f"{name}_mean"), r.get(f"{name}_sd")
        if mean is None:
            return ""
        return f"{mean:.5f}" + ("" if sd is None else f" ({sd:.5f})")

    # Headers keep a space after the Greek delta: a terminal that draws it double-width overlaps the space, not the
    # next letter, and the column widths stay correct either way.
    header = ["Model", "k", "Δ AIC", "Δ BIC", "ho-NLL (sd)", "ho-mAUC (sd)", "rel.", "RMSD"]
    body = [[
        r["model"],
        number(r.get("fit_n_params_mean"), ".0f"),
        number(r["fit_aic_mean"] - reference["fit_aic_mean"], "+,.1f"),
        number(r["fit_bic_mean"] - reference["fit_bic_mean"], "+,.1f"),
        with_sd(r, "predictive_ho_nll_per_obs"),
        with_sd(r, "predictive_ho_mauc"),
        number(r.get("ability_marginal_reliability_mean"), ".4f"),
        number(r.get("rmsd_rmsd_mean_mean"), ".4f"),
    ] for r in rows]
    widths = [max(len(line[k]) for line in [header] + body) for k in range(len(header))]
    align = lambda cells: "| " + " | ".join(
        cell.ljust(w) if k == 0 else cell.rjust(w) for k, (cell, w) in enumerate(zip(cells, widths))) + " |"
    lines = ["", align(header), "|" + "|".join("-" * (w + 2) for w in widths) + "|", *map(align, body),
             f"(Δ from {reference['model']}; negative favors the row model)"]
    return "\n".join(lines)


GENERATOR_ORDER = ["2PL", "3PL", "LPE", "MiRM", "Spline"]
PRECISION_METRICS = (("characteristic_icc_rmse", "ICC RMSE"), ("ability_rmse", "θ RMSE"))


def precision_tables(full: list[dict]) -> tuple[str, dict]:
    """Precision study (each fitted model x each generator), one table per metric: ICC RMSE over the plot grid and
    ability RMSE. Rows are the fitted models, columns the generators ('_from' dropped); each cell is the mean (sd) over
    the replicates, four decimals; per column ** marks the lowest mean and * the second lowest. Returns the printed
    text and the wide records ({metric: [row, ...]}) for tables/precision_<metric>.csv."""
    rows = [r for r in full if r.get("study") == "precision"]
    if not rows:
        return "", {}
    name = lambda g: str(g).removesuffix("_from")
    present = {name(r["generator"]) for r in rows}
    generators = [g for g in GENERATOR_ORDER if g in present] + sorted(present - set(GENERATOR_ORDER))
    models = sorted({r["model"] for r in rows}, key=_model_rank)
    sizes = sorted({(r.get("n_students"), r.get("n_items")) for r in rows})
    seeds = {(r["model"], name(r["generator"])): set() for r in rows}
    for r in rows:
        seeds[(r["model"], name(r["generator"]))].add(r.get("seed"))
    n_reps = sorted({len(v) for v in seeds.values()})

    blocks, records = [], {}
    for key, label in PRECISION_METRICS:
        stats = {}
        for model in models:
            for g in generators:
                values = np.array([r[key] for r in rows if r["model"] == model and name(r["generator"]) == g
                                   and r.get(key) is not None and np.isfinite(r[key])], dtype=float)
                if values.size:
                    stats[(model, g)] = (float(values.mean()), float(values.std(ddof=1)) if values.size > 1 else 0.0)
        marks = {}
        for g in generators:
            ranked = sorted((stats[(m, g)][0], m) for m in models if (m, g) in stats)
            for mark, (_, m) in zip(("**", "*"), ranked):
                marks[(m, g)] = mark
        header = [label] + generators
        body = [[model] + [(f"{stats[(model, g)][0]:.4f}{marks.get((model, g), '')} ({stats[(model, g)][1]:.4f})"
                            if (model, g) in stats else "") for g in generators] for model in models]
        widths = [max(len(line[k]) for line in [header] + body) for k in range(len(header))]
        align = lambda cells: "| " + " | ".join(
            cell.ljust(w) if k == 0 else cell.rjust(w) for k, (cell, w) in enumerate(zip(cells, widths))) + " |"
        blocks += ["", align(header), "|" + "|".join("-" * (w + 2) for w in widths) + "|", *map(align, body)]
        records[key] = [{"model": model, **{f"{g}_{suffix}": (stats[(model, g)][j] if (model, g) in stats else None)
                                            for g in generators for j, suffix in enumerate(("mean", "sd"))}}
                        for model in models]
    size = ", ".join(f"{s} students x {i} items" for s, i in sizes)
    reps = "/".join(str(n) for n in n_reps)
    blocks.append(f"(precision study: rows fitted model, columns generator; mean (sd) over {reps} replicates, {size}; "
                  f"per column ** lowest, * second lowest)")
    return "\n".join(blocks), records


# ------------------------------------------------------------
# Saving
# ------------------------------------------------------------

def save_results(args, out_path, results, artifact):
    if not (out_path / "jobs.pt").exists() and results is not None:
        jobs = []
        for i, row in enumerate(results):
            path = out_path / "checkpoints" / f"{i}.pt"
            model = row["result"]["model"]
            torch.save(model.state_dict(), path)

            result = {k: v for k, v in row["result"].items() if k != "model"}
            result["checkpoint"] = str(path.relative_to(out_path))

            jobs.append({**{k: v for k, v in row.items() if k not in ("dataset", "test_dataset", "metadata")},
                         "result": result, "n_items": model.n_items})
        torch.save(jobs, out_path / "jobs.pt")

    if artifact is None:
        return

    torch.save(artifact, out_path / "artifact.pt")
    tables = artifact["tables"]
    for name, records in tables.items():
        if records:
            write_csv(out_path / "tables" / f"{name}.csv", records)
    for key, records in precision_tables(tables.get("full", []))[1].items():
        write_csv(out_path / "tables" / f"precision_{key}.csv", records)

    half = args.plot_half_bound
    theta = torch.linspace(-half, half, args.plot_nodes)
    theta_np = theta.numpy()
    synthetic = args.data_name == "synthetic"
    densities = artifact.get("densities", {})

    models = {}
    for row in results:
        if row["explore"] or row["prediction"] or (synthetic and row.get("seed") != 0):
            continue
        condition = {k: v for k, v in row.items() if k not in NOT_IDENTIFIERS}
        models[(_label(row), _condition(condition), condition.get("seed"))] = row
    reference_2pl = next((row for key, row in models.items() if key[0] == "2PL"), None)

    # Training history
    if not synthetic:
        from mirm.run.utils.viz.icc import _style
        fig, axes = viz.figure(1, 2, size=(5.0, 3.4))
        for axis, key, title in zip(axes[0], ("nll", "penalty"), ("NLL per response", "PRM penalty per response")):
            for k, ((label, _, _), row) in enumerate(models.items()):
                history = row["result"]["history"]
                color, linestyle, _ = _style(k)
                axis.plot([h["epoch"] for h in history], [h[key] for h in history], color=color, linestyle=linestyle,
                          linewidth=1.2, label=label)
            axis.set_title(title)
            axis.set_xlabel("epoch")
            axis.legend(fontsize=6)
            axis.grid(alpha=0.15)
        viz.save(fig, out_path / "plots" / "history.png")

    # (c, d) grid: training NLL, AIC and BIC per configuration (c vertical, d horizontal)
    conditions = {}
    for r in tables["grid"]:
        if "c" in r:
            conditions.setdefault(r["condition"], {}).setdefault(r["seed"], []).append(r)
    for condition, by_seed in conditions.items():
        max_c = max(r["c"] for rs in by_seed.values() for r in rs)
        max_d = max(r["d"] for rs in by_seed.values() for r in rs)
        for seed, rs in by_seed.items():
            M = {metric: np.full((max_c, max_d), np.nan) for metric in ("fit_nll_total", "fit_aic", "fit_bic")}
            for r in rs:
                for metric in M:
                    M[metric][r["c"] - 1, r["d"] - 1] = r[metric]
            fig, axes = viz.figure(1, 3, size=(4.2, 3.4))
            for axis, (metric, title) in zip(axes[0], zip(M, ("Δ train NLL", "Δ AIC", "Δ BIC"))):
                viz.heatmap(axis, M[metric] - np.nanmin(M[metric]), xticks=range(1, max_d + 1),
                            yticks=range(1, max_c + 1), xlabel="d", ylabel="c", title=title, winner="min",
                            annotate="{:.0f}")
            viz.save(fig, out_path / "plots" / "grid" / f"{_slug(condition)}_seed{seed}.png")

    # ICC overview, Q3, item pages, item-wise configuration
    with torch.no_grad():
        for (label, condition, seed), row in models.items():
            model = row["result"]["model"]
            stem = f"{_slug(label, condition)}_seed{seed}"
            p = model.p_correct_grid(theta).numpy()
            highlight = np.flatnonzero(_dip_depth(p) >= args.mono_tol) if isinstance(model, BinaryMiRM) else None
            density = densities.get((label, condition, seed))

            fig, axes = viz.figure(size=(5.5, 4.0))
            viz.icc_overview(axes[0, 0], theta_np, p, highlight=highlight, density=density, title=label)
            viz.save(fig, out_path / "plots" / "icc" / f"{stem}.png")

            if not isinstance(model, BinaryMiRM):
                continue

            if not synthetic:
                characteristics = model.characteristics(theta)
                char = {name: characteristics[name].numpy() for name in CHARACTERISTICS}
                limits = {}
                for name in ("discrimination", "acceleration", "information"):
                    values = char[name][np.isfinite(char[name])]
                    low, high = (np.quantile(values, [0.005, 0.995]) if values.size else (0.0, 1.0))
                    pad = 0.05 * (high - low + 1e-12)
                    limits[name] = (min(0.0, low) - pad, max(0.0, high) + pad)
                n_items = min(model.n_items, args.max_ic)
                for page, start in enumerate(range(0, n_items, 6), 1):
                    chunk = range(start, min(start + 6, n_items))
                    fig, axes = viz.figure(len(chunk), len(CHARACTERISTICS), size=(3.0, 2.2))
                    for r, j in enumerate(chunk):
                        viz.characteristics_row(axes[r], theta_np, {name: char[name][j] for name in CHARACTERISTICS},
                                                limits=limits, row_label=f"item {j}", first_row=r == 0,
                                                last_row=r == len(chunk) - 1)
                    viz.save(fig, out_path / "items" / f"{stem}_{page:03d}.png")

            if model.c.unique().numel() > 1 or model.d.unique().numel() > 1:
                counts = np.zeros((model.c_max, model.d_max))
                np.add.at(counts, (model.c.numpy() - 1, model.d.numpy() - 1), 1)
                fig, axes = viz.figure(size=(4.6, 3.8))
                viz.heatmap(axes[0, 0], counts, xticks=range(1, model.d_max + 1), yticks=range(1, model.c_max + 1),
                            xlabel="d", ylabel="c", title=f"item-wise (c, d): {label}", annotate="{:.0f}")
                viz.save(fig, out_path / "plots" / f"itemwise_cd_{stem}.png")

    for record in artifact["matrices"]:
        if synthetic and record.get("seed") != 0:
            continue
        fig, axes = viz.figure(size=(6.0, 5.0))
        viz.heatmap(axes[0, 0], record["q3"], title=f"Q3: {record['model']}", annotate=None, center=0.0)
        viz.save(fig, out_path / "plots" / "q3" / f"{_slug(record['model'], record['condition'])}_seed{record.get('seed')}.png")

    # Selection consistency across the splits (one panel per run)
    if tables["selection"] and not synthetic:
        cfg, splits = selection_configs(tables["selection"])
        viz.consistency_figure([(args.data_fname, cfg, splits)], out_path / "plots" / "cd_consistency.png")

    # Held-out summary
    if not synthetic:
        main = sorted((r for r in tables["main"] if r.get("predictive_ho_mauc_mean") is not None),
                      key=lambda r: _model_rank(r["model"]))
        if main:
            fig, axes = viz.figure(1, 2, size=(5.0, 3.6))
            labels = [r["model"] for r in main]
            viz.bars(axes[0, 0], [r["predictive_ho_mauc_mean"] for r in main],
                     errors=[r["predictive_ho_mauc_sd"] or 0.0 for r in main], labels=labels, best="max", title="held-out mAUC")
            viz.bars(axes[0, 1], [r["predictive_ho_nll_per_obs_mean"] for r in main],
                     errors=[r["predictive_ho_nll_per_obs_sd"] or 0.0 for r in main], labels=labels, best="min", title="held-out NLL per response")
            viz.save(fig, out_path / "plots" / "predictive.png")

    # Non-monotonicity audit: rest-score bins on the ICC of flagged items
    if tables["dips"]:
        z_rest = {}
        for record in tables["dips"]:
            if record["reference"] == "rest":
                z_rest[(record["model"], record["condition"], record["seed"], record["item"])] = record["z"]
        with torch.no_grad():
            for record in artifact["bins"]:
                if record["reference"] != "rest":
                    continue
                key = (record["model"], record["condition"], record["seed"])
                if key not in models:
                    continue
                model = models[key]["result"]["model"]
                j = record["item"]
                z = z_rest.get((*key, j), float("nan"))
                reference_icc = None if reference_2pl is None else \
                    reference_2pl["result"]["model"].p_correct_grid(theta)[j].numpy()

                fig, axes = viz.figure(size=(5.0, 3.8))
                viz.dip_panel(axes[0, 0], theta_np, model.p_correct_grid(theta)[j].numpy(), record["edges"],
                              (record["p"], record["sd"], record["n"]), z=z, density=densities.get(key),
                              reference_icc=reference_icc, compared=record.get("compared"),
                              title=f"{record['model']} item {j}")
                viz.save(fig, out_path / "nonmono_audit" / f"{_slug(*key)}_item{j}.png")

        if z_rest:
            fig, axes = viz.figure(size=(max(5.0, 0.35 * len(z_rest)), 3.6))
            viz.bars(axes[0, 0], list(z_rest.values()), labels=[f"{k[3]}" for k in z_rest], title="rest-score dip z by item", ylabel="z")
            viz.save(fig, out_path / "nonmono_audit" / "z_rest.png")

    # Intercept audit (keyed data)
    if tables["intercept"]:
        for key in dict.fromkeys((r["model"], r["condition"], r["seed"]) for r in tables["intercept"]):
            examples = [b for b in artifact["bins"]
                        if b["reference"] == "intercept" and (b["model"], b["condition"], b["seed"]) == key]
            if not examples:
                continue
            fig, axes = viz.figure(1, len(examples), size=(4.2, 3.4))
            edges = audit._normspace(-half, half, examples[0]["rates"].shape[1] + 1)
            for axis, example in zip(axes[0], examples):
                centers = (edges[:-1] + edges[1:]) / 2
                K = example["rates"].shape[0]
                viz.curves(axis, centers, example["rates"], labels=[f"choice {k + 1}" + (" (key)" if k + 1 == example["answer"] else "") for k in range(K)],
                           ylim=(-0.02, 1.02), title=f"item {example['item']}", xlabel="θ", ylabel="choice rate", marker="auto")
            viz.save(fig, out_path / "nonmono_audit" / f"intercept_{_slug(*key)}.png")

    # Dimensionality audit: domain-restricted PRM margins (empirical multi-domain data only)
    if not synthetic and models:
        metadata = next(iter(models.values()))["metadata"]
        mapping = metadata.get("idx_to_item") or {}
        keys = [str(mapping.get(i, mapping.get(str(i), ""))) for i in range(len(mapping))]
        names, masks = _domains(keys) if keys else ([], None)
        if len(names) > 1:
            records = []
            with torch.no_grad():
                for (label, condition, seed), row in models.items():
                    model = row["result"]["model"]
                    if not isinstance(model, BinaryMiRM):
                        continue
                    margins = audit.domain_margins(model, masks, theta)
                    fig, axes = viz.figure(1, 2, size=(5.0, 3.6))
                    viz.curves(axes[0, 0], theta_np, margins["c_norm"].numpy(), labels=names, title=f"C-margin by domain: {label}", xlabel="θ")
                    viz.curves(axes[0, 1], theta_np, margins["d_norm"].numpy(), labels=names, title=f"D-margin by domain: {label}", xlabel="θ")
                    for axis in axes[0]:
                        axis.axhline(0.0, color="black", linewidth=0.8)
                    viz.save(fig, out_path / "ndim_audit" / f"margins_{_slug(label, condition)}_seed{seed}.png")
                    records.extend(
                        {"model": label, "condition": condition, "seed": seed, "domain": names[k], "theta": float(t),
                         "c_norm": float(margins["c_norm"][k, i]), "d_norm": float(margins["d_norm"][k, i])}
                        for k in range(len(names)) for i, t in enumerate(theta_np)
                    )
            if records:
                write_csv(out_path / "ndim_audit" / "margins.csv", records)

    if synthetic:
        _simulation_plots(args, out_path, tables)


RECOVERY_TRUTHS = [(1, 1), (1, 2), (2, 2), (2, 1)]      # rows of the (c, d) recovery matrix, top to bottom
RECOVERY_SELECTIONS = [(1, 1), (1, 2), (2, 2), (2, 1)]  # its columns (same order), then "others"


def _simulation_plots(args, out_path, tables) -> None:
    precision = [r for r in tables["main"] if r.get("study") == "precision"]
    if precision:
        generators = sorted({r["generator"] for r in precision})
        names = sorted({r["model"] for r in precision}, key=_model_rank)
        fig, axes = viz.figure(1, 2, size=(6.0, 3.8))
        for axis, metric, title in zip(axes[0], ("characteristic_icc_rmse", "ability_rmse"), ("ICC RMSE", "ability RMSE")):
            values = np.full((len(generators), len(names)), np.nan)
            errors = np.zeros_like(values)
            for r in precision:
                g, m = generators.index(r["generator"]), names.index(r["model"])
                values[g, m] = r.get(f"{metric}_mean") or np.nan
                errors[g, m] = r.get(f"{metric}_sd") or 0.0
            viz.bars(axis, values, errors=errors, labels=names, groups=[f"truth {g}" for g in generators], title=title, annotate=None)
        viz.save(fig, out_path / "plots" / "precision.png")

    sensitivity = [r for r in tables["main"] if r.get("study") == "sensitivity"]
    if sensitivity:
        # the precision study fits the sensitivity generator at the full size (n_students, n_items) with the same
        # seeds and arms, so its rows are the next point of both sensitivity panels
        generators = {r.get("generator") for r in sensitivity}
        seen = {(r["model"], r["n_students"], r["n_items"]) for r in sensitivity}
        sensitivity = sensitivity + [r for r in precision if r.get("generator") in generators
                                     and (r["model"], r["n_students"], r["n_items"]) not in seen]
        names = sorted({r["model"] for r in sensitivity}, key=_model_rank)
        mirm = next((m for m in names if m.startswith("MiRM")), None)
        groups = []
        for metric, ylabel in (("characteristic_icc_rmse", "ICC RMSE"), ("ability_rmse", "ability RMSE")):
            panels = []
            for key, fixed in (("n_students", "n_items"), ("n_items", "n_students")):
                fixed_value = getattr(args, fixed)
                xs = sorted({r[key] for r in sensitivity if r[fixed] == fixed_value})

                def stat(m, x, suffix, default):
                    value = next((r.get(f"{metric}_{suffix}") for r in sensitivity
                                  if r["model"] == m and r[key] == x and r[fixed] == fixed_value), None)
                    return default if value is None else value

                panels.append({"x": xs, "bar_label": mirm or "MiRM",
                               "bars": [stat(mirm, x, "mean", np.nan) for x in xs],
                               "bar_errors": [stat(mirm, x, "sd", 0.0) for x in xs],
                               "lines": {m: [stat(m, x, "mean", np.nan) for x in xs] for m in names if m != mirm},
                               "line_errors": {m: [stat(m, x, "sd", 0.0) for x in xs] for m in names if m != mirm},
                               "title": f"vs {key} ({fixed} = {fixed_value})", "xlabel": key})
            groups.append((ylabel, panels))
        viz.grouped_row(groups, out_path / "plots" / "sensitivity.png")
        viz.grouped_row(groups, out_path / "plots" / "sensitivity_plain.png", errors=False)

    mono = [r for r in tables["full"] if r.get("study") == "monotonicity" and r.get("dip_tp") is not None]
    if mono:
        def cells_of(rs):
            """Per-fit counts, rows truth and columns fit, monotone first: [[mono->mono, mono->dip], [dip->mono,
            dip->dip]] (the dip is the positive class of dip_tp / dip_fp / dip_fn / dip_tn)."""
            return np.array([[[r["dip_tn"], r["dip_fp"]], [r["dip_fn"], r["dip_tp"]]] for r in rs], dtype=float)

        densities = sorted({r["density"] for r in mono})
        for density in [*densities, None]:
            fig, axes = viz.figure(1, 2, size=(5.4, 3.0))
            for axis, prm in zip(axes[0], (False, True)):
                rs = [r for r in mono if (density is None or r["density"] == density) and r["model"].endswith("noPRM") != prm]
                if not rs:
                    axis.axis("off")
                    continue
                cells = cells_of(rs)
                total = cells.sum(axis=0)
                rates = total / np.maximum(total.sum(axis=1, keepdims=True), 1)
                mean, sd = cells.mean(axis=0), cells.std(axis=0, ddof=1) if len(cells) > 1 else np.zeros((2, 2))
                text = np.array([[f"{mean[i, j]:.1f} ± {sd[i, j]:.1f}\n({100 * rates[i, j]:.1f}%)" for j in range(2)]
                                 for i in range(2)], dtype=object)
                scope = "all conditions" if density is None else f"density {density}"
                (tn, fp), (fn, tp) = total                     # binary_summary takes the dip as the positive class
                viz.confusion(axis, total, row_labels=["true: monotone", "true: dip"],
                              col_labels=["fit: monotone", "fit: dip"], text=text,
                              title=f"{'PRM' if prm else 'no PRM'} ({scope}, n={len(rs)})",
                              ylabel=not prm, summary=viz.binary_summary([[tp, fn], [fp, tn]]))
            name = "confusion_all.png" if density is None else f"confusion_density{density}.png"
            viz.save(fig, out_path / "plots" / name)

        # one-row summary: matrices over all conditions + main-effect lines of the two error rates per factor
        factor_keys = [k for k in ("density", "class_spread", "treat_spread", "loading_scale")
                       if len({r[k] for r in mono}) > 1]
        arms = {"no PRM": [r for r in mono if r["model"].endswith("noPRM")],
                "PRM": [r for r in mono if not r["model"].endswith("noPRM")]}
        arms = {arm: rs for arm, rs in arms.items() if rs}

        def rate(r, metric):
            if metric == "false non-monotone":
                n = r["dip_fp"] + r["dip_tn"]
                return r["dip_fp"] / n if n else np.nan
            n = r["dip_fn"] + r["dip_tp"]
            return r["dip_fn"] / n if n else np.nan

        factors = {k: sorted({r[k] for r in mono}) for k in factor_keys}
        lines = {}
        for arm, rs in arms.items():
            lines[arm] = {}
            for metric in ("false non-monotone", "false monotone"):
                lines[arm][metric] = {}
                for k, levels in factors.items():
                    means, sds = [], []
                    for level in levels:
                        values = np.array([rate(r, metric) for r in rs if r[k] == level], dtype=float)
                        values = values[np.isfinite(values)]
                        means.append(float(values.mean()) if values.size else np.nan)
                        sds.append(float(values.std(ddof=1)) if values.size > 1 else 0.0)
                    lines[arm][metric][k] = (means, sds)
        # monotone-by-construction arms fitted in the study (2PL): no dip is ever fitted
        reference = {name: {"false non-monotone": 0.0, "false monotone": 1.0}
                     for name in sorted({r["model"] for r in tables["full"] if r.get("study") == "monotonicity"
                                         and r["model"].split("_")[0] in ("2PL", "3PL", "LPE")}, key=_model_rank)}
        if arms and factors:
            for name, errors in (("monotonicity.png", True), ("monotonicity_plain.png", False)):
                viz.monotonicity_figure({arm: cells_of(rs) for arm, rs in arms.items()}, factors, lines,
                                        out_path / "plots" / name, reference=reference, errors=errors)

        cells = {}
        for r in mono:
            cell = (r["density"], r["class_spread"], r["treat_spread"], r["loading_scale"], not r["model"].endswith("noPRM"))
            cells.setdefault(cell, []).append(r)
        keys = sorted({cell[:4] for cell in cells})
        fig, axes = viz.figure(2, 1, size=(max(6.0, 0.28 * len(keys)), 3.4))
        for axis, (num, den, title) in zip(axes[:, 0], (("dip_fp", "dip_tn", "false positive rate"), ("dip_tp", "dip_fn", "power"))):
            values = np.full((2, len(keys)), np.nan)
            for k, key in enumerate(keys):
                for g, prm in enumerate((False, True)):
                    rs = cells.get((*key, prm), [])
                    rates = [r[num] / max(r[num] + r[den], 1) for r in rs]
                    values[g, k] = np.mean(rates) if rates else np.nan
            viz.bars(axis, values, labels=[f"d{d} c{c} t{t} l{l}" for d, c, t, l in keys], groups=["no PRM", "PRM"], title=title, annotate=None)
        viz.save(fig, out_path / "plots" / "fpr_power.png")

    recovery = [r for r in tables["items"] if r.get("study") == "recovery"]
    if recovery:
        selected = [r for r in recovery if r["model"].endswith("_per_item") and r.get("config_item_true_c") is not None]
        # fixed layout: truths (1,1), (1,2), (2,2), (2,1) top to bottom (other true configurations, e.g. (2,3) of the
        # heavy mixture, below them); selections in the same order, then every other selection as "others"
        present = {(int(r["config_item_true_c"]), int(r["config_item_true_d"])) for r in selected}
        truths = RECOVERY_TRUTHS + sorted(present - set(RECOVERY_TRUTHS))
        estimates = RECOVERY_SELECTIONS + ["others"]
        counts = np.zeros((len(truths), len(estimates)))
        for r in selected:
            fit = (int(r["config_item_fit_c"]), int(r["config_item_fit_d"]))
            counts[truths.index((int(r["config_item_true_c"]), int(r["config_item_true_d"]))),
                   estimates.index(fit) if fit in RECOVERY_SELECTIONS else len(estimates) - 1] += 1
        shown = [t for t in truths if t in present]                  # truths of the RMSE bars (present ones only)

        true_of = {(r["seed"], r["item"]): (int(r["config_item_true_c"]), int(r["config_item_true_d"])) for r in selected}
        names = sorted({r["model"] for r in recovery}, key=_model_rank)
        rmse = np.full((len(names), len(shown)), np.nan)
        spread = np.zeros_like(rmse)
        for m, name in enumerate(names):
            for t, truth in enumerate(shown):
                values = [r["characteristic_item_icc_rmse"] for r in recovery
                          if r["model"] == name and true_of.get((r["seed"], r["item"])) == truth
                          and r.get("characteristic_item_icc_rmse") is not None]
                if values:
                    rmse[m, t], spread[m, t] = np.mean(values), np.std(values, ddof=1) if len(values) > 1 else 0.0

        row_labels = [f"({c},{d})" for c, d in truths]
        col_labels = [f"({e[0]},{e[1]})" if e != "others" else "others" for e in estimates]
        params = [r for r in tables.get("parameters", []) if r["model"].endswith("_per_item")]
        # two versions: (a) matrix, ICC RMSE bars, parameter scatter; (b) without the ICC RMSE bars
        for panels, name in (("abc", "recovery.png"), ("ac", "recovery_two_panel.png")):
            fig, axes = viz.figure(1, len(panels), size=(4.8, 3.9))
            row = list(axes[0])
            viz.confusion(row[0], counts, row_labels=row_labels, col_labels=col_labels,
                          title="per-item (c, d) recovery")
            if "b" in panels:
                viz.bars(row[1], rmse, errors=spread, labels=[f"truth ({c},{d})" for c, d in shown], groups=names,
                         title="item ICC RMSE by true configuration", ylabel="ICC RMSE", annotate=None)
            viz.parameter_scatter(row[-1], [r["true_alpha"] for r in params], [r["fit_alpha"] for r in params],
                                  [r["true_delta"] for r in params], [r["fit_delta"] for r in params],
                                  title=f"microstate parameters (items with recovered (c, d))")
            _equal_height(fig, row, 0, counts.shape[1] / counts.shape[0])
            viz.save(fig, out_path / "plots" / name)


def _equal_height(fig, row: list, index: int, aspect: float, iterations: int = 20) -> None:
    """Give row[index] (a panel with square cells, width / height = aspect) the full height of its row: its column is
    widened or narrowed (and the figure with it) until its box is aspect x the common panel height; the other panels
    keep their widths, so every panel keeps its own proportions and all panels have the same height."""
    grid = row[0].get_subplotspec().get_gridspec()
    width, height = fig.get_size_inches()
    for _ in range(iterations):
        fig.tight_layout()
        boxes = [axis.get_position(original=True) for axis in row]
        widths = [box.width * width for box in boxes]
        target = boxes[index].height * height * aspect
        if abs(target - widths[index]) < 0.002:
            break
        width += target - widths[index]
        widths[index] = target
        grid.set_width_ratios(widths)
        fig.set_size_inches(width, height)


def selection_configs(records: list[dict]) -> tuple[dict, list[str]]:
    """{split: {item: (c, d)}} with "full" first, from the selection table of one run."""
    cfg = {}
    for r in records:
        cfg.setdefault(str(r["split"]), {})[int(r["item"])] = (int(r["c"]), int(r["d"]))
    splits = ["full"] + sorted(s for s in cfg if s != "full")
    return cfg, splits


def load_results(args, out_path):
    jobs = torch.load(out_path / "jobs.pt", map_location="cpu", weights_only=False)

    results = []
    for row in jobs:
        result = dict(row["result"])
        ctor = {k: row[k] for k in ("c", "d", "n_basis") if k in row}
        model = build_model(row["arm"], row["n_items"], args.n_quad, **ctor)
        model.load_state_dict(torch.load(out_path / result.pop("checkpoint"), map_location="cpu"))
        result["model"] = model
        results.append({**row, "result": result})
    return results


# ------------------------------------------------------------
# PRM ablation across runs (domain anchors + full PRM / no-PRM)
# ------------------------------------------------------------

def _load_run(path: Path):
    meta = json.loads((path / "run_meta.json").read_text(encoding="utf-8"))
    args = SimpleNamespace(**meta)
    args.data_dir = Path(args.data_dir)
    results = load_results(args, path)
    rows = [r for r in results if r["arm"] == "MiRM" and not r["explore"] and not r["prediction"]]
    row = next((r for r in rows if str(r["model"]).endswith("_per_item")), rows[0])
    _, metadata = load_dataset(args.data_dir / args.data_name / args.data_fname)
    mapping = metadata["idx_to_item"]
    keys = [str(mapping.get(i, mapping.get(str(i)))) for i in range(len(mapping))]
    args.train_nll_per_obs = (row.get("fit_metrics") or {}).get("nll_per_obs")
    return row["result"]["model"], keys, args


def _reflected_fits_better(model, reference, theta: torch.Tensor) -> bool:
    """True when model's ICCs match reference's ICCs at -theta better than at theta (N(0, 1)-weighted squared
    difference over all items): the theta direction of a fit without PRM is not identified."""
    with torch.no_grad():
        P = model.p_correct_grid(theta)
        weight = torch.exp(-0.5 * theta.square())[None, :]
        same = float((weight * (P - reference.p_correct_grid(theta)).square()).sum())
        reflected = float((weight * (P - reference.p_correct_grid(-theta)).square()).sum())
    return reflected < same


def prm_ablation(anchors: dict[str, Path], full_prm: Path, full_noprm: Path, out_path: Path, *,
                 max_points: int = 20000, elev: float = 25.0, azim: float = -52.0, score_samples: int = 400,
                 flip_noprm: str | bool = "auto", insets: Optional[dict] = None) -> Path:
    """Full-bank theta paths of the PRM and no-PRM fits in the domain (anchor) PRM coordinates, over a cloud of the
    students' domain abilities (MAP of each anchor on the student's responses to that domain; students who answered
    every domain), with the ICCs of both fits colored along theta as the color key; plus the domain margins.

    flip_noprm: the no-PRM fit is reflected (theta -> -theta, i.e. alpha -> -alpha; the likelihood and the train NLL
    do not change) when True, or with "auto" when its ICCs match the PRM fit's ICCs better reflected. insets: keyword
    arguments of viz.fold_inset (low- and high-theta zooms)."""
    out_path = Path(out_path)
    runs = {name: _load_run(Path(path)) for name, path in anchors.items()}
    prm_model, keys, args = _load_run(Path(full_prm))
    noprm_model, noprm_keys, noprm_args = _load_run(Path(full_noprm))
    if flip_noprm == "auto":
        half = args.plot_half_bound
        flip_noprm = _reflected_fits_better(noprm_model, prm_model, torch.linspace(-half, half, args.plot_nodes))
    if flip_noprm:
        with torch.no_grad():
            noprm_model.alpha_raw.neg_()  # logit = -alpha * theta + delta: theta -> -theta
    print(f"no-PRM fit {'reflected (theta -> -theta)' if flip_noprm else 'kept as fitted'}: {full_noprm}", flush=True)
    train_nll = {"PRM": args.train_nll_per_obs, "no PRM": noprm_args.train_nll_per_obs}
    if keys != noprm_keys:
        raise ValueError("PRM and no-PRM full runs do not use the same items.")
    for name, (_, _, anchor_args) in runs.items():
        if not anchor_args.prm:
            raise ValueError(f"Anchor run {name} must use PRM.")

    lookup = {key: i for i, key in enumerate(keys)}
    anchor_models = {name: (model, torch.tensor([lookup[key] for key in anchor_keys]))
                     for name, (model, anchor_keys, _) in runs.items()}
    names = list(anchors)
    domains, masks = _domains(keys)

    half = args.plot_half_bound
    theta = torch.linspace(-half, half, args.plot_nodes)
    theta_np = theta.numpy()

    # students: domain abilities from the anchor models on the full-bank responses of each domain
    full_dataset, _, _ = load_job_dataset(args, {"prediction": False})
    data = full_dataset.data
    with torch.no_grad():
        abilities = []
        for name in names:
            anchor, item_ids = anchor_models[name]
            to_anchor = torch.full((len(keys),), -1, dtype=torch.long)
            to_anchor[item_ids] = torch.arange(len(item_ids))
            local = to_anchor[data[:, 1].long()]
            part = data[local >= 0].clone()
            part[:, 1] = local[local >= 0].to(part.dtype)
            abilities.append(audit.map_abilities(anchor, part, full_dataset.n_students, theta).numpy())
    students = np.stack(abilities, axis=1)
    students = students[np.isfinite(students).all(axis=1)]
    (out_path / "tables").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path / "tables" / "student_coordinates.npz", coordinates=students, domains=np.array(names))

    paths, trajectory, margins_records = {}, [], []
    with torch.no_grad():
        for condition, model in (("no PRM", noprm_model), ("PRM", prm_model)):
            coordinates = audit.trajectory(model, anchor_models, theta)
            X = np.stack([coordinates[name].numpy() for name in names], axis=1)
            paths[condition] = X
            trajectory.extend({"condition": condition, "theta": float(t), **{name: float(X[i, k]) for k, name in enumerate(names)}}
                              for i, t in enumerate(theta_np))
            margins = audit.domain_margins(model, masks, theta)
            margins_records.extend(
                {"condition": condition, "domain": domains[k], "theta": float(t),
                 "c_norm": float(margins["c_norm"][k, i]), "d_norm": float(margins["d_norm"][k, i])}
                for k in range(len(domains)) for i, t in enumerate(theta_np)
            )
    write_csv(out_path / "tables" / "trajectory.csv", trajectory)
    write_csv(out_path / "tables" / "margins.csv", margins_records)

    cmaps = viz.theta_cmaps()
    fig = plt.figure(figsize=(15.0, 7.5))
    gs = fig.add_gridspec(2, 3, width_ratios=(1.4, 1.4, 0.9))
    axis3d = fig.add_subplot(gs[:, :2], projection="3d")
    viz.clean_3d(axis3d)
    drawn = viz.cloud3d(axis3d, students, max_points=max_points)
    shown = [viz.path3d_gradient(axis3d, paths[condition], theta_np, cmaps[condition], label=condition,
                                 t_range=(theta_np[0], theta_np[-1])) for condition in ("no PRM", "PRM")]
    extent = np.vstack(shown + ([np.quantile(drawn, [0.002, 0.998], axis=0)] if len(drawn) else []))
    for setter, k in ((axis3d.set_xlim, 0), (axis3d.set_ylim, 1), (axis3d.set_zlim, 2)):
        setter(extent[:, k].min() - 0.4, extent[:, k].max() + 0.6)  # room for the arrowheads
    for setter, name in zip((axis3d.set_xlabel, axis3d.set_ylabel, axis3d.set_zlabel), names):
        setter(f"{name} PRM coordinate")
    axis3d.set_title("full-bank θ path in domain coordinates "
                     "(color: θ as in the ICC panels;  ● θ = −4,  arrowhead θ = +4)")
    axis3d.legend(fontsize=8, loc="upper right")
    axis3d.view_init(elev=elev, azim=azim)
    score_records = []
    with torch.no_grad():
        for axis, (condition, model) in zip((fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])),
                                            (("no PRM", noprm_model), ("PRM", prm_model))):
            curve, share = audit.generalized_score_curve(model, theta, n_samples=score_samples, seed=args.seed)
            title = f"full {condition}"
            if train_nll[condition] is not None:  # train NLL per response; the PRM panel adds the gap to no PRM
                title += f"\ntrain NLL {train_nll[condition]:.5f}/obs"
                if condition == "PRM" and train_nll["no PRM"] is not None:
                    title += f"  (Δ {train_nll['PRM'] - train_nll['no PRM']:+.5f})"
            viz.theta_iccs(axis, theta_np, model.p_correct_grid(theta).numpy(), cmaps[condition],
                           title=title, curve=curve, curve_label="generalized score $G_R(θ)/L$",
                           certified=share >= 0.5)
            score_records.extend({"condition": condition, "theta": float(t), "score": float(g), "certified": float(c)}
                                 for t, g, c in zip(theta_np, curve, share))
    write_csv(out_path / "tables" / "generalized_score.csv", score_records)
    fig.tight_layout()
    shown = viz.fold_inset(fig, axis3d, paths, theta_np, students, cmaps, **(insets or {}))
    if "low" in shown:
        low = shown["low"]
        print(f"low zoom: half-width {low['half']:.2f}, near-crossing at theta = {low['theta']:.2f}, "
              f"start-to-crossing distance {low['start_to_crossing']:.2f}, centre shifted {low['shift']:.2f}", flush=True)
    if "high" in shown:
        high = shown["high"]
        print(f"high zoom: turn at theta = {high['theta']:.2f}, inset {high['side']} "
              f"({high['center'][0]:.3f}, {high['center'][1]:.3f})", flush=True)
    viz.save(fig, out_path / "plots" / "prm_ablation.png", tight=False)

    fig, axes = viz.figure(2, 2, size=(5.0, 3.4))
    for r, condition in enumerate(("no PRM", "PRM")):
        for c, side in enumerate(("c_norm", "d_norm")):
            Y = [[m[side] for m in margins_records if m["condition"] == condition and m["domain"] == domain] for domain in domains]
            viz.curves(axes[r, c], theta_np, Y, labels=domains, title=f"{condition}: {side}", xlabel="θ")
            axes[r, c].axhline(0.0, color="black", linewidth=0.8)
    viz.save(fig, out_path / "plots" / "domain_margins.png")
    return out_path


# ------------------------------------------------------------
# Cross-run figures (selection consistency, dip showcase)
# ------------------------------------------------------------

DEFAULT_SPLITS = ["full", "seed0", "seed1", "seed3", "seed4", "seed6", "seed8", "seed9"]


def cd_consistency(panels: list[tuple[str, str]], out_path: Path, sort: str = "zigzag") -> Path:
    """One tile panel per run in display order; a source is a run directory (its tables/selection.csv) or "blank:N"
    for an empty N-item placeholder."""
    loaded = []
    for title, source in panels:
        if str(source).startswith("blank:"):
            loaded.append((title, None, None, int(str(source).split(":", 1)[1])))
            continue
        with open(Path(source) / "tables" / "selection.csv", encoding="utf-8") as handle:
            cfg, splits = selection_configs(list(csv.DictReader(handle)))
        loaded.append((title, cfg, splits, 0))
    splits = next((p[2] for p in loaded if p[2] is not None), DEFAULT_SPLITS)
    figure_panels = [(t, cfg, sp or splits) if cfg is not None else (t, None, splits, n) for t, cfg, sp, n in loaded]
    return viz.consistency_figure(figure_panels, Path(out_path), sort=sort)


SHOWCASE = (  # (run key, item index, title, note)
    ("R", 22, "Reading 22 (CR466Q03)", "Work Right Q03\njoint of three yes/no choices"),
    ("S", 81, "Science 81 (CS649Q01)", "Weather Balloon Q01\n4-choice multiple choice"),
    ("M", 42, "Math 42 (CM998Q04)", "Bike Rental Q04\njoint of two yes/no choices"),
)


def dip_showcase(runs: dict[str, Path], out_path: Path, items=SHOWCASE, reference: str = "rest") -> Path:
    """Dip significance panels of selected items side by side: per-item MiRM ICC, translucent 2PL ICC, binned
    proportions against the rest score, the z of the peak-vs-dip comparison and the students' posterior strip."""
    fig, axes = viz.figure(1, len(items), size=(4.2, 3.7))
    for axis, (key, item, title, note) in zip(axes[0], items):
        path = Path(runs[key])
        args = SimpleNamespace(**json.loads((path / "run_meta.json").read_text(encoding="utf-8")))
        args.data_dir = Path(args.data_dir)
        results = load_results(args, path)
        full_rows = [r for r in results if not r["prediction"] and not r["explore"]]
        row = next(r for r in full_rows if str(r["model"]).endswith("_per_item"))
        two_pl = next((r for r in full_rows if r["arm"] == "2PL"), None)
        dataset, _, metadata = load_job_dataset(args, row)
        model = row["result"]["model"]

        half = args.plot_half_bound
        theta = torch.linspace(-half, half, args.plot_nodes)
        with torch.no_grad():
            if reference == "rest":
                abilities = {"rest": audit.normal_scores(audit.rest_score(dataset, item))}
            else:
                abilities = {reference: two_pl["result"]["model"].map_theta(dataset.data)}
            found = audit.dip_significance(model, dataset, item, abilities, theta_min=-half, theta_max=half,
                                           n_bins=getattr(args, "dip_bins", 20),
                                           tail_bins=getattr(args, "dip_tail_bins", 1))
            icc = model.p_correct_grid(theta)[item].numpy()
            reference_icc = None if two_pl is None else two_pl["result"]["model"].p_correct_grid(theta)[item].numpy()
            density = audit.population_density(model, dataset, theta).numpy()
        tag = next(iter(abilities))
        probs = found["binned_probs"][tag].numpy() if found["binned_probs"] else None
        z = (found["referenced_significance"] or {}).get(tag, float("nan"))
        if probs is None:
            probs = np.full((3, len(found["edges"]) - 1), np.nan)
        viz.dip_panel(axis, theta.numpy(), icc, found["edges"], probs, z=z, density=density,
                      reference_icc=reference_icc, compared=(found["compared"] or {}).get(tag, found["bins"]), title=title,
                      note=f"{note}\ndepth {found['significance']:.3f}",
                      points_label=f"{reference}-score bins")
    out_path = Path(out_path)
    viz.save(fig, out_path)
    return out_path
