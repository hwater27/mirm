from typing import Any

import numpy as np

from mirm.data.datasets import KeyedIRTDataset, ScoredIRTDataset
from mirm.data.generate import generate, generate_dip, generate_from
from mirm.data.io import load_dataset
from mirm.data.utils import split_students, top_items
from mirm.model import MODEL_NAMES, build_model
from mirm.eval.train import trainer
from mirm.eval.selection import select_configuration
from mirm.eval.metrics import fit_metrics
from mirm.run.utils.parallel import report
from mirm.run.utils.from_bank import load_bank

RECOVERY_MIXTURE = {"1,1": 1 / 3, "2,3": 1 / 3, "3,1": 1 / 3}

_DATASETS: dict[tuple, tuple[ScoredIRTDataset, dict]] = {}
_CACHE_SIZE = 32  # datasets kept per process (oldest dropped first)


def _remember(key, value) -> None:
    _DATASETS[key] = value
    while len(_DATASETS) > _CACHE_SIZE:
        _DATASETS.pop(next(iter(_DATASETS)))


def load_empirical(args) -> tuple[ScoredIRTDataset | KeyedIRTDataset, dict]:
    """The empirical dataset as loaded (scored or keyed), restricted to the --top-items most answered items.

    The restriction is applied once to the whole file, before any train/test split, so every split and every
    analysis uses the same items in the same order (item i of the subset is metadata["selected_items"][i])."""
    path = args.data_dir / args.data_name / args.data_fname
    if not path.exists():
        raise FileNotFoundError(f"Data file {path} not found")
    original, metadata = load_dataset(path)

    n_top = getattr(args, "top_items", None)
    if n_top:
        original, old_items, _ = top_items(original, n_top)
        mapping = metadata.get("idx_to_item") or {}
        metadata["selected_items"] = old_items.tolist()
        metadata["idx_to_item"] = {
            i: mapping.get(int(old), mapping.get(str(int(old)), int(old))) for i, old in enumerate(old_items.tolist())
        }
    return original, metadata


def load_job_dataset(args, job: dict[str, Any]) -> tuple[ScoredIRTDataset, dict]:
    if "generator" in job:
        key = (
            job["generator"],
            job["seed"],
            job["density"],
            job["n_students"],
            job["n_items"],
            job["study"],
            job.get("class_spread"),
            job.get("treat_spread"),
            job.get("loading_scale"),
        )
        cached = _DATASETS.get(key)
        if cached is None:
            if job["generator"] in ("MiRM_from", "MiRM_from_mix", "dip_from") or job["generator"].endswith("_from"):
                kind = {"MiRM_from": "mono", "MiRM_from_mix": "mix", "dip_from": "dip"}.get(
                    job["generator"], job["generator"].removesuffix("_from"))      # 2PL_from -> 2PL, ...
                dip_options = None
                if kind == "dip":
                    dip_options = {"class_spread": job["class_spread"], "treat_spread": job["treat_spread"],
                                   "loading_scale": job["loading_scale"]}
                dataset, metadata = generate_from(
                    load_bank(args.from_bank), kind,
                    n_students=job["n_students"],
                    n_items=job["n_items"],
                    density=job["density"],
                    seed=job["seed"],
                    n_quad=args.n_quad,
                    dip_options=dip_options,
                )
            elif job["generator"] == "dip":
                dataset, metadata = generate_dip(
                    n_students=job["n_students"],
                    n_items=job["n_items"],
                    density=job["density"],
                    class_spread=job["class_spread"],
                    treat_spread=job["treat_spread"],
                    loading_scale=job["loading_scale"],
                    seed=job["seed"],
                    n_quad=args.n_quad,
                )
            else:
                mixture = job["generator"] == "MiRM_mix"
                dataset, metadata = generate(
                    generator="MiRM" if mixture else job["generator"],
                    n_students=job["n_students"],
                    n_items=job["n_items"],
                    density=job["density"],
                    generator_config={"mixture": RECOVERY_MIXTURE} if mixture else None,
                    seed=job["seed"],
                    n_quad=args.n_quad,
                )
            _remember(key, (dataset, metadata))
        else:
            dataset, metadata = cached

    else:
        key = (args.data_dir / args.data_name / args.data_fname, getattr(args, "top_items", None))
        cached = _DATASETS.get(key)
        if cached is None:
            original, metadata = load_empirical(args)
            dataset = original.to_scored() if isinstance(original, KeyedIRTDataset) else original
            _remember(key, (dataset, metadata))
        else:
            dataset, metadata = cached

    if job.get("prediction"):
        split = split_students(dataset, test_ratio=args.test_ratio, seed=job["seed"])
        dataset, test_dataset = split["train"], split["test"]
    else:
        test_dataset = None

    return dataset, test_dataset, metadata


def _prm_options(args, **overrides) -> dict[str, Any]:
    return {
        "additional_epochs": args.additional_epochs,
        "force_feasible": args.force_feasible,
        "PHR_strength": args.PHR_strength,
        "worst_m": args.worst_m,
        "form_masks": None if args.form_masks is None else np.load(args.form_masks),
        "n_nodes": args.prm_nodes or None,
        "cut_scan": args.prm_cut_scan,
        "certify_nodes": args.certify_nodes,
        "certify_tolerance": args.certify_tolerance,
        "certify_depth": args.certify_depth,
        **overrides,
    }


def resolve_device(spec: str | None) -> str:
    """'cpu', 'cuda', 'cuda:N' or a comma-separated list of CUDA devices; with a list (or bare 'cuda' and several
    visible GPUs) the parallel workers are spread over the devices by worker index."""
    spec = (spec or "cpu").strip()
    if spec == "cpu":
        return "cpu"
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(f"--device {spec}: CUDA is not available")
    devices = [d.strip() for d in spec.split(",") if d.strip()]
    if devices == ["cuda"]:
        devices = [f"cuda:{k}" for k in range(torch.cuda.device_count())]
    import multiprocessing as mp
    identity = mp.current_process()._identity
    worker = identity[0] - 1 if identity else 0
    return devices[worker % len(devices)]


def _train_options(args) -> dict[str, Any]:
    return {
        "device": resolve_device(getattr(args, "device", "cpu")),
        "lr": args.lr,
        "max_iter": args.max_iter,
        "history_size": args.history_size,
        "early_stop_tol": args.early_stop_tol,
        "early_stop_patience": args.early_stop_patience,
        "chunk_students": args.chunk_students,
        "chunk_mem_gb": args.chunk_mem_gb,
    }


def run_job(args, job: dict[str, Any]) -> dict[str, Any]:
    dataset, _, _ = load_job_dataset(args, job)

    n_items = dataset.n_items
    n_quad = args.n_quad

    model_name = job["arm"]
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Invalid arm: {model_name}")

    use_prm = model_name == "MiRM" and job.get("prm", args.prm)
    prm_options = _prm_options(args) if use_prm else None
    selection = None

    if job.get("select"):
        selection = select_configuration(
            dataset,
            n_quad=n_quad,
            cap=args.select_cap,
            max_epochs=args.select_max_epochs,
            pool=args.select_pool,
            stable_rounds=args.select_stable,
            train_options={**_train_options(args), "early_stop_tol": args.select_early_stop_tol},
            prm_options=_prm_options(args, n_nodes=args.select_prm_nodes or 2 * n_quad, tol=args.select_prm_tol,
                                     certify=False) if use_prm else None,
            label=f"[select {job.get('study', '')} seed={job.get('seed')}]",
            verbose=False,
            progress=report,
            seed=args.seed,
        )
        model = selection.pop("cell").trimmed(selection["c"], selection["d"])
        report(f"refit ({len(selection['rounds'])} rounds, converged {selection['converged']})")
        job = {**job, "c": selection["c"].numpy(), "d": selection["d"].numpy()}
    else:
        ctor = {k: job[k] for k in ("c", "d", "n_basis") if k in job}
        model = build_model(model_name, n_items, n_quad, **ctor)

    if model_name == "Spline":
        model.initialize_from_data(dataset.data)

    train_options = _train_options(args)
    result = trainer(model=model, dataset=dataset, max_epochs=args.max_epochs, prm_options=prm_options,
                     verbose=False, seed=args.seed, **train_options)

    if selection is not None:
        selection["c"], selection["d"] = selection["c"].numpy(), selection["d"].numpy()
        result["selection"] = selection

    fit = fit_metrics(model, dataset)
    return {
        **job,
        "model": str(model),
        "result": result,
        "fit_metrics": fit,
    }
