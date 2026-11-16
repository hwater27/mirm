"""Random reassignment of the selected per-item (c, d) configurations (PISA 2015 Mathematics).

Takes the per-item configurations selected on the full data in a finished application run, permutes them among the
items (each item receives the (c, d) pair of another item, so the multiset of configurations and the number of free
parameters are unchanged), and refits MiRM under every permuted assignment with the training and PRM settings of
that run. With the parameter count fixed, differences in AIC and BIC equal twice the difference in training NLL.

The per-item fit stored in the run starts from the states of the selection (warm start), which a permuted
assignment cannot use. The comparison therefore also refits the selected assignment itself from random starts
("selected, cold start"): permuted fits are compared with these, and the stored fit is reported for reference.

Place this file next to the package folders (v4_8, v4_9) and run it there:
    python -u exp_reassign.py --num-workers 4 2>&1 | tee exp_reassign.log

Every fit is saved as soon as it ends (<out>/<label>.json and .pt); a rerun skips the fits already saved.
"""
import argparse
import csv
import json
import os
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PACKAGE = Path(os.environ.get("MIRM_PACKAGE", HERE / "v4_8")).resolve()  # also read by the spawned workers
sys.path.insert(0, str(PACKAGE))

import numpy as np
import torch

from mirm.model import build_model
from mirm.eval.train import trainer
from mirm.eval.metrics import fit_metrics
from mirm.run.utils.jobs import load_job_dataset, _train_options, _prm_options
from mirm.run.utils.parallel import map_parallel

DEFAULT_RUN = "runs/application/20261019_133728_pisa_2015_M"


def load_run(run: Path, device: str | None) -> tuple[SimpleNamespace, dict]:
    """Arguments of the run (paths resolved against the package folder) and its full-data per-item row."""
    args = SimpleNamespace(**json.loads((run / "run_meta.json").read_text(encoding="utf-8")))
    data_dir = Path(args.data_dir)
    args.data_dir = data_dir if data_dir.is_absolute() else (PACKAGE / data_dir).resolve()
    args.form_masks = None if args.form_masks in (None, "None") else Path(args.form_masks)
    if device is not None:
        args.device = device

    jobs = torch.load(run / "jobs.pt", map_location="cpu", weights_only=False)
    rows = [r for r in jobs if r.get("select") and not r["prediction"] and not r.get("explore")]
    if len(rows) != 1:
        raise ValueError(f"Expected one full-data per-item fit in {run}, found {len(rows)}")
    return args, rows[0]


def fit(args: SimpleNamespace, out_dir: Path, job: dict) -> dict:
    label = f"{job['generator']}_{job['seed']:03d}"
    path = out_dir / f"{label}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    dataset, _, _ = load_job_dataset(args, {"prediction": False})
    torch.manual_seed(job["init_seed"])
    model = build_model("MiRM", dataset.n_items, args.n_quad, c=job["c"], d=job["d"])
    result = trainer(model=model, dataset=dataset, max_epochs=args.max_epochs,
                     prm_options=_prm_options(args) if args.prm else None,
                     verbose=False, seed=args.seed, **_train_options(args))
    metrics = fit_metrics(model, dataset)

    record = {
        "label": label, "kind": job["generator"], "seed": job["seed"], "init_seed": job["init_seed"],
        "n_moved": job["n_moved"], **{k: float(v) for k, v in metrics.items()},
        "feasible": bool(result["prm_profile"]["feasible"]),
        "epochs": int(result["elapsed_epoch"]), "returned_epoch": int(result["returned_epoch"]),
        "minutes": float(result["elapsed_time"] / 60),
        "c": [int(v) for v in job["c"]], "d": [int(v) for v in job["d"]],
    }
    torch.save(model.state_dict(), out_dir / f"{label}.pt")
    path.write_text(json.dumps(record), encoding="utf-8")
    return record


def summarize(stored: dict, records: list[dict], out_dir: Path) -> None:
    cold = [r for r in records if r["kind"] == "selected"]
    perm = [r for r in records if r["kind"] == "permuted"]
    base = float(np.mean([r["nll_total"] for r in cold])) if cold else stored["nll_total"]
    base_name = "mean of the cold-start fits of the selected assignment" if cold else "the stored per-item fit"

    rows = [{"label": "stored per-item fit (warm start)", "kind": "stored", **stored}] + cold + perm
    for r in rows:
        r["delta_nll"] = r["nll_total"] - base
        r["delta_bic"] = 2.0 * r["delta_nll"]
    fields = ["label", "kind", "n_moved", "nll_total", "nll_per_obs", "n_params", "aic", "bic", "delta_nll",
              "delta_bic", "feasible", "epochs", "minutes"]
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'fit':<36} {'moved':>5} {'NLL':>13} {'NLL/obs':>9} {'k':>5} {'d NLL':>9} {'d BIC':>9} {'feasible':>8}")
    for r in rows:
        print(f"{r['label']:<36} {str(r.get('n_moved', '')):>5} {r['nll_total']:>13.2f} {r['nll_per_obs']:>9.5f} "
              f"{int(r['n_params']):>5} {r['delta_nll']:>+9.1f} {r['delta_bic']:>+9.1f} {str(r.get('feasible', '')):>8}")
    print(f"(d NLL, d BIC: difference from {base_name}; positive is worse)")

    def line(name, group):
        if not group:
            return
        values = np.array([r["delta_bic"] for r in group])
        sd = values.std(ddof=1) if len(values) > 1 else float("nan")
        print(f"{name:<28} n = {len(values):>2} | d BIC mean {values.mean():+.1f}, sd {sd:.1f}, "
              f"min {values.min():+.1f}, max {values.max():+.1f}")

    print()
    line("selected, cold start", cold)
    line("permuted", perm)
    if cold and perm:
        worst_cold = max(r["nll_total"] for r in cold)
        above = sum(r["nll_total"] > worst_cold for r in perm)
        print(f"permuted fits with a higher NLL than every cold-start fit of the selected assignment: {above}/{len(perm)}")
    if any(not r["feasible"] for r in cold + perm):
        print("note: some fits ended PRM-infeasible; their NLL is not comparable with the feasible fits")
    print(f"saved {out_dir / 'summary.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=PACKAGE / DEFAULT_RUN, help="finished application run")
    parser.add_argument("--out-dir", type=Path, default=None, help="default: ./runs_reassign/<run name>")
    parser.add_argument("--permutations", type=int, default=10)
    parser.add_argument("--controls", type=int, default=3, help="cold-start refits of the selected assignment")
    parser.add_argument("--device", type=str, default=None, help="default: the device of the run")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    options = parser.parse_args()

    args, row = load_run(options.run, options.device)
    c, d = np.asarray(row["c"]), np.asarray(row["d"])
    out_dir = options.out_dir or HERE / "runs_reassign" / options.run.name
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = [{"arm": "MiRM", "study": "reassign", "generator": "selected", "seed": k, "init_seed": args.seed + 1000 + k,
             "c": c, "d": d, "n_moved": 0} for k in range(options.controls)]
    for k in range(options.permutations):
        order = np.random.default_rng(k).permutation(len(c))
        moved = int(((c[order] != c) | (d[order] != d)).sum())
        jobs.append({"arm": "MiRM", "study": "reassign", "generator": "permuted", "seed": k,
                     "init_seed": args.seed + k, "c": c[order], "d": d[order], "n_moved": moved})

    print(f"package {PACKAGE}\nrun     {options.run}\ndata    {args.data_dir / args.data_name / args.data_fname}\n"
          f"items {len(c)}, PRM {args.prm}, max epochs {args.max_epochs} (+{args.additional_epochs}), "
          f"device {args.device}\n============ [ {len(jobs)} fits, {options.num_workers} workers ] ============", flush=True)

    records = map_parallel(partial(fit, args, out_dir), jobs, max_workers=options.num_workers,
                           threads_per_worker=options.threads_per_worker, verbose=True)
    summarize({k: float(v) for k, v in row["fit_metrics"].items()}, records, out_dir)


if __name__ == "__main__":
    main()
