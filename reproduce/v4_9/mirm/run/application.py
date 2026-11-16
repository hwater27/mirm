import argparse
from pathlib import Path
from functools import partial

from mirm.run.utils.artifacts import run_meta, analyze, save_results, load_results, summary_table
from mirm.run.utils.parallel import map_parallel
from mirm.run.utils.jobs import run_job


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    """Training, PRM and (c,d)-selection options shared by the application and the simulation."""
    # Training settings (sparse direct MML with L-BFGS)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--history-size", type=int, default=100)
    parser.add_argument("--early-stop-tol", type=float, default=1e-9)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--chunk-students", type=int, default=0, help=">0 fixed chunk size, 0 automatic, <0 never chunk")
    parser.add_argument("--chunk-mem-gb", type=float, default=12.0)
    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu, cuda, cuda:N or cuda:0,cuda:1 (training only; workers are spread over the listed "
                             "devices; analysis stays on the CPU)")
    # Monotonicity settings
    parser.add_argument("--prm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--PHR-strength", type=float, default=1.0)
    parser.add_argument("--force-feasible", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--additional-epochs", type=int, default=500)
    parser.add_argument("--prm-nodes", type=int, default=0,
                        help="Gauss-Hermite order of the fixed PRM nodes of every other PRM fit (0: --n-quad)")
    parser.add_argument("--prm-cut-scan", type=int, default=32,
                        help="interior scan points per gap when the PRM cuts move")
    parser.add_argument("--certify-nodes", type=int, default=1601)
    parser.add_argument("--certify-tolerance", type=float, default=1e-3)
    parser.add_argument("--certify-depth", type=int, default=12)
    # Per-item (c,d) selection (cell rule) and the refit
    parser.add_argument("--select-cap", type=int, default=5)
    parser.add_argument("--select-max-epochs", type=int, default=5000,
                        help="epoch budget of all selection rounds together")
    parser.add_argument("--select-pool", choices=["geo", "arith"], default="geo")
    parser.add_argument("--select-stable", type=int, default=2,
                        help="consecutive rounds without a change that end the selection")
    parser.add_argument("--select-early-stop-tol", type=float, default=1e-7,
                        help="early-stopping tolerance of every selection round")
    parser.add_argument("--select-prm-nodes", type=int, default=0,
                        help="Gauss-Hermite order of the fixed PRM nodes of the selection rounds (0: 2 * --n-quad)")
    parser.add_argument("--select-prm-tol", type=float, default=1e-5,
                        help="training feasibility tolerance of the selection rounds on the normalized PRM margins")


def parse():
    parser = argparse.ArgumentParser(description='Run MiRM empirical study',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Endpoint settings
    parser.add_argument("--out-path", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default="../../../data/processed")
    parser.add_argument("--data-name", type=str, choices=["pisa_2015", "pisa_2018", "ednet", "eedi"], required=True)
    parser.add_argument("--data-fname", type=str, required=True)
    parser.add_argument("--top-items", type=int, default=None,
                        help="analyze only the most answered items (default: 100 for EdNet, all items otherwise)")
    parser.add_argument("--out-dir", type=Path, default="./runs/application")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--analysis-workers", type=int, default=10,
                        help="worker processes of the analysis (CPU; the fits use --num-workers)")
    parser.add_argument("--threads-per-worker", type=int, default=1)

    # Model settings
    parser.add_argument("--arms", nargs="+", type=str,
                        default=["2PL", "3PL", "LPE", "RH", "3PLQ", "Spline", "MiRM (2,2)", "MiRM (5,5)", "MiRM per-item"])
    parser.add_argument("--n-quad", type=int, default=41)
    parser.add_argument("--spline-basis", type=int, default=18)
    parser.add_argument("--grid", action=argparse.BooleanOptionalAction, default=False,
                        help="also fit MiRM (1,1)..(max,max) on the full data (no prediction) for the (c,d) heatmaps")
    parser.add_argument("--grid-max", type=int, default=6)

    # Prediction settings
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--k-fold", type=int, default=5)
    parser.add_argument("--test-seeds", nargs="*", type=int, default=[1, 3, 8, 0, 6, 4, 9],  # Boltzmann constant
                        help="prediction splits; '--test-seeds' with no value fits the full data only")
    parser.add_argument("--prm-conditions", nargs="+", choices=["prm", "no-prm"], default=None,
                        help="fit every job under each listed condition in one worker pool and save one run per "
                             "condition (tag suffix _prm / _noprm); default: the single condition of --prm")

    add_training_arguments(parser)
    parser.add_argument("--worst-m", type=int, default=None)
    parser.add_argument("--form-masks", type=Path, default=None)

    # Plot settings
    parser.add_argument("--plot-nodes", type=int, default=401)
    parser.add_argument("--plot-half-bound", type=float, default=4.0)
    parser.add_argument("--max-ic", type=int, default=300)

    # Monotonicity analysis
    parser.add_argument("--check-item-type", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-intercept", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mono-tol", type=float, default=0.05)
    parser.add_argument("--dip-bins", type=int, default=20)
    parser.add_argument("--dip-tail-bins", type=int, default=1)

    args = parser.parse_args()
    if args.top_items is None and args.data_name == "ednet":
        args.top_items = 100
    return args


def arm_job(args, arm: str) -> dict:
    """'MiRM (c,d)' -> fixed configuration, 'MiRM per-item' -> (c,d) selection, other names -> baselines."""
    if arm == "MiRM per-item":
        return {"arm": "MiRM", "select": True}
    if arm.startswith("MiRM"):
        c, d = (int(v) for v in arm[arm.index("(") + 1:arm.index(")")].split(","))
        return {"arm": "MiRM", "c": c, "d": d}
    if arm == "Spline":
        return {"arm": arm, "n_basis": args.spline_basis}
    return {"arm": arm}


def build_jobs(args) -> list[dict]:
    jobs = []
    for prediction, seed in [(False, None), *((True, s) for s in args.test_seeds)]:
        for arm in args.arms:
            jobs.append({**arm_job(args, arm), "prediction": prediction, "seed": seed, "explore": False})
    if args.grid:
        for c in range(1, args.grid_max + 1):
            for d in range(1, args.grid_max + 1):
                jobs.append({"arm": "MiRM", "c": c, "d": d, "prediction": False, "seed": None, "explore": True})
    return jobs


def finish(args, out_path, results) -> None:
    save_results(args, out_path, results, None)
    artifact = analyze(args, results)
    if args.verbose:
        print(summary_table(artifact["tables"]["main"]))
    save_results(args, out_path, results, artifact)
    if args.verbose:
        print(f"Run completed. Results saved to {out_path}")


def main():
    args = parse()

    if args.out_path is not None:
        results = load_results(args, args.out_path)
        artifact = analyze(args, results)
        if args.verbose:
            print(summary_table(artifact["tables"]["main"]))
        save_results(args, args.out_path, results, artifact)
        if args.verbose:
            print(f"Run completed. Results saved to {args.out_path}")
        return

    conditions = args.prm_conditions or ["prm" if args.prm else "no-prm"]
    split = args.prm_conditions is not None
    runs = {}
    for condition in (conditions if split else []):
        run_args = argparse.Namespace(**{**vars(args), "prm": condition == "prm", "prm_conditions": None,
                                         "tag": f"{args.tag}_{condition.replace('-', '')}"})
        runs[condition] = (run_args, run_meta(run_args))
    out_path = None if split else run_meta(args)
    base = build_jobs(args)
    # interleave the conditions so that both start at once (e.g., PRM and no-PRM per-item fits on two workers)
    jobs = [{**job, "prm": condition == "prm"} if split else job for job in base for condition in conditions]
    if args.verbose:
        print(f"============ [ {len(jobs)} jobs, {args.num_workers} workers ] ============")

    results = map_parallel(
        partial(run_job, args),
        jobs,
        max_workers=args.num_workers,
        threads_per_worker=args.threads_per_worker,
        verbose=args.verbose,
    )

    if not split:
        finish(args, out_path, results)
        return
    for condition, (run_args, run_path) in runs.items():
        rows = [{k: v for k, v in row.items() if k != "prm"} for row in results if row["prm"] == (condition == "prm")]
        finish(run_args, run_path, rows)


if __name__ == "__main__":
    main()
