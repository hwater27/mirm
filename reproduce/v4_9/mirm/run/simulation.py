import argparse
from pathlib import Path
from functools import partial

from mirm.run.application import add_training_arguments
from mirm.run.utils.artifacts import run_meta, analyze, save_results, load_results, precision_tables
from mirm.run.utils.parallel import map_parallel
from mirm.run.utils.jobs import run_job


def parse():
    parser = argparse.ArgumentParser(description='Run MiRM simulation study',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Endpoint settings
    parser.add_argument("--out-path", type=Path, default=None)
    parser.add_argument("--data-name", type=str, choices=["synthetic"], default="synthetic")
    parser.add_argument("--data-fname", type=str, choices=["smoke", "pilot", "heavy", "from", "custom"], required=True)
    parser.add_argument("--out-dir", type=Path, default="./runs/simulation")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=30)
    parser.add_argument("--analysis-workers", type=int, default=10,
                        help="worker processes of the analysis (CPU; the fits use --num-workers)")
    parser.add_argument("--threads-per-worker", type=int, default=1)

    # Data settings
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--generators", nargs="+", type=str, default=["2PL", "3PL", "LPE", "MiRM", "Spline"])
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--n-students", type=int)
    parser.add_argument("--n-items", type=int)
    # Monotonicity
    parser.add_argument("--run-monotonicity", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--monotonicity-replicates", type=int, default=10)
    parser.add_argument("--monotonicity-densities", nargs="+", type=float)
    parser.add_argument("--class-spreads", nargs="+", type=float)
    parser.add_argument("--treat-spreads", nargs="+", type=float)
    parser.add_argument("--loading-scales", nargs="+", type=float)
    # Sensitivity
    parser.add_argument("--run-sensitivity", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sensitivity-n-students", nargs="+", type=int)
    parser.add_argument("--sensitivity-n-items", nargs="+", type=int)
    # Parameter and configuration recovery: items mixed from (1,1), (2,3), (3,1) in fixed proportions
    parser.add_argument("--run-recovery", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--recovery-replicates", type=int, default=30)

    # Model settings
    parser.add_argument("--arms", nargs="+", type=str, default=["2PL", "3PL", "LPE", "MiRM"])
    parser.add_argument("--n-quad", type=int, default=41)

    add_training_arguments(parser)

    # Plot settings
    parser.add_argument("--plot-nodes", type=int, default=401)
    parser.add_argument("--plot-half-bound", type=float, default=4.0)
    parser.add_argument("--max-ic", type=int, default=300)

    # Monotonicity analysis
    parser.add_argument("--mono-tol", type=float, default=0.05)

    # 'from' profile: the heavy design with items taken from fitted empirical models
    parser.add_argument("--from-runs", nargs="+", type=Path, default=None,
                        help="finished application runs whose fitted items form the bank (profile 'from'): the MiRM "
                             "generators use --from-model, the 2PL/3PL/LPE/Spline generators the full-data fit of "
                             "the same arm")
    parser.add_argument("--from-model", type=str, default="MiRM_per_item", help="fitted model the MiRM items come from")
    parser.add_argument("--from-mono-tol", type=float, default=0.01, help="largest ICC drop of a monotone bank item")
    parser.add_argument("--from-dip-min", type=float, default=0.05, help="smallest ICC drop of a bank dip item")
    parser.add_argument("--from-dip-max", type=float, default=0.30, help="largest ICC drop of a bank dip item")

    return parser.parse_args()


def _default_profile(args: argparse.Namespace) -> argparse.Namespace:
    # generators of the MiRM-based studies; the 'from' profile swaps in the empirical item bank for every generator
    args.mirm_generator, args.recovery_generator, args.dip_generator = "MiRM", "MiRM_mix", "dip"
    if args.data_fname == "smoke":
        args.replicates = 2
        args.generators = ["2PL", "MiRM"]
        args.density = 1.0
        args.n_students = 128
        args.n_items = 8
        args.run_monotonicity = False
        args.run_sensitivity = False
        args.run_recovery = True
        args.recovery_replicates = 1
        args.max_epochs = 3
        args.select_max_epochs = 6
    elif args.data_fname == "pilot":
        args.replicates = 5
        args.generators = ["2PL", "3PL", "MiRM"]
        args.density = 1.0
        args.n_students = 1024
        args.n_items = 32
        args.run_monotonicity = True
        args.monotonicity_replicates = 3
        args.monotonicity_densities = [0.5, 1.0]
        args.class_spreads = [0.75]
        args.treat_spreads = [1.0]
        args.loading_scales = [0.5]
        args.run_sensitivity = True
        args.sensitivity_n_students = [128, 256, 512]
        args.sensitivity_n_items = [8, 16]
        args.run_recovery = True
        args.recovery_replicates = 3
        args.max_epochs = 20
        args.select_max_epochs = 100
    elif args.data_fname in ("heavy", "from"):
        args.replicates = 30
        args.generators = ["2PL", "3PL", "LPE", "MiRM", "Spline"]
        args.density = 1.0
        args.n_students = 4096
        args.n_items = 128
        args.run_monotonicity = True
        args.monotonicity_replicates = 10
        args.monotonicity_densities = [0.2, 0.5, 1.0]
        args.class_spreads = [0.0, 0.75, 1.5]
        args.treat_spreads = [0.0, 1.0, 2.0]
        args.loading_scales = [0.0, 0.5, 1.0]
        args.run_sensitivity = True
        args.sensitivity_n_students = [128, 256, 512, 1024, 2048]
        args.sensitivity_n_items = [8, 16, 32, 64]
        args.run_recovery = True
        args.recovery_replicates = 30
        args.max_epochs = 500
    if args.data_fname == "from":
        if not args.from_runs:
            raise ValueError("profile 'from' requires --from-runs")
        args.generators = [f"{g}_from" for g in args.generators]   # every generator takes its items from the runs
        args.mirm_generator, args.recovery_generator, args.dip_generator = "MiRM_from", "MiRM_from_mix", "dip_from"
    elif args.data_fname == "custom":
        if args.n_students is None or args.n_items is None:
            raise ValueError("custom requires --n-students and --n-items")
        if args.run_monotonicity and not args.monotonicity_densities:
            raise ValueError("custom monotonicity requires --monotonicity-densities")
        if args.run_sensitivity and (
            not args.sensitivity_n_students or not args.sensitivity_n_items
        ):
            raise ValueError("custom sensitivity requires n-students and n-items grids")

    return args


MONOTONICITY_SKIP = ("3PL", "LPE")  # arms not fitted in the monotonicity study


def _arm_jobs(args, *, seed, generator, density, n_students, n_items, study, **others):
    """MiRM is fitted as (2,2) in every study (with and without PRM in the monotonicity study); the recovery study
    adds the per-item (c,d) selection. The monotonicity study skips 3PL and LPE (monotone by construction, like 2PL,
    which is kept as the monotone reference)."""
    jobs = []
    for arm in args.arms:
        if study == "monotonicity" and arm in MONOTONICITY_SKIP:
            continue
        base = {
            "prediction": False,
            "seed": seed,
            "generator": generator,
            "density": density,
            "n_students": n_students,
            "n_items": n_items,
            "study": study,
            **others,
        }
        if arm == "MiRM":
            if study == "monotonicity":
                for prm in [True, False]:
                    jobs.append({**base, "arm": arm, "c": 2, "d": 2, "explore": False, "prm": prm})
            else:
                jobs.append({**base, "arm": arm, "c": 2, "d": 2, "explore": False})
            if study == "recovery":
                jobs.append({**base, "arm": arm, "select": True, "explore": False})
        else:
            jobs.append({**base, "arm": arm, "explore": False})
    return jobs


def main():
    args = parse()
    args = _default_profile(args)

    out_path = args.out_path
    if out_path is not None:
        if (out_path / "from_bank.pt").exists():
            args.from_bank = str(out_path / "from_bank.pt")
        results = load_results(args, out_path)
    else:
        out_path = run_meta(args)
        if args.data_fname == "from":
            from mirm.run.utils.from_bank import build_bank
            build_bank(args.from_runs, model_name=args.from_model, mono_tol=args.from_mono_tol,
                       dip_min=args.from_dip_min, dip_max=args.from_dip_max, out_path=out_path / "from_bank.pt",
                       parametric=tuple(g.removesuffix("_from") for g in args.generators if g != "MiRM_from"))
            args.from_bank = str(out_path / "from_bank.pt")

        args.worst_m = None
        args.form_masks = None

        jobs = []
        # recovery first: its per-item selections are the longest jobs
        if args.run_recovery:
            for seed in range(args.recovery_replicates):
                jobs.extend(_arm_jobs(
                    args,
                    seed=seed,
                    generator=args.recovery_generator,
                    density=args.density,
                    n_students=args.n_students,
                    n_items=args.n_items,
                    study="recovery",
                ))

        for seed in range(args.replicates):
            for generator in args.generators:
                jobs.extend(_arm_jobs(
                    args,
                    seed=seed,
                    generator=generator,
                    density=args.density,
                    n_students=args.n_students,
                    n_items=args.n_items,
                    study="precision",
                ))

        if args.run_monotonicity:
            for seed in range(args.monotonicity_replicates):
                for density in args.monotonicity_densities:
                    for class_spread in args.class_spreads:
                        for treat_spread in args.treat_spreads:
                            for loading_scale in args.loading_scales:
                                dip_structure = {
                                    "class_spread": class_spread,
                                    "treat_spread": treat_spread,
                                    "loading_scale": loading_scale,
                                }
                                jobs.extend(_arm_jobs(
                                    args,
                                    seed=seed,
                                    generator=args.dip_generator,
                                    density=density,
                                    n_students=args.n_students,
                                    n_items=args.n_items,
                                    study="monotonicity",
                                    **dip_structure,
                                ))

        if args.run_sensitivity:
            for seed in range(args.replicates):

                n_students = args.n_students
                for n_items in args.sensitivity_n_items:
                    jobs.extend(_arm_jobs(
                        args,
                        seed=seed,
                        generator=args.mirm_generator,
                        density=args.density,
                        n_students=n_students,
                        n_items=n_items,
                        study="sensitivity",
                    ))

                n_items = args.n_items
                for n_students in args.sensitivity_n_students:
                    jobs.extend(_arm_jobs(
                        args,
                        seed=seed,
                        generator=args.mirm_generator,
                        density=args.density,
                        n_students=n_students,
                        n_items=n_items,
                        study="sensitivity",
                    ))

        jobs = sorted(jobs, key=lambda job: not job.get("select", False))  # per-item selections first (stable)
        results = map_parallel(
            partial(run_job, args),
            jobs,
            max_workers=args.num_workers,
            threads_per_worker=args.threads_per_worker,
            verbose=args.verbose,
        )

        save_results(args, out_path, results, None)

    artifact = analyze(args, results)
    if args.verbose:
        print(precision_tables(artifact["tables"]["full"])[0])

    save_results(args, out_path, results, artifact)
    if args.verbose:
        print(f"\nRun completed. Results saved to {out_path}")


if __name__ == "__main__":
    main()
