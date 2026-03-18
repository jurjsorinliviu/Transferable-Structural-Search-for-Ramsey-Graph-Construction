from __future__ import annotations

import argparse
import json
from dataclasses import replace

from experiment_config import default_config
from run_psi_ramsey_experiments import available_suites, run_all, run_suite, run_suite_replicates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PSI-Ramsey paper experiment suites.")
    parser.add_argument(
        "--suite",
        default="all",
        choices=["all", *available_suites()],
        help="Experiment suite to run.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a lightweight smoke version with fewer witnesses and samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the default random seed.",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=1,
        help="Number of repeated runs with incremented seeds for the selected suite.",
    )
    parser.add_argument("--sampled-subsets", type=int, default=None, help="Override the sampled subset count.")
    parser.add_argument("--search-iterations", type=int, default=None, help="Override the local-search iteration budget.")
    parser.add_argument("--teacher-top-k", type=int, default=None, help="Override the number of teacher motifs kept.")
    parser.add_argument("--max-shift-pool", type=int, default=None, help="Override the extracted structure pool size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = default_config()
    if args.seed is not None:
        config.random_seed = args.seed
    overrides = {}
    if args.sampled_subsets is not None:
        overrides["sampled_subsets"] = args.sampled_subsets
    if args.search_iterations is not None:
        overrides["search_iterations"] = args.search_iterations
    if args.teacher_top_k is not None:
        overrides["teacher_top_k"] = args.teacher_top_k
    if args.max_shift_pool is not None:
        overrides["max_shift_pool"] = args.max_shift_pool
    if overrides:
        config = replace(config, **overrides)

    if args.suite == "all":
        if args.replicates != 1:
            raise ValueError("--replicates is only supported with a single suite, not --suite all")
        reports = run_all(config=config, smoke=args.smoke)
        summary = {suite: len(report["results"]) for suite, report in reports.items()}
        print(json.dumps({"suite": "all", "rows_per_suite": summary, "smoke": args.smoke}, indent=2))
        return

    if args.replicates > 1:
        summary = run_suite_replicates(config=config, suite_name=args.suite, smoke=args.smoke, replicates=args.replicates)
        print(
            json.dumps(
                {
                    "suite": args.suite,
                    "replicates": args.replicates,
                    "smoke": args.smoke,
                    "base_seed": config.random_seed,
                    "output_dir": str(config.results_dir),
                },
                indent=2,
            )
        )
        return

    report = run_suite(config=config, suite_name=args.suite, smoke=args.smoke)
    print(
        json.dumps(
            {
                "suite": args.suite,
                "targets": len(report["results"]),
                "smoke": args.smoke,
                "output_dir": report["config"]["results_dir"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
