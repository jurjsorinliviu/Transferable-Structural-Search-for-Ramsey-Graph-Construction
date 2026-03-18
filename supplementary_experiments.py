from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from experiment_config import ExperimentConfig, default_config
from run_psi_ramsey_experiments import run_suite, run_suite_replicates


SUPPLEMENTARY_RESULTS_DIR = Path("results") / "psi_ramsey_supplementary"


@dataclass(slots=True)
class SupplementaryExperiment:
    name: str
    suite: str
    description: str
    reviewer_goal: str
    config_overrides: dict[str, object] = field(default_factory=dict)
    replicates: int = 1
    smoke_supported: bool = True


def build_experiment_plan() -> list[SupplementaryExperiment]:
    return [
        SupplementaryExperiment(
            name="transfer_seed_robustness",
            suite="transfer",
            replicates=8,
            description=(
                "Repeated transfer runs across incremented seeds to show that the balanced-transfer "
                "story is not a single-seed artifact."
            ),
            reviewer_goal="Stability and repeatability of the main transfer claim.",
        ),
        SupplementaryExperiment(
            name="search_seed_robustness",
            suite="search",
            replicates=8,
            description=(
                "Repeated search-suite runs to test whether the search ranking remains competitive "
                "under seed variation."
            ),
            reviewer_goal="Stability of the search-side conclusions.",
        ),
        SupplementaryExperiment(
            name="transfer_compute_low_budget",
            suite="transfer",
            config_overrides={
                "sampled_subsets": 900,
                "search_iterations": 320,
                "transfer_refine_iterations": 90,
            },
            description=(
                "Low-budget transfer stress test with reduced search and sampling budgets."
            ),
            reviewer_goal="Check whether the method degrades gracefully under tighter budgets.",
        ),
        SupplementaryExperiment(
            name="transfer_compute_high_budget",
            suite="transfer",
            config_overrides={
                "sampled_subsets": 2200,
                "search_iterations": 900,
                "transfer_refine_iterations": 260,
            },
            description=(
                "High-budget transfer stress test to examine whether the ranking remains favorable "
                "when more refinement is allowed."
            ),
            reviewer_goal="Budget-sensitivity analysis for the transfer claim.",
        ),
        SupplementaryExperiment(
            name="search_compute_high_budget",
            suite="search",
            config_overrides={
                "sampled_subsets": 2200,
                "search_iterations": 950,
                "transfer_refine_iterations": 240,
            },
            description=(
                "Higher-budget search suite to test whether the main search ordering changes under "
                "more aggressive refinement."
            ),
            reviewer_goal="Budget-sensitivity analysis for the search claim.",
        ),
        SupplementaryExperiment(
            name="mixed_r_transfer_neighborhood",
            suite="transfer",
            config_overrides={
                "target_same_r_only": False,
            },
            description=(
                "Transfer-neighborhood stress test using mixed-r teacher pools rather than same-r-only pools."
            ),
            reviewer_goal="Directly test how local the transferable structure really is.",
        ),
        SupplementaryExperiment(
            name="high_resolution_structure",
            suite="transfer",
            config_overrides={
                "normalized_shift_bins": 32,
                "teacher_top_k": 20,
                "max_shift_pool": 32,
            },
            description=(
                "Higher-resolution structural distillation with more bins and a larger extracted motif pool."
            ),
            reviewer_goal="Sensitivity to structural resolution choices.",
        ),
        SupplementaryExperiment(
            name="compact_structure",
            suite="transfer",
            config_overrides={
                "normalized_shift_bins": 16,
                "teacher_top_k": 10,
                "max_shift_pool": 16,
                "max_shared_bins": 8,
            },
            description=(
                "Compact structural distillation with fewer bins and a smaller motif pool."
            ),
            reviewer_goal="Check whether conclusions persist under a more compressed representation.",
        ),
        SupplementaryExperiment(
            name="exact_supervision_stress",
            suite="transfer",
            config_overrides={
                "exact_supervision_iterations": 150,
                "transfer_distribution_repair_steps_r3": 36,
                "exact_repair_steps": 10,
            },
            description=(
                "Stronger exact-supervision setting for weak R(3,s) cells."
            ),
            reviewer_goal="Show how much the weak-cell behavior depends on exact supervision strength.",
        ),
        SupplementaryExperiment(
            name="ablation_seed_robustness",
            suite="ablations",
            replicates=6,
            description=(
                "Replicated ablation runs to verify that the component ordering is not seed-fragile."
            ),
            reviewer_goal="Strengthen confidence in the ablation narrative.",
        ),
        SupplementaryExperiment(
            name="interpretability_seed_robustness",
            suite="interpretability",
            replicates=6,
            description=(
                "Replicated interpretability/frontier runs to test the stability of the structure-validity separation."
            ),
            reviewer_goal="Stability of the structure-validity frontier.",
        ),
    ]


def apply_overrides(config: ExperimentConfig, overrides: dict[str, object]) -> ExperimentConfig:
    return replace(config, **overrides) if overrides else config


def experiment_output_dir(name: str) -> Path:
    return SUPPLEMENTARY_RESULTS_DIR / name


def serialize_plan(plan: list[SupplementaryExperiment]) -> list[dict[str, object]]:
    serialized = []
    for item in plan:
        raw = asdict(item)
        serialized.append(raw)
    return serialized


def write_plan_file(plan: list[SupplementaryExperiment], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"experiments": serialize_plan(plan)}, indent=2), encoding="utf-8")


def run_experiment(exp: SupplementaryExperiment, smoke: bool) -> dict[str, object]:
    config = default_config()
    config = apply_overrides(config, exp.config_overrides)
    config = replace(config, results_dir=experiment_output_dir(exp.name))

    if exp.replicates > 1:
        summary = run_suite_replicates(config=config, suite_name=exp.suite, smoke=smoke, replicates=exp.replicates)
        return {
            "name": exp.name,
            "suite": exp.suite,
            "mode": "replicates",
            "replicates": exp.replicates,
            "results_dir": str(config.results_dir),
            "description": exp.description,
            "reviewer_goal": exp.reviewer_goal,
            "config": {
                **exp.config_overrides,
                "random_seed": config.random_seed,
                "results_dir": str(config.results_dir),
            },
            "summary_file": str(config.results_dir / exp.suite / "replicates_summary.json"),
        }

    report = run_suite(config=config, suite_name=exp.suite, smoke=smoke)
    return {
        "name": exp.name,
        "suite": exp.suite,
        "mode": "single",
        "replicates": 1,
        "results_dir": str(config.results_dir),
        "description": exp.description,
        "reviewer_goal": exp.reviewer_goal,
        "config": {
            **exp.config_overrides,
            "random_seed": config.random_seed,
            "results_dir": str(config.results_dir),
        },
        "report_file": str(config.results_dir / exp.suite / "report.json"),
        "targets": len(report["results"]),
    }


def overview_markdown(results: list[dict[str, object]]) -> str:
    lines = [
        "# Supplementary Experiments",
        "",
        "This directory contains reviewer-oriented supplementary experiments that extend the main PSI-Ramsey evaluation without changing the main-manuscript result tables.",
        "",
        "## Blocks",
    ]
    for item in results:
        lines.append(f"- `{item['name']}` ({item['suite']})")
        lines.append(f"  Goal: {item['reviewer_goal']}")
        lines.append(f"  Description: {item['description']}")
        lines.append(f"  Output dir: `{item['results_dir']}`")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PSI-Ramsey supplementary reviewer-oriented experiments.")
    parser.add_argument(
        "--experiment",
        default="all",
        help="Experiment name to run, or 'all'.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run lightweight smoke versions where supported.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the supplementary experiment plan and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_experiment_plan()
    write_plan_file(plan, SUPPLEMENTARY_RESULTS_DIR / "supplementary_experiment_plan.json")

    if args.list:
        print(json.dumps({"experiments": serialize_plan(plan)}, indent=2))
        return

    selected = plan if args.experiment == "all" else [item for item in plan if item.name == args.experiment]
    if not selected:
        names = ", ".join(item.name for item in plan)
        raise ValueError(f"Unknown supplementary experiment '{args.experiment}'. Available: {names}")

    results = [run_experiment(item, smoke=args.smoke and item.smoke_supported) for item in selected]
    SUPPLEMENTARY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (SUPPLEMENTARY_RESULTS_DIR / "supplementary_results_manifest.json").write_text(
        json.dumps({"results": results}, indent=2),
        encoding="utf-8",
    )
    (SUPPLEMENTARY_RESULTS_DIR / "overview.md").write_text(overview_markdown(results), encoding="utf-8")
    print(json.dumps({"ran": [item["name"] for item in results], "smoke": args.smoke}, indent=2))


if __name__ == "__main__":
    main()
