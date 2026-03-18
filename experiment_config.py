from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_WITNESS_DIR = ROOT_DIR / "related work" / "ramsey_number_bounds" / "improved_bounds"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "psi_ramsey_experiment_results.json"
DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "psi_ramsey"


@dataclass(slots=True)
class ExperimentConfig:
    witness_dir: Path = DEFAULT_WITNESS_DIR
    output_path: Path = DEFAULT_OUTPUT_PATH
    results_dir: Path = DEFAULT_RESULTS_DIR
    random_seed: int = 1234
    sampled_subsets: int = 1500
    teacher_top_k: int = 16
    normalized_shift_bins: int = 24
    shared_support_threshold: float = 0.5
    max_shift_pool: int = 24
    max_shared_bins: int = 12
    contrastive_margin: float = 0.02
    pairwise_support_threshold: float = 0.5
    pairwise_bonus: float = 0.35
    max_pair_candidates: int = 48
    partition_blocks: int = 4
    partition_support_threshold: float = 0.5
    partition_contrastive_margin: float = 0.02
    partition_pair_bonus: float = 0.35
    target_same_r_only: bool = True
    teacher_trace_runs: int = 2
    teacher_trace_iterations: int = 80
    teacher_trace_samples: int = 300
    teacher_trace_perturbations: int = 3
    search_iterations: int = 600
    search_accept_slack: float = 0.01
    search_explore_prob: float = 0.08
    search_correction_ratio: float = 0.035
    search_max_corrections: int = 8
    transfer_refine_iterations: int = 180
    transfer_refine_max_initial_corrections: int = 12
    transfer_refine_correction_ratio: float = 0.03
    transfer_refine_max_corrections: int = 6
    transfer_motif_polish_iterations: int = 60
    transfer_distribution_trials: int = 3
    transfer_distribution_repair_steps: int = 12
    transfer_distribution_repair_steps_r3: int = 24
    transfer_distribution_repair_edge_candidates: int = 24
    exact_supervision_iterations: int = 90
    exact_supervision_candidate_edges: int = 18
    exact_supervision_shift_candidates: int = 6
    exact_supervision_pair_candidates: int = 12
    transfer_selection_alignment_weight: float = 0.08
    transfer_selection_correction_weight: float = 0.01
    transfer_selection_objective_tolerance: float = 0.12
    portfolio_alignment_weight: float = 0.0
    exact_repair_steps: int = 6
    exact_repair_edge_candidates: int = 10
    exact_repair_clique_cutoff: int = 8000
    metadata: dict[str, str] = field(
        default_factory=lambda: {
            "paper": "PSI-Ramsey",
            "prototype": "root-level structural transfer scaffold",
            "inspiration": "PSI-NN only",
        }
    )


def default_config() -> ExperimentConfig:
    return ExperimentConfig()
