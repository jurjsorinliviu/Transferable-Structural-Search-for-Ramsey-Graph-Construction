from __future__ import annotations

import random
from dataclasses import dataclass

from experiment_config import ExperimentConfig
from ramsey_circulant import shift_to_bin
from ramsey_data import RamseyWitness
from ramsey_search import SearchResult, SearchStep, circulant_local_search
from ramsey_teacher import TeacherRepresentation


@dataclass(slots=True)
class TrajectoryDistillation:
    accepted_add: dict[int, float]
    accepted_remove: dict[int, float]
    rejected_add: dict[int, float]
    rejected_remove: dict[int, float]


def perturb_initial_shifts(base_shifts: list[int], n: int, rng: random.Random, swaps: int) -> list[int]:
    current = set(base_shifts)
    pool = list(range(1, (n // 2) + 1))
    for _ in range(swaps):
        if current and rng.random() < 0.5:
            current.remove(rng.choice(list(current)))
        available = [shift for shift in pool if shift not in current]
        if available:
            current.add(rng.choice(available))
    return sorted(current)


def collect_teacher_search_traces(
    witness: RamseyWitness,
    teacher: TeacherRepresentation,
    config: ExperimentConfig,
    rng: random.Random,
) -> list[SearchResult]:
    traces: list[SearchResult] = []
    base_shifts = teacher.top_shifts[: max(1, round(witness.density * (witness.n - 1) / 2))]
    for run_index in range(config.teacher_trace_runs):
        run_rng = random.Random(rng.randint(0, 10**9))
        initial_shifts = perturb_initial_shifts(base_shifts, witness.n, run_rng, config.teacher_trace_perturbations)
        traces.append(
            circulant_local_search(
                target=witness,
                initial_shifts=initial_shifts,
                prioritized_shifts=teacher.top_shifts,
                initial_corrections=None,
                samples=config.teacher_trace_samples,
                iterations=config.teacher_trace_iterations,
                accept_slack=config.search_accept_slack,
                explore_prob=config.search_explore_prob,
                correction_ratio=config.search_correction_ratio,
                max_corrections=config.search_max_corrections,
                normalized_bins=config.normalized_shift_bins,
                pair_weights=None,
                operator_bias=None,
                rng=run_rng,
            )
        )
    return traces


def distill_teacher_trajectories(
    traces: list[SearchResult],
    witness_n: int,
    normalized_bins: int,
) -> TrajectoryDistillation:
    accepted_add: dict[int, float] = {}
    accepted_remove: dict[int, float] = {}
    rejected_add: dict[int, float] = {}
    rejected_remove: dict[int, float] = {}
    total = max(1, len(traces))
    for trace in traces:
        for step in trace.steps:
            for shift in step.added_shifts:
                shift_bin = shift_to_bin(shift, witness_n, normalized_bins)
                target = accepted_add if step.accepted else rejected_add
                target[shift_bin] = target.get(shift_bin, 0.0) + 1.0
            for shift in step.removed_shifts:
                shift_bin = shift_to_bin(shift, witness_n, normalized_bins)
                target = accepted_remove if step.accepted else rejected_remove
                target[shift_bin] = target.get(shift_bin, 0.0) + 1.0
    return TrajectoryDistillation(
        accepted_add={shift_bin: value / total for shift_bin, value in accepted_add.items()},
        accepted_remove={shift_bin: value / total for shift_bin, value in accepted_remove.items()},
        rejected_add={shift_bin: value / total for shift_bin, value in rejected_add.items()},
        rejected_remove={shift_bin: value / total for shift_bin, value in rejected_remove.items()},
    )
