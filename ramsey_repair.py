from __future__ import annotations

import random
from dataclasses import dataclass

from ramsey_data import (
    RamseyWitness,
    exact_count_k_cliques,
    exact_k_clique_edge_counts,
    subset_is_independent,
    violation_rates_from_subsets,
)
from ramsey_reconstruct import toggle_edge
from ramsey_search import build_sample_cache, sampled_objective


@dataclass(slots=True)
class RepairResult:
    adjacency: list[list[int]]
    initial_score: float
    final_score: float
    applied_steps: int
    exact_count_before: int
    exact_count_after: int


def clone_adjacency(adjacency: list[list[int]]) -> list[list[int]]:
    return [row[:] for row in adjacency]


def sampled_independent_nonedge_counts(
    adjacency: list[list[int]],
    independent_subsets: list[tuple[int, ...]],
) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for subset in independent_subsets:
        if not subset_is_independent(adjacency, subset):
            continue
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                edge = (subset[i], subset[j]) if subset[i] < subset[j] else (subset[j], subset[i])
                counts[edge] = counts.get(edge, 0) + 1
    return counts


def repair_priority_key(
    target: RamseyWitness,
    exact_count: int,
    rates: dict[str, float],
    score: float,
) -> tuple[float, ...]:
    if target.r <= 4:
        return (
            float(exact_count),
            float(rates["sampled_r_clique_rate"]),
            float(rates["sampled_s_independent_rate"]),
            float(score),
        )
    return (
        float(rates["sampled_r_clique_rate"]),
        float(rates["sampled_s_independent_rate"]),
        float(score),
    )


def greedy_exact_repair(
    target: RamseyWitness,
    adjacency: list[list[int]],
    samples: int,
    max_steps: int,
    candidate_edges: int,
    clique_cutoff: int,
    rng: random.Random,
) -> RepairResult:
    current = clone_adjacency(adjacency)
    clique_subsets, independent_subsets = build_sample_cache(target, samples, rng)
    current_score = sampled_objective(target, current, clique_subsets, independent_subsets)
    current_rates = violation_rates_from_subsets(current, clique_subsets, independent_subsets)
    current_exact = exact_count_k_cliques(current, target.r, cutoff=clique_cutoff) if target.r <= 4 else -1
    initial_score = current_score
    exact_before = current_exact
    applied_steps = 0

    for _ in range(max_steps):
        hotspot_counts: dict[tuple[int, int], int] = {}
        if target.r <= 4 and current_exact != 0:
            hotspot_counts = exact_k_clique_edge_counts(current, target.r, clique_cutoff=clique_cutoff)
        if not hotspot_counts:
            hotspot_counts = sampled_independent_nonedge_counts(current, independent_subsets)
        if not hotspot_counts:
            break

        current_key = repair_priority_key(target, current_exact, current_rates, current_score)
        best_key: tuple[float, ...] | None = None
        best_state: tuple[list[list[int]], float, dict[str, float], int] | None = None
        ranked_edges = sorted(hotspot_counts.items(), key=lambda item: (-item[1], item[0]))
        for (i, j), _ in ranked_edges[:candidate_edges]:
            candidate = clone_adjacency(current)
            toggle_edge(candidate, i, j)
            candidate_score = sampled_objective(target, candidate, clique_subsets, independent_subsets)
            candidate_rates = violation_rates_from_subsets(candidate, clique_subsets, independent_subsets)
            candidate_exact = exact_count_k_cliques(candidate, target.r, cutoff=clique_cutoff) if target.r <= 4 else -1
            candidate_key = repair_priority_key(target, candidate_exact, candidate_rates, candidate_score)
            if candidate_key < current_key and (best_key is None or candidate_key < best_key):
                best_key = candidate_key
                best_state = (candidate, candidate_score, candidate_rates, candidate_exact)

        if best_state is None:
            break
        current, current_score, current_rates, current_exact = best_state
        applied_steps += 1

    return RepairResult(
        adjacency=current,
        initial_score=initial_score,
        final_score=current_score,
        applied_steps=applied_steps,
        exact_count_before=exact_before,
        exact_count_after=current_exact,
    )
