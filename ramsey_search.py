from __future__ import annotations

import math
import random
from dataclasses import dataclass

from ramsey_circulant import shift_to_bin
from ramsey_data import RamseyWitness, cyclic_shift_profile, exact_count_k_cliques, exact_k_clique_edge_counts, subset_is_clique, subset_is_independent
from ramsey_metrics import graph_density
from ramsey_reconstruct import estimate_shift_budget, graph_from_shifts_and_corrections


@dataclass(slots=True)
class SearchResult:
    adjacency: list[list[int]]
    shifts: list[int]
    corrections: list[tuple[int, int]]
    steps: list["SearchStep"]
    initial_score: float
    best_score: float
    accepted_moves: int
    iterations: int


@dataclass(slots=True)
class SearchStep:
    accepted: bool
    move_kind: str
    added_shifts: list[int]
    removed_shifts: list[int]
    score_before: float
    score_after: float


@dataclass(slots=True)
class ExactAdjacencySearchResult:
    adjacency: list[list[int]]
    steps: list["SearchStep"]
    initial_score: float
    best_score: float
    accepted_moves: int
    iterations: int
    exact_count: int


def sampled_objective(
    target: RamseyWitness,
    adjacency: list[list[int]],
    clique_subsets: list[tuple[int, ...]],
    independent_subsets: list[tuple[int, ...]],
) -> float:
    clique_hits = sum(1 for subset in clique_subsets if subset_is_clique(adjacency, subset))
    independent_hits = sum(1 for subset in independent_subsets if subset_is_independent(adjacency, subset))
    clique_rate = clique_hits / max(1, len(clique_subsets))
    independent_rate = independent_hits / max(1, len(independent_subsets))
    density_penalty = abs(graph_density(adjacency) - target.density)
    target_shift_budget = estimate_shift_budget(target.n, target.density)
    current_shift_budget = sum(adjacency[0]) // 2 if adjacency else 0
    shift_budget_penalty = abs(current_shift_budget - target_shift_budget) / max(1, target_shift_budget)
    exact_clique_penalty = 0.0
    if target.r <= 4:
        exact_count = exact_count_k_cliques(adjacency, target.r, cutoff=20000)
        exact_clique_penalty = 5.0 * (math.log1p(exact_count) / math.log1p(20000))
    return exact_clique_penalty + 2.5 * clique_rate + 2.0 * independent_rate + 0.5 * density_penalty + 0.2 * shift_budget_penalty


def build_sample_cache(target: RamseyWitness, samples: int, rng: random.Random) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    nodes = list(range(target.n))
    clique_subsets = [tuple(sorted(rng.sample(nodes, target.r))) for _ in range(samples)]
    independent_subsets = [tuple(sorted(rng.sample(nodes, target.s))) for _ in range(samples)]
    return clique_subsets, independent_subsets


def estimate_correction_budget(n: int, ratio: float, max_corrections: int) -> int:
    return min(max_corrections, max(2, round(n * ratio)))


def ordered_candidate_pool(n: int, prioritized: list[int]) -> list[int]:
    full = list(range(1, (n // 2) + 1))
    seen = set()
    ordered = []
    for shift in prioritized + full:
        if shift not in seen and 0 < shift <= n // 2:
            ordered.append(shift)
            seen.add(shift)
    return ordered


def pair_key(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def choose_shift(
    candidates: list[int],
    prioritized_pool: list[int],
    current: set[int],
    target_budget: int,
    n: int,
    normalized_bins: int | None,
    pair_weights: dict[tuple[int, int], float] | None,
    operator_bias: dict[int, float] | None,
    rng: random.Random,
) -> int:
    if not candidates:
        raise ValueError("choose_shift requires at least one candidate")
    focus_width = min(len(prioritized_pool), max(8, 2 * max(1, target_budget)))
    focus_candidates = [shift for shift in prioritized_pool[:focus_width] if shift in candidates]
    candidate_pool = focus_candidates if focus_candidates and rng.random() < 0.75 else candidates
    if normalized_bins is None:
        return rng.choice(candidate_pool)

    current_bins = [shift_to_bin(shift, n, normalized_bins) for shift in current]
    scored_candidates = []
    for shift in candidate_pool:
        shift_bin = shift_to_bin(shift, n, normalized_bins)
        pair_score = 0.0 if not pair_weights else sum(
            pair_weights.get(pair_key(shift_bin, current_bin), 0.0) for current_bin in current_bins
        )
        operator_score = 0.0 if not operator_bias else operator_bias.get(shift_bin, 0.0)
        scored_candidates.append((pair_score + operator_score, shift))
    scored_candidates.sort(reverse=True)
    top_score = scored_candidates[0][0]
    best = [shift for score, shift in scored_candidates if score == top_score]
    return rng.choice(best)


def choose_removal_shift(
    occupied: list[int],
    n: int,
    normalized_bins: int | None,
    operator_bias: dict[int, float] | None,
    rng: random.Random,
) -> int:
    if not occupied or normalized_bins is None or not operator_bias:
        return rng.choice(occupied)
    scored = []
    for shift in occupied:
        shift_bin = shift_to_bin(shift, n, normalized_bins)
        scored.append((operator_bias.get(shift_bin, 0.0), shift))
    scored.sort(reverse=True)
    top_score = scored[0][0]
    candidates = [shift for score, shift in scored if score == top_score]
    return rng.choice(candidates)


def observed_violation_edges(
    adjacency: list[list[int]],
    clique_subsets: list[tuple[int, ...]],
    independent_subsets: list[tuple[int, ...]],
    rng: random.Random,
) -> list[tuple[int, int]]:
    for subset in clique_subsets:
        if subset_is_clique(adjacency, subset):
            edges = [pair_key(subset[i], subset[j]) for i in range(len(subset)) for j in range(i + 1, len(subset))]
            rng.shuffle(edges)
            return edges
    for subset in independent_subsets:
        if subset_is_independent(adjacency, subset):
            edges = [pair_key(subset[i], subset[j]) for i in range(len(subset)) for j in range(i + 1, len(subset))]
            rng.shuffle(edges)
            return edges
    return []


def propose_correction_update(
    current: set[tuple[int, int]],
    adjacency: list[list[int]],
    clique_subsets: list[tuple[int, ...]],
    independent_subsets: list[tuple[int, ...]],
    correction_budget: int,
    n: int,
    rng: random.Random,
) -> set[tuple[int, int]]:
    proposal = set(current)
    guided_edges = observed_violation_edges(adjacency, clique_subsets, independent_subsets, rng)
    candidate_edge = guided_edges[0] if guided_edges else pair_key(*sorted(rng.sample(range(n), 2)))

    if len(proposal) < correction_budget and candidate_edge not in proposal:
        proposal.add(candidate_edge)
        return proposal

    move_type = rng.random()
    if move_type < 0.35 and proposal:
        proposal.remove(rng.choice(list(proposal)))
        return proposal
    if move_type < 0.7 and proposal:
        removed = rng.choice(list(proposal))
        proposal.remove(removed)
        proposal.add(candidate_edge)
        return proposal
    if candidate_edge in proposal:
        proposal.remove(candidate_edge)
    elif len(proposal) < correction_budget:
        proposal.add(candidate_edge)
    elif proposal:
        proposal.remove(rng.choice(list(proposal)))
        proposal.add(candidate_edge)
    return proposal


def propose_shift_update(
    current: set[int],
    pool: list[int],
    target_budget: int,
    n: int,
    normalized_bins: int | None,
    pair_weights: dict[tuple[int, int], float] | None,
    add_operator_bias: dict[int, float] | None,
    remove_operator_bias: dict[int, float] | None,
    rng: random.Random,
) -> set[int]:
    proposal = set(current)
    current_budget = len(proposal)
    available = [shift for shift in pool if shift not in proposal]
    occupied = [shift for shift in proposal if shift in pool]

    if current_budget < target_budget and available:
        proposal.add(
            choose_shift(
                available,
                pool,
                current,
                target_budget,
                n,
                normalized_bins,
                pair_weights,
                add_operator_bias,
                rng,
            )
        )
        return proposal
    if current_budget > target_budget and occupied:
        proposal.remove(choose_removal_shift(occupied, n, normalized_bins, remove_operator_bias, rng))
        return proposal

    move_type = rng.random()
    if move_type < 0.4 and available:
        proposal.add(
            choose_shift(
                available,
                pool,
                current,
                target_budget,
                n,
                normalized_bins,
                pair_weights,
                add_operator_bias,
                rng,
            )
        )
    elif move_type < 0.8 and occupied:
        proposal.remove(choose_removal_shift(occupied, n, normalized_bins, remove_operator_bias, rng))
    elif available and occupied:
        proposal.remove(choose_removal_shift(occupied, n, normalized_bins, remove_operator_bias, rng))
        proposal.add(
            choose_shift(
                available,
                pool,
                current,
                target_budget,
                n,
                normalized_bins,
                pair_weights,
                add_operator_bias,
                rng,
            )
        )
    elif available:
        proposal.add(
            choose_shift(
                available,
                pool,
                current,
                target_budget,
                n,
                normalized_bins,
                pair_weights,
                add_operator_bias,
                rng,
            )
        )
    elif occupied:
        proposal.remove(choose_removal_shift(occupied, n, normalized_bins, remove_operator_bias, rng))
    return proposal


def circulant_local_search(
    target: RamseyWitness,
    initial_shifts: list[int],
    prioritized_shifts: list[int],
    initial_corrections: list[tuple[int, int]] | set[tuple[int, int]] | None,
    samples: int,
    iterations: int,
    accept_slack: float,
    explore_prob: float,
    correction_ratio: float,
    max_corrections: int,
    normalized_bins: int | None,
    pair_weights: dict[tuple[int, int], float] | None,
    operator_bias: dict[str, dict[int, float]] | None,
    rng: random.Random,
) -> SearchResult:
    clique_subsets, independent_subsets = build_sample_cache(target, samples, rng)
    pool = ordered_candidate_pool(target.n, prioritized_shifts)
    target_budget = estimate_shift_budget(target.n, target.density)
    correction_budget = estimate_correction_budget(target.n, correction_ratio, max_corrections)
    current = set(initial_shifts)
    current_corrections: set[tuple[int, int]] = set(initial_corrections or [])
    current_adjacency = graph_from_shifts_and_corrections(target.n, current, current_corrections)
    current_score = sampled_objective(target, current_adjacency, clique_subsets, independent_subsets)
    initial_score = current_score

    best = set(current)
    best_corrections = set(current_corrections)
    best_adjacency = current_adjacency
    best_score = current_score
    accepted_moves = 0
    steps: list[SearchStep] = []

    for _ in range(iterations):
        if rng.random() < 0.7:
            proposal = propose_shift_update(
                current,
                pool,
                target_budget,
                target.n,
                normalized_bins,
                pair_weights,
                None if operator_bias is None else operator_bias.get("add"),
                None if operator_bias is None else operator_bias.get("remove"),
                rng,
            )
            proposal_corrections = set(current_corrections)
            move_kind = "shift"
        else:
            proposal = set(current)
            proposal_corrections = propose_correction_update(
                current_corrections,
                current_adjacency,
                clique_subsets,
                independent_subsets,
                correction_budget,
                target.n,
                rng,
            )
            move_kind = "correction"
        proposal_adjacency = graph_from_shifts_and_corrections(target.n, proposal, proposal_corrections)
        proposal_score = sampled_objective(target, proposal_adjacency, clique_subsets, independent_subsets)
        added_shifts = sorted(proposal - current)
        removed_shifts = sorted(current - proposal)
        accepted = proposal_score <= current_score + accept_slack or rng.random() < explore_prob
        steps.append(
            SearchStep(
                accepted=accepted,
                move_kind=move_kind,
                added_shifts=added_shifts,
                removed_shifts=removed_shifts,
                score_before=current_score,
                score_after=proposal_score,
            )
        )

        if accepted:
            current = proposal
            current_corrections = proposal_corrections
            current_adjacency = proposal_adjacency
            current_score = proposal_score
            accepted_moves += 1
            if proposal_score < best_score:
                best = set(proposal)
                best_corrections = set(proposal_corrections)
                best_adjacency = proposal_adjacency
                best_score = proposal_score

    return SearchResult(
        adjacency=best_adjacency,
        shifts=sorted(best),
        corrections=sorted(best_corrections),
        steps=steps,
        initial_score=initial_score,
        best_score=best_score,
        accepted_moves=accepted_moves,
        iterations=iterations,
    )


def clone_adjacency(adjacency: list[list[int]]) -> list[list[int]]:
    return [row[:] for row in adjacency]


def shift_of_pair(n: int, i: int, j: int) -> int:
    delta = abs(j - i)
    return min(delta, n - delta)


def safe_to_add_triangle_edge(adjacency: list[list[int]], i: int, j: int) -> bool:
    return not any(adjacency[i][k] == 1 and adjacency[j][k] == 1 for k in range(len(adjacency)))


def exact_triangle_objective(
    target: RamseyWitness,
    adjacency: list[list[int]],
    independent_subsets: list[tuple[int, ...]],
    target_shift_profile: dict[int, float] | None,
    clique_cutoff: int,
) -> tuple[float, float, float, float]:
    exact_count = exact_count_k_cliques(adjacency, 3, cutoff=clique_cutoff)
    independent_hits = sum(1 for subset in independent_subsets if subset_is_independent(adjacency, subset))
    independent_rate = independent_hits / max(1, len(independent_subsets))
    density_penalty = abs(graph_density(adjacency) - target.density)
    profile_penalty = 0.0
    if target_shift_profile:
        current_profile = cyclic_shift_profile(adjacency)
        all_shifts = set(target_shift_profile) | set(current_profile)
        profile_penalty = sum(abs(target_shift_profile.get(shift, 0.0) - current_profile.get(shift, 0.0)) for shift in all_shifts)
        profile_penalty /= max(1, len(all_shifts))
    return (
        float(exact_count),
        float(independent_rate),
        float(density_penalty),
        float(profile_penalty),
    )


def ranked_shift_deficits(
    adjacency: list[list[int]],
    target_shift_profile: dict[int, float] | None,
) -> list[int]:
    if not target_shift_profile:
        return []
    current_profile = cyclic_shift_profile(adjacency)
    deficits = sorted(
        (
            (target_shift_profile.get(shift, 0.0) - current_profile.get(shift, 0.0), shift)
            for shift in target_shift_profile
        ),
        reverse=True,
    )
    return [shift for deficit, shift in deficits if deficit > 1e-9]


def exact_triangle_local_search(
    target: RamseyWitness,
    initial_adjacency: list[list[int]],
    target_shift_profile: dict[int, float] | None,
    samples: int,
    iterations: int,
    candidate_edges: int,
    shift_candidates: int,
    pair_candidates: int,
    accept_slack: float,
    explore_prob: float,
    clique_cutoff: int,
    rng: random.Random,
) -> ExactAdjacencySearchResult:
    _, independent_subsets = build_sample_cache(target, samples, rng)
    current = clone_adjacency(initial_adjacency)
    current_key = exact_triangle_objective(target, current, independent_subsets, target_shift_profile, clique_cutoff)
    initial_score = float(sum(current_key))
    best = clone_adjacency(current)
    best_key = current_key
    accepted_moves = 0
    steps: list[SearchStep] = []

    for _ in range(iterations):
        exact_count = int(current_key[0])
        proposals: list[tuple[list[list[int]], str]] = []
        if exact_count > 0:
            hotspot_counts = exact_k_clique_edge_counts(current, 3, clique_cutoff=clique_cutoff)
            ranked_edges = sorted(hotspot_counts.items(), key=lambda item: (-item[1], item[0]))
            for (i, j), _ in ranked_edges[:candidate_edges]:
                candidate = clone_adjacency(current)
                candidate[i][j] = 0
                candidate[j][i] = 0
                proposals.append((candidate, "triangle_remove"))
        else:
            for shift in ranked_shift_deficits(current, target_shift_profile)[:shift_candidates]:
                pairs: list[tuple[int, int]] = []
                for i in range(target.n):
                    j = (i + shift) % target.n
                    if i >= j or current[i][j] == 1:
                        continue
                    if safe_to_add_triangle_edge(current, i, j):
                        pairs.append((i, j))
                if len(pairs) > pair_candidates:
                    pairs = rng.sample(pairs, pair_candidates)
                for i, j in pairs:
                    candidate = clone_adjacency(current)
                    candidate[i][j] = 1
                    candidate[j][i] = 1
                    proposals.append((candidate, "profile_add"))
            if not proposals:
                ranked_edges = sorted(
                    (
                        (shift_of_pair(target.n, i, j), i, j)
                        for i in range(target.n)
                        for j in range(i + 1, target.n)
                        if current[i][j] == 1
                    ),
                    reverse=True,
                )
                for _, i, j in ranked_edges[:pair_candidates]:
                    candidate = clone_adjacency(current)
                    candidate[i][j] = 0
                    candidate[j][i] = 0
                    proposals.append((candidate, "density_remove"))
        if not proposals:
            break

        best_proposal: tuple[list[list[int]], tuple[float, float, float, float], str] | None = None
        for candidate, move_kind in proposals:
            candidate_key = exact_triangle_objective(target, candidate, independent_subsets, target_shift_profile, clique_cutoff)
            if best_proposal is None or candidate_key < best_proposal[1]:
                best_proposal = (candidate, candidate_key, move_kind)
        if best_proposal is None:
            break

        proposal, proposal_key, move_kind = best_proposal
        accepted = proposal_key <= tuple(value + accept_slack for value in current_key) or rng.random() < explore_prob
        steps.append(
            SearchStep(
                accepted=accepted,
                move_kind=move_kind,
                added_shifts=[],
                removed_shifts=[],
                score_before=float(sum(current_key)),
                score_after=float(sum(proposal_key)),
            )
        )
        if accepted:
            current = proposal
            current_key = proposal_key
            accepted_moves += 1
            if proposal_key < best_key:
                best = clone_adjacency(proposal)
                best_key = proposal_key

    return ExactAdjacencySearchResult(
        adjacency=best,
        steps=steps,
        initial_score=initial_score,
        best_score=float(sum(best_key)),
        accepted_moves=accepted_moves,
        iterations=iterations,
        exact_count=int(best_key[0]),
    )
