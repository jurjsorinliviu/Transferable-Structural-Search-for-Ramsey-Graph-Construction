from __future__ import annotations

import random
from dataclasses import dataclass

from ramsey_data import RamseyWitness
from ramsey_reconstruct import empty_graph, pair_key, toggle_edge
from ramsey_search import (
    SearchStep,
    build_sample_cache,
    estimate_correction_budget,
    observed_violation_edges,
    sampled_objective,
)


@dataclass(slots=True)
class PartitionTemplate:
    blocks: int
    block_scores: dict[tuple[int, int], float]
    block_support: dict[tuple[int, int], float]
    ranked_pairs: list[tuple[int, int]]
    shared_pairs: list[tuple[int, int]]
    target_density: float


@dataclass(slots=True)
class PartitionSearchResult:
    adjacency: list[list[int]]
    active_pairs: list[tuple[int, int]]
    corrections: list[tuple[int, int]]
    steps: list[SearchStep]
    initial_score: float
    best_score: float
    accepted_moves: int
    iterations: int


def equal_partition_labels(n: int, blocks: int) -> list[int]:
    return [min(blocks - 1, (vertex * blocks) // max(1, n)) for vertex in range(n)]


def partition_pair_pool(blocks: int) -> list[tuple[int, int]]:
    return [(left, right) for left in range(blocks) for right in range(left, blocks)]


def block_density_profile(adjacency: list[list[int]], blocks: int) -> dict[tuple[int, int], float]:
    labels = equal_partition_labels(len(adjacency), blocks)
    counts = {pair: 0 for pair in partition_pair_pool(blocks)}
    totals = {pair: 0 for pair in partition_pair_pool(blocks)}
    for i in range(len(adjacency)):
        for j in range(i + 1, len(adjacency)):
            pair = pair_key(labels[i], labels[j])
            totals[pair] += 1
            counts[pair] += adjacency[i][j]
    return {
        pair: (counts[pair] / totals[pair]) if totals[pair] else 0.0
        for pair in partition_pair_pool(blocks)
    }


def distill_partition_template(
    witnesses: list[RamseyWitness],
    blocks: int,
    support_threshold: float,
    contrastive_margin: float,
) -> PartitionTemplate:
    score_sum = {pair: 0.0 for pair in partition_pair_pool(blocks)}
    support = {pair: 0.0 for pair in partition_pair_pool(blocks)}
    mean_density = sum(witness.density for witness in witnesses) / max(1, len(witnesses))
    for witness in witnesses:
        profile = block_density_profile(witness.adjacency, blocks)
        for pair, density in profile.items():
            excess = max(0.0, density - witness.density)
            score_sum[pair] += excess
            if excess >= contrastive_margin:
                support[pair] += 1.0
    count = max(1, len(witnesses))
    block_scores = {pair: score_sum[pair] / count for pair in score_sum}
    block_support = {pair: support[pair] / count for pair in support}
    ranked_pairs = [
        pair
        for pair, weight in sorted(
            block_scores.items(),
            key=lambda item: (item[1], block_support[item[0]]),
            reverse=True,
        )
        if weight > 0.0
    ]
    shared_pairs = [pair for pair in ranked_pairs if block_support[pair] >= support_threshold]
    if not shared_pairs:
        shared_pairs = ranked_pairs[: max(1, min(4, len(ranked_pairs)))]
    return PartitionTemplate(
        blocks=blocks,
        block_scores=block_scores,
        block_support=block_support,
        ranked_pairs=ranked_pairs,
        shared_pairs=shared_pairs,
        target_density=mean_density,
    )


def estimate_active_pair_budget(blocks: int, density: float) -> int:
    possible = len(partition_pair_pool(blocks))
    return max(1, round(possible * density))


def graph_from_partition_template(
    n: int,
    blocks: int,
    active_pairs: set[tuple[int, int]] | list[tuple[int, int]],
    corrections: set[tuple[int, int]] | None = None,
) -> list[list[int]]:
    labels = equal_partition_labels(n, blocks)
    active = set(active_pairs)
    adjacency = empty_graph(n)
    for i in range(n):
        for j in range(i + 1, n):
            if pair_key(labels[i], labels[j]) in active:
                adjacency[i][j] = 1
                adjacency[j][i] = 1
    for i, j in sorted(corrections or set()):
        if 0 <= i < n and 0 <= j < n and i != j:
            toggle_edge(adjacency, i, j)
    return adjacency


def reconstruct_from_partition_template(target: RamseyWitness, template: PartitionTemplate) -> list[list[int]]:
    budget = estimate_active_pair_budget(template.blocks, template.target_density)
    chosen = list(template.shared_pairs[:budget])
    if len(chosen) < budget:
        for pair in template.ranked_pairs:
            if pair not in chosen:
                chosen.append(pair)
            if len(chosen) >= budget:
                break
    return graph_from_partition_template(target.n, template.blocks, set(chosen))


def choose_block_pair(
    candidates: list[tuple[int, int]],
    ranked_pairs: list[tuple[int, int]],
    current: set[tuple[int, int]],
    pair_bonus: float,
    score_map: dict[tuple[int, int], float],
    rng: random.Random,
) -> tuple[int, int]:
    if not candidates:
        raise ValueError("choose_block_pair requires candidates")
    focus = [pair for pair in ranked_pairs[: max(4, len(ranked_pairs) // 2)] if pair in candidates]
    candidate_pool = focus if focus and rng.random() < 0.75 else candidates
    best_pair = None
    best_score = None
    for pair in candidate_pool:
        score = score_map.get(pair, 0.0)
        score += pair_bonus * sum(score_map.get(existing, 0.0) for existing in current if existing != pair)
        if best_score is None or score > best_score:
            best_pair = pair
            best_score = score
    return best_pair if best_pair is not None else rng.choice(candidate_pool)


def propose_partition_update(
    current: set[tuple[int, int]],
    template: PartitionTemplate,
    target_budget: int,
    rng: random.Random,
    pair_bonus: float,
) -> set[tuple[int, int]]:
    proposal = set(current)
    pool = partition_pair_pool(template.blocks)
    available = [pair for pair in pool if pair not in proposal]
    occupied = [pair for pair in proposal if pair in pool]
    if len(proposal) < target_budget and available:
        proposal.add(choose_block_pair(available, template.ranked_pairs, proposal, pair_bonus, template.block_scores, rng))
        return proposal
    if len(proposal) > target_budget and occupied:
        proposal.remove(rng.choice(occupied))
        return proposal
    move_type = rng.random()
    if move_type < 0.4 and available:
        proposal.add(choose_block_pair(available, template.ranked_pairs, proposal, pair_bonus, template.block_scores, rng))
    elif move_type < 0.8 and occupied:
        proposal.remove(rng.choice(occupied))
    elif available and occupied:
        proposal.remove(rng.choice(occupied))
        proposal.add(choose_block_pair(available, template.ranked_pairs, proposal, pair_bonus, template.block_scores, rng))
    return proposal


def propose_partition_correction_update(
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
    if candidate_edge in proposal:
        proposal.remove(candidate_edge)
    elif proposal and rng.random() < 0.5:
        proposal.remove(rng.choice(list(proposal)))
    elif proposal:
        proposal.remove(rng.choice(list(proposal)))
        proposal.add(candidate_edge)
    return proposal


def partition_local_search(
    target: RamseyWitness,
    template: PartitionTemplate,
    iterations: int,
    samples: int,
    accept_slack: float,
    explore_prob: float,
    correction_ratio: float,
    max_corrections: int,
    pair_bonus: float,
    rng: random.Random,
) -> PartitionSearchResult:
    clique_subsets, independent_subsets = build_sample_cache(target, samples, rng)
    target_budget = estimate_active_pair_budget(template.blocks, target.density)
    current = set(template.shared_pairs[:target_budget] or template.ranked_pairs[:target_budget])
    current_corrections: set[tuple[int, int]] = set()
    correction_budget = estimate_correction_budget(target.n, correction_ratio, max_corrections)
    current_adjacency = graph_from_partition_template(target.n, template.blocks, current, current_corrections)
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
            proposal = propose_partition_update(current, template, target_budget, rng, pair_bonus)
            proposal_corrections = set(current_corrections)
            move_kind = "partition"
        else:
            proposal = set(current)
            proposal_corrections = propose_partition_correction_update(
                current_corrections,
                current_adjacency,
                clique_subsets,
                independent_subsets,
                correction_budget,
                target.n,
                rng,
            )
            move_kind = "correction"
        proposal_adjacency = graph_from_partition_template(target.n, template.blocks, proposal, proposal_corrections)
        proposal_score = sampled_objective(target, proposal_adjacency, clique_subsets, independent_subsets)
        accepted = proposal_score <= current_score + accept_slack or rng.random() < explore_prob
        steps.append(
            SearchStep(
                accepted=accepted,
                move_kind=move_kind,
                added_shifts=[],
                removed_shifts=[],
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

    return PartitionSearchResult(
        adjacency=best_adjacency,
        active_pairs=sorted(best),
        corrections=sorted(best_corrections),
        steps=steps,
        initial_score=initial_score,
        best_score=best_score,
        accepted_moves=accepted_moves,
        iterations=iterations,
    )
