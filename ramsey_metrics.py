from __future__ import annotations

from ramsey_data import (
    RamseyWitness,
    cyclic_shift_profile,
    exact_count_k_cliques,
    exact_has_k_clique,
    sample_violation_rates,
    violation_rates_from_subsets,
)


def graph_density(adjacency: list[list[int]]) -> float:
    n = len(adjacency)
    if n <= 1:
        return 0.0
    edges = sum(sum(row) for row in adjacency) // 2
    return edges / (n * (n - 1) / 2)


def degree_sequence(adjacency: list[list[int]]) -> list[int]:
    return [sum(row) for row in adjacency]


def degree_mae(reference: RamseyWitness, adjacency: list[list[int]]) -> float:
    ref_degrees = reference.degrees
    cand_degrees = degree_sequence(adjacency)
    return sum(abs(a - b) for a, b in zip(ref_degrees, cand_degrees)) / reference.n


def density_error(reference: RamseyWitness, adjacency: list[list[int]]) -> float:
    return abs(graph_density(adjacency) - reference.density)


def motif_overlap(reference: RamseyWitness, adjacency: list[list[int]]) -> float:
    ref = cyclic_shift_profile(reference.adjacency)
    cand = cyclic_shift_profile(adjacency)
    shifts = sorted(set(ref) | set(cand))
    overlap = 0.0
    for shift in shifts:
        overlap += min(ref.get(shift, 0.0), cand.get(shift, 0.0))
    return overlap / max(1, len(shifts))


def edge_disagreement_rate(reference: RamseyWitness, adjacency: list[list[int]]) -> float:
    disagreements = 0
    total = 0
    for i in range(reference.n):
        for j in range(i + 1, reference.n):
            total += 1
            if reference.adjacency[i][j] != adjacency[i][j]:
                disagreements += 1
    return disagreements / max(1, total)


def top_shift_set(adjacency: list[list[int]], top_k: int) -> set[int]:
    profile = cyclic_shift_profile(adjacency)
    ranked = sorted(profile.items(), key=lambda item: item[1], reverse=True)
    return {shift for shift, _ in ranked[:top_k]}


def top_shift_alignment(reference: RamseyWitness, adjacency: list[list[int]], top_k: int) -> dict[str, float]:
    ref_top = top_shift_set(reference.adjacency, top_k)
    cand_top = top_shift_set(adjacency, top_k)
    overlap = len(ref_top & cand_top)
    precision = overlap / max(1, len(cand_top))
    recall = overlap / max(1, len(ref_top))
    union = len(ref_top | cand_top)
    jaccard = overlap / max(1, union)
    return {
        "top_shift_precision": precision,
        "top_shift_recall": recall,
        "top_shift_jaccard": jaccard,
    }


def evaluate_candidate(
    name: str,
    target: RamseyWitness,
    adjacency: list[list[int]],
    sampled_subsets: int,
    top_k: int,
    rng,
    sample_cache: tuple[list[tuple[int, ...]], list[tuple[int, ...]]] | None = None,
) -> dict[str, float | str]:
    metrics: dict[str, float | str] = {
        "name": name,
        "density_error": density_error(target, adjacency),
        "degree_mae": degree_mae(target, adjacency),
        "motif_overlap": motif_overlap(target, adjacency),
        "edge_disagreement_rate": edge_disagreement_rate(target, adjacency),
    }
    if target.r <= 4:
        exact_count = exact_count_k_cliques(adjacency, target.r)
        metrics["exact_r_clique_count"] = exact_count
        metrics["exact_r_clique_free"] = 1.0 if exact_count == 0 else 0.0
    else:
        metrics["exact_r_clique_count"] = -1.0
        metrics["exact_r_clique_free"] = -1.0
    metrics.update(top_shift_alignment(target, adjacency, top_k=top_k))
    if sample_cache is None:
        metrics.update(sample_violation_rates(target, adjacency, sampled_subsets, rng))
    else:
        clique_subsets, independent_subsets = sample_cache
        metrics.update(violation_rates_from_subsets(adjacency, clique_subsets, independent_subsets))
    metrics["validity_proxy"] = max(
        0.0,
        1.0 - float(metrics["sampled_r_clique_rate"]) - float(metrics["sampled_s_independent_rate"]),
    )
    if target.r <= 4 and exact_has_k_clique(adjacency, target.r):
        metrics["validity_proxy"] = min(metrics["validity_proxy"], 0.5)
    return metrics
