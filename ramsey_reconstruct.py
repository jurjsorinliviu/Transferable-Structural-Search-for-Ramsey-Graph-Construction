from __future__ import annotations

from ramsey_data import cyclic_shift_profile
from ramsey_circulant import bins_to_ratios, bins_to_shifts
from ramsey_structure import ExtractedStructure


def empty_graph(n: int) -> list[list[int]]:
    return [[0 for _ in range(n)] for _ in range(n)]


def add_circulant_shift(adjacency: list[list[int]], shift: int) -> None:
    n = len(adjacency)
    for i in range(n):
        j = (i + shift) % n
        if i == j:
            continue
        adjacency[i][j] = 1
        adjacency[j][i] = 1


def toggle_edge(adjacency: list[list[int]], i: int, j: int) -> None:
    if i == j:
        return
    adjacency[i][j] = 1 - adjacency[i][j]
    adjacency[j][i] = adjacency[i][j]


def estimate_shift_budget(n: int, density: float) -> int:
    if n <= 1:
        return 0
    return max(1, round(density * (n - 1) / 2))


def pair_key(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def ordered_bins_from_structure(structure: ExtractedStructure, budget: int, pairwise_bonus: float) -> list[int]:
    candidates = list(dict.fromkeys(structure.shared_bins + structure.ranked_bins))
    if not candidates or budget <= 0:
        return []

    selected: list[int] = []
    remaining = set(candidates)
    while remaining and len(selected) < budget:
        best_bin = None
        best_score = None
        for shift_bin in remaining:
            unary_score = structure.bin_weights.get(shift_bin, 0.0)
            pair_score = sum(
                structure.pair_weights.get(pair_key(shift_bin, chosen), 0.0)
                for chosen in selected
            )
            operator_score = structure.operator_add_bias.get(shift_bin, 0.0)
            score = unary_score + operator_score + pairwise_bonus * pair_score
            if best_score is None or score > best_score:
                best_bin = shift_bin
                best_score = score
        if best_bin is None:
            break
        selected.append(best_bin)
        remaining.remove(best_bin)

    if len(selected) < budget:
        for shift_bin in structure.ranked_bins:
            if shift_bin not in selected:
                selected.append(shift_bin)
            if len(selected) >= budget:
                break
    return selected[:budget]


def select_shifts_from_structure(
    n: int,
    structure: ExtractedStructure,
    normalized_bins: int,
    pairwise_bonus: float,
) -> list[int]:
    estimated_shifts = estimate_shift_budget(n, structure.target_density)
    ordered_bins = ordered_bins_from_structure(structure, estimated_shifts, pairwise_bonus)
    return bins_to_shifts(ordered_bins, n, normalized_bins)[:estimated_shifts]


def graph_from_shifts(n: int, shifts: list[int] | set[int]) -> list[list[int]]:
    adjacency = empty_graph(n)
    for shift in sorted(set(shifts)):
        if 0 < shift <= n // 2:
            add_circulant_shift(adjacency, shift)
    return adjacency


def graph_from_shifts_and_corrections(
    n: int,
    shifts: list[int] | set[int],
    corrections: set[tuple[int, int]],
) -> list[list[int]]:
    adjacency = graph_from_shifts(n, shifts)
    for i, j in sorted(corrections):
        if 0 <= i < n and 0 <= j < n and i != j:
            toggle_edge(adjacency, i, j)
    return adjacency


def infer_corrections_from_adjacency(
    adjacency: list[list[int]],
    shifts: list[int] | set[int],
) -> list[tuple[int, int]]:
    base = graph_from_shifts(len(adjacency), shifts)
    corrections: list[tuple[int, int]] = []
    for i in range(len(adjacency)):
        for j in range(i + 1, len(adjacency)):
            if adjacency[i][j] != base[i][j]:
                corrections.append((i, j))
    return corrections


def infer_shifts_from_adjacency(adjacency: list[list[int]], shift_budget: int) -> list[int]:
    profile = cyclic_shift_profile(adjacency)
    ranked = sorted(profile.items(), key=lambda item: item[1], reverse=True)
    return [shift for shift, _ in ranked[: max(0, shift_budget)]]


def reconstruct_from_structure(
    n: int,
    structure: ExtractedStructure,
    normalized_bins: int,
    pairwise_bonus: float,
) -> list[list[int]]:
    return graph_from_shifts(n, select_shifts_from_structure(n, structure, normalized_bins, pairwise_bonus))


def structure_ratios(structure: ExtractedStructure, normalized_bins: int) -> dict[str, list[float]]:
    return {
        "shared": bins_to_ratios(structure.shared_bins, normalized_bins),
        "ranked": bins_to_ratios(structure.ranked_bins, normalized_bins),
    }
