from __future__ import annotations

import random

from ramsey_circulant import bins_to_shifts
from ramsey_data import RamseyWitness, cyclic_shift_profile
from ramsey_reconstruct import empty_graph, reconstruct_from_structure
from ramsey_structure import ExtractedStructure
from ramsey_teacher import build_teacher_representation


def random_density_baseline(n: int, density: float, rng: random.Random) -> list[list[int]]:
    adjacency = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            edge = 1 if rng.random() < density else 0
            adjacency[i][j] = edge
            adjacency[j][i] = edge
    return adjacency


def random_circulant_baseline(n: int, density: float, rng: random.Random) -> list[list[int]]:
    adjacency = empty_graph(n)
    if n <= 1:
        return adjacency
    shift_budget = max(1, round(density * (n - 1) / 2))
    available = list(range(1, (n // 2) + 1))
    rng.shuffle(available)
    chosen = sorted(available[:shift_budget])
    for shift in chosen:
        for i in range(n):
            j = (i + shift) % n
            if i == j:
                continue
            adjacency[i][j] = 1
            adjacency[j][i] = 1
    return adjacency


def nearest_neighbor_transfer(target: RamseyWitness, train_pool: list[RamseyWitness], normalized_bins: int) -> list[list[int]]:
    if not train_pool:
        raise ValueError("nearest_neighbor_transfer requires at least one training witness")
    nearest = min(train_pool, key=lambda item: abs(item.n - target.n))
    rep = build_teacher_representation(nearest, top_k=24, normalized_bins=normalized_bins)
    structure = ExtractedStructure(
        shared_bins=rep.normalized_top_bins[:8],
        ranked_bins=rep.normalized_top_bins[:24],
        relation_buckets={"high": rep.normalized_top_bins[:4], "mid": rep.normalized_top_bins[4:8], "low": rep.normalized_top_bins[8:24]},
        bin_weights={shift_bin: rep.normalized_contrastive_profile.get(shift_bin, 0.0) for shift_bin in rep.normalized_top_bins[:24]},
        pair_weights={
            pair: weight
            for pair, weight in rep.normalized_pair_profile.items()
            if pair[0] in rep.normalized_top_bins[:24] and pair[1] in rep.normalized_top_bins[:24]
        },
        top_pairs=[
            pair
            for pair, _ in sorted(
                rep.normalized_pair_profile.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:24]
        ],
        operator_add_bias={},
        operator_remove_bias={},
        target_density=nearest.density,
        target_degree=sum(nearest.degrees) / len(nearest.degrees),
    )
    return reconstruct_from_structure(target.n, structure, normalized_bins, pairwise_bonus=0.0)


def structured_seed_transfer(target: RamseyWitness, train_pool: list[RamseyWitness], normalized_bins: int) -> list[list[int]]:
    if not train_pool:
        raise ValueError("structured_seed_transfer requires at least one training witness")
    seed = min(train_pool, key=lambda item: abs(item.n - target.n))
    profile = cyclic_shift_profile(seed.adjacency)
    ranked_shifts = [shift for shift, value in sorted(profile.items(), key=lambda item: item[1], reverse=True) if value > 0.0]
    rep = build_teacher_representation(seed, top_k=24, normalized_bins=normalized_bins)
    shared = rep.normalized_top_bins[: max(4, min(12, len(rep.normalized_top_bins)))]
    structure = ExtractedStructure(
        shared_bins=shared,
        ranked_bins=rep.normalized_top_bins[:24],
        relation_buckets={"high": shared[:4], "mid": shared[4:8], "low": rep.normalized_top_bins[8:24]},
        bin_weights={shift_bin: rep.normalized_contrastive_profile.get(shift_bin, 0.0) for shift_bin in rep.normalized_top_bins[:24]},
        pair_weights={
            pair: weight
            for pair, weight in rep.normalized_pair_profile.items()
            if pair[0] in rep.normalized_top_bins[:24] and pair[1] in rep.normalized_top_bins[:24]
        },
        top_pairs=[
            pair
            for pair, _ in sorted(
                rep.normalized_pair_profile.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:24]
        ],
        operator_add_bias={},
        operator_remove_bias={},
        target_density=target.density,
        target_degree=sum(target.degrees) / len(target.degrees),
    )
    return reconstruct_from_structure(target.n, structure, normalized_bins, pairwise_bonus=0.0)


def scaled_nearest_seed_shifts(target: RamseyWitness, train_pool: list[RamseyWitness], normalized_bins: int) -> list[int]:
    if not train_pool:
        raise ValueError("scaled_nearest_seed_shifts requires at least one training witness")
    nearest = min(train_pool, key=lambda item: abs(item.n - target.n))
    rep = build_teacher_representation(nearest, top_k=24, normalized_bins=normalized_bins)
    return bins_to_shifts(rep.normalized_top_bins, target.n, normalized_bins)
