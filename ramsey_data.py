from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from pathlib import Path


WITNESS_PATTERN = re.compile(r"R\((\d+),\s*(\d+)\)\s*[_>]=\s*(\d+)\.txt")


@dataclass(slots=True)
class RamseyWitness:
    path: Path
    r: int
    s: int
    claimed_bound: int
    n: int
    adjacency: list[list[int]]

    @property
    def density(self) -> float:
        edges = sum(sum(row) for row in self.adjacency) // 2
        return edges / (self.n * (self.n - 1) / 2)

    @property
    def degrees(self) -> list[int]:
        return [sum(row) for row in self.adjacency]


def parse_witness_path(path: Path) -> tuple[int, int, int]:
    match = WITNESS_PATTERN.match(path.name)
    if not match:
        raise ValueError(f"Unsupported witness filename: {path.name}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def load_adjacency_matrix(path: Path) -> list[list[int]]:
    content = path.read_text(encoding="utf-8")
    values = [int(token) for token in content.replace("[", " ").replace("]", " ").split()]
    side = int(math.isqrt(len(values)))
    if side * side != len(values):
        raise ValueError(f"Matrix in {path.name} is not square")
    matrix = [values[i * side : (i + 1) * side] for i in range(side)]
    return matrix


def load_witnesses(witness_dir: Path) -> list[RamseyWitness]:
    witnesses: list[RamseyWitness] = []
    for path in sorted(witness_dir.glob("*.txt")):
        r, s, claimed_bound = parse_witness_path(path)
        adjacency = load_adjacency_matrix(path)
        witnesses.append(
            RamseyWitness(
                path=path,
                r=r,
                s=s,
                claimed_bound=claimed_bound,
                n=len(adjacency),
                adjacency=adjacency,
            )
        )
    return witnesses


def cyclic_shift_profile(adjacency: list[list[int]]) -> dict[int, float]:
    n = len(adjacency)
    counts: dict[int, int] = {}
    totals: dict[int, int] = {}
    for i in range(n):
        for j in range(i + 1, n):
            delta = j - i
            shift = min(delta, n - delta)
            counts.setdefault(shift, 0)
            totals.setdefault(shift, 0)
            totals[shift] += 1
            counts[shift] += adjacency[i][j]
    return {shift: counts[shift] / totals[shift] for shift in sorted(totals)}


def subset_is_clique(adjacency: list[list[int]], nodes: tuple[int, ...]) -> bool:
    size = len(nodes)
    for i in range(size):
        for j in range(i + 1, size):
            if adjacency[nodes[i]][nodes[j]] == 0:
                return False
    return True


def subset_is_independent(adjacency: list[list[int]], nodes: tuple[int, ...]) -> bool:
    size = len(nodes)
    for i in range(size):
        for j in range(i + 1, size):
            if adjacency[nodes[i]][nodes[j]] == 1:
                return False
    return True


def sample_violation_rates(
    witness: RamseyWitness,
    adjacency: list[list[int]],
    samples: int,
    rng: random.Random,
) -> dict[str, float]:
    clique_subsets, independent_subsets = sample_target_subsets(witness, samples, rng)
    return violation_rates_from_subsets(adjacency, clique_subsets, independent_subsets)


def sample_target_subsets(
    witness: RamseyWitness,
    samples: int,
    rng: random.Random,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    nodes = list(range(witness.n))
    clique_subsets = [tuple(sorted(rng.sample(nodes, witness.r))) for _ in range(samples)]
    independent_subsets = [tuple(sorted(rng.sample(nodes, witness.s))) for _ in range(samples)]
    return clique_subsets, independent_subsets


def violation_rates_from_subsets(
    adjacency: list[list[int]],
    clique_subsets: list[tuple[int, ...]],
    independent_subsets: list[tuple[int, ...]],
) -> dict[str, float]:
    clique_hits = 0
    independent_hits = 0
    for clique_nodes in clique_subsets:
        if subset_is_clique(adjacency, clique_nodes):
            clique_hits += 1
    for independent_nodes in independent_subsets:
        if subset_is_independent(adjacency, independent_nodes):
            independent_hits += 1
    return {
        "sampled_r_clique_rate": clique_hits / max(1, len(clique_subsets)),
        "sampled_s_independent_rate": independent_hits / max(1, len(independent_subsets)),
    }


def adjacency_bitmasks(adjacency: list[list[int]]) -> list[int]:
    masks: list[int] = []
    for row in adjacency:
        mask = 0
        for j, value in enumerate(row):
            if value:
                mask |= 1 << j
        masks.append(mask)
    return masks


def iter_bits(mask: int):
    while mask:
        lsb = mask & -mask
        yield lsb.bit_length() - 1
        mask ^= lsb


def exact_has_k_clique(adjacency: list[list[int]], k: int) -> bool:
    if k <= 1:
        return True
    masks = adjacency_bitmasks(adjacency)
    n = len(adjacency)

    def search(candidates: int, depth: int) -> bool:
        if depth == 0:
            return True
        while candidates:
            lsb = candidates & -candidates
            v = lsb.bit_length() - 1
            candidates ^= lsb
            next_candidates = candidates & masks[v]
            if next_candidates.bit_count() >= depth - 1 and search(next_candidates, depth - 1):
                return True
        return False

    all_nodes = (1 << n) - 1
    return search(all_nodes, k)


def exact_count_k_cliques(adjacency: list[list[int]], k: int, cutoff: int | None = None) -> int:
    if k <= 1:
        return len(adjacency)
    masks = adjacency_bitmasks(adjacency)
    n = len(adjacency)
    total = 0

    def count(candidates: int, depth: int) -> None:
        nonlocal total
        if cutoff is not None and total >= cutoff:
            return
        if depth == 1:
            total += candidates.bit_count()
            return
        local = candidates
        while local:
            if cutoff is not None and total >= cutoff:
                return
            lsb = local & -local
            v = lsb.bit_length() - 1
            local ^= lsb
            next_candidates = local & masks[v]
            if next_candidates.bit_count() >= depth - 1:
                count(next_candidates, depth - 1)

    all_nodes = (1 << n) - 1
    count(all_nodes, k)
    return min(total, cutoff) if cutoff is not None else total


def exact_k_clique_edge_counts(
    adjacency: list[list[int]],
    k: int,
    clique_cutoff: int | None = None,
) -> dict[tuple[int, int], int]:
    if k <= 1:
        return {}
    masks = adjacency_bitmasks(adjacency)
    n = len(adjacency)
    total = 0
    edge_counts: dict[tuple[int, int], int] = {}

    def add_clique_edges(clique: list[int]) -> None:
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                edge = (clique[i], clique[j]) if clique[i] < clique[j] else (clique[j], clique[i])
                edge_counts[edge] = edge_counts.get(edge, 0) + 1

    def count(prefix: list[int], candidates: int, depth: int) -> None:
        nonlocal total
        if clique_cutoff is not None and total >= clique_cutoff:
            return
        if depth == 0:
            total += 1
            add_clique_edges(prefix)
            return
        local = candidates
        while local:
            if clique_cutoff is not None and total >= clique_cutoff:
                return
            lsb = local & -local
            v = lsb.bit_length() - 1
            local ^= lsb
            next_candidates = local & masks[v]
            if next_candidates.bit_count() >= depth - 1:
                count(prefix + [v], next_candidates, depth - 1)

    all_nodes = (1 << n) - 1
    count([], all_nodes, k)
    return edge_counts
