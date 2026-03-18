from __future__ import annotations

from itertools import combinations
from dataclasses import dataclass

from ramsey_circulant import profile_to_bins
from ramsey_data import RamseyWitness, cyclic_shift_profile


@dataclass(slots=True)
class TeacherRepresentation:
    witness_name: str
    r: int
    s: int
    n: int
    density: float
    degree_mean: float
    degree_range: tuple[int, int]
    shift_profile: dict[int, float]
    normalized_shift_profile: dict[int, float]
    normalized_contrastive_profile: dict[int, float]
    normalized_pair_profile: dict[tuple[int, int], float]
    top_shifts: list[int]
    normalized_top_bins: list[int]


def build_teacher_representation(witness: RamseyWitness, top_k: int, normalized_bins: int) -> TeacherRepresentation:
    shift_profile = cyclic_shift_profile(witness.adjacency)
    ranked = sorted(shift_profile.items(), key=lambda item: item[1], reverse=True)
    top_shifts = [shift for shift, _ in ranked[:top_k]]
    normalized_shift_profile = profile_to_bins(shift_profile, witness.n, normalized_bins)
    normalized_contrastive_profile = {
        bin_index: max(0.0, weight - witness.density)
        for bin_index, weight in normalized_shift_profile.items()
    }
    active_bins = [bin_index for bin_index, weight in normalized_contrastive_profile.items() if weight > 0.0]
    normalized_pair_profile: dict[tuple[int, int], float] = {}
    for left, right in combinations(sorted(active_bins), 2):
        normalized_pair_profile[(left, right)] = min(
            normalized_contrastive_profile[left],
            normalized_contrastive_profile[right],
        )
    normalized_ranked = sorted(normalized_shift_profile.items(), key=lambda item: item[1], reverse=True)
    normalized_top_bins = [bin_index for bin_index, _ in normalized_ranked[:top_k]]
    degrees = witness.degrees
    return TeacherRepresentation(
        witness_name=witness.path.name,
        r=witness.r,
        s=witness.s,
        n=witness.n,
        density=witness.density,
        degree_mean=sum(degrees) / len(degrees),
        degree_range=(min(degrees), max(degrees)),
        shift_profile=shift_profile,
        normalized_shift_profile=normalized_shift_profile,
        normalized_contrastive_profile=normalized_contrastive_profile,
        normalized_pair_profile=normalized_pair_profile,
        top_shifts=top_shifts,
        normalized_top_bins=normalized_top_bins,
    )
