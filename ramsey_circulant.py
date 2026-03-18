from __future__ import annotations


def max_circulant_shift(n: int) -> int:
    return max(1, n // 2)


def shift_to_bin(shift: int, n: int, normalized_bins: int) -> int:
    max_shift = max_circulant_shift(n)
    ratio = shift / max_shift
    return max(1, min(normalized_bins, round(ratio * normalized_bins)))


def bin_to_shift(bin_index: int, n: int, normalized_bins: int) -> int:
    max_shift = max_circulant_shift(n)
    shift = round((bin_index / normalized_bins) * max_shift)
    return max(1, min(max_shift, shift))


def profile_to_bins(shift_profile: dict[int, float], n: int, normalized_bins: int) -> dict[int, float]:
    binned: dict[int, float] = {}
    for shift, weight in shift_profile.items():
        bin_index = shift_to_bin(shift, n, normalized_bins)
        binned[bin_index] = binned.get(bin_index, 0.0) + weight
    return binned


def bins_to_ratios(bin_indices: list[int], normalized_bins: int) -> list[float]:
    return [round(bin_index / normalized_bins, 4) for bin_index in bin_indices]


def bins_to_shifts(bin_indices: list[int], n: int, normalized_bins: int) -> list[int]:
    shifts: list[int] = []
    seen = set()
    for bin_index in bin_indices:
        shift = bin_to_shift(bin_index, n, normalized_bins)
        if shift not in seen:
            shifts.append(shift)
            seen.add(shift)
    return shifts
