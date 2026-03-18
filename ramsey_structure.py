from __future__ import annotations

from dataclasses import dataclass

from ramsey_student import StudentDistillation


@dataclass(slots=True)
class ExtractedStructure:
    shared_bins: list[int]
    ranked_bins: list[int]
    relation_buckets: dict[str, list[int]]
    bin_weights: dict[int, float]
    pair_weights: dict[tuple[int, int], float]
    top_pairs: list[tuple[int, int]]
    operator_add_bias: dict[int, float]
    operator_remove_bias: dict[int, float]
    target_density: float
    target_degree: float


def extract_structure(
    distilled: StudentDistillation,
    support_threshold: float,
    max_shift_pool: int,
    max_shared_bins: int,
    contrastive_margin: float,
    pairwise_support_threshold: float,
) -> ExtractedStructure:
    ranked = sorted(
        distilled.contrastive_weights.items(),
        key=lambda item: (
            item[1],
            distilled.positive_excess_support.get(item[0], 0.0),
            distilled.motif_weights.get(item[0], 0.0),
        ),
        reverse=True,
    )
    ranked_bins = [
        shift_bin
        for shift_bin, contrastive_weight in ranked
        if contrastive_weight > 0.0
    ][:max_shift_pool]
    if len(ranked_bins) < max_shift_pool:
        fallback_ranked = sorted(
            distilled.motif_weights.items(),
            key=lambda item: (
                distilled.support.get(item[0], 0.0),
                item[1],
            ),
            reverse=True,
        )
        for shift_bin, _ in fallback_ranked:
            if shift_bin not in ranked_bins:
                ranked_bins.append(shift_bin)
            if len(ranked_bins) >= max_shift_pool:
                break
    shared_bins = [
        shift
        for shift, weight in ranked
        if distilled.positive_excess_support.get(shift, 0.0) >= support_threshold and weight >= contrastive_margin
    ][:max_shared_bins]
    if not shared_bins:
        shared_bins = ranked_bins[: min(max_shared_bins, len(ranked_bins))]

    relation_buckets = {"high": [], "mid": [], "low": []}
    if ranked_bins:
        max_weight = max(distilled.contrastive_weights.get(shift, 0.0) for shift in ranked_bins)
        for shift in ranked_bins[:max_shift_pool]:
            weight = distilled.contrastive_weights.get(shift, 0.0)
            ratio = 0.0 if max_weight == 0.0 else weight / max_weight
            if ratio >= 0.75:
                relation_buckets["high"].append(shift)
            elif ratio >= 0.4:
                relation_buckets["mid"].append(shift)
            else:
                relation_buckets["low"].append(shift)

    bin_weights = {shift: distilled.contrastive_weights.get(shift, 0.0) for shift in ranked_bins}
    active_ranked = set(ranked_bins)
    pair_weights = {
        pair: weight
        for pair, weight in distilled.pair_weights.items()
        if pair[0] in active_ranked
        and pair[1] in active_ranked
        and distilled.pair_support.get(pair, 0.0) >= pairwise_support_threshold
        and weight > 0.0
    }
    top_pairs = [
        pair
        for pair, _ in sorted(pair_weights.items(), key=lambda item: item[1], reverse=True)[: max_shift_pool]
    ]

    return ExtractedStructure(
        shared_bins=shared_bins,
        ranked_bins=ranked_bins,
        relation_buckets=relation_buckets,
        bin_weights=bin_weights,
        pair_weights=pair_weights,
        top_pairs=top_pairs,
        operator_add_bias={
            shift: distilled.operator_add_bias.get(shift, 0.0)
            for shift in ranked_bins
            if distilled.operator_add_bias.get(shift, 0.0) != 0.0
        },
        operator_remove_bias={
            shift: distilled.operator_remove_bias.get(shift, 0.0)
            for shift in ranked_bins
            if distilled.operator_remove_bias.get(shift, 0.0) != 0.0
        },
        target_density=distilled.mean_density,
        target_degree=distilled.mean_degree,
    )
