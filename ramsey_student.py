from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ramsey_teacher import TeacherRepresentation

if TYPE_CHECKING:
    from ramsey_trajectory import TrajectoryDistillation


@dataclass(slots=True)
class StudentDistillation:
    train_count: int
    mean_density: float
    mean_degree: float
    support: dict[int, float]
    motif_weights: dict[int, float]
    contrastive_weights: dict[int, float]
    positive_excess_support: dict[int, float]
    pair_weights: dict[tuple[int, int], float]
    pair_support: dict[tuple[int, int], float]
    operator_add_bias: dict[int, float]
    operator_remove_bias: dict[int, float]


def distill_student(representations: list[TeacherRepresentation]) -> StudentDistillation:
    if not representations:
        raise ValueError("At least one teacher representation is required")

    support: dict[int, float] = {}
    motif_weights: dict[int, float] = {}
    contrastive_weights: dict[int, float] = {}
    positive_excess_support: dict[int, float] = {}
    pair_weights: dict[tuple[int, int], float] = {}
    pair_support: dict[tuple[int, int], float] = {}
    for rep in representations:
        seen = set()
        seen_excess = set()
        for shift_bin, weight in rep.normalized_shift_profile.items():
            motif_weights[shift_bin] = motif_weights.get(shift_bin, 0.0) + weight
            excess = max(0.0, weight - rep.density)
            contrastive_weights[shift_bin] = contrastive_weights.get(shift_bin, 0.0) + excess
            if weight > 0.0:
                seen.add(shift_bin)
            if excess > 0.0:
                seen_excess.add(shift_bin)
        for shift_bin in seen:
            support[shift_bin] = support.get(shift_bin, 0.0) + 1.0
        for shift_bin in seen_excess:
            positive_excess_support[shift_bin] = positive_excess_support.get(shift_bin, 0.0) + 1.0
        for pair, weight in rep.normalized_pair_profile.items():
            pair_weights[pair] = pair_weights.get(pair, 0.0) + weight
            if weight > 0.0:
                pair_support[pair] = pair_support.get(pair, 0.0) + 1.0

    count = len(representations)
    motif_weights = {shift: weight / count for shift, weight in motif_weights.items()}
    support = {shift: value / count for shift, value in support.items()}
    contrastive_weights = {shift: weight / count for shift, weight in contrastive_weights.items()}
    positive_excess_support = {shift: value / count for shift, value in positive_excess_support.items()}
    pair_weights = {pair: weight / count for pair, weight in pair_weights.items()}
    pair_support = {pair: value / count for pair, value in pair_support.items()}

    return StudentDistillation(
        train_count=count,
        mean_density=sum(rep.density for rep in representations) / count,
        mean_degree=sum(rep.degree_mean for rep in representations) / count,
        support=support,
        motif_weights=motif_weights,
        contrastive_weights=contrastive_weights,
        positive_excess_support=positive_excess_support,
        pair_weights=pair_weights,
        pair_support=pair_support,
        operator_add_bias={},
        operator_remove_bias={},
    )


def attach_trajectory_priors(
    distilled: StudentDistillation,
    trajectories: list[TrajectoryDistillation],
) -> StudentDistillation:
    if not trajectories:
        return distilled

    add_bias: dict[int, float] = {}
    remove_bias: dict[int, float] = {}
    count = len(trajectories)
    for item in trajectories:
        for shift_bin, value in item.accepted_add.items():
            add_bias[shift_bin] = add_bias.get(shift_bin, 0.0) + value
        for shift_bin, value in item.rejected_add.items():
            add_bias[shift_bin] = add_bias.get(shift_bin, 0.0) - value
        for shift_bin, value in item.accepted_remove.items():
            remove_bias[shift_bin] = remove_bias.get(shift_bin, 0.0) + value
        for shift_bin, value in item.rejected_remove.items():
            remove_bias[shift_bin] = remove_bias.get(shift_bin, 0.0) - value

    return StudentDistillation(
        train_count=distilled.train_count,
        mean_density=distilled.mean_density,
        mean_degree=distilled.mean_degree,
        support=distilled.support,
        motif_weights=distilled.motif_weights,
        contrastive_weights=distilled.contrastive_weights,
        positive_excess_support=distilled.positive_excess_support,
        pair_weights=distilled.pair_weights,
        pair_support=distilled.pair_support,
        operator_add_bias={shift_bin: value / count for shift_bin, value in add_bias.items()},
        operator_remove_bias={shift_bin: value / count for shift_bin, value in remove_bias.items()},
    )
