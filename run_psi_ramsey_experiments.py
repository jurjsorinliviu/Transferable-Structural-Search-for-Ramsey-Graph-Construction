from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

from experiment_config import ExperimentConfig, default_config
from ramsey_baselines import (
    nearest_neighbor_transfer,
    random_circulant_baseline,
    random_density_baseline,
    scaled_nearest_seed_shifts,
    structured_seed_transfer,
)
from ramsey_circulant import bins_to_shifts, profile_to_bins
from ramsey_data import (
    RamseyWitness,
    cyclic_shift_profile,
    exact_count_k_cliques,
    load_witnesses,
    sample_target_subsets,
)
from ramsey_metrics import density_error, edge_disagreement_rate, evaluate_candidate, graph_density, motif_overlap, top_shift_alignment
from ramsey_partition import (
    PartitionSearchResult,
    distill_partition_template,
    partition_local_search,
    reconstruct_from_partition_template,
)
from ramsey_reconstruct import (
    estimate_shift_budget,
    graph_from_shifts,
    infer_corrections_from_adjacency,
    infer_shifts_from_adjacency,
    reconstruct_from_structure,
    select_shifts_from_structure,
    structure_ratios,
)
from ramsey_repair import greedy_exact_repair
from ramsey_search import ExactAdjacencySearchResult, SearchResult, build_sample_cache, circulant_local_search, exact_triangle_local_search, sampled_objective
from ramsey_structure import ExtractedStructure, extract_structure
from ramsey_student import StudentDistillation, attach_trajectory_priors, distill_student
from ramsey_teacher import TeacherRepresentation, build_teacher_representation
from ramsey_trajectory import collect_teacher_search_traces, distill_teacher_trajectories
from ramsey_circulant import shift_to_bin


@dataclass(slots=True)
class CandidateArtifact:
    adjacency: list[list[int]]
    metadata: dict[str, int | float | str]
    shifts: list[int] | None = None
    corrections: list[tuple[int, int]] | None = None


CandidateBuilder = Callable[
    [RamseyWitness, list[RamseyWitness], list[TeacherRepresentation], StudentDistillation, ExtractedStructure, ExperimentConfig, random.Random],
    CandidateArtifact,
]


def training_pool_for_target(target: RamseyWitness, witnesses: list[RamseyWitness], same_r_only: bool) -> list[RamseyWitness]:
    pool = [w for w in witnesses if w.path != target.path]
    if same_r_only:
        pool = [w for w in pool if w.r == target.r]
    return pool


def teacher_mean_transfer(
    target: RamseyWitness,
    train_pool: list[RamseyWitness],
    teacher_reps: list[TeacherRepresentation],
    distilled: StudentDistillation,
    structure: ExtractedStructure,
    config: ExperimentConfig,
    rng: random.Random,
) -> CandidateArtifact:
    weight_sum: dict[int, float] = {}
    support: dict[int, float] = {}
    for rep in teacher_reps:
        seen = set()
        for shift_bin, weight in rep.normalized_shift_profile.items():
            weight_sum[shift_bin] = weight_sum.get(shift_bin, 0.0) + weight
            if weight > 0.0:
                seen.add(shift_bin)
        for shift_bin in seen:
            support[shift_bin] = support.get(shift_bin, 0.0) + 1.0
    count = max(1, len(teacher_reps))
    ranked = sorted(weight_sum.items(), key=lambda item: item[1], reverse=True)
    shared = [shift_bin for shift_bin, _ in ranked if support.get(shift_bin, 0.0) / count >= config.shared_support_threshold]
    teacher_structure = ExtractedStructure(
        shared_bins=shared[: config.max_shift_pool],
        ranked_bins=[shift_bin for shift_bin, _ in ranked[: config.max_shift_pool]],
        relation_buckets={"high": shared[:4], "mid": shared[4:8], "low": shared[8: config.max_shift_pool]},
        bin_weights={shift_bin: weight_sum.get(shift_bin, 0.0) / count for shift_bin, _ in ranked[: config.max_shift_pool]},
        pair_weights={},
        top_pairs=[],
        operator_add_bias={},
        operator_remove_bias={},
        target_density=sum(rep.density for rep in teacher_reps) / count,
        target_degree=sum(rep.degree_mean for rep in teacher_reps) / count,
    )
    shifts = select_shifts_from_structure(
        target.n,
        teacher_structure,
        config.normalized_shift_bins,
        config.pairwise_bonus,
    )
    return CandidateArtifact(
        adjacency=graph_from_shifts(target.n, shifts),
        metadata={},
        shifts=shifts,
        corrections=[],
    )


def student_no_filter_transfer(
    target: RamseyWitness,
    train_pool: list[RamseyWitness],
    teacher_reps: list[TeacherRepresentation],
    distilled: StudentDistillation,
    structure: ExtractedStructure,
    config: ExperimentConfig,
    rng: random.Random,
) -> CandidateArtifact:
    loose_structure = extract_structure(
        distilled,
        support_threshold=0.0,
        max_shift_pool=config.max_shift_pool,
        max_shared_bins=config.max_shared_bins,
        contrastive_margin=0.0,
        pairwise_support_threshold=0.0,
    )
    shifts = select_shifts_from_structure(
        target.n,
        loose_structure,
        config.normalized_shift_bins,
        config.pairwise_bonus,
    )
    return CandidateArtifact(
        adjacency=graph_from_shifts(target.n, shifts),
        metadata={},
        shifts=shifts,
        corrections=[],
    )


def random_builder(
    target: RamseyWitness,
    train_pool: list[RamseyWitness],
    teacher_reps: list[TeacherRepresentation],
    distilled: StudentDistillation,
    structure: ExtractedStructure,
    config: ExperimentConfig,
    rng: random.Random,
) -> CandidateArtifact:
    return CandidateArtifact(random_density_baseline(target.n, target.density, rng), {})


def random_circulant_builder(
    target: RamseyWitness,
    train_pool: list[RamseyWitness],
    teacher_reps: list[TeacherRepresentation],
    distilled: StudentDistillation,
    structure: ExtractedStructure,
    config: ExperimentConfig,
    rng: random.Random,
) -> CandidateArtifact:
    adjacency = random_circulant_baseline(target.n, target.density, rng)
    shifts = infer_shifts_from_adjacency(adjacency, estimate_shift_budget(target.n, target.density))
    return CandidateArtifact(adjacency=adjacency, metadata={}, shifts=shifts, corrections=[])


def nearest_builder(
    target: RamseyWitness,
    train_pool: list[RamseyWitness],
    teacher_reps: list[TeacherRepresentation],
    distilled: StudentDistillation,
    structure: ExtractedStructure,
    config: ExperimentConfig,
    rng: random.Random,
) -> CandidateArtifact:
    adjacency = nearest_neighbor_transfer(target, train_pool, config.normalized_shift_bins)
    shifts = infer_shifts_from_adjacency(adjacency, estimate_shift_budget(target.n, target.density))
    return CandidateArtifact(adjacency=adjacency, metadata={}, shifts=shifts, corrections=[])


def structured_seed_builder(
    target: RamseyWitness,
    train_pool: list[RamseyWitness],
    teacher_reps: list[TeacherRepresentation],
    distilled: StudentDistillation,
    structure: ExtractedStructure,
    config: ExperimentConfig,
    rng: random.Random,
) -> CandidateArtifact:
    adjacency = structured_seed_transfer(target, train_pool, config.normalized_shift_bins)
    shifts = infer_shifts_from_adjacency(adjacency, estimate_shift_budget(target.n, target.density))
    return CandidateArtifact(adjacency=adjacency, metadata={}, shifts=shifts, corrections=[])


def scaled_nearest_builder(
    target: RamseyWitness,
    train_pool: list[RamseyWitness],
    teacher_reps: list[TeacherRepresentation],
    distilled: StudentDistillation,
    structure: ExtractedStructure,
    config: ExperimentConfig,
    rng: random.Random,
) -> CandidateArtifact:
    shifts = scaled_nearest_seed_shifts(target, train_pool, config.normalized_shift_bins)
    return CandidateArtifact(adjacency=graph_from_shifts(target.n, shifts), metadata={}, shifts=shifts, corrections=[])


def psi_ramsey_builder(
    target: RamseyWitness,
    train_pool: list[RamseyWitness],
    teacher_reps: list[TeacherRepresentation],
    distilled: StudentDistillation,
    structure: ExtractedStructure,
    config: ExperimentConfig,
    rng: random.Random,
) -> CandidateArtifact:
    shifts = select_shifts_from_structure(
        target.n,
        structure,
        config.normalized_shift_bins,
        config.pairwise_bonus,
    )
    return CandidateArtifact(adjacency=graph_from_shifts(target.n, shifts), metadata={}, shifts=shifts, corrections=[])


def partition_transfer_builder(
    target: RamseyWitness,
    train_pool: list[RamseyWitness],
    teacher_reps: list[TeacherRepresentation],
    distilled: StudentDistillation,
    structure: ExtractedStructure,
    config: ExperimentConfig,
    rng: random.Random,
) -> CandidateArtifact:
    template = distill_partition_template(
        train_pool,
        blocks=config.partition_blocks,
        support_threshold=config.partition_support_threshold,
        contrastive_margin=config.partition_contrastive_margin,
    )
    return CandidateArtifact(reconstruct_from_partition_template(target, template), {})


def teacher_profile_transfer_builder(
    target: RamseyWitness,
    train_pool: list[RamseyWitness],
    teacher_reps: list[TeacherRepresentation],
    distilled: StudentDistillation,
    structure: ExtractedStructure,
    config: ExperimentConfig,
    rng: random.Random,
) -> CandidateArtifact:
    profile_reference = average_teacher_profile(teacher_reps)
    shift_budget = estimate_shift_budget(target.n, target.density)
    ranked_bins = [shift_bin for shift_bin, _ in sorted(profile_reference.items(), key=lambda item: item[1], reverse=True)]
    shifts = bins_to_shifts(ranked_bins[: max(1, shift_budget)], target.n, config.normalized_shift_bins)
    return CandidateArtifact(
        adjacency=graph_from_shifts(target.n, shifts),
        metadata={},
        shifts=shifts,
        corrections=[],
    )


def shift_pair_count(n: int, shift: int) -> int:
    if shift <= 0 or shift > n // 2:
        return 0
    if n % 2 == 0 and shift == n // 2:
        return n // 2
    return n


def interpolated_teacher_shift_profile(
    target: RamseyWitness,
    teacher_reps: list[TeacherRepresentation],
) -> dict[int, float]:
    max_shift = max(1, target.n // 2)
    if not teacher_reps:
        return {shift: target.density for shift in range(1, max_shift + 1)}
    profile: dict[int, float] = {}
    for shift in range(1, max_shift + 1):
        ratio = shift / max_shift
        total = 0.0
        count = 0
        for rep in teacher_reps:
            teacher_max_shift = max(1, rep.n // 2)
            mapped_shift = max(1, min(teacher_max_shift, round(ratio * teacher_max_shift)))
            total += rep.shift_profile.get(mapped_shift, 0.0)
            count += 1
        profile[shift] = total / max(1, count)
    return profile


def scale_shift_profile_to_density(
    n: int,
    shift_profile: dict[int, float],
    target_density: float,
) -> dict[int, float]:
    total_pairs = 0
    expected_edges = 0.0
    for shift, probability in shift_profile.items():
        count = shift_pair_count(n, shift)
        total_pairs += count
        expected_edges += count * probability
    if total_pairs <= 0:
        return shift_profile
    current_density = expected_edges / total_pairs
    if current_density <= 0.0:
        return {shift: target_density for shift in shift_profile}
    scale = target_density / current_density
    return {shift: min(1.0, max(0.0, probability * scale)) for shift, probability in shift_profile.items()}


def adjacency_from_shift_distribution(
    n: int,
    shift_profile: dict[int, float],
    rng: random.Random,
) -> list[list[int]]:
    adjacency = [[0 for _ in range(n)] for _ in range(n)]
    pairs_by_shift: dict[int, list[tuple[int, int]]] = {}
    for i in range(n):
        for j in range(i + 1, n):
            delta = j - i
            shift = min(delta, n - delta)
            pairs_by_shift.setdefault(shift, []).append((i, j))
    for shift, pairs in pairs_by_shift.items():
        probability = min(1.0, max(0.0, shift_profile.get(shift, 0.0)))
        edge_count = round(probability * len(pairs))
        if edge_count <= 0:
            continue
        if edge_count >= len(pairs):
            chosen = pairs
        else:
            chosen = rng.sample(pairs, edge_count)
        for i, j in chosen:
            adjacency[i][j] = 1
            adjacency[j][i] = 1
    return adjacency


def shift_pairs(n: int, shift: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for i in range(n):
        j = (i + shift) % n
        if i < j:
            pairs.append((i, j))
    return pairs


def motif_only_projection(target: RamseyWitness, adjacency: list[list[int]]) -> list[list[int]]:
    target_profile = cyclic_shift_profile(target.adjacency)
    projected = [[0 for _ in range(target.n)] for _ in range(target.n)]
    max_shift = max(1, target.n // 2)
    for shift in range(1, max_shift + 1):
        pairs = shift_pairs(target.n, shift)
        if not pairs:
            continue
        target_count = min(len(pairs), max(0, round(target_profile.get(shift, 0.0) * len(pairs))))
        ranked_pairs = sorted(pairs, key=lambda pair: adjacency[pair[0]][pair[1]], reverse=True)
        for i, j in ranked_pairs[:target_count]:
            projected[i][j] = 1
            projected[j][i] = 1
    return projected


def distribution_profile_transfer_builder(
    target: RamseyWitness,
    train_pool: list[RamseyWitness],
    teacher_reps: list[TeacherRepresentation],
    distilled: StudentDistillation,
    structure: ExtractedStructure,
    config: ExperimentConfig,
    rng: random.Random,
) -> CandidateArtifact:
    local_rng = random.Random(config.random_seed * 1000 + target.r * 100 + target.s * 10 + target.n)
    profile_reference = average_teacher_profile(teacher_reps)
    base_profile = interpolated_teacher_shift_profile(target, teacher_reps)
    scaled_profile = scale_shift_profile_to_density(target.n, base_profile, target.density)
    best_choice: tuple[float, float, CandidateArtifact] | None = None
    repair_steps = (
        config.transfer_distribution_repair_steps_r3
        if target.r == 3
        else config.transfer_distribution_repair_steps
    )
    for _ in range(max(1, config.transfer_distribution_trials)):
        adjacency = adjacency_from_shift_distribution(target.n, scaled_profile, local_rng)
        repair_result = greedy_exact_repair(
            target=target,
            adjacency=adjacency,
            samples=min(700, config.sampled_subsets),
            max_steps=repair_steps,
            candidate_edges=config.transfer_distribution_repair_edge_candidates,
            clique_cutoff=max(config.exact_repair_clique_cutoff, 20000),
            rng=local_rng,
        )
        artifact = CandidateArtifact(
            adjacency=repair_result.adjacency,
            metadata={
                "transfer_distribution_profile": 1,
                "transfer_distribution_repair_steps": repair_result.applied_steps,
                "repair_steps_applied": repair_result.applied_steps,
                "repair_initial_score": repair_result.initial_score,
                "repair_final_score": repair_result.final_score,
                "repair_exact_count_before": repair_result.exact_count_before,
                "repair_exact_count_after": repair_result.exact_count_after,
            },
            shifts=None,
            corrections=None,
        )
        alignment = teacher_profile_alignment_score(
            target,
            artifact.adjacency,
            config.normalized_shift_bins,
            profile_reference,
        )
        score = objective_score(target, artifact.adjacency, min(400, config.sampled_subsets), local_rng)
        choice = (score, -alignment, artifact)
        if best_choice is None or choice[:2] < best_choice[:2]:
            best_choice = choice
    if best_choice is None:
        return random_builder(target, train_pool, teacher_reps, distilled, structure, config, rng)
    return best_choice[2]


def portfolio_transfer_builder(
    target: RamseyWitness,
    train_pool: list[RamseyWitness],
    teacher_reps: list[TeacherRepresentation],
    distilled: StudentDistillation,
    structure: ExtractedStructure,
    config: ExperimentConfig,
    rng: random.Random,
) -> CandidateArtifact:
    profile_reference = average_teacher_profile(teacher_reps)
    candidates = [
        ("structured_seed", structured_seed_builder(target, train_pool, teacher_reps, distilled, structure, config, rng)),
        ("nearest_neighbor_transfer", nearest_builder(target, train_pool, teacher_reps, distilled, structure, config, rng)),
        ("teacher_mean_transfer", teacher_mean_transfer(target, train_pool, teacher_reps, distilled, structure, config, rng)),
        ("teacher_profile_transfer", teacher_profile_transfer_builder(target, train_pool, teacher_reps, distilled, structure, config, rng)),
        ("distribution_profile_transfer", distribution_profile_transfer_builder(target, train_pool, teacher_reps, distilled, structure, config, rng)),
        ("psi_ramsey_transfer", psi_ramsey_builder(target, train_pool, teacher_reps, distilled, structure, config, rng)),
        ("partition_transfer", partition_transfer_builder(target, train_pool, teacher_reps, distilled, structure, config, rng)),
        ("random_circulant", random_circulant_builder(target, train_pool, teacher_reps, distilled, structure, config, rng)),
        ("exact_triangle_transfer", exact_triangle_transfer_builder(target, train_pool, teacher_reps, distilled, structure, config, rng)),
    ]
    refined_candidates = [
        (
            name,
            refine_transfer_artifact(target, artifact, structure, profile_reference, config, rng),
        )
        for name, artifact in candidates
    ]
    polished_candidates = [
        (
            name,
            motif_polish_artifact(target, artifact, profile_reference, config, rng),
        )
        for name, artifact in refined_candidates
        if not artifact.metadata.get("transfer_diversity", 0)
    ]
    return best_artifact_by_objective(
        target,
        refined_candidates
        + [(f"{name}_motif", artifact) for name, artifact in polished_candidates],
        min(400, config.sampled_subsets),
        config,
        structure,
        profile_reference,
        *transfer_selection_weights(config),
        rng,
    )


def structure_oracle_transfer_builder(
    target: RamseyWitness,
    train_pool: list[RamseyWitness],
    teacher_reps: list[TeacherRepresentation],
    distilled: StudentDistillation,
    structure: ExtractedStructure,
    config: ExperimentConfig,
    rng: random.Random,
) -> CandidateArtifact:
    profile_reference = average_teacher_profile(teacher_reps)
    base_candidates = [
        ("random_density", random_builder(target, train_pool, teacher_reps, distilled, structure, config, rng)),
        ("random_circulant", random_circulant_builder(target, train_pool, teacher_reps, distilled, structure, config, rng)),
        ("structured_seed", structured_seed_builder(target, train_pool, teacher_reps, distilled, structure, config, rng)),
        ("nearest_neighbor_transfer", nearest_builder(target, train_pool, teacher_reps, distilled, structure, config, rng)),
        ("distribution_profile_transfer", distribution_profile_transfer_builder(target, train_pool, teacher_reps, distilled, structure, config, rng)),
        ("exact_triangle_transfer", exact_triangle_transfer_builder(target, train_pool, teacher_reps, distilled, structure, config, rng)),
        ("partition_transfer", partition_transfer_builder(target, train_pool, teacher_reps, distilled, structure, config, rng)),
        ("portfolio_transfer", portfolio_transfer_builder(target, train_pool, teacher_reps, distilled, structure, config, rng)),
    ]
    candidates: list[tuple[str, CandidateArtifact]] = []
    for name, artifact in base_candidates:
        candidates.append((name, artifact))
        if not artifact.metadata.get("transfer_diversity", 0):
            candidates.append(
                (
                    f"{name}_motif",
                    motif_polish_artifact(target, artifact, profile_reference, config, rng),
                )
            )
            projected = CandidateArtifact(
                adjacency=motif_only_projection(target, artifact.adjacency),
                metadata={
                    "structure_oracle_projection": 1,
                    "structure_oracle_projection_source": name,
                    "transfer_diversity": 1,
                },
                shifts=None,
                corrections=None,
            )
            candidates.append((f"{name}_projection", projected))

    evaluated: list[tuple[str, CandidateArtifact, dict[str, float]]] = []
    for name, artifact in candidates:
        repaired = repair_artifact(target, artifact, config, rng)
        top_shift = top_shift_alignment(target, repaired.adjacency, top_k=min(8, config.max_shift_pool))
        evaluated.append(
            (
                name,
                repaired,
                {
                    "motif_overlap": float(motif_overlap(target, repaired.adjacency)),
                    "edge_disagreement_rate": float(edge_disagreement_rate(target, repaired.adjacency)),
                    "top_shift_jaccard": float(top_shift["top_shift_jaccard"]),
                    "top_shift_precision": float(top_shift["top_shift_precision"]),
                    "density_error": float(density_error(target, repaired.adjacency)),
                },
            )
        )

    rank_score: dict[str, float] = {name: 0.0 for name, _, _ in evaluated}
    metric_specs = [
        ("motif_overlap", True),
        ("edge_disagreement_rate", False),
        ("top_shift_jaccard", True),
        ("top_shift_precision", True),
    ]
    for metric_name, descending in metric_specs:
        ordered = sorted(evaluated, key=lambda item: item[2][metric_name], reverse=descending)
        for rank, (name, _, _) in enumerate(ordered, start=1):
            rank_score[name] += float(rank)

    best_name = ""
    best_score: tuple[float, float] | None = None
    best_artifact: CandidateArtifact | None = None
    for name, artifact, metrics in evaluated:
        score = (rank_score[name], metrics["density_error"])
        if best_score is None or score < best_score:
            best_score = score
            best_name = name
            best_artifact = artifact
    if best_artifact is None:
        return portfolio_transfer_builder(target, train_pool, teacher_reps, distilled, structure, config, rng)
    metadata = dict(best_artifact.metadata)
    metadata["structure_oracle"] = 1
    metadata["structure_oracle_component"] = best_name
    return CandidateArtifact(
        adjacency=best_artifact.adjacency,
        metadata=metadata,
        shifts=None if best_artifact.shifts is None else list(best_artifact.shifts),
        corrections=None if best_artifact.corrections is None else list(best_artifact.corrections),
    )


def search_artifact(result: SearchResult) -> CandidateArtifact:
    return CandidateArtifact(
        adjacency=result.adjacency,
        metadata={
            "search_initial_score": result.initial_score,
            "search_best_score": result.best_score,
            "search_accepted_moves": result.accepted_moves,
            "search_iterations": result.iterations,
            "search_shift_count": len(result.shifts),
            "search_correction_count": len(result.corrections),
        },
        shifts=list(result.shifts),
        corrections=list(result.corrections),
    )


def exact_search_artifact(result: ExactAdjacencySearchResult) -> CandidateArtifact:
    return CandidateArtifact(
        adjacency=result.adjacency,
        metadata={
            "search_initial_score": result.initial_score,
            "search_best_score": result.best_score,
            "search_accepted_moves": result.accepted_moves,
            "search_iterations": result.iterations,
            "search_exact_count": result.exact_count,
            "search_family": "exact_triangle",
            "transfer_diversity": 1,
        },
        shifts=None,
        corrections=None,
    )


def partition_search_artifact(result: PartitionSearchResult, blocks: int) -> CandidateArtifact:
    return CandidateArtifact(
        adjacency=result.adjacency,
        metadata={
            "search_initial_score": result.initial_score,
            "search_best_score": result.best_score,
            "search_accepted_moves": result.accepted_moves,
            "search_iterations": result.iterations,
            "search_shift_count": len(result.active_pairs),
            "search_correction_count": len(result.corrections),
            "partition_blocks": blocks,
        },
    )


def objective_score(
    target: RamseyWitness,
    adjacency: list[list[int]],
    samples: int,
    rng: random.Random,
) -> float:
    clique_subsets, independent_subsets = build_sample_cache(target, samples, rng)
    return sampled_objective(target, adjacency, clique_subsets, independent_subsets)


def repair_artifact(
    target: RamseyWitness,
    artifact: CandidateArtifact,
    config: ExperimentConfig,
    rng: random.Random,
) -> CandidateArtifact:
    if artifact.metadata.get("repair_steps_applied", None) is not None:
        return artifact
    repair_result = greedy_exact_repair(
        target=target,
        adjacency=artifact.adjacency,
        samples=min(400, config.sampled_subsets),
        max_steps=config.exact_repair_steps,
        candidate_edges=config.exact_repair_edge_candidates,
        clique_cutoff=config.exact_repair_clique_cutoff,
        rng=rng,
    )
    metadata = dict(artifact.metadata)
    metadata["repair_steps_applied"] = repair_result.applied_steps
    metadata["repair_initial_score"] = repair_result.initial_score
    metadata["repair_final_score"] = repair_result.final_score
    metadata["repair_exact_count_before"] = repair_result.exact_count_before
    metadata["repair_exact_count_after"] = repair_result.exact_count_after
    shifts = None if artifact.shifts is None else list(artifact.shifts)
    corrections = None
    if shifts is not None:
        corrections = infer_corrections_from_adjacency(repair_result.adjacency, shifts)
    return CandidateArtifact(
        adjacency=repair_result.adjacency,
        metadata=metadata,
        shifts=shifts,
        corrections=corrections,
    )


def transfer_selection_weights(config: ExperimentConfig) -> tuple[float, float]:
    return config.transfer_selection_alignment_weight, config.transfer_selection_correction_weight


def refine_transfer_artifact(
    target: RamseyWitness,
    artifact: CandidateArtifact,
    structure: ExtractedStructure,
    profile_reference: dict[int, float] | None,
    config: ExperimentConfig,
    rng: random.Random,
) -> CandidateArtifact:
    if config.transfer_refine_iterations <= 0:
        return artifact
    if artifact.metadata.get("transfer_diversity", 0):
        return repair_artifact(target, artifact, config, rng)
    repaired = repair_artifact(target, artifact, config, rng)
    shift_budget = estimate_shift_budget(target.n, target.density)
    inferred = list(repaired.shifts) if repaired.shifts is not None else infer_shifts_from_adjacency(repaired.adjacency, shift_budget)
    initial_corrections = (
        list(repaired.corrections)
        if repaired.corrections is not None
        else infer_corrections_from_adjacency(repaired.adjacency, inferred)
    )
    if len(initial_corrections) > config.transfer_refine_max_initial_corrections:
        initial_corrections = initial_corrections[: config.transfer_refine_max_initial_corrections]
    prioritized = list(
        dict.fromkeys(
            inferred + bins_to_shifts(structure.ranked_bins, target.n, config.normalized_shift_bins)
        )
    )
    refined = circulant_local_search(
        target=target,
        initial_shifts=inferred,
        prioritized_shifts=prioritized,
        initial_corrections=initial_corrections,
        samples=min(500, config.sampled_subsets),
        iterations=config.transfer_refine_iterations,
        accept_slack=config.search_accept_slack,
        explore_prob=config.search_explore_prob,
        correction_ratio=config.transfer_refine_correction_ratio,
        max_corrections=config.transfer_refine_max_corrections,
        normalized_bins=config.normalized_shift_bins,
        pair_weights=structure.pair_weights,
        operator_bias={"add": structure.operator_add_bias, "remove": structure.operator_remove_bias},
        rng=rng,
    )
    refined_artifact = search_artifact(refined)
    refined_artifact.metadata["transfer_refine_iterations"] = config.transfer_refine_iterations
    refined_artifact.metadata["transfer_refine_seed_shift_count"] = len(inferred)
    refined_artifact.metadata["transfer_refine_seed_correction_count"] = len(initial_corrections)
    refined_artifact.metadata["transfer_refine_prioritized_count"] = len(prioritized)
    projected_shifts = infer_shifts_from_adjacency(refined_artifact.adjacency, shift_budget)
    projected_artifact = CandidateArtifact(
        adjacency=graph_from_shifts(target.n, projected_shifts),
        metadata={
            "transfer_projected": 1,
            "transfer_projected_shift_count": len(projected_shifts),
        },
        shifts=projected_shifts,
        corrections=[],
    )
    return best_artifact_by_objective(
        target,
        [
            ("transfer_seed", repaired),
            ("transfer_refined", refined_artifact),
            ("transfer_projected", projected_artifact),
        ],
        min(400, config.sampled_subsets),
        config,
        structure,
        profile_reference,
        *transfer_selection_weights(config),
        rng,
    )


def structural_alignment_score(
    target: RamseyWitness,
    adjacency: list[list[int]],
    structure: ExtractedStructure | None,
    normalized_bins: int,
) -> float:
    if structure is None:
        return 0.0
    shift_budget = estimate_shift_budget(target.n, target.density)
    inferred = infer_shifts_from_adjacency(adjacency, shift_budget)
    if not inferred:
        return 0.0
    candidate_bins = {shift_to_bin(shift, target.n, normalized_bins) for shift in inferred}
    shared = set(structure.shared_bins)
    ranked = set(structure.ranked_bins[: max(1, shift_budget)])
    shared_overlap = len(candidate_bins & shared) / max(1, len(shared)) if shared else 0.0
    ranked_overlap = len(candidate_bins & ranked) / max(1, len(ranked)) if ranked else 0.0
    structure_total = sum(max(0.0, weight) for weight in structure.bin_weights.values())
    profile_overlap = 0.0
    if structure_total > 0.0:
        candidate_profile = profile_to_bins(cyclic_shift_profile(adjacency), target.n, normalized_bins)
        candidate_total = sum(max(0.0, weight) for weight in candidate_profile.values())
        if candidate_total > 0.0:
            structure_norm = {
                shift_bin: max(0.0, weight) / structure_total
                for shift_bin, weight in structure.bin_weights.items()
                if weight > 0.0
            }
            candidate_norm = {
                shift_bin: max(0.0, weight) / candidate_total
                for shift_bin, weight in candidate_profile.items()
                if weight > 0.0
            }
            all_bins = set(structure_norm) | set(candidate_norm)
            profile_overlap = sum(
                min(structure_norm.get(shift_bin, 0.0), candidate_norm.get(shift_bin, 0.0))
                for shift_bin in all_bins
            )
    return 0.5 * profile_overlap + 0.3 * shared_overlap + 0.2 * ranked_overlap


def average_teacher_profile(teacher_reps: list[TeacherRepresentation]) -> dict[int, float]:
    if not teacher_reps:
        return {}
    totals: dict[int, float] = {}
    for rep in teacher_reps:
        for shift_bin, weight in rep.normalized_shift_profile.items():
            totals[shift_bin] = totals.get(shift_bin, 0.0) + weight
    count = float(len(teacher_reps))
    averaged = {shift_bin: value / count for shift_bin, value in totals.items()}
    total = sum(averaged.values())
    if total <= 0.0:
        return {}
    return {shift_bin: value / total for shift_bin, value in averaged.items() if value > 0.0}


def teacher_profile_alignment_score(
    target: RamseyWitness,
    adjacency: list[list[int]],
    normalized_bins: int,
    profile_reference: dict[int, float] | None,
) -> float:
    if not profile_reference:
        return 0.0
    candidate_profile = profile_to_bins(cyclic_shift_profile(adjacency), target.n, normalized_bins)
    total = sum(candidate_profile.values())
    if total <= 0.0:
        return 0.0
    candidate_norm = {shift_bin: value / total for shift_bin, value in candidate_profile.items() if value > 0.0}
    all_bins = set(profile_reference) | set(candidate_norm)
    return sum(min(profile_reference.get(shift_bin, 0.0), candidate_norm.get(shift_bin, 0.0)) for shift_bin in all_bins)


def ranked_bins_from_profile(profile_reference: dict[int, float] | None) -> list[int]:
    if not profile_reference:
        return []
    return [shift_bin for shift_bin, _ in sorted(profile_reference.items(), key=lambda item: item[1], reverse=True)]


def profile_shift_candidates(
    target: RamseyWitness,
    current_shifts: list[int],
    profile_reference: dict[int, float] | None,
    normalized_bins: int,
) -> list[list[int]]:
    shift_budget = max(1, estimate_shift_budget(target.n, target.density))
    ranked_bins = ranked_bins_from_profile(profile_reference)
    ranked_shifts = bins_to_shifts(ranked_bins, target.n, normalized_bins)
    current_unique = list(dict.fromkeys(shift for shift in current_shifts if 0 < shift <= target.n // 2))
    ranked_unique = list(dict.fromkeys(ranked_shifts))
    candidates: list[list[int]] = []

    pure_profile = ranked_unique[:shift_budget]
    if pure_profile:
        candidates.append(pure_profile)

    if current_unique:
        profile_first = pure_profile[: max(1, shift_budget // 2)]
        merged_front = list(dict.fromkeys(profile_first + current_unique + ranked_unique))[:shift_budget]
        merged_back = list(dict.fromkeys(current_unique + ranked_unique))[:shift_budget]
        candidates.extend([merged_front, merged_back])

    if current_unique and ranked_unique:
        for replace_count in range(1, min(3, shift_budget) + 1):
            kept = current_unique[: max(0, shift_budget - replace_count)]
            injected = [shift for shift in ranked_unique if shift not in kept][:replace_count]
            proposal = list(dict.fromkeys(injected + kept + ranked_unique))[:shift_budget]
            if proposal:
                candidates.append(proposal)

    deduped: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for proposal in candidates:
        key = tuple(sorted(set(proposal)))
        if key and key not in seen:
            seen.add(key)
            deduped.append(list(key))
    return deduped


def motif_polish_artifact(
    target: RamseyWitness,
    artifact: CandidateArtifact,
    profile_reference: dict[int, float] | None,
    config: ExperimentConfig,
    rng: random.Random,
) -> CandidateArtifact:
    if config.transfer_motif_polish_iterations <= 0 or not profile_reference:
        return artifact
    shift_budget = max(1, estimate_shift_budget(target.n, target.density))
    base_shifts = (
        list(artifact.shifts)
        if artifact.shifts is not None
        else infer_shifts_from_adjacency(artifact.adjacency, shift_budget)
    )
    if not base_shifts:
        return artifact
    base_corrections = (
        list(artifact.corrections)
        if artifact.corrections is not None
        else infer_corrections_from_adjacency(artifact.adjacency, base_shifts)
    )
    base_score = objective_score(target, artifact.adjacency, min(400, config.sampled_subsets), rng)
    best_artifact = artifact
    best_alignment = teacher_profile_alignment_score(
        target,
        artifact.adjacency,
        config.normalized_shift_bins,
        profile_reference,
    )
    remaining_corrections = base_corrections[: config.transfer_refine_max_corrections]
    proposals = profile_shift_candidates(target, base_shifts, profile_reference, config.normalized_shift_bins)
    if not proposals:
        return artifact
    for proposal in proposals[: config.transfer_motif_polish_iterations]:
        candidate = CandidateArtifact(
            adjacency=graph_from_shifts(target.n, proposal),
            metadata={
                "transfer_motif_polish": 1,
                "transfer_motif_polish_shift_count": len(proposal),
            },
            shifts=proposal,
            corrections=[],
        )
        if remaining_corrections:
            corrected = CandidateArtifact(
                adjacency=graph_from_shifts(target.n, proposal),
                metadata=dict(candidate.metadata),
                shifts=proposal,
                corrections=remaining_corrections,
            )
            corrected.adjacency = graph_from_shifts(target.n, proposal)
            for i, j in remaining_corrections:
                corrected.adjacency[i][j] = 1 - corrected.adjacency[i][j]
                corrected.adjacency[j][i] = corrected.adjacency[i][j]
            candidate = corrected
        candidate = repair_artifact(target, candidate, config, rng)
        candidate_score = objective_score(target, candidate.adjacency, min(400, config.sampled_subsets), rng)
        if candidate_score > base_score + config.transfer_selection_objective_tolerance:
            continue
        candidate_alignment = teacher_profile_alignment_score(
            target,
            candidate.adjacency,
            config.normalized_shift_bins,
            profile_reference,
        )
        if candidate_alignment > best_alignment + 1e-9 or (
            abs(candidate_alignment - best_alignment) <= 1e-9 and candidate_score < base_score
        ):
            metadata = dict(candidate.metadata)
            metadata["transfer_motif_polish_base_score"] = base_score
            metadata["transfer_motif_polish_alignment"] = candidate_alignment
            best_artifact = CandidateArtifact(
                adjacency=candidate.adjacency,
                metadata=metadata,
                shifts=None if candidate.shifts is None else list(candidate.shifts),
                corrections=None if candidate.corrections is None else list(candidate.corrections),
            )
            best_alignment = candidate_alignment
            base_score = candidate_score
    return best_artifact


def artifact_correction_count(artifact: CandidateArtifact) -> int:
    if artifact.corrections is not None:
        return len(artifact.corrections)
    value = artifact.metadata.get("search_correction_count", "")
    if value == "":
        return 0
    return int(float(value))


def select_by_objective_then_structure(
    scored: list[tuple[float, float, str, CandidateArtifact]],
    tolerance: float,
) -> CandidateArtifact:
    if not scored:
        raise ValueError("select_by_objective_then_structure requires at least one candidate")
    best_objective = min(item[0] for item in scored)
    eligible = [item for item in scored if item[0] <= best_objective + tolerance]
    eligible.sort(key=lambda item: (item[1], item[0], item[2]))
    return eligible[0][3]


def best_artifact_by_objective(
    target: RamseyWitness,
    candidates: list[tuple[str, CandidateArtifact]],
    samples: int,
    config: ExperimentConfig,
    structure: ExtractedStructure | None,
    profile_reference: dict[int, float] | None,
    alignment_weight: float,
    correction_weight: float,
    rng: random.Random,
) -> CandidateArtifact:
    scored: list[tuple[int | None, float, float, str, CandidateArtifact]] = []
    for name, artifact in candidates:
        repaired = repair_artifact(target, artifact, config, rng)
        score = objective_score(target, repaired.adjacency, samples, rng)
        exact_count = exact_count_k_cliques(repaired.adjacency, target.r) if target.r <= 4 else None
        structure_alignment = structural_alignment_score(target, repaired.adjacency, structure, config.normalized_shift_bins)
        profile_alignment = teacher_profile_alignment_score(
            target,
            repaired.adjacency,
            config.normalized_shift_bins,
            profile_reference,
        )
        alignment = 0.5 * structure_alignment + 0.5 * profile_alignment
        correction_penalty = correction_weight * artifact_correction_count(repaired)
        selection_score = score - (alignment_weight * alignment) + correction_penalty
        metadata = dict(repaired.metadata)
        metadata["portfolio_component"] = name
        metadata["portfolio_objective"] = score
        metadata["portfolio_alignment"] = alignment
        metadata["portfolio_structure_alignment"] = structure_alignment
        metadata["portfolio_profile_alignment"] = profile_alignment
        metadata["portfolio_correction_penalty"] = correction_penalty
        metadata["portfolio_selection_score"] = selection_score
        if exact_count is not None:
            metadata["portfolio_exact_r_clique_count"] = exact_count
        scored.append(
            (
                exact_count,
                score,
                selection_score,
                name,
                CandidateArtifact(
                    repaired.adjacency,
                    metadata,
                    shifts=None if repaired.shifts is None else list(repaired.shifts),
                    corrections=None if repaired.corrections is None else list(repaired.corrections),
                ),
            )
        )
    exact_free = [item for item in scored if item[0] == 0]
    if exact_free:
        scored = exact_free
    if alignment_weight > 0.0 or correction_weight > 0.0:
        return select_by_objective_then_structure(
            [(item[1], item[2], item[3], item[4]) for item in scored],
            config.transfer_selection_objective_tolerance,
        )
    scored.sort(key=lambda item: (item[1], item[2], item[3]))
    return scored[0][4]


def exact_triangle_transfer_builder(
    target: RamseyWitness,
    train_pool: list[RamseyWitness],
    teacher_reps: list[TeacherRepresentation],
    distilled: StudentDistillation,
    structure: ExtractedStructure,
    config: ExperimentConfig,
    rng: random.Random,
) -> CandidateArtifact:
    if target.r != 3:
        return distribution_profile_transfer_builder(target, train_pool, teacher_reps, distilled, structure, config, rng)
    local_rng = random.Random(config.random_seed * 5000 + target.r * 100 + target.s * 10 + target.n)
    target_shift_profile = scale_shift_profile_to_density(
        target.n,
        interpolated_teacher_shift_profile(target, teacher_reps),
        target.density,
    )
    seed_builders = [
        ("distribution_profile_transfer", distribution_profile_transfer_builder),
        ("random_circulant", random_circulant_builder),
        ("random_density", random_builder),
    ]
    best_artifact: CandidateArtifact | None = None
    best_key: tuple[float, float, float] | None = None
    for seed_name, builder in seed_builders:
        seed = builder(target, train_pool, teacher_reps, distilled, structure, config, local_rng)
        result = exact_triangle_local_search(
            target=target,
            initial_adjacency=seed.adjacency,
            target_shift_profile=target_shift_profile,
            samples=min(500, config.sampled_subsets),
            iterations=config.exact_supervision_iterations,
            candidate_edges=config.exact_supervision_candidate_edges,
            shift_candidates=config.exact_supervision_shift_candidates,
            pair_candidates=config.exact_supervision_pair_candidates,
            accept_slack=config.search_accept_slack,
            explore_prob=config.search_explore_prob,
            clique_cutoff=max(config.exact_repair_clique_cutoff, 20000),
            rng=local_rng,
        )
        artifact = exact_search_artifact(result)
        metadata = dict(artifact.metadata)
        metadata["exact_supervision_seed"] = seed_name
        artifact = CandidateArtifact(
            adjacency=artifact.adjacency,
            metadata=metadata,
            shifts=None,
            corrections=None,
        )
        candidate_key = (
            float(metadata.get("search_exact_count", 0.0)),
            abs(graph_density(artifact.adjacency) - target.density),
            float(metadata.get("search_best_score", 0.0)),
        )
        if best_key is None or candidate_key < best_key:
            best_key = candidate_key
            best_artifact = artifact
    if best_artifact is None:
        return distribution_profile_transfer_builder(target, train_pool, teacher_reps, distilled, structure, config, rng)
    return best_artifact


def random_local_search_builder(
    target: RamseyWitness,
    train_pool: list[RamseyWitness],
    teacher_reps: list[TeacherRepresentation],
    distilled: StudentDistillation,
    structure: ExtractedStructure,
    config: ExperimentConfig,
    rng: random.Random,
) -> CandidateArtifact:
    initial = sorted(set(range(1, (target.n // 2) + 1)))
    rng.shuffle(initial)
    initial = initial[: max(1, round(target.density * (target.n - 1) / 2))]
    result = circulant_local_search(
        target=target,
        initial_shifts=initial,
        prioritized_shifts=[],
        initial_corrections=None,
        samples=config.sampled_subsets,
        iterations=config.search_iterations,
        accept_slack=config.search_accept_slack,
        explore_prob=config.search_explore_prob,
        correction_ratio=config.search_correction_ratio,
        max_corrections=config.search_max_corrections,
        normalized_bins=None,
        pair_weights=None,
        operator_bias=None,
        rng=rng,
    )
    return search_artifact(result)


def structured_seed_local_search_builder(
    target: RamseyWitness,
    train_pool: list[RamseyWitness],
    teacher_reps: list[TeacherRepresentation],
    distilled: StudentDistillation,
    structure: ExtractedStructure,
    config: ExperimentConfig,
    rng: random.Random,
) -> CandidateArtifact:
    ranked = scaled_nearest_seed_shifts(target, train_pool, config.normalized_shift_bins)
    initial = ranked[: max(1, round(target.density * (target.n - 1) / 2))]
    result = circulant_local_search(
        target=target,
        initial_shifts=initial,
        prioritized_shifts=ranked,
        initial_corrections=None,
        samples=config.sampled_subsets,
        iterations=config.search_iterations,
        accept_slack=config.search_accept_slack,
        explore_prob=config.search_explore_prob,
        correction_ratio=config.search_correction_ratio,
        max_corrections=config.search_max_corrections,
        normalized_bins=None,
        pair_weights=None,
        operator_bias=None,
        rng=rng,
    )
    return search_artifact(result)


def psi_ramsey_guided_search_builder(
    target: RamseyWitness,
    train_pool: list[RamseyWitness],
    teacher_reps: list[TeacherRepresentation],
    distilled: StudentDistillation,
    structure: ExtractedStructure,
    config: ExperimentConfig,
    rng: random.Random,
) -> CandidateArtifact:
    initial = select_shifts_from_structure(
        target.n,
        structure,
        config.normalized_shift_bins,
        config.pairwise_bonus,
    )
    prioritized = bins_to_shifts(structure.ranked_bins, target.n, config.normalized_shift_bins)
    result = circulant_local_search(
        target=target,
        initial_shifts=initial,
        prioritized_shifts=prioritized,
        initial_corrections=None,
        samples=config.sampled_subsets,
        iterations=config.search_iterations,
        accept_slack=config.search_accept_slack,
        explore_prob=config.search_explore_prob,
        correction_ratio=config.search_correction_ratio,
        max_corrections=config.search_max_corrections,
        normalized_bins=config.normalized_shift_bins,
        pair_weights=structure.pair_weights,
        operator_bias={"add": structure.operator_add_bias, "remove": structure.operator_remove_bias},
        rng=rng,
    )
    return search_artifact(result)


def partition_guided_search_builder(
    target: RamseyWitness,
    train_pool: list[RamseyWitness],
    teacher_reps: list[TeacherRepresentation],
    distilled: StudentDistillation,
    structure: ExtractedStructure,
    config: ExperimentConfig,
    rng: random.Random,
) -> CandidateArtifact:
    template = distill_partition_template(
        train_pool,
        blocks=config.partition_blocks,
        support_threshold=config.partition_support_threshold,
        contrastive_margin=config.partition_contrastive_margin,
    )
    result = partition_local_search(
        target=target,
        template=template,
        iterations=config.search_iterations,
        samples=config.sampled_subsets,
        accept_slack=config.search_accept_slack,
        explore_prob=config.search_explore_prob,
        correction_ratio=config.search_correction_ratio,
        max_corrections=config.search_max_corrections,
        pair_bonus=config.partition_pair_bonus,
        rng=rng,
    )
    return partition_search_artifact(result, template.blocks)


def portfolio_guided_search_builder(
    target: RamseyWitness,
    train_pool: list[RamseyWitness],
    teacher_reps: list[TeacherRepresentation],
    distilled: StudentDistillation,
    structure: ExtractedStructure,
    config: ExperimentConfig,
    rng: random.Random,
) -> CandidateArtifact:
    candidate_count = 4
    split_iterations = max(10, config.search_iterations // (candidate_count + 2))
    portfolio_config = replace(config, search_iterations=split_iterations)
    candidates = [
        ("random_local_search", random_local_search_builder(target, train_pool, teacher_reps, distilled, structure, portfolio_config, rng)),
        ("structured_seed_local_search", structured_seed_local_search_builder(target, train_pool, teacher_reps, distilled, structure, portfolio_config, rng)),
        ("psi_ramsey_guided_search", psi_ramsey_guided_search_builder(target, train_pool, teacher_reps, distilled, structure, portfolio_config, rng)),
        ("partition_guided_search", partition_guided_search_builder(target, train_pool, teacher_reps, distilled, structure, portfolio_config, rng)),
    ]
    artifact = best_artifact_by_objective(
        target,
        candidates,
        min(400, config.sampled_subsets),
        config,
        structure,
        None,
        0.0,
        0.0,
        rng,
    )

    remaining_iterations = max(0, config.search_iterations - split_iterations * candidate_count)
    if remaining_iterations > 0:
        shift_budget = estimate_shift_budget(target.n, target.density)
        inferred = infer_shifts_from_adjacency(artifact.adjacency, shift_budget)
        prioritized = list(
            dict.fromkeys(
                inferred + bins_to_shifts(structure.ranked_bins, target.n, config.normalized_shift_bins)
            )
        )
        refined_result = circulant_local_search(
            target=target,
            initial_shifts=(list(artifact.shifts) if artifact.shifts is not None else inferred),
            prioritized_shifts=prioritized,
            initial_corrections=(
                list(artifact.corrections)
                if artifact.corrections is not None
                else infer_corrections_from_adjacency(artifact.adjacency, inferred)
            ),
            samples=config.sampled_subsets,
            iterations=remaining_iterations,
            accept_slack=config.search_accept_slack,
            explore_prob=config.search_explore_prob,
            correction_ratio=config.search_correction_ratio,
            max_corrections=config.search_max_corrections,
            normalized_bins=config.normalized_shift_bins,
            pair_weights=structure.pair_weights,
            operator_bias={"add": structure.operator_add_bias, "remove": structure.operator_remove_bias},
            rng=rng,
        )
        refined_artifact = search_artifact(refined_result)
        refined_artifact.metadata["portfolio_refine_iterations"] = remaining_iterations
        refined_artifact.metadata["portfolio_seed_component"] = artifact.metadata.get("portfolio_component", "")
        artifact = best_artifact_by_objective(
            target,
            [("portfolio_seed", artifact), ("portfolio_refined", refined_artifact)],
            min(400, config.sampled_subsets),
            config,
            structure,
            None,
            0.0,
            0.0,
            rng,
        )

    artifact.metadata["search_portfolio_size"] = len(candidates)
    artifact.metadata["search_iterations"] = split_iterations * candidate_count + remaining_iterations
    return artifact


SUITES: dict[str, list[tuple[str, CandidateBuilder]]] = {
    "transfer": [
        ("random_density", random_builder),
        ("random_circulant", random_circulant_builder),
        ("structured_seed", structured_seed_builder),
        ("nearest_neighbor_transfer", nearest_builder),
        ("scaled_nearest_transfer", scaled_nearest_builder),
        ("exact_triangle_transfer", exact_triangle_transfer_builder),
        ("partition_transfer", partition_transfer_builder),
        ("portfolio_transfer", portfolio_transfer_builder),
        ("structure_oracle_transfer", structure_oracle_transfer_builder),
        ("teacher_mean_transfer", teacher_mean_transfer),
        ("psi_ramsey_transfer", psi_ramsey_builder),
    ],
    "matched_compute": [
        ("random_density", random_builder),
        ("random_circulant", random_circulant_builder),
        ("structured_seed", structured_seed_builder),
        ("scaled_nearest_transfer", scaled_nearest_builder),
        ("exact_triangle_transfer", exact_triangle_transfer_builder),
        ("partition_transfer", partition_transfer_builder),
        ("portfolio_transfer", portfolio_transfer_builder),
        ("psi_ramsey_transfer", psi_ramsey_builder),
    ],
    "ablations": [
        ("teacher_mean_transfer", teacher_mean_transfer),
        ("student_no_filter_transfer", student_no_filter_transfer),
        ("structured_seed", structured_seed_builder),
        ("psi_ramsey_transfer", psi_ramsey_builder),
    ],
    "interpretability": [
        ("random_circulant", random_circulant_builder),
        ("nearest_neighbor_transfer", nearest_builder),
        ("scaled_nearest_transfer", scaled_nearest_builder),
        ("exact_triangle_transfer", exact_triangle_transfer_builder),
        ("partition_transfer", partition_transfer_builder),
        ("portfolio_transfer", portfolio_transfer_builder),
        ("structure_oracle_transfer", structure_oracle_transfer_builder),
        ("structured_seed", structured_seed_builder),
        ("teacher_mean_transfer", teacher_mean_transfer),
        ("psi_ramsey_transfer", psi_ramsey_builder),
    ],
    "search": [
        ("random_local_search", random_local_search_builder),
        ("structured_seed_local_search", structured_seed_local_search_builder),
        ("psi_ramsey_guided_search", psi_ramsey_guided_search_builder),
        ("partition_guided_search", partition_guided_search_builder),
        ("portfolio_guided_search", portfolio_guided_search_builder),
    ],
}


def available_suites() -> list[str]:
    return sorted(SUITES)


def serialize_config(config: ExperimentConfig) -> dict[str, object]:
    raw = asdict(config)
    raw["witness_dir"] = str(config.witness_dir)
    raw["output_path"] = str(config.output_path)
    raw["results_dir"] = str(config.results_dir)
    return raw


def suite_output_dir(config: ExperimentConfig, suite_name: str) -> Path:
    return config.results_dir / suite_name


def summary_csv_rows(report: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for target_row in report["results"]:
        for baseline in target_row["baselines"]:
            rows.append(
                {
                    "suite": report["suite"],
                    "target": target_row["target"],
                    "r": target_row["r"],
                    "s": target_row["s"],
                    "baseline": baseline["name"],
                    "density_error": baseline["density_error"],
                    "degree_mae": baseline["degree_mae"],
                    "motif_overlap": baseline["motif_overlap"],
                    "edge_disagreement_rate": baseline["edge_disagreement_rate"],
                    "exact_r_clique_count": baseline["exact_r_clique_count"],
                    "exact_r_clique_free": baseline["exact_r_clique_free"],
                    "top_shift_precision": baseline["top_shift_precision"],
                    "top_shift_recall": baseline["top_shift_recall"],
                    "top_shift_jaccard": baseline["top_shift_jaccard"],
                    "sampled_r_clique_rate": baseline["sampled_r_clique_rate"],
                    "sampled_s_independent_rate": baseline["sampled_s_independent_rate"],
                    "validity_proxy": baseline["validity_proxy"],
                    "search_initial_score": baseline.get("search_initial_score", ""),
                    "search_best_score": baseline.get("search_best_score", ""),
                    "search_accepted_moves": baseline.get("search_accepted_moves", ""),
                    "search_iterations": baseline.get("search_iterations", ""),
                    "search_shift_count": baseline.get("search_shift_count", ""),
                    "search_correction_count": baseline.get("search_correction_count", ""),
                    "partition_blocks": baseline.get("partition_blocks", ""),
                    "portfolio_component": baseline.get("portfolio_component", ""),
                    "portfolio_seed_component": baseline.get("portfolio_seed_component", ""),
                    "portfolio_objective": baseline.get("portfolio_objective", ""),
                    "portfolio_alignment": baseline.get("portfolio_alignment", ""),
                    "portfolio_structure_alignment": baseline.get("portfolio_structure_alignment", ""),
                    "portfolio_profile_alignment": baseline.get("portfolio_profile_alignment", ""),
                    "portfolio_correction_penalty": baseline.get("portfolio_correction_penalty", ""),
                    "portfolio_selection_score": baseline.get("portfolio_selection_score", ""),
                    "portfolio_refine_iterations": baseline.get("portfolio_refine_iterations", ""),
                    "transfer_refine_iterations": baseline.get("transfer_refine_iterations", ""),
                    "transfer_refine_seed_shift_count": baseline.get("transfer_refine_seed_shift_count", ""),
                    "transfer_refine_seed_correction_count": baseline.get("transfer_refine_seed_correction_count", ""),
                    "transfer_refine_prioritized_count": baseline.get("transfer_refine_prioritized_count", ""),
                    "search_portfolio_size": baseline.get("search_portfolio_size", ""),
                    "repair_steps_applied": baseline.get("repair_steps_applied", ""),
                    "repair_initial_score": baseline.get("repair_initial_score", ""),
                    "repair_final_score": baseline.get("repair_final_score", ""),
                    "repair_exact_count_before": baseline.get("repair_exact_count_before", ""),
                    "repair_exact_count_after": baseline.get("repair_exact_count_after", ""),
                }
            )
    return rows


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def effective_config(config: ExperimentConfig, smoke: bool) -> ExperimentConfig:
    if not smoke:
        return config
    return replace(
        config,
        sampled_subsets=200,
        teacher_top_k=min(config.teacher_top_k, 8),
        max_shift_pool=min(config.max_shift_pool, 12),
        teacher_trace_iterations=min(config.teacher_trace_iterations, 24),
        teacher_trace_samples=min(config.teacher_trace_samples, 120),
        search_iterations=min(config.search_iterations, 120),
        transfer_refine_iterations=min(config.transfer_refine_iterations, 40),
    )


def aggregate_baseline_metrics(rows: list[dict[str, object]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["baseline"]), []).append(row)

    metrics = [
        "density_error",
        "degree_mae",
        "motif_overlap",
        "edge_disagreement_rate",
        "exact_r_clique_count",
        "exact_r_clique_free",
        "top_shift_precision",
        "top_shift_recall",
        "top_shift_jaccard",
        "sampled_r_clique_rate",
        "sampled_s_independent_rate",
        "validity_proxy",
    ]
    summary: dict[str, dict[str, float | int]] = {}
    for baseline, baseline_rows in grouped.items():
        baseline_summary: dict[str, float | int] = {"targets": len(baseline_rows)}
        for metric in metrics:
            baseline_summary[f"mean_{metric}"] = sum(float(row[metric]) for row in baseline_rows) / len(baseline_rows)
        search_metrics = [
            "search_initial_score",
            "search_best_score",
            "search_accepted_moves",
            "search_iterations",
            "search_shift_count",
            "search_correction_count",
            "partition_blocks",
            "portfolio_objective",
            "portfolio_alignment",
            "portfolio_structure_alignment",
            "portfolio_profile_alignment",
            "portfolio_correction_penalty",
            "portfolio_selection_score",
            "portfolio_refine_iterations",
            "transfer_refine_iterations",
            "transfer_refine_seed_shift_count",
            "transfer_refine_seed_correction_count",
            "transfer_refine_prioritized_count",
            "search_portfolio_size",
            "repair_steps_applied",
            "repair_initial_score",
            "repair_final_score",
            "repair_exact_count_before",
            "repair_exact_count_after",
        ]
        for metric in search_metrics:
            values = [float(row[metric]) for row in baseline_rows if row[metric] != ""]
            if values:
                baseline_summary[f"mean_{metric}"] = sum(values) / len(values)
        summary[baseline] = baseline_summary
    return summary


def ranking_summary(rows: list[dict[str, object]], suite_name: str) -> dict[str, object]:
    metric_preferences = {
        "density_error": "min",
        "degree_mae": "min",
        "edge_disagreement_rate": "min",
        "exact_r_clique_count": "min",
        "motif_overlap": "max",
        "top_shift_jaccard": "max",
        "validity_proxy": "max",
    }
    if suite_name == "search":
        metric_preferences["search_best_score"] = "min"

    mean_ranks: dict[str, dict[str, float]] = {}
    winner_counts: dict[str, dict[str, int]] = {}
    overall_totals: dict[str, float] = {}
    overall_counts: dict[str, int] = {}
    targets = sorted({str(row["target"]) for row in rows})

    for metric, preference in metric_preferences.items():
        per_baseline_totals: dict[str, float] = {}
        per_baseline_counts: dict[str, int] = {}
        per_baseline_wins: dict[str, int] = {}
        for target in targets:
            target_rows = [row for row in rows if str(row["target"]) == target and row.get(metric, "") != ""]
            ranked = sorted(
                target_rows,
                key=lambda row: float(row[metric]),
                reverse=(preference == "max"),
            )
            if not ranked:
                continue
            best_value = float(ranked[0][metric])
            rank_lookup: dict[str, float] = {}
            index = 0
            while index < len(ranked):
                group_start = index
                group_value = float(ranked[index][metric])
                while index < len(ranked) and float(ranked[index][metric]) == group_value:
                    index += 1
                average_rank = ((group_start + 1) + index) / 2.0
                for tied_row in ranked[group_start:index]:
                    rank_lookup[str(tied_row["baseline"])] = average_rank
            for row in ranked:
                baseline = str(row["baseline"])
                per_baseline_totals[baseline] = per_baseline_totals.get(baseline, 0.0) + rank_lookup[baseline]
                per_baseline_counts[baseline] = per_baseline_counts.get(baseline, 0) + 1
                if float(row[metric]) == best_value:
                    per_baseline_wins[baseline] = per_baseline_wins.get(baseline, 0) + 1
        mean_ranks[metric] = {
            baseline: per_baseline_totals[baseline] / per_baseline_counts[baseline]
            for baseline in sorted(per_baseline_totals)
        }
        winner_counts[metric] = {baseline: per_baseline_wins.get(baseline, 0) for baseline in sorted(per_baseline_totals)}
        for baseline, mean_rank in mean_ranks[metric].items():
            overall_totals[baseline] = overall_totals.get(baseline, 0.0) + mean_rank
            overall_counts[baseline] = overall_counts.get(baseline, 0) + 1

    return {
        "targets": len(targets),
        "metric_preferences": metric_preferences,
        "mean_ranks": mean_ranks,
        "winner_counts": winner_counts,
        "overall_mean_rank": {
            baseline: overall_totals[baseline] / overall_counts[baseline]
            for baseline in sorted(overall_totals)
        },
    }


def overview_markdown(global_summary: dict[str, dict[str, object]]) -> str:
    lines = ["# PSI-Ramsey Experiment Overview", ""]
    for suite_name in sorted(global_summary):
        suite_summary = global_summary[suite_name]
        ranking = suite_summary["ranking"]
        aggregate = suite_summary["aggregate"]
        ordered = sorted(ranking["overall_mean_rank"].items(), key=lambda item: item[1])
        lines.append(f"## {suite_name}")
        lines.append(f"- Targets: {ranking['targets']}")
        lines.append(
            "- Overall mean rank: "
            + ", ".join(f"{baseline}={score:.3f}" for baseline, score in ordered[:3])
        )
        if suite_name == "search":
            for baseline, score in ordered[:3]:
                best_score = aggregate[baseline].get("mean_search_best_score")
                validity = aggregate[baseline].get("mean_validity_proxy")
                if best_score is None:
                    continue
                lines.append(
                    f"- {baseline}: mean_search_best_score={float(best_score):.4f}, "
                    f"mean_validity_proxy={float(validity):.4f}"
                )
        else:
            for baseline, score in ordered[:3]:
                clique_count = aggregate[baseline].get("mean_exact_r_clique_count")
                density_error = aggregate[baseline].get("mean_density_error")
                lines.append(
                    f"- {baseline}: mean_exact_r_clique_count={float(clique_count):.1f}, "
                    f"mean_density_error={float(density_error):.4f}"
                )
        lines.append("")
    return "\n".join(lines)


def run_suite(config: ExperimentConfig | None = None, suite_name: str = "transfer", smoke: bool = False) -> dict[str, object]:
    if suite_name not in SUITES:
        raise ValueError(f"Unknown suite '{suite_name}'. Available suites: {', '.join(available_suites())}")

    config = effective_config(default_config() if config is None else config, smoke)
    rng = random.Random(config.random_seed)
    witnesses = load_witnesses(config.witness_dir)
    if smoke:
        witnesses = witnesses[:2]

    results: list[dict[str, object]] = []
    builders = SUITES[suite_name]
    trajectory_cache: dict[str, object] = {}
    for target in witnesses:
        train_pool = training_pool_for_target(target, witnesses, config.target_same_r_only)
        if not train_pool:
            continue

        teacher_reps = []
        teacher_trajectory_distillations = []
        for witness in train_pool:
            rep = build_teacher_representation(witness, config.teacher_top_k, config.normalized_shift_bins)
            teacher_reps.append(rep)
            cache_key = str(witness.path)
            cached = trajectory_cache.get(cache_key)
            if cached is None:
                traces = collect_teacher_search_traces(witness, rep, config, rng)
                cached = distill_teacher_trajectories(traces, witness.n, config.normalized_shift_bins)
                trajectory_cache[cache_key] = cached
            teacher_trajectory_distillations.append(cached)
        distilled = attach_trajectory_priors(distill_student(teacher_reps), teacher_trajectory_distillations)
        structure = extract_structure(
            distilled,
            support_threshold=config.shared_support_threshold,
            max_shift_pool=config.max_shift_pool,
            max_shared_bins=config.max_shared_bins,
            contrastive_margin=config.contrastive_margin,
            pairwise_support_threshold=config.pairwise_support_threshold,
        )
        eval_rng = random.Random(config.random_seed + (target.n * 1000) + (target.r * 100) + target.s)
        evaluation_sample_cache = sample_target_subsets(target, config.sampled_subsets, eval_rng)
        target_prioritized_shifts = select_shifts_from_structure(
            target.n,
            structure,
            config.normalized_shift_bins,
            config.pairwise_bonus,
        )

        baselines = []
        for name, builder in builders:
            artifact = builder(target, train_pool, teacher_reps, distilled, structure, config, rng)
            artifact = repair_artifact(target, artifact, config, rng)
            baseline_metrics = evaluate_candidate(
                name,
                target,
                artifact.adjacency,
                config.sampled_subsets,
                top_k=min(8, config.max_shift_pool),
                rng=rng,
                sample_cache=evaluation_sample_cache,
            )
            baseline_metrics.update(artifact.metadata)
            baselines.append(baseline_metrics)

        target_result = {
            "target": target.path.name,
            "r": target.r,
            "s": target.s,
            "claimed_bound": target.claimed_bound,
            "train_pool": [item.path.name for item in train_pool],
            "shared_structure_bins": structure.shared_bins,
            "ranked_structure_bins": structure.ranked_bins,
            "top_structure_pairs": [list(pair) for pair in structure.top_pairs],
            "structure_ratios": structure_ratios(structure, config.normalized_shift_bins),
            "target_prioritized_shifts": target_prioritized_shifts,
            "baselines": baselines,
        }
        results.append(target_result)

    output_dir = suite_output_dir(config, suite_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "suite": suite_name,
        "smoke": smoke,
        "config": serialize_config(config),
        "results": results,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    rows = summary_csv_rows(report)
    write_summary_csv(output_dir / "summary.csv", rows)
    (output_dir / "aggregate.json").write_text(json.dumps(aggregate_baseline_metrics(rows), indent=2), encoding="utf-8")
    (output_dir / "ranking.json").write_text(json.dumps(ranking_summary(rows, suite_name), indent=2), encoding="utf-8")
    return report


def run_all(config: ExperimentConfig | None = None, smoke: bool = False) -> dict[str, dict[str, object]]:
    config = default_config() if config is None else config
    reports: dict[str, dict[str, object]] = {}
    for suite_name in available_suites():
        reports[suite_name] = run_suite(config=config, suite_name=suite_name, smoke=smoke)
    global_summary = {
        suite_name: {
            "aggregate": aggregate_baseline_metrics(summary_csv_rows(report)),
            "ranking": ranking_summary(summary_csv_rows(report), suite_name),
        }
        for suite_name, report in reports.items()
    }
    config.results_dir.mkdir(parents=True, exist_ok=True)
    (config.results_dir / "all_suites_summary.json").write_text(json.dumps(global_summary, indent=2), encoding="utf-8")
    (config.results_dir / "overview.md").write_text(overview_markdown(global_summary), encoding="utf-8")
    return reports


def run_suite_replicates(
    config: ExperimentConfig | None = None,
    suite_name: str = "transfer",
    smoke: bool = False,
    replicates: int = 1,
) -> dict[str, object]:
    config = default_config() if config is None else config
    seed_summaries: list[dict[str, object]] = []
    for replicate in range(replicates):
        replicate_config = replace(config, random_seed=config.random_seed + replicate)
        report = run_suite(config=replicate_config, suite_name=suite_name, smoke=smoke)
        seed_summaries.append(
            {
                "replicate": replicate,
                "seed": replicate_config.random_seed,
                "aggregate": aggregate_baseline_metrics(summary_csv_rows(report)),
                "ranking": ranking_summary(summary_csv_rows(report), suite_name),
            }
        )

    aggregate_means: dict[str, dict[str, float]] = {}
    for seed_summary in seed_summaries:
        for baseline, metrics in seed_summary["aggregate"].items():
            baseline_summary = aggregate_means.setdefault(baseline, {})
            for metric_name, value in metrics.items():
                if metric_name == "targets":
                    continue
                baseline_summary[metric_name] = baseline_summary.get(metric_name, 0.0) + float(value)
    for baseline, metrics in aggregate_means.items():
        for metric_name in list(metrics):
            metrics[metric_name] /= replicates

    output_dir = suite_output_dir(config, suite_name)
    summary = {
        "suite": suite_name,
        "smoke": smoke,
        "replicates": replicates,
        "base_seed": config.random_seed,
        "aggregate_means": aggregate_means,
        "replicate_summaries": seed_summaries,
    }
    (output_dir / "replicates_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run() -> dict[str, object]:
    report = run_suite(default_config(), suite_name="transfer", smoke=False)
    default_config().output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    report = run()
    print(f"Wrote {len(report['results'])} experiment rows to {default_config().output_path.name}")
