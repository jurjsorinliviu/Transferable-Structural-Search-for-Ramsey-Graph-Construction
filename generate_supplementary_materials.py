from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
MAIN_RESULTS_DIR = ROOT / "results" / "psi_ramsey"
SUPP_RESULTS_DIR = ROOT / "results" / "psi_ramsey_supplementary"
OUTPUT_DIR = SUPP_RESULTS_DIR / "materials"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def style_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)


def structural_score(metrics: dict[str, float]) -> float:
    return 0.5 * float(metrics["mean_motif_overlap"]) + 0.5 * float(metrics["mean_top_shift_jaccard"])


def metric_series_from_replicates(summary: dict, baseline: str, metric: str, source: str) -> list[float]:
    values: list[float] = []
    for replicate in summary["replicate_summaries"]:
        if source == "ranking":
            value = replicate["ranking"]["overall_mean_rank"][baseline]
        else:
            value = replicate["aggregate"][baseline][metric]
        values.append(float(value))
    return values


def stats(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return mean(values), pstdev(values)


def build_replicate_table(
    summary: dict,
    baseline_order: list[str],
    metric_specs: list[tuple[str, str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for baseline in baseline_order:
        row: dict[str, object] = {"Method": baseline}
        for source, metric, label in metric_specs:
            values = metric_series_from_replicates(summary, baseline, metric, source)
            metric_mean, metric_std = stats(values)
            row[f"{label} Mean"] = round(metric_mean, 4)
            row[f"{label} Std"] = round(metric_std, 4)
        rows.append(row)
    return rows


def scenario_record(name: str, suite: str) -> tuple[dict, dict]:
    aggregate = load_json(SUPP_RESULTS_DIR / name / suite / "aggregate.json")
    ranking = load_json(SUPP_RESULTS_DIR / name / suite / "ranking.json")
    return aggregate, ranking


def baseline_order_from_ranking(ranking: dict) -> list[str]:
    ordered = sorted(ranking["overall_mean_rank"].items(), key=lambda item: item[1])
    return [name for name, _ in ordered]


def create_table_s1(transfer_summary: dict) -> Path:
    first_ranking = transfer_summary["replicate_summaries"][0]["ranking"]
    order = baseline_order_from_ranking(first_ranking)
    rows = build_replicate_table(
        transfer_summary,
        order,
        [
            ("ranking", "overall_mean_rank", "Overall Mean Rank"),
            ("aggregate", "mean_exact_r_clique_count", "Exact r-Clique Count"),
            ("aggregate", "mean_motif_overlap", "Motif Overlap"),
            ("aggregate", "mean_top_shift_jaccard", "Top-Shift Jaccard"),
            ("aggregate", "mean_validity_proxy", "Validity Proxy"),
        ],
    )
    path = TABLES_DIR / "table_s1_transfer_seed_robustness.csv"
    write_csv(path, rows)
    return path


def create_table_s2(search_summary: dict) -> Path:
    first_ranking = search_summary["replicate_summaries"][0]["ranking"]
    order = baseline_order_from_ranking(first_ranking)
    rows = build_replicate_table(
        search_summary,
        order,
        [
            ("ranking", "overall_mean_rank", "Overall Mean Rank"),
            ("aggregate", "mean_search_best_score", "Search Best Score"),
            ("aggregate", "mean_exact_r_clique_count", "Exact r-Clique Count"),
            ("aggregate", "mean_validity_proxy", "Validity Proxy"),
        ],
    )
    path = TABLES_DIR / "table_s2_search_seed_robustness.csv"
    write_csv(path, rows)
    return path


def create_table_s3(ablation_summary: dict) -> Path:
    first_ranking = ablation_summary["replicate_summaries"][0]["ranking"]
    order = baseline_order_from_ranking(first_ranking)
    rows = build_replicate_table(
        ablation_summary,
        order,
        [
            ("ranking", "overall_mean_rank", "Overall Mean Rank"),
            ("aggregate", "mean_exact_r_clique_count", "Exact r-Clique Count"),
            ("aggregate", "mean_density_error", "Density Error"),
            ("aggregate", "mean_motif_overlap", "Motif Overlap"),
        ],
    )
    path = TABLES_DIR / "table_s3_ablation_seed_robustness.csv"
    write_csv(path, rows)
    return path


def create_table_s4(interpret_summary: dict) -> Path:
    first_ranking = interpret_summary["replicate_summaries"][0]["ranking"]
    order = baseline_order_from_ranking(first_ranking)
    rows = build_replicate_table(
        interpret_summary,
        order,
        [
            ("ranking", "overall_mean_rank", "Overall Mean Rank"),
            ("aggregate", "mean_exact_r_clique_count", "Exact r-Clique Count"),
            ("aggregate", "mean_density_error", "Density Error"),
            ("aggregate", "mean_motif_overlap", "Motif Overlap"),
            ("aggregate", "mean_top_shift_jaccard", "Top-Shift Jaccard"),
            ("aggregate", "mean_validity_proxy", "Validity Proxy"),
        ],
    )
    for row in rows:
        row["Structural Score Mean"] = round(
            0.5 * float(row["Motif Overlap Mean"]) + 0.5 * float(row["Top-Shift Jaccard Mean"]),
            4,
        )
    path = TABLES_DIR / "table_s4_interpretability_seed_robustness.csv"
    write_csv(path, rows)
    return path


def create_table_s5(main_transfer_agg: dict, main_transfer_ranking: dict) -> Path:
    scenarios = [
        ("main_transfer", "Main transfer suite", main_transfer_agg, main_transfer_ranking),
        ("transfer_compute_low_budget", "Low-budget transfer", *scenario_record("transfer_compute_low_budget", "transfer")),
        ("transfer_compute_high_budget", "High-budget transfer", *scenario_record("transfer_compute_high_budget", "transfer")),
        ("mixed_r_transfer_neighborhood", "Mixed-r teacher pool", *scenario_record("mixed_r_transfer_neighborhood", "transfer")),
        ("high_resolution_structure", "High-resolution structure", *scenario_record("high_resolution_structure", "transfer")),
        ("compact_structure", "Compact structure", *scenario_record("compact_structure", "transfer")),
        ("exact_supervision_stress", "Stronger exact supervision", *scenario_record("exact_supervision_stress", "transfer")),
    ]
    rows: list[dict[str, object]] = []
    for key, label, aggregate, ranking in scenarios:
        portfolio = aggregate["portfolio_transfer"]
        exact_triangle = aggregate["exact_triangle_transfer"]
        oracle = aggregate["structure_oracle_transfer"]
        rows.append(
            {
                "Scenario": label,
                "Portfolio Overall Mean Rank": round(float(ranking["overall_mean_rank"]["portfolio_transfer"]), 4),
                "Portfolio Exact r-Clique Count": round(float(portfolio["mean_exact_r_clique_count"]), 1),
                "Portfolio Validity Proxy": round(float(portfolio["mean_validity_proxy"]), 4),
                "Exact Triangle Exact r-Clique Count": round(float(exact_triangle["mean_exact_r_clique_count"]), 1),
                "Exact Triangle Validity Proxy": round(float(exact_triangle["mean_validity_proxy"]), 4),
                "Structure Oracle Exact r-Clique Count": round(float(oracle["mean_exact_r_clique_count"]), 1),
                "Structure Oracle Structural Score": round(structural_score(oracle), 4),
            }
        )
    path = TABLES_DIR / "table_s5_transfer_sensitivity.csv"
    write_csv(path, rows)
    return path


def create_table_s6(main_search_agg: dict, main_search_ranking: dict) -> Path:
    high_search_agg, high_search_ranking = scenario_record("search_compute_high_budget", "search")
    scenarios = [
        ("main_search", "Main search suite", main_search_agg, main_search_ranking),
        ("search_compute_high_budget", "High-budget search", high_search_agg, high_search_ranking),
    ]
    rows: list[dict[str, object]] = []
    tracked = [
        "portfolio_guided_search",
        "structured_seed_local_search",
        "random_local_search",
    ]
    for _, label, aggregate, ranking in scenarios:
        row: dict[str, object] = {"Scenario": label}
        for method in tracked:
            prefix = method.replace("_", " ").title()
            row[f"{prefix} Overall Mean Rank"] = round(float(ranking["overall_mean_rank"][method]), 4)
            row[f"{prefix} Search Best Score"] = round(float(aggregate[method]["mean_search_best_score"]), 4)
            row[f"{prefix} Exact r-Clique Count"] = round(float(aggregate[method]["mean_exact_r_clique_count"]), 1)
        rows.append(row)
    path = TABLES_DIR / "table_s6_search_budget_sensitivity.csv"
    write_csv(path, rows)
    return path


def create_figure_s1(transfer_summary: dict) -> dict:
    methods = [
        "portfolio_transfer",
        "structure_oracle_transfer",
        "exact_triangle_transfer",
        "random_circulant",
    ]
    labels = ["Portfolio", "Structure Oracle", "Exact Triangle", "Random Circulant"]
    data = [metric_series_from_replicates(transfer_summary, method, "overall_mean_rank", "ranking") for method in methods]

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    box = ax.boxplot(data, patch_artist=True, tick_labels=labels)
    colors = ["#1b5e20", "#b71c1c", "#ef6c00", "#1565c0"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for median in box["medians"]:
        median.set_color("black")
        median.set_linewidth(1.2)
    ax.set_ylabel("Overall mean rank")
    ax.set_title("Transfer Seed Robustness")
    style_axes(ax)
    fig.tight_layout()

    path = FIGURES_DIR / "figure_s1_transfer_seed_robustness.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "figure": "S1",
        "title": "Transfer Seed Robustness",
        "path": str(path),
        "caption": (
            "Distribution of overall mean ranks across eight replicated transfer runs. "
            "Lower values are better. The figure shows that portfolio_transfer remains "
            "the most stable balanced method across seed variation."
        ),
    }


def create_figure_s2(search_summary: dict) -> dict:
    methods = [
        "portfolio_guided_search",
        "structured_seed_local_search",
        "random_local_search",
        "psi_ramsey_guided_search",
    ]
    labels = ["Portfolio Guided", "Structured Seed", "Random Local", "Proposed Framework Guided"]
    data = [metric_series_from_replicates(search_summary, method, "overall_mean_rank", "ranking") for method in methods]

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    box = ax.boxplot(data, patch_artist=True, tick_labels=labels)
    colors = ["#1b5e20", "#1565c0", "#455a64", "#8e24aa"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for median in box["medians"]:
        median.set_color("black")
        median.set_linewidth(1.2)
    ax.set_ylabel("Overall mean rank")
    ax.set_title("Search Seed Robustness")
    style_axes(ax)
    fig.tight_layout()

    path = FIGURES_DIR / "figure_s2_search_seed_robustness.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "figure": "S2",
        "title": "Search Seed Robustness",
        "path": str(path),
        "caption": (
            "Distribution of overall mean ranks across eight replicated search runs. "
            "Lower values are better. Portfolio-guided search remains competitive and "
            "stable under seed variation."
        ),
    }


def create_figure_s3(main_transfer_agg: dict) -> dict:
    scenarios = [
        ("Main", main_transfer_agg),
        ("Low budget", scenario_record("transfer_compute_low_budget", "transfer")[0]),
        ("High budget", scenario_record("transfer_compute_high_budget", "transfer")[0]),
        ("Mixed-r", scenario_record("mixed_r_transfer_neighborhood", "transfer")[0]),
        ("High-res", scenario_record("high_resolution_structure", "transfer")[0]),
        ("Compact", scenario_record("compact_structure", "transfer")[0]),
        ("Exact-stress", scenario_record("exact_supervision_stress", "transfer")[0]),
    ]
    methods = [
        ("portfolio_transfer", "Portfolio"),
        ("exact_triangle_transfer", "Exact Triangle"),
        ("structure_oracle_transfer", "Structure Oracle"),
    ]

    x = np.arange(len(scenarios))
    width = 0.24
    colors = ["#1b5e20", "#ef6c00", "#b71c1c"]
    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    for idx, (method, label) in enumerate(methods):
        values = [float(aggregate[method]["mean_exact_r_clique_count"]) for _, aggregate in scenarios]
        ax.bar(
            x + (idx - 1) * width,
            values,
            width=width,
            label=label,
            color=colors[idx],
            edgecolor="black",
            linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in scenarios], rotation=15)
    ax.set_ylabel("Mean exact r-clique count")
    ax.set_title("Transfer Sensitivity Across Supplementary Scenarios")
    style_axes(ax)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()

    path = FIGURES_DIR / "figure_s3_transfer_sensitivity.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "figure": "S3",
        "title": "Transfer Sensitivity Across Supplementary Scenarios",
        "path": str(path),
        "caption": (
            "Comparison of mean exact r-clique counts for three representative transfer methods "
            "across the main setting and six supplementary transfer stress tests. Lower values are better."
        ),
    }


def create_figure_s4(main_transfer_agg: dict, main_transfer_ranking: dict) -> dict:
    scenarios = [
        ("Main", main_transfer_agg, main_transfer_ranking),
        ("Low budget", *scenario_record("transfer_compute_low_budget", "transfer")),
        ("High budget", *scenario_record("transfer_compute_high_budget", "transfer")),
        ("Mixed-r", *scenario_record("mixed_r_transfer_neighborhood", "transfer")),
        ("High-res", *scenario_record("high_resolution_structure", "transfer")),
        ("Compact", *scenario_record("compact_structure", "transfer")),
        ("Exact-stress", *scenario_record("exact_supervision_stress", "transfer")),
    ]
    x = np.arange(len(scenarios))
    portfolio_exact = [float(aggregate["portfolio_transfer"]["mean_exact_r_clique_count"]) for _, aggregate, _ in scenarios]
    portfolio_rank = [float(ranking["overall_mean_rank"]["portfolio_transfer"]) for _, _, ranking in scenarios]

    fig, ax1 = plt.subplots(figsize=(10.4, 5.6))
    bars = ax1.bar(x, portfolio_exact, width=0.6, color="#1b5e20", alpha=0.78, edgecolor="black", linewidth=0.4, label="Portfolio exact count")
    ax1.set_ylabel("Portfolio mean exact r-clique count", color="#1b5e20")
    ax1.tick_params(axis="y", labelcolor="#1b5e20")
    ax1.set_xticks(x)
    ax1.set_xticklabels([label for label, _, _ in scenarios], rotation=15)
    style_axes(ax1)

    ax2 = ax1.twinx()
    line = ax2.plot(x, portfolio_rank, color="#b71c1c", marker="o", linewidth=2.2, label="Portfolio overall mean rank")
    ax2.set_ylabel("Portfolio overall mean rank", color="#b71c1c")
    ax2.tick_params(axis="y", labelcolor="#b71c1c")

    ax1.set_title("Portfolio Transfer Sensitivity Across Supplementary Scenarios")
    legend_handles = [bars, line[0]]
    legend_labels = ["Portfolio exact count", "Portfolio overall mean rank"]
    ax1.legend(legend_handles, legend_labels, frameon=False, loc="upper right")
    fig.tight_layout()

    path = FIGURES_DIR / "figure_s4_portfolio_transfer_sensitivity.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "figure": "S4",
        "title": "Portfolio Transfer Sensitivity Across Supplementary Scenarios",
        "path": str(path),
        "caption": (
            "Sensitivity of the portfolio transfer method across supplementary transfer scenarios. "
            "Bars show mean exact r-clique counts and the line shows overall mean rank. Lower values are better on both axes."
        ),
    }


def create_materials_summary(
    main_transfer_agg: dict,
    main_transfer_ranking: dict,
    main_search_agg: dict,
    main_search_ranking: dict,
    transfer_summary: dict,
    search_summary: dict,
    ablation_summary: dict,
    interpret_summary: dict,
) -> dict:
    summary = {
        "main": {
            "transfer_portfolio_overall_mean_rank": main_transfer_ranking["overall_mean_rank"]["portfolio_transfer"],
            "search_portfolio_overall_mean_rank": main_search_ranking["overall_mean_rank"]["portfolio_guided_search"],
        },
        "seed_robustness": {
            "transfer_portfolio_mean_rank_mean_std": stats(
                metric_series_from_replicates(transfer_summary, "portfolio_transfer", "overall_mean_rank", "ranking")
            ),
            "search_portfolio_mean_rank_mean_std": stats(
                metric_series_from_replicates(search_summary, "portfolio_guided_search", "overall_mean_rank", "ranking")
            ),
        },
        "transfer_scenarios": {},
    }
    scenario_names = [
        "transfer_compute_low_budget",
        "transfer_compute_high_budget",
        "mixed_r_transfer_neighborhood",
        "high_resolution_structure",
        "compact_structure",
        "exact_supervision_stress",
    ]
    for name in scenario_names:
        aggregate, ranking = scenario_record(name, "transfer")
        summary["transfer_scenarios"][name] = {
            "portfolio_overall_mean_rank": ranking["overall_mean_rank"]["portfolio_transfer"],
            "portfolio_exact_r_clique_count": aggregate["portfolio_transfer"]["mean_exact_r_clique_count"],
            "portfolio_validity_proxy": aggregate["portfolio_transfer"]["mean_validity_proxy"],
            "oracle_structural_score": structural_score(aggregate["structure_oracle_transfer"]),
        }
    summary["ablation"] = {
        baseline: ablation_summary["aggregate_means"][baseline]["mean_exact_r_clique_count"]
        for baseline in ablation_summary["aggregate_means"]
    }
    summary["interpretability"] = {
        baseline: {
            "exact_r_clique_count": interpret_summary["aggregate_means"][baseline]["mean_exact_r_clique_count"],
            "structural_score": 0.5
            * interpret_summary["aggregate_means"][baseline]["mean_motif_overlap"]
            + 0.5 * interpret_summary["aggregate_means"][baseline]["mean_top_shift_jaccard"],
        }
        for baseline in interpret_summary["aggregate_means"]
    }
    return summary


def main() -> None:
    ensure_dirs()

    main_transfer_agg = load_json(MAIN_RESULTS_DIR / "transfer" / "aggregate.json")
    main_transfer_ranking = load_json(MAIN_RESULTS_DIR / "transfer" / "ranking.json")
    main_search_agg = load_json(MAIN_RESULTS_DIR / "search" / "aggregate.json")
    main_search_ranking = load_json(MAIN_RESULTS_DIR / "search" / "ranking.json")

    transfer_summary = load_json(SUPP_RESULTS_DIR / "transfer_seed_robustness" / "transfer" / "replicates_summary.json")
    search_summary = load_json(SUPP_RESULTS_DIR / "search_seed_robustness" / "search" / "replicates_summary.json")
    ablation_summary = load_json(SUPP_RESULTS_DIR / "ablation_seed_robustness" / "ablations" / "replicates_summary.json")
    interpret_summary = load_json(SUPP_RESULTS_DIR / "interpretability_seed_robustness" / "interpretability" / "replicates_summary.json")

    table_paths = [
        create_table_s1(transfer_summary),
        create_table_s2(search_summary),
        create_table_s3(ablation_summary),
        create_table_s4(interpret_summary),
        create_table_s5(main_transfer_agg, main_transfer_ranking),
        create_table_s6(main_search_agg, main_search_ranking),
    ]

    figures = [
        create_figure_s1(transfer_summary),
        create_figure_s2(search_summary),
        create_figure_s3(main_transfer_agg),
        create_figure_s4(main_transfer_agg, main_transfer_ranking),
    ]

    summary = create_materials_summary(
        main_transfer_agg,
        main_transfer_ranking,
        main_search_agg,
        main_search_ranking,
        transfer_summary,
        search_summary,
        ablation_summary,
        interpret_summary,
    )
    summary_path = OUTPUT_DIR / "supplementary_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    manifest = {
        "tables": [str(path) for path in table_paths],
        "figures": figures,
        "summary": str(summary_path),
    }
    manifest_path = OUTPUT_DIR / "supplementary_materials_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
