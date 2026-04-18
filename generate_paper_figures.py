from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "psi_ramsey"
FIGURES_DIR = RESULTS_DIR / "paper_figures"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_figures_dir() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)


def pretty_target_label(raw: str) -> str:
    match = re.search(r"R\(\d+,\s*\d+\)", raw)
    return match.group(0).replace(", ", ",") if match else raw


def thousands_formatter(x, _pos) -> str:
    """Format tick labels with comma thousands separator for numbers with 5+ digits."""
    val = int(round(x))
    if abs(val) >= 10000:
        return f"{val:,}"
    return str(val)


THOUSANDS_FMT = ticker.FuncFormatter(thousands_formatter)


def annotate_points(ax: plt.Axes, xs, ys, labels, offsets=None) -> None:
    if offsets is None:
        offsets = [(6, 6)] * len(labels)
    for x, y, label, offset in zip(xs, ys, labels, offsets):
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=offset,
            fontsize=9,
        )


def generate_figure_1(transfer_agg: dict) -> dict:
    methods = [
        "portfolio_transfer",
        "structure_oracle_transfer",
        "hierarchical_coarsening_transfer",
        "flag_density_transfer",
        "random_circulant",
        "random_density",
        "exact_triangle_transfer",
    ]
    labels = [
        "Portfolio",
        "Structure Oracle",
        "Hier. Coarsening",
        "Flag Density",
        "Random Circulant",
        "Random Density",
        "Exact Triangle",
    ]

    xs = []
    ys = []
    for method in methods:
        stats = transfer_agg[method]
        xs.append(stats["mean_exact_r_clique_count"])
        ys.append(
            0.5 * stats["mean_motif_overlap"]
            + 0.5 * stats["mean_top_shift_jaccard"]
        )

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    colors = ["#1b5e20", "#b71c1c", "#00838f", "#f57f17", "#1565c0", "#6a1b9a", "#ef6c00"]
    markers = ["o", "o", "^", "^", "o", "o", "o"]
    handles = []
    for x, y, color, marker, label in zip(xs, ys, colors, markers, labels):
        sc = ax.scatter(x, y, s=130, c=color, edgecolors="black", linewidths=0.7, marker=marker, label=label, zorder=3)
        handles.append(sc)
    ax.legend(
        handles=handles,
        labels=labels,
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        fontsize=9,
        ncol=1,
        borderpad=0.8,
    )
    ax.set_xlabel("Mean exact r-clique count", fontsize=11)
    ax.set_ylabel("Structural agreement score", fontsize=11)
    ax.set_title("Structure-Validity Frontier", fontsize=12)
    ax.xaxis.set_major_formatter(THOUSANDS_FMT)
    style_axes(ax)
    fig.tight_layout()

    path = FIGURES_DIR / "figure_1_structure_validity_frontier.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "figure": 1,
        "title": "Structure-Validity Frontier",
        "path": str(path),
        "caption": (
            "Tradeoff between structural agreement and Ramsey-valid transfer. "
            "The horizontal axis represents mean exact r-clique count, where lower is better. "
            "The vertical axis represents structural agreement, measured as the average of motif "
            "overlap and top-shift Jaccard similarity. Triangular markers denote the two new methods "
            "introduced in the revision (Hierarchical Coarsening Transfer and Flag Algebra Density Transfer). "
            "The figure shows that structure_oracle_transfer lies on the structure-dominant frontier, "
            "whereas portfolio_transfer attains the strongest balance between structural agreement and "
            "Ramsey-valid construction. The new methods occupy distinct positions in the frontier space."
        ),
    }


def generate_figure_2(search_agg: dict) -> dict:
    methods = [
        "portfolio_guided_search",
        "structured_seed_local_search",
        "random_local_search",
        "psi_ramsey_guided_search",
        "partition_guided_search",
    ]
    labels = [
        "Portfolio Guided",
        "Structured Seed",
        "Random Local",
        "PSI Guided",
        "Partition Guided",
    ]

    xs = [search_agg[m]["mean_search_best_score"] for m in methods]
    ys = [search_agg[m]["mean_exact_r_clique_count"] for m in methods]

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    colors = ["#1b5e20", "#1565c0", "#37474f", "#8e24aa", "#ef6c00"]
    ax.scatter(xs, ys, s=95, c=colors, edgecolors="black", linewidths=0.7)
    offsets = [
        (12, 10),   # Portfolio Guided
        (8, 6),     # Structured Seed
        (-52, 6),   # Random Local
        (6, 8),     # PSI Guided
        (6, 6),     # Partition Guided
    ]
    annotate_points(ax, xs, ys, labels, offsets=offsets)
    ax.set_xlabel("Mean search best score")
    ax.set_ylabel("Mean exact r-clique count (log scale)")
    ax.set_yscale("log")
    ax.set_xlim(min(xs) - 0.08, max(xs) + 0.08)
    ax.set_title("Search Tradeoff Between Refinement Score and Exact Validity")
    style_axes(ax)
    fig.tight_layout()

    path = FIGURES_DIR / "figure_2_search_tradeoff.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "figure": 2,
        "title": "Search Tradeoff Between Refinement Score and Exact Validity",
        "path": str(path),
        "caption": (
            "Comparison of search-time methods by mean search best score and mean exact r-clique count. "
            "Lower values are better on both axes. A logarithmic vertical axis is used to make the contrast "
            "between the main search methods and the partition-based outlier readable in a single panel. "
            "The figure shows that portfolio_guided_search provides the strongest balanced search position "
            "among the compared methods."
        ),
    }


def baseline_map(report_entry: dict) -> dict:
    return {item["name"]: item for item in report_entry["baselines"]}


def generate_figure_3(transfer_report: dict) -> dict:
    targets = [pretty_target_label(row["target"]) for row in transfer_report["results"]]
    methods = [
        "portfolio_transfer",
        "hierarchical_coarsening_transfer",
        "flag_density_transfer",
        "structure_oracle_transfer",
        "random_circulant",
        "exact_triangle_transfer",
    ]
    labels = [
        "Portfolio",
        "Hier. Coarsening",
        "Flag Density",
        "Structure Oracle",
        "Random Circulant",
        "Exact Triangle",
    ]

    values = {method: [] for method in methods}
    for row in transfer_report["results"]:
        bmap = baseline_map(row)
        for method in methods:
            values[method].append(bmap[method]["exact_r_clique_count"])

    fig, ax = plt.subplots(figsize=(11.0, 5.6))
    x = np.arange(len(targets))
    n_methods = len(methods)
    width = 0.13
    colors = ["#1b5e20", "#00838f", "#f57f17", "#b71c1c", "#1565c0", "#ef6c00"]

    for idx, method in enumerate(methods):
        offset = (idx - (n_methods - 1) / 2.0) * width
        ax.bar(
            x + offset,
            values[method],
            width=width,
            label=labels[idx],
            color=colors[idx],
            edgecolor="black",
            linewidth=0.4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=15)
    ax.set_ylabel("Exact r-clique count")
    ax.set_xlabel("Target Ramsey cell")
    ax.set_title("Per-Target Exact Clique Counts for Transfer Methods")
    ax.yaxis.set_major_formatter(THOUSANDS_FMT)
    style_axes(ax)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()

    path = FIGURES_DIR / "figure_3_per_target_transfer_counts.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "figure": 3,
        "title": "Per-Target Exact Clique Counts for Transfer Methods",
        "path": str(path),
        "caption": (
            "Per-target comparison of exact r-clique counts across the main transfer baselines, "
            "including the two new methods introduced in the revision. "
            "Lower values are better. The figure shows that Hierarchical Coarsening Transfer "
            "achieves very low clique counts on the R(3,s) cells, while Flag Algebra Density Transfer "
            "is competitive on R(3,s) but produces higher counts on the R(4,s) cells. "
            "Portfolio transfer remains the most balanced method across the full target set."
        ),
    }


def main() -> None:
    ensure_figures_dir()
    transfer_agg = load_json(RESULTS_DIR / "transfer" / "aggregate.json")
    search_agg = load_json(RESULTS_DIR / "search" / "aggregate.json")
    transfer_report = load_json(RESULTS_DIR / "transfer" / "report.json")

    manifest = {
        "source_files": {
            "transfer_aggregate": str(RESULTS_DIR / "transfer" / "aggregate.json"),
            "search_aggregate": str(RESULTS_DIR / "search" / "aggregate.json"),
            "transfer_report": str(RESULTS_DIR / "transfer" / "report.json"),
        },
        "figures": [
            generate_figure_1(transfer_agg),
            generate_figure_2(search_agg),
            generate_figure_3(transfer_report),
        ],
    }

    manifest_path = FIGURES_DIR / "paper_figures_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    main()
