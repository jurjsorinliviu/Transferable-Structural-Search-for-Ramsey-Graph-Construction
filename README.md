# From Cell-Specific Heuristics to Transferable Structural Search for Ramsey Graph Construction

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Results](https://img.shields.io/badge/results-main%20%2B%20supplementary-success.svg)](#headline-results)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/jurjsorinliviu/Transferable-Structural-Search-for-Ramsey-Graph-Construction)

> **Scope notice**
>
> This repository contains implementation code, experiment outputs, paper figures, and manuscript artifacts accompanying the study of transferable structural search in Ramsey graph construction.  
> The manuscript and supplementary document are the authoritative sources for formal claims, definitions, and evaluation details.

<img width="1855" height="1291" alt="Proposed_Framework" src="https://github.com/user-attachments/assets/86c04825-9ad2-47f8-b878-55dc7afaea09" />

This repository contains a proof-of-concept framework for **transferable structural search in Ramsey graph construction**. It adapts a teacher-student structure-distillation pattern inspired by [Ψ-NN](https://www.nature.com/articles/s41467-025-64624-3) and applies it to Ramsey witnesses, structural motif extraction, candidate reconstruction, local refinement, and portfolio-based selection across related Ramsey cells.

---

## 🔭 Overview

The project studies whether useful search structure can be transferred across related Ramsey cells instead of rediscovered independently for each target.

The current evaluation uses five target cells:

- `R(3,13)`
- `R(3,18)`
- `R(4,13)`
- `R(4,14)`
- `R(4,15)`

The main experimental suites are:

- `transfer`
- `matched_compute`
- `search`
- `ablations`
- `interpretability`

The supplementary package adds 11 reviewer-oriented robustness and stress-test blocks, including:

- seed robustness,
- budget sensitivity,
- mixed-`r` teacher neighborhoods,
- structural-resolution sensitivity,
- stronger exact supervision.

---

## Method at a Glance

1. Build teacher representations from known Ramsey witnesses and teacher search traces.
2. Distill a compact student summary of recurring structural patterns across related cells.
3. Extract reusable structural priors such as ranked bins, pairwise motif relations, and global targets.
4. Reconstruct candidate graphs for unseen target cells from multiple transfer families.
5. Refine candidates with balanced local search and optional exact supervision for weak `R(3,s)` cells.
6. Select the strongest candidate through a portfolio mechanism rather than a single fixed transfer rule.

---

## 📊 Headline Results

The main result files in [`results/psi_ramsey`](results/psi_ramsey) show:

| Suite            | Main finding                                                 | Source                                                       |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Transfer         | `portfolio_transfer` is the strongest balanced transfer method with overall mean rank `4.186` and mean exact `r`-clique count `3834.2`. | `results/psi_ramsey/transfer/aggregate.json`, `results/psi_ramsey/transfer/ranking.json` |
| Matched compute  | `portfolio_transfer` remains first overall under controlled compute with overall mean rank `3.471` and mean exact `r`-clique count `2602.8`. | `results/psi_ramsey/matched_compute/aggregate.json`, `results/psi_ramsey/matched_compute/ranking.json` |
| Search           | `portfolio_guided_search` is the strongest balanced search method in the synchronized runs with overall mean rank `2.538`. | `results/psi_ramsey/search/aggregate.json`, `results/psi_ramsey/search/ranking.json` |
| Ablations        | `structured_seed` ranks best inside the restricted ablation family, while the distilled transfer variants substantially change the exact validity profile. | `results/psi_ramsey/ablations/aggregate.json`, `results/psi_ramsey/ablations/ranking.json` |
| Interpretability | `portfolio_transfer` and `structure_oracle_transfer` separate balanced Ramsey-valid transfer from structure-dominant witness-shape recovery. | `results/psi_ramsey/interpretability/aggregate.json`, `results/psi_ramsey/interpretability/ranking.json` |

Supplementary materials in [`results/psi_ramsey_supplementary`](results/psi_ramsey_supplementary) extend these results with:

- 6 summary tables,
- 4 supplementary figures,
- 11 extra experiment blocks,
- a generated supplementary manifest and summary file.

---

## 📑 Main Paper Tables and Figures

The repository already contains the artifacts behind the main manuscript results.

### Main tables

| Manuscript table                                         | Backing result files                                         |
| -------------------------------------------------------- | ------------------------------------------------------------ |
| Table 1. Transfer Performance Across Target Ramsey Cells | `results/psi_ramsey/transfer/summary.csv`, `aggregate.json`, `ranking.json` |
| Table 2. Matched-Compute Comparison                      | `results/psi_ramsey/matched_compute/summary.csv`, `aggregate.json`, `ranking.json` |
| Table 3. Search-Suite Performance                        | `results/psi_ramsey/search/summary.csv`, `aggregate.json`, `ranking.json` |
| Table 4. Ablation Study                                  | `results/psi_ramsey/ablations/summary.csv`, `aggregate.json`, `ranking.json` |
| Table 5. Structural Frontier Analysis                    | `results/psi_ramsey/interpretability/summary.csv`, `aggregate.json`, `ranking.json` |

### Main figures

Generated by `generate_paper_figures.py` and stored in [`results/psi_ramsey/paper_figures`](results/psi_ramsey/paper_figures):

| Manuscript figure                                            | File                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Figure 1. Structure-Validity Frontier                        | `results/psi_ramsey/paper_figures/figure_1_structure_validity_frontier.png` |
| Figure 2. Search Tradeoff Between Refinement Score and Exact Validity | `results/psi_ramsey/paper_figures/figure_2_search_tradeoff.png` |
| Figure 3. Per-Target Exact Clique Counts for Transfer Methods | `results/psi_ramsey/paper_figures/figure_3_per_target_transfer_counts.png` |
| Figure manifest                                              | `results/psi_ramsey/paper_figures/paper_figures_manifest.json` |

### Supplementary tables and figures

Generated by `generate_supplementary_materials.py` and stored in [`results/psi_ramsey_supplementary/materials`](results/psi_ramsey_supplementary/materials):

| Supplementary artifact                                       | File                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Table S1. Transfer Seed Robustness                           | `results/psi_ramsey_supplementary/materials/tables/table_s1_transfer_seed_robustness.csv` |
| Table S2. Search Seed Robustness                             | `results/psi_ramsey_supplementary/materials/tables/table_s2_search_seed_robustness.csv` |
| Table S3. Ablation Seed Robustness                           | `results/psi_ramsey_supplementary/materials/tables/table_s3_ablation_seed_robustness.csv` |
| Table S4. Interpretability Seed Robustness                   | `results/psi_ramsey_supplementary/materials/tables/table_s4_interpretability_seed_robustness.csv` |
| Table S5. Transfer Sensitivity                               | `results/psi_ramsey_supplementary/materials/tables/table_s5_transfer_sensitivity.csv` |
| Table S6. Search Budget Sensitivity                          | `results/psi_ramsey_supplementary/materials/tables/table_s6_search_budget_sensitivity.csv` |
| Figure S1. Transfer Seed Robustness                          | `results/psi_ramsey_supplementary/materials/figures/figure_s1_transfer_seed_robustness.png` |
| Figure S2. Search Seed Robustness                            | `results/psi_ramsey_supplementary/materials/figures/figure_s2_search_seed_robustness.png` |
| Figure S3. Transfer Sensitivity Across Supplementary Scenarios | `results/psi_ramsey_supplementary/materials/figures/figure_s3_transfer_sensitivity.png` |
| Figure S4. Portfolio Transfer Sensitivity                    | `results/psi_ramsey_supplementary/materials/figures/figure_s4_portfolio_transfer_sensitivity.png` |
| Supplementary manifest                                       | `results/psi_ramsey_supplementary/materials/supplementary_materials_manifest.json` |

---

## 🗂️ Repository Structure

```text
Transferable-Structural-Search-for-Ramsey-Graph-Construction/
|-- main.py
|-- run_psi_ramsey_experiments.py
|-- supplementary_experiments.py
|-- generate_paper_figures.py
|-- generate_supplementary_materials.py
|-- experiment_config.py
|-- ramsey_*.py
|-- fig1_psi_ramsey_framework.html
|-- results/
|   |-- psi_ramsey/
|   `-- psi_ramsey_supplementary/
`-- related work/
```

### Core files

| File                                  | Purpose                                                      |
| ------------------------------------- | ------------------------------------------------------------ |
| `main.py`                             | CLI entry point for running the main paper suites.           |
| `run_psi_ramsey_experiments.py`       | Main experiment engine and suite orchestration.              |
| `supplementary_experiments.py`        | Runs the reviewer-oriented supplementary experiment blocks.  |
| `generate_paper_figures.py`           | Builds the main paper figures from saved JSON results.       |
| `generate_supplementary_materials.py` | Builds supplementary tables, figures, and manifests.         |
| `experiment_config.py`                | Default experiment configuration and output locations.       |
| `ramsey_data.py`                      | Witness loading, exact clique counting, sampling, and graph utilities. |
| `ramsey_teacher.py`                   | Teacher representation construction.                         |
| `ramsey_student.py`                   | Student distillation and trajectory priors.                  |
| `ramsey_structure.py`                 | Explicit structure extraction.                               |
| `ramsey_reconstruct.py`               | Candidate reconstruction from extracted structure.           |
| `ramsey_search.py`                    | Local search, sampled objective, and exact small-cell refinement. |
| `ramsey_repair.py`                    | Greedy exact-repair helpers.                                 |
| `ramsey_metrics.py`                   | Structural and validity-related metrics.                     |
| `ramsey_baselines.py`                 | Baseline candidate generators.                               |

---

## 🧩 Data and Witnesses

By default, the code reads witness graphs from:

`related work/ramsey_number_bounds/improved_bounds`

These witness files are based on [publicly available Ramsey-number bound resources](https://github.com/google-research/google-research/tree/master/ramsey_number_bounds/improved_bounds) associated with recent computational [Ramsey-search work](https://arxiv.org/abs/2603.09172), including the Google Research `ramsey_number_bounds/improved_bounds` repository.

The filename pattern is parsed from entries such as:

`R(4, 13) _= 139.txt`

The default experiment config is defined in [`experiment_config.py`](experiment_config.py), including:

- witness directory,
- random seed,
- search budget,
- transfer refinement budget,
- structural binning parameters,
- output directories for main and supplementary runs.

---

## 💻 Running Locally

### 1. Python environment

The main experiment pipeline uses the Python standard library. Figure generation and supplementary material generation additionally use:

- `numpy`
- `matplotlib`

A simple local setup is:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Run the main suites

Run all main paper suites:

```bash
python main.py --suite all
```

Run one suite at a time:

```bash
python main.py --suite transfer
python main.py --suite matched_compute
python main.py --suite search
python main.py --suite ablations
python main.py --suite interpretability
```

Run a smoke test:

```bash
python main.py --suite transfer --smoke
```

Run repeated seeds for one suite:

```bash
python main.py --suite transfer --replicates 4
```

Override a few parameters:

```bash
python main.py --suite transfer --seed 2026 --sampled-subsets 1800 --search-iterations 700 --teacher-top-k 20 --max-shift-pool 28
```

### 3. Generate the main paper figures

```bash
python generate_paper_figures.py
```

This writes:

`results/psi_ramsey/paper_figures`

### 4. Run supplementary experiments

List available supplementary blocks:

```bash
python supplementary_experiments.py --list
```

Run all supplementary experiments:

```bash
python supplementary_experiments.py --experiment all
```

Run one supplementary block:

```bash
python supplementary_experiments.py --experiment transfer_seed_robustness
python supplementary_experiments.py --experiment search_compute_high_budget
```

Run a smoke version:

```bash
python supplementary_experiments.py --experiment all --smoke
```

### 5. Generate supplementary tables and figures

```bash
python generate_supplementary_materials.py
```

This writes:

`results/psi_ramsey_supplementary/materials`

---

## ☁️ Running in GitHub Codespaces

This repository now includes:

- a lightweight [`requirements.txt`](requirements.txt)
- a minimal [`.devcontainer/devcontainer.json`](.devcontainer/devcontainer.json)

That means a new Codespace should start with Python available and the proposed framework dependencies installed automatically.

Suggested Codespaces workflow:

1. Open the repository in Codespaces.
2. Wait for the container to finish its post-create setup.
3. Open a terminal in the repo root.
4. Run the experiment or figure-generation commands directly.

Typical Codespaces commands:

```bash
python main.py --suite all
python generate_paper_figures.py
python supplementary_experiments.py --experiment all
python generate_supplementary_materials.py
```

If you only want a quick verification in Codespaces, use:

```bash
python main.py --suite transfer --smoke
python supplementary_experiments.py --experiment transfer_compute_low_budget --smoke
```

---

## 🧪 Results Layout

### Main results

Main paper outputs are under [`results/psi_ramsey`](results/psi_ramsey):

| Path                                         | Content                                   |
| -------------------------------------------- | ----------------------------------------- |
| `results/psi_ramsey/transfer/report.json`    | Per-target transfer results.              |
| `results/psi_ramsey/transfer/summary.csv`    | Transfer summary table.                   |
| `results/psi_ramsey/transfer/aggregate.json` | Aggregated transfer metrics by method.    |
| `results/psi_ramsey/transfer/ranking.json`   | Overall and per-metric transfer rankings. |
| `results/psi_ramsey/matched_compute/*`       | Matched-budget evaluation outputs.        |
| `results/psi_ramsey/search/*`                | Search-suite outputs.                     |
| `results/psi_ramsey/ablations/*`             | Ablation outputs.                         |
| `results/psi_ramsey/interpretability/*`      | Interpretability/frontier outputs.        |
| `results/psi_ramsey/all_suites_summary.json` | Cross-suite summary.                      |
| `results/psi_ramsey/overview.md`             | Human-readable overview of main results.  |

### Supplementary results

Supplementary runs live under [`results/psi_ramsey_supplementary`](results/psi_ramsey_supplementary):

| Path                                                         | Content                             |
| ------------------------------------------------------------ | ----------------------------------- |
| `results/psi_ramsey_supplementary/transfer_seed_robustness/*` | Transfer seed robustness outputs.   |
| `results/psi_ramsey_supplementary/search_seed_robustness/*`  | Search seed robustness outputs.     |
| `results/psi_ramsey_supplementary/transfer_compute_low_budget/*` | Low-budget transfer stress test.    |
| `results/psi_ramsey_supplementary/transfer_compute_high_budget/*` | High-budget transfer stress test.   |
| `results/psi_ramsey_supplementary/mixed_r_transfer_neighborhood/*` | Mixed-`r` teacher-pool stress test. |
| `results/psi_ramsey_supplementary/high_resolution_structure/*` | High-resolution structure variant.  |
| `results/psi_ramsey_supplementary/compact_structure/*`       | Compact structure variant.          |
| `results/psi_ramsey_supplementary/exact_supervision_stress/*` | Stronger exact-supervision setting. |
| `results/psi_ramsey_supplementary/search_compute_high_budget/*` | High-budget search stress test.     |
| `results/psi_ramsey_supplementary/ablation_seed_robustness/*` | Ablation seed robustness.           |
| `results/psi_ramsey_supplementary/interpretability_seed_robustness/*` | Interpretability seed robustness.   |

---

## ▶️ Suggested Reproduction Flow

If you want the shortest full reproduction path, use:

```bash
python main.py --suite all
python generate_paper_figures.py
python supplementary_experiments.py --experiment all
python generate_supplementary_materials.py
```

If you only want to inspect the existing generated outputs, start with:

- `results/psi_ramsey/overview.md`
- `results/psi_ramsey/all_suites_summary.json`
- `results/psi_ramsey/paper_figures/paper_figures_manifest.json`
- `results/psi_ramsey_supplementary/materials/supplementary_materials_manifest.json`

---

## 🙏 Acknowledgments and Upstream Sources

This work was methodologically inspired in part by **Ψ-NN**, especially its teacher-student distillation and explicit structure-discovery workflow:

- Z. Liu et al., *Automatic Network Structure Discovery of Physics Informed Neural Networks via Knowledge Distillation*, *Nature Communications*, vol. 16, Art. no. 9558, 2025, doi: `10.1038/s41467-025-64624-3`
- Ψ-NN GitHub repository: <https://github.com/ZitiLiu/Psi-NN>

The Ramsey witness and lower-bound data used in this repository are based on publicly available Ramsey-number construction resources associated with recent computational Ramsey-search work, including:

- A. Nagda, P. Raghavan, and A. Thakurta, *Reinforced Generation of Combinatorial Structures: Ramsey Numbers*, 2026, doi: `10.48550/arXiv.2603.09172`
- Google Research Ramsey bounds repository: <https://github.com/google-research/google-research/tree/master/ramsey_number_bounds/improved_bounds>

This repository builds on those prior resources for methodological inspiration and/or input witness data, while introducing a separate transferable-structure framework, evaluation pipeline, and supplementary robustness study.

---

## 📚 How to Cite

If you use this repository, code structure, or generated experimental outputs, please cite the current pre-publication repository version:

```bibtex
@misc{jurj_transferable_structural_search_ramsey_2026,
  author       = {Sorin Liviu Jurj},
  title        = {From Cell-Specific Heuristics to Transferable Structural Search for Ramsey Graph Construction},
  year         = {2026},
  howpublished = {\url{https://github.com/jurjsorinliviu/Transferable-Structural-Search-for-Ramsey-Graph-Construction}},
  note         = {GitHub repository, pre-publication research prototype}
}
```

The citation will be updated once the manuscript is formally published.

---

## 📝 Notes

- This repository is a research prototype, not a packaged library.
- The experiment pipeline is file-based and writes results directly into `results/`.
- The root-level runtime dependencies for the project are listed in `requirements.txt`.
- A minimal Codespaces/devcontainer setup is provided in `.devcontainer/devcontainer.json`.
- The witness graphs currently come from the local `related work/ramsey_number_bounds/improved_bounds` directory.
- Smoke runs are available for quick validation, but they are not the paper results.
