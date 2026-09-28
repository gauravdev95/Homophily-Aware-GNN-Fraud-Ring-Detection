# Homophily-Aware GNN for Financial Fraud Ring Detection

Detecting **fraud rings** in Bitcoin transaction networks using **homophily-aware
Graph Neural Networks**, built on the
[Elliptic++ dataset](https://github.com/git-disl/EllipticPlusPlus) (Elmougy & Liu, KDD 2023).

## Motivation

Financial fraud rarely happens in isolation — illicit actors transact with each other,
forming dense **fraud rings**. In transaction graphs this shows up as *homophily*:
nodes of the same class (illicit / licit) tend to be connected. Standard GNNs assume
homophily implicitly, but fraud graphs are only *partially* homophilic, and blindly
smoothing over heterophilic edges hurts detection. This project:

1. Quantifies homophily in the Elliptic++ transaction graph (node/edge homophily,
   class-wise assortativity, per-timestep drift).
2. Benchmarks standard baselines (tabular ML + vanilla GNNs).
3. Designs and evaluates **homophily-aware GNN variants** that adapt message passing
   to local homophily structure.

---

## Project status — where we are

| Phase | Status | What was done |
|---|---|---|
| **Phase 1 — Setup** | ✅ Complete | Repo scaffolding, Elliptic++ dataset downloaded & verified (`scripts/verify_data.py` → PASS), one-command setup (`python start.py`), `requirements.txt`, `SETUP.md` guide |
| **Phase 2 — Exploratory Data Analysis** | ✅ Complete | Executed notebook `notebooks/01_data_exploration.ipynb` (32 cells, 0 errors), 9 figures in `results/figures/eda/`, findings in `results/eda_summary.md`, leakage audit in `results/leakage_analysis.md` |
| **Phase 3 — Graph construction & modeling** | ⬜ Not started | Baselines → homophily-aware GNNs → fraud-ring evaluation (see Roadmap) |

**Headline findings from Phase 2** (all computed, none estimated):

- **Illicit→illicit linkage is 5.9× the random-mixing rate** — the core homophily
  signal this project is built to exploit. (Licit edges, by contrast, mix more
  broadly than chance at 0.72×.)
- **The graph is temporally clean**: exactly **0 cross-step edges** out of 234,355 —
  the 49 weakly connected components map 1:1 onto the 49 time steps, so temporal
  train/test splits are graph-clean by construction and message passing can never
  leak across time.
- **No label leakage**: no feature determines the label (max |correlation| with the
  illicit indicator is 0.267); 0 zero-variance features.
- **Top discriminative features** (illicit vs licit, |Cohen's d|): `Local_feature_53`
  (−1.162), `Local_feature_55` (−0.997), `Local_feature_89` (−0.990),
  `Local_feature_90` (−0.987).

Full detail: [`results/eda_summary.md`](results/eda_summary.md) ·
[`results/leakage_analysis.md`](results/leakage_analysis.md)

---

## Quick start

One command sets up everything — virtualenv, dependencies, dataset download,
verification:

```bash
git clone https://github.com/gauravdev95/Homophily-Aware-GNN-Fraud-Ring-Detection.git
cd Homophily-Aware-GNN-Fraud-Ring-Detection
python start.py
```

`start.py` (stdlib only, Windows/macOS/Linux) creates `.venv/`, installs
`requirements.txt`, downloads any missing dataset files into `data/raw/elliptic_pp/`,
and runs `scripts/verify_data.py`. Full guide, manual `pip install` path, and
troubleshooting: **[SETUP.md](SETUP.md)**.

---

## Dataset — in depth

**Source:** Elliptic++ Transactions (Elmougy & Liu, KDD 2023) —
[official repo](https://github.com/git-disl/EllipticPlusPlus) ·
[paper (arXiv:2306.06108)](https://arxiv.org/abs/2306.06108).
The Actors/Wallet dataset is intentionally out of scope for this project.

| File | Shape | Content | Location |
|---|---|---|---|
| `txs_features.csv` | 203,769 × 184 | `txId` + 183 features (~694.8 MB) | [data-v1 release](https://github.com/gauravdev95/Homophily-Aware-GNN-Fraud-Ring-Detection/releases/tag/data-v1) ([direct download](https://github.com/gauravdev95/Homophily-Aware-GNN-Fraud-Ring-Detection/releases/download/data-v1/txs_features.csv), no login needed) |
| `txs_classes.csv` | 203,769 × 2 | `txId`, `class` | committed in `data/raw/elliptic_pp/` |
| `txs_edgelist.csv` | 234,355 × 2 | directed money-flow edges `txId1 → txId2` | committed in `data/raw/elliptic_pp/` |

(`txs_features.csv` exceeds GitHub's 100 MB/file limit, so it lives on the release;
the two small CSVs are versioned in git. `python start.py` fetches everything
automatically.)

### Feature taxonomy (183 features)

| Group | Count | Notes |
|---|---|---|
| `Time step` | 1 | Integer 1–49; every edge stays inside one step |
| `Local_feature_1` … `Local_feature_93` | 93 | Anonymized, standardized (≈ mean 0, std 1). Referenced by name only — meanings not interpreted, per official docs |
| `Aggregate_feature_1` … `Aggregate_feature_72` | 72 | Anonymized neighborhood aggregates, heterogeneous scales |
| Named statistics | 17 | `in/out_txs_degree`, `total_BTC`, `fees`, `size`, `num_input/output_addresses`, `in/out_BTC_min/max/mean/median/total` |

### Labels

| Class | Meaning | Count | Share |
|---|---|---|---|
| `1` | illicit | 4,545 | 2.23% |
| `2` | licit | 42,019 | 20.62% |
| `3` | unknown | 157,205 | 77.15% |

Only 22.85% of transactions are labeled (licit:illicit ≈ 9.25:1). Unknowns are
**kept** — the modeling setup is semi-supervised, not filtered. Label prior drift
over the 49 steps is negligible (+0.026).

### Graph structure

- Directed money-flow edges; 100% go forward in time (dst step ≥ src step).
- Heavy-tailed degrees: in-degree max 284, out-degree max 472 (mean 1.15).
- 49 weakly connected components ⇔ 49 time steps (exact 1:1 mapping, verified).
- 0 self-loops, 0 duplicate edges, 0 dangling endpoints, 0 isolated nodes.

---

## Data quality & cleaning

### What was verified (Phase 1 + 2)

- **ID consistency:** feature-ID set == class-ID set exactly (203,769 each); 0 duplicate `txId`s.
- **Graph integrity:** 0 self-loops, 0 duplicate edges, 0 dangling endpoints, 0 isolated nodes.
- **No leakage:** no deterministic feature→label mapping; max |r| with illicit = 0.267
  (`Aggregate_feature_57`), with licit = 0.517; 0 zero-variance
  features; cross-time leakage structurally impossible (0 cross-step edges).
- **Near-duplicates:** 10 features have a partner with |r| > 0.999 — drop one of each
  pair before modeling.

### Missing values — what exists and what to do

**What exists:** 16,405 missing cells (0.04% of all feature cells), confined to the
17 named statistical columns — the **same 965 rows are missing all 17** (a clean
co-missing block). Their labels: 0 illicit, 519 licit, 446 unknown — i.e.
missingness is disproportionately licit/unknown, never illicit.

**What we did in Phase 2:** nothing destructive — no rows dropped, no imputation,
unknowns kept. EDA is descriptive; the missing block is reported as-is
(`eda_missing_values.png`, `results/eda_summary.md` §2).

**What to do when nulls appear (modeling policy for Phase 3):** pick one option
*before* training and document it:

| Option | How | Trade-off |
|---|---|---|
| **(a) Drop the 965 rows** | simplest | Loses 0.47% of data; slightly shifts the label mix (missing rows are 53.8% licit vs 20.6% overall) |
| **(b) Impute + missingness indicator** ✅ recommended | median/mean-impute the 17 columns **and** add a binary `is_missing_stats` column | Keeps all rows; lets the model learn that missingness itself is informative (it correlates with licit) |
| **(c) Exclude the 17 named-stat columns** | drop the columns | Sidesteps the issue but discards the most interpretable features |

Rules that always apply:

- **Fit imputers/scalers on the training split only**, then apply to validation/test —
  computing them over all data leaks test statistics into training.
- **Never use the label** when imputing features.
- Keep the raw CSVs untouched — all cleaning happens in code on loaded copies
  (`scripts/verify_data.py` is strictly read-only).

---

## Project structure

```
├── start.py                 # One-command setup (venv, deps, data, verify)
├── requirements.txt         # pip install -r requirements.txt
├── SETUP.md                 # Full setup guide + troubleshooting
├── configs/                 # Experiment configs (YAML)
├── data/
│   ├── README.md            # Dataset sources & download instructions
│   ├── raw/elliptic_pp/     # txs CSVs (small ones committed; features via release)
│   └── processed/           # Derived artifacts (gitignored)
├── notebooks/
│   ├── 01_data_exploration.ipynb   # ✅ Phase 2, executed
│   ├── 02_graph_construction.ipynb # ⬜ Phase 3
│   ├── 03_homophily_analysis.ipynb # ⬜ Phase 3
│   ├── 04_baseline_models.ipynb    # ⬜ Phase 3
│   └── 05_gnn_experiments.ipynb    # ⬜ Phase 3
├── src/
│   ├── data/                # Dataset loading, graph building
│   ├── models/              # GNN architectures (incl. homophily-aware variants)
│   ├── training/            # Training loops, losses, sampling
│   ├── evaluation/          # Metrics, fraud-ring evaluation
│   └── explainability/      # GNNExplainer, attention, SHAP
├── scripts/
│   ├── verify_data.py       # Read-only dataset sanity checks
│   ├── build_eda_notebook.py / run_notebook.py / gen_eda_reports.py
├── results/
│   ├── eda_summary.md       # Phase 2 findings (computed statistics)
│   ├── leakage_analysis.md  # Leakage audit
│   └── figures/eda/         # 9 EDA plots
├── models/                  # Saved weights (gitignored)
└── tests/                   # Unit tests
```

---

## Roadmap

- [x] Repository scaffolding + Elliptic++ dataset setup & verification
- [x] `01` Data exploration — distributions, missing values, temporal structure,
      feature analysis, preliminary homophily (see `results/eda_summary.md`)
- [ ] `02` Graph construction — directed tx-tx graph; **temporal** splits
      (train steps 1–34, val 35–41, test 42–49), never random
- [ ] `03` Homophily analysis — edge/node homophily, class-wise assortativity,
      per-timestep drift, structural baseline (0.9537) to beat
- [ ] `04` Baseline models — LogReg / RandomForest / XGBoost / MLP
      (class-weighted losses, PR-AUC — not accuracy)
- [ ] `05` GNN experiments — GCN, GAT, GraphSAGE → **homophily-aware variants**
      (class-conditional / heterophily-aware message passing)
- [ ] Fraud-ring level evaluation + explainability; ablation with/without
      aggregate features (construction semantics unverified — see
      `results/leakage_analysis.md`)

---

## Citation

If you use the Elliptic++ dataset, please cite:

```bibtex
@inproceedings{elmougy2023demystifying,
  title={Demystifying Fraudulent Transactions and Illicit Nodes in the Bitcoin Network for Financial Forensics},
  author={Elmougy, Youssef and Liu, Ling},
  booktitle={Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '23)},
  year={2023}
}
```

## License

Code in this repository is released under the MIT License (see `LICENSE` when added).
The Elliptic++ dataset is released by its authors (Georgia Tech) for research use —
see the [official repository](https://github.com/git-disl/EllipticPlusPlus) for terms.
