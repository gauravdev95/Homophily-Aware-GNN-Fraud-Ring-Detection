# Homophily-Aware GNN for Financial Fraud Ring Detection

A research project on detecting **fraud rings** in Bitcoin transaction networks using
**homophily-aware Graph Neural Networks**, built on the
[Elliptic++ dataset](https://github.com/git-disl/EllipticPlusPlus) (Elmougy & Liu, KDD 2023).

## Motivation

Financial fraud rarely happens in isolation — illicit actors transact with each other,
forming dense **fraud rings**. In transaction graphs this shows up as *homophily*:
nodes of the same class (illicit / licit) tend to be connected. Standard GNNs assume
homophily implicitly, but fraud graphs are only *partially* homophilic, and blindly
smoothing over heterophilic edges hurts detection. This project:

1. Quantifies homophily in the Elliptic++ transaction graph (node/edge homophily, class-wise assortativity, per-timestep drift).
2. Benchmarks standard baselines (tabular ML + vanilla GNNs).
3. Designs and evaluates **homophily-aware GNN variants** that adapt message passing to local homophily structure.

> **Current status:** repository + dataset setup complete. No models built or trained yet.

## Quick start

One command sets up everything — virtualenv, dependencies, dataset download,
verification:

```bash
python start.py
```

`start.py` (stdlib only, works on Windows/macOS/Linux) will:

1. Create `.venv/` and install `requirements.txt`
2. Download any missing dataset files into `data/raw/elliptic_pp/`
   (`txs_features.csv` comes from the
   [data-v1 release](https://github.com/gauravdev95/Homophily-Aware-GNN-Fraud-Ring-Detection/releases/tag/data-v1),
   Google Drive is the automatic fallback)
3. Run `scripts/verify_data.py` to sanity-check the dataset

Options: `--no-download` (fail instead of downloading), `--no-verify`,
`--reinstall` (force-reinstall deps).

Full step-by-step instructions, manual `pip install` path, and troubleshooting:
**[SETUP.md](SETUP.md)**.

## Dataset

**Elliptic++ Transactions Dataset** — 203,769 Bitcoin transactions, 234,355 money-flow edges,
49 timesteps, 183 features per transaction.

| File | Description |
|---|---|
| `data/raw/elliptic_pp/txs_features.csv` | 183 features per transaction |
| `data/raw/elliptic_pp/txs_classes.csv` | Labels: `1` = illicit, `2` = licit, `3` = unknown |
| `data/raw/elliptic_pp/txs_edgelist.csv` | Transaction → transaction money-flow edges |

The two small CSVs (`txs_classes.csv`, `txs_edgelist.csv`) are committed in
`data/raw/elliptic_pp/`; the 663 MB `txs_features.csv` is fetched from the
[data-v1 release](https://github.com/gauravdev95/Homophily-Aware-GNN-Fraud-Ring-Detection/releases/tag/data-v1)
(see `data/README.md` for details and official download sources).
The Actors/Wallet dataset is intentionally out of scope for now.

## Project structure

```
├── configs/               # Experiment configs (YAML)
├── data/
│   ├── raw/elliptic_pp/   # Official Elliptic++ txs CSVs (gitignored, see data/README.md)
│   └── processed/         # Derived graph/processed artifacts (gitignored)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_graph_construction.ipynb
│   ├── 03_homophily_analysis.ipynb
│   ├── 04_baseline_models.ipynb
│   └── 05_gnn_experiments.ipynb
├── src/
│   ├── data/              # Dataset loading, graph building
│   ├── models/            # GNN architectures (incl. homophily-aware variants)
│   ├── training/          # Training loops, losses, sampling
│   ├── evaluation/        # Metrics, fraud-ring evaluation
│   └── explainability/    # GNN explainability (GNNExplainer, attention, SHAP)
├── scripts/               # CLI entry points (train, evaluate, download data)
├── models/                # Saved model weights (gitignored)
├── results/
│   ├── figures/           # Plots
│   ├── metrics/           # Metric JSONs/CSVs
│   └── logs/              # Training logs (gitignored)
└── tests/                 # Unit tests
```

## Setup

```bash
# 1. Clone
git clone https://github.com/gauravdev95/Homophily-Aware-GNN-Fraud-Ring-Detection.git
cd Homophily-Aware-GNN-Fraud-Ring-Detection

# 2. Python environment (3.10+)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # (to be added with the training stack)

# 3. Download the dataset (see data/README.md for details)
python scripts/download_data.py    # (to be added)
# or manually place txs_*.csv into data/raw/elliptic_pp/
```

## Roadmap

- [x] Repository scaffolding + Elliptic++ transactions dataset setup & verification
- [ ] `01` Data exploration (distributions, missing values, temporal structure)
- [ ] `02` Graph construction (tx-tx money-flow graph, temporal splits)
- [ ] `03` Homophily analysis (edge/node homophily, illicit-subgraph density)
- [ ] `04` Baseline models (LogReg / RandomForest / XGBoost / MLP)
- [ ] `05` GNN experiments (GCN, GAT, GraphSAGE → homophily-aware variants)
- [ ] Fraud-ring level evaluation + explainability

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
