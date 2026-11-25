# Homophily Analysis Summary — Elliptic++ Transactions

*Phase 3 / Step 4 · descriptive statistics only — no fraud-ring or camouflage claims; no model training.*

## 1. Edge-type homophily (directed)

- Labelled-edge homophily h_lab = **0.9537** vs random-mixing baseline **0.9043** (confirmed: identical to Phase 2 EDA).
- Illicit -> illicit linkage is **5.9x** the random-mixing rate; licit -> licit is **0.72x** (below chance).
- P(dst=illicit | src=illicit) = 0.2961; P(dst=licit | src=licit) = 0.6810.
- Reading: the aggregate h_lab is dominated by the licit majority; the informative view is per-class conditioning — illicit nodes link to illicit nodes well above chance, licit nodes mix more broadly.

## 2. Node-level homophily

- **illicit**: mean same-label neighbour fraction 0.510 (median 0.500); coverage 54.7% of nodes have >=1 labelled neighbour; fraction with zero same-label neighbours 0.451.
- **licit**: mean same-label neighbour fraction 0.982 (median 1.000); coverage 79.5% of nodes have >=1 labelled neighbour; fraction with zero same-label neighbours 0.011.

## 3. Homophily by timestep

- Per-step h_lab ranges 0.755–0.999 (mean 0.939); slope -0.00001 per step (corr -0.001) — no strong drift.
- Illicit rate among labelled nodes ranges 0.0028–0.3597; slope -0.000438 per step (corr -0.066).

## 4. Feature homophily

- Mean |standardised local-feature difference|: real edges 0.4725 vs random pairs 0.4839 (ratio 0.98). Neighbours are more similar than random pairs, but the difference is modest — consistent with weak overall homophily, not with a strong feature-based cluster structure.

## 5. Structural baseline (transductive upper bound)

- Majority-neighbour prediction on labelled test nodes (n=8,841, coverage 73.1%): PR-AUC 0.3932, ROC-AUC 0.7256, F1@0.5 0.5158 (covered nodes only).
- This peeks at neighbours' true labels, so it is an upper-bound reference only — NOT a fair model and NOT comparable to the Step 5 baselines, which never see test labels.

## 6. What this means for the GNN (Step 6)

- A plain homophily-assuming GNN (GCN-style uniform averaging) is a weak prior here: aggregate homophily is close to the random-mixing baseline and licit nodes mix broadly. The class-conditional signal (illicit 5.9x enrichment) argues for heterophily-aware / class-conditional message passing, and for evaluating on whether the model beats the structural baseline above — especially on cross-class edges.

## 7. Figures

- `homophily_edge_type_heatmap.png`
- `homophily_enrichment_bars.png`
- `homophily_by_timestep.png`
- `homophily_node_histograms.png`
- `homophily_feature_similarity.png`
