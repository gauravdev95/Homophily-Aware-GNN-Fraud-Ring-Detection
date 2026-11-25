# GNN Experiments Summary — Elliptic++ Transactions

*Phase 3 / Step 6 · temporal splits (train 1–34, val 35–41, test 42–49) · supervised on labelled nodes only (unknowns are graph context, never targets) · no SMOTE, no oversampling · class-weighted loss only · accuracy never headlined · transductive message passing (all node features/edges visible; loss on train labels only).*

## 1. Setup

- Models: GCN, GAT (4 heads), GraphSAGE (mean), H2GCN-style (ego separate, 1-hop + 2-hop channels), GatedGCN (learned per-node ego/neighbour gate).
- Features: all 183 + missingness indicator, median imputation fit on TRAIN only, StandardScaler fit on train only — the identical pipeline to Step 5 (MLP baseline).
- Aggregation over INCOMING neighbours (money senders), respecting the directed money-flow direction.
- Training: full-batch, Adam lr 0.01, weight decay 5e-4, dropout 0.5, hidden 64, 2 layers, up to 200 epochs, early stopping on val PR-AUC (patience 20).
- Model selection on VAL using PR-AUC; final numbers on TEST.

## 2. Results (PR-AUC primary)

| Model | Val PR-AUC | Test PR-AUC | Test ROC-AUC | Test F1@0.5 | Test precision | Test recall | Fit (s) | Best epoch |
|---|---|---|---|---|---|---|---|---|
| sage | 0.7996 | 0.4746 | 0.8689 | 0.4631 | 0.4731 | 0.4534 | 303.6 | 140 |
| h2gcn | 0.7824 | 0.4102 | 0.8746 | 0.4235 | 0.3866 | 0.4681 | 953.1 | 86 |
| gated_gcn | 0.6881 | 0.3449 | 0.8656 | 0.3420 | 0.2599 | 0.5000 | 210.5 | 22 |
| gcn | 0.5944 | 0.2577 | 0.7943 | 0.3203 | 0.2547 | 0.4314 | 171.7 | 130 |
| gat | 0.4708 | 0.2502 | 0.8212 | 0.1724 | 0.0957 | 0.8701 | 133.4 | 36 |

**Best model (by val PR-AUC): sage** (val PR-AUC 0.7996, test PR-AUC 0.4746).

**XGBoost bar (Step 5): test PR-AUC 0.5652. Best GNN does NOT beat it on test PR-AUC (-0.0906).**

## 3. Confusion matrices (test, threshold 0.5)

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| sage | 8,227 | 206 | 223 | 185 |
| h2gcn | 8,130 | 303 | 217 | 191 |
| gated_gcn | 7,852 | 581 | 204 | 204 |
| gcn | 7,918 | 515 | 232 | 176 |
| gat | 5,078 | 3,355 | 53 | 355 |

## 4. Fraud-ring evaluation (descriptive, test labelled nodes)

Best model (sage) on test: 391 predicted-illicit nodes vs 408 true illicit (precision 0.4731, recall 0.4534).

- Predicted-illicit subgraph: 330 weak components, largest 6 nodes, 290 singletons. True-illicit subgraph: 332 components, largest 12 nodes, 298 singletons.
- Predicted illicit→illicit edge rate: 0.2877 (true-illicit test rate 0.3838; observed overall rate 0.2961).

These are descriptive statistics of the predicted subgraph, not proof of organised fraud rings — the homophily statistics describe linkage patterns only.

## 5. Explainability

### 5.1 GAT attention by edge type

| Edge type | n edges | Mean attention | Median attention |
|---|---|---|---|
| illicit->illicit | 998 | 0.4505 | 0.4965 |
| illicit->licit | 915 | 0.0653 | 0.0343 |
| licit->illicit | 781 | 0.2659 | 0.4328 |
| licit->licit | 33,930 | 0.2572 | 0.2241 |
| unknown-involved | 197,731 | 0.3412 | 0.4722 |

### 5.2 Feature-group ablation (best model)

| Feature set | n features | Val PR-AUC | Test PR-AUC |
|---|---|---|---|
| local_only | 95 | 0.7291 | 0.4143 |
| local_plus_aggregate | 167 | 0.7840 | 0.4143 |
| all features | 184 | 0.7996 | 0.4746 |

## 6. Figures

- `gnn_pr_auc_bars.png`
- `gnn_pr_curves.png`
- `gnn_val_test_drop.png`
- `gnn_training_curves.png`
- `gnn_fraud_ring.png`
- `gnn_gat_attention.png`
- `gnn_ablation.png`

## 7. Reading the results

- **Temporal gap.** Every model drops from val to test — future transactions are genuinely harder, exactly what the temporal split is designed to expose.
- **ROC-AUC flatters on imbalanced data** (2.23% illicit); PR-AUC is the primary metric.
- **Homophily-aware variants** (H2GCN-style, GatedGCN) were motivated by Step 4: illicit nodes link to illicit nodes at 5.9x chance, but node homophily is bimodal (45% of illicit nodes have zero same-label neighbours), so uniform smoothing hurts — ego separation and per-node gating address exactly that.
