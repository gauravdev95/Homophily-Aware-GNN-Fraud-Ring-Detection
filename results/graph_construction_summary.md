# Graph Construction Summary — Elliptic++ Transactions

*Phase 3 / Step 3 · directed money-flow graph + temporal splits · all statistics computed, not estimated.*

## 1. Graph

| Metric | Value |
|---|---|
| Nodes (transactions) | 203,769 |
| Directed edges (money flow) | 234,355 |
| Density | 5.644e-06 |
| Weakly connected components | 49 |
| Mean / max in-degree | 1.15 / 284 |
| Mean / max out-degree | 1.15 / 472 |

## 2. Validation

| Check | Result | Detail |
|---|---|---|
| no_cross_step_edges | PASS | 0 cross-step edges |
| components_match_timesteps | PASS | 49 components, 1:1 with 49 timesteps: True |
| no_self_loops | PASS | 0 self-loops |
| no_isolated_nodes | PASS | 0 isolated nodes |

## 3. Temporal splits (no random splitting)

| Split | Timesteps | Nodes | Edges | Illicit | Licit | Unknown |
|---|---|---|---|---|---|---|
| train | 1-34 | 136,265 | 156,843 | 3,462 | 26,432 | 106,371 |
| val | 35-41 | 30,680 | 35,360 | 675 | 7,154 | 22,851 |
| test | 42-49 | 36,824 | 42,152 | 408 | 8,433 | 27,983 |

All edges are intra-timestep (0 cross-step edges verified), so no edge crosses a split boundary — splits are graph-clean by construction.

## 4. Artifacts

- `data/processed/node_splits.csv` — txId, timestep, split, class
- `data/processed/graph_splits.json` — split definitions + statistics
- `results/graph_stats.json`, `results/graph_construction_summary.md`
- Figures: `results/figures/graph/`

## 5. Figures

- `graph_split_sizes.png`
- `graph_labels_per_split.png`
- `graph_nodes_per_timestep.png`
