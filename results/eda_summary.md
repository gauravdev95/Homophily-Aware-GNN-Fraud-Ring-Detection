# EDA Summary — Elliptic++ Transactions Dataset

*Phase 2 of the Homophily-Aware GNN for Fraud Ring Detection project · generated from
`notebooks/01_data_exploration.ipynb` (executed end-to-end) · all statistics computed, not estimated.*

## 1. Dataset inventory

| File | Shape | Content |
|---|---|---|
| `txs_features.csv` | 203,769 × 184 | `txId` + 183 features (~694.8 MB on disk) |
| `txs_classes.csv` | 203,769 × 2 | `txId`, `class` (1=illicit, 2=licit, 3=unknown) |
| `txs_edgelist.csv` | 234,355 × 2 | directed money-flow edges `txId1 → txId2` |

Feature columns: `Time step` (int, 1–49) + 93 anonymized standardized local features
(`Local_feature_1..93`) + 72 anonymized aggregate features (`Aggregate_feature_1..72`)
+ 17 named statistical columns (`in_txs_degree`, `out_txs_degree`,
`total_BTC`, `fees`, `size`, `num_input_addresses`, `num_output_addresses`,
`in_BTC_min/max/mean/median/total`, `out_BTC_min/max/mean/median/total`).
Per official documentation the local features are anonymized — they are referenced by
name only, never interpreted.

## 2. Data quality — what was verified

- **ID consistency:** feature-ID set == class-ID set exactly (203,769 each);   duplicate `txId`: features=0, classes=0.
- **Graph integrity:** self-loops=0, duplicate edges=0,   dangling endpoints=0, isolated (zero-edge) nodes=0   (0.00%).
- **Missing values:** 16,405 cells (0.04% of all feature cells),   confined to the 17 named statistical columns — the same 965 rows are missing all 17   (a clean co-missing block). Missing-row classes:   illicit=0, licit=519, unknown=446   (vs overall 2.23%/20.62%/77.15%).
- **Temporal direction of edges:** 100.00% of edges go forward in time   (dst step ≥ src step), confirming the `txId1 → txId2` money-flow direction empirically.

## 3. Labels and class imbalance

| Class | Count | Share |
|---|---|---|
| illicit (1) | 4,545 | 2.23% |
| licit (2) | 42,019 | 20.62% |
| unknown (3) | 157,205 | 77.15% |

Licit:illicit ≈ 9.25:1; only 22.85% of
transactions are labeled. Temporal label-prior drift is negligible (+0.026); per-step illicit rates and
volumes are plotted in `eda_labels_over_time.png` (all 49 steps non-empty).

## 4. Graph structure

- Degrees are heavy-tailed: in-degree mean 1.15, max 284;   out-degree mean 1.15, max 472;   27.15% of nodes have in-degree 0 (source-only),   18.37% out-degree 0 (sink-only).
- 49 weakly connected components; the largest holds 7,880 nodes   (3.87% of all transactions); 0 singleton components.
- Temporal structure is exact, not approximate: **0 cross-step edges** out of 234,355   (every edge is intra-step), and the 49 components map 1:1 onto the 49 time   steps (verified: no component spans more than one step). Message passing therefore never   crosses time steps, and temporal train/test splits are graph-clean by construction.

## 5. Feature analysis highlights

- Standardization check: the 93 local features are centered near 0 with std near 1
  (`eda_feature_group_scales.png`); the 72 anonymized aggregates and 17 named stats are on
  heterogeneous raw scales (see the group table in the notebook).
- Top discriminative features (illicit vs licit, |Cohen's d|):

| Feature | Cohen's d |
|---|---|
| Local_feature_53 | -1.162 |
| Local_feature_55 | -0.997 |
| Local_feature_89 | -0.990 |
| Local_feature_90 | -0.987 |
| Local_feature_91 | -0.758 |
| Local_feature_52 | -0.749 |

- Correlation structure (stratified sample, n≈20.5k): mean |r| within and across the
  local / anonymized-aggregate / named-stat blocks plus the top-5 correlated pairs are
  reported in the notebook (`eda_correlation.png`).
- Feature↔label association ceiling is strong (0.517) — a single feature carries substantial label signal.

## 6. Preliminary graph homophily

Metric (directed): labeled edge homophily
$h_{lab}$ = fraction of both-ends-labeled edges connecting same-class endpoints;
$h_{all}$ = same with *unknown* as a third class; per-class $P(dst=c\mid src=c)$;
random-mixing baseline $h_{base} = \sum_c p_c^2$ from labeled endpoint proportions.

| Metric | Value |
|---|---|
| $h_{lab}$ (observed) | 0.9537 |
| random-mixing baseline | 0.9043 |
| $h_{all}$ (3-class, unknown-dominated) | 0.7113 |
| P(dst=illicit \| src=illicit) | 0.2961 |
| P(dst=licit \| src=illicit) | 0.2714 |
| P(dst=licit \| src=licit) | 0.6810 |

Interpretation: homophily in the labeled subgraph is close to the random-mixing baseline (weak homophily signal). The aggregate figure is
dominated by the licit majority, so per-class conditioning is the informative view:

| src class | P(dst = same class \| src) | random-mixing rate | enrichment |
|---|---|---|---|
| illicit | 0.2961 | 0.0504 | 5.9x |
| licit | 0.6810 | 0.9496 | 0.72x |

Illicit transactions link to other illicit transactions well above the random-mixing rate,
while licit transactions mix more broadly than chance, transacting with unknown/illicit
counterparties. The full directed edge-type matrix is in `eda_homophily.png`. No claims
about fraud rings or camouflage are made — this is a descriptive statistic, not evidence
of organized structure.

## 7. Limitations

1. The 17 named statistical columns' and 72 anonymized aggregates' construction (neighborhood
   definition, temporal masking) cannot
   be verified from the CSVs — see `leakage_analysis.md`.
2. 77% of nodes are unlabeled; homophily/label statistics on the labeled subgraph need not
   generalize to unknown nodes.
3. Local feature semantics are anonymized by the dataset authors; distributional differences
   cannot be mapped to domain meaning.
4. Correlation/feature-importance figures use a stratified sample (~20.5k rows), not the full
   203k rows; global moments (means/stds) are exact chunked computations.
5. 965 rows lack all aggregate features; any model using them must define
   an explicit missingness policy.

## 8. Recommendations for Phase 3 (graph construction & temporal splitting)

1. **Temporal split, not random:** the label prior is essentially stable over time (negligible (+0.026));
   use
   forward-chaining splits (e.g. train on steps 1–34, validate 35–41, test 42–49, as in the
   original Elliptic protocol family) so evaluation measures generalization to future
   transactions.
2. **Missingness policy:** decide before modeling — options are (a) drop the 965 rows,
   (b) zero/mean-impute with a missingness indicator, or (c) exclude the named-stat columns.
   Document the choice; the missing rows are disproportionately licit/unknown.
3. **Leakage audit of aggregates:** cross-step leakage via neighborhoods is structurally
   ruled out (0 cross-step edges), but the construction semantics of the 72 anonymized
   aggregates and 17 named statistics are unverified from the CSVs — check the dataset
   paper/construction code for the exact neighborhood definition, and run Phase-3
   ablations with/without aggregates regardless.
4. **Class imbalance:** keep all unknowns (semi-supervised setting); do not oversample.
   Plan class-weighted losses and PR-AUC-style metrics rather than accuracy.
5. **Graph construction:** directed money-flow graph (validated: exactly 0 cross-step edges
   out of 234,355); the 49 components map 1:1 onto the 49 time steps, so temporal
   train/test splits are graph-clean by construction. Every node has at least one edge
   (0 isolated nodes, 0 singleton components), so no special isolate handling is needed
   in message passing.
6. **Class-conditional structure should be modeled:** illicit→illicit linkage is
   5.9x the random-mixing rate while licit edges mix more broadly
   than chance; a GNN should therefore model class-conditional/heterophilous message
   passing rather than assuming uniform homophily, and be evaluated on whether it beats
   the structural baseline (0.9537), especially on cross-class edges.

## Figures (`results/figures/eda/`)

`eda_class_distribution.png`, `eda_labels_over_time.png`, `eda_missing_values.png`,
`eda_feature_group_scales.png`, `eda_degree_distributions.png`, `eda_connected_components.png`,
`eda_feature_distributions.png`, `eda_correlation.png`, `eda_homophily.png`
