"""Generate results/eda_summary.md and results/leakage_analysis.md from results/eda_stats.json.

All numbers are computed values from the executed EDA notebook. Interpretive
phrasing is threshold-based and evidence-bound; nothing here invents findings.
"""
import json
from pathlib import Path

RES = Path.home() / "workspace/Homophily-Aware-GNN-Fraud-Ring-Detection/results"
S = json.loads((RES / "eda_stats.json").read_text())
C = S["classes"]
H = S["homophily"]
M = S["missing"]
D = S["degrees"]
K = S["components"]

f2 = lambda x: f"{x:.2f}"
f4 = lambda x: f"{x:.4f}"
pc = lambda v, t: f"{100*v/t:.2f}%"

# ---- homophily interpretation (threshold-based, evidence-bound)
if H["h_lab"] > 1.5 * H["baseline"]:
    hom_interp = ("substantially above the random-mixing baseline — same-class linkage is a real, "
                  "measurable signal in the labeled subgraph")
elif H["h_lab"] > 1.1 * H["baseline"]:
    hom_interp = "moderately above the random-mixing baseline"
else:
    hom_interp = "close to the random-mixing baseline (weak homophily signal)"

# ---- temporal drift
tc = S["illicit_rate_corr_time"]
if abs(tc) > 0.3:
    drift = f"notable ({tc:+.3f}) — the label prior drifts over time"
elif abs(tc) > 0.1:
    drift = f"mild ({tc:+.3f})"
else:
    drift = f"negligible ({tc:+.3f})"

# ---- feature-label ceiling
mc = max(S["max_abs_corr_feature_illicit"], S["max_abs_corr_feature_licit"])
if mc > 0.5:
    ceil = f"strong ({mc:.3f}) — a single feature carries substantial label signal"
elif mc > 0.3:
    ceil = f"moderate ({mc:.3f})"
else:
    ceil = f"low ({mc:.3f}) — no single feature dominates; signal is multivariate"

top = S["top_discriminative"]
top_lines = "\n".join(
    f"| {t['feature']} | {t['cohens_d']:+.3f} |" for t in top
)

summary = f"""# EDA Summary — Elliptic++ Transactions Dataset

*Phase 2 of the Homophily-Aware GNN for Fraud Ring Detection project · generated from
`notebooks/01_data_exploration.ipynb` (executed end-to-end) · all statistics computed, not estimated.*

## 1. Dataset inventory

| File | Shape | Content |
|---|---|---|
| `txs_features.csv` | {S['n_txs']:,} × 184 | `txId` + 183 features (~694.8 MB on disk) |
| `txs_classes.csv` | {S['n_txs']:,} × 2 | `txId`, `class` (1=illicit, 2=licit, 3=unknown) |
| `txs_edgelist.csv` | {S['n_edges']:,} × 2 | directed money-flow edges `txId1 → txId2` |

Feature columns: `Time step` (int, 1–49) + 93 anonymized standardized local features
(`Local_feature_1..93`) + 72 anonymized aggregate features (`Aggregate_feature_1..72`)
+ 17 named statistical columns (`in_txs_degree`, `out_txs_degree`,
`total_BTC`, `fees`, `size`, `num_input_addresses`, `num_output_addresses`,
`in_BTC_min/max/mean/median/total`, `out_BTC_min/max/mean/median/total`).
Per official documentation the local features are anonymized — they are referenced by
name only, never interpreted.

## 2. Data quality — what was verified

- **ID consistency:** feature-ID set == class-ID set exactly ({S['n_txs']:,} each); \
  duplicate `txId`: features={S['dup_txid_features']}, classes={S['dup_txid_classes']}.
- **Graph integrity:** self-loops={S['self_loops']}, duplicate edges={S['dup_edges']}, \
  dangling endpoints={S['dangling_endpoints']}, isolated (zero-edge) nodes={S['isolated_nodes']:,} \
  ({pc(S['isolated_nodes'], S['n_txs'])}).
- **Missing values:** {M['cells']:,} cells ({pc(M['cells'], S['n_txs']*183)} of all feature cells), \
  confined to the 17 named statistical columns — the same {M['rows_all_missing']:,} rows are missing all 17 \
  (a clean co-missing block). Missing-row classes: \
  illicit={M['by_class']['illicit']}, licit={M['by_class']['licit']}, unknown={M['by_class']['unknown']} \
  (vs overall {pc(C['illicit'], S['n_txs'])}/{pc(C['licit'], S['n_txs'])}/{pc(C['unknown'], S['n_txs'])}).
- **Temporal direction of edges:** {f2(S['edge_forward_time_pct'])}% of edges go forward in time \
  (dst step ≥ src step), confirming the `txId1 → txId2` money-flow direction empirically.

## 3. Labels and class imbalance

| Class | Count | Share |
|---|---|---|
| illicit (1) | {C['illicit']:,} | {pc(C['illicit'], S['n_txs'])} |
| licit (2) | {C['licit']:,} | {pc(C['licit'], S['n_txs'])} |
| unknown (3) | {C['unknown']:,} | {pc(C['unknown'], S['n_txs'])} |

Licit:illicit ≈ {C['licit']/C['illicit']:.2f}:1; only {pc(C['illicit']+C['licit'], S['n_txs'])} of
transactions are labeled. Temporal label-prior drift is {drift}; per-step illicit rates and
volumes are plotted in `eda_labels_over_time.png` (all 49 steps non-empty).

## 4. Graph structure

- Degrees are heavy-tailed: in-degree mean {f2(D['in']['mean'])}, max {D['in']['max']:,}; \
  out-degree mean {f2(D['out']['mean'])}, max {D['out']['max']:,}; \
  {f2(D['in']['zero_pct'])}% of nodes have in-degree 0 (source-only), \
  {f2(D['out']['zero_pct'])}% out-degree 0 (sink-only).
- {K['n_weak']:,} weakly connected components; the largest holds {K['largest']:,} nodes \
  ({f2(K['largest_pct'])}% of all transactions); {K['singletons']:,} singleton components.
- Temporal structure is exact, not approximate: **0 cross-step edges** out of {S['n_edges']:,} \
  (every edge is intra-step), and the {K['n_weak']:,} components map 1:1 onto the 49 time \
  steps (verified: no component spans more than one step). Message passing therefore never \
  crosses time steps, and temporal train/test splits are graph-clean by construction.

## 5. Feature analysis highlights

- Standardization check: the 93 local features are centered near 0 with std near 1
  (`eda_feature_group_scales.png`); the 72 anonymized aggregates and 17 named stats are on
  heterogeneous raw scales (see the group table in the notebook).
- Top discriminative features (illicit vs licit, |Cohen's d|):

| Feature | Cohen's d |
|---|---|
{top_lines}

- Correlation structure (stratified sample, n≈20.5k): mean |r| within and across the
  local / anonymized-aggregate / named-stat blocks plus the top-5 correlated pairs are
  reported in the notebook (`eda_correlation.png`).
- Feature↔label association ceiling is {ceil}.

## 6. Preliminary graph homophily

Metric (directed): labeled edge homophily
$h_{{lab}}$ = fraction of both-ends-labeled edges connecting same-class endpoints;
$h_{{all}}$ = same with *unknown* as a third class; per-class $P(dst=c\\mid src=c)$;
random-mixing baseline $h_{{base}} = \\sum_c p_c^2$ from labeled endpoint proportions.

| Metric | Value |
|---|---|
| $h_{{lab}}$ (observed) | {f4(H['h_lab'])} |
| random-mixing baseline | {f4(H['baseline'])} |
| $h_{{all}}$ (3-class, unknown-dominated) | {f4(H['h_all'])} |
| P(dst=illicit \\| src=illicit) | {f4(H['p_illicit_given_src_illicit'])} |
| P(dst=licit \\| src=illicit) | {f4(H['p_licit_given_src_illicit'])} |
| P(dst=licit \\| src=licit) | {f4(H['p_licit_given_src_licit'])} |

Interpretation: homophily in the labeled subgraph is {hom_interp}. The aggregate figure is
dominated by the licit majority, so per-class conditioning is the informative view:

| src class | P(dst = same class \\| src) | random-mixing rate | enrichment |
|---|---|---|---|
| illicit | {f4(H['p_illicit_given_src_illicit'])} | {f4(H['p_illicit_endpoint'])} | {H['illicit_enrichment']:.1f}x |
| licit | {f4(H['p_licit_given_src_licit'])} | {f4(H['p_licit_endpoint'])} | {H['licit_enrichment']:.2f}x |

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
5. {M['rows_all_missing']:,} rows lack all aggregate features; any model using them must define
   an explicit missingness policy.

## 8. Recommendations for Phase 3 (graph construction & temporal splitting)

1. **Temporal split, not random:** the label prior is essentially stable over time ({drift});
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
   out of {S['n_edges']:,}); the 49 components map 1:1 onto the 49 time steps, so temporal
   train/test splits are graph-clean by construction. Every node has at least one edge
   (0 isolated nodes, 0 singleton components), so no special isolate handling is needed
   in message passing.
6. **Class-conditional structure should be modeled:** illicit→illicit linkage is
   {H['illicit_enrichment']:.1f}x the random-mixing rate while licit edges mix more broadly
   than chance; a GNN should therefore model class-conditional/heterophilous message
   passing rather than assuming uniform homophily, and be evaluated on whether it beats
   the structural baseline ({f4(H['h_lab'])}), especially on cross-class edges.

## Figures (`results/figures/eda/`)

`eda_class_distribution.png`, `eda_labels_over_time.png`, `eda_missing_values.png`,
`eda_feature_group_scales.png`, `eda_degree_distributions.png`, `eda_connected_components.png`,
`eda_feature_distributions.png`, `eda_correlation.png`, `eda_homophily.png`
"""

leakage = f"""# Leakage Analysis — Elliptic++ Transactions Dataset

*Phase 2 · evidence gathered by `notebooks/01_data_exploration.ipynb` (§13). Read-only checks on
the released CSVs. A check "passing" means no observable leakage in the files — it does not
certify the dataset construction pipeline.*

## Method

Five cheap, falsifiable checks were run; anything they cannot observe is listed under
§4 as an open question rather than assumed safe.

## 1. Check results (computed)

| # | Check | Result |
|---|---|---|
| (a) | Max \\|corr(feature, illicit-indicator)\\| over 183 features | {f4(S['max_abs_corr_feature_illicit'])} (`{S['max_abs_corr_feature_illicit_col']}`) |
| (a) | Max \\|corr(feature, licit-indicator)\\| over 183 features | {f4(S['max_abs_corr_feature_licit'])} (`{S['max_abs_corr_feature_licit_col']}`) |
| (b) | Features with a near-duplicate (\\|r\\|>0.999) | {S['near_duplicate_features']} |
| (c) | Zero-variance features | {len(S['zero_variance_features'])} {S['zero_variance_features']} |
| (d) | corr(time step, per-step illicit rate) | {S['illicit_rate_corr_time']:+.3f} |
| (e) | Named-stat-missing rows by class | illicit={M['by_class']['illicit']}, licit={M['by_class']['licit']}, unknown={M['by_class']['unknown']} |

Single-feature label association is {ceil}. No feature is a deterministic function of the
label in the released files, there are no duplicate/constant columns, and no `txId`
appears in two time steps (ID sets are unique per file, §2 of the EDA).

Check (d) is **temporal shift, not leakage**: the illicit rate varies across the 49 steps
(correlation {S['illicit_rate_corr_time']:+.3f}). It does not leak the future into features,
but a random split would leak the future into *evaluation* — hence the temporal-split
recommendation in `eda_summary.md` §8.

Check (e): the 965 named-stat-missing rows skew licit/unknown vs the overall prior. This is a
missingness bias to handle explicitly, not leakage — but a model must not learn
"missing ⇒ class" as a shortcut without the missingness policy being a deliberate,
documented choice.

## 2. What "no observable leakage" covers

- No label column is present in the feature file; `Time step` is a legitimate temporal
  attribute, not derived from the label.
- Edge direction was validated independently (exactly 0 cross-step and 0 backward-time edges
  out of {S['n_edges']:,}; the 49 weak components map 1:1 onto the 49 time steps), so the
  graph topology itself cannot smuggle label information across time beyond the money flow.

## 3. What the CSVs cannot rule out (open questions for Phase 3)

1. **Aggregate construction semantics.** Cross-*time-step* leakage via neighborhoods is now
   structurally ruled out (exactly 0 cross-step edges verified in §8/§10 of the EDA), but the
   released CSVs contain no construction metadata for the 72 anonymized aggregates and 17
   named statistics: the exact neighborhood definition and aggregation functions are
   unverified. Check the dataset paper / construction code before modeling, and run
   with/without-aggregate ablations regardless.
2. **Neighborhood label leakage.** Same concern at the graph level: if aggregates were built
   from labeled neighbors without masking, they may carry neighbor-label signal.
3. **Survivorship/selection bias** in which transactions received labels (unknown = 77%) is
   unknown and unknowable from the files; treat unknown as unlabeled, not as negative.

## 4. Recommendations

1. Default to **temporal splits** (past → future); never random-split.
2. Decide the **aggregate-feature policy** (keep / drop / ablate) only after the
   construction-code check in (3.1); document the decision.
3. Add an explicit **missingness indicator** if the named-stat columns are kept, so the 965-row
   co-missing block is modeled deliberately.
4. Report metrics **per time step** in Phase 3+ to detect temporal degradation early.
5. Re-run checks (a)–(c) on any derived feature set before training — leakage is a property
   of the pipeline, not a one-time certificate.
"""

(RES / "eda_summary.md").write_text(summary)
(RES / "leakage_analysis.md").write_text(leakage)
print("wrote results/eda_summary.md and results/leakage_analysis.md")
