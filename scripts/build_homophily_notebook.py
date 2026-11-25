"""Build notebooks/03_homophily_analysis.ipynb (Phase 3, Step 4).

Generates the notebook programmatically with real code cells only (no IPython
magics), so it can be executed headless via scripts/run_notebook.py.
Computes, from the real Elliptic++ data:
  1. directed 3x3 edge-type matrix + per-class homophily vs random-mixing
  2. node-level homophily distribution for labeled nodes, split by class
  3. homophily by timestep (all 49 steps) — drift analysis
  4. feature homophily: standardised local-feature distance across real
     edges vs random node pairs (descriptive only)
  5. structural baseline: majority-neighbour label prediction on labeled
     test nodes — a transductive upper-bound reference, NOT a fair model
Raw CSVs are read-only. No fraud-ring or camouflage claims are made:
all statistics here are descriptive.
"""
import nbformat as nbf
from pathlib import Path

NB_PATH = (Path.home() / "workspace/Homophily-Aware-GNN-Fraud-Ring-Detection"
           / "notebooks/03_homophily_analysis.ipynb")


def md(src):
    return nbf.v4.new_markdown_cell(src.strip("\n"))


def code(src):
    return nbf.v4.new_code_cell(src.strip("\n"))


cells = []

cells.append(md("""
# 03 — Homophily Analysis

**Phase 3 / Step 4 of the Homophily-Aware GNN for Fraud Ring Detection project.**

Quantifies the homophily signal that the GNN is meant to exploit: do
transactions prefer to transact with transactions of the same class?
All statistics are computed from the real Elliptic++ data
(`node_splits.csv` + raw CSVs, adjacency rebuilt with
`scripts/build_graph.py`).

**What is computed**
1. Directed 3×3 edge-type matrix + per-class homophily P(dst=c|src=c)
   with random-mixing baseline and enrichment — recomputed here and
   checked against the Phase 2 EDA values.
2. Node-level homophily for labeled nodes (fraction of labeled neighbours
   with the same label), split by class.
3. Homophily by timestep across all 49 steps (drift analysis).
4. Feature homophily: mean absolute difference of standardised local
   features across real edges vs random node pairs (descriptive only).
5. Structural baseline: majority-neighbour label prediction on labeled
   test nodes — a **transductive upper-bound reference** that peeks at
   neighbour labels, NOT a fair model.

**Constraints**
- Raw CSVs are read-only — nothing here modifies them.
- No model training, no SMOTE/oversampling, no label removal.
- Unknown labels (class 3) are kept; labelled-only statistics say so.
- No claims about fraud rings or camouflage — descriptive statistics only.
- Figures → `../results/figures/homophily/` · reports →
  `../results/homophily_stats.json`, `../results/homophily_analysis.md`.
"""))

cells.append(code("""
import sys
sys.path.insert(0, "../scripts")
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import build_graph as bg

ROOT = Path("..").resolve()
FIG_DIR = ROOT / "results" / "figures" / "homophily"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CLASS_NAMES = ["illicit", "licit", "unknown"]

classes, feats, edges = bg.load_data()
g = bg.build_graph(classes, feats, edges)
cls = classes.set_index("txId")["class"].reindex(g["tx_ids"]).to_numpy() - 1
src, dst = g["src"], g["dst"]
step_of = g["step_of"]
print(f"nodes {g['n_nodes']:,}, edges {g['n_edges']:,}")
print("label counts:", {c: int((cls == i).sum())
                        for i, c in enumerate(CLASS_NAMES)})
"""))

cells.append(code("""
# 1. Directed edge-type matrix (3x3) + per-class homophily
M = np.zeros((3, 3), dtype=np.int64)
np.add.at(M, (cls[src], cls[dst]), 1)
row_norm = M / M.sum(axis=1, keepdims=True)

lab = np.ix_([0, 1], [0, 1])
M_lab = M[lab]
h_lab = float(np.trace(M_lab) / M_lab.sum())
h_all = float(np.trace(M) / M.sum())

# random-mixing baseline from LABELED endpoint proportions
# (endpoints of edges whose BOTH ends are labelled, as in Phase 2)
lab_edge = (cls[src] < 2) & (cls[dst] < 2)
lab_end = np.concatenate([cls[src][lab_edge], cls[dst][lab_edge]])
p = np.bincount(lab_end, minlength=2) / len(lab_end)
h_base = float((p ** 2).sum())
enrich = {CLASS_NAMES[c]: float(row_norm[c, c] / p[c]) for c in (0, 1)}
p_all = np.bincount(np.concatenate([cls[src], cls[dst]]), minlength=3) / (2 * len(src))
enrich["unknown"] = float(row_norm[2, 2] / p_all[2])

print("edge-type counts (rows=src, cols=dst):")
print(pd.DataFrame(M, index=CLASS_NAMES, columns=CLASS_NAMES).to_string())
print("\\nP(dst | src):")
print(pd.DataFrame(np.round(row_norm, 4), index=CLASS_NAMES,
                   columns=CLASS_NAMES).to_string())
print(f"\\nh_lab        = {h_lab:.4f}")
print(f"h_all        = {h_all:.4f}")
print(f"baseline     = {h_base:.4f}  (p_illicit={p[0]:.4f}, p_licit={p[1]:.4f})")
print(f"enrichment   = { {k: round(v, 2) for k, v in enrich.items()} }")

# confirmation against Phase 2 EDA (results/eda_stats.json)
ref = json.loads((ROOT / "results" / "eda_stats.json").read_text())["homophily"]
assert abs(h_lab - ref["h_lab"]) < 1e-9, "h_lab differs from Phase 2"
assert abs(h_base - ref["baseline"]) < 1e-9, "baseline differs from Phase 2"
assert np.array_equal(M, np.array([[ref["edge_type_counts"][str(i + 1)][str(j + 1)]
                                    for j in range(3)] for i in range(3)])), \\
    "edge-type matrix differs from Phase 2"
print("\\nconfirmed: matches Phase 2 EDA values exactly")
"""))

cells.append(code("""
# 2. Node-level homophily for labeled nodes
# neighbours = in-neighbours + out-neighbours (undirected view of the flow)
adj = g["adj"]
A_sym = ((adj + adj.T) > 0).astype(np.int8).tocsr()
Y = np.zeros((g["n_nodes"], 3), dtype=np.int8)
Y[np.arange(g["n_nodes"]), cls] = 1
nbr_counts = np.asarray(A_sym @ Y)          # (n, 3) labelled-neighbour counts
lab_nbr = nbr_counts[:, 0] + nbr_counts[:, 1]
is_lab = cls < 2
same = nbr_counts[np.arange(g["n_nodes"]), cls]
with np.errstate(invalid="ignore", divide="ignore"):
    node_hom = np.where(lab_nbr > 0, same / np.maximum(lab_nbr, 1), np.nan)

node_stats = {}
for c in (0, 1):
    h = node_hom[is_lab & (cls == c)]
    h = h[~np.isnan(h)]
    node_stats[CLASS_NAMES[c]] = {
        "n_nodes": int((is_lab & (cls == c)).sum()),
        "n_with_labelled_neighbour": int(len(h)),
        "coverage": float(len(h) / max((is_lab & (cls == c)).sum(), 1)),
        "mean": float(h.mean()),
        "median": float(np.median(h)),
        "frac_zero": float((h == 0).mean()),
        "frac_one": float((h == 1).mean()),
    }
    print(f"{CLASS_NAMES[c]:8s} n={node_stats[CLASS_NAMES[c]]['n_nodes']:,}  "
          f"coverage={node_stats[CLASS_NAMES[c]]['coverage']:.3f}  "
          f"mean={h.mean():.3f}  median={np.median(h):.3f}  "
          f"frac(h=0)={node_stats[CLASS_NAMES[c]]['frac_zero']:.3f}  "
          f"frac(h=1)={node_stats[CLASS_NAMES[c]]['frac_one']:.3f}")
"""))

cells.append(code("""
# 3. Homophily by timestep (all 49 steps) — drift analysis
rows = []
for s in range(1, 50):
    e = step_of[src] == s
    Ms = np.zeros((3, 3), dtype=np.int64)
    np.add.at(Ms, (cls[src[e]], cls[dst[e]]), 1)
    Ms = Ms[:2, :2]
    lab_e = Ms.sum()
    h_s = float(np.trace(Ms) / lab_e) if lab_e else float("nan")
    nodes_s = step_of == s
    lab_s = cls[nodes_s] < 2
    il_rate = float((cls[nodes_s][lab_s] == 0).mean())
    rows.append({"step": s, "h_lab": h_s, "illicit_rate": il_rate,
                 "n_nodes": int(nodes_s.sum()),
                 "n_labelled_edges": int(lab_e)})
per_step = pd.DataFrame(rows)
h_trend = float(np.polyfit(per_step["step"], per_step["h_lab"], 1)[0])
r_trend = float(np.polyfit(per_step["step"], per_step["illicit_rate"], 1)[0])
corr_h = float(np.corrcoef(per_step["step"], per_step["h_lab"])[0, 1])
corr_r = float(np.corrcoef(per_step["step"], per_step["illicit_rate"])[0, 1])
print(per_step.round(4).to_string(index=False))
print(f"\\nh_lab: min={per_step['h_lab'].min():.3f} (step "
      f"{int(per_step.loc[per_step['h_lab'].idxmin(), 'step'])})  "
      f"max={per_step['h_lab'].max():.3f} (step "
      f"{int(per_step.loc[per_step['h_lab'].idxmax(), 'step'])})  "
      f"mean={per_step['h_lab'].mean():.3f}")
print(f"illicit rate: min={per_step['illicit_rate'].min():.4f}  "
      f"max={per_step['illicit_rate'].max():.4f}  "
      f"mean={per_step['illicit_rate'].mean():.4f}")
print(f"drift: h_lab slope={h_trend:+.5f}/step (corr={corr_h:+.3f}); "
      f"illicit-rate slope={r_trend:+.6f}/step (corr={corr_r:+.3f})")
"""))

cells.append(code("""
# 4. Feature homophily (descriptive only)
# Mean absolute difference of standardised local features (93 cols)
# across real edges vs random node pairs. Local features are anonymized:
# only the aggregate distance is reported, never a per-feature meaning.
rng = np.random.default_rng(42)
local_cols = [f"Local_feature_{i}" for i in range(1, 94)]
X = pd.read_csv(ROOT / "data" / "raw" / "elliptic_pp" / "txs_features.csv",
                usecols=local_cols, dtype=np.float32).to_numpy()
mu, sd = X.mean(axis=0), X.std(axis=0)
Z = ((X - mu) / np.where(sd > 0, sd, 1)).astype(np.float32)
del X

n_sample = 20_000
e_idx = rng.choice(len(src), size=n_sample, replace=False)
real_d = np.abs(Z[src[e_idx]] - Z[dst[e_idx]]).mean(axis=1)
a = rng.integers(0, g["n_nodes"], size=n_sample)
b = rng.integers(0, g["n_nodes"], size=n_sample)
rand_d = np.abs(Z[a] - Z[b]).mean(axis=1)
feat_hom = {
    "n_sampled_pairs": n_sample,
    "real_edges_mean_abs_diff": float(real_d.mean()),
    "real_edges_std_abs_diff": float(real_d.std()),
    "random_pairs_mean_abs_diff": float(rand_d.mean()),
    "random_pairs_std_abs_diff": float(rand_d.std()),
    "ratio_real_over_random": float(real_d.mean() / rand_d.mean()),
}
print(f"mean |dz| real edges   = {real_d.mean():.4f} (std {real_d.std():.4f})")
print(f"mean |dz| random pairs = {rand_d.mean():.4f} (std {rand_d.std():.4f})")
print(f"ratio real/random      = {feat_hom['ratio_real_over_random']:.3f} "
      "(<1 means neighbours are more similar than random pairs)")
"""))

cells.append(code("""
# 5. Structural baseline: majority-neighbour label prediction
# Score = fraction of a node's LABELED neighbours that are illicit.
# This PEEKS at neighbour labels (transductive), so it is an upper-bound
# reference only — NOT a fair model and NOT comparable to the Step 5
# baselines, which never see test labels.
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             f1_score, confusion_matrix)

splits = pd.read_csv(ROOT / "data" / "processed" / "node_splits.csv")
test_mask = (splits["split"] == "test").to_numpy() & is_lab
y_test = (cls[test_mask] == 0).astype(int)          # 1 = illicit
with np.errstate(invalid="ignore", divide="ignore"):
    score_all = np.where(lab_nbr > 0,
                         nbr_counts[:, 0] / np.maximum(lab_nbr, 1), 0.0)
score = score_all[test_mask]
covered = lab_nbr[test_mask] > 0

def metrics_for(y, s):
    pred = (s > 0.5).astype(int)
    return {
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
        "f1_at_0.5": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }

struct = {
    "n_test_labelled": int(test_mask.sum()),
    "n_covered": int(covered.sum()),
    "coverage": float(covered.mean()),
    "covered_only": metrics_for(y_test[covered], score[covered]),
    "all_with_fallback_licit": metrics_for(y_test, score),
}
print(f"test labelled nodes: {struct['n_test_labelled']:,}  "
      f"covered (has labelled neighbour): {struct['n_covered']:,} "
      f"({struct['coverage']:.1%})")
for k in ("covered_only", "all_with_fallback_licit"):
    m = struct[k]
    print(f"{k:26s} PR-AUC={m['pr_auc']:.4f}  ROC-AUC={m['roc_auc']:.4f}  "
          f"F1@0.5={m['f1_at_0.5']:.4f}")
    print(f"{'':26s} confusion [[TN, FP], [FN, TP]] = {m['confusion_matrix']}")
print("\\nNOTE: transductive — scores use neighbours' true labels, "
      "so this is an upper bound, not a model.")
"""))

cells.append(code("""
# 6. Figures -> results/figures/homophily/
paths = []

# 6.1 edge-type heatmap (row-normalised, counts annotated)
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(row_norm, cmap="Blues", vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{row_norm[i, j]:.2f}\\n({M[i, j]:,})",
                ha="center", va="center", fontsize=10,
                color="white" if row_norm[i, j] > 0.55 else "black")
ax.set_xticks(range(3), [f"dst {c}" for c in CLASS_NAMES])
ax.set_yticks(range(3), [f"src {c}" for c in CLASS_NAMES])
ax.set_title("P(dst class | src class) — directed edge types")
fig.colorbar(im, ax=ax, label="fraction of src-class edges")
fig.tight_layout()
fp = FIG_DIR / "homophily_edge_type_heatmap.png"
fig.savefig(fp, dpi=150); plt.close(fig); paths.append(str(fp))

# 6.2 per-class enrichment bars (observed / random-mixing)
fig, ax = plt.subplots(figsize=(7, 4.5))
names = ["illicit", "licit", "unknown"]
vals = [enrich[c] for c in names]
cols = ["#E45756" if v > 1 else "#4C78A8" for v in vals]
ax.bar(names, vals, color=cols)
ax.axhline(1.0, color="black", ls="--", lw=1, label="random mixing (=1)")
for i, v in enumerate(vals):
    ax.text(i, v + 0.15, f"{v:.2f}x", ha="center", fontsize=11)
ax.set_ylabel("enrichment (observed / random-mixing)")
ax.set_title("Same-class linkage enrichment by class")
ax.legend()
fig.tight_layout()
fp = FIG_DIR / "homophily_enrichment_bars.png"
fig.savefig(fp, dpi=150); plt.close(fig); paths.append(str(fp))

# 6.3 homophily by timestep
fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
a1.plot(per_step["step"], per_step["h_lab"], marker="o", ms=3, color="#4C78A8")
a1.axhline(h_lab, color="#E45756", ls="--", lw=1,
           label=f"overall h_lab={h_lab:.3f}")
a1.set_ylabel("h_lab per step")
a1.set_title("Labelled-edge homophily by timestep")
a1.legend()
a2.plot(per_step["step"], per_step["illicit_rate"], marker="o", ms=3,
        color="#E45756")
a2.set_xlabel("timestep")
a2.set_ylabel("illicit / labelled")
a2.set_title("Illicit rate among labelled nodes by timestep")
fig.tight_layout()
fp = FIG_DIR / "homophily_by_timestep.png"
fig.savefig(fp, dpi=150); plt.close(fig); paths.append(str(fp))

# 6.4 node-level homophily histograms, split by class
fig, ax = plt.subplots(figsize=(8, 4.5))
for c in (0, 1):
    h = node_hom[is_lab & (cls == c)]
    h = h[~np.isnan(h)]
    ax.hist(h, bins=20, range=(0, 1), alpha=0.6,
            label=f"{CLASS_NAMES[c]} (n={len(h):,})")
ax.set_xlabel("fraction of labelled neighbours with the same label")
ax.set_ylabel("nodes")
ax.set_title("Node-level homophily distribution (labelled nodes)")
ax.legend()
fig.tight_layout()
fp = FIG_DIR / "homophily_node_histograms.png"
fig.savefig(fp, dpi=150); plt.close(fig); paths.append(str(fp))

# 6.5 feature similarity: real edges vs random pairs
fig, ax = plt.subplots(figsize=(7, 4.5))
labels = ["real edges", "random pairs"]
means = [feat_hom["real_edges_mean_abs_diff"],
         feat_hom["random_pairs_mean_abs_diff"]]
stds = [feat_hom["real_edges_std_abs_diff"],
        feat_hom["random_pairs_std_abs_diff"]]
ax.bar(labels, means, yerr=stds, capsize=6,
       color=["#4C78A8", "#9E9E9E"])
for i, v in enumerate(means):
    ax.text(i, v + stds[i] + 0.02, f"{v:.3f}", ha="center", fontsize=11)
ax.set_ylabel("mean |standardised local-feature difference|")
ax.set_title("Feature similarity across edges vs random pairs "
             f"(n={n_sample:,}, ratio {feat_hom['ratio_real_over_random']:.2f})")
fig.tight_layout()
fp = FIG_DIR / "homophily_feature_similarity.png"
fig.savefig(fp, dpi=150); plt.close(fig); paths.append(str(fp))

for fp in paths:
    print("saved", Path(fp).name)
"""))

cells.append(code("""
# 7. Reports -> results/homophily_stats.json, results/homophily_analysis.md
homophily_stats = {
    "edge_type_counts": {str(i + 1): {str(j + 1): int(M[i, j])
                                      for j in range(3)} for i in range(3)},
    "p_dst_given_src": {CLASS_NAMES[i]: {CLASS_NAMES[j]: float(row_norm[i, j])
                                          for j in range(3)} for i in range(3)},
    "h_lab": h_lab,
    "h_all": h_all,
    "random_mixing_baseline": h_base,
    "labelled_endpoint_proportions": {"illicit": float(p[0]),
                                      "licit": float(p[1])},
    "enrichment": {k: float(v) for k, v in enrich.items()},
    "node_homophily": node_stats,
    "per_timestep": per_step.round(4).to_dict("records"),
    "drift": {"h_lab_slope_per_step": h_trend, "h_lab_corr_with_step": corr_h,
              "illicit_rate_slope_per_step": r_trend,
              "illicit_rate_corr_with_step": corr_r},
    "feature_homophily": feat_hom,
    "structural_baseline_majority_neighbour": struct,
}
(ROOT / "results" / "homophily_stats.json").write_text(
    json.dumps(homophily_stats, indent=2))
print("wrote results/homophily_stats.json")

md_lines = [
    "# Homophily Analysis Summary — Elliptic++ Transactions",
    "",
    "*Phase 3 / Step 4 · descriptive statistics only — no fraud-ring or "
    "camouflage claims; no model training.*",
    "",
    "## 1. Edge-type homophily (directed)",
    "",
    f"- Labelled-edge homophily h_lab = **{h_lab:.4f}** vs random-mixing "
    f"baseline **{h_base:.4f}** (confirmed: identical to Phase 2 EDA).",
    f"- Illicit -> illicit linkage is **{enrich['illicit']:.1f}x** the "
    "random-mixing rate; licit -> licit is "
    f"**{enrich['licit']:.2f}x** (below chance).",
    f"- P(dst=illicit | src=illicit) = {row_norm[0, 0]:.4f}; "
    f"P(dst=licit | src=licit) = {row_norm[1, 1]:.4f}.",
    "- Reading: the aggregate h_lab is dominated by the licit majority; "
    "the informative view is per-class conditioning — illicit nodes link "
    "to illicit nodes well above chance, licit nodes mix more broadly.",
    "",
    "## 2. Node-level homophily",
    "",
]
for c in ("illicit", "licit"):
    d = node_stats[c]
    md_lines.append(
        f"- **{c}**: mean same-label neighbour fraction "
        f"{d['mean']:.3f} (median {d['median']:.3f}); coverage "
        f"{d['coverage']:.1%} of nodes have >=1 labelled neighbour; "
        f"fraction with zero same-label neighbours {d['frac_zero']:.3f}.")
md_lines += [
    "",
    "## 3. Homophily by timestep",
    "",
    f"- Per-step h_lab ranges {per_step['h_lab'].min():.3f}–"
    f"{per_step['h_lab'].max():.3f} (mean {per_step['h_lab'].mean():.3f}); "
    f"slope {h_trend:+.5f} per step (corr {corr_h:+.3f}) — no strong drift.",
    f"- Illicit rate among labelled nodes ranges "
    f"{per_step['illicit_rate'].min():.4f}–"
    f"{per_step['illicit_rate'].max():.4f}; slope {r_trend:+.6f} per step "
    f"(corr {corr_r:+.3f}).",
    "",
    "## 4. Feature homophily",
    "",
    f"- Mean |standardised local-feature difference|: real edges "
    f"{feat_hom['real_edges_mean_abs_diff']:.4f} vs random pairs "
    f"{feat_hom['random_pairs_mean_abs_diff']:.4f} "
    f"(ratio {feat_hom['ratio_real_over_random']:.2f}). Neighbours are "
    "more similar than random pairs, but the difference is modest — "
    "consistent with weak overall homophily, not with a strong "
    "feature-based cluster structure.",
    "",
    "## 5. Structural baseline (transductive upper bound)",
    "",
    f"- Majority-neighbour prediction on labelled test nodes "
    f"(n={struct['n_test_labelled']:,}, coverage {struct['coverage']:.1%}): "
    f"PR-AUC {struct['covered_only']['pr_auc']:.4f}, ROC-AUC "
    f"{struct['covered_only']['roc_auc']:.4f}, F1@0.5 "
    f"{struct['covered_only']['f1_at_0.5']:.4f} (covered nodes only).",
    "- This peeks at neighbours' true labels, so it is an upper-bound "
    "reference only — NOT a fair model and NOT comparable to the Step 5 "
    "baselines, which never see test labels.",
    "",
    "## 6. What this means for the GNN (Step 6)",
    "",
    "- A plain homophily-assuming GNN (GCN-style uniform averaging) is a "
    "weak prior here: aggregate homophily is close to the random-mixing "
    "baseline and licit nodes mix broadly. The class-conditional signal "
    "(illicit 5.9x enrichment) argues for heterophily-aware / "
    "class-conditional message passing, and for evaluating on whether the "
    "model beats the structural baseline above — especially on "
    "cross-class edges.",
    "",
    "## 7. Figures",
    "",
]
for fp in paths:
    md_lines.append(f"- `{Path(fp).name}`")
md_lines.append("")
(ROOT / "results" / "homophily_analysis.md").write_text("\\n".join(md_lines))
print("wrote results/homophily_analysis.md")
"""))

cells.append(md("""
## Summary

- **Edge homophily recomputed and confirmed** against Phase 2: h_lab 0.9537
  vs baseline 0.9043; illicit→illicit enrichment 5.9×, licit 0.72×.
- **Node-level homophily** and **per-timestep drift** quantified — no strong
  temporal drift in homophily or illicit rate.
- **Feature homophily** is modest: neighbours are more similar than random
  pairs, but not dramatically so.
- **Structural baseline** (majority-neighbour, transductive upper bound)
  recorded for Step 6 to beat — it peeks at neighbour labels, so it is
  not a fair model.
- **Artifacts:** `results/homophily_stats.json`,
  `results/homophily_analysis.md`, figures in `results/figures/homophily/`
- **Next:** `04_baseline_models.ipynb` — LogReg / RandomForest / XGBoost /
  MLP / Dummy on the temporal splits (Step 5).
"""))

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3",
                             "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python", "version": "3.10"}
nbf.write(nb, NB_PATH)
print(f"wrote {NB_PATH} ({len(cells)} cells)")
