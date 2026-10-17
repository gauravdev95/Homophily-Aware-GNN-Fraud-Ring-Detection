"""Build notebooks/02_graph_construction.ipynb (Phase 3, Step 3).

Generates the notebook programmatically with real code cells only (no IPython
magics), so it can be executed headless via scripts/run_notebook.py.
The heavy lifting lives in scripts/build_graph.py; notebook cells call it.
Raw CSVs are read-only.
"""
import nbformat as nbf
from pathlib import Path

NB_PATH = (Path.home() / "workspace/Homophily-Aware-GNN-Fraud-Ring-Detection"
           / "notebooks/02_graph_construction.ipynb")


def md(src):
    return nbf.v4.new_markdown_cell(src.strip("\n"))


def code(src):
    return nbf.v4.new_code_cell(src.strip("\n"))


cells = []

cells.append(md("""
# 02 — Graph Construction & Temporal Split

**Phase 3 / Step 3 of the Homophily-Aware GNN for Fraud Ring Detection project.**

Builds the **directed transaction → transaction money-flow graph** from
`txs_edgelist.csv`, validates its structure, and defines **temporal**
train/val/test splits — no random splitting, so evaluation measures
generalization to future transactions.

**Splits** (per EDA recommendation, Elliptic protocol family):
- train = timesteps 1–34
- val = timesteps 35–41
- test = timesteps 42–49

**Constraints**
- Raw CSVs are read-only — nothing here modifies them.
- No model training, no SMOTE/oversampling, no label removal.
- Unknown labels (class 3) are kept (semi-supervised setting).
- Figures → `../results/figures/graph/` · artifacts → `../data/processed/`.
"""))

cells.append(code("""
import sys
sys.path.insert(0, "../scripts")
from pathlib import Path
import build_graph as bg

print("scripts/build_graph.py loaded")
print("splits:", bg.SPLITS)
"""))

cells.append(code("""
# 1. Load raw data (read-only)
classes, feats, edges = bg.load_data()
print(f"classes : {classes.shape}  (txId, class)")
print(f"features: {feats.shape}  (txId, Time step)")
print(f"edges   : {edges.shape}  (txId1 -> txId2)")
"""))

cells.append(code("""
# 2. Build the directed money-flow graph
g = bg.build_graph(classes, feats, edges)
print(f"nodes          : {g['n_nodes']:,}")
print(f"directed edges : {g['n_edges']:,}")
print(f"density        : {g['density']:.3e}")
print(f"weak components: {g['n_components']}")
print(f"in-degree  mean/max : {g['in_deg'].mean():.2f} / {g['in_deg'].max()}")
print(f"out-degree mean/max : {g['out_deg'].mean():.2f} / {g['out_deg'].max()}")
"""))

cells.append(code("""
# 3. Structural validation
checks = bg.validate(g)
for name, (passed, detail) in checks.items():
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
assert all(p for p, _ in checks.values()), "validation failed"
"""))

cells.append(code("""
# 4. Temporal splits (never random)
node_splits = bg.make_splits(classes, g["step_of"], g["tx_ids"])
stats = bg.split_stats(node_splits, g)
print(f"{'split':6s} {'steps':>7s} {'nodes':>9s} {'edges':>9s} "
      f"{'illicit':>8s} {'licit':>8s} {'unknown':>8s}")
for s in ("train", "val", "test"):
    d = stats[s]
    print(f"{s:6s} {d['timesteps']:>7s} {d['n_nodes']:>9,} {d['n_edges']:>9,} "
          f"{d['n_illicit']:>8,} {d['n_licit']:>8,} {d['n_unknown']:>8,}")
print("\\nclass totals check:",
      sum(stats[s]['n_illicit'] for s in stats) == 4545,
      sum(stats[s]['n_licit'] for s in stats) == 42019,
      sum(stats[s]['n_unknown'] for s in stats) == 157205)
print("edge total check:",
      sum(stats[s]['n_edges'] for s in stats) == g['n_edges'])
"""))

cells.append(code("""
# 5. Persist processed artifacts (gitignored; raw data untouched)
from pathlib import Path
bg.PROCESSED.mkdir(parents=True, exist_ok=True)
node_splits.to_csv(bg.PROCESSED / "node_splits.csv", index=False)
print("wrote", bg.PROCESSED / "node_splits.csv", node_splits.shape)
import json
(bg.PROCESSED / "graph_splits.json").write_text(
    json.dumps({"splits": bg.SPLITS, "stats": stats}, indent=2))
print("wrote", bg.PROCESSED / "graph_splits.json")
print(node_splits.head(3).to_string(index=False))
"""))

cells.append(code("""
# 6. Figures (saved to ../results/figures/graph/)
fig_paths = bg.make_figures(node_splits, stats)
for p in fig_paths:
    print("saved", Path(p).name)
bg.write_reports(g, checks, stats, fig_paths)
print("wrote results/graph_stats.json, results/graph_construction_summary.md")
"""))

cells.append(md("""
## Summary

- **Directed graph:** 203,769 nodes · 234,355 edges · density 5.6e-06
- **Validation:** 0 cross-step edges · 49 components 1:1 with 49 timesteps ·
  0 self-loops · 0 isolated nodes — all PASS
- **Temporal splits:** train (steps 1–34) 136,265 nodes ·
  val (steps 35–41) 30,680 nodes · test (steps 42–49) 36,824 nodes.
  No edge crosses a split boundary (graph-clean by construction).
- **Artifacts:** `data/processed/node_splits.csv`, `data/processed/graph_splits.json`
- **Next:** `03_homophily_analysis.ipynb` — edge/node homophily, class-wise
  assortativity, per-timestep drift.
"""))

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3",
                             "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python", "version": "3.10"}
nbf.write(nb, NB_PATH)
print(f"wrote {NB_PATH} ({len(cells)} cells)")
