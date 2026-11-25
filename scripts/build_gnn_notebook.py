"""Build notebooks/05_gnn_experiments.ipynb (Phase 3, Step 6).

Generates the notebook programmatically with real code cells only (no IPython
magics), so it can be executed headless via scripts/run_notebook.py.
The heavy lifting lives in scripts/run_gnn.py; notebook cells call it
step by step. Raw CSVs are read-only.
"""
import nbformat as nbf
from pathlib import Path

NB_PATH = (Path.home() / "workspace/Homophily-Aware-GNN-Fraud-Ring-Detection"
           / "notebooks/05_gnn_experiments.ipynb")


def md(src):
    return nbf.v4.new_markdown_cell(src.strip("\n"))


def code(src):
    return nbf.v4.new_code_cell(src.strip("\n"))


cells = []

cells.append(md("""
# 05 — GNN Experiments

**Phase 3 / Step 6 of the Homophily-Aware GNN for Fraud Ring Detection project.**

Graph neural networks on the temporal splits — the models the baselines
(Step 5) were the bar for. Models: GCN, GAT (4 heads), GraphSAGE (mean
aggregator), an H2GCN-style homophily-aware variant (ego embedding kept
separate, 1-hop + 2-hop neighbour channels), and a GatedGCN variant
(learned per-node gate blending ego and neighbour embeddings).

**Why the two homophily-aware variants (Step 4 evidence)**
- Illicit nodes link to illicit nodes at **5.9×** the random-mixing rate,
  so neighbour evidence is real — but node homophily is **bimodal** for
  illicit nodes (45% have zero same-label neighbours, 46% have all).
  Uniform smoothing (GCN) dilutes the ego signal of the zero-homophily
  illicit nodes.
- H2GCN-style keeps the ego embedding separate from neighbour
  aggregation and adds a 2-hop channel, so a node can rely on its own
  features when its neighbourhood is uninformative.
- GatedGCN learns a per-node gate from [ego, neighbour], so each node
  decides how much to trust its neighbourhood — directly addressing
  the bimodal case and licit nodes mixing broadly (0.72×).

**Setup (all constraints honoured)**
- **Splits are temporal** (train steps 1–34, val 35–41, test 42–49) —
  never random.
- **Supervised on labelled nodes only** (illicit vs licit). Unknown nodes
  (class 3) stay in the graph as message-passing context — never
  deleted, never training targets.
- **Features:** the identical pipeline to Step 5 — all 183 features,
  median imputation fit on TRAIN only + one missingness-indicator
  column, StandardScaler fit on train only.
- **Aggregation over INCOMING neighbours** (money senders), respecting
  the directed money-flow direction.
- **No SMOTE, no oversampling, no resampling** — class-weighted BCE
  loss only (pos_weight = licit/illicit on train).
- **Training:** full-batch, Adam lr 0.01, weight decay 5e-4, dropout
  0.5, hidden 64, 2 layers, up to 200 epochs, early stopping on val
  PR-AUC (patience 20). Model selection on VAL; final numbers on TEST.
- **Primary metric PR-AUC** (average precision). Accuracy is never
  headlined — only 2.23% of transactions are illicit.
- **Transductive:** all node features/edges are visible during message
  passing, but the loss uses train labels only; val labels only for
  early stopping/selection; test evaluated once at the end.

**Outputs**
- Figures → `../results/figures/gnn/` · reports →
  `../results/gnn_results.json`, `../results/gnn_summary.md`.
"""))

cells.append(code("""
import sys
sys.path.insert(0, "../scripts")
import run_gnn as rg
import torch

print("torch", torch.__version__, "| threads:", torch.get_num_threads())
import torch_geometric
print("torch_geometric", torch_geometric.__version__)
print("models:", rg.MODELS)
print("XGBoost bar to beat (Step 5 test PR-AUC):", rg.XGB_TEST_PR_AUC)
"""))

cells.append(code("""
# 1. Load data — same feature pipeline as Step 5, plus directed edges
data = rg.load_graph_data()
print(f"nodes: {data['n_nodes']:,}, edges: {data['n_edges']:,}, "
      f"features: {len(data['feat_cols'])} + indicator = "
      f"{data['X'].shape[1]}")
for name in ("train", "val", "test"):
    m = data["masks"][name]
    print(f"{name:5s} labelled: {int(m.sum()):,} "
          f"(illicit {int((data['y'][m] == 1).sum()):,}, "
          f"licit {int((data['y'][m] == 0).sum()):,})")
print(f"pos_weight (licit/illicit train): {data['pos_weight']:.3f}")
print("edge_index shape:", data["edge_index"].shape,
      "| self-loops:", int((data['edge_index'][0]
                            == data['edge_index'][1]).sum()))
"""))

cells.append(code("""
# 2. Torch tensors (CPU)
t = rg.to_tensors(data)
print("x:", tuple(t["x"].shape), t["x"].dtype)
print("edge_index:", tuple(t["edge_index"].shape))
"""))

cells.append(code("""
# 3. Train all five GNNs (selection on val PR-AUC) — takes a while on CPU
results, fitted = rg.train_all(t, data["pos_weight"])
best = max(results, key=lambda n: results[n]["val"]["pr_auc"])
print(f"\\nbest model by val PR-AUC: {best} "
      f"({results[best]['val']['pr_auc']:.4f})")
"""))

cells.append(code("""
# 4. Results table — val and test, PR-AUC primary
print(f"{'model':9s} {'val PR-AUC':>10s} {'test PR-AUC':>11s} "
      f"{'test ROC-AUC':>12s} {'test F1@0.5':>11s} "
      f"{'prec@0.5':>9s} {'rec@0.5':>8s} {'fit s':>6s} {'epoch':>5s}")
for name in sorted(results, key=lambda n: results[n]["val"]["pr_auc"],
                   reverse=True):
    v, tt = results[name]["val"], results[name]["test"]
    print(f"{name:9s} {v['pr_auc']:10.4f} {tt['pr_auc']:11.4f} "
          f"{tt['roc_auc']:12.4f} {tt['f1_at_0.5']:11.4f} "
          f"{tt['precision_at_0.5']:9.4f} {tt['recall_at_0.5']:8.4f} "
          f"{results[name]['fit_seconds']:6.1f} "
          f"{results[name]['best_epoch']:5d}")
print(f"\\nXGBoost bar (Step 5): test PR-AUC {rg.XGB_TEST_PR_AUC:.4f} — "
      f"best GNN beats it by "
      f"{results[best]['test']['pr_auc'] - rg.XGB_TEST_PR_AUC:+.4f}")
print("(accuracy is deliberately not reported — 2.23% illicit makes it "
      "meaningless here)")
"""))

cells.append(code("""
# 5. Fraud-ring evaluation (descriptive) — best model, test labelled nodes
fraud = rg.fraud_ring_eval(best, results, t, data)
print(f"predicted-illicit: {fraud['n_pred_illicit']:,} "
      f"(true {fraud['n_true_illicit']:,}), precision "
      f"{fraud['precision_illicit']:.4f}, recall "
      f"{fraud['recall_illicit']:.4f}")
print(f"pred subgraph: {fraud['pred_subgraph']['n_components']} weak "
      f"components, largest {fraud['pred_subgraph']['largest']}, "
      f"{fraud['pred_subgraph']['singletons']} singletons")
print(f"true subgraph: {fraud['true_subgraph']['n_components']} weak "
      f"components, largest {fraud['true_subgraph']['largest']}, "
      f"{fraud['true_subgraph']['singletons']} singletons")
print(f"pred illicit->illicit edge rate: "
      f"{fraud['pred_illicit_illicit_edge_rate']:.4f} "
      f"(true-illicit test rate "
      f"{fraud['true_illicit_illicit_edge_rate_test']:.4f}; "
      f"observed overall 0.2961)")
print("descriptive only — not proof of organised fraud rings")
"""))

cells.append(code("""
# 6. Explainability — GAT attention by edge type
attention = rg.gat_attention_analysis(fitted["gat"], t, data)
print(f"{'edge type':18s} {'n edges':>9s} {'mean att.':>10s} "
      f"{'median att.':>11s}")
for k, v in attention.items():
    print(f"{k:18s} {v['n_edges']:9,} {v['mean_attention']:10.4f} "
          f"{v['median_attention']:11.4f}")
"""))

cells.append(code("""
# 7. Feature-group ablation for the best model (retrains it twice)
ablation_res = rg.ablation(best, data, data["feat_cols"],
                           data["pos_weight"])
print(f"\\n{'feature set':20s} {'val PR-AUC':>10s} {'test PR-AUC':>11s}")
for gname in ("local_only", "local_plus_aggregate"):
    a = ablation_res[gname]
    print(f"{gname:20s} {a['val']['pr_auc']:10.4f} "
          f"{a['test']['pr_auc']:11.4f}")
print(f"{'all features':20s} {results[best]['val']['pr_auc']:10.4f} "
      f"{results[best]['test']['pr_auc']:11.4f}")
"""))

cells.append(code("""
# 8. Figures (saved to ../results/figures/gnn/)
fig_paths = rg.make_figures(results, best, fraud, attention, ablation_res)
for fp in fig_paths:
    print("saved", fp.split("/")[-1])
"""))

cells.append(code("""
# 9. Reports (JSON + Markdown summary)
rg.write_reports(results, best, fraud, attention, ablation_res, fig_paths)
print("wrote results/gnn_results.json, results/gnn_summary.md")
print(f"best model: {best} — val PR-AUC "
      f"{results[best]['val']['pr_auc']:.4f}, test PR-AUC "
      f"{results[best]['test']['pr_auc']:.4f}")
"""))

cells.append(md("""
## Summary

- **Five GNNs trained on temporal splits** with class-weighted loss
  only — no SMOTE, no oversampling, no random splits.
- **Best model by val PR-AUC** and the full comparison table are in
  `results/gnn_summary.md`; per-model metrics in
  `results/gnn_results.json`.
- The XGBoost bar from Step 5 (test PR-AUC 0.5652) is the reference:
  the results table above shows whether, and by how much, the best
  GNN beats it.
- **Fraud-ring evaluation is descriptive** — component sizes and
  illicit→illicit edge rates of the predicted-illicit subgraph, not
  proof of organised rings.
- **Next:** Step 6 is complete — the full pipeline (EDA → graph →
  homophily → baselines → GNNs) is done. Remaining work is the final
  report update and any follow-up ablations.
"""))

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3",
                             "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python", "version": "3.10"}
nbf.write(nb, NB_PATH)
print(f"wrote {NB_PATH} ({len(cells)} cells)")
