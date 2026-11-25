"""Build notebooks/04_baseline_models.ipynb (Phase 3, Step 5).

Generates the notebook programmatically with real code cells only (no IPython
magics), so it can be executed headless via scripts/run_notebook.py.
The heavy lifting lives in scripts/run_baselines.py; notebook cells call it
step by step. Raw CSVs are read-only.
"""
import nbformat as nbf
from pathlib import Path

NB_PATH = (Path.home() / "workspace/Homophily-Aware-GNN-Fraud-Ring-Detection"
           / "notebooks/04_baseline_models.ipynb")


def md(src):
    return nbf.v4.new_markdown_cell(src.strip("\n"))


def code(src):
    return nbf.v4.new_code_cell(src.strip("\n"))


cells = []

cells.append(md("""
# 04 — Baseline Models

**Phase 3 / Step 5 of the Homophily-Aware GNN for Fraud Ring Detection project.**

Non-graph baselines on the temporal splits — the bar the GNN (Step 6) has
to beat. Models: Logistic Regression, Random Forest, XGBoost, MLP, and a
stratified Dummy classifier as the reference floor.

**Setup (all constraints honoured)**
- **Splits are temporal** (train steps 1–34, val 35–41, test 42–49) —
  never random.
- **Supervised on labelled nodes only** (illicit vs licit). Unknown nodes
  (class 3) stay in the dataset — they are never deleted — but have no
  target and take no part in training/evaluation.
- **Features:** all 183 (Time step + 93 local + 72 aggregate + 17 named
  stats).
- **Missing values:** median imputation fit on TRAIN only + one binary
  missingness-indicator column for the 17 named-stat block. The label is
  never used for imputation.
- **Standardisation:** StandardScaler fit on train only; applied to
  LogReg/MLP only (tree models use unscaled features).
- **No SMOTE, no oversampling, no resampling** — class weighting only.
- **Model selection on VAL using PR-AUC;** final numbers on TEST.
- **Primary metric PR-AUC** (average precision). Accuracy is never
  headlined — only 2.23% of transactions are illicit.

**Outputs**
- Figures → `../results/figures/baselines/` · reports →
  `../results/baseline_results.json`, `../results/baseline_summary.md`.
"""))

cells.append(code("""
import sys
sys.path.insert(0, "../scripts")
import run_baselines as rb

print("scripts/run_baselines.py loaded")
print("named-stat columns:", len(rb.NAMED_STATS))
"""))

cells.append(code("""
# 1. Load data (read-only): all 183 features + labels + temporal splits
df = rb.load_all()
feat_cols = rb.feature_columns(df)
print(f"rows: {len(df):,}  feature columns: {len(feat_cols)}")
print("splits:", df["split"].value_counts().to_dict())
print("classes:", df["class"].value_counts().to_dict())
print("missing cells in named stats:",
      int(df[rb.NAMED_STATS].isna().sum().sum()))
print("rows missing all 17 named stats:",
      int(df[rb.NAMED_STATS].isna().all(axis=1).sum()))
"""))

cells.append(code("""
# 2. Labelled train/val/test matrices (unknowns excluded — no target)
Xy = rb.split_xy(df, feat_cols)
for name in ("train", "val", "test"):
    X, y = Xy[name]
    print(f"{name:5s} labelled: {len(y):,}  "
          f"(illicit {int(y.sum()):,}, licit {int((y == 0).sum()):,})  "
          f"X shape {X.shape}")
print("\\nunknown nodes stay in df (never deleted) but have no target: "
      f"{int((df['class'] == 3).sum()):,} rows")
"""))

cells.append(code("""
# 3. Train all models (selection on val PR-AUC) — takes a few minutes
results, fitted = rb.train_all(Xy, feat_cols)
best = max(results, key=lambda n: results[n]["val"]["pr_auc"])
print(f"\\nbest model by val PR-AUC: {best} "
      f"({results[best]['val']['pr_auc']:.4f})")
"""))

cells.append(code("""
# 4. Results table — val and test, PR-AUC primary
print(f"{'model':17s} {'val PR-AUC':>10s} {'test PR-AUC':>11s} "
      f"{'test ROC-AUC':>12s} {'test F1@0.5':>11s} "
      f"{'prec@0.5':>9s} {'rec@0.5':>8s}")
for name in sorted(results, key=lambda n: results[n]["val"]["pr_auc"],
                   reverse=True):
    v, t = results[name]["val"], results[name]["test"]
    print(f"{name:17s} {v['pr_auc']:10.4f} {t['pr_auc']:11.4f} "
          f"{t['roc_auc']:12.4f} {t['f1_at_0.5']:11.4f} "
          f"{t['precision_at_0.5']:9.4f} {t['recall_at_0.5']:8.4f}")
print("\\n(accuracy is deliberately not reported — 2.23% illicit makes it "
      "meaningless here)")
"""))

cells.append(code("""
# 5. Figures (saved to ../results/figures/baselines/)
fig_paths = rb.make_figures(results, feat_cols, fitted)
for fp in fig_paths:
    print("saved", fp.split("/")[-1])
"""))

cells.append(code("""
# 6. Reports (JSON + Markdown summary)
rb.write_reports(results, feat_cols, fig_paths, best)
print("wrote results/baseline_results.json, results/baseline_summary.md")
print(f"best model: {best} — val PR-AUC "
      f"{results[best]['val']['pr_auc']:.4f}, test PR-AUC "
      f"{results[best]['test']['pr_auc']:.4f}")
"""))

cells.append(md("""
## Summary

- **Baselines trained on temporal splits** with class weighting only —
  no SMOTE, no oversampling, no random splits.
- **Best model by val PR-AUC** and the full comparison table are in
  `results/baseline_summary.md`; per-model metrics in
  `results/baseline_results.json`.
- These numbers are the bar for Step 6: the GNN must beat the best
  baseline's test PR-AUC — especially on cross-class edges, where the
  homophily analysis (notebook 03) showed the signal is weakest.
- **Next:** `05_gnn_experiments.ipynb` — GCN / GAT / GraphSAGE and
  homophily-aware variants (Step 6, not started).
"""))

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3",
                             "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python", "version": "3.10"}
nbf.write(nb, NB_PATH)
print(f"wrote {NB_PATH} ({len(cells)} cells)")
