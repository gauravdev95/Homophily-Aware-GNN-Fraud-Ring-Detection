"""Build notebooks/01_data_exploration.ipynb (Phase 2 EDA) programmatically.

The notebook performs the full 14-section EDA with real code cells only
(no IPython magics), so it can be executed headless. Raw CSVs are read-only.
"""
import nbformat as nbf
from pathlib import Path

NB_PATH = Path.home() / "workspace/Homophily-Aware-GNN-Fraud-Ring-Detection/notebooks/01_data_exploration.ipynb"


def md(src):
    return nbf.v4.new_markdown_cell(src.strip("\n"))


def code(src):
    return nbf.v4.new_code_cell(src.strip("\n"))


cells = []

# ---------------------------------------------------------------- 0. header
cells.append(md("""
# 01 — Exploratory Data Analysis: Elliptic++ Transactions Dataset

**Phase 2 of the Homophily-Aware GNN for Fraud Ring Detection project.**

Complete, research-quality EDA of the three raw Elliptic++ transaction files
(`txs_features.csv`, `txs_classes.csv`, `txs_edgelist.csv`). Raw files are read
**read-only** — nothing here modifies them.

**Official references**
- Y. Elmougy & L. Liu, *Demystifying Fraudulent Transactions and Illicit Nodes in the
  Bitcoin Network for Financial Forensics*, KDD 2023. Code/data: https://github.com/git-disl/EllipticPlusPlus
- Documented facts used below: 203,769 transactions, 234,355 money-flow edges, 49 time steps,
  183 features, classes 1=illicit / 2=licit / 3=unknown. Feature columns are `txId`,
  `Time step`, 93 anonymized standardized local features (`Local_feature_1..93`),
  72 anonymized aggregate features (`Aggregate_feature_1..72`), and 17 named statistical
  columns (`in_txs_degree`, `out_txs_degree`, `total_BTC`, `fees`, `size`,
  `num_input_addresses`, `num_output_addresses`, `in_BTC_min/max/mean/median/total`,
  `out_BTC_min/max/mean/median/total`).

**Scope & constraints**
- No model training, no SMOTE/oversampling, no label removal, no train/test splitting.
  (The small stratified sample in §3 exists only for EDA plots — not for modeling.)
- Feature meanings come only from official documentation; anonymized local features are
  referenced by name and never interpreted.
- The ~695 MB features file is processed in chunks (§3); later sections reuse the
  accumulated statistics plus a small local sample.
- Figures → `../results/figures/eda/` · key numbers → `../results/eda_stats.json`.
"""))

cells.append(code("""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: every figure is saved to disk
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

RNG = np.random.default_rng(42)

RAW = Path("../data/raw/elliptic_pp")
PROC = Path("../data/processed"); PROC.mkdir(parents=True, exist_ok=True)
FIG = Path("../results/figures/eda"); FIG.mkdir(parents=True, exist_ok=True)
RES = Path("../results"); RES.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {1: "illicit", 2: "licit", 3: "unknown"}
CLASS_COLORS = {1: "#d62728", 2: "#2ca02c", 3: "#7f7f7f"}
AGG_COLS = ["in_txs_degree", "out_txs_degree", "total_BTC", "fees", "size",
            "num_input_addresses", "num_output_addresses",
            "in_BTC_min", "in_BTC_max", "in_BTC_mean", "in_BTC_median", "in_BTC_total",
            "out_BTC_min", "out_BTC_max", "out_BTC_mean", "out_BTC_median", "out_BTC_total"]

plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})
print("paths ok:", RAW.exists(), "| fig dir:", FIG)
"""))

# ------------------------------------------------- 1. shape / dtypes / memory
cells.append(md("## 1. Dataset shape, columns, dtypes and memory usage"))
cells.append(code("""
classes = pd.read_csv(RAW / "txs_classes.csv")
edgelist = pd.read_csv(RAW / "txs_edgelist.csv")
print("txs_classes :", classes.shape, dict(classes.dtypes.astype(str)),
      f"| mem={classes.memory_usage(deep=True).sum()/1e6:.2f} MB")
print("txs_edgelist:", edgelist.shape, dict(edgelist.dtypes.astype(str)),
      f"| mem={edgelist.memory_usage(deep=True).sum()/1e6:.2f} MB")
print("\\nclass value counts (1=illicit, 2=licit, 3=unknown):")
print(classes["class"].value_counts().sort_index().to_dict())
"""))

cells.append(code("""
# Chunked single pass over the ~695 MB features file (25k rows at a time, memory-safe).
FEAT_PATH = RAW / "txs_features.csv"
CHUNK = 25_000
cls_map = dict(zip(classes["txId"].to_numpy(), classes["class"].to_numpy()))

feature_cols, col_dtypes = None, {}
n_feat_rows, dup_feat = 0, 0
missing = None
seen_ids = set()
sums, sumsq, cnts = {}, {}, {}
ts_min, ts_max = np.inf, -np.inf
ts_values = set()
sample_parts = []
LIC_FRAC, UNK_FRAC = 8000 / 42019, 8000 / 157205  # ~8k licit + ~8k unknown, for plots only

for chunk in pd.read_csv(FEAT_PATH, chunksize=CHUNK):
    if feature_cols is None:
        feature_cols = [c for c in chunk.columns if c != "txId"]
        col_dtypes = {c: str(chunk[c].dtype) for c in chunk.columns}
        for k in (1, 2, 3):
            sums[k] = np.zeros(len(feature_cols))
            sumsq[k] = np.zeros(len(feature_cols))
            cnts[k] = np.zeros(len(feature_cols))
    n_feat_rows += len(chunk)
    m = chunk.isna().sum()
    missing = m if missing is None else missing + m
    ids = chunk["txId"].to_numpy()
    dup_feat += int(pd.Series(ids).duplicated().sum()) + len(set(ids) & seen_ids)
    seen_ids |= set(ids)
    tsv = chunk["Time step"].to_numpy()
    ts_min, ts_max = min(ts_min, tsv.min()), max(ts_max, tsv.max())
    ts_values.update(np.unique(tsv).tolist())
    cl = chunk["txId"].map(cls_map).to_numpy()
    X = chunk[feature_cols].to_numpy(dtype=np.float64)
    for k in (1, 2, 3):
        sel = X[cl == k]
        if len(sel):
            sums[k] += np.nansum(sel, axis=0)
            sumsq[k] += np.nansum(sel * sel, axis=0)
            cnts[k] += np.sum(~np.isnan(sel), axis=0)
    r = RNG.random(len(chunk))
    take = (cl == 1) | ((cl == 2) & (r < LIC_FRAC)) | ((cl == 3) & (r < UNK_FRAC))
    part = chunk.loc[take, ["txId"] + feature_cols].copy()
    part["cls"] = cl[take]
    sample_parts.append(part)

sample = pd.concat(sample_parts, ignore_index=True)
sample[feature_cols] = sample[feature_cols].astype(np.float32)
sample.to_pickle(PROC / "eda_sample.pkl")
LOCAL_COLS = [c for c in feature_cols if c.startswith("Local_feature_")]
AGG_ANON = [c for c in feature_cols if c.startswith("Aggregate_feature_")]
assert feature_cols == ["Time step"] + LOCAL_COLS + AGG_ANON + AGG_COLS
assert len(LOCAL_COLS) == 93 and len(AGG_ANON) == 72 and len(feature_cols) == 183
print(f"feature groups: Time step=1, local={len(LOCAL_COLS)}, aggregate*={len(AGG_ANON)}, named={len(AGG_COLS)}")
print(f"rows={n_feat_rows:,} cols={len(feature_cols)+1} (txId + 183 features)")
print("dtypes:", pd.Series(col_dtypes).value_counts().to_dict())
print(f"'Time step': min={ts_min:.0f} max={ts_max:.0f} nunique={len(ts_values)}")
print(f"duplicate txId in features: {dup_feat}")
print("plot sample:", sample.shape, "| per-class:", sample["cls"].value_counts().sort_index().to_dict())
"""))

cells.append(code("""
ITEMSIZE = {"int64": 8, "float64": 8, "int32": 4, "float32": 4}
feat_mem_mb = n_feat_rows * sum(ITEMSIZE.get(col_dtypes[c], 8) for c in ["txId"] + feature_cols) / 1e6
print("=== Schema ===")
print(f"txs_features.csv : {n_feat_rows:,} rows x {len(feature_cols)+1} cols "
      "[txId(int64), Time step(int64), 93 x Local_feature_k (float64, standardized), "
      "72 x Aggregate_feature_k (float64), 17 x named stats (float64)]")
print(f"  estimated in-memory size: ~{feat_mem_mb:.0f} MB (on-disk: 694.8 MB CSV)")
print(f"txs_classes.csv  : {classes.shape[0]:,} rows x 2 cols [txId, class] "
      f"| mem={classes.memory_usage(deep=True).sum()/1e6:.2f} MB")
print(f"txs_edgelist.csv : {edgelist.shape[0]:,} rows x 2 cols [txId1 -> txId2] "
      f"| mem={edgelist.memory_usage(deep=True).sum()/1e6:.2f} MB")
"""))

# ------------------------------------------------- 2. ID consistency
cells.append(md("## 2. Transaction ID consistency and duplicates"))
cells.append(code("""
dup_classes = int(classes.duplicated("txId").sum())
dup_edges = int(edgelist.duplicated().sum())
feat_ids = seen_ids
class_ids = set(classes["txId"].tolist())
edge_endpoints = set(edgelist["txId1"].tolist()) | set(edgelist["txId2"].tolist())
dangling = edge_endpoints - feat_ids
isolated = feat_ids - edge_endpoints
print(f"duplicate txId: features={dup_feat}, classes={dup_classes}, duplicate edges={dup_edges}")
print(f"ID sets equal (features vs classes): {feat_ids == class_ids} "
      f"(|features|={len(feat_ids):,}, |classes|={len(class_ids):,})")
print(f"edge endpoints: {len(edge_endpoints):,} unique | dangling endpoints: {len(dangling)}")
print(f"transactions with zero edges (isolated nodes): {len(isolated):,} "
      f"({100*len(isolated)/len(feat_ids):.2f}%)")
"""))

# ------------------------------------------------- 3. feature groups
cells.append(md("## 3. Feature groups and statistics"))
cells.append(code("""
mean = {k: sums[k] / np.maximum(cnts[k], 1) for k in (1, 2, 3)}
var = {k: np.maximum(sumsq[k] / np.maximum(cnts[k], 1) - mean[k] ** 2, 0) for k in (1, 2, 3)}
tot_c = cnts[1] + cnts[2] + cnts[3]
overall_mean = (sums[1] + sums[2] + sums[3]) / np.maximum(tot_c, 1)
overall_var = np.maximum((sumsq[1] + sumsq[2] + sumsq[3]) / np.maximum(tot_c, 1) - overall_mean ** 2, 0)
overall_std = np.sqrt(overall_var)

groups = {"Time step": ["Time step"], "local (93)": LOCAL_COLS,
        "aggregate* (72)": AGG_ANON, "named stats (17)": AGG_COLS}
print(f"{'group':<20}{'n_feat':>7} {'mean_of_means':>14} {'mean_std':>10} {'max_std':>10}")
for gname, gcols in groups.items():
    idx = [feature_cols.index(c) for c in gcols]
    print(f"{gname:<20}{len(idx):>7} {overall_mean[idx].mean():>14.4f} "
          f"{overall_std[idx].mean():>10.4f} {overall_std[idx].max():>10.4f}")

top_std = np.argsort(overall_std)[::-1][:10]
print("\\ntop-10 features by overall std:")
for i in top_std:
    g = ("named" if feature_cols[i] in AGG_COLS
         else "agg*" if feature_cols[i] in AGG_ANON
         else "time" if feature_cols[i] == "Time step" else "local")
    print(f"  {feature_cols[i]:<22} [{g:>5}] mean={overall_mean[i]:12.4f} std={overall_std[i]:12.4f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
li = [feature_cols.index(c) for c in LOCAL_COLS]
ax[0].hist(overall_mean[li], bins=50, color="#1f77b4", edgecolor="white")
ax[0].axvline(0, color="k", ls="--", lw=1)
ax[0].set_title("Local features: distribution of per-feature means")
ax[0].set_xlabel("mean"); ax[0].set_ylabel("count")
ax[1].hist(overall_std[li], bins=50, color="#ff7f0e", edgecolor="white")
ax[1].axvline(1, color="k", ls="--", lw=1)
ax[1].set_title("Local features: distribution of per-feature stds")
ax[1].set_xlabel("std"); ax[1].set_ylabel("count")
fig.suptitle("Standardization check — 93 anonymized local features")
fig.savefig(FIG / "eda_feature_group_scales.png"); plt.close(fig)
print("saved eda_feature_group_scales.png")
"""))

# ------------------------------------------------- 4. missing values
cells.append(md("## 4. Missing values and their distribution"))
cells.append(code("""
miss_cols = missing[missing > 0]
print(f"columns with any missing: {len(miss_cols)} / {len(feature_cols)}")
print(f"total missing cells: {int(miss_cols.sum()):,} "
      f"({100*miss_cols.sum()/(n_feat_rows*len(feature_cols)):.3f}% of all feature cells)")
assert (miss_cols == 965).all() and set(miss_cols.index) == set(AGG_COLS)

agg = pd.read_csv(FEAT_PATH, usecols=["txId"] + AGG_COLS)  # light second pass, 18 cols
any_miss = agg[AGG_COLS].isna().any(axis=1)
all_miss = agg[AGG_COLS].isna().all(axis=1)
print(f"rows with >=1 missing agg value: {int(any_miss.sum()):,}")
print(f"rows with all 17 agg missing   : {int(all_miss.sum()):,}  "
      f"-> co-missing block: {bool((any_miss == all_miss).all())}")
miss_cls = agg.loc[all_miss, "txId"].map(cls_map).value_counts().sort_index()
print("class distribution of the 965 all-missing rows vs overall:")
for k in (1, 2, 3):
    print(f"  {CLASS_NAMES[k]:<8}: {miss_cls.get(k, 0):>5} rows "
          f"({100*miss_cls.get(k, 0)/965:.2f}% of missing rows vs "
          f"{100*(classes['class'] == k).mean():.2f}% overall)")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].bar(range(len(AGG_COLS)), [miss_cols[c] for c in AGG_COLS], color="#9467bd")
ax[0].set_xticks(range(len(AGG_COLS)))
ax[0].set_xticklabels(AGG_COLS, rotation=60, ha="right", fontsize=8)
ax[0].set_title("Missing values per aggregate column (all 965)")
ax[0].set_ylabel("rows missing")
ax[1].bar([CLASS_NAMES[k] for k in (1, 2, 3)], [miss_cls.get(k, 0) for k in (1, 2, 3)],
           color=[CLASS_COLORS[k] for k in (1, 2, 3)])
ax[1].set_title("Class of the 965 all-missing rows")
ax[1].set_ylabel("rows")
fig.suptitle("Missing values live only in the 17 named statistical columns")
fig.savefig(FIG / "eda_missing_values.png"); plt.close(fig)
print("saved eda_missing_values.png")
"""))

# ------------------------------------------------- 5/6. labels + imbalance
cells.append(md("## 5 & 6. Labels: illicit / licit / unknown, and class imbalance"))
cells.append(code("""
counts = classes["class"].value_counts().sort_index()
total = len(classes)
print("label distribution:")
for k in (1, 2, 3):
    print(f"  class {k} ({CLASS_NAMES[k]:<8}): {counts[k]:>7,}  ({100*counts[k]/total:5.2f}%)")
print(f"\\nlicit:illicit ratio = {counts[2]/counts[1]:.2f} : 1")
print(f"labeled (illicit+licit) share = {100*(counts[1]+counts[2])/total:.2f}%")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
labels = [f"{k}\\n{CLASS_NAMES[k]}" for k in (1, 2, 3)]
vals = [counts[k] for k in (1, 2, 3)]
cols = [CLASS_COLORS[k] for k in (1, 2, 3)]
ax[0].bar(labels, vals, color=cols); ax[0].set_title("Class counts (linear)"); ax[0].set_ylabel("transactions")
for x, v in zip(labels, vals):
    ax[0].text(x, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
ax[1].bar(labels, vals, color=cols); ax[1].set_yscale("log"); ax[1].set_title("Class counts (log scale)")
ax[1].set_ylabel("transactions (log)")
fig.suptitle("Severe class imbalance; 77% of transactions are unlabeled (unknown)")
fig.savefig(FIG / "eda_class_distribution.png"); plt.close(fig)
print("saved eda_class_distribution.png")
"""))

# ------------------------------------------------- 7. time steps
cells.append(md("## 7. Transactions and labels across all time steps"))
cells.append(code("""
ts = pd.read_csv(FEAT_PATH, usecols=["txId", "Time step"])  # light pass: 2 columns
ts["cls"] = ts["txId"].map(cls_map)
ct = pd.crosstab(ts["Time step"], ts["cls"]).reindex(columns=[1, 2, 3], fill_value=0).reindex(range(1, 50), fill_value=0)
assert (ct.sum(axis=1) > 0).all(), "every time step must contain transactions"
illicit_rate = ct[1] / ct.sum(axis=1)
print(f"time steps covered: {ct.index.min()}-{ct.index.max()} ({len(ct)} steps, all non-empty)")
print(f"txs per step: min={ct.sum(axis=1).min():,} max={ct.sum(axis=1).max():,}")
print(f"steps with zero illicit txs: {int((ct[1] == 0).sum())}")
print(f"illicit rate per step: min={illicit_rate.min():.4f} max={illicit_rate.max():.4f} "
      f"corr(step, illicit_rate)={np.corrcoef(ct.index, illicit_rate)[0, 1]:.3f}")

fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
bottom = np.zeros(len(ct))
for k in (1, 2, 3):
    share = ct[k] / ct.sum(axis=1)
    ax[0].bar(ct.index, share, bottom=bottom, color=CLASS_COLORS[k],
              label=f"{k}={CLASS_NAMES[k]}", width=0.9)
    bottom += share
ax[0].set_ylabel("class share"); ax[0].set_title("Label composition per time step"); ax[0].legend(ncol=3)
ax[1].bar(ct.index, ct.sum(axis=1), color="#1f77b4", width=0.9, label="tx count")
ax[1].set_ylabel("transactions"); ax[1].set_xlabel("time step")
ax1b = ax[1].twinx()
ax1b.plot(ct.index, illicit_rate, color="#d62728", marker="o", ms=3, label="illicit rate")
ax1b.set_ylabel("illicit rate", color="#d62728")
fig.suptitle("Temporal coverage: all 49 time steps non-empty; label mix varies over time")
fig.savefig(FIG / "eda_labels_over_time.png"); plt.close(fig)
print("saved eda_labels_over_time.png")
"""))

# ------------------------------------------------- 8. graph structure
cells.append(md("## 8. Graph structure: edges, nodes, self-loops, duplicates, dangling IDs"))
cells.append(code("""
nodes = np.array(sorted(feat_ids))
nidx = {n: i for i, n in enumerate(nodes)}
src = edgelist["txId1"].map(nidx).to_numpy()
dst = edgelist["txId2"].map(nidx).to_numpy()
n = len(nodes); m = len(edgelist)
self_loops = int((edgelist["txId1"] == edgelist["txId2"]).sum())
print(f"nodes={n:,} edges={m:,} (directed money-flow, txId1 -> txId2)")
print(f"self-loops={self_loops} | duplicate edges={dup_edges} | dangling endpoints={len(dangling)}")
print(f"density = {m/(n*(n-1)):.3e} | avg out-degree = {m/n:.3f}")
# Temporal direction check: money should flow forward in time (inputs older than outputs).
step_of = dict(zip(ts["txId"], ts["Time step"]))
s1 = edgelist["txId1"].map(step_of).to_numpy()
s2 = edgelist["txId2"].map(step_of).to_numpy()
fwd = float(np.mean(s2 >= s1)); same = float(np.mean(s2 == s1))
cross = int((s1 != s2).sum())
assert cross == 0 and fwd == 1.0
print(f"edges with dst_step >= src_step: {100*fwd:.2f}% (of which same-step: {100*same:.2f}%)")
print(f"cross-step edges (exact count): {cross} — every edge is intra-step")
"""))

# ------------------------------------------------- 9. degrees
cells.append(md("## 9. In/out/total degree distributions"))
cells.append(code("""
in_deg = np.bincount(dst, minlength=n)
out_deg = np.bincount(src, minlength=n)
tot_deg = in_deg + out_deg
for name, d in [("in", in_deg), ("out", out_deg), ("total", tot_deg)]:
    print(f"{name:>5}-degree: mean={d.mean():6.2f} median={np.median(d):5.0f} max={d.max():6d} "
          f"| zeros={100*np.mean(d==0):5.2f}% | >=10: {100*np.mean(d>=10):5.2f}%")

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for a, (name, d) in zip(ax, [("in", in_deg), ("out", out_deg), ("total", tot_deg)]):
    bins = np.logspace(0, np.log10(d.max() + 1), 30)
    hist_c, edges_ = np.histogram(d[d > 0], bins=bins)
    ctr = np.sqrt(edges_[:-1] * edges_[1:])
    a.loglog(ctr, hist_c, "o-", ms=4)
    a.set_title(f"{name}-degree (log-log)"); a.set_xlabel("degree"); a.set_ylabel("nodes")
    a.grid(True, which="both", ls=":", alpha=0.4)
fig.suptitle("Heavy-tailed degree distributions (zero-degree mass excluded from plot, reported above)")
fig.savefig(FIG / "eda_degree_distributions.png"); plt.close(fig)
print("saved eda_degree_distributions.png")
"""))

# ------------------------------------------------- 10. components
cells.append(md("## 10. Connected components"))
cells.append(code("""
A = csr_matrix((np.ones(m), (src, dst)), shape=(n, n))
n_comp, comp = connected_components(A, directed=True, connection="weak")
sizes = np.bincount(comp)
order = np.argsort(sizes)[::-1]
sizes = sizes[order]
largest_label = order[0]
mask_largest = comp == largest_label
print(f"weakly connected components: {n_comp:,}")
print(f"largest component: {sizes[0]:,} nodes ({100*sizes[0]/n:.2f}% of all txs)")
print(f"singleton components: {int(np.sum(sizes == 1)):,} | components with <10 nodes: {int(np.sum(sizes < 10)):,}")
# Do components align with time steps? (All edges are intra-step, so each component
# should sit inside a single time step.)
node_step = np.array([step_of[v] for v in nodes])
multi_step = sum(len(np.unique(node_step[comp == c])) != 1 for c in range(n_comp))
print(f"components spanning more than one time step: {multi_step}")
assert multi_step == 0 and n_comp == len(ts_values)
print(f"=> the {n_comp} weak components map 1:1 onto the {len(ts_values)} time steps (verified exactly)")
node_cls = np.array([cls_map[v] for v in nodes])
for k in (1, 2, 3):
    kk = node_cls == k
    print(f"  {CLASS_NAMES[k]:<8}: {int((mask_largest & kk).sum()):>7,} / {int(kk.sum()):<7,} in largest "
          f"({100*(mask_largest & kk).sum()/max(kk.sum(), 1):.1f}%)")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].loglog(np.arange(1, len(sizes) + 1), sizes, ".", ms=3)
ax[0].set_title("Component sizes (rank-ordered, log-log)")
ax[0].set_xlabel("rank"); ax[0].set_ylabel("size")
ax[0].grid(True, which="both", ls=":", alpha=0.4)
ax[1].hist(np.log10(sizes[sizes > 1]), bins=30, color="#2ca02c", edgecolor="white")
ax[1].set_title("Size distribution (non-singletons, log10)")
ax[1].set_xlabel("log10(size)"); ax[1].set_ylabel("components")
fig.suptitle(f"{n_comp:,} weakly connected components")
fig.savefig(FIG / "eda_connected_components.png"); plt.close(fig)
print("saved eda_connected_components.png")
"""))

# ------------------------------------------------- 11. feature distributions
cells.append(md("## 11. Important feature distributions: illicit vs licit"))
cells.append(code("""
# Standardized mean difference (Cohen's d) between illicit (1) and licit (2), per feature.
d = {}
for i, c in enumerate(feature_cols):
    if c == "Time step":
        continue
    s1, s2 = np.sqrt(var[1][i]), np.sqrt(var[2][i])
    pooled = np.sqrt((s1 ** 2 + s2 ** 2) / 2)
    d[c] = (mean[1][i] - mean[2][i]) / pooled if pooled > 0 else 0.0
ranked = sorted(d.items(), key=lambda kv: -abs(kv[1]))[:6]
print("top-6 discriminative features, illicit vs licit (|Cohen's d|):")
for c, v in ranked:
    g = "named" if c in AGG_COLS else ("agg*" if c in AGG_ANON else "local")
    i = feature_cols.index(c)
    print(f"  {c:<22} [{g:>5}] d={v:+.3f}  mean_illicit={mean[1][i]:+.3f}  mean_licit={mean[2][i]:+.3f}")

fig, ax = plt.subplots(2, 3, figsize=(13, 7))
for a, (c, v) in zip(ax.ravel(), ranked):
    g = ("named stat" if c in AGG_COLS
         else "aggregate* (anonymized)" if c in AGG_ANON else "local (anonymized)")
    for k, lbl in [(1, "illicit"), (2, "licit")]:
        vals = sample.loc[sample["cls"] == k, c].dropna().to_numpy()
        a.hist(vals, bins=50, alpha=0.55, color=CLASS_COLORS[k], label=lbl, log=True)
    a.set_title(f"{c}\\n[{g}] d={v:+.2f}", fontsize=10)
    a.set_yscale("log")
handles, labels_ = ax[0, 0].get_legend_handles_labels()
fig.legend(handles, labels_, loc="upper right")
fig.suptitle("Illicit vs licit distributions of the 6 most discriminative features\\n"
             "(local feature semantics are anonymized in the official release — names only, no interpretation)")
fig.savefig(FIG / "eda_feature_distributions.png"); plt.close(fig)
print("saved eda_feature_distributions.png")
"""))

# ------------------------------------------------- 12. correlation
cells.append(md("## 12. Correlation analysis"))
cells.append(code("""
Xc = sample[feature_cols].to_numpy(dtype=np.float64)
Xc = Xc - np.nanmean(Xc, axis=0)
Xc = np.where(np.isnan(Xc), 0, Xc)  # mean-impute: only the 965 agg-missing rows affected
C = np.corrcoef(Xc, rowvar=False)
np.fill_diagonal(C, 0)

def block_mean_abs(r0, r1, c0, c1):
    b = np.abs(C[r0:r1, c0:c1])
    if r0 == c0 and r1 == c1:
        iu = np.triu_indices(b.shape[0], 1)
        return float(b[iu].mean())
    return float(b.mean())

# feature_cols order: [Time step] + 93 local + 72 aggregate* + 17 named
print("mean |correlation|:")
print(f"  local-local   : {block_mean_abs(1, 94, 1, 94):.3f}")
print(f"  aggregate*-a*: {block_mean_abs(94, 166, 94, 166):.3f}")
print(f"  named-named   : {block_mean_abs(166, 183, 166, 183):.3f}")
print(f"  local-agg*    : {block_mean_abs(1, 94, 94, 166):.3f}")
print(f"  local-named   : {block_mean_abs(1, 94, 166, 183):.3f}")
print(f"  agg*-named    : {block_mean_abs(94, 166, 166, 183):.3f}")
print(f"  time-vs-rest  : {float(np.abs(C[0, 1:]).mean()):.3f}")
iu = np.triu_indices(183, 1)
top = np.argsort(np.abs(C[iu]))[::-1][:5]
print("top-5 |r| pairs:")
for a_, b_ in zip(iu[0][top], iu[1][top]):
    print(f"  {feature_cols[a_]:<22} x {feature_cols[b_]:<22} r={C[a_, b_]:+.3f}")

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(C, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
for b_ in (1, 94, 166):
    ax.axhline(b_ - 0.5, color="k", lw=1.2)
    ax.axvline(b_ - 0.5, color="k", lw=1.2)
ax.set_xticks([0, 47, 130, 174])
ax.set_xticklabels(["Time step", "local (93)", "aggregate* (72)", "named (17)"], fontsize=9)
ax.set_yticks([0, 47, 130, 174])
ax.set_yticklabels(["Time step", "local (93)", "aggregate* (72)", "named (17)"], fontsize=9)
ax.set_title(f"Feature correlation matrix (stratified sample, n={len(sample):,})")
fig.colorbar(im, ax=ax, label="Pearson r")
fig.savefig(FIG / "eda_correlation.png"); plt.close(fig)
print("saved eda_correlation.png")
"""))

# ------------------------------------------------- 13. leakage
cells.append(md("""
## 13. Potential data leakage — investigation

Checks below are evidence-gathering only. A check "passing" means no *observable*
leakage in the released CSVs; dataset-construction choices (e.g. how the aggregate
features — the 72 anonymized aggregates and 17 named statistics — were computed over
neighborhoods and time) cannot be verified from the CSVs
alone and are flagged as open questions for Phase 3.
"""))
cells.append(code("""
print("=== Leakage checks (evidence only; see results/leakage_analysis.md) ===")
# (a) feature <-> label association ceiling
y_ill = (sample["cls"] == 1).to_numpy(float)
y_lic = (sample["cls"] == 2).to_numpy(float)
max_ill, max_lic, arg_ill, arg_lic = 0.0, 0.0, None, None
for i, c in enumerate(feature_cols):
    v = Xc[:, i]
    r_ill = np.corrcoef(v, y_ill)[0, 1]
    r_lic = np.corrcoef(v, y_lic)[0, 1]
    if abs(r_ill) > max_ill:
        max_ill, arg_ill = abs(r_ill), c
    if abs(r_lic) > max_lic:
        max_lic, arg_lic = abs(r_lic), c
print(f"(a) max |r(feature, illicit-indicator)| = {max_ill:.3f} [{arg_ill}]")
print(f"    max |r(feature, licit-indicator)|   = {max_lic:.3f} [{arg_lic}]")
# (b) near-duplicate columns
nd = int((((np.abs(C) > 0.999).sum(axis=1) - 1) > 0).sum())
print(f"(b) features with a near-duplicate (|r|>0.999): {nd}")
# (c) zero-variance features
zv = [feature_cols[i] for i in range(183) if overall_std[i] == 0]
print(f"(c) zero-variance features: {len(zv)} {zv}")
# (d) temporal shift of the label prior (affects splitting, not leakage per se)
print(f"(d) corr(time step, per-step illicit rate) = {np.corrcoef(ct.index, illicit_rate)[0, 1]:+.3f}")
# (e) named-stat missingness by class (from §4)
print("(e) named-stat-missing rows by class: " +
      ", ".join(f"{CLASS_NAMES[k]}={miss_cls.get(k, 0)}" for k in (1, 2, 3)))
"""))

# ------------------------------------------------- 14. homophily
cells.append(md("""
## 14. Preliminary graph homophily

**Metric definition.** Let the transaction graph be directed, $G=(V,E)$, with money flowing
$u \\to v$, and labels $y_v \\in \\{1{=}\\text{illicit},\\ 2{=}\\text{licit},\\ 3{=}\\text{unknown}\\}$.

- **Labeled directed edge homophily**
  $$h_{lab} = \\frac{|\\{(u,v) \\in E : y_u, y_v \\in \\{1,2\\},\\ y_u = y_v\\}|}
                   {|\\{(u,v) \\in E : y_u, y_v \\in \\{1,2\\}\\}|}$$
  the fraction of edges whose *both* endpoints are labeled that connect same-class endpoints.
- **3-class homophily** $h_{all} = |\\{(u,v)\\in E : y_u=y_v\\}| / |E|$, treating *unknown*
  as a third class (reported for completeness; it is dominated by unknown–unknown edges).
- **Per-class conditional homophily** $h_{c\\to c} = P(y_v = c \\mid y_u = c)$ over directed
  edges out of class $c$ — e.g. how often an illicit transaction's money flows to another
  illicit transaction.
- **Random-mixing baseline**: with $p_c$ the fraction of labeled edge-endpoints in class $c$,
  $h_{base} = \\sum_c p_c^2$ is the homophily expected if edges connected labeled endpoints
  at random. Comparing $h_{lab}$ against $h_{base}$ tells whether same-class linkage exceeds chance.
"""))
cells.append(code("""
node_cls = np.array([cls_map[v] for v in nodes])
sc, dc = node_cls[src], node_cls[dst]
ct_e = pd.crosstab(pd.Series(sc, name="src"), pd.Series(dc, name="dst"))
ct_e = ct_e.reindex(index=[1, 2, 3], columns=[1, 2, 3], fill_value=0)
print("directed edge-type counts [src x dst]:")
print(ct_e.to_string())

lab = (sc < 3) & (dc < 3)
h_lab = float(np.mean(sc[lab] == dc[lab]))
h_all = float(np.mean(sc == dc))
lab_endpoints = np.concatenate([sc[lab], dc[lab]])
p_ill_end = float(np.mean(lab_endpoints == 1)); p_lic_end = float(np.mean(lab_endpoints == 2))
etc = {str(r): {str(c): int(ct_e.loc[r, c]) for c in (1, 2, 3)} for r in (1, 2, 3)}  # src -> dst
p_ill_given_src_ill = etc["1"]["1"] / sum(etc["1"].values())
p_lic_given_src_lic = etc["2"]["2"] / sum(etc["2"].values())
h_base = float(p_ill_end ** 2 + p_lic_end ** 2)
print(f"\\nlabeled directed edge homophily h_lab = {h_lab:.4f}  (both-ends-labeled edges: {lab.sum():,})")
print(f"3-class homophily h_all = {h_all:.4f}  (unknown treated as a class)")
print(f"random-mixing baseline (labeled) = {h_base:.4f}")
for k in (1, 2):
    kk = sc == k
    print(f"P(dst={CLASS_NAMES[k]} | src={CLASS_NAMES[k]}) = {np.mean(dc[kk] == k):.4f}  "
          f"(n_src_edges={kk.sum():,})")
print(f"P(dst=licit | src=illicit) = {np.mean(dc[sc == 1] == 2):.4f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
prop = ct_e / ct_e.values.sum()
im = ax[0].imshow(prop.values, norm=LogNorm(), cmap="YlOrRd", aspect="auto")
for i in range(3):
    for j in range(3):
        ax[0].text(j, i, f"{100*prop.values[i, j]:.1f}%", ha="center", va="center", fontsize=10)
ax[0].set_xticks([0, 1, 2]); ax[0].set_xticklabels(["illicit", "licit", "unknown"])
ax[0].set_yticks([0, 1, 2]); ax[0].set_yticklabels(["illicit", "licit", "unknown"])
ax[0].set_xlabel("dst class"); ax[0].set_ylabel("src class")
ax[0].set_title("Directed edge-type proportions (log color)")
ax[1].bar(["h_lab\\n(observed)", "random-mixing\\nbaseline", "h_all\\n(3-class)"],
          [h_lab, h_base, h_all], color=["#1f77b4", "#7f7f7f", "#ff7f0e"])
ax[1].set_ylim(0, 1); ax[1].set_ylabel("homophily")
ax[1].set_title("Homophily vs. chance")
fig.suptitle("Preliminary graph homophily (directed money-flow edges)")
fig.savefig(FIG / "eda_homophily.png"); plt.close(fig)
print("saved eda_homophily.png")
"""))

# ------------------------------------------------- 15. save stats
cells.append(code("""
def _j(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return o

STATS = {
    "n_txs": int(n_feat_rows), "n_features": 183, "n_edges": int(m),
    "classes": {CLASS_NAMES[k]: int(counts[k]) for k in (1, 2, 3)},
    "time_steps": [1, 49],
    "dup_txid_features": int(dup_feat), "dup_txid_classes": int(dup_classes),
    "dup_edges": int(dup_edges), "dangling_endpoints": int(len(dangling)),
    "isolated_nodes": int(len(isolated)), "self_loops": int(self_loops),
    "missing": {"n_cols": 17, "rows_all_missing": int(all_miss.sum()), "cells": 17 * 965,
                "by_class": {CLASS_NAMES[k]: int(miss_cls.get(k, 0)) for k in (1, 2, 3)}},
    "degrees": {name: {"mean": float(d.mean()), "max": int(d.max()),
                       "zero_pct": float(100 * np.mean(d == 0))}
                for name, d in [("in", in_deg), ("out", out_deg), ("total", tot_deg)]},
    "components": {"n_weak": int(n_comp), "largest": int(sizes[0]),
                   "largest_pct": float(100 * sizes[0] / n),
                   "singletons": int(np.sum(sizes == 1)),
                   "align_with_time_steps": bool(multi_step == 0 and n_comp == len(ts_values))},
    "homophily": {"h_lab": h_lab, "h_all": h_all, "baseline": h_base,
                  "p_illicit_given_src_illicit": p_ill_given_src_ill,
                  "p_licit_given_src_illicit": etc["1"]["2"] / sum(etc["1"].values()),
                  "p_licit_given_src_licit": p_lic_given_src_lic,
                  "p_illicit_endpoint": p_ill_end, "p_licit_endpoint": p_lic_end,
                  "illicit_enrichment": p_ill_given_src_ill / p_ill_end if p_ill_end else 0.0,
                  "licit_enrichment": p_lic_given_src_lic / p_lic_end if p_lic_end else 0.0,
                  "edge_type_counts": etc},
    "edge_forward_time_pct": float(100 * fwd),
    "cross_step_edges": int(cross),
    "illicit_rate_corr_time": float(np.corrcoef(ct.index, illicit_rate)[0, 1]),
    "max_abs_corr_feature_illicit": float(max_ill),
    "max_abs_corr_feature_illicit_col": str(arg_ill),
    "max_abs_corr_feature_licit": float(max_lic),
    "max_abs_corr_feature_licit_col": str(arg_lic),
    "near_duplicate_features": int(nd),
    "zero_variance_features": [str(z) for z in zv],
    "top_discriminative": [{"feature": c, "cohens_d": float(v)} for c, v in ranked],
}
(RES / "eda_stats.json").write_text(json.dumps(STATS, indent=2, default=_j))
print("wrote results/eda_stats.json")
"""))

cells.append(md("""
## Key takeaways

Full write-up with computed statistics, limitations and Phase-3 recommendations:
- `../results/eda_summary.md`
- `../results/leakage_analysis.md`

**Next: Phase 3 — graph construction and temporal splitting** (not started; no modeling done here).
"""))

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata.language_info = {"name": "python", "version": "3.10"}
nbf.write(nb, NB_PATH)
print(f"wrote {NB_PATH} with {len(cells)} cells")
