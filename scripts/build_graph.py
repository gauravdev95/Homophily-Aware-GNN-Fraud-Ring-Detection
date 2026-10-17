"""Phase 3 / Step 3 — Graph construction & temporal split.

Builds the directed transaction -> transaction money-flow graph from the raw
Elliptic++ CSVs (read-only; raw files are never modified), validates its
structure, and defines temporal train/val/test splits.

Splits follow the EDA recommendation and the original Elliptic protocol family:
    train = timesteps  1-34
    val   = timesteps 35-41
    test  = timesteps 42-49
No random splitting — evaluation must measure generalization to future
transactions. All 234,355 edges are intra-timestep (verified), so no edge
crosses a split boundary: splits are graph-clean by construction.

Artifacts (data/processed/):
    node_splits.csv   txId, timestep, split, class   (one row per node)
    graph_splits.json split definitions + graph statistics

Reports (results/):
    graph_stats.json
    graph_construction_summary.md
    figures/graph/*.png

Only stdlib + pandas + numpy + scipy + matplotlib (no networkx needed).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "elliptic_pp"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIG_DIR = RESULTS / "figures" / "graph"

SPLITS = {"train": (1, 34), "val": (35, 41), "test": (42, 49)}
CLASS_NAMES = {1: "illicit", 2: "licit", 3: "unknown"}


# ---------------------------------------------------------------- loading

def load_data():
    """Load raw CSVs. Returns (classes, timesteps, edges) as DataFrames."""
    classes = pd.read_csv(RAW / "txs_classes.csv")          # txId, class
    feats = pd.read_csv(RAW / "txs_features.csv",
                        usecols=["txId", "Time step"])       # txId, timestep
    edges = pd.read_csv(RAW / "txs_edgelist.csv")            # txId1, txId2
    return classes, feats, edges


# ---------------------------------------------------------------- graph

class UnionFind:
    """Iterative union-find with path compression (weak components)."""

    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def build_graph(classes: pd.DataFrame, feats: pd.DataFrame,
                edges: pd.DataFrame) -> dict:
    """Build the directed graph and compute structural statistics."""
    tx_ids = classes["txId"].to_numpy()
    n = len(tx_ids)
    index = {t: i for i, t in enumerate(tx_ids)}

    src = edges["txId1"].map(index).to_numpy()
    dst = edges["txId2"].map(index).to_numpy()
    adj = sparse.csr_matrix((np.ones(len(src), dtype=np.int8), (src, dst)),
                            shape=(n, n))

    step_of = feats.set_index("txId")["Time step"].reindex(tx_ids).to_numpy()

    # weak components via union-find (treat edges as undirected)
    uf = UnionFind(n)
    for a, b in zip(src.tolist(), dst.tolist()):
        uf.union(a, b)
    roots = np.array([uf.find(i) for i in range(n)], dtype=np.int64)
    _, comp_ids = np.unique(roots, return_inverse=True)
    n_comp = int(comp_ids.max()) + 1

    # per-component timestep check: every component must live in one step
    comp_step = np.full(n_comp, -1, dtype=np.int64)
    comp_ok = True
    for node, c in enumerate(comp_ids):
        s = step_of[node]
        if comp_step[c] == -1:
            comp_step[c] = s
        elif comp_step[c] != s:
            comp_ok = False
            break

    in_deg = np.asarray(adj.sum(axis=0)).ravel()
    out_deg = np.asarray(adj.sum(axis=1)).ravel()

    return {
        "n_nodes": n,
        "n_edges": int(len(src)),
        "adj": adj,
        "tx_ids": tx_ids,
        "step_of": step_of,
        "src": src,
        "dst": dst,
        "n_components": n_comp,
        "comp_ids": comp_ids,
        "components_single_step": bool(comp_ok),
        "in_deg": in_deg,
        "out_deg": out_deg,
        "density": float(len(src) / (n * (n - 1))),
    }


def validate(g: dict) -> dict:
    """Structural validation checks. Returns {name: (passed, detail)}."""
    cross = int((g["step_of"][g["src"]] != g["step_of"][g["dst"]]).sum())
    self_loops = int((g["src"] == g["dst"]).sum())
    checks = {
        "no_cross_step_edges": (cross == 0, f"{cross} cross-step edges"),
        "components_match_timesteps": (
            g["n_components"] == 49 and g["components_single_step"],
            f"{g['n_components']} components, 1:1 with 49 timesteps: "
            f"{g['components_single_step']}"),
        "no_self_loops": (self_loops == 0, f"{self_loops} self-loops"),
        "no_isolated_nodes": (
            bool(((g["in_deg"] + g["out_deg"]) > 0).all()),
            f"{int(((g['in_deg'] + g['out_deg']) == 0).sum())} isolated nodes"),
    }
    return checks


# ---------------------------------------------------------------- split

def make_splits(classes: pd.DataFrame, step_of: np.ndarray,
                tx_ids: np.ndarray) -> pd.DataFrame:
    """Temporal node split assignment. Returns node_splits DataFrame."""
    steps = pd.Series(step_of, index=tx_ids, name="timestep")
    split = pd.Series(index=tx_ids, dtype="object")
    for name, (lo, hi) in SPLITS.items():
        mask = (steps.values >= lo) & (steps.values <= hi)
        split.iloc[mask] = name
    node_splits = pd.DataFrame({
        "txId": tx_ids,
        "timestep": steps.values,
        "split": split.values,
        "class": classes.set_index("txId")["class"].reindex(tx_ids).values,
    })
    assert node_splits["split"].notna().all(), "unassigned nodes!"
    return node_splits


def split_stats(node_splits: pd.DataFrame, g: dict) -> dict:
    """Per-split node/edge/label statistics."""
    step_of = g["step_of"]
    out = {}
    for name in ("train", "val", "test"):
        nodes = node_splits[node_splits["split"] == name]
        lo, hi = SPLITS[name]
        in_split = np.isin(step_of[g["src"]], np.arange(lo, hi + 1))
        # every edge is intra-timestep, so src-in-split <=> edge-in-split
        n_edges = int(in_split.sum())
        labels = nodes["class"].value_counts()
        out[name] = {
            "timesteps": f"{SPLITS[name][0]}-{SPLITS[name][1]}",
            "n_nodes": int(len(nodes)),
            "n_edges": n_edges,
            "n_illicit": int(labels.get(1, 0)),
            "n_licit": int(labels.get(2, 0)),
            "n_unknown": int(labels.get(3, 0)),
        }
    return out


# ---------------------------------------------------------------- figures

def make_figures(node_splits: pd.DataFrame, stats: dict) -> list[str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    order = ["train", "val", "test"]
    colors = {"train": "#4C78A8", "val": "#F58518", "test": "#54A24B"}

    # 1. nodes & edges per split
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(order))
    nodes = [stats[s]["n_nodes"] for s in order]
    edges = [stats[s]["n_edges"] for s in order]
    ax.bar(x - 0.2, nodes, 0.4, label="nodes",
           color=[colors[s] for s in order])
    ax.bar(x + 0.2, edges, 0.4, label="edges",
           color=[colors[s] for s in order], alpha=0.55, hatch="//")
    for i, s in enumerate(order):
        ax.text(i - 0.2, nodes[i], f"{nodes[i]:,}", ha="center", va="bottom",
                fontsize=9)
        ax.text(i + 0.2, edges[i], f"{edges[i]:,}", ha="center", va="bottom",
                fontsize=9)
    ax.set_xticks(x, [f"{s}\n(steps {stats[s]['timesteps']})" for s in order])
    ax.set_title("Graph size per temporal split")
    ax.legend()
    fig.tight_layout()
    p = FIG_DIR / "graph_split_sizes.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p))

    # 2. label distribution per split (stacked)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(order))
    for c, col in [(3, "#9E9E9E"), (2, "#4C78A8"), (1, "#E45756")]:
        vals = np.array([stats[s][f"n_{CLASS_NAMES[c]}"] for s in order])
        ax.bar(order, vals, bottom=bottom, label=CLASS_NAMES[c], color=col)
        bottom += vals
    for i, s in enumerate(order):
        ax.text(i, bottom[i], f"{int(bottom[i]):,}", ha="center", va="bottom",
                fontsize=9)
    ax.set_title("Label distribution per split (unknowns kept)")
    ax.legend(title="class")
    fig.tight_layout()
    p = FIG_DIR / "graph_labels_per_split.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p))

    # 3. nodes per timestep, colored by split
    per_step = node_splits.groupby(["timestep", "split"]).size().unstack(
        fill_value=0).reindex(columns=order, fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    bottom = np.zeros(len(per_step))
    for s in order:
        vals = per_step[s].to_numpy()
        ax.bar(per_step.index, vals, bottom=bottom, label=s,
               color=colors[s], width=0.9)
        bottom += vals
    for lo, hi, s in [(1, 34, "train"), (35, 41, "val"), (42, 49, "test")]:
        ax.axvspan(lo - 0.5, hi + 0.5, color=colors[s], alpha=0.07)
    ax.set_xlabel("timestep")
    ax.set_ylabel("nodes")
    ax.set_title("Nodes per timestep by split (shaded regions)")
    ax.legend()
    fig.tight_layout()
    p = FIG_DIR / "graph_nodes_per_timestep.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p))

    return paths


# ---------------------------------------------------------------- reports

def write_reports(g: dict, checks: dict, stats: dict,
                  fig_paths: list[str]) -> None:
    graph_stats = {
        "n_nodes": g["n_nodes"],
        "n_edges": g["n_edges"],
        "density": g["density"],
        "n_weak_components": g["n_components"],
        "components_single_step": g["components_single_step"],
        "in_degree": {"mean": float(g["in_deg"].mean()),
                      "max": int(g["in_deg"].max())},
        "out_degree": {"mean": float(g["out_deg"].mean()),
                       "max": int(g["out_deg"].max())},
        "validation": {k: {"passed": v[0], "detail": v[1]}
                       for k, v in checks.items()},
        "splits": stats,
    }
    (RESULTS / "graph_stats.json").write_text(json.dumps(graph_stats, indent=2))

    lines = [
        "# Graph Construction Summary — Elliptic++ Transactions",
        "",
        "*Phase 3 / Step 3 · directed money-flow graph + temporal splits · "
        "all statistics computed, not estimated.*",
        "",
        "## 1. Graph",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Nodes (transactions) | {g['n_nodes']:,} |",
        f"| Directed edges (money flow) | {g['n_edges']:,} |",
        f"| Density | {g['density']:.3e} |",
        f"| Weakly connected components | {g['n_components']} |",
        f"| Mean / max in-degree | {g['in_deg'].mean():.2f} / {g['in_deg'].max()} |",
        f"| Mean / max out-degree | {g['out_deg'].mean():.2f} / {g['out_deg'].max()} |",
        "",
        "## 2. Validation",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    for name, (passed, detail) in checks.items():
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} | {detail} |")
    lines += [
        "",
        "## 3. Temporal splits (no random splitting)",
        "",
        "| Split | Timesteps | Nodes | Edges | Illicit | Licit | Unknown |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in ("train", "val", "test"):
        d = stats[s]
        lines.append(
            f"| {s} | {d['timesteps']} | {d['n_nodes']:,} | {d['n_edges']:,} | "
            f"{d['n_illicit']:,} | {d['n_licit']:,} | {d['n_unknown']:,} |")
    lines += [
        "",
        "All edges are intra-timestep (0 cross-step edges verified), so no edge "
        "crosses a split boundary — splits are graph-clean by construction.",
        "",
        "## 4. Artifacts",
        "",
        "- `data/processed/node_splits.csv` — txId, timestep, split, class",
        "- `data/processed/graph_splits.json` — split definitions + statistics",
        "- `results/graph_stats.json`, `results/graph_construction_summary.md`",
        "- Figures: `results/figures/graph/`",
        "",
        "## 5. Figures",
        "",
    ]
    for p in fig_paths:
        lines.append(f"- `{Path(p).name}`")
    lines.append("")
    (RESULTS / "graph_construction_summary.md").write_text("\n".join(lines))


# ---------------------------------------------------------------- main

def main() -> int:
    print("== loading raw data ==")
    classes, feats, edges = load_data()
    print(f"  nodes: {len(classes):,}, edges: {len(edges):,}")

    print("\n== building directed graph ==")
    g = build_graph(classes, feats, edges)
    print(f"  nodes={g['n_nodes']:,}, edges={g['n_edges']:,}, "
          f"density={g['density']:.3e}")
    print(f"  weak components={g['n_components']}")

    print("\n== validation ==")
    checks = validate(g)
    all_ok = True
    for name, (passed, detail) in checks.items():
        print(f"  [{ 'PASS' if passed else 'FAIL' }] {name}: {detail}")
        all_ok &= passed
    if not all_ok:
        print("\nRESULT: FAIL")
        return 1

    print("\n== temporal splits ==")
    node_splits = make_splits(classes, g["step_of"], g["tx_ids"])
    stats = split_stats(node_splits, g)
    for s in ("train", "val", "test"):
        d = stats[s]
        print(f"  {s:5s} steps {d['timesteps']:>5s}: "
              f"{d['n_nodes']:,} nodes, {d['n_edges']:,} edges "
              f"(illicit {d['n_illicit']:,}, licit {d['n_licit']:,}, "
              f"unknown {d['n_unknown']:,})")

    print("\n== saving artifacts ==")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    node_splits.to_csv(PROCESSED / "node_splits.csv", index=False)
    (PROCESSED / "graph_splits.json").write_text(json.dumps(
        {"splits": SPLITS, "stats": stats}, indent=2))
    print(f"  {PROCESSED / 'node_splits.csv'}")
    print(f"  {PROCESSED / 'graph_splits.json'}")

    print("\n== figures ==")
    fig_paths = make_figures(node_splits, stats)
    for p in fig_paths:
        print(f"  {p}")

    write_reports(g, checks, stats, fig_paths)
    print("\nreports: results/graph_stats.json, "
          "results/graph_construction_summary.md")
    print("\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
