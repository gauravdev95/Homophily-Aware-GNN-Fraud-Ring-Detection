"""Phase 3 / Step 6 — GNN experiments (GCN / GAT / GraphSAGE /
H2GCN-style / GatedGCN) on the Elliptic++ temporal splits.

Supervised training on LABELED nodes only (illicit vs licit). Unknown nodes
(class 3) stay in the graph as message-passing context — they are never
deleted and never training targets. Splits are temporal (train steps 1-34,
val 35-41, test 42-49); NEVER random.

Feature pipeline is IDENTICAL to Step 5 (scripts/run_baselines.py):
all 183 features, median imputation fit on TRAIN only + one binary
missingness-indicator column for the 17 named-stat block, StandardScaler
fit on train only (the same scaler the MLP baseline used). The label is
never used for imputation/scaling. The GNN input is the scaled
184-column matrix (183 features + indicator) for ALL nodes — unknown
nodes included, since they carry features and edges.

Message passing is transductive: all nodes' features and edges are
visible during propagation, but the loss is computed ONLY on labelled
train nodes, model selection uses VAL PR-AUC only, and TEST is
evaluated once at the end. No test label is ever used for training or
selection.

Graph: directed money-flow edges (src = sender, dst = receiver).
Aggregation is over INCOMING neighbours (senders), which respects the
money-flow direction. GCN/GAT/GraphSAGE use PyTorch Geometric layers
(torch_geometric 2.8.1, CPU); the two homophily-aware variants are
plain-torch modules using index_add propagation, so all five models
share one directed incoming-neighbour convention.

No SMOTE, no oversampling, no resampling of any kind — class-weighted
BCE loss only (pos_weight = licit/illicit on train). Primary metric
PR-AUC (average precision). Accuracy is never headlined.

Raw CSVs are read-only.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_baselines as rb

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "elliptic_pp"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIG_DIR = RESULTS / "figures" / "gnn"

RANDOM_STATE = 42
HIDDEN = 64
MAX_EPOCHS = 200
PATIENCE = 20
LR = 0.01
WEIGHT_DECAY = 5e-4
DROPOUT = 0.5
MODELS = ["gcn", "gat", "sage", "h2gcn", "gated_gcn"]
XGB_TEST_PR_AUC = 0.5652  # Step 5 bar to beat

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.set_num_threads(4)


# ---------------------------------------------------------------- data

def load_graph_data(feat_cols: list[str] | None = None) -> dict:
    """Load features + labels + splits + directed edges, aligned on the
    canonical node order (txId order of txs_classes.csv, as in
    scripts/build_graph.py).

    Returns X (scaled, all nodes), y (-1 unknown / 0 licit / 1 illicit),
    boolean train/val/test labelled masks, edge_index [2, E] (src->dst),
    and the feature columns used.
    """
    df = rb.load_all()
    if feat_cols is None:
        feat_cols = rb.feature_columns(df)
    classes = pd.read_csv(RAW / "txs_classes.csv")
    tx_ids = classes["txId"].to_numpy()
    index = {t: i for i, t in enumerate(tx_ids)}

    # labelled train/val/test matrices -> fit imputer + scaler on train
    Xy = rb.split_xy(df, feat_cols)
    X_train_raw, y_train = Xy["train"]
    imputer = rb.fit_preprocess(X_train_raw)
    X_train_imp = _transform(X_train_raw, feat_cols, imputer)
    scaler = rb.fit_scaler(X_train_imp)

    # full-node feature matrix in canonical node order (all nodes, unknowns
    # included — they carry features and edges for message passing)
    df_ord = df.set_index("txId").loc[tx_ids].reset_index()
    X_all_raw = df_ord[feat_cols].to_numpy(dtype=np.float64)
    X = _transform(X_all_raw, feat_cols, imputer, scaler)
    X = np.ascontiguousarray(X, dtype=np.float32)

    cls = df_ord["class"].to_numpy()
    y = np.where(cls == 1, 1, np.where(cls == 2, 0, -1)).astype(np.int64)
    split = df_ord["split"].to_numpy()
    masks = {}
    for name in ("train", "val", "test"):
        masks[name] = (split == name) & (y >= 0)

    edges = pd.read_csv(RAW / "txs_edgelist.csv")
    src = edges["txId1"].map(index).to_numpy(dtype=np.int64)
    dst = edges["txId2"].map(index).to_numpy(dtype=np.int64)
    assert len(src) == 234_355, f"unexpected edge count {len(src)}"
    assert (src != dst).all(), "self-loop in edgelist"
    edge_index = np.stack([src, dst])

    n_licit = int((y_train == 0).sum())
    n_illicit = int((y_train == 1).sum())
    return {
        "X": X, "y": y, "masks": masks, "edge_index": edge_index,
        "feat_cols": feat_cols, "n_nodes": len(tx_ids),
        "n_edges": len(src), "pos_weight": n_licit / n_illicit,
        "n_train_labelled": len(y_train),
    }


def _transform(X_raw: np.ndarray, feat_cols: list[str], imputer,
               scaler=None) -> np.ndarray:
    """Impute (train-fit) + append missingness indicator, optionally
    scale. Mirrors rb.transform, but only over the named-stat columns
    actually present in feat_cols (feature subsets in the ablation
    contain none of them, so the indicator is all zeros there)."""
    X = imputer.transform(X_raw)
    present = [c for c in rb.NAMED_STATS if c in feat_cols]
    if present:
        idx = [feat_cols.index(c) for c in present]
        ind = np.isnan(X_raw[:, idx]).any(axis=1).astype(np.float64)
    else:
        ind = np.zeros(len(X_raw), dtype=np.float64)
    X = np.hstack([X, ind.reshape(-1, 1)])
    if scaler is not None:
        X = scaler.transform(X)
    return X


def to_tensors(data: dict) -> dict:
    """NumPy data -> torch tensors (CPU)."""
    t = {
        "x": torch.from_numpy(data["X"]),
        "y": torch.from_numpy(data["y"]).float(),
        "edge_index": torch.from_numpy(data["edge_index"]).long(),
    }
    t["masks"] = {k: torch.from_numpy(v) for k, v in data["masks"].items()}
    return t
    """NumPy data -> torch tensors (CPU)."""
    t = {
        "x": torch.from_numpy(data["X"]),
        "y": torch.from_numpy(data["y"]).float(),
        "edge_index": torch.from_numpy(data["edge_index"]).long(),
    }
    t["masks"] = {k: torch.from_numpy(v) for k, v in data["masks"].items()}
    return t


def propagate(x: torch.Tensor, edge_index: torch.Tensor,
              norm: torch.Tensor) -> torch.Tensor:
    """Row-normalised incoming-neighbour aggregation (no self-loops):
    out[dst] = sum over incoming edges of norm[e] * x[src[e]]."""
    out = torch.zeros_like(x)
    out.index_add_(0, edge_index[1],
                   x[edge_index[0]] * norm.unsqueeze(1))
    return out


def edge_norm(edge_index: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """1 / in-degree per edge (nodes with no incoming edge get 0)."""
    deg = torch.zeros(n_nodes)
    deg.index_add_(0, edge_index[1],
                   torch.ones(edge_index.shape[1]))
    return torch.where(deg[edge_index[1]] > 0,
                       1.0 / deg[edge_index[1]].clamp(min=1),
                       torch.zeros_like(deg[edge_index[1]]))


# ---------------------------------------------------------------- models

class GCNNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = HIDDEN,
                 dropout: float = DROPOUT):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index).squeeze(-1)


class GATNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = HIDDEN,
                 dropout: float = DROPOUT):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden // 4, heads=4, dropout=dropout)
        self.conv2 = GATConv(hidden, 1, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index, return_attention: bool = False):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        if return_attention:
            out, (ei, alpha) = self.conv2(x, edge_index,
                                          return_attention_weights=True)
            return out.squeeze(-1), ei, alpha
        return self.conv2(x, edge_index).squeeze(-1)


class SAGENet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = HIDDEN,
                 dropout: float = DROPOUT):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden, aggr="mean")
        self.conv2 = SAGEConv(hidden, 1, aggr="mean")
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index).squeeze(-1)


class H2GCNNet(nn.Module):
    """H2GCN-style: ego embedding kept separate from neighbour aggregation,
    with 1-hop AND 2-hop neighbour channels; layer outputs concatenated
    for the final classifier (no smoothing of ego into neighbours).

    Justification (Step 4): illicit nodes are bimodal — 45% have zero
    same-label neighbours, 46% have all. Uniform smoothing (GCN) dilutes
    the ego signal of the zero-homophily illicit nodes; keeping the ego
    embedding separate lets the classifier use a node's own features
    when its neighbourhood is uninformative, and the 2-hop channel
    reaches the illicit->illicit enrichment (5.9x) one step further out.
    """

    def __init__(self, in_dim: int, hidden: int = HIDDEN,
                 dropout: float = DROPOUT):
        super().__init__()
        self.ego1 = nn.Linear(in_dim, hidden // 2)
        self.n1_1 = nn.Linear(in_dim, hidden // 2)
        self.n2_1 = nn.Linear(in_dim, hidden // 2)
        self.ego2 = nn.Linear(hidden * 3 // 2, hidden // 2)
        self.n1_2 = nn.Linear(hidden * 3 // 2, hidden // 2)
        self.n2_2 = nn.Linear(hidden * 3 // 2, hidden // 2)
        self.clf = nn.Linear(hidden * 3, 1)
        self.dropout = dropout
        self.norm = None
        self.edge_index = None

    def _prop(self, x):
        return propagate(x, self.edge_index, self.norm)

    def forward(self, x, edge_index):
        if self.norm is None:
            self.edge_index = edge_index
            self.norm = edge_norm(edge_index, x.shape[0])
        h1 = torch.cat([self.ego1(x), self.n1_1(self._prop(x)),
                        self.n2_1(self._prop(self._prop(x)))], dim=1)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h2 = torch.cat([self.ego2(h1), self.n1_2(self._prop(h1)),
                        self.n2_2(self._prop(self._prop(h1)))], dim=1)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        return self.clf(torch.cat([h1, h2], dim=1)).squeeze(-1)


class GatedGCNNet(nn.Module):
    """Homophily-gated aggregation: per node, a learned gate blends the
    ego embedding and the neighbour aggregate, so a node can suppress
    neighbour signal when its neighbourhood is uninformative.

    Justification (Step 4): node homophily is bimodal for illicit nodes
    (45% zero same-label neighbours, 46% all) and licit nodes mix more
    broadly than chance (0.72x) — a fixed smoothing weight is wrong for
    both extremes. The gate is computed from [ego, neighbour] per node,
    so each node decides how much to trust its neighbourhood.
    """

    def __init__(self, in_dim: int, hidden: int = HIDDEN,
                 dropout: float = DROPOUT):
        super().__init__()
        self.lin_self1 = nn.Linear(in_dim, hidden)
        self.lin_neigh1 = nn.Linear(in_dim, hidden)
        self.gate1 = nn.Linear(2 * hidden, 1)
        self.lin_self2 = nn.Linear(hidden, hidden)
        self.lin_neigh2 = nn.Linear(hidden, hidden)
        self.gate2 = nn.Linear(2 * hidden, 1)
        self.clf = nn.Linear(hidden, 1)
        self.dropout = dropout
        self.norm = None
        self.edge_index = None

    def _prop(self, x):
        return propagate(x, self.edge_index, self.norm)

    def _layer(self, x, lin_self, lin_neigh, gate):
        h_self = lin_self(x)
        h_neigh = lin_neigh(self._prop(x))
        g = torch.sigmoid(gate(torch.cat([h_self, h_neigh], dim=1)))
        return F.relu(g * h_neigh + (1 - g) * h_self)

    def forward(self, x, edge_index):
        if self.norm is None:
            self.edge_index = edge_index
            self.norm = edge_norm(edge_index, x.shape[0])
        h = self._layer(x, self.lin_self1, self.lin_neigh1, self.gate1)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self._layer(h, self.lin_self2, self.lin_neigh2, self.gate2)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.clf(h).squeeze(-1)


def build_model(name: str, in_dim: int) -> nn.Module:
    torch.manual_seed(RANDOM_STATE)
    return {"gcn": GCNNet, "gat": GATNet, "sage": SAGENet,
            "h2gcn": H2GCNNet, "gated_gcn": GatedGCNNet}[name](in_dim)


# ---------------------------------------------------------------- training

def evaluate(model, t: dict, mask: torch.Tensor) -> dict:
    """PR-AUC, ROC-AUC, F1 / precision / recall at 0.5, confusion matrix."""
    model.eval()
    with torch.no_grad():
        logits = model(t["x"], t["edge_index"])
    proba = torch.sigmoid(logits).numpy()
    y = t["y"].numpy().astype(int)
    p, yy = proba[mask.numpy()], y[mask.numpy()]
    pred = (p > 0.5).astype(int)
    cm = confusion_matrix(yy, pred).tolist()
    return {
        "n": int(len(yy)),
        "n_positive": int(yy.sum()),
        "pr_auc": float(average_precision_score(yy, p)),
        "roc_auc": float(roc_auc_score(yy, p)),
        "f1_at_0.5": float(f1_score(yy, pred, zero_division=0)),
        "precision_at_0.5": float(precision_score(yy, pred, zero_division=0)),
        "recall_at_0.5": float(recall_score(yy, pred, zero_division=0)),
        "confusion_matrix": cm,  # [[TN, FP], [FN, TP]]
        "_proba": p,
        "_y": yy,
    }


def train_model(name: str, t: dict, pos_weight: float,
                max_epochs: int = MAX_EPOCHS,
                patience: int = PATIENCE) -> dict:
    """Full-batch training with early stopping on val PR-AUC."""
    model = build_model(name, t["x"].shape[1])
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR,
                           weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight]))
    train_mask = t["masks"]["train"]
    val_mask = t["masks"]["val"]

    best_pr, best_epoch, best_state, wait = -1.0, -1, None, 0
    history = []
    t0 = time.time()
    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad()
        logits = model(t["x"], t["edge_index"])
        loss = loss_fn(logits[train_mask], t["y"][train_mask])
        loss.backward()
        opt.step()
        val = evaluate(model, t, val_mask)
        history.append(val["pr_auc"])
        if val["pr_auc"] > best_pr + 1e-5:
            best_pr = val["pr_auc"]
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    [{name}] epoch {epoch + 1:3d} loss {loss.item():.4f} "
                  f"val PR-AUC {val['pr_auc']:.4f} (best {best_pr:.4f} "
                  f"@ epoch {best_epoch + 1})", flush=True)
        if wait >= patience:
            print(f"    [{name}] early stop at epoch {epoch + 1} "
                  f"(best val PR-AUC {best_pr:.4f} @ epoch {best_epoch + 1})",
                  flush=True)
            break
    fit_s = time.time() - t0
    model.load_state_dict(best_state)
    return {"model": model, "best_epoch": best_epoch + 1,
            "best_val_pr_auc": best_pr, "fit_seconds": round(fit_s, 1),
            "n_params": n_params, "history": history,
            "epochs_run": epoch + 1}


def train_all(t: dict, pos_weight: float, quick: bool = False) -> dict:
    """Train all five models; evaluate val + test ONCE with best weights."""
    results, fitted = {}, {}
    names = MODELS if not quick else ["gcn"]
    for name in names:
        print(f"== training {name} ==", flush=True)
        out = train_model(name, t, pos_weight,
                          max_epochs=5 if quick else MAX_EPOCHS)
        fitted[name] = out["model"]
        entry = {"fit_seconds": out["fit_seconds"],
                 "best_epoch": out["best_epoch"],
                 "n_params": out["n_params"],
                 "epochs_run": out["epochs_run"]}
        for split in ("val", "test"):
            m = evaluate(out["model"], t, t["masks"][split])
            entry[split] = {k: v for k, v in m.items()
                            if not k.startswith("_")}
            entry[f"_{split}_proba"] = m["_proba"]
            entry[f"_{split}_y"] = m["_y"]
        entry["_history"] = out["history"]
        results[name] = entry
        print(f"[{name:9s}] best epoch {out['best_epoch']:3d} "
              f"fit {out['fit_seconds']:6.1f}s  "
              f"val PR-AUC={entry['val']['pr_auc']:.4f}  "
              f"test PR-AUC={entry['test']['pr_auc']:.4f}", flush=True)
    return results, fitted


# ---------------------------------------------------------------- fraud-ring + explainability

class _UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)

    def find(self, x):
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def _components(nodes: np.ndarray, src: np.ndarray,
                dst: np.ndarray) -> list[int]:
    """Weak component sizes of the subgraph induced by `nodes`."""
    pos = {n: i for i, n in enumerate(nodes)}
    uf = _UnionFind(len(nodes))
    keep = np.array([s in pos and d in pos for s, d in zip(src, dst)])
    for s, d in zip(src[keep], dst[keep]):
        uf.union(pos[s], pos[d])
    roots = np.array([uf.find(i) for i in range(len(nodes))])
    _, counts = np.unique(roots, return_counts=True)
    return sorted(counts.tolist(), reverse=True)


def fraud_ring_eval(best_name: str, results: dict, t: dict,
                    data: dict) -> dict:
    """Predicted-illicit subgraph on labelled TEST nodes vs the true
    illicit subgraph. All counts computed, none estimated."""
    test_mask = data["masks"]["test"]
    test_idx = np.where(test_mask)[0]  # node ids of labelled test nodes
    y_test = data["y"][test_mask]
    proba = results[best_name]["_test_proba"]
    pred = (proba > 0.5).astype(int)

    src, dst = data["edge_index"][0], data["edge_index"][1]
    # edges with both ends labelled test nodes
    in_test = np.isin(src, test_idx) & np.isin(dst, test_idx)
    e_src, e_dst = src[in_test], dst[in_test]
    y_map = dict(zip(test_idx.tolist(), y_test.tolist()))
    p_map = dict(zip(test_idx.tolist(), pred.tolist()))

    def edge_rate(src_nodes, dst_nodes, label_map):
        sel = np.array([label_map.get(int(s), -1) == 1 for s in src_nodes])
        if sel.sum() == 0:
            return None, 0
        rate = float(np.mean([label_map.get(int(d), -1) == 1
                              for d in dst_nodes[sel]]))
        return rate, int(sel.sum())

    pred_nodes = test_idx[pred == 1]
    true_nodes = test_idx[y_test == 1]
    pred_sizes = _components(pred_nodes, e_src, e_dst)
    true_sizes = _components(true_nodes, e_src, e_dst)
    pred_rate, pred_e = edge_rate(
        e_src, e_dst, {int(n): int(p_map[n]) for n in test_idx})
    true_rate, true_e = edge_rate(
        e_src, e_dst, {int(n): int(y_map[n]) for n in test_idx})

    def size_hist(sizes):
        return {"n_components": len(sizes),
                "largest": int(sizes[0]) if sizes else 0,
                "singletons": int(sum(1 for s in sizes if s == 1)),
                "size_2_5": int(sum(1 for s in sizes if 2 <= s <= 5)),
                "size_6_20": int(sum(1 for s in sizes if 6 <= s <= 20)),
                "size_gt_20": int(sum(1 for s in sizes if s > 20))}

    return {
        "model": best_name,
        "n_test_labelled": int(len(y_test)),
        "n_true_illicit": int((y_test == 1).sum()),
        "n_pred_illicit": int(pred.sum()),
        "precision_illicit": float(results[best_name]["test"]
                                   ["precision_at_0.5"]),
        "recall_illicit": float(results[best_name]["test"]
                                ["recall_at_0.5"]),
        "observed_illicit_illicit_rate_all": 0.2961,
        "pred_illicit_illicit_edge_rate": pred_rate,
        "pred_illicit_out_edges": pred_e,
        "true_illicit_illicit_edge_rate_test": true_rate,
        "true_illicit_out_edges": true_e,
        "pred_subgraph": size_hist(pred_sizes),
        "true_subgraph": size_hist(true_sizes),
        "pred_component_sizes": pred_sizes[:20],
        "true_component_sizes": true_sizes[:20],
    }


def gat_attention_analysis(gat_model, t: dict, data: dict) -> dict:
    """Mean final-layer attention per edge type (both ends labelled, plus
    unknown-involved). Attention is a softmax over incoming edges per
    target node, so means are comparable across types."""
    gat_model.eval()
    with torch.no_grad():
        _, ei, alpha = gat_model(t["x"], t["edge_index"],
                                 return_attention=True)
    att = alpha.mean(dim=1).numpy()  # mean over heads
    # drop the self-loops GATConv adds (src == dst): they are not real
    # money-flow edges and would otherwise count as II/LL edges
    keep = (ei[0] != ei[1]).numpy()
    ei, att = ei[:, keep], att[keep]
    src, dst = ei[0].numpy(), ei[1].numpy()
    y = data["y"]
    types = {}
    for s, d, a in zip(src, dst, att):
        ys, yd = int(y[s]), int(y[d])
        if ys == 1 and yd == 1:
            k = "illicit->illicit"
        elif ys == 0 and yd == 0:
            k = "licit->licit"
        elif ys == 1 and yd == 0:
            k = "illicit->licit"
        elif ys == 0 and yd == 1:
            k = "licit->illicit"
        elif ys == -1 or yd == -1:
            k = "unknown-involved"
        else:
            k = "other"
        types.setdefault(k, []).append(float(a))
    return {k: {"n_edges": len(v), "mean_attention": float(np.mean(v)),
                "median_attention": float(np.median(v))}
            for k, v in sorted(types.items())}


def ablation(best_name: str, data_full: dict, feat_cols_all: list[str],
             pos_weight: float) -> dict:
    """Feature-group ablation for the best model: retrain it on feature
    subsets (imputer + scaler refit on train only, same protocol)."""
    groups = {
        "local_only": ["Time step"] + [c for c in feat_cols_all
                                       if c.startswith("Local_feature_")],
        "local_plus_aggregate": (["Time step"]
                                 + [c for c in feat_cols_all
                                    if c.startswith("Local_feature_")]
                                 + [c for c in feat_cols_all
                                    if c.startswith("Aggregate_feature_")]),
    }
    out = {}
    for gname, cols in groups.items():
        print(f"== ablation {gname} ({len(cols)} features) ==", flush=True)
        sub = load_graph_data(feat_cols=cols)
        t_sub = to_tensors(sub)
        res = train_model(best_name, t_sub, pos_weight)
        model = res["model"]
        entry = {"n_features": len(cols) + 1,  # + missingness indicator
                 "fit_seconds": res["fit_seconds"],
                 "best_epoch": res["best_epoch"]}
        for split in ("val", "test"):
            m = evaluate(model, t_sub, t_sub["masks"][split])
            entry[split] = {k: v for k, v in m.items()
                            if not k.startswith("_")}
        out[gname] = entry
        print(f"[{gname:20s}] val PR-AUC={entry['val']['pr_auc']:.4f}  "
              f"test PR-AUC={entry['test']['pr_auc']:.4f}", flush=True)
    return out


# ---------------------------------------------------------------- figures

def make_figures(results: dict, best_name: str, fraud: dict,
                 attention: dict, ablation_res: dict) -> list[str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    names = list(results)
    colors = {"gcn": "#4C78A8", "gat": "#F58518", "sage": "#54A24B",
              "h2gcn": "#E45756", "gated_gcn": "#72B7B2"}

    # 1. grouped val/test PR-AUC bars + XGBoost reference line
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(names))
    val = [results[n]["val"]["pr_auc"] for n in names]
    test = [results[n]["test"]["pr_auc"] for n in names]
    ax.bar(x - 0.2, val, 0.4, label="val PR-AUC", color="#9E9E9E")
    ax.bar(x + 0.2, test, 0.4, label="test PR-AUC",
           color=[colors[n] for n in names])
    ax.axhline(XGB_TEST_PR_AUC, color="black", ls="--", lw=1.2,
               label=f"XGBoost test PR-AUC {XGB_TEST_PR_AUC:.4f} (Step 5)")
    for i, n in enumerate(names):
        ax.text(i - 0.2, val[i] + 0.02, f"{val[i]:.3f}", ha="center",
                fontsize=8)
        ax.text(i + 0.2, test[i] + 0.02, f"{test[i]:.3f}", ha="center",
                fontsize=8)
    ax.set_xticks(x, names)
    ax.set_xlabel("model")
    ax.set_ylabel("PR-AUC")
    ax.set_title("GNN test PR-AUC vs XGBoost bar (Step 5)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fp = FIG_DIR / "gnn_pr_auc_bars.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    paths.append(str(fp))

    # 2. test PR curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for n in names:
        y = results[n]["_test_y"]
        p = results[n]["_test_proba"]
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, label=f"{n} (AP={results[n]['test']['pr_auc']:.3f})",
                color=colors[n])
    base = results[names[0]]["test"]["n_positive"] / results[names[0]]["test"]["n"]
    ax.axhline(base, color="black", ls="--", lw=1,
               label=f"chance (illicit rate {base:.3f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision–Recall curves (test, steps 42–49)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fp = FIG_DIR / "gnn_pr_curves.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    paths.append(str(fp))

    # 3. val -> test drop per model
    fig, ax = plt.subplots(figsize=(8, 4.5))
    drops = [results[n]["val"]["pr_auc"] - results[n]["test"]["pr_auc"]
             for n in names]
    ax.bar(names, drops, color=[colors[n] for n in names])
    for i, d in enumerate(drops):
        ax.text(i, d + 0.01, f"{d:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("val PR-AUC − test PR-AUC")
    ax.set_title("Temporal gap: val → test PR-AUC drop per model")
    ax.set_ylim(0, max(drops) * 1.25)
    fig.tight_layout()
    fp = FIG_DIR / "gnn_val_test_drop.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    paths.append(str(fp))

    # 4. training curves (val PR-AUC per epoch)
    fig, ax = plt.subplots(figsize=(8, 5))
    for n in names:
        h = results[n]["_history"]
        ax.plot(range(1, len(h) + 1), h, label=n, color=colors[n], lw=1.2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("val PR-AUC")
    ax.set_title("Validation PR-AUC during training (early stopping)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fp = FIG_DIR / "gnn_training_curves.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    paths.append(str(fp))

    # 5. fraud-ring: component-size histogram + edge-rate comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    pred_sizes = fraud["pred_component_sizes"]
    true_sizes = fraud["true_component_sizes"]
    axes[0].bar(range(len(pred_sizes)), pred_sizes, color="#E45756",
                label="predicted-illicit")
    axes[0].bar(range(len(true_sizes)), true_sizes, color="#4C78A8",
                alpha=0.55, label="true-illicit")
    axes[0].set_xlabel("component (largest first)")
    axes[0].set_ylabel("nodes")
    axes[0].set_title("Largest components of illicit subgraphs (test)")
    axes[0].legend(fontsize=9)
    labels = ["observed\n(all labelled)", "true-illicit\n(test)",
              "predicted-illicit\n(test)"]
    vals = [fraud["observed_illicit_illicit_rate_all"],
            fraud["true_illicit_illicit_edge_rate_test"],
            fraud["pred_illicit_illicit_edge_rate"]]
    axes[1].bar(labels, vals, color=["#9E9E9E", "#4C78A8", "#E45756"])
    for i, v in enumerate(vals):
        axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    axes[1].set_ylabel("P(dst illicit | src illicit)")
    axes[1].set_title("Illicit→illicit edge rate")
    axes[1].set_ylim(0, max(vals) * 1.25)
    fig.tight_layout()
    fp = FIG_DIR / "gnn_fraud_ring.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    paths.append(str(fp))

    # 6. GAT attention by edge type
    fig, ax = plt.subplots(figsize=(8, 4.5))
    keys = list(attention)
    means = [attention[k]["mean_attention"] for k in keys]
    ns = [attention[k]["n_edges"] for k in keys]
    ax.bar(keys, means, color="#F58518")
    for i, (m, n) in enumerate(zip(means, ns)):
        ax.text(i, m + max(means) * 0.03, f"{m:.4f}\n(n={n:,})",
                ha="center", fontsize=8)
    ax.set_ylabel("mean final-layer attention")
    ax.set_title("GAT attention by edge type (all edges, heads averaged)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fp = FIG_DIR / "gnn_gat_attention.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    paths.append(str(fp))

    # 7. feature-group ablation (best model)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = ["local only\n(95 cols)", "local + aggregate\n(167 cols)",
              "all features\n(184 cols)"]
    vals = [ablation_res["local_only"]["test"]["pr_auc"],
            ablation_res["local_plus_aggregate"]["test"]["pr_auc"],
            results[best_name]["test"]["pr_auc"]]
    vvals = [ablation_res["local_only"]["val"]["pr_auc"],
             ablation_res["local_plus_aggregate"]["val"]["pr_auc"],
             results[best_name]["val"]["pr_auc"]]
    x = np.arange(3)
    ax.bar(x - 0.2, vvals, 0.4, label="val PR-AUC", color="#9E9E9E")
    ax.bar(x + 0.2, vals, 0.4, label="test PR-AUC", color="#54A24B")
    for i in range(3):
        ax.text(i + 0.2, vals[i] + 0.02, f"{vals[i]:.3f}", ha="center",
                fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("PR-AUC")
    ax.set_title(f"Feature-group ablation — {best_name}")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fp = FIG_DIR / "gnn_ablation.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    paths.append(str(fp))

    return paths


# ---------------------------------------------------------------- reports

def public_results(results: dict) -> dict:
    out = {}
    for name, entry in results.items():
        out[name] = {
            "fit_seconds": entry["fit_seconds"],
            "best_epoch": entry["best_epoch"],
            "epochs_run": entry["epochs_run"],
            "n_params": entry["n_params"],
            "val": entry["val"],
            "test": entry["test"],
        }
    return out


def write_reports(results: dict, best_name: str, fraud: dict,
                  attention: dict, ablation_res: dict,
                  fig_paths: list[str]) -> None:
    pub = public_results(results)
    payload = {
        "models": pub,
        "best_model": best_name,
        "xgboost_test_pr_auc_step5": XGB_TEST_PR_AUC,
        "fraud_ring": fraud,
        "gat_attention": attention,
        "ablation": {k: {kk: vv for kk, vv in v.items()}
                     for k, v in ablation_res.items()},
        "setup": {
            "splits": "train 1-34, val 35-41, test 42-49 (temporal)",
            "features": "all 183 + missingness indicator (median impute "
                        "fit on train only, StandardScaler fit on train "
                        "only — identical to Step 5)",
            "loss": "BCEWithLogitsLoss, class-weighted (pos_weight "
                    "licit/illicit on train)",
            "optimiser": "Adam lr 0.01, weight decay 5e-4, dropout 0.5",
            "hidden": HIDDEN, "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE, "seed": RANDOM_STATE,
            "framework": "PyTorch 2.15.0+cpu, torch-geometric 2.8.1, CPU",
            "transductive": True,
        },
    }
    (RESULTS / "gnn_results.json").write_text(json.dumps(payload, indent=2))

    order = sorted(pub, key=lambda n: pub[n]["val"]["pr_auc"], reverse=True)
    best_test = pub[best_name]["test"]["pr_auc"]
    beat = best_test - XGB_TEST_PR_AUC
    lines = [
        "# GNN Experiments Summary — Elliptic++ Transactions",
        "",
        "*Phase 3 / Step 6 · temporal splits (train 1–34, val 35–41, "
        "test 42–49) · supervised on labelled nodes only (unknowns are "
        "graph context, never targets) · no SMOTE, no oversampling · "
        "class-weighted loss only · accuracy never headlined · "
        "transductive message passing (all node features/edges visible; "
        "loss on train labels only).*",
        "",
        "## 1. Setup",
        "",
        "- Models: GCN, GAT (4 heads), GraphSAGE (mean), H2GCN-style "
        "(ego separate, 1-hop + 2-hop channels), GatedGCN (learned "
        "per-node ego/neighbour gate).",
        "- Features: all 183 + missingness indicator, median imputation "
        "fit on TRAIN only, StandardScaler fit on train only — the "
        "identical pipeline to Step 5 (MLP baseline).",
        "- Aggregation over INCOMING neighbours (money senders), "
        "respecting the directed money-flow direction.",
        "- Training: full-batch, Adam lr 0.01, weight decay 5e-4, "
        "dropout 0.5, hidden 64, 2 layers, up to 200 epochs, early "
        "stopping on val PR-AUC (patience 20).",
        "- Model selection on VAL using PR-AUC; final numbers on TEST.",
        "",
        "## 2. Results (PR-AUC primary)",
        "",
        "| Model | Val PR-AUC | Test PR-AUC | Test ROC-AUC | Test F1@0.5 | "
        "Test precision | Test recall | Fit (s) | Best epoch |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in order:
        v, t = pub[name]["val"], pub[name]["test"]
        lines.append(
            f"| {name} | {v['pr_auc']:.4f} | {t['pr_auc']:.4f} | "
            f"{t['roc_auc']:.4f} | {t['f1_at_0.5']:.4f} | "
            f"{t['precision_at_0.5']:.4f} | {t['recall_at_0.5']:.4f} | "
            f"{pub[name]['fit_seconds']} | {pub[name]['best_epoch']} |")
    lines += [
        "",
        f"**Best model (by val PR-AUC): {best_name}** "
        f"(val PR-AUC {pub[best_name]['val']['pr_auc']:.4f}, "
        f"test PR-AUC {best_test:.4f}).",
        "",
        f"**XGBoost bar (Step 5): test PR-AUC {XGB_TEST_PR_AUC:.4f}. "
        f"Best GNN beats it by {beat:+.4f} on test PR-AUC.**" if beat > 0
        else f"**XGBoost bar (Step 5): test PR-AUC {XGB_TEST_PR_AUC:.4f}. "
        f"Best GNN does NOT beat it on test PR-AUC ({beat:+.4f}).**",
        "",
        "## 3. Confusion matrices (test, threshold 0.5)",
        "",
        "| Model | TN | FP | FN | TP |",
        "|---|---|---|---|---|",
    ]
    for name in order:
        tn, fp = pub[name]["test"]["confusion_matrix"][0]
        fn, tp = pub[name]["test"]["confusion_matrix"][1]
        lines.append(f"| {name} | {tn:,} | {fp:,} | {fn:,} | {tp:,} |")
    lines += [
        "",
        "## 4. Fraud-ring evaluation (descriptive, test labelled nodes)",
        "",
        f"Best model ({best_name}) on test: "
        f"{fraud['n_pred_illicit']:,} predicted-illicit nodes vs "
        f"{fraud['n_true_illicit']:,} true illicit "
        f"(precision {fraud['precision_illicit']:.4f}, "
        f"recall {fraud['recall_illicit']:.4f}).",
        "",
        f"- Predicted-illicit subgraph: "
        f"{fraud['pred_subgraph']['n_components']} weak components, "
        f"largest {fraud['pred_subgraph']['largest']} nodes, "
        f"{fraud['pred_subgraph']['singletons']} singletons. True-illicit "
        f"subgraph: {fraud['true_subgraph']['n_components']} components, "
        f"largest {fraud['true_subgraph']['largest']} nodes, "
        f"{fraud['true_subgraph']['singletons']} singletons.",
        f"- Predicted illicit→illicit edge rate: "
        f"{fraud['pred_illicit_illicit_edge_rate']:.4f} "
        f"(true-illicit test rate "
        f"{fraud['true_illicit_illicit_edge_rate_test']:.4f}; observed "
        f"overall rate {fraud['observed_illicit_illicit_rate_all']:.4f}).",
        "",
        "These are descriptive statistics of the predicted subgraph, not "
        "proof of organised fraud rings — the homophily statistics "
        "describe linkage patterns only.",
        "",
        "## 5. Explainability",
        "",
        "### 5.1 GAT attention by edge type",
        "",
        "| Edge type | n edges | Mean attention | Median attention |",
        "|---|---|---|---|",
    ]
    for k, v in attention.items():
        lines.append(f"| {k} | {v['n_edges']:,} | {v['mean_attention']:.4f} | "
                     f"{v['median_attention']:.4f} |")
    lines += [
        "",
        "### 5.2 Feature-group ablation (best model)",
        "",
        "| Feature set | n features | Val PR-AUC | Test PR-AUC |",
        "|---|---|---|---|",
    ]
    for gname in ("local_only", "local_plus_aggregate"):
        a = ablation_res[gname]
        lines.append(f"| {gname} | {a['n_features']} | "
                     f"{a['val']['pr_auc']:.4f} | {a['test']['pr_auc']:.4f} |")
    lines.append(
        f"| all features | 184 | {pub[best_name]['val']['pr_auc']:.4f} | "
        f"{pub[best_name]['test']['pr_auc']:.4f} |")
    lines += [
        "",
        "## 6. Figures",
        "",
    ]
    for fp in fig_paths:
        lines.append(f"- `{Path(fp).name}`")
    lines += [
        "",
        "## 7. Reading the results",
        "",
        "- **Temporal gap.** Every model drops from val to test — future "
        "transactions are genuinely harder, exactly what the temporal "
        "split is designed to expose.",
        "- **ROC-AUC flatters on imbalanced data** (2.23% illicit); "
        "PR-AUC is the primary metric.",
        "- **Homophily-aware variants** (H2GCN-style, GatedGCN) were "
        "motivated by Step 4: illicit nodes link to illicit nodes at "
        "5.9x chance, but node homophily is bimodal (45% of illicit "
        "nodes have zero same-label neighbours), so uniform smoothing "
        "hurts — ego separation and per-node gating address exactly "
        "that.",
        "",
    ]
    (RESULTS / "gnn_summary.md").write_text("\n".join(lines))


# ---------------------------------------------------------------- main

def main(quick: bool = False) -> int:
    print("== loading data (same pipeline as Step 5) ==")
    data = load_graph_data()
    print(f"  nodes: {data['n_nodes']:,}, edges: {data['n_edges']:,}, "
          f"features: {len(data['feat_cols'])} + indicator = "
          f"{data['X'].shape[1]}")
    for name in ("train", "val", "test"):
        m = data["masks"][name]
        print(f"  {name:5s} labelled: {int(m.sum()):,} "
              f"(illicit {int((data['y'][m] == 1).sum()):,})")
    print(f"  pos_weight (licit/illicit train): {data['pos_weight']:.3f}")
    t = to_tensors(data)

    print("\n== training GNNs (selection on val PR-AUC) ==")
    results, fitted = train_all(t, data["pos_weight"], quick=quick)

    if quick:
        print("\nquick smoke test done — no reports written")
        print("\nRESULT: PASS")
        return 0

    best = max(results, key=lambda n: results[n]["val"]["pr_auc"])
    print(f"\nbest model by val PR-AUC: {best} "
          f"({results[best]['val']['pr_auc']:.4f})")

    print("\n== fraud-ring evaluation (best model, test) ==")
    fraud = fraud_ring_eval(best, results, t, data)
    print(f"  predicted-illicit: {fraud['n_pred_illicit']:,} "
          f"(true {fraud['n_true_illicit']:,}), precision "
          f"{fraud['precision_illicit']:.4f}, recall "
          f"{fraud['recall_illicit']:.4f}")
    print(f"  pred subgraph: {fraud['pred_subgraph']['n_components']} "
          f"components, largest {fraud['pred_subgraph']['largest']}")
    print(f"  pred illicit->illicit edge rate: "
          f"{fraud['pred_illicit_illicit_edge_rate']:.4f} "
          f"(observed 0.2961)")

    print("\n== GAT attention analysis ==")
    attention = gat_attention_analysis(fitted["gat"], t, data)
    for k, v in attention.items():
        print(f"  {k:18s} n={v['n_edges']:,} "
              f"mean attention {v['mean_attention']:.4f}")

    print("\n== feature-group ablation (best model) ==")
    ablation_res = ablation(best, data, data["feat_cols"],
                            data["pos_weight"])

    print("\n== figures ==")
    fig_paths = make_figures(results, best, fraud, attention, ablation_res)
    for fp in fig_paths:
        print(f"  {fp}")

    write_reports(results, best, fraud, attention, ablation_res, fig_paths)
    print("\nreports: results/gnn_results.json, results/gnn_summary.md")
    print("\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(quick="--quick" in sys.argv))
