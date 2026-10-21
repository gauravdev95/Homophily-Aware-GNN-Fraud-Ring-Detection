"""Teacher demo figure — real transaction flow around one fraud transaction.

Picks a REAL illicit transaction from Elliptic++ that has a readable
neighbourhood (a few senders and receivers, incl. other illicit nodes),
and draws it as a left-to-right money-flow diagram:

    senders  ->  FRAUD (centre)  ->  receivers

Red = illicit (fraud), Green = licit, Grey = unknown.
Arrows = money-flow direction. No invented data — every node/edge comes
from txs_classes.csv + txs_edgelist.csv (+ timestep from txs_features.csv).

Output: results/figures/graph/transaction_flow_sample.png
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "elliptic_pp"
OUT = ROOT / "results" / "figures" / "graph" / "transaction_flow_sample.png"

COLORS = {1: "#E45756", 2: "#54A24B", 3: "#9E9E9E"}
NAMES = {1: "illicit (fraud)", 2: "licit (genuine)", 3: "unknown"}
MAX_NODES = 30


def pick_centre(classes, edges):
    """Best demo centre: illicit node, readable degree, >=1 illicit neighbour,
    no node that is both a sender and a receiver (keeps 3 clean columns)."""
    cls = dict(zip(classes["txId"], classes["class"]))
    in_n = defaultdict(list)
    out_n = defaultdict(list)
    for a, b in zip(edges["txId1"], edges["txId2"]):
        out_n[a].append(b)
        in_n[b].append(a)

    best = None
    for tx, c in cls.items():
        if c != 1:
            continue
        senders = set(in_n.get(tx, []))
        receivers = set(out_n.get(tx, []))
        if senders & receivers:
            continue
        if len(senders) < 2 or len(receivers) < 2:
            continue  # need flow on BOTH sides for a readable demo
        deg = len(senders) + len(receivers)
        if not (4 <= deg <= 14):
            continue
        n_illicit_nb = sum(1 for nb in senders | receivers
                           if cls.get(nb) == 1)
        if n_illicit_nb < 1:
            continue
        score = (n_illicit_nb, -deg)
        if best is None or score > best[0]:
            best = (score, tx, senders, receivers)
    _, tx, senders, receivers = best
    return tx, senders, receivers, cls, in_n, out_n


def main():
    classes = pd.read_csv(RAW / "txs_classes.csv")
    edges = pd.read_csv(RAW / "txs_edgelist.csv")
    feats = pd.read_csv(RAW / "txs_features.csv", usecols=["txId", "Time step"])
    step_of = dict(zip(feats["txId"], feats["Time step"]))

    centre, senders, receivers, cls, _, _ = pick_centre(classes, edges)

    # trim unknowns if the neighbourhood is too big (keep all illicit/licit)
    def trim(nodes):
        nodes = sorted(nodes)
        if len(nodes) <= MAX_NODES:
            return nodes
        keep = [n for n in nodes if cls.get(n) != 3]
        rest = [n for n in nodes if cls.get(n) == 3]
        keep += rest[: MAX_NODES - len(keep)]
        return sorted(keep)

    senders = trim(senders)
    receivers = trim(receivers)

    # ---- layout: 3 columns, money flows left -> right ----
    def col_positions(nodes, x):
        # illicit first, then licit, then unknown — tidy vertical order
        nodes = sorted(nodes, key=lambda n: (cls.get(n), n))
        ys = [i - (len(nodes) - 1) / 2 for i in range(len(nodes))]
        return {n: (x, y) for n, y in zip(nodes, ys)}

    pos = {centre: (1.0, 0.0)}
    pos.update(col_positions(senders, 0.0))
    pos.update(col_positions(receivers, 2.0))

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(-max(len(senders), len(receivers)) / 2 - 1.2,
                max(len(senders), len(receivers)) / 2 + 1.2)
    ax.axis("off")

    # column headers (flow stages)
    for x, label in [(0.0, "SENDERS\n(paise bhejne wale)"),
                     (1.0, "FRAUD TRANSACTION\n(centre)"),
                     (2.0, "RECEIVERS\n(paise paane wale)")]:
        ax.text(x, ax.get_ylim()[1] - 0.3, label, ha="center", va="top",
                fontsize=11, fontweight="bold", color="#333333")

    # arrows: senders -> centre -> receivers
    for s in senders:
        ax.add_patch(FancyArrowPatch(
            (pos[s][0] + 0.13, pos[s][1]), (pos[centre][0] - 0.18, pos[centre][1]),
            arrowstyle="-|>", mutation_scale=16, linewidth=1.4,
            color="#555555", alpha=0.75,
            connectionstyle="arc3,rad=0.08", shrinkA=0, shrinkB=0))
    for r in receivers:
        ax.add_patch(FancyArrowPatch(
            (pos[centre][0] + 0.18, pos[centre][1]), (pos[r][0] - 0.13, pos[r][1]),
            arrowstyle="-|>", mutation_scale=16, linewidth=1.4,
            color="#555555", alpha=0.75,
            connectionstyle="arc3,rad=0.08", shrinkA=0, shrinkB=0))

    # nodes (labels BELOW the circle so full TX ids stay readable)
    for n, (x, y) in pos.items():
        c = cls.get(n, 3)
        big = (n == centre)
        ax.scatter([x], [y], s=1100 if big else 650,
                   color=COLORS[c], edgecolor="black",
                   linewidth=2 if big else 1.2, zorder=3)
        ax.text(x, y - 0.38, f"TX {n}", ha="center", va="top",
                fontsize=9 if big else 8.5,
                fontweight="bold" if big else "normal", color="#222222",
                zorder=4)
        if big:
            ax.annotate("FRAUD", (x, y + 0.55), ha="center",
                        fontsize=11, fontweight="bold", color="#C0392B")

    # legend + source note
    for c in (1, 2, 3):
        ax.scatter([], [], s=200, color=COLORS[c], edgecolor="black",
                   label=NAMES[c])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=3,
              frameon=False, fontsize=10)
    ax.set_title(
        "Real transaction flow around a fraud transaction  (Elliptic++ data)",
        fontsize=14, fontweight="bold", pad=28)
    fig.text(0.5, 0.02,
             f"Every node and arrow is real data: centre TX {centre} "
             f"(timestep {step_of[centre]}), "
             f"{len(senders)} senders, {len(receivers)} receivers.  "
             "Arrows show money-flow direction.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"saved {OUT}")
    print(f"centre TX {centre} (timestep {step_of[centre]})")
    print(f"senders ({len(senders)}): {senders}")
    print(f"receivers ({len(receivers)}): {receivers}")
    print("sender classes:", {int(cls[s]) for s in senders},
          "| receiver classes:", {int(cls[r]) for r in receivers})


if __name__ == "__main__":
    main()
