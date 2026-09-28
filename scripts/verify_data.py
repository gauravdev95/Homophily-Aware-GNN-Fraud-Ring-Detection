"""Read-only verification of the raw Elliptic++ transaction CSVs.

Checks: file sizes, row/column counts, missing values, class distribution,
txId consistency across files, and edgelist integrity. Never modifies raw data.
"""
from pathlib import Path
import sys

import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "elliptic_pp"

EXPECTED = {
    "txs_features.csv": {"rows": 203_769, "cols": 184},   # txId + 183 features
    "txs_classes.csv": {"rows": 203_769, "cols": 2},      # txId, class
    "txs_edgelist.csv": {"rows": 234_355, "cols": 2},     # txId1, txId2
}
EXPECTED_CLASSES = {1: 4_545, 2: 42_019, 3: 157_205}  # illicit / licit / unknown


def main() -> int:
    ok = True
    print("== file sizes ==")
    for name in EXPECTED:
        p = RAW / name
        if not p.exists():
            print(f"  MISSING: {name}")
            ok = False
            continue
        print(f"  {name}: {p.stat().st_size / 1e6:.1f} MB")

    print("\n== shapes ==")
    feats = pd.read_csv(RAW / "txs_features.csv")
    classes = pd.read_csv(RAW / "txs_classes.csv")
    edges = pd.read_csv(RAW / "txs_edgelist.csv")
    for name, df in [("txs_features.csv", feats), ("txs_classes.csv", classes),
                     ("txs_edgelist.csv", edges)]:
        exp = EXPECTED[name]
        status = "OK" if (len(df) == exp["rows"] and df.shape[1] == exp["cols"]) else "MISMATCH"
        if status != "OK":
            ok = False
        print(f"  {name}: rows={len(df)} (exp {exp['rows']}), cols={df.shape[1]} (exp {exp['cols']}) -> {status}")

    print("\n== missing values ==")
    for name, df in [("txs_features.csv", feats), ("txs_classes.csv", classes),
                     ("txs_edgelist.csv", edges)]:
        n = int(df.isna().sum().sum())
        print(f"  {name}: {n} missing cells" + ("" if n == 0 else "  <-- CHECK"))

    print("\n== class distribution (txs_classes.csv) ==")
    dist = classes["class"].value_counts().sort_index()
    for c, exp_n in EXPECTED_CLASSES.items():
        n = int(dist.get(c, 0))
        status = "OK" if n == exp_n else "MISMATCH"
        if status != "OK":
            ok = False
        label = {1: "illicit", 2: "licit", 3: "unknown"}[c]
        print(f"  class {c} ({label}): {n} (exp {exp_n}) -> {status}")
    extra = set(dist.index) - set(EXPECTED_CLASSES)
    if extra:
        print(f"  unexpected class values: {sorted(extra)}")
        ok = False

    print("\n== txId consistency ==")
    f_ids, c_ids = set(feats["txId"]), set(classes["txId"])
    print(f"  features txIds: {len(f_ids)}, classes txIds: {len(c_ids)}")
    print(f"  features == classes txId sets: {f_ids == c_ids}")
    if f_ids != c_ids:
        ok = False
    e1, e2 = set(edges["txId1"]), set(edges["txId2"])
    dangling = (e1 | e2) - f_ids
    print(f"  edgelist endpoints: {len(e1 | e2)} unique, dangling (not in features): {len(dangling)}")
    self_loops = int((edges["txId1"] == edges["txId2"]).sum())
    dupes = int(edges.duplicated().sum())
    print(f"  self-loops: {self_loops}, duplicate edges: {dupes}")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
