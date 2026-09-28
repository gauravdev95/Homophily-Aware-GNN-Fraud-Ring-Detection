"""Read-only verification of the raw Elliptic++ transaction CSVs.

Checks: file sizes, row/column counts, missing values, class distribution,
txId consistency across files, and edgelist integrity. Never modifies raw data.

Uses pandas when available; falls back to a pure-stdlib (csv module)
implementation when pandas cannot be imported — e.g. on Windows machines
where an Application Control / Code Integrity policy blocks pandas'
compiled .pyd files. Pass --no-pandas to force the stdlib path.
"""
from pathlib import Path
import sys

RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "elliptic_pp"

EXPECTED = {
    "txs_features.csv": {"rows": 203_769, "cols": 184},   # txId + 183 features
    "txs_classes.csv": {"rows": 203_769, "cols": 2},      # txId, class
    "txs_edgelist.csv": {"rows": 234_355, "cols": 2},     # txId1, txId2
}
EXPECTED_CLASSES = {1: 4_545, 2: 42_019, 3: 157_205}  # illicit / licit / unknown
CLASS_NAMES = {1: "illicit", 2: "licit", 3: "unknown"}


def _missing_files():
    return [n for n in EXPECTED if not (RAW / n).exists()]


def _print_sizes():
    ok = True
    print("== file sizes ==")
    for name in EXPECTED:
        p = RAW / name
        if not p.exists():
            print(f"  MISSING: {name}")
            ok = False
            continue
        print(f"  {name}: {p.stat().st_size / 1e6:.1f} MB")
    return ok


# ---------------------------------------------------------------- pandas engine

def verify_pandas(pd) -> int:
    ok = _print_sizes()
    if _missing_files():
        print("\nRESULT: FAIL")
        return 1

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
        print(f"  class {c} ({CLASS_NAMES[c]}): {n} (exp {exp_n}) -> {status}")
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
    if dangling or self_loops or dupes:
        ok = False

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ---------------------------------------------------------------- stdlib engine

def verify_stdlib() -> int:
    """Same checks using only the csv module (no pandas/numpy). Slower,
    but works on machines where compiled extensions are blocked by policy."""
    import csv

    ok = _print_sizes()
    if _missing_files():
        print("\nRESULT: FAIL")
        return 1

    print("\n== shapes ==")
    shapes = {}
    # one streaming pass per file collects everything that file needs
    f_ids: set[str] = set()
    c_ids: set[str] = set()
    class_counts: dict[str, int] = {}
    endpoints: set[str] = set()
    edge_set: set[tuple[str, str]] = set()
    self_loops = 0
    dupes = 0
    missing = dict.fromkeys(EXPECTED, 0)

    for name, exp in EXPECTED.items():
        rows, cols = 0, 0
        with open(RAW / name, newline="", encoding="utf-8") as fh:
            rdr = csv.reader(fh)
            try:
                header = next(rdr)
            except StopIteration:
                header = []
            cols = len(header)
            idx = {h: i for i, h in enumerate(header)}
            for row in rdr:
                rows += 1
                missing[name] += sum(1 for c in row if c.strip() == "")
                if name == "txs_features.csv":
                    f_ids.add(row[idx["txId"]])
                elif name == "txs_classes.csv":
                    c_ids.add(row[idx["txId"]])
                    v = row[idx["class"]].strip()
                    class_counts[v] = class_counts.get(v, 0) + 1
                elif name == "txs_edgelist.csv":
                    a, b = row[idx["txId1"]].strip(), row[idx["txId2"]].strip()
                    endpoints.add(a)
                    endpoints.add(b)
                    if a == b:
                        self_loops += 1
                    if (a, b) in edge_set:
                        dupes += 1
                    else:
                        edge_set.add((a, b))
        shapes[name] = (rows, cols)
        status = "OK" if (rows == exp["rows"] and cols == exp["cols"]) else "MISMATCH"
        if status != "OK":
            ok = False
        print(f"  {name}: rows={rows} (exp {exp['rows']}), cols={cols} (exp {exp['cols']}) -> {status}")

    print("\n== missing values ==")
    for name in EXPECTED:
        n = missing[name]
        print(f"  {name}: {n} missing cells" + ("" if n == 0 else "  <-- CHECK"))

    print("\n== class distribution (txs_classes.csv) ==")
    for c, exp_n in EXPECTED_CLASSES.items():
        n = class_counts.get(str(c), 0)
        status = "OK" if n == exp_n else "MISMATCH"
        if status != "OK":
            ok = False
        print(f"  class {c} ({CLASS_NAMES[c]}): {n} (exp {exp_n}) -> {status}")
    extra = sorted(v for v in class_counts if v not in {str(c) for c in EXPECTED_CLASSES})
    if extra:
        print(f"  unexpected class values: {extra}")
        ok = False

    print("\n== txId consistency ==")
    print(f"  features txIds: {len(f_ids)}, classes txIds: {len(c_ids)}")
    print(f"  features == classes txId sets: {f_ids == c_ids}")
    if f_ids != c_ids:
        ok = False
    dangling = endpoints - f_ids
    print(f"  edgelist endpoints: {len(endpoints)} unique, dangling (not in features): {len(dangling)}")
    print(f"  self-loops: {self_loops}, duplicate edges: {dupes}")
    if dangling or self_loops or dupes:
        ok = False

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ---------------------------------------------------------------- entry point

def main() -> int:
    force_stdlib = "--no-pandas" in sys.argv
    pd = None
    pd_err = ""
    if not force_stdlib:
        try:
            import pandas as _pd  # noqa: F401
            pd = _pd
        except ImportError as exc:
            pd_err = str(exc).strip().splitlines()[0] if str(exc).strip() else repr(exc)

    if pd is not None:
        print("engine: pandas")
        return verify_pandas(pd)

    print("engine: pure-stdlib fallback (pandas could not be imported)")
    if pd_err:
        print(f"        import failed: {pd_err[:160]}")
    print("        (same checks, slower — see SETUP.md if a Windows policy blocks pandas)")
    return verify_stdlib()


if __name__ == "__main__":
    sys.exit(main())
