# Setup Guide

This guide gets the project running from scratch on **Windows, macOS, or Linux**.
Two paths: the automatic one (recommended) and the manual one.

---

## Option A — Automatic (recommended)

One command does everything: virtualenv, dependencies, dataset download, verification.

```bash
git clone https://github.com/gauravdev95/Homophily-Aware-GNN-Fraud-Ring-Detection.git
cd Homophily-Aware-GNN-Fraud-Ring-Detection
python start.py
```

`start.py` needs nothing installed except Python itself (stdlib only). It will:

1. Check Python ≥ 3.10
2. Create `.venv/` in the project folder (skipped if it already exists)
3. Install `requirements.txt` into it
4. Download any missing dataset files into `data/raw/elliptic_pp/`
   - `txs_features.csv` (663 MB) → from the
     [data-v1 release](https://github.com/gauravdev95/Homophily-Aware-GNN-Fraud-Ring-Detection/releases/tag/data-v1),
     with Google Drive as automatic fallback
   - `txs_classes.csv` / `txs_edgelist.csv` → already in the repo; re-downloaded
     from the repo/Drive only if missing
5. Run `scripts/verify_data.py` to sanity-check the dataset

Useful flags:

| Flag | Effect |
|---|---|
| `--no-download` | Fail if data files are missing instead of downloading them |
| `--no-verify` | Skip the verification step |
| `--reinstall` | Force-reinstall all dependencies |

After it finishes:

```bash
# activate the environment (each new terminal)
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

---

## Option B — Manual (pip install)

If you prefer to do each step yourself:

```bash
git clone https://github.com/gauravdev95/Homophily-Aware-GNN-Fraud-Ring-Detection.git
cd Homophily-Aware-GNN-Fraud-Ring-Detection

# 1. virtualenv
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 2. dependencies (this is the plain `pip install` path)
pip install -r requirements.txt

# 3. dataset — two small files are already in the repo; fetch the big one:
#    direct link (no login needed):
#    https://github.com/gauravdev95/Homophily-Aware-GNN-Fraud-Ring-Detection/releases/download/data-v1/txs_features.csv
#    and place it at data/raw/elliptic_pp/txs_features.csv
#    (official upstream source: see data/README.md)

# 4. verify (read-only, never modifies raw data)
python scripts/verify_data.py
```

`pip install -r requirements.txt` works on its own in any Python 3.10+
virtualenv — that is the canonical dependency list. `start.py` is just
automation around it.

---

## Requirements

- **Python 3.10+** ([download](https://www.python.org/downloads/))
  — check with `python --version`
- **git**
- **~2 GB free disk** (~700 MB dataset + ~1 GB virtualenv)
- Internet access (PyPI + dataset download)

`requirements.txt` installs: pandas, numpy, scipy, matplotlib,
scikit-learn, nbformat (plus their transitive deps).

---

## Dataset layout

After setup, `data/raw/elliptic_pp/` must contain:

| File | Size | Source |
|---|---|---|
| `txs_features.csv` | 663 MB | [data-v1 release](https://github.com/gauravdev95/Homophily-Aware-GNN-Fraud-Ring-Detection/releases/tag/data-v1) (direct link, no login) |
| `txs_classes.csv` | 2.4 MB | committed in this repo |
| `txs_edgelist.csv` | 4.5 MB | committed in this repo |

The canonical upstream source is the official Elliptic++ Google Drive folder —
see `data/README.md`. Raw files are never modified by any script in this repo.

`scripts/verify_data.py` checks: file sizes, row/column counts
(203,769 / 203,769 / 234,355 rows), class distribution
(4,545 illicit / 42,019 licit / 157,205 unknown), `txId` consistency across
files, and edgelist integrity.

---

## Troubleshooting

**`Python 3.10+ required`**
→ Install a newer Python from python.org and re-run. On Debian/Ubuntu:
`sudo apt install python3-venv` if venv creation fails.

**`pip install` is slow / times out**
→ Re-run `python start.py` (or `pip install -r requirements.txt`) — pip resumes
cleanly. Big wheels (scipy, pandas) just take a few minutes on slow networks.

**Dataset download fails**
→ Download manually from the links in the table above (or the Drive folder in
`data/README.md`) and place the files in `data/raw/elliptic_pp/`, then re-run
with `python start.py --no-download`.

**`verify_data.py` reports MISMATCH**
→ A file is corrupted or truncated. Delete it from `data/raw/elliptic_pp/`
and let `start.py` re-download it.

**Windows: `python` not recognized**
→ Reinstall Python with **"Add python.exe to PATH"** checked, then open a new
terminal.

---

## What's next

- `notebooks/01_data_exploration.ipynb` — exploratory data analysis (Phase 2, complete)
- `results/eda_summary.md` — key findings and leakage analysis
- Phase 3 (modeling) has not started yet — see `results/eda_summary.md` for the
  recommended next steps.
