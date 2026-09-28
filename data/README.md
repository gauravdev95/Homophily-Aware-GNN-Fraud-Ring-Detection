# Data

## Source dataset: Elliptic++ (Transactions)

- **Official repository:** https://github.com/git-disl/EllipticPlusPlus
- **Paper:** Youssef Elmougy and Ling Liu, *"Demystifying Fraudulent Transactions and
  Illicit Nodes in the Bitcoin Network for Financial Forensics"*, KDD 2023
  ([arXiv:2306.06108](https://arxiv.org/abs/2306.06108))
- **Download location:** the official repo distributes the CSVs via Google Drive
  ("DATASET CAN BE FOUND HERE" in its README):
  https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l

### Files used in this project

| File | Rows | Columns | Description |
|---|---|---|---|
| `txs_features.csv` | 203,769 | 184 (`txId` + 183 features) | Feature matrix, one row per transaction |
| `txs_classes.csv` | 203,769 | 2 (`txId`, `class`) | Labels: `1` = illicit, `2` = licit, `3` = unknown |
| `txs_edgelist.csv` | 234,355 | 2 (`txId1`, `txId2`) | Directed money-flow edges, tx → tx |

Official statistics: 49 timesteps; 4,545 illicit / 42,019 licit / 157,205 unknown
transactions.

> The **Actors (wallet addresses) dataset** (`wallets_*.csv`, `AddrAddr_edgelist.csv`,
> …) is **intentionally not downloaded** — out of scope for this phase of the project.

### How to download

Option A — with [`gdown`](https://github.com/wkentaro/gdown):

```bash
pip install gdown
cd data/raw/elliptic_pp
gdown --fuzzy "https://drive.google.com/uc?id=19q09IFhfkOOBOXvn_dKhWjILJtjCcsjc" -O txs_features.csv
gdown --fuzzy "https://drive.google.com/uc?id=1DiBxn8TXdbJqoSw58pYUeaqO3oOKhuQO" -O txs_classes.csv
gdown --fuzzy "https://drive.google.com/uc?id=1Q2yG_CIDvfdGP-fKVPSw979EYgQukjz5" -O txs_edgelist.csv
```

Option B — open the
[Drive folder](https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l?usp=sharing)
in a browser and download the three `txs_*.csv` files into `data/raw/elliptic_pp/`.

### Verification

After downloading, sanity-check the raw files (read-only — **never modify raw data**):

```bash
python scripts/verify_data.py
```

Expected: `txs_features.csv` and `txs_classes.csv` have 203,769 data rows each,
`txs_edgelist.csv` has 234,355 data rows, `txId`s match across files, and class
counts are 4,545 / 42,019 / 157,205 for classes 1 / 2 / 3.

## Version control policy

Raw data is **excluded from git** (see the root `.gitignore`):

- `data/raw/**` — never committed (files are hundreds of MB; GitHub rejects files > 100 MB
  and Git LFS quotas make it impractical for a dataset of this size).
- `data/processed/**` — regenerated from raw; not committed.
- Only `.gitkeep` placeholders and this README are tracked under `data/`.

Anyone reproducing this work downloads the dataset from the official source above.
