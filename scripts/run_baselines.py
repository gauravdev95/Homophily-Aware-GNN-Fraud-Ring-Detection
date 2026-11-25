"""Phase 3 / Step 5 — Baseline models (LogReg / RandomForest / XGBoost /
MLP / Dummy) on the Elliptic++ temporal splits.

Supervised training on LABELED nodes only (illicit vs licit). Unknown nodes
(class 3) stay in the dataset — they are never deleted — but have no target
and are excluded from training/evaluation. Splits are temporal
(train steps 1-34, val 35-41, test 42-49); NEVER random.

Missing values: median imputation fit on TRAIN only + one binary
missingness-indicator column for the 17 named-stat block (the same 965
rows miss all 17). The label is never used for imputation.
Standardisation: StandardScaler fit on train only, applied to LogReg/MLP;
tree models use the unscaled (imputed) features.

No SMOTE, no oversampling, no resampling of any kind — class weighting
only. Model selection on VAL using PR-AUC; final numbers reported on TEST.
Primary metric PR-AUC (average precision). Accuracy is never headlined
(only 2.23% illicit — accuracy is meaningless here).

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
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "elliptic_pp"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIG_DIR = RESULTS / "figures" / "baselines"

# the 17 named statistical columns (all missing-value cells live here)
NAMED_STATS = [
    "in_txs_degree", "out_txs_degree", "total_BTC", "fees", "size",
    "num_input_addresses", "num_output_addresses",
    "in_BTC_min", "in_BTC_max", "in_BTC_mean", "in_BTC_median", "in_BTC_total",
    "out_BTC_min", "out_BTC_max", "out_BTC_mean", "out_BTC_median",
    "out_BTC_total",
]
RANDOM_STATE = 42


# ---------------------------------------------------------------- data

def load_all() -> pd.DataFrame:
    """Load features + labels + temporal splits, merged on txId.

    Returns one row per node with all 183 features, `class`
    (1=illicit, 2=licit, 3=unknown) and `split` (train/val/test).
    Unknown nodes are kept in the frame (no target, never deleted).
    """
    feats = pd.read_csv(RAW / "txs_features.csv")
    classes = pd.read_csv(RAW / "txs_classes.csv")
    splits = pd.read_csv(PROCESSED / "node_splits.csv",
                         usecols=["txId", "split"])
    df = feats.merge(classes, on="txId", how="inner")
    df = df.merge(splits, on="txId", how="inner")
    assert len(df) == 203_769, f"unexpected row count {len(df)}"
    assert set(NAMED_STATS) <= set(df.columns), "named-stat columns missing"
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All 183 feature columns (everything except ids/labels/split)."""
    return [c for c in df.columns if c not in ("txId", "class", "split")]


def split_xy(df: pd.DataFrame, feat_cols: list[str]):
    """Return {split: (X, y)} for LABELED nodes only (illicit=1, licit=0).

    Unknown nodes have no target and are excluded from X/y — they remain
    in `df`, they are just not part of supervised training/evaluation.
    """
    out = {}
    for name in ("train", "val", "test"):
        sub = df[df["split"] == name]
        lab = sub[sub["class"] < 3]
        X = lab[feat_cols].to_numpy(dtype=np.float64)
        y = (lab["class"] == 1).astype(int).to_numpy()
        out[name] = (X, y)
    return out


# ---------------------------------------------------------------- preprocessing

def fit_preprocess(X_train: np.ndarray):
    """Median imputer fit on TRAIN only. Returns (imputer, named_idx)."""
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train)
    return imputer


def missing_indicator(X_raw: np.ndarray, feat_cols: list[str]) -> np.ndarray:
    """1 if the row is missing any of the 17 named-stat values, else 0."""
    idx = [feat_cols.index(c) for c in NAMED_STATS]
    return np.isnan(X_raw[:, idx]).any(axis=1).astype(np.float64)


def transform(X_raw: np.ndarray, feat_cols: list[str], imputer,
              scaler=None) -> np.ndarray:
    """Impute (train-fit) + append missingness indicator; optionally scale."""
    X = imputer.transform(X_raw)
    ind = missing_indicator(X_raw, feat_cols).reshape(-1, 1)
    X = np.hstack([X, ind])
    if scaler is not None:
        X = scaler.transform(X)
    return X


def fit_scaler(X_train_imputed: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train_imputed)
    return scaler


# ---------------------------------------------------------------- models

def get_models(n_licit_train: int, n_illicit_train: int) -> dict:
    """Model name -> (estimator, needs_scaling). Class weighting only."""
    spw = n_licit_train / n_illicit_train
    return {
        "logreg": (LogisticRegression(class_weight="balanced", max_iter=1000,
                                      random_state=RANDOM_STATE), True),
        "random_forest": (RandomForestClassifier(
            class_weight="balanced_subsample", n_estimators=300, n_jobs=-1,
            random_state=RANDOM_STATE), False),
        "xgboost": (XGBClassifier(
            scale_pos_weight=spw, n_estimators=300, max_depth=6,
            learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", n_jobs=-1, eval_metric="logloss",
            random_state=RANDOM_STATE), False),
        "mlp": (MLPClassifier(hidden_layer_sizes=(128, 64),
                              early_stopping=True, max_iter=300,
                              random_state=RANDOM_STATE), True),
        "dummy_stratified": (DummyClassifier(strategy="stratified",
                                             random_state=RANDOM_STATE), False),
    }


def evaluate(model, X: np.ndarray, y: np.ndarray) -> dict:
    """PR-AUC, ROC-AUC, F1 / precision / recall at 0.5, confusion matrix."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:  # pragma: no cover - all models used here expose predict_proba
        proba = model.decision_function(X)
        proba = (proba - proba.min()) / (proba.max() - proba.min() + 1e-12)
    pred = (proba > 0.5).astype(int)
    cm = confusion_matrix(y, pred).tolist()
    return {
        "n": int(len(y)),
        "n_positive": int(y.sum()),
        "pr_auc": float(average_precision_score(y, proba)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "f1_at_0.5": float(f1_score(y, pred, zero_division=0)),
        "precision_at_0.5": float(precision_score(y, pred, zero_division=0)),
        "recall_at_0.5": float(recall_score(y, pred, zero_division=0)),
        "confusion_matrix": cm,  # [[TN, FP], [FN, TP]]
        "_proba": proba,
        "_y": y,
    }


def train_all(Xy: dict, feat_cols: list[str]) -> dict:
    """Fit every model on train, evaluate on val and test. Returns results."""
    X_train_raw, y_train = Xy["train"]
    imputer = fit_preprocess(X_train_raw)
    # scaler is fit on the imputed + indicator train matrix
    X_train_imp = transform(X_train_raw, feat_cols, imputer)
    scaler = fit_scaler(X_train_imp)

    X = {}
    for name in ("train", "val", "test"):
        X_raw, _ = Xy[name]
        X[name] = {
            "scaled": transform(X_raw, feat_cols, imputer, scaler),
            "unscaled": transform(X_raw, feat_cols, imputer),
        }
    n_licit = int((y_train == 0).sum())
    n_illicit = int((y_train == 1).sum())
    print(f"train labelled: {len(y_train):,} "
          f"(illicit {n_illicit:,}, licit {n_licit:,}); "
          f"scale_pos_weight={n_licit / n_illicit:.3f}")

    models = get_models(n_licit, n_illicit)
    results = {}
    fitted = {}
    for name, (model, needs_scaling) in models.items():
        key = "scaled" if needs_scaling else "unscaled"
        t0 = time.time()
        model.fit(X["train"][key], y_train)
        fit_s = time.time() - t0
        fitted[name] = model
        entry = {"scaled": needs_scaling, "fit_seconds": round(fit_s, 1)}
        for split in ("val", "test"):
            m = evaluate(model, X[split][key], Xy[split][1])
            entry[split] = {k: v for k, v in m.items()
                            if not k.startswith("_")}
            entry[f"_{split}_proba"] = m["_proba"]
            entry[f"_{split}_y"] = m["_y"]
        results[name] = entry
        print(f"[{name:17s}] fit {fit_s:6.1f}s  "
              f"val PR-AUC={entry['val']['pr_auc']:.4f}  "
              f"test PR-AUC={entry['test']['pr_auc']:.4f}")
    return results, fitted


# ---------------------------------------------------------------- figures

def make_figures(results: dict, feat_cols: list[str],
                 fitted: dict) -> list[str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    names = list(results)
    colors = {"logreg": "#4C78A8", "random_forest": "#F58518",
              "xgboost": "#54A24B", "mlp": "#E45756",
              "dummy_stratified": "#9E9E9E"}

    # 1. PR curves on test
    fig, ax = plt.subplots(figsize=(8, 6))
    for name in names:
        y = results[name]["_test_y"]
        p = results[name]["_test_proba"]
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, label=f"{name} "
                f"(AP={results[name]['test']['pr_auc']:.3f})",
                color=colors[name])
    base = results["dummy_stratified"]["test"]["n_positive"] / \
        results["dummy_stratified"]["test"]["n"]
    ax.axhline(base, color="black", ls="--", lw=1,
               label=f"chance (illicit rate {base:.3f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision–Recall curves (test, steps 42–49)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fp = FIG_DIR / "baselines_pr_curves.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    paths.append(str(fp))

    # 2. ROC curves on test
    fig, ax = plt.subplots(figsize=(8, 6))
    for name in names:
        y = results[name]["_test_y"]
        p = results[name]["_test_proba"]
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name} "
                f"(AUC={results[name]['test']['roc_auc']:.3f})",
                color=colors[name])
    ax.plot([0, 1], [0, 1], color="black", ls="--", lw=1, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC curves (test, steps 42–49)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fp = FIG_DIR / "baselines_roc_curves.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    paths.append(str(fp))

    # 3. top-15 importances for the tree models
    all_feats = feat_cols + ["missing_named_stats"]
    for name in ("random_forest", "xgboost"):
        model = fitted[name]
        imp = model.feature_importances_
        top = np.argsort(imp)[::-1][:15]
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.barh([all_feats[i] for i in top][::-1], imp[top][::-1],
                color=colors[name])
        ax.set_xlabel("importance")
        ax.set_title(f"Top-15 feature importances — {name} (test split model)")
        fig.tight_layout()
        fp = FIG_DIR / f"baselines_importance_{name}.png"
        fig.savefig(fp, dpi=150)
        plt.close(fig)
        paths.append(str(fp))

    # 4. confusion matrices on test (one panel per model)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    for ax, name in zip(axes, names):
        cm = np.array(results[name]["test"]["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        fontsize=11,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["pred licit", "pred illicit"])
        ax.set_yticks([0, 1], ["true licit", "true illicit"])
        ax.set_title(f"{name}\nF1@0.5="
                     f"{results[name]['test']['f1_at_0.5']:.3f}")
    axes[len(names)].axis("off")
    fig.suptitle("Confusion matrices on test (threshold 0.5)", fontsize=13)
    fig.tight_layout()
    fp = FIG_DIR / "baselines_confusion_matrices.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    paths.append(str(fp))

    return paths


# ---------------------------------------------------------------- reports

def public_results(results: dict) -> dict:
    """Strip proba/y arrays; keep JSON-serialisable metrics only."""
    out = {}
    for name, entry in results.items():
        out[name] = {
            "scaled_features": entry["scaled"],
            "fit_seconds": entry["fit_seconds"],
            "val": entry["val"],
            "test": entry["test"],
        }
    return out


def write_reports(results: dict, feat_cols: list[str], fig_paths: list[str],
                  best_model: str) -> None:
    pub = public_results(results)
    (RESULTS / "baseline_results.json").write_text(json.dumps(pub, indent=2))

    order = sorted(pub, key=lambda n: pub[n]["val"]["pr_auc"], reverse=True)
    lines = [
        "# Baseline Models Summary — Elliptic++ Transactions",
        "",
        "*Phase 3 / Step 5 · temporal splits (train 1–34, val 35–41, "
        "test 42–49) · supervised on labelled nodes only · no SMOTE, "
        "no oversampling · class weighting only · accuracy never headlined.*",
        "",
        "## 1. Setup",
        "",
        "- Features: all 183 (Time step + 93 local + 72 aggregate + "
        "17 named stats).",
        "- Missing values: median imputation fit on TRAIN only + one "
        "binary missingness-indicator column for the 17 named-stat block.",
        "- Standardisation: StandardScaler fit on train only; applied to "
        "LogReg/MLP only (tree models use unscaled features).",
        "- Unknown nodes (class 3) are kept in the dataset but have no "
        "target — they take no part in supervised training/evaluation.",
        "- Model selection on VAL using PR-AUC; final numbers on TEST.",
        "",
        "## 2. Results (PR-AUC primary)",
        "",
        "| Model | Val PR-AUC | Test PR-AUC | Test ROC-AUC | Test F1@0.5 | "
        "Test precision | Test recall |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in order:
        v, t = pub[name]["val"], pub[name]["test"]
        lines.append(
            f"| {name} | {v['pr_auc']:.4f} | {t['pr_auc']:.4f} | "
            f"{t['roc_auc']:.4f} | {t['f1_at_0.5']:.4f} | "
            f"{t['precision_at_0.5']:.4f} | {t['recall_at_0.5']:.4f} |")
    lines += [
        "",
        f"**Best model (by val PR-AUC): {best_model}** "
        f"(val PR-AUC {pub[best_model]['val']['pr_auc']:.4f}, "
        f"test PR-AUC {pub[best_model]['test']['pr_auc']:.4f}).",
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
        "## 4. Figures",
        "",
    ]
    for fp in fig_paths:
        lines.append(f"- `{Path(fp).name}`")
    lines.append("")
    (RESULTS / "baseline_summary.md").write_text("\n".join(lines))


# ---------------------------------------------------------------- main

def main() -> int:
    print("== loading data ==")
    df = load_all()
    feat_cols = feature_columns(df)
    print(f"  rows: {len(df):,}  features: {len(feat_cols)} "
          f"(+1 missingness indicator at train time)")
    Xy = split_xy(df, feat_cols)
    for name in ("train", "val", "test"):
        X, y = Xy[name]
        print(f"  {name:5s} labelled: {len(y):,} "
              f"(illicit {int(y.sum()):,}, licit {int((y == 0).sum()):,})")

    print("\n== training (selection on val PR-AUC) ==")
    results, fitted = train_all(Xy, feat_cols)

    best = max(results, key=lambda n: results[n]["val"]["pr_auc"])
    print(f"\nbest model by val PR-AUC: {best} "
          f"({results[best]['val']['pr_auc']:.4f})")

    print("\n== figures ==")
    fig_paths = make_figures(results, feat_cols, fitted)
    for fp in fig_paths:
        print(f"  {fp}")

    write_reports(results, feat_cols, fig_paths, best)
    print("\nreports: results/baseline_results.json, "
          "results/baseline_summary.md")
    print("\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
