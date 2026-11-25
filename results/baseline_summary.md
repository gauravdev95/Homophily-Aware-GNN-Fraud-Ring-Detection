# Baseline Models Summary — Elliptic++ Transactions

*Phase 3 / Step 5 · temporal splits (train 1–34, val 35–41, test 42–49) · supervised on labelled nodes only · no SMOTE, no oversampling · class weighting only · accuracy never headlined.*

## 1. Setup

- Features: all 183 (Time step + 93 local + 72 aggregate + 17 named stats).
- Missing values: median imputation fit on TRAIN only + one binary missingness-indicator column for the 17 named-stat block.
- Standardisation: StandardScaler fit on train only; applied to LogReg/MLP only (tree models use unscaled features).
- Unknown nodes (class 3) are kept in the dataset but have no target — they take no part in supervised training/evaluation.
- Model selection on VAL using PR-AUC; final numbers on TEST.

## 2. Results (PR-AUC primary)

| Model | Val PR-AUC | Test PR-AUC | Test ROC-AUC | Test F1@0.5 | Test precision | Test recall |
|---|---|---|---|---|---|---|
| xgboost | 0.9344 | 0.5652 | 0.8542 | 0.5928 | 0.7615 | 0.4853 |
| random_forest | 0.9277 | 0.5407 | 0.8346 | 0.5860 | 0.9609 | 0.4216 |
| mlp | 0.6404 | 0.3948 | 0.8541 | 0.3432 | 0.6940 | 0.2279 |
| logreg | 0.3823 | 0.2025 | 0.8571 | 0.2144 | 0.1238 | 0.7990 |
| dummy_stratified | 0.0882 | 0.0461 | 0.4991 | 0.0644 | 0.0455 | 0.1103 |

**Best model (by val PR-AUC): xgboost** (val PR-AUC 0.9344, test PR-AUC 0.5652).

## 3. Confusion matrices (test, threshold 0.5)

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| xgboost | 8,371 | 62 | 210 | 198 |
| random_forest | 8,426 | 7 | 236 | 172 |
| mlp | 8,392 | 41 | 315 | 93 |
| logreg | 6,126 | 2,307 | 82 | 326 |
| dummy_stratified | 7,488 | 945 | 363 | 45 |

## 4. Figures

- `baselines_pr_curves.png`
- `baselines_roc_curves.png`
- `baselines_importance_random_forest.png`
- `baselines_importance_xgboost.png`
- `baselines_confusion_matrices.png`

## 5. Reading the results

- **Temporal gap is real.** Every model drops sharply from val (steps 35–41)
  to test (steps 42–49): XGBoost 0.934 → 0.565, Random Forest 0.928 → 0.541.
  Transactions in the future are genuinely harder to score — this is
  exactly what the temporal split is designed to expose, and a random split
  would have hidden it.
- **ROC-AUC flatters on imbalanced data.** LogReg has the highest test
  ROC-AUC (0.8571) but the second-worst PR-AUC (0.2025). With only 2.23%
  positives, PR-AUC is the honest metric; ROC-AUC is reported only for
  completeness.
- **Trees win.** Gradient boosting and random forests clearly beat the
  linear model and the MLP on these features, consistent with the
  heterogeneous feature scales (standardised locals vs raw aggregates).
- **What Step 6 must beat:** XGBoost test PR-AUC **0.5652** — and the
  structural baseline from notebook 03 (majority-neighbour, transductive
  PR-AUC 0.3932 on covered test nodes) is already cleared by every
  non-dummy model.
