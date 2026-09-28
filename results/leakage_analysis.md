# Leakage Analysis — Elliptic++ Transactions Dataset

*Phase 2 · evidence gathered by `notebooks/01_data_exploration.ipynb` (§13). Read-only checks on
the released CSVs. A check "passing" means no observable leakage in the files — it does not
certify the dataset construction pipeline.*

## Method

Five cheap, falsifiable checks were run; anything they cannot observe is listed under
§4 as an open question rather than assumed safe.

## 1. Check results (computed)

| # | Check | Result |
|---|---|---|
| (a) | Max \|corr(feature, illicit-indicator)\| over 183 features | 0.2675 (`Aggregate_feature_57`) |
| (a) | Max \|corr(feature, licit-indicator)\| over 183 features | 0.5166 (`Local_feature_53`) |
| (b) | Features with a near-duplicate (\|r\|>0.999) | 10 |
| (c) | Zero-variance features | 0 [] |
| (d) | corr(time step, per-step illicit rate) | +0.026 |
| (e) | Named-stat-missing rows by class | illicit=0, licit=519, unknown=446 |

Single-feature label association is strong (0.517) — a single feature carries substantial label signal. No feature is a deterministic function of the
label in the released files, there are no duplicate/constant columns, and no `txId`
appears in two time steps (ID sets are unique per file, §2 of the EDA).

Check (d) is **temporal shift, not leakage**: the illicit rate varies across the 49 steps
(correlation +0.026). It does not leak the future into features,
but a random split would leak the future into *evaluation* — hence the temporal-split
recommendation in `eda_summary.md` §8.

Check (e): the 965 named-stat-missing rows skew licit/unknown vs the overall prior. This is a
missingness bias to handle explicitly, not leakage — but a model must not learn
"missing ⇒ class" as a shortcut without the missingness policy being a deliberate,
documented choice.

## 2. What "no observable leakage" covers

- No label column is present in the feature file; `Time step` is a legitimate temporal
  attribute, not derived from the label.
- Edge direction was validated independently (exactly 0 cross-step and 0 backward-time edges
  out of 234,355; the 49 weak components map 1:1 onto the 49 time steps), so the
  graph topology itself cannot smuggle label information across time beyond the money flow.

## 3. What the CSVs cannot rule out (open questions for Phase 3)

1. **Aggregate construction semantics.** Cross-*time-step* leakage via neighborhoods is now
   structurally ruled out (exactly 0 cross-step edges verified in §8/§10 of the EDA), but the
   released CSVs contain no construction metadata for the 72 anonymized aggregates and 17
   named statistics: the exact neighborhood definition and aggregation functions are
   unverified. Check the dataset paper / construction code before modeling, and run
   with/without-aggregate ablations regardless.
2. **Neighborhood label leakage.** Same concern at the graph level: if aggregates were built
   from labeled neighbors without masking, they may carry neighbor-label signal.
3. **Survivorship/selection bias** in which transactions received labels (unknown = 77%) is
   unknown and unknowable from the files; treat unknown as unlabeled, not as negative.

## 4. Recommendations

1. Default to **temporal splits** (past → future); never random-split.
2. Decide the **aggregate-feature policy** (keep / drop / ablate) only after the
   construction-code check in (3.1); document the decision.
3. Add an explicit **missingness indicator** if the named-stat columns are kept, so the 965-row
   co-missing block is modeled deliberately.
4. Report metrics **per time step** in Phase 3+ to detect temporal degradation early.
5. Re-run checks (a)–(c) on any derived feature set before training — leakage is a property
   of the pipeline, not a one-time certificate.
