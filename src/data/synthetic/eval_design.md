# Evaluation Design Decisions — SFP Research

Design decisions and known confounds for using the synthetic dataset in SFP evaluation.
Companion to `synth_data_structure.md`.

---

## 1. Using `claims_all.csv` as an Evaluation Dataset

### The question

All model versions (v1, v2a, v2b, v3a, v3b) are scored on **all 70,000 rows** — including
rows that were in each model's own training window. Is this a problem?

### Decision

Separate the two use cases:

| Use case | Training rows OK? | Rationale |
|---|---|---|
| SFP symptom analysis (score drift, decision rate inflation, label mechanism bias) | Yes | Need all versions scored on the same claims to track loop progression across time |
| Model performance evaluation (AUC, precision/recall, calibration) | No — data leakage | Use OOT holdout windows only |

**`claims_all.csv` must not be used as a unified evaluation dataset for performance metrics.**
For Build 03 (Unbiased Evaluation), restrict to each model's designated OOT window.

---

## 2. Cross-Model Common Holdout

### The question

Each model version has its own OOT holdout at a different time period (v1: May–Oct 2021;
v2a: May–Oct 2024). Raw performance metrics across these windows are not comparable.
Is a common holdout needed, and does one exist?

### What already exists

Every row in `claims_all.csv` carries scores for all versions. The **v2a OOT window
(2024-05 to 2024-10, ~4,051 rows)** is the natural common evaluation window:

- After v1's training window (2016–2021)
- After v2a's training window (2022–Apr 2024)
- All model version scores present

This means v1, v2a, v2b, v3a, and v3b can all be compared on the same 4,051 claims —
a genuine out-of-time window for every version simultaneously.

### Remaining limitation

Labels in this window (`model_v1_observed_outcome`) are SFP-contaminated: scrapped cars
carry forced label = 1 regardless of true outcome. Standard AUC against this label is biased.
Build 03 must apply selective-labels correction (IPS weighting) before any performance
comparison across versions on this common window.

---

## 3. Enrichment Data Inflation and the OOT Holdout

### The question

`used_car_price_index` is a time-varying feature reflecting UK used-car market conditions.
It shifts `vehicle_value`, which in turn shifts `repair_to_value_ratio` (the strongest DGP
predictor). Does using the most recent data as the OOT holdout conflate market effects with
SFP effects?

### `used_car_price_index` values across the synthetic timeline

| Year | Index | Context |
|---|---|---|
| 2016 | 1.00 | Baseline |
| 2020 | 0.97 | COVID demand drop |
| 2021 | 1.18 | Post-lockdown rebound |
| 2022 | 1.28 | Semiconductor shortage peak |
| 2023 | 1.18 | Normalising |
| 2024 | 1.10 | Further normalisation (OOT period) |

v2a's training window (2022–Apr 2024) spans the price peak and partial deflation.
The OOT (May–Oct 2024) sits at index ≈ 1.10, below the training average.

### Directional effects

Lower price index → lower `vehicle_value` → higher `repair_to_value_ratio` → more genuine
total losses. This is the **opposite direction to SFP**:

```
SFP effect:     inflates predicted scrap rate (model bias ↑)
Price deflation: increases genuine TL rate (DGP reality ↑)
```

This means the two effects partially offset in the OOT window. More importantly, if v2a's
decision rate in OOT is still elevated above v1's despite the market moving toward more
genuine total losses (which would justify a higher rate), the *excess* is a **stronger**
attribution to SFP contamination.

### Required controls in analysis

1. Include `used_car_price_index` as a control variable when comparing decision rates
   across time periods. Do not compare raw rates across different year bands without
   adjusting for the price index.

2. When stratifying OOT vs training performance, report whether the gap is consistent
   across `used_car_price_index` bins. A gap that holds at constant index is cleaner
   SFP evidence than one that only appears at extremes.

3. Document this confound explicitly in the dissertation. The price index is an
   observed feature in the model, so partial self-correction occurs — but it was
   not designed as a debiasing mechanism.

### Decision

Using the most recent data (2024) as the OOT holdout is appropriate. The temporal
distribution shift introduced by price index variation is real and expected — it mirrors
the real Insurance Company. setting where enrichment values change independently of retraining.
The SFP and inflation effects operate in opposite directions, which if anything makes
detected SFP symptoms *more conservative* (harder to see), not spurious.

---

## Summary Table

| Design question | Decision | Key constraint |
|---|---|---|
| Use `claims_all.csv` for evaluation? | Only for SFP symptom analysis, not performance metrics | Training rows cause leakage if used for AUC |
| Common holdout for cross-version comparison? | v2a OOT (May–Oct 2024, ~4k rows) serves as de facto common holdout | Labels still SFP-contaminated; requires Build 03 correction |
| Most recent data as OOT despite inflation? | Yes — appropriate | Price deflation moves DGP in opposite direction to SFP; control for `used_car_price_index` in cross-time comparisons |
