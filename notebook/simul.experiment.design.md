# Simulation Experiment Design
**SFP Loop Detection — Synthetic Data Evaluation Strategy**

---

## Overview

The synthetic dataset (`claims_all.csv`, 70,000 rows, 2016–2024) contains predictions from
five model versions: v1, v2a, v2b, v3a, v3b. Each version was scored retroactively on every
row, regardless of training window. This document records three evaluation design decisions
that arise from this structure.

---

## Design Decision 1 — When is `claims_all.csv` safe to use as an evaluation set?

### Problem

Every model version's scores exist on rows that were used to train that same model.
For example, v2a was trained on 2022–Apr 2024 data, but `model_v2a_score` is present for
all 70k rows including its own training window. Evaluating v2a performance on those rows
is data leakage.

### Decision: separate by use case

| Use case | Use training rows? | Reason |
|---|---|---|
| SFP symptom analysis — score drift, decision rate inflation, label mechanism bias | Yes | Comparing version scores on the *same* claims is required to track how the loop evolves across model generations |
| Model performance evaluation — AUC, precision, recall, calibration | No — leakage | Must restrict to each model's OOT holdout window |

**Rule:** `claims_all.csv` as a whole is appropriate for SFP pattern analysis only.
For any performance metric, use the per-model OOT windows defined in `synth_data_structure.md`.

---

## Design Decision 2 — Is a common holdout needed for cross-model comparison?

### Problem

Each model version's OOT window covers a different calendar period:

| Model | Training window | OOT holdout |
|---|---|---|
| v1 | 2016-01 → 2021-04 | 2021-05 → 2021-10 |
| v2a | 2022-01 → 2024-04 | 2024-05 → 2024-10 |
| v2b, v3a, v3b | similar to v2a | 2024-05 → 2024-10 |

Comparing raw AUC from v1's OOT (2021) against v2a's OOT (2024) is not valid — different
claim populations, different market conditions, different label contamination levels.

### Decision: the v2a OOT window serves as the de facto common holdout

The **2024-05 to 2024-10 window (~4,051 rows)** satisfies three requirements simultaneously:

1. Outside v1's training window (2016–2021) — v1 has not seen these claims in training
2. Outside v2a's training window (2022–Apr 2024) — v2a has not seen them either
3. All model version scores are present in `claims_all.csv` for this window

No additional dataset needs to be constructed. This window is the natural common evaluation
ground where v1, v2a, v2b, v3a, and v3b can all be compared on equal footing.

### Remaining constraint: SFP-contaminated labels

Labels in this window (`model_v1_observed_outcome`) are not clean ground truth. Claims
scrapped by v1 carry a forced label of 1 — never verified in a garage. Standard AUC
computed against these labels is biased upward for models that mimic v1's scrap decisions.

**Build 03 (Unbiased Evaluation) must apply selective-labels correction (IPS weighting)
to this window before any cross-version performance comparison.**

---

## Design Decision 3 — Does enrichment data inflation confound the OOT holdout?

### Problem

`used_car_price_index` is a time-varying feature that reflects UK used-car market conditions.
It directly affects `vehicle_value` and therefore `repair_to_value_ratio` — the strongest
predictor in the DGP. The index is not constant across the evaluation timeline:

| Year | Index | Market context |
|---|---|---|
| 2016 | 1.00 | Baseline |
| 2020 | 0.97 | COVID-19 demand drop |
| 2021 | 1.18 | Post-lockdown rebound |
| 2022 | 1.28 | Semiconductor shortage peak |
| 2023 | 1.18 | Normalising |
| 2024 | 1.10 | Further normalisation — OOT period |

v2a's training window (2022–Apr 2024) straddles the price peak and partial recovery.
The OOT (May–Oct 2024) sits at index ≈ 1.10, **below the training-period average**.

### Directional effect of price deflation

Higher `used_car_price_index` → higher `vehicle_value` → lower `repair_to_value_ratio`
→ fewer genuine total losses (car is worth more; repair cost less likely to exceed its value).

In the OOT period, prices are lower than during training, which means:
- `repair_to_value_ratio` is higher in OOT
- Genuine total loss rate is higher in OOT than during peak-price training rows

This moves the DGP in the **opposite direction to SFP**:

```
SFP effect:        inflates predicted scrap rate (model bias, upward)
Price deflation:   increases genuine TL rate (real-world effect, also upward)
```

### Why this is not a disqualifying confound

Because the two effects push in the same direction (both increase scrap/TL rate), detecting
SFP on top of this background is actually more conservative — harder. If v2a's decision rate
in the OOT window still exceeds v1's after accounting for the price shift, that excess is a
stronger attribution to SFP contamination, not a weaker one.

The key risk is the opposite: **attributing SFP symptoms to price effects** when it is
actually the model's bias driving the elevated rate. This is why `used_car_price_index`
must be controlled for in any cross-time comparison.

### Required controls

1. Include `used_car_price_index` as a control variable when comparing decision rates
   across time periods. Never compare raw scrap rates across different year bands without
   adjusting for the index.

2. When stratifying OOT vs training-period results, check whether performance gaps are
   consistent across `used_car_price_index` bins. A gap that holds at constant index
   values is cleaner SFP evidence than one concentrated at index extremes.

3. The dissertation should document this confound explicitly. The index is an observed
   feature in the model input, so partial self-correction occurs at inference time — but
   this was not designed as a debiasing mechanism and does not fully remove the confound.

### Decision: using 2024 OOT is appropriate

The temporal distribution shift introduced by the price index mirrors the real Allianz
setting (enrichment tables are refreshed independently of model retraining). Using the
most recent data as the holdout is methodologically sound. The inflation effect and the
SFP effect operate in the same directional sense, making any detected SFP symptom a
conservative lower bound rather than an artefact.

---

## Summary

| Design question | Decision | Key constraint |
|---|---|---|
| Use `claims_all.csv` for evaluation? | SFP pattern analysis only; not for AUC/performance metrics | Training rows cause leakage if used for performance evaluation |
| Common holdout for cross-model comparison? | v2a OOT (May–Oct 2024, ~4k rows) is the de facto common window | Labels are SFP-contaminated; Build 03 IPS correction required |
| Most recent data as OOT despite inflation? | Yes — appropriate and conservative | Control for `used_car_price_index` in all cross-time comparisons; document as a known confound |

---

*Companion documents: `src/data/synthetic/synth_data_structure.md`, `src/data/synthetic/eval_design.md`*
