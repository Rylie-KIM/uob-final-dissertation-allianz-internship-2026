> ⚠️ **WARNING — SIMULATION PURPOSE**
>
> This dataset is a **synthetic simulation** of real Insurance A Cop. UK motor claims data.
> The goal is to faithfully reproduce the structure, business logic, and label mechanisms
> of the actual production system so that the SFP detection framework can be developed and
> validated **before access to the real data is granted**.
>
> Every design decision here — data splits, scrapping thresholds, enrichment join logic,
> label mechanisms, model training windows — is chosen to match what actually happens at
> Insurance A Cop., as understood from business discussions. When real data arrives, the synthetic
> files (`claims_pre_v1.parquet`, `claims_v1_log.parquet`) should be swapped out and the
> framework should run unchanged.
>
> **Key constraint this simulation must respect:** In the real Insurance A Cop. data, there is no
> oracle or ground truth label. Pre-ML handler/engineer decisions (pre-model data) have
> been disposed of under data retention policy. Even if they existed, handler decisions
> would not constitute a clean oracle — only engineer physical inspections would, and it
> is not always possible to distinguish which type of decision was recorded. The synthetic
> data must therefore be designed so that all detection and analysis methods work **without
> access to any oracle column**. Any use of `garage_outcome` (the internal DGP variable)
> is for generation mechanics only and must not be exposed as a column or used in
> framework logic.

---

# Synthetic Dataset Schema
**UK Motor Insurance Claims — SFP Loop Research**
**Domain: Claims Operations — Total Loss Prediction (is the car repairable?)**
Total columns: 39 (claims table) | Output files: claims_pre_v1.parquet, claims_v1_log.parquet, vehicle_enrichment.parquet | Rows: 10,000

---

> **The SFP loop in this domain:**
> Model predicts `total_loss = 1` → car is scrapped immediately → outcome recorded as 1 (self-fulfilling).
> Model predicts `total_loss = 0` → car goes to garage → actual repair outcome observed.
> When v2 retrains on v1 log data, false-positive total-loss predictions from v1
> appear as confirmed positives — v2 learns to over-predict total loss.

---

## Data Generation Overview

```
[Step 1]   Generate 10,000 base claim features
            claim_id, policy_id, dates, vehicle_make, vehicle_model, vehicle_type,
            damage_type, damage_location, damage_severity, agent_channel, coverage_type,
            booleans, vehicle_age_years, mileage, driver_age, prior_claims_count

[Step 1b]  Join enrichment table on (vehicle_make, vehicle_model, manufacture_year)
            → vehicle_value        = typical_market_value_gbp × age_depreciation_factor
            → repair_estimate_gbp  = part_cost_index × f(damage_severity, damage_location)
            → repair_to_value_ratio = repair_estimate_gbp / vehicle_value

[Step 2]   Internal DGP — compute garage_outcome (not saved as column)
            logit = f(repair_to_value_ratio, damage_severity, damage_location, ...)
            garage_outcome = Bernoulli(sigmoid(logit))   ← used only in Steps 3 & 5

[Step 3]   Simulate pre-ML era decisions (2016–2021 rows, all pre-2022)
            Human rules + handler judgment → pre_ml_decision, pre_ml_label
            pre_ml_decision = 1  →  scrapped  →  pre_ml_label = 1              (self-fulfilling)
            pre_ml_decision = 0  →  garage    →  pre_ml_label = garage_outcome  (actual result)

[Step 4]   Train Model v1 on pre-ML labels
            Effective training window (after maturation buffer + OOT):
              Train + Test : 2016-01 → 2021-04  (80/20 random split)
              OOT holdout  : 2021-05 → 2021-10  (6 months, temporally latest)
              Excluded     : 2021-11 → 2021-12  (2-month maturation buffer)
            XGBClassifier.fit(X_train, y=pre_ml_label[train_rows])
            →  model_v1_score applied to all 10,000 rows (score all, train on subset)

[Step 5]   Apply v1 policy: absolute cutoff score ≥ 0.872  →  model_v1_decision
            model_v1_decision = 1  →  scrapped  →  model_v1_observed_outcome = 1              (self-fulfilling)
            model_v1_decision = 0  →  garage    →  model_v1_observed_outcome = garage_outcome  (actual result)

[Step 6]   Train Model v2 — two variants generated side by side
            train_and_apply_v2(df, X_all, option="A")  → model_v2a_*
            train_and_apply_v2(df, X_all, option="B")  → model_v2b_*

            Option A (option="A"): v1 log only — REAL Insurance A Cop. SCENARIO
              Train + Test : 2022-01 → 2024-04  (target = model_v1_observed_outcome)
              OOT holdout  : 2024-05 → 2024-10  (SFP-contaminated labels — see note below)
              Excluded     : 2024-11 → 2024-12  (maturation buffer)
              Note: pre_ml_label unavailable at retraining time — v2 trained on v1 data only

            Option B (option="B"): pre-ML mix + v1 log — RESEARCH COMPARISON ONLY
              Train + Test : 2020-01 → 2024-04  (pre_ml_label for 2020–2021 rows;
                                                  model_v1_observed_outcome for 2022–2024 rows)
              OOT holdout  : 2024-05 → 2024-10  (same OOT as Option A)
              Excluded     : 2024-11 → 2024-12  (maturation buffer)
              Note: does not represent any real Allianz model.
                    - Not v2: v2 used v1 log only (pre_ml_label was already disposed)
                    - Not v3: v3 tried 2023+ data only, dropped pre-COVID (different approach)
                    - Purpose: shows how SFP signal is diluted when an unbiased prior
                      (pre_ml_label) is available alongside contaminated v1 labels.
                      Analytical baseline for dissertation comparison only.

            ⚠ OOT note: Both options share the same OOT holdout (May–Oct 2024), which is drawn
              from the v1 production log. Scrapped cars in this period have forced label = 1.
              OOT AUC against model_v1_observed_outcome is therefore a biased metric.
              Use selective-labels-corrected AUC (Build 03) for valid OOT evaluation.

            →  model_v2a_score, model_v2b_score applied to all rows

[Step 7]   Apply v2 policy (same absolute cutoff ≥ 0.872) to BOTH variants
            →  model_v2a_decision, model_v2b_decision
            Save: claims_pre_v1.parquet      (2016–2021 rows)
                  claims_v1_log.parquet      (2022–2024 rows)
                  vehicle_enrichment.parquet (enrichment lookup table)
```

> **v3 status:** A v3 refresh was attempted at Insurance A Cop. (end of 2024 / early 2025) but was not deployed. The failure mode was not that precision could not reach 0.985 — a threshold can always be tightened to hold precision — but that doing so caused recall to collapse to an operationally unacceptable level. The model became too conservative to be useful: it fast-tracked only a tiny fraction of genuine total losses, undermining the core purpose of the system. This is consistent with SFP loop contamination: positive labels in training are inflated by v2's false positives, making the decision boundary imprecise, and forcing a high threshold to hold precision suppresses true positives alongside them. With no clean signal available (pre_ml_label disposed of; all training data now SFP-contaminated), v3 had nothing to learn from except the biases inherited from v1 and v2. Generating a synthetic v3 is an optional dissertation extension to show the loop deepening across three generations.

---

## Timeline Context

**What is known vs. what is defined for simulation:**

| What's known from real data | What we define for the synthetic simulation |
|---|---|
| Research data available from ~2018 (exact cutoff TBC) | Pre-ML era: 2016–2021 (6 years, our choice) |
| v1 and v2 exist; exact deployment dates unknown | v1 deployed 2022 (our choice); v1 era: 2022–2024 |
| v2 trained on v1 log only, all pre-2022 data | Simulated as: v2 trains on 2022–2024 v1 log (same mechanism, shifted dates) |
| v3 attempted 2025; tried 2023+ data; dropped pre-COVID | Not yet simulated (details uncertain) |
| pre_ml_label disposed (data retention) | Simulated as: pre_ml_label exists 2016–2021, then treated as inaccessible |

**Synthetic event timeline (our design choices):**

| Synthetic period | Event |
|---|---|
| 2016–2021 | Pre-ML era (synthetic): human rules + handler judgment → `pre_ml_label` generated |
| 2020–2021 | COVID period within the simulation — claims behaviour changed |
| 2016-01 → 2021-04 | **v1 training + test data** (80/20 split, after maturation buffer and OOT) |
| 2021-05 → 2021-10 | **v1 OOT holdout** — temporally latest labelled data before v1 deployment |
| 2021-11 → 2021-12 | **v1 maturation buffer** — excluded from training |
| 2022 | **Model v1 deployed** in simulation; begins generating `model_v1_observed_outcome` |
| 2022-01 → 2024-04 | **v2 training + test data** (v1 log only; same mechanism as real v2 trained on pre-2022 v1 log) |
| 2024-05 → 2024-10 | **v2 OOT holdout** — drawn from v1 log; SFP-contaminated |
| 2024-11 → 2024-12 | **v2 maturation buffer** — excluded from v2 training |
| 2025 | **v3 refresh** — not yet simulated; real attempt used 2023+ data, dropped pre-COVID |

> GDPR constraint: cannot use data older than 8 years at time of model build (internal policy).
> Enrichment data updated regularly independent of model retraining cycle.
> Synthetic `claim_date` range: **2016-01-01 → 2024-12-31**

> **Data retention (real Insurance A Cop. constraint):** The `pre_ml_label` dataset (human-era handler decisions, 2016–2021) has been disposed of due to data protection and regulatory requirements. It is no longer available for retrospective analysis. This has two consequences: (1) v2 was trained exclusively on v1-generated data — the pre-ML signal was unavailable at retraining time; (2) there is now no biased-oracle baseline for comparing model-era decisions against the pre-ML era. See oracle discussion below.

---

## Model Training Protocol

Confirmed methodology from internal discussions (documented in `README.md`).

| Parameter | Value | Notes |
|---|---|---|
| **Target maturation time** | ~2 months | Total loss outcome takes time to finalise (repair confirmed or claim settled). Most recent 2 months excluded from training. |
| **OOT holdout** | ~6 months | Temporally latest non-excluded data. Always *after* training data — not randomly sampled. Tests generalisation to future claims. |
| **Train / Test split** | 80 / 20 | Random split on the remaining data (after removing maturation buffer and OOT). Used for parameter learning and final performance reporting. |
| **Minimum training rows** | 10,000 | Per model version. With a 9-year span (2016–2024), v1 occupies ~59% and v2a ~26% of total rows. To satisfy this floor for all versions, `base_features.N_ROWS` must be ≥ ~40,000. Enforced at runtime in `model.py` (`MIN_TRAIN_ROWS`). |

```
Full data timeline (per model version)
──────────────────────────────────────────────────────────────────────►
│           Training + Test (80/20 random split)      │  OOT (6m) │excl│
│                                                     │           │(2m)│
```

### Applied splits — synthetic data dates

| Split | v1 (base: 2016–2021 pre-ML log) | v2a — real scenario (base: 2022–2024 v1 log) | v2b — research comparison only (base: 2020–2024) |
|---|---|---|---|
| **Maturation buffer (excluded)** | Nov–Dec 2021 | Nov–Dec 2024 | Nov–Dec 2024 |
| **OOT holdout** | May–Oct 2021 | May–Oct 2024 | May–Oct 2024 |
| **Train + Test (80/20)** | 2016-01 → 2021-04 | 2022-01 → 2024-04 | 2020-01 → 2024-04 |

> **v2a vs v2b clarification:**
> v2a represents the real scenario — v2 trained on v1 log only, with no pre-ML data available (disposed).
> v2b is a research-only comparison scenario — it mixes pre-ML labels with v1 log to show how SFP is partially diluted when an unbiased prior exists. This does not reflect anything that happened at Allianz: the "drop pre-COVID data" consideration came from the v3 retraining attempt (2025), which tried 2023+ data only — a different approach entirely. v2b exists solely as an analytical baseline for the dissertation.

> **SFP implication for OOT evaluation:**
> The v2 OOT holdout (May–Oct 2024) is drawn from the v1 production log period.
> Every claim in that period was subject to v1's scrapping decisions.
> Therefore `model_v1_observed_outcome` for OOT claims is also SFP-contaminated:
> scrapped cars in OOT have forced label = 1 (never verified in garage).
> **Standard OOT AUC against `model_v1_observed_outcome` is not an unbiased measure of v2's true detection ability.**
> Build 03 (Unbiased Evaluation) must apply selective-labels correction to the OOT set, not just the training set.

> **SFP implication for the maturation buffer:**
> The 2-month exclusion window is operationally necessary for label reliability.
> However, this also means the *most recently SFP-affected* claims (Nov–Dec 2024 for v2) are
> invisible during both training and OOT evaluation. The loop continues to deepen in this blind spot
> between the cut-off date and the next model deployment.

---

## 1. Identifiers

| Column | Type | Values / Range |
|---|---|---|
| `claim_id` | string | `CLM000001` format, zero-padded, unique per row |
| `policy_id` | string | `POL000001` format; ~70% unique (7,000 distinct policies for 10,000 rows); ~75% appear once, ~20% twice, ~5% three or more times |

---

## 2. Date Fields

| Column | Type | Values / Range | Notes |
|---|---|---|---|
| `claim_date` | date (YYYY-MM-DD) | 2016-01-01 → 2024-12-31 | Date claim was registered; seasonality affects volume (winter → more collisions, summer → more thefts); year also determines which model was active (pre-2022 = human era, 2022+ = v1 in production) |
| `incident_date` | date (YYYY-MM-DD) | 2016-01-01 → `claim_date` | Date of the actual accident; always ≤ `claim_date`; large gaps surface as `report_delay_days`; incident in a different calendar year than `claim_date` can indicate a delayed complex claim |
| `policy_start_date` | date (YYYY-MM-DD) | 2012-01-01 → `claim_date` | Start of the **current** annual policy period — not the customer's first-ever policy; new policies (< 6 months old) tend to have higher claim severity and less underwriting history; does not capture customer loyalty — see `customer_tenure_years` |
| `policy_expiry_date` | date (YYYY-MM-DD) | `policy_start_date` + 365 days | End of the current annual policy period; claims filed in the final 30 days before expiry are worth flagging as a timing anomaly for operational review |
| `report_delay_days` | int | 0 → 30 | Derived: `claim_date` − `incident_date`; both extremes are informative — long delays (> 14 days) suggest severe physical damage or time spent fabricating a narrative; very short delays on high-severity claims may indicate a premeditated incident |

---

## 3. Categorical Fields

| Column | Type | Categories (enum) | Notes |
|---|---|---|---|
| `vehicle_make` | string | `Ford`, `BMW`, `Toyota`, `Volkswagen`, `Vauxhall`, `Audi`, `Honda`, `Mercedes`, `Nissan`, `Hyundai` | UK top-10 brands; join key for enrichment table; premium brands (BMW, Audi, Mercedes) → higher `part_cost_index` → inflated `repair_estimate_gbp` at the same damage severity |
| `vehicle_model` | string | e.g. `Fiesta`, `3 Series`, `Corolla`, `Golf`, `Corsa` | Specific model within make; join key alongside `vehicle_make`; volume models (Fiesta, Corsa, Polo) have cheaper, widely available parts; niche or performance models have scarce/expensive parts |
| `vehicle_type` | string | `Sedan`, `SUV`, `Hatchback`, `Estate`, `Van`, `Motorbike` | Vehicle body type; SUVs and Vans have higher structural repair costs due to size; Motorbikes have distinct damage profiles and rarely follow the same `repair_to_value` total loss pathway |
| `damage_type` | string | `collision`, `flood`, `fire`, `vandalism`, `theft_damage` | How the damage occurred; `fire` is the strongest non-ratio predictor in the DGP (+0.4); `flood` and `fire` frequently result in total loss regardless of severity rating; `theft_damage` can indicate a staged or fabricated incident |
| `damage_location` | string | `front`, `rear`, `side`, `roof`, `multiple` | Where on the car; `multiple` is the highest total loss risk category in the DGP (+1.0); `roof` damage (rollover, hail) is typically expensive to repair; `front` and `rear` alone are rarely sufficient for total loss unless combined with high mileage or age |
| `damage_severity` | string | `minor`, `moderate`, `severe` | Handler's initial subjective assessment at claim intake, recorded **before** garage inspection; `severe` is a key DGP predictor (+1.5); susceptible to claimant framing and handler experience — one source of pre-ML bias |
| `agent_channel` | string | `online`, `broker`, `direct`, `app`, `phone` | Channel through which claim was filed; `online` and `app` claims have lower handler oversight at intake; `broker` involvement typically means a more thoroughly documented claim; channel also correlates with customer demographics |
| `coverage_type` | string | `third_party`, `third_party_fire_theft`, `comprehensive` | Policy coverage tier; `third_party` cannot claim for own vehicle damage — presence in a vehicle damage claim row would be a data anomaly to flag; `tpft` covers fire and theft only; `comprehensive` customers can claim for all damage types and do so more frequently |
| `repair_decision` | string | `sent_to_garage`, `scrapped_by_model`, `scrapped_by_handler` | What actually happened to the car; **key SFP mechanism column** — `scrapped_by_model` and `scrapped_by_handler` rows receive a forced label of 1 without garage verification, generating the self-fulfilling signal; era-dependent (2016–2021: handler decision, 2022+: model v1 decision) |

---

## 4. Boolean Fields

| Column | Type | Values | Notes |
|---|---|---|---|
| `is_weekend_claim` | bool | `True` / `False` | Derived from `claim_date`; weekend claims have lower handler availability at intake, which can affect the speed and thoroughness of initial damage assessment |
| `has_prior_claims` | bool | `True` / `False` | Binary flag for any prior claim history; repeat claimants tend to have higher claim severity on average; a simpler companion to `prior_claims_count` — use both together for full picture |
| `is_high_value_vehicle` | bool | `True` / `False` | `vehicle_value` > 75th percentile of the dataset; note this is a **relative** threshold — the same physical car may cross this boundary depending on dataset composition; high-value vehicles set a higher absolute repair cost bar for total loss |
| `car_driveable` | bool | `True` / `False` | Whether the car could be driven away after the accident; `False` is a strong proxy for severe structural damage; a driveable car subsequently classified as total loss is a potential over-claim signal worth flagging |

---

## 5. Numerical Fields

| Column | Type | Range | Distribution | Notes |
|---|---|---|---|---|
| `vehicle_age_years` | int | 0 → 20 | Right-skewed (most 1–8) | **Key DGP feature** (+0.8 at > 12 yrs): older cars → lower `vehicle_value` → easier to breach total loss threshold; depreciation modelled at 8%/yr (floor at 10% of new value) |
| `manufacture_year` | int | 2004 → 2024 | Derived | `YEAR(claim_date) − vehicle_age_years`; join key for enrichment; newer models (post-2018) have higher `part_cost_index` due to ADAS sensors and complex bumper assemblies |
| `vehicle_value` | float | 500 → 80,000 (GBP) | Derived | `typical_market_value_gbp × age_depreciation_factor` from enrichment join; **denominator of `repair_to_value_ratio`** — errors here propagate directly to the strongest DGP predictor |
| `mileage` | int | 0 → 200,000 | Right-skewed | **Key DGP feature** (+0.6 at > 120,000 miles): high mileage reduces residual value and repair worthiness; captures wear independent of `vehicle_age_years` (a young car can have high mileage) |
| `repair_estimate_gbp` | float | 200 → 25,000 (GBP) | Derived | `part_cost_index × f(damage_severity, damage_location)` from enrichment join; **numerator of `repair_to_value_ratio`**; luxury brand parts are disproportionately expensive relative to vehicle value — a Mercedes A-Class repair can total the car where a Fiesta repair would not |
| `repair_to_value_ratio` | float | 0.01 → 2.00 | Derived | `repair_estimate_gbp / vehicle_value`; **strongest DGP predictor** (+2.5 at > 0.8); industry standard total loss threshold is 0.7–0.8; values > 1.0 mean repair costs exceed the car's worth; clip at 2.0 in preprocessing |
| `driver_age` | int | 17 → 85 | Near-normal (μ=42, σ=15) | Age of main driver; young drivers (17–25) attract higher premiums and are statistically higher risk; drivers 75+ also have elevated risk profiles due to reaction time; age 35–55 is typically the lowest-risk band |
| `prior_claims_count` | int | 0 → 10 | Poisson (λ=0.8) | Number of previous claims on the same `policy_id`; counts ≥ 3 may indicate a high-risk policyholder profile worth additional scrutiny; more granular than `has_prior_claims` — use this for modelling, use `has_prior_claims` as a quick binary flag |
| `customer_tenure_years` | int | 0 → 20 | Right-skewed | Years the policyholder has held cover with this insurer **across all renewals**; `policy_start_date` resets to the current annual period at each renewal, so it does not capture long-term loyalty — this column does; long-tenure customers (> 5 years) tend to have more stable and predictable claim patterns; a short `policy_start_date` combined with high tenure indicates a recent renewal, not a new customer |

---

## 6. Pre-ML Era Columns

Human decisions from 2016–2021, before v1 was deployed.
`NaN` for 2022–2024 rows.

| Column | Type | Values | Notes |
|---|---|---|---|
| `pre_ml_decision` | int | 0 / 1 | 1 = handler scrapped the car (self-fulfilling); 0 = sent to garage |
| `pre_ml_label` | int | 0 / 1 | **v1 training target.** If scrapped: always 1. If garage: actual repair outcome. Already biased — scrapped cars never verified |

> **Target column nulls:** `pre_ml_label` has `NaN` only for 2022–2024 rows (by design — those rows belong to the model era, not the pre-ML era). Within the 2016–2021 rows, `pre_ml_label` is **always populated** — every claim in that period received either a handler scrap decision (label = 1) or a garage outcome (label = 0 or 1). There are no missing target values within the training window for v1.
>
> Similarly, `model_v1_observed_outcome` has no nulls within the 2022–2024 rows: every post-ML claim either received a model scrap decision (forced label = 1) or was sent to garage and received an observed outcome. The target column is always fully populated within each model's training window.

---

## 7. Model Columns (SFP Loop Core)

Naming convention: `model_v{n}_score`, `model_v{n}_decision`, `model_v{n}_observed_outcome`.

| Column | Type | Range / Values | Real-world status | Notes |
|---|---|---|---|---|
| `model_v1_score` | float | 0.00 → 1.00 | Superseded by v2 | P(total_loss) from v1; trained on `pre_ml_label`; exact real training dates unknown — 2016–2021 is our synthetic choice |
| `model_v1_decision` | int | 0 / 1 | Superseded | 1 if `model_v1_score` ≥ **0.872** → scrapped; 0 → sent to garage |
| `model_v1_observed_outcome` | int | 0 / 1 | Superseded | **v2's training target.** Scrapped cars forced to 1 (self-fulfilling); garage cars get actual repair result |
| `model_v2a_score` | float | 0.00 → 1.00 | **Currently deployed (real v2)** | P(total_loss) from v2; trained on `model_v1_observed_outcome` (v1 log only); real v2 trained on all pre-2022 v1 data — simulated here as 2022–2024 |
| `model_v2a_decision` | int | 0 / 1 | **Currently deployed** | 1 if `model_v2a_score` ≥ **0.872** → scrapped; 0 → garage |
| `model_v2b_score` | float | 0.00 → 1.00 | Research comparison only — not a real model | P(total_loss) trained on mixed labels (pre_ml_label 2020–2021 + model_v1_observed_outcome 2022–2024); shows SFP dilution when unbiased prior is available; does not represent v2 (pre_ml disposed) or v3 (tried 2023+ data, different approach) |
| `model_v2b_decision` | int | 0 / 1 | Research comparison only | 1 if `model_v2b_score` ≥ **0.872** → scrapped; 0 → garage |

> **v3 not simulated yet.** Real v3 (attempted 2025) trained on v2 log data using 2023+ only (pre-COVID period dropped). Exact training window uncertain. Synthetic v3 generation is an optional extension — would add `model_v3_score`, `model_v3_decision` columns to demonstrate SFP deepening across three generations.

> **Threshold logic (every model version identical):**
> Model outputs probability via `predict_proba(X)[:, 1]`
> → **absolute** decision rule: scrap iff `score ≥ 0.872` (the real cutoff, tuned for precision ≥ 0.985)
> → this is **not** a percentile — the scrap *rate* floats with the score distribution, so v2's score drift surfaces as a higher scrap rate (the headline SFP signal).

---

## Data Generating Process

### Internal DGP (not saved to dataset)

The DGP is used only to generate realistic `garage_outcome` values for claims that are
sent to a garage (i.e., not scrapped). It is **never saved as a column**.

```
logit = -2.0
      + 2.5 × (repair_to_value_ratio > 0.8)
      + 1.5 × (damage_severity == 'severe')
      + 1.0 × (damage_location == 'multiple')
      + 0.8 × (vehicle_age_years > 12)
      + 0.6 × (mileage > 120,000)
      + 0.4 × (damage_type == 'fire')
      + N(0, 0.3)

garage_outcome = Bernoulli(sigmoid(logit))   # ~10% positive; used in Steps 3 & 5 only
```

### Human Decision Process (pre-ML era, 2016–2021)

```
Rule A: repair_to_value_ratio > 0.9
         → pre_ml_decision = 1 → repair_decision = 'scrapped_by_handler'
         → pre_ml_label = 1    (self-fulfilling, garage never checked)

Rule B: damage_severity == 'severe' AND vehicle_age_years > 15
         → pre_ml_decision = 1 → repair_decision = 'scrapped_by_handler'
         → pre_ml_label = 1    (self-fulfilling, garage never checked)

Otherwise:
         → pre_ml_decision = 0 → repair_decision = 'sent_to_garage'
         → pre_ml_label = garage_outcome  (actual repair result from garage)

2022–2024 rows: pre_ml_decision = NaN, pre_ml_label = NaN
```

### SFP Progression Across Versions

```
pre_ml_label              ← human bias (rule-based, inconsistent)
                             [real data: disposed; synthetic: 2016–2021, dates our choice]
      ↓ v1 trains on pre_ml_label (exact real dates unknown; synthetic: 2016–2021)
model_v1_observed_outcome ← ML bias (systematic, self-fulfilling)
                             [real data: pre-2022 v1 log; synthetic: 2022–2024 v1 log]
      ↓ v2 trains on model_v1_observed_outcome ONLY (pre_ml_label disposed at retraining time)
model_v2a_score/decision  ← ML bias amplified (SFP deepening) [CURRENTLY DEPLOYED]

      ↓ v3 trains on v2 log (attempted 2025, 2023+ data only, pre-COVID dropped)
[v3 not deployed]         ← recall collapsed when precision held at ≥ 0.985
                             [not simulated in synthetic data yet]
```

---

## Data Generation Pipeline

### Step 1 — Generate base features (all rows)

```
Identifiers : claim_id (sequential), policy_id (~70% unique)
Dates       : claim_date (2016–2024) → incident_date → policy_start/expiry → report_delay_days
Categorical : vehicle_make, vehicle_model, vehicle_type, damage_type,
              damage_location, damage_severity, agent_channel, coverage_type
Boolean     : is_weekend_claim, has_prior_claims, is_high_value_vehicle,
              car_driveable
Numerical   : vehicle_age_years, mileage, driver_age, prior_claims_count,
              customer_tenure_years
```

### Step 1b — Join enrichment table

```
manufacture_year        = YEAR(claim_date) − vehicle_age_years
age_depreciation_factor = max(0.1, 1 − 0.08 × vehicle_age_years)

JOIN vehicle_enrichment ON (vehicle_make, vehicle_model, manufacture_year)

vehicle_value         = typical_market_value_gbp × age_depreciation_factor
repair_estimate_gbp   = part_cost_index × base_repair_cost(damage_severity, damage_location)
repair_to_value_ratio = repair_estimate_gbp / vehicle_value
```

> **Null handling after join:** If any row fails to match the enrichment table (e.g. rare make/model/year combination), the join produces null values for `typical_market_value_gbp` and `part_cost_index`. These are imputed rather than dropped — median imputation by `vehicle_make` group, falling back to dataset median if the group is also sparse. Derived fields (`vehicle_value`, `repair_estimate_gbp`, `repair_to_value_ratio`) are then recomputed on the imputed values. No rows are removed due to enrichment join failures.

#### `age_depreciation_factor` — straight-line 8%/yr with a 10% floor

```
age_depreciation_factor = max(0.10, 1.0 − 0.08 × vehicle_age_years)
```

| vehicle_age_years | calculation          | factor |
|---|---|---|
| 0  (new)          | 1.00 − 0.00 = 1.00  | 1.00   |
| 5                 | 1.00 − 0.40 = 0.60  | 0.60   |
| 10                | 1.00 − 0.80 = 0.20  | 0.20   |
| 12                | 1.00 − 0.96 = 0.04 → floor | **0.10** |
| 15+               | floor applies        | **0.10** |

The 10% floor prevents `vehicle_value` from hitting zero for very old cars,
keeping `repair_to_value_ratio` finite even for 20-year-old vehicles.

#### `base_repair_cost(damage_severity, damage_location)` — lookup table (GBP)

Base repair cost before any brand multiplier is applied.
Values represent a mid-market UK vehicle (part_cost_index = 1.0).

```python
BASE_REPAIR_COST = {
    # (damage_severity, damage_location) → base cost in GBP
    ("minor",    "front"):    500,
    ("minor",    "rear"):     450,
    ("minor",    "side"):     400,
    ("minor",    "roof"):     600,
    ("minor",    "multiple"): 800,

    ("moderate", "front"):   2_000,
    ("moderate", "rear"):    1_800,
    ("moderate", "side"):    1_500,
    ("moderate", "roof"):    2_500,
    ("moderate", "multiple"):3_500,

    ("severe",   "front"):   5_000,
    ("severe",   "rear"):    4_500,
    ("severe",   "side"):    4_000,
    ("severe",   "roof"):    6_000,
    ("severe",   "multiple"):8_000,
}
```

**Design rationale:**
- `multiple` is the most expensive location at every severity level — consistent with the DGP weight (+1.0) and real-world structural repair costs.
- `roof` exceeds `front`/`rear` at moderate/severe because rollover and hail damage require full panel replacement.
- Severity multipliers are roughly ×4–5 from minor → severe, which matches UK bodyshop data ranges.
- `part_cost_index` is applied on top: `repair_estimate_gbp = part_cost_index × BASE_REPAIR_COST[(severity, location)]`.
  A BMW (index ≈ 1.8) with severe multiple damage → £8,000 × 1.8 = **£14,400**.
  A Vauxhall Corsa (index ≈ 0.7) with minor rear damage → £450 × 0.7 = **£315**.

### Step 2 — Internal DGP (garage_outcome, not saved)

```
garage_outcome = Bernoulli(sigmoid(logit))   # used in Steps 3 & 5; NOT a column
```

### Step 3 — Simulate pre-ML era decisions (2016–2021 rows)

```
Rule A or B → pre_ml_decision = 1 → pre_ml_label = 1              (self-fulfilling)
Otherwise   → pre_ml_decision = 0 → pre_ml_label = garage_outcome  (actual result)
```

### Step 4 — Train Model v1

Real dates unknown — 2016–2021 is our synthetic choice for the pre-ML era.

```
model_v1       = XGBClassifier().fit(X_2016_2021, y=pre_ml_label)
                 # real training window: unknown; synthetic: 2016–2021
model_v1_score = model_v1.predict_proba(X_all)[:, 1]   # all 10,000 rows
```

### Step 5 — Apply v1 policy (SFP loop begins)

```
SCRAP_THRESHOLD = 0.872   # absolute P(total_loss) cutoff

model_v1_decision = 1  if model_v1_score ≥ SCRAP_THRESHOLD  → repair_decision = 'scrapped_by_model'
                  = 0  otherwise                             → repair_decision = 'sent_to_garage'

model_v1_observed_outcome:
  decision = 1  →  1              (self-fulfilling — garage never checked)
  decision = 0  →  garage_outcome  (actual repair result)
```

### Step 6 — Train Model v2 (parameterised window)

Real v2 was trained on all v1-generated data (pre-2022). In the synthetic simulation this
corresponds to the 2022–2024 v1 log (same mechanism, different dates because we placed
v1 deployment at 2022). Option B is not a real Allianz scenario — it is a research
comparison baseline for the dissertation.

```python
# Actual signature in src/data/synthetic/generate/model.py
def train_and_apply_v2(df, X_all, option: str = "A") -> pd.DataFrame:
    # option "A": REAL v2 scenario — v1 log only, full SFP contamination
    # option "B": RESEARCH COMPARISON — mixes pre-ML labels with v1 log
    #             (not v2: pre_ml disposed; not v3: v3 used 2023+ data only)
    ...
    # writes columns model_v2{a|b}_score and model_v2{a|b}_decision
```

```
--- Option A: v1 log only (REAL v2 scenario) ---  (suffix "a")
Training rows : 2022–2024  (target = model_v1_observed_outcome)
               [real v2: all pre-2022 v1 log; synthetic dates shifted]
model_v2a = XGBClassifier().fit(X_2022_2024, y=model_v1_observed_outcome)


--- Option B: pre-ML mix + v1 log (RESEARCH COMPARISON ONLY) ---  (suffix "b")
Training rows : 2020–2021  (target = pre_ml_label)
              + 2022–2024  (target = model_v1_observed_outcome)
[not a real Allianz model: pre_ml_label was disposed before v2 retraining;
 v3's approach (2023+ data, pre-COVID dropped) was entirely different]

combined_label = pre_ml_label              for 2020–2021 rows
                 model_v1_observed_outcome  for 2022–2024 rows
model_v2b = XGBClassifier().fit(X_2020_2024, y=combined_label)


--- Result ---
v2a is the real scenario: deployed (currently live), shows SFP symptoms —
score drift upward and higher scrap rate than v1.
v2b is a research baseline only: shows SFP dilution when an unbiased prior
exists alongside contaminated labels. Both variants are scored on all rows:

model_v2a_score = model_v2a.predict_proba(X_all)[:, 1]
model_v2b_score = model_v2b.predict_proba(X_all)[:, 1]
```

### Step 7 — Apply v2 policy and save

```
model_v2{a,b}_decision = 1  if model_v2{a,b}_score ≥ 0.872   (absolute cutoff)
                       = 0  otherwise

Output files (exactly as written by run.py):
  claims_pre_v1.parquet      2016–2021 rows — saved BEFORE v1 is trained, so it
                              carries NO model columns.
                              columns: base features + enrichment-derived fields
                                       + pre_ml_decision + pre_ml_label + repair_decision

  claims_v1_log.parquet      2022–2024 rows
                              columns: base features + enrichment-derived fields
                                       + pre_ml_decision (NaN) + pre_ml_label (NaN)
                                       + repair_decision
                                       + model_v1_score + model_v1_decision
                                       + model_v1_observed_outcome
                                       + model_v2a_score + model_v2a_decision
                                       + model_v2b_score + model_v2b_decision

  vehicle_enrichment.parquet enrichment lookup table
                              rows: ~200 (10 makes × ~4 models × ~5 year bands)
```

> **Note:** v1 is scored on *all* rows during generation, but only the 2022–2024
> slice (`claims_v1_log`) is persisted with model columns. The pre-v1 file is the
> human-era ground for v1's training target and deliberately predates any model output.

---

### SFP Signal to Verify After Generation

Observable checks — no oracle needed:

```
1. Score drift:
   mean(model_v2a_score) > mean(model_v1_score)
   → v2 assigns higher total loss probability on average

2. Decision rate inflation:
   rate(model_v2a_decision=1) > rate(model_v1_decision=1)
   → v2 scraps more cars than v1 did
   (observed: v1 ≈ 19% → v2a ≈ 21.5% under the absolute 0.872 cutoff)

3. Label mechanism bias:
   rate(model_v1_observed_outcome=1 | model_v1_decision=1) = 1.0   ← always (self-fulfilling)
   rate(model_v1_observed_outcome=1 | model_v1_decision=0) << 1.0  ← much lower
   → gap confirms label noise from scrapping mechanism

4. Pre-ML vs v1 comparison (synthetic data only — not possible on real Insurance A Cop. data):
   rate(pre_ml_label=1) vs rate(model_v1_observed_outcome=1)
   → if v1 observed rate > pre_ml rate: SFP amplified by ML
   ⚠ pre_ml_label has been disposed of at Insurance A Cop.; this check cannot be run on
     real data. It is available in synthetic experiments only.
```

### Oracle Data: What We Have and What We Don't

**In the synthetic dataset:** `garage_outcome` is the true oracle — it is generated by the DGP and used internally to produce realistic labels, but it is deliberately not saved to any output file. This simulates the real-world constraint.

**In the real Insurance A Cop. data:** There is genuinely no oracle. The original plan was to use `pre_ml_label` (human agent decisions, 2016–2021) as a biased-but-independent reference point — not a true oracle, since human handlers also scrapped cars without garage verification (self-fulfilling), but at least a different signal from the ML model. That data has since been disposed of due to data protection regulations. What remains is exclusively the v1 and v2 production logs.

This means:
- For scrapped cars (decision = 1), the true repair outcome is **permanently unknown** across all eras
- There is no pre-ML baseline to compare model-era behaviour against
- All SFP detection must operate solely on observed production log data

The table below shows how each SFP check shifts in meaning when a true oracle is available versus not.

**Why the absence of oracle values matters:**

| Check | Without oracle (this dataset) | With oracle (ground truth) |
|---|---|---|
| **Score drift** | Cannot determine whether higher mean score reflects genuine risk increase or model bias | Calibration error computable directly: `ECE = |mean(score) − rate(oracle=1)|`; drift attributable to SFP if ECE widens across versions |
| **Decision rate inflation** | Cannot separate legitimate rate increase from false-positive inflation | Decompose by oracle: `rate(decision=1 \| oracle=0)` rising → FP inflation confirmed; `rate(decision=1 \| oracle=1)` rising → appropriate sensitivity gain |
| **Label mechanism bias** | `rate(observed=1 \| decision=1) = 1.0` by construction (tautological); `rate(observed=1 \| decision=0)` unobservable | True precision and recall computable: `Precision = rate(oracle=1 \| decision=1)`, `Recall = rate(decision=1 \| oracle=1)`; counterfactual harm quantifiable as `rate(oracle=1 \| decision=0)` — claims that were scrapped but were actually repairable |
| **Pre-ML vs v1 comparison** | Both distributions are themselves biased observed labels | AUC against oracle computable per model version; if `AUC(v2, oracle) < AUC(v1, oracle)` despite v2 training on more data: SFP degradation confirmed |

**Core distinction:** without oracle, the four checks measure *symptoms* of SFP (anomalous patterns in observed data). With oracle, they measure *actual harm* — false positive rates, recall loss, and counterfactual outcomes for cases the model chose not to investigate.

**Label framing:** The problem here is **contaminated labels**, not missing labels. Every claim in the training window has a label — but labels for scrapped cars are structurally forced to 1 by the model's own past decisions, not by independent verification. This distinction matters for methodology: missing-label techniques (semi-supervised learning, imputation) are the wrong tool. The right frame is selective labels / label contamination, where observed outcomes are a biased subset of true outcomes depending on which action was taken.

**Implication for this research:** The absence of oracle labels is not a data quality gap — it is the defining structural feature of the SFP problem, and in the real Insurance A Cop. case it is compounded by the regulatory disposal of the pre-ML data. The model's own decisions determine which outcomes are ever observed, and the historical human-era decisions that might have served as a reference are gone. This leaves the SFP detection framework with no ground truth to compare against — it must identify the loop from symptoms alone: score drift, decision rate inflation, and label mechanism bias detectable in the production log itself. This constraint will be discussed in the dissertation as the core motivation for detection methods that do not require oracle access.

---

## Preprocessing Required

| Column | Preprocessing needed |
|---|---|
| `vehicle_make`, `vehicle_model`, `vehicle_type`, `damage_type`, `damage_location`, `damage_severity`, `agent_channel`, `coverage_type` | One-hot or ordinal encoding |
| `repair_decision` | One-hot encoding |
| `claim_date`, `incident_date`, `policy_start_date`, `policy_expiry_date` | Extract year, month, day-of-week; compute `report_delay_days`, `days_since_policy_start` |
| `vehicle_value`, `repair_estimate_gbp` | Log transform |
| `repair_to_value_ratio` | Already normalised; clip at 2.0 |
| `is_weekend_claim`, `has_prior_claims`, `is_high_value_vehicle`, `car_driveable` | Cast bool → int (0/1) |
| `pre_ml_label`, `pre_ml_decision` | NaN for 2022–2024 rows — exclude from v1 training |
| `model_v1_observed_outcome` | Flag decision=1 rows as self-fulfilling for label-noise-aware training |
| `customer_tenure_years` | Integer count; no encoding needed; use as-is in tree models |

---

## Enrichment Table (`vehicle_enrichment.parquet`)

Separate lookup table joined on (`vehicle_make`, `vehicle_model`, `manufacture_year`).
Updated regularly at Insurance A Cop. independent of model retraining cycle.

| Column | Type | Notes |
|---|---|---|
| `vehicle_make` | string | Join key |
| `vehicle_model` | string | Join key |
| `manufacture_year` | int | Join key |
| `typical_market_value_gbp` | float | Base market value; used to compute `vehicle_value` after age depreciation |
| `part_cost_index` | float | Composite repair cost index (1.0 = average); luxury > 1.0, economy < 1.0 |

**Derived fields after join:**
```
age_depreciation_factor = max(0.1, 1 − 0.08 × vehicle_age_years)
vehicle_value           = typical_market_value_gbp × age_depreciation_factor
repair_estimate_gbp     = part_cost_index × base_repair_cost(damage_severity, damage_location)
repair_to_value_ratio   = repair_estimate_gbp / vehicle_value
```

**Representative models per make:**

| Make | Models |
|---|---|
| Ford | Fiesta, Focus, Mondeo, Kuga, Transit |
| BMW | 1 Series, 3 Series, 5 Series, X3, X5 |
| Toyota | Yaris, Corolla, RAV4, Camry |
| Volkswagen | Polo, Golf, Passat, Tiguan |
| Vauxhall | Corsa, Astra, Mokka, Insignia |
| Audi | A3, A4, A6, Q3, Q5 |
| Honda | Jazz, Civic, HR-V, CR-V |
| Mercedes | A-Class, C-Class, E-Class, GLC |
| Nissan | Micra, Juke, Qashqai, Leaf |
| Hyundai | i10, i20, i30, Tucson |

---

## Column Count Summary (Claims Table)

| Group | Count |
|---|---|
| Identifiers | 2 |
| Date fields | 5 |
| Categorical | 9 |
| Boolean | 4 |
| Numerical | 9 |
| Pre-ML era | 2 |
| Model columns | 7 (v1 ×3 + v2a ×2 + v2b ×2) |
| **Total** | **38** |

> Counts are conceptual claims-table fields. The saved files differ by design:
> `claims_pre_v1` omits the 7 model columns (saved before training);
> `claims_v1_log` includes them. Both also carry the 6 enrichment-derived fields
> (`manufacture_year`, `typical_market_value_gbp`, `part_cost_index`,
> `vehicle_value`, `repair_estimate_gbp`, `repair_to_value_ratio`).

---

## Notes on Real Data Format (for when Insurance A Cop. data arrives)

- Format: **Parquet files + database tables** (not CSV)
- Code style: scripts over notebooks for reproducible pipelines; use **ruff** linter
- PII handling required for customer data columns once NDA signed
- Synthetic data is a placeholder — swap `claims_pre_v1.parquet` and `claims_v1_log.parquet` with real Insurance A Cop. data; framework runs unchanged
