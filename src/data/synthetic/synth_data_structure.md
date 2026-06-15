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

[Step 4]   Train Model v1 on pre-ML labels (2016–2021, all pre-2022 including COVID)
            XGBClassifier.fit(X_pre_ml, y=pre_ml_label)
            →  model_v1_score applied to all 10,000 rows

[Step 5]   Apply v1 policy: 90th percentile threshold  →  model_v1_decision
            model_v1_decision = 1  →  scrapped  →  model_v1_observed_outcome = 1              (self-fulfilling)
            model_v1_decision = 0  →  garage    →  model_v1_observed_outcome = garage_outcome  (actual result)

[Step 6]   Train Model v2 — parameterised training window
            build_v2_training_data(window_start_year, include_pre_v1)
            Option A (window_start_year=2022, include_pre_v1=False): v1 log only (2022–2024)
            Option B (window_start_year=2020, include_pre_v1=True):  COVID + v1 log (2020–2024)
            →  model_v2_score applied to all rows

[Step 7]   Apply v2 policy: 90th percentile threshold  →  model_v2_decision
            Save: claims_pre_v1.parquet      (2016–2021 rows)
                  claims_v1_log.parquet      (2022–2024 rows)
                  vehicle_enrichment.parquet (enrichment lookup table)
```

> **v3 plan:** generate `model_v2_observed_outcome` using same structure as Step 5,
> then train v3. Shows SFP deepening over 3 generations. Optional extension.

---

## Timeline Context

| Period | Event |
|---|---|
| 2014–2021 | Pre-ML era: human rules + claim handler judgment + garage assessment |
| 2020–2021 | COVID period — claims volume and behaviour changed significantly |
| 2022 | **Model v1 trained** on all pre-2022 data (2014–2021, including COVID era); deployed to production |
| 2022–present | v1 in production; log data accumulates (model calls + observed outcomes) |
| 2025 | Refresh attempt (v2): dropped pre-COVID historical data (2014–2019); trained on 2022–2024 log only → performance drops → SFP suspected → parked |

> GDPR constraint: cannot use data older than 8 years at time of model build (Allianz internal policy).
> Allianz uses case-by-case window decisions — not always the maximum allowed.
> Enrichment data updated regularly independent of model retraining cycle.
> Synthetic `claim_date` range: **2016-01-01 → 2024-12-31**

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

---

## 7. Model Columns (SFP Loop Core)

Naming convention: `model_v{n}_score`, `model_v{n}_decision`, `model_v{n}_observed_outcome`

| Column | Type | Range / Values | Notes |
|---|---|---|---|
| `model_v1_score` | float | 0.00 → 1.00 | P(total_loss) from v1; `predict_proba(X)[:, 1]`; trained on `pre_ml_label` (2016–2021) |
| `model_v1_decision` | int | 0 / 1 | Binary: 1 if `model_v1_score` ≥ 90th percentile threshold → scrapped; 0 → sent to garage |
| `model_v1_observed_outcome` | int | 0 / 1 | **v2 training target.** If scrapped (decision=1): always 1 (self-fulfilling). If garage (decision=0): actual repair outcome |
| `model_v2_score` | float | 0.00 → 1.00 | P(total_loss) from v2; `predict_proba(X)[:, 1]`; trained on `model_v1_observed_outcome` — SFP-biased |
| `model_v2_decision` | int | 0 / 1 | Binary: 1 if `model_v2_score` ≥ 90th percentile threshold → scrapped; 0 → sent to garage |

> **Threshold logic (v1 and v2 identical):**
> Model outputs probability via `predict_proba(X)[:, 1]`
> → threshold = 90th percentile of that score
> → binary decision: scrap (1) or send to garage (0)

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
      ↓ v1 trains on pre_ml_label
model_v1_observed_outcome ← ML bias (systematic, self-fulfilling)
      ↓ v2 trains on model_v1_observed_outcome
model_v2_decision         ← ML bias amplified (SFP deepening)
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

```
model_v1       = XGBClassifier().fit(X_2016_2021, y=pre_ml_label)
model_v1_score = model_v1.predict_proba(X_all)[:, 1]   # all 10,000 rows
```

### Step 5 — Apply v1 policy (SFP loop begins)

```
threshold = 90th percentile of model_v1_score

model_v1_decision = 1  if model_v1_score ≥ threshold  → repair_decision = 'scrapped_by_model'
                  = 0  otherwise                        → repair_decision = 'sent_to_garage'

model_v1_observed_outcome:
  decision = 1  →  1              (self-fulfilling — garage never checked)
  decision = 0  →  garage_outcome  (actual repair result)
```

### Step 6 — Train Model v2 (parameterised window)

```python
def build_v2_training_data(
    pre_v1_df,            # claims_pre_v1.parquet  (2016–2021)
    v1_log_df,            # claims_v1_log.parquet  (2022–2024)
    window_start_year: int,
    include_pre_v1: bool,
) -> pd.DataFrame:
    if include_pre_v1:
        pre_v1_window = pre_v1_df[pre_v1_df["claim_year"] >= window_start_year]
        return pd.concat([pre_v1_window, v1_log_df])
    else:
        return v1_log_df
```

```
--- Option A: v1 log only ---
window_start_year = 2022,  include_pre_v1 = False
Training rows     : 2022–2024  (target = model_v1_observed_outcome)

model_v2a = XGBClassifier().fit(X_2022_2024, y=model_v1_observed_outcome)


--- Option B: drop pre-COVID, keep 2020 onwards ---
window_start_year = 2020,  include_pre_v1 = True
Training rows     : 2020–2021  (target = pre_ml_label)
                  + 2022–2024  (target = model_v1_observed_outcome)

combined_label = pre_ml_label              for 2020–2021 rows
                 model_v1_observed_outcome  for 2022–2024 rows

model_v2b = XGBClassifier().fit(X_2020_2024, y=combined_label)


--- Result ---
Both options produce worse performance than v1.
SFP is the common cause: self-fulfilling labels from v1 contaminate v2 training
regardless of window choice.

model_v2_score = model_v2.predict_proba(X_all)[:, 1]
```

### Step 7 — Apply v2 policy and save

```
model_v2_decision = 1  if model_v2_score ≥ 90th percentile threshold
                  = 0  otherwise

Output files:
  claims_pre_v1.parquet      2016–2021 rows
                              columns: all base features + pre_ml_decision + pre_ml_label
                                       + model_v1_score + model_v1_decision
                                       + model_v1_observed_outcome
                                       + model_v2_score + model_v2_decision

  claims_v1_log.parquet      2022–2024 rows
                              columns: all base features + model_v1_score + model_v1_decision
                                       + model_v1_observed_outcome
                                       + model_v2_score + model_v2_decision
                                       (pre_ml_decision = NaN, pre_ml_label = NaN)

  vehicle_enrichment.parquet enrichment lookup table
                              rows: ~200 (10 makes × ~4 models × ~5 year bands)
```

---

### SFP Signal to Verify After Generation

Observable checks — no oracle needed:

```
1. Score drift:
   mean(model_v2_score) > mean(model_v1_score)
   → v2 assigns higher total loss probability on average

2. Decision rate inflation:
   rate(model_v2_decision=1) > rate(model_v1_decision=1)
   → v2 scraps more cars than v1 did

3. Label mechanism bias:
   rate(model_v1_observed_outcome=1 | model_v1_decision=1) = 1.0   ← always (self-fulfilling)
   rate(model_v1_observed_outcome=1 | model_v1_decision=0) << 1.0  ← much lower
   → gap confirms label noise from scrapping mechanism

4. Pre-ML vs v1 comparison:
   rate(pre_ml_label=1) vs rate(model_v1_observed_outcome=1)
   → if v1 observed rate > pre_ml rate: SFP amplified by ML
```

### What Changes When an Oracle Label Is Available

In this dataset, no oracle column exists — `garage_outcome` is used only internally during
generation and is never saved. All four SFP checks above operate on **observed** labels only.

The table below shows how each check shifts in meaning when a true ground-truth label is available,
and why the absence of oracle values constitutes a structural limitation of real-world SFP analysis.

| Check | Without oracle (this dataset) | With oracle (ground truth) |
|---|---|---|
| **Score drift** | Cannot determine whether higher mean score reflects genuine risk increase or model bias | Calibration error computable directly: `ECE = |mean(score) − rate(oracle=1)|`; drift attributable to SFP if ECE widens across versions |
| **Decision rate inflation** | Cannot separate legitimate rate increase from false-positive inflation | Decompose by oracle: `rate(decision=1 \| oracle=0)` rising → FP inflation confirmed; `rate(decision=1 \| oracle=1)` rising → appropriate sensitivity gain |
| **Label mechanism bias** | `rate(observed=1 \| decision=1) = 1.0` by construction (tautological); `rate(observed=1 \| decision=0)` unobservable | True precision and recall computable: `Precision = rate(oracle=1 \| decision=1)`, `Recall = rate(decision=1 \| oracle=1)`; counterfactual harm quantifiable as `rate(oracle=1 \| decision=0)` — claims that were scrapped but were actually repairable |
| **Pre-ML vs v1 comparison** | Both distributions are themselves biased observed labels | AUC against oracle computable per model version; if `AUC(v2, oracle) < AUC(v1, oracle)` despite v2 training on more data: SFP degradation confirmed |

**Core distinction:** without oracle, the four checks measure *symptoms* of SFP (anomalous
patterns in observed data). With oracle, they measure *actual harm* — false positive rates,
recall loss, and counterfactual outcomes for cases the model chose not to investigate.

**Implication for this research:** The absence of oracle labels (i.e., unknown true repair
outcomes for scrapped cars) is not a data quality gap — it is the defining structural feature
of the SFP problem. The model's own decision determines which outcomes are observable, making
unbiased evaluation fundamentally impossible without external intervention (randomisation,
audits, or natural experiments). This limitation will be discussed in the dissertation as a
core constraint on any post-hoc SFP detection method that relies solely on production log data.

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
Updated regularly at Allianz independent of model retraining cycle.

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
| Model columns | 5 |
| **Total** | **36** |

---

## Notes on Real Data Format (for when Allianz data arrives)

- Format: **Parquet files + database tables** (not CSV)
- Code style: scripts over notebooks for reproducible pipelines; use **ruff** linter
- PII handling required for customer data columns once NDA signed
- Synthetic data is a placeholder — swap `claims_pre_v1.parquet` and `claims_v1_log.parquet` with real Allianz data; framework runs unchanged
