# uob-final-dissertation-Insurance A Cop.-internship-2026


Identifying and Mitigating Self-Fulfilling Prophecy Loops in Machine Learning 

- A self-fulfilling prophecy in machine learning occurs when a model's predictions influence the outcome in a way that reinforces the model's original prediction, creating a feedback loop.
- This can lead to biased or inaccurate results, especially in systems that interact with human behaviour or decision-making processes. 
- Identifying these loops can be tricky. 
- Can we create a framework to identify these loops and suggest for improvements in the future?
- method 1) We can start from unbiased performance evaluation and intervention analysis to assess model driven impact.
- method 2) Then potentially exploring randomisation strategies and causal inference technics for mitigation.
- A dataset containing insurance claims, 
- along with several versions of model predictions, is prepared and available for analysis.


# Business Context
## Summary of 2 expected benefits from the ML model 
1. reduce the process, cost, and time for total loss cars 
2. predicting total loss car accuarately is important, since if the car is classified as a total loss, the insurance company has to pay the whole car vlaue. 

## Detailed Context 
This model is an internal service known as the **Fast Track Total Loss** model. Without it, every damaged car would be sent to a garage, where an engineer assesses whether it can be repaired or must be written off. That process is costly: the insurer pays for the garage's assessment time and must provide a replacement car to the customer while theirs is being evaluated.

The model's purpose is to **bypass the garage entirely for clear-cut total losses** — cars where the damage is so severe that scrapping is the certain outcome. By fast-tracking these cases straight to write-off, Insurance A Cop. reduces garage costs and speeds up settlement for the customer.

This is why **high precision (≥ 0.985) is the business-critical constraint**. A false positive — scrapping a car that could have been repaired — forces Insurance A Cop. to pay out the full market value of the vehicle instead of the (lower) repair cost. That is a direct and significant financial loss. The model must therefore be extremely conservative: it should only fast-track a total loss when it is near-certain.

| Decision | Outcome | Cost implication |
|----------|---------|-----------------|
| `predict = total loss` → scrap (correct) | Genuine total loss bypasses garage | Saves garage assessment + hire car cost |
| `predict = total loss` → scrap (**wrong**) | Repairable car scrapped | Insurer pays full car value instead of repair cost — **high financial loss** |
| `predict = repairable` → send to garage | Garage confirms outcome | Garage + hire car costs incurred, but no catastrophic loss |

The model **does not need to catch every total loss** — missed total losses (false negatives) simply proceed to the garage as normal. The asymmetric cost structure is why recall is monitored but not the optimisation target.

### What "precision ≥ 0.985" is measured against

A critical constraint for this research: the 98.5% precision figure is **not** measured against engineer-confirmed ground truth labels. It is measured against the observed outcomes in the production log — which are themselves SFP-contaminated (scrapped cars are recorded as total loss without garage verification). There is no held-out subset of claims where every car was independently assessed by an engineer that could serve as a clean evaluation set.

This means the model's reported precision is a measure of internal consistency with prior decisions, not a measure of true accuracy. It is the "contaminated labels" problem — not missing labels, but labels that are structurally biased by the model's own past decisions. This is a core motivation for the SFP detection framework.

### Why there is no ground truth (oracle) label

Even if pre-model data existed (it has been disposed of under Insurance A Cop.'s data retention policy), it would not constitute a clean oracle:
- **Handler decisions** (call centre staff) reflect inconsistent human judgment and are themselves self-fulfilling — a handler who scraps a car without sending it to a garage generates a label with no independent verification.
- **Engineer decisions** (garage physical inspection) are the closest thing to ground truth, but it is not always possible to determine from historical records whether a decision was made by a handler or an engineer.

The model's goal is to replicate engineer-level certainty (98.5% precision) but it is evaluated against a log that conflates handler and engineer decisions, with scrapped cars never verified at all. This structural absence of oracle labels is not a data quality gap — it is the defining feature of the SFP problem in this domain.


# Model Training Methodology
- **Target maturation time**: ~1–2 months. The total loss outcome (whether a car is genuinely repairable or not) takes time to be confirmed. The most recent 2 months of data are excluded from training to ensure labels are fully matured and not provisional.
- **Out-of-time (OOT) holdout**: ~6 months of the most recent (non-excluded) data is held out as an out-of-time validation set. This tests whether the model generalises to future data and is not just overfitting to historical patterns.
- **Train / test split**: 80-20 random split on the remaining data (after removing the maturation buffer and OOT holdout).
- **Evaluation metric**: Primary training metric is **precision**, with a target threshold of **≥ 0.985**. Recall is computed alongside precision but is not the optimisation target — the model is tuned to minimise false positives (incorrectly scrapping repairable cars) rather than to maximise fraud/total-loss recall.
- **No calibration**: XGBoost outputs raw probability-like scores but they are not calibrated. Since the model is used purely for ranking/triaging claims (not for making expected-value decisions), well-calibrated probabilities are not required and the calibration step is omitted to keep the pipeline simple. Note: if scores are used as propensity weights (e.g. for IPS correction), poor calibration can distort debiasing — this is a known limitation flagged in the SFP mitigation analysis.
- **Decision threshold**: the scrapping policy applies an **absolute score cutoff** — a car is fast-tracked to scrap only when `model_score ≥ 0.872` (the threshold tuned on validation to hold precision ≥ 0.985). This is **not** a percentile/top-N rule. Because the cutoff is fixed in score space, the *scrap rate* is free to move with the score distribution — which is precisely how score drift in a later model version becomes observable as a higher scrap rate (the headline SFP signal). See `src/data/synthetic/generate/model.py` (`SCRAP_THRESHOLD`).

```
Full data timeline
──────────────────────────────────────────────────────────────────────►
│        Training + Test (80/20 split)        │   OOT (6m)  │ excl. │
│                                             │             │  (2m) │
```

- The OOT holdout is temporally separated — it always comes *after* training data, not randomly sampled from it. This reflects real deployment conditions where the model is applied to future unseen claims.

## Data Split Roles

| Split | When used | Purpose |
|---|---|---|
| **Train** | During training | Learn model parameters |
| **Validation** | During training | Hyperparameter tuning, early stopping |
| **Test** | After training | Report final performance metrics |
| **OOT** | After training | Verify the model holds up on future data |

```
Full dataset
├── Train       ┐
├── Validation  ┼── same time period, random split
├── Test        ┘
└── OOT             ← temporally later data
```


# Training Process (Insurance A Cop.-aligned)

## Data & Model Timeline

### 1. Production log — what was happening and when

| Period | Model running | Who made scrap decisions | Label generated | Researcher access |
|---|---|---|---|---|
| Before ~2018 | None (pre-ML era) | Handlers + engineers | `pre_ml_label` | ✗ Outside retention window |
| ~2018 – pre-v2 deployment | None → v1 (transition date unknown) | Handlers/engineers → Model v1 | `pre_ml_label` → `model_v1_observed_outcome` | ✗ `pre_ml_label` disposed; v1 log available from ~2018 (TBC) |
| ~2022 – present | **v2** *(currently live)* | Model v2 | `model_v2_observed_outcome` | ✓ v2 scores + decisions |

> v1's exact deployment and end dates are not confirmed. The ~2022 figure for v2 deployment is approximate. Log data start date (~2018) is also approximate — to be confirmed when real data is provided.

---

### 2. Model training — what each model learned from

| Model | Trained on | Training label | Training window | Deployment |
|---|---|---|---|---|
| **v1** | Pre-ML era production log | `pre_ml_label` (human handler/engineer decisions) | Unknown — we define this for the synthetic simulation | ✓ Deployed (dates TBC) |
| **v2** *(currently live)* | v1 production log only | `model_v1_observed_outcome` | All v1-generated data, pre-2022 only; `pre_ml_label` already disposed at retraining time | ✓ Currently active |
| **v3** *(not deployed)* | v2 production log | `model_v2_observed_outcome` | Attempted 2025; tried 2023+ data only (dropped pre-COVID period) — exact window uncertain | ✗ Recall collapsed when precision held at ≥ 0.985 |

> The "drop pre-COVID data" consideration came from the v3 retraining attempt — it was not part of v2's training design.

---

### 3. What this research can and cannot access

| | Pre-ML period (~2018 – v2 deployment) | v2 production period (~2022 – present) |
|---|---|---|
| **Claim features** | ✓ from ~2018 (TBC) | ✓ |
| **Model scores** | ✓ `model_v1_score` (v1 log, pre-2022) | ✓ `model_v2_score` |
| **Model decisions** | ✓ `model_v1_decision` | ✓ `model_v2_decision` |
| **Observed outcome label** | ✓ `model_v1_observed_outcome` | ✓ `model_v2_observed_outcome` |
| **Pre-ML targets** | ✗ `pre_ml_label` disposed (retention policy) | ✗ N/A |
| **Ground truth / oracle** | ✗ None — scrapped cars never garage-verified; handler vs engineer indistinguishable in records | ✗ None |

> **Synthetic data note:** Pre-ML era dates and v1 deployment dates are unknown in real data. For the synthetic simulation these are set by us. The simulation generates `pre_ml_label` from a rule-based DGP to reproduce v1's training — this column has no real-world counterpart available to the researcher.

---

The synthetic generator reproduces the **two-generation** training process exactly as it ran at Insurance A Cop., so the SFP loop is baked into the data rather than asserted. Code lives in `src/data/synthetic/generate/model.py`; this section is the canonical description of what that code does and why.

### The scrapping policy (shared by every model version)

```python
SCRAP_THRESHOLD = 0.872          # absolute P(total_loss) cutoff (real Insurance A Cop. value)

def apply_policy(scores):        # scores = model.predict_proba(X)[:, 1]
    return (scores >= SCRAP_THRESHOLD).astype(int)   # 1 → scrap, 0 → garage
```

- **Absolute cutoff, not a percentile.** Scrap only when the model is near-certain. The scrap *volume* therefore floats with the score distribution — this is what lets v2's upward score drift show up as a higher scrap rate.
- `decision = 1` → car scrapped → `observed_outcome` **forced to 1** (the car is gone; the garage never sees it → self-fulfilling label).
- `decision = 0` → car sent to garage → `observed_outcome` = the **true** repair result.

### Model v1 — trained on the pre-ML (human) era

```python
# Train rows: 2016–2021, label = pre_ml_label (handler decisions; already biased)
model_v1 = XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")
model_v1.fit(X_train_2016_2021, y=pre_ml_label)

df["model_v1_score"]    = model_v1.predict_proba(X_all)[:, 1]
df["model_v1_decision"] = apply_policy(df["model_v1_score"])     # SFP loop begins

# Forced-positive label mechanism
df["model_v1_observed_outcome"] = np.where(
    df["model_v1_decision"] == 1, 1, garage_outcome              # 1 if scrapped, else true outcome
)
```

> **Data retention note:** The `pre_ml_label` dataset (human-era handler decisions, 2016–2021) used to train v1 is no longer available at Insurance A Cop.. It has been disposed of due to data protection and regulatory requirements. This means v1's training data cannot be reconstructed or audited retrospectively — a constraint that informs why the SFP loop is difficult to untangle without the original pre-ML labels.

### Model v2 — currently deployed; trained exclusively on v1-generated data

v2 is the **currently active model** at Insurance A Cop.. It was retrained using only claims data generated under v1's scrapping policy — the pre-ML dataset was no longer available at the time of retraining. This makes v2 the first model version whose training labels are entirely contaminated by the SFP loop: every positive label it learned from was either a v1-scrapped car (forced positive) or a garage-confirmed outcome, with no unbiased pre-ML signal mixed in.

```python
# v1 log only (pre-ML data unavailable) — this is the real Insurance A Cop. scenario
#   rows 2022–2024, target = model_v1_observed_outcome
model_v2a.fit(X_2022_2024, y=model_v1_observed_outcome)
```

For dissertation analysis, a second synthetic variant (v2b) is also generated to serve as a counterfactual — showing what v2 would have looked like if the pre-ML data had still been available:

```python
# Counterfactual only — not what happened at Insurance A Cop.
#   rows 2020–2021 → target = pre_ml_label
#   rows 2022–2024 → target = model_v1_observed_outcome
model_v2b.fit(X_2020_2024, y=combined_label)
```

Both variants are scored on all rows and saved with their own columns
(`model_v2a_score`, `model_v2a_decision`, `model_v2b_score`, `model_v2b_decision`).
**v2a represents the real Insurance A Cop. v2; v2b is a synthetic counterfactual for comparison.**

### Model v3 — refresh attempted but not deployed

A v3 refresh was attempted after v2 but was not put into production; v2 remains the active model. The failure mode was not that precision fell below 0.985 — it is possible to tune a threshold that holds precision — but that doing so caused recall to collapse to a level the business considered unacceptably low. The model became too conservative: it only fast-tracked a tiny fraction of genuine total losses, undermining the core operational benefit of bypassing the garage for clear-cut cases.

This is the expected signature of a model trained entirely on SFP-contaminated labels: the positive class in training data is inflated with false positives (cars v2 scrapped that were actually repairable), so v3 learns an imprecise decision boundary. Holding precision requires tightening the threshold to the point where true positives are also suppressed — recall suffers as a direct consequence. With neither the pre-ML labels nor an unbiased holdout available, v3 had no clean signal to learn from.

**Additional reason v3 was shelved — Control Expert replacement (confirmed 2026-06-22):**
Allianz acquired a third-party company called **Control Expert**, whose platform includes a total loss prediction capability. The business decision was taken to retire the in-house FTTL model in favour of Control Expert — without conducting a proper comparative evaluation. Control Expert is not yet integrated; implementation is expected in **early 2027**. In the interim, v2 remains live and the team is looking at incremental improvements rather than a full retraining cycle.

One of the identified short-term improvements is **updating the enrichment table** to cover newer make/model/year combinations that currently produce join misses and degrade the model's score quality for newer vehicles. See `src/data/synthetic/generate/enrichment.py` for the synthetic equivalent.

### What the loop looks like in the generated data

| Model | Training target | Scrap rate | Status |
|---|---|---|---|
| v1 | `pre_ml_label` (human era) | ~19% | Deployed (superseded) |
| **v2a** *(real)* | `model_v1_observed_outcome` (v1 log only) | **~21.5%** ↑ | **Currently deployed** |
| v2b *(counterfactual)* | mixed (pre-ML + v1 log) | ~19% | Synthetic only — not real |
| v3 | `model_v2_observed_outcome` | — | Attempted; not deployed (recall collapsed when precision held at ≥ 0.985) |

v2a inflates the scrap rate even though true repair feasibility has not changed — the self-fulfilling labels from v1 push v2 to over-predict total loss. This is the observable signature the detection framework (Build 02) is built to catch. v2b is included in the synthetic data purely as a counterfactual baseline; it does not reflect what happened at Insurance A Cop.. To regenerate the datasets after any change to the policy or windows, run:

```bash
python src/data/synthetic/run.py
```

> Full column-level schema, the data-generating process, and the SFP verification checks live in `src/data/synthetic/synth_data_structure.md`.


# Claim Intake Channels — FNOL vs ENOL

## Definitions

| Term | Full name | Description |
|---|---|---|
| **FNOL** | First Notification of Loss | Traditional phone-based claim intake: a call-centre handler takes the claim, can ask follow-up questions, and records detailed damage information through a free-form conversation |
| **ENOL** | Electronic Notification of Loss | Online self-service claim intake, recently introduced at Allianz UK. The claimant follows a structured digital form; the set of fields collected is fixed by the form path rather than adaptive |

## Why the channel distinction matters for the FTTL model

The data that flows into the FTTL model differs structurally between the two channels:

- **FNOL (phone):** Handler-mediated — can probe ambiguities, collect richer damage descriptions, and exercise judgment on severity ratings. Data quality tends to be higher but is subject to handler variability and interpretation.
- **ENOL (online):** Fixed-path form — only the fields on the form are collected. Missing values and data errors may differ in pattern from FNOL. The claimant self-describes damage without handler prompting, which can affect `damage_severity` and `damage_location` accuracy.

**Score distribution hypothesis (from Luna, 2026-06-22):** More severe total-loss incidents are likely to trigger a phone call rather than an online form — the claimant is in a worse situation and seeks direct support. This means FNOL claims may have systematically higher model scores than ENOL claims, not because of model bias but because of genuine case-mix differences between the two populations.

## Current operational context

A ticket in the team's sprint (confirmed 2026-06-22) is specifically investigating ENOL vs FNOL differences in:
- Feature distributions (means, modes, missing value rates, error rates per column)
- Model output scores (is the score distribution materially different between channels?)

The analysis is primarily checking whether anything "massively concerning" surfaces — the expectation is that some difference will exist and is explainable by case-mix, not by a data pipeline issue.

## Relevance to SFP research

The ENOL/FNOL split is a **potential confound** for SFP detection. If the proportion of ENOL vs FNOL claims changes over time (e.g. as online filing becomes more common), this could cause a shift in the feature distribution that is unrelated to the SFP loop. When detecting score drift across model versions, it is worth checking whether shifts in channel mix can explain part of the observed drift before attributing it to SFP amplification.