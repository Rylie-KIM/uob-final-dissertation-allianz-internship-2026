# uob-final-dissertation-allianz-internship-2026


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

The model's purpose is to **bypass the garage entirely for clear-cut total losses** — cars where the damage is so severe that scrapping is the certain outcome. By fast-tracking these cases straight to write-off, Allianz reduces garage costs and speeds up settlement for the customer.

This is why **high precision (≥ 0.985) is the business-critical constraint**. A false positive — scrapping a car that could have been repaired — forces Allianz to pay out the full market value of the vehicle instead of the (lower) repair cost. That is a direct and significant financial loss. The model must therefore be extremely conservative: it should only fast-track a total loss when it is near-certain.

| Decision | Outcome | Cost implication |
|----------|---------|-----------------|
| `predict = total loss` → scrap (correct) | Genuine total loss bypasses garage | Saves garage assessment + hire car cost |
| `predict = total loss` → scrap (**wrong**) | Repairable car scrapped | Insurer pays full car value instead of repair cost — **high financial loss** |
| `predict = repairable` → send to garage | Garage confirms outcome | Garage + hire car costs incurred, but no catastrophic loss |

The model **does not need to catch every total loss** — missed total losses (false negatives) simply proceed to the garage as normal. The asymmetric cost structure is why recall is monitored but not the optimisation target.


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


# Training Process (Allianz-aligned)

The synthetic generator reproduces the **two-generation** training process exactly as it ran at Allianz, so the SFP loop is baked into the data rather than asserted. Code lives in `src/data/synthetic/generate/model.py`; this section is the canonical description of what that code does and why.

### The scrapping policy (shared by every model version)

```python
SCRAP_THRESHOLD = 0.872          # absolute P(total_loss) cutoff (real Allianz value)

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

### Model v2 — two retraining windows, both contaminated

v2 is the **refresh attempt that underperformed and got the project parked**. Two windows are generated so the dissertation can show that SFP — not the window choice — is the common cause of the degradation. Both train their target on `model_v1_observed_outcome`, which carries v1's forced positives.

```python
# Option A — v1 log only (pre-COVID data dropped)
#   rows 2022–2024, target = model_v1_observed_outcome
model_v2a.fit(X_2022_2024, y=model_v1_observed_outcome)

# Option B — keep 2020 onwards (COVID + v1 log)
#   rows 2020–2021 → target = pre_ml_label
#   rows 2022–2024 → target = model_v1_observed_outcome
model_v2b.fit(X_2020_2024, y=combined_label)
```

Both versions are scored on all rows and saved with their own columns
(`model_v2a_score`, `model_v2a_decision`, `model_v2b_score`, `model_v2b_decision`).

### What the loop looks like in the generated data

| Model | Training target | Scrap rate |
|---|---|---|
| v1 | `pre_ml_label` (human era) | ~19% |
| **v2a** | `model_v1_observed_outcome` (v1 log only) | **~21.5%** ↑ |
| v2b | mixed (pre-ML + v1 log) | ~19% |

v2a inflates the scrap rate even though true repair feasibility has not changed — the self-fulfilling labels from v1 push v2 to over-predict total loss. This is the observable signature the detection framework (Build 02) is built to catch. To regenerate the datasets after any change to the policy or windows, run:

```bash
python src/data/synthetic/run.py
```

> Full column-level schema, the data-generating process, and the SFP verification checks live in `src/data/synthetic/synth_data_structure.md`.