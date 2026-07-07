# uob-final-dissertation-internship-2026


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

## ⭐ Pre-ML Baseline — the human era, before any model existed (reference figures)

> **These are the most important reference numbers for the whole SFP story.** They quantify the *pre-model, human-based* loop that every model version inherited.

Before any ML model was deployed, whether a car was scrapped (written off as total loss) was determined by a mix of **predefined rules** (e.g. *if a car flips over it is automatically a total loss*) and **handler judgement**.

| Pre-ML figure | Value | What it means for SFP |
|---|---|---|
| **% of all cars scrapped** | **15%** | The human-era baseline scrap rate — the reference point the real ML-era rate (≈ 19% → 21.5% post-deployment) should be compared against. Scrap inflation above 15% is the headline symptom. (**Real Allianz pre-model figures**; the synthetic DGP reproduces the loop with its own analogue rates ~18.4% → 18.6% — see the model-versions table further down.) |
| **% of scrapped cars fast-tracked for TL** (handler-identified, **did *not* go to the garage**) | **43%** | Nearly half of all write-offs were **forced labels with no garage verification** — a handler decided total loss and the car was never independently assessed. This is the **pre-ML human-based SFP loop quantified**: 43% of the positive labels in the pre-ML era are structurally unverifiable (`pre_ml_label` contamination). The remaining ~57% of scrapped cars did reach a garage and carry an engineer (ground-truth) outcome. |

**Why this matters.** v1 was trained on `pre_ml_label`, in which ~43% of the total-loss labels are handler-forced and never garage-checked (see §"Why there is no ground truth"). This figure is the concrete size of the contamination v1 inherited *before* the model-based loop ever began — the human loop the ML loop then amplified. (Source figures provided by the Allianz team; to be reconciled with the real logs when available.)

### Implied true total-loss rate — the class prior α

The 15% / 43% split pins down the **true total-loss rate (class prior α = P(y=1))** as a **sharp bound: α ∈ [8.55%, 15%], point estimate ≈ 15%.** In the pre-ML era, of *all* cars:

- **93.55% reached a garage and carry a ground-truth outcome** — 85% engineer-confirmed repairable (non-TL) + 8.55% engineer-confirmed total loss.
- **only 6.45% were fast-tracked** (handler-forced, no garage visit) — the oracle is permanently destroyed for these; their true status is unverifiable.

α is therefore partially identified by the 6.45% unverified slice alone: **8.55%** if every forced label was actually repairable, up to **15%** if every forced label was a genuine total loss. This is a **sharp** bound (not merely conservative) because the entire 85% non-scrapped region is garage-observed, so the missed-total-loss error in it is structurally **zero** — all uncertainty is confined to the 6.45% fast-track slice. The **point estimate sits near the upper bound (≈ 15%)**: fast-tracking was rule-driven on unambiguous write-offs (e.g. a flipped car), so those forced labels are almost all genuine total losses.

**This is the calibration anchor for the whole SFP story.** The pre-ML scrap *decision* rate (15%) almost exactly matches α — evidence the human era was well-calibrated *before* any model contamination entered the labels. The SFP fingerprint is the **post-deployment scrap rate inflating above α** (≈ 19% → 21.5%): once the model starts fast-tracking on its own forced-positive labels, it scraps more cars than the true total-loss rate can justify.

(α here is the *marginal* class prior P(y=1); it is distinct from the scrap precision π_scrap = P(y=1 | scrap), the positive rate *within* the scrapped subgroup. See `literatures/notes/p28.md`.)

## Summary of 2 expected benefits from the ML model 
1. reduce the process, cost, and time for total loss cars 
2. predicting total loss car accuarately is important, since if the car is classified as a total loss, the insurance company has to pay the whole car vlaue. 

## Detailed Context 
This model is an internal service known as the **Fast Track Total Loss** model. Without it, every damaged car would be sent to a garage, where an engineer assesses whether it can be repaired or must be written off. That process is costly: the insurer pays for the garage's assessment time and must provide a replacement car to the customer while theirs is being evaluated.

The model's purpose is to **bypass the garage entirely for clear-cut total losses** — cars where the damage is so severe that scrapping is the certain outcome. By fast-tracking these cases straight to write-off, Insurance Company. reduces garage costs and speeds up settlement for the customer.

This is why **high precision (≥ 0.985) is the business-critical constraint**. A false positive — scrapping a car that could have been repaired — forces Insurance Company. to pay out the full market value of the vehicle instead of the (lower) repair cost. That is a direct and significant financial loss. The model must therefore be extremely conservative: it should only fast-track a total loss when it is near-certain.

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

Even if pre-model data existed (it has been disposed of under Insurance Company.'s data retention policy), it would not constitute a clean oracle. The two human label sources are **not** on an equal footing (supervisor decision, 2026-07-06):

- **Engineer decisions** (garage physical inspection) **are the only ground truth in the entire system.** A car a garage engineer physically assessed has a verified repair-feasibility outcome. This engineer = ground-truth status applies **only to cars sent to a garage**; once a car is scrapped (by a handler or by a model) no engineer ever sees it and the oracle is permanently gone.
- **Handler decisions** (call centre staff) are treated **unconditionally as a biased data generator — never as ground truth.** A call handler does not have the engineering knowledge to judge repair feasibility, so a handler who writes a car off without sending it to a garage produces a forced label with no independent verification. This is the original, **human-based SFP loop** that predates the ML model: handler judgment → write-off → the outcome is recorded as total loss with no way to check it → that biased label becomes evidence for the next decision. Because of this, handler-originated labels are regarded as **contaminated by construction, regardless of how confident any individual handler was** — they are not a weaker oracle, they are not an oracle at all.

Two practical problems compound this: (a) in the pre-ML records it is not always possible to tell whether a given `pre_ml_label` came from an engineer (ground truth) or a handler (biased generator), so even the genuinely reliable engineer labels cannot be cleanly isolated and recovered; and (b) the pre-ML data is disposed of anyway.

The model's goal was never to reproduce handler judgment — **v1 was built to *outperform* the call handler** and approach engineer-level certainty (98.5% precision). But it is trained on, and evaluated against, a log that conflates biased handler decisions with reliable engineer decisions, with scrapped cars never verified at all. This structural absence of oracle labels is not a data quality gap — it is the defining feature of the SFP problem in this domain. Note there are therefore **two nested SFP loops**: the *human-based* loop (handlers writing off without verification, embedded in `pre_ml_label`) and the *model-based* loop (§ Training Process) that inherits and amplifies it.


# Model Training Methodology
- **Target maturation time**: ~1–2 months. The total loss outcome (whether a car is genuinely repairable or not) takes time to be confirmed. The most recent 2 months of data are excluded from training to ensure labels are fully matured and not provisional.
- **Out-of-time (OOT) holdout**: ~6 months of the most recent (non-excluded) data is held out as an out-of-time validation set. This tests whether the model generalises to future data and is not just overfitting to historical patterns.
- **Train / test split**: 80-20 random split on the remaining data (after removing the maturation buffer and OOT holdout).
- **Evaluation metric**: Primary training metric is **precision**, with a target threshold of **≥ 0.985**. Recall is computed alongside precision but is not the optimisation target — the model is tuned to minimise false positives (incorrectly scrapping repairable cars) rather than to maximise fraud/total-loss recall.
- **No calibration**: XGBoost outputs raw probability-like scores but they are not calibrated. Since the model is used purely for ranking/triaging claims (not for making expected-value decisions), well-calibrated probabilities are not required and the calibration step is omitted to keep the pipeline simple. Note: if scores are used as propensity weights (e.g. for IPS correction), poor calibration can distort debiasing — this is a known limitation flagged in the SFP mitigation analysis.
- **Decision threshold**: the scrapping policy applies an **absolute score cutoff** — a car is fast-tracked to scrap only when `model_score ≥ threshold`. This is **not** a percentile/top-N rule. Because the cutoff is fixed in score space, the *scrap rate* is free to move with the score distribution — which is precisely how score drift in a later model version becomes observable as a higher scrap rate (the headline SFP signal). See `src/data/synthetic/generate/model.py` (`SCRAP_THRESHOLD`).

  > **Threshold is per-model, not a universal constant.** What is invariant across model versions is the **business constraint (precision ≥ 0.985)** and the **policy *form*** (an absolute score cutoff, not a percentile). The actual cutoff *value* differs by version: each model's score distribution is different, so a **different** absolute threshold is required to hold the same ≥ 0.985 precision target. **`0.872` is specifically v2's tuned threshold** (the real Insurance Company. value used by the synthetic generator). **v1 and v3 were tuned to different threshold values** (exact figures not confirmed), each calibrated on validation to keep precision ≥ 0.985. So "the threshold" should always be read as "the precision-≥-0.985 cutoff *for that model version*", and `0.872` is the v2 instance of it.
  >
  > **Operational note — threshold change history (v2):** v2's threshold was briefly changed away from 0.872 at some point during production (exact value and dates not confirmed). Performance degraded and the threshold was promptly reverted to 0.872. **For the purposes of this research and the dataset, v2's threshold is treated as constant at 0.872 throughout the production period.** The brief deviation is not modelled separately and is not reflected in the decision columns in the production log. This simplification is consistent with Allianz's operational understanding of the dataset.
  >
  > **⚠️ Researcher concern (2026-06-25):** The advice to ignore this deviation is *operational*, not verified. If the brief threshold change overlapped a retraining data window, then the decisions and forced labels generated during that window were produced under a *different* cutoff — injecting a small, undocumented inconsistency into exactly the label data the SFP analysis depends on. "Treat the threshold as constant at 0.872" is therefore adopted as a **working assumption whose safety is not confirmed**, and is logged here as an open risk to revisit when the real decision logs become available.

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


# Training Process (Insurance Company.-aligned)

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

The synthetic generator reproduces the **two-generation** training process exactly as it ran at Insurance Company., so the SFP loop is baked into the data rather than asserted. Code lives in `src/data/synthetic/generate/model.py`; this section is the canonical description of what that code does and why.

### The scrapping policy (form shared by every model version; threshold value τ_v is tuned per-version)

```python
# Each version's decision threshold τ_v is TUNED at deployment, not hardcoded: it is the
# lowest (most permissive) score cutoff that holds precision ≥ TARGET_PRECISION (0.985) on a
# held-out validation slice, scored against that version's (SFP-contaminated) training label.
# See generate/model.py: _tune_threshold() + the fit/validation split in train_and_apply().

def apply_policy(scores, tau_v):     # scores = model.predict_proba(X)[:, 1]
    return (scores >= tau_v).astype(int)   # 1 → scrap, 0 → garage
```

- **The *form* (absolute cutoff tuned to precision ≥ 0.985) is shared; the threshold *value* τ_v is tuned per-version.** Each model's score distribution differs, so the cutoff that holds ≥ 0.985 precision differs by version. `SCRAP_THRESHOLD = 0.872` (in `src/config.py`) is the documented **real-world value for v2**, kept as the fallback τ_v when the precision target is unreachable on a validation slice; the synthetic generator's own tuned values come out near it (e.g. v1 ≈ 0.852, v2a ≈ 0.906 — distribution-dependent).
- **Absolute cutoff, not a percentile.** Scrap only when the model is near-certain. The scrap *volume* therefore floats with the score distribution — this is what lets v2's upward score drift show up as a higher scrap rate.
- `decision = 1` → car scrapped → `observed_outcome` **forced to 1** (the car is gone; the garage never sees it → self-fulfilling label).
- `decision = 0` → car sent to garage → `observed_outcome` = the **true** repair result.

### Model v1 — trained on the pre-ML (human) era

**Purpose of v1: to outperform the call handler**, not to imitate it. In the pre-ML era clear-cut write-offs were decided by call handlers, who lack the engineering knowledge to judge repair feasibility (see "Why there is no ground truth"). v1 was introduced to make those fast-track decisions more accurately and consistently than a handler could. But it was *trained* on `pre_ml_label` — labels a biased handler generator produced — so v1 inherits the human-based SFP loop it was meant to improve on: it learns from a biased teacher while aiming to beat it.

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

> **⚠️ Hard constraint — v1's training data is gone and cannot be accessed.** The `pre_ml_label` dataset (human-era handler/engineer decisions, 2016–2021) that v1 was trained on has been **permanently disposed of under Insurance Company.'s data-protection and regulatory (retention-policy) obligations**. It is **not archived, not recoverable, and the research has no access to it** — v1's training data can never be reconstructed, re-scored, or audited retrospectively, and the true pre-ML class prior (α, above) can only be *bounded*, never measured directly.
>
> **This is a first-class experiment-design constraint, not a footnote.** Every method in this project must be designed to work *without* v1's original training data or the pre-ML labels:
> - No approach may assume access to `pre_ml_label` or a clean pre-ML holdout — the counterfactual v2b (which mixes in `pre_ml_label`) is therefore a **synthetic-only** device and can never be reproduced on real Allianz data.
> - v1 can only be studied through the **artefacts it left behind** (its production log: `model_v1_score`, `model_v1_decision`, `model_v1_observed_outcome`), never through its inputs.
> - Any debiasing / unbiased-evaluation / causal-mitigation design that would require the original training labels to validate against is **infeasible on real data** and must instead rely on the v1 log, garage-verified rows, and bounding arguments.
>
> This irreversibility is a core reason the SFP loop is hard to untangle: the one dataset that could anchor the chain to an uncontaminated ground truth no longer exists.

### Model v2 — currently deployed; trained exclusively on v1-generated data

v2 is the **currently active model** at Insurance Company.. It was retrained using only claims data generated under v1's scrapping policy — the pre-ML dataset was no longer available at the time of retraining. This makes v2 the first model version whose training labels are entirely contaminated by the SFP loop: every positive label it learned from was either a v1-scrapped car (forced positive) or a garage-confirmed outcome, with no unbiased pre-ML signal mixed in.

```python
# v1 log only (pre-ML data unavailable) — this is the real Insurance Company. scenario
#   rows 2022–2024, target = model_v1_observed_outcome
model_v2a.fit(X_2022_2024, y=model_v1_observed_outcome)
```

For dissertation analysis, a second synthetic variant (v2b) is also generated to serve as a counterfactual — showing what v2 would have looked like if the pre-ML data had still been available:

```python
# Counterfactual only — not what happened at Insurance Company.
#   rows 2020–2021 → target = pre_ml_label
#   rows 2022–2024 → target = model_v1_observed_outcome
model_v2b.fit(X_2020_2024, y=combined_label)
```

Both variants are scored on all rows and saved with their own columns
(`model_v2a_score`, `model_v2a_decision`, `model_v2b_score`, `model_v2b_decision`).
**v2a represents the real Insurance Company. v2; v2b is a synthetic counterfactual for comparison.**

### Model v3 — refresh attempted but not deployed

A v3 refresh was attempted after v2 but was not put into production; v2 remains the active model. The failure mode was not that precision fell below 0.985 — it is possible to tune a threshold that holds precision — but that doing so caused recall to collapse to a level the business considered unacceptably low. The model became too conservative: it only fast-tracked a tiny fraction of genuine total losses, undermining the core operational benefit of bypassing the garage for clear-cut cases.

This is the expected signature of a model trained entirely on SFP-contaminated labels: the positive class in training data is inflated with false positives (cars v2 scrapped that were actually repairable), so v3 learns an imprecise decision boundary. Holding precision requires tightening the threshold to the point where true positives are also suppressed — recall suffers as a direct consequence. With neither the pre-ML labels nor an unbiased holdout available, v3 had no clean signal to learn from.

**Additional reason v3 was shelved — Control Expert replacement (confirmed 2026-06-22):**
Allianz acquired a third-party company called **Control Expert**, whose platform includes a total loss prediction capability. The business decision was taken to retire the in-house FTTL model in favour of Control Expert — without conducting a proper comparative evaluation. Control Expert is not yet integrated; implementation is expected in **early 2027**. In the interim, v2 remains live and the team is looking at incremental improvements rather than a full retraining cycle.

One of the identified short-term improvements is **updating the enrichment table** to cover newer make/model/year combinations that currently produce join misses and degrade the model's score quality for newer vehicles. See `src/data/synthetic/generate/enrichment.py` for the synthetic equivalent.

### Enrichment Table — Update Cycle and Open Questions

The enrichment table is updated approximately every **6, 9, or 12 months**, independently of any model retraining cycle. However, the exact mechanics of these updates are **not yet confirmed**:

- **Static-on-entry:** Once a per-ABI-code vehicle entry is added, its value fields (`typical_market_value_gbp`, `part_cost_index`) are frozen and never revisited — only new make/model/year rows are appended over time.
- **Yearly price refresh:** Existing entries are updated when new prices are set — e.g., a BMW 3 Series (2020) might have its `typical_market_value_gbp` revised in subsequent update cycles to reflect used-car market conditions at the time of the update.
- **Manufacture-year expansion:** New rows are appended as new manufacture years enter the fleet (e.g., a 2025 model year row added in a 2025/2026 update), with no changes to existing rows.

It is not yet known which of these mechanisms (or which combination) applies. This uncertainty matters for SFP analysis: if enrichment values are refreshed over time, then `repair_to_value_ratio` can shift for identical vehicles across training windows purely due to enrichment changes, independently of any SFP loop. Until this is confirmed, enrichment-driven score drift cannot be cleanly separated from model-driven drift.

### Preprocessing and Training-Window Divergence Across Versions (confirmed 2026-06-25)

A meeting on 2026-06-25 confirmed that v1, v2, and v3 differ from one another along **three independent axes**, not just their training labels:

1. **Training label source** (already documented): v1 ← `pre_ml_label`; v2 ← v1 log; v3 ← v2 log.
2. **Training window size**: each version was trained on a different span of history (e.g. v3 dropped the pre-COVID period). The windows are **not nested or aligned** across versions.
3. **Data preprocessing pipeline**: the feature preprocessing/cleaning steps were **re-implemented differently for each version** — they are *not* a single shared, versioned pipeline. The same raw claim can therefore become different model-ready features under v1 vs v2 vs v3.

**Why this matters for SFP detection.** The headline SFP signal is upward score drift across versions (`mean(v2_score) > mean(v1_score)`). But at least three confounds can produce that same signal *without* any SFP loop:

| Confound | Mechanism | Where flagged |
|---|---|---|
| Enrichment table refresh | `repair_to_value_ratio` shifts for identical vehicles between cycles | Enrichment section above |
| **Preprocessing divergence** | Same raw claim → different model-ready features under v1 vs v2 vs v3 pipelines | **NEW — this section** |
| Channel mix shift (FNOL/ENOL) | Case-mix changes the score distribution over time | FNOL/ENOL section below |

Score drift must therefore be **decomposed**, not read directly as SFP. Build 02 cannot attribute drift to the loop until the preprocessing, enrichment, and channel-mix contributions are ruled out or differenced away. This is a *measurement-confound* problem layered on top of the *label-contamination* problem, and it raises the evidentiary bar for any SFP claim against the real data. The cleanest comparison holds the preprocessing pipeline fixed and varies only the training label/window — achievable in the synthetic DGP, but **not** in the real production logs, where the three pipelines are baked in and cannot be retrospectively aligned.

#### What "preprocessing differs" concretely means — worked examples

"The preprocessing was re-implemented differently" is not abstract: the *same raw column* becomes a *different feature vector* under each version. Two representative mechanisms:

**Example A — `damage_severity`: one-hot vs ordinal (output schema differs).**
Raw column: `damage_severity ∈ {minor, moderate, severe}`.

| Version | Encoder | Output columns | A `severe` claim becomes |
|---|---|---|---|
| v1 | `OneHotEncoder()` | 3 cols: `sev_minor, sev_moderate, sev_severe` | `[0, 0, 1]` |
| v2 | `OrdinalEncoder()` | 1 col: `severity` (minor=0, moderate=1, severe=2) | `[2]` |

Same raw `severe` → occupies **3 feature slots in v1, 1 in v2**, and the model interprets it differently (v1 assumes *no* ordering between levels; v2 imposes a linear minor<moderate<severe).

**Example B — `vehicle_make`: rare-category bucketing (55 vs 41 columns).**
Say the data has **55 distinct makes**, 15 of them rare (each < 0.1% of claims: Ferrari, Lamborghini, …).

| Version | Logic | `vehicle_make` expands to | A **Ferrari** claim | A **BMW** claim |
|---|---|---|---|---|
| v1 | `pd.get_dummies` (one col per value) | **55 columns** | `make_Ferrari = 1` | `make_BMW = 1` (of 55) |
| v2 | group rare (<0.1%) → `OTHER`, then one-hot | **41 columns** (40 common + `OTHER`) | `make_OTHER = 1` (no `make_Ferrari` column exists) | `make_BMW = 1` (of 41) |

So one raw column expands to **55 slots under v1, 41 under v2**, and the *same* Ferrari claim is a unique feature in v1 but "misc bucket" in v2.

**Why this is an SFP confound.** When v1→v2 moves a claim's score, part of that movement is *"the model now encodes Ferrari as OTHER"* — an encoding artefact with **nothing to do with the SFP loop**. Reading "mean score went up" as SFP would mis-attribute it. This is exactly the measurement confound above.

**Three levels of divergence (a design knob for the synthetic data).** Encoding-logic difference and output-schema difference are related but not identical:

| Level | Output schema (columns) | Processing logic | Example |
|---|---|---|---|
| (a) | same | same | fully identical |
| (b) | **same** | **different** | same columns, different missing-value fill (median 4.0 vs 0) |
| (c) | different | different | the one-hot examples above (column count changes) |

On **real** data we do not choose — the repos did whatever they did (typically level c). On **synthetic** data this is a **deliberate knob**: level (a) gives Build 01 the clean isolation it needs to *prove* the SFP mechanism (only label/window varies), while level (c) *reproduces* the #10 confound so the detector can be stress-tested against encoding noise. The synthetic generator supports both, defaulting to (a).

### Model artefacts and reproduction — clone & run, not re-implement (updated 2026-07-01)

Confirmed with the team (2026-07-01): each production version is preserved as a **pickled scikit-learn `Pipeline`** — the preprocessing steps *and* the model are frozen together in one `.pkl`, not stored as separately readable code plus a bare model. This has three hard consequences for how this project must consume them:

1. **The preprocessing logic is not independently re-implementable — and must not be guessed.** A version's preprocessing lives *inside* its pickled pipeline, not in any file we can read. The correct way to obtain a version's features/scores is to **clone that version's model repository and run its own pipeline** in a matching environment — never to re-code "what it probably did". (Re-implementing by guess is exactly what would inject a spurious #10-style artefact.)

2. **Un-pickling requires that version's own code *and* its exact library stack on the import path.** `joblib.load("v1_pipeline.pkl")` reconstructs custom transformer classes (e.g. a bespoke rare-category grouper) by *importing* them — so the class definitions from v1's repo must be installed in the environment doing the load, at the pinned library versions. This is *why* each version needs its own isolated environment (`env-v1`/`env-v2`/`env-v3`); the pickle format makes the per-version env non-optional, not merely tidy. Version repos are pulled in as a **git dependency of each `env-vX` pinned to a commit SHA** (not as git submodules — submodules were tried before and abandoned for their detached-HEAD / recursive-clone friction).

   **Every model version has a genuinely different, incompatible environment — confirmed, not a "maybe".** And a frequent misunderstanding must be killed here: *"can't we just install all three repos as packages into one environment?"* **No — and the reason is not the repo code, it is the third-party numeric stack.** A pickle stores class *references* + fitted *state*, never source code; loading it re-imports both the repo's custom classes **and** the exact library classes the pipeline was built from (`xgboost.sklearn.XGBClassifier`, `sklearn.pipeline.Pipeline`, numpy dtypes), all of which must resolve at a compatible version or the load fails. A Python environment is a **flat** library pool — **one version of each library, shared by everything installed in it**; installing packages does *not* isolate their dependencies (unlike npm's nested `node_modules`). So installing v1+v2+v3 into one `.venv` forces `xgboost`/`scikit-learn`/`numpy` to a *single* resolved version; when v1 and v3 require different, incompatible versions (which for this project they do), that is physically unsatisfiable — pip errors, or one version's pickle silently fails to load at runtime. The repo code installs fine either way; it is the numeric stack that cannot coexist. **Physically separating the environments is the only thing that provides isolation** — hence `env-v1`/`env-v2`/`env-v3` are mandatory, not a tidiness choice.

3. **`predict` needs only the pickle; `retrain` needs the repo's training code.** A pickled pipeline can `predict_proba(raw)` but cannot re-train itself. The mitigation re-evaluation step (retrain on corrected labels) therefore requires each version's **training script from its repo**, run in that version's env. **A version available only as a prediction pickle, with no runnable training code, can be scored but not re-trained** — it drops to symptom tracking and is excluded from the quantitative before/after mitigation comparison (cf. `problem.md` §2.5 #7/#9). Confirming that runnable training code (not just the pickle) exists per version is a prerequisite for the mitigation experiment.

**Consequence for the code path (real vs synthetic).** Because real scoring is `pipeline.predict_proba(raw)`, the project holds **one execution contract for both data sources**: a version is always a *pipeline artefact* that is loaded and run.

- **Real:** `clone repo → build env-vX → joblib.load(vX_pipeline.pkl) → predict_proba(raw)`. Preprocessing happens *inside* the pipeline; `features_<v>.parquet` is no longer a scoring input but an optional artefact extracted for confound analysis (`pipe[:-1].transform(raw)`).
- **Synthetic:** to honour "always code for real; if real is unavailable, synthetic must reproduce the real conditions", the synthetic generator likewise emits a **per-version pipeline pickle** (shared preprocessing wrapped with the model), consumed by the *same* `predict.py`/`retrain.py`. The only thing that differs is how the pipeline was built — one shared synthetic preprocessor (identical by construction, level a) vs each repo's own (genuinely different, level c). This keeps the analysis code identical across real and synthetic; only the artefact provenance changes.

> **Status.** Points 1–3 are the confirmed reproduction model. The synthetic-side change (emit per-version pipeline pickles instead of `features_<tag>.parquet` directly) is an agreed direction affecting `src/data/synthetic/run.py` (`export_version_features`) and is **documented here ahead of the code refactor**; until that lands, the synthetic path still writes feature parquets. See `src/DESIGN.md` and `src/STRUCTURE.md`.

### What the loop looks like in the generated data

Each version tunes its own τ_v to hold precision ≥ 0.985 against the (contaminated) label. Measured on each version's OOT window (`evaluate.py` for the contaminated view, `verify_sfp_oracle.py` for the synthetic-only oracle view):

| Model | Training target | Scrap rate | Contaminated prec | **Oracle prec (true)** | cont−oracle gap | Status |
|---|---|---|---|---|---|---|
| v1 | `pre_ml_label` (human era) | ~12.5% | 0.976 | 0.974 | 0.002 | Deployed (superseded) |
| **v2a** *(real)* | `model_v1_observed_outcome` (v1 log only) | **~18.4%** ↑ | 0.988 | 0.980 | 0.008 ↑ | **Currently deployed** |
| v2b *(counterfactual)* | mixed (pre-ML + v1 log) | ~18.3% | 0.992 | 0.985 | 0.007 | Synthetic only — not real |
| v3a | `model_v2a_observed_outcome` (2023+) | **~18.6%** ↑ | 0.988 | 0.972 | **0.016** ↑ | Attempted; not deployed (real) |
| v3b *(counterfactual)* | `model_v2b_observed_outcome` (2023+) | ~18.8% | 0.983 | 0.972 | 0.011 | Synthetic only — not real |

v2a/v3a inflate the scrap rate (~12.5% → ~18.4% → ~18.6% across v1→v2a→v3a) — v1's self-fulfilling labels push later versions to over-predict total loss. **The key subtlety:** because each version re-tunes τ_v to hold precision against the *contaminated* label, the monitored (contaminated) precision stays pinned near/above 0.985 and **recall does not collapse in this DGP** (unlike the real v3). The loop instead surfaces as a **widening contaminated-vs-oracle precision gap** (0.002 → 0.008 → 0.016) — the harm the business's own metrics cannot see. **Read the gap, not the oracle-precision level:** the *level* of oracle precision is not directly comparable across versions here because the OOT windows differ in true-TL base rate (≈16.6% at v1's 2021 window vs ≈22.1% at the 2024 windows) — a year-over-year feature-drift confound, **separate from the SFP loop**, now under investigation in `notebook/00_feature_drift_EDA.ipynb`. Within a fixed window the gap isolates the loop. See `problem.md` §"How the loop manifests" and `verify_sfp_oracle.py`. v2b/v3b are counterfactual baselines only. To regenerate the datasets after any change to the policy or windows, run:

```bash
python src/data/synthetic/run.py
```

> Full column-level schema, the data-generating process, and the SFP verification checks live in `src/data/synthetic/synth_data_structure.md`.


# Deployment & Serving Infrastructure — AML → FastAPI

The team is currently **migrating the model's serving layer from Azure Machine Learning (AML) to FastAPI**. This is a *serving/inference* migration, not a change of where models are trained. The two stages have different jobs and are being split, not swapped:

| Stage | Where | What happens | Why it lives there |
|---|---|---|---|
| **Training** | **AML** *(stays — likely, not yet confirmed)* | Read large data, attach GPU/large compute, track experiments, register/version models, retain governance & reproducibility | Heavy, runs only on the retraining cadence; no strong reason to move the data-access/compute/governance stack |
| **Serving** | **FastAPI** *(migration in progress)* | Load a registered model, expose `/predict`, run input validation (Pydantic), preprocessing, auth, routing | Light, always-on; needs custom request logic and portability |

```
[AML]                                  [FastAPI]
data → train → register model    →     load model → /predict API
(MLflow / Model Registry)              (Pydantic validation, preprocess, inference)
experiment tracking, versioning,       real-time serving, auth, routing
governance
```

**The link between the two is the model registry.** Training finishes in AML and the model is registered (with a version pinned); the FastAPI layer pulls a specific version's artifact from that registry, holds it in memory, and serves inference only. Training is heavy and intermittent (per retraining cycle); serving is light and always up — hence the split.

**Why serving is being pulled out of AML into FastAPI:**
- **Flexibility** — input validation (Pydantic), preprocessing, auth, multi-model routing are all free-form in code; AML's built-in managed online endpoints are more constrained for this.
- **Portability** — FastAPI is just a container and runs anywhere, avoiding AML endpoint lock-in.
- **Integration** — fits naturally on top of an existing backend microservice ecosystem.
- **Cost / control** — traffic, scaling, and latency are controlled directly.

> **Status caveat (2026-06-25):** The serving migration (AML endpoint → FastAPI) is the confirmed/observed part. "Training stays in AML" is the **current expectation, not yet verified** — FastAPI is a serving framework, not a training tool, so any phrasing like "moving training to FastAPI" would be a loose use of terms and should be re-checked against what the team actually means. To confirm when more detail is available.

## Relevance to SFP research

This is operational/MLOps context rather than a direct SFP mechanism, but it matters for two reasons: (1) the **preprocessing-divergence confound** (see training section) interacts with serving — if FastAPI re-implements preprocessing separately from AML training, the train/serve skew becomes another axis on which "the same raw claim" yields different model-ready features; (2) the registry-based split is where decision logs and scores are emitted, i.e. where the production data the SFP analysis depends on is actually generated.


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


# Application Architecture — Split by Concern (Two Layers), Not by Model Version

## The question (raised 2026-07-01)

Because each model version (v1, v2, v3) was trained in a **different environment** with a **different data-preprocessing pipeline** (see "Preprocessing and Training-Window Divergence Across Versions", and the AML → FastAPI serving split), a suggestion was raised to split the project into **three separate applications — one per model version, each with its own `pyproject.toml`**.

## Resolution — isolate per-version *inputs*, share the *analysis*

Splitting *the whole thing* three ways is the wrong cut. The right cut is by **concern (layer)**, not by **version**:

- What genuinely differs per version → the **model artefact, its environment, its preprocessing, its (re)training, and its scoring**. These are isolated **one app/env per version**.
- What is **identical** across versions → the **SFP detection, mitigation, and re-evaluation algorithm**. This is a **single, version-agnostic application**.

This is the **two-layer design already documented** in `src/DESIGN.md`, `src/STRUCTURE.md`, and `src/ENV_MANAGEMENT.md`. The "each model runs on a separate application" framing is exactly the **Version Layer** — it does **not** extend to the detection/mitigation core.

| Layer | Scope | One per version? | Env | `pyproject.toml` |
|---|---|---|---|---|
| **Version Layer** (per-version model worker) | preprocessing → (re)train → score, for ONE version's artefact | **Yes** — `src/model/envs/v1│v2│v3`, isolated & frozen | own pinned env each | one **per version** |
| **Analysis Layer** | detector + mitigator + re-evaluation, version-agnostic | **No — single shared app** | one evolving `.venv` | **one, repo root** |

The two layers **never share a process**: the Version Layer emits `features_<v>.parquet` + `*_scores.parquet` to disk; Analysis Layer reads them and merges on `claim_id`. (Same decoupling as the offline-scoring design in `src/DESIGN.md`.)

## Why the detection/mitigation core must NOT be triplicated

1. **The SFP signal lives *between* versions.** The headline test is cross-version score drift / temporal prediction correlation (v1 → v2 → v3). Three independent apps with no shared join could not compute it — the signal is in the joins, not inside any single version.
2. **Validity of the comparison requires identical code.** Applying the *same* detector to every version is what makes "v2 drifted vs v1" a like-for-like claim. Three copies of the algorithm would be free to diverge — a scientific confound, not just a maintenance cost.
3. **"Same algorithm" is the correct instinct.** The detection/mitigation maths does not depend on which version produced a score; it consumes scores + labels that are already version-tagged columns.

So: **isolate the version-specific *inputs* (env + preprocessing + scores); keep the version-agnostic *analysis* single.**

## Where preprocessing lives (the part that was under-specified)

Per-version preprocessing divergence is real (confirmed 2026-06-25) and belongs in the **Version Layer**: each version's preprocessing code runs **inside that version's own env** and emits that version's `features_<v>.parquet`. It is **not** a shared module in the analysis app — that would re-introduce the train/serve-skew and preprocessing-divergence confounds the per-version contract exists to remove. (On synthetic data the files are identical by construction; on real data they genuinely differ — see "Per-version feature matrices" in `src/DESIGN.md`.)

## Re-evaluation after mitigation — does different preprocessing break it?

**Concern:** after mitigating we must retrain each model and check the SFP loop actually shrank. Do the differing preprocessing pipelines contaminate that comparison?

**Answer: no — provided each version's preprocessing is held FIXED across its own before/after.** The invariant:

> Mitigation changes the **training labels / sample weights / training data** — which is **downstream of preprocessing**. It does **not** change the preprocessing pipeline. So when version *v* is retrained on the corrected data, it reuses *v*'s **same** preprocessing as before, and the before → after Δ is attributable to the mitigation, not to a preprocessing change.

Two distinct axes — do not conflate them:

| Comparison | Preprocessing | Valid? |
|---|---|---|
| **Within a version**, pre- vs post-mitigation (the re-eval test) | **MUST be held fixed** (reuse *v*'s own pipeline both times) | ✓ Δ attributable to mitigation |
| **Across versions**, v1 vs v2 vs v3 | genuinely differs — already isolated into `features_<v>.parquet` | ✓ confound differenced away by the per-version contract |

Cross-version preprocessing differences therefore do **not** break re-evaluation: they are a *between-version* confound already controlled by the per-version features, whereas the re-eval test is a *within-version* before/after where preprocessing is held constant by construction.

**Where retraining runs.** Retraining is a **Version Layer** activity — it happens in each version's own env, on its own preprocessing, using the mitigation-corrected training set produced by the **Analysis Layer**. The before/after score comparison is then a **Analysis Layer** activity (read the new `*_scores.parquet`, re-run the detector). This is why the Version Layer owns *(re)train + score*, not merely *score*.


# Environment & Dependency Management (uv)

The Allianz team standard is **[uv](https://docs.astral.sh/uv/)** — a fast, Rust-based drop-in replacement for `pip` + `venv`. The team workflow is to **create a separate virtual environment per model version** and activate it both when *training* that version and when *scoring* with it. This matches the per-version isolation design already documented in `src/ENV_MANAGEMENT.md` and `src/DESIGN.md` (`env-v1` / `env-v2` / `env-v3` — **one isolated environment per version, all three managed separately**) — uv is simply the tooling the team uses to build and manage those environments.

> The original team setup note was written for **Windows**. The macOS equivalents are given alongside below — the only real differences are the install command and the activation path (`.venv\Scripts\activate` vs `source .venv/bin/activate`).

## Why uv?

- **Speed** — uv resolves and installs dependencies an order of magnitude faster than `pip`, so rebuilding a per-version environment from scratch is cheap.
- **Reproducibility** — uv writes a `uv.lock` file that pins the *exact* resolved version of every (transitive) dependency. `uv sync` recreates a byte-for-byte identical environment on any machine. This matters here because each model version must be scored in a frozen, reproducible environment — the same reason each version keeps its own pinned spec (`env-v1` / `env-v2` / `env-v3`).
- **Single source of truth** — dependencies live in `pyproject.toml` (a core list) and `uv.lock` (the resolved pins). You don't hand-edit these; uv keeps them in sync as you `add` / `remove` packages.
- **Per-model-version isolation** — when model versions need conflicting packages (e.g. a different XGBoost / scikit-learn release per version), **each version gets its own environment** (`env-v1`, `env-v2`, `env-v3` — all three managed separately). You activate the right one before training or scoring that version, so the environments never meet. See `src/ENV_MANAGEMENT.md`.

## Two environment tiers (managed differently, on purpose)

This project deliberately uses **two kinds of environment**, managed by **two different uv mechanisms**, because they have opposite lifecycles:

| Tier | What runs in it | uv mechanism | Spec files | Lifecycle |
|---|---|---|---|---|
| **Analysis env** (`.venv`) | The SFP pipeline, detector, mitigator, EDA, notebooks — everything that does **not** load a model | `uv add` / `uv sync` | `pyproject.toml` + `uv.lock` (repo root) | **Evolving** — packages are added as the research grows |
| **Per-version scoring envs** (`env-v1`, `env-v2`, `env-v3`) | Nothing but `predict.py`, used to re-score **one** model version's serialised artefact offline | Built per version, one env each | One **independent** pinned spec per version | **Frozen** — write-once; rebuilt only to reproduce, never casually mutated |

**Why the analysis env uses `uv add` + `pyproject.toml` + `uv.lock`.** It is a *single, growing* dependency set. `uv add` keeps `pyproject.toml` (what we want) and `uv.lock` (the exact resolved pins) in sync automatically — one source of truth, trivial to extend, and `uv sync` reproduces it anywhere.

**Why the per-version envs are managed separately, *not* in that `pyproject.toml`.** Each per-version env is a *frozen reproduction* of the libraries its model was serialised with. It must (a) match that version's pins exactly, (b) stay isolated from the other versions (v1's XGBoost must not move when v2's does), and (c) **not** be perturbed when you `uv add` something to the analysis env. Folding them into the analysis `pyproject.toml` would couple all of that together — the exact opposite of isolation. So each version keeps its own spec, outside the analysis project, and is scored in its own process (offline → parquet; the analysis runtime loads no model).

**Two ways to pin a per-version env** (choose per how strict you need to be):

| Option | How | Captures | When to use |
|---|---|---|---|
| **Standard — pinned `requirements.txt`** | `uv venv env-vX` + `uv pip install -r vX/requirements.txt` (pins written with `==`) | Direct deps only, no lockfile | Adequate when the env is frozen and every package is pinned exactly; simplest |
| **Stricter — per-version `pyproject.toml` + `uv.lock`** | each version is its own tiny uv project; `cd env-vX && uv sync` | **Transitive** deps + hashes → byte-for-byte reproducible | Dissertation-grade reproducibility, or when upstream transitive versions are unstable |

This project manages **v1, v2 and v3 each as a fully separate environment** regardless of which option is used — the two options only differ in *how tightly each one is pinned*. See `src/ENV_MANAGEMENT.md` for the concrete layout and commands.

## Initial Setup

### 1. Install uv

```bash
# Either OS (uv is on PyPI)
pip install uv
```

```bash
# macOS — recommended standalone installers (no existing Python needed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# or, with Homebrew:
brew install uv
```

### 2. Initialize a new project (only when starting a fresh project)

```bash
uv init <project_name>
```

This scaffolds a `pyproject.toml`. For an existing project that already has a `pyproject.toml` / `uv.lock`, skip this step and go straight to `uv sync`.

### 3. Create and activate the environment

`uv sync` reads `pyproject.toml` + `uv.lock` and builds the environment at `.venv`. Run it whenever the lockfile changes to bring your environment up to date.

```bash
# Both OS — create / update the environment
uv sync          # new environment is created at .venv
```

**Windows** (PowerShell / cmd):

```bat
.venv\Scripts\activate
```

**macOS / Linux** (bash / zsh):

```bash
source .venv/bin/activate
```

## Dependency Management

Dependencies are managed through `pyproject.toml`. **Do not edit it by hand** to add packages — let uv manage it so `pyproject.toml` and `uv.lock` stay consistent. With the uv environment active (or run from the project root — uv finds `.venv` automatically):

```bash
# add a dependency
uv add polars

# add a specific version constraint
uv add "scikit-learn>=1.5.1"

# remove a dependency
uv remove polars
```

Each command updates `pyproject.toml`, re-resolves `uv.lock`, and installs into `.venv` in one step. Commit both `pyproject.toml` and `uv.lock` so teammates get the identical environment from `uv sync`.

> Further reading: the [uv documentation](https://docs.astral.sh/uv/).