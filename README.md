# ML Self-Fulfilling Prophecy Detection & Mitigation

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

> **✅ Update 2026-08-05 — the inputs to this bound are now measurable.** v1's training data survives **including `pre_ml_label`** (see § "Data availability"), so the three figures this bound is built from — the 15% scrap rate, the 43% fast-track share, the implied 6.45% unverified slice — no longer have to be taken as team-provided numbers; they can be **computed directly from v1's training rows**. The bound's endpoints may therefore shift.
>
> **⚠️ But α remains bounded, not identified.** The surviving `pre_ml_label` records the *forced* 1 for fast-tracked cars, not their true status — the oracle for those 6.45% was destroyed by the scrapping itself, and no surviving dataset recovers it. Data availability is not oracle availability. 🔎 Recompute the three input figures on real rows and restate the interval.

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

Pre-model data **does exist** (✅ corrected 2026-08-05 — `pre_ml_label` survives in v1's training data, § "Data availability"; it was previously recorded here as disposed of under the retention policy). It nonetheless **does not constitute a clean oracle** — the argument below was always written as a conditional ("even if it existed…") and its substance is unaffected by the correction. The two human label sources are **not** on an equal footing (supervisor decision, 2026-07-06):

- **Engineer decisions** (garage physical inspection) **are the only ground truth in the entire system.** A car a garage engineer physically assessed has a verified repair-feasibility outcome. This engineer = ground-truth status applies **only to cars sent to a garage**; once a car is scrapped (by a handler or by a model) no engineer ever sees it and the oracle is permanently gone.
- **Handler decisions** (call centre staff) are treated **unconditionally as a biased data generator — never as ground truth.** A call handler does not have the engineering knowledge to judge repair feasibility, so a handler who writes a car off without sending it to a garage produces a forced label with no independent verification. This is the original, **human-based SFP loop** that predates the ML model: handler judgment → write-off → the outcome is recorded as total loss with no way to check it → that biased label becomes evidence for the next decision. Because of this, handler-originated labels are regarded as **contaminated by construction, regardless of how confident any individual handler was** — they are not a weaker oracle, they are not an oracle at all.

One practical problem compounds this: in the pre-ML records it is not always possible to tell whether a given `pre_ml_label` came from an engineer (ground truth) or a handler (biased generator), so even the genuinely reliable engineer labels cannot be cleanly isolated and recovered.

> **✅ Corrected 2026-08-05.** A second problem was previously listed here — "the pre-ML data is disposed of anyway" — and it is **false**: `pre_ml_label` survives inside v1's training data (§ "Data availability"). This makes the *provenance* problem above the **sole and now binding** obstacle: the labels are in hand, but engineer-origin and handler-origin rows still cannot be told apart, so the data's existence does not by itself yield a clean oracle. 🔎 Check whether the surviving extract carries any provenance/source field that would separate them — if it does, a partial engineer-only oracle becomes recoverable for the first time, which would be a materially new result.

The model's goal was never to reproduce handler judgment — **v1 was built to *outperform* the call handler** and approach engineer-level certainty (98.5% precision). But it is trained on, and evaluated against, a log that conflates biased handler decisions with reliable engineer decisions, with scrapped cars never verified at all. This structural absence of oracle labels is not a data quality gap — it is the defining feature of the SFP problem in this domain. Note there are therefore **two nested SFP loops**: the *human-based* loop (handlers writing off without verification, embedded in `pre_ml_label`) and the *model-based* loop (§ Training Process) that inherits and amplifies it.


# Model Training Methodology
- **Target maturation time**: ~1–2 months. The total loss outcome (whether a car is genuinely repairable or not) takes time to be confirmed. The most recent 2 months of data are excluded from training to ensure labels are fully matured and not provisional.
- **Out-of-time (OOT) holdout**: ~6 months of the most recent (non-excluded) data is held out as an out-of-time validation set. This tests whether the model generalises to future data and is not just overfitting to historical patterns.
- **Train / test split**: 80-20 random split on the remaining data (after removing the maturation buffer and OOT holdout).
- **Evaluation metric**: Primary training metric is **precision**, with a target threshold of **≥ 0.985**. Recall is computed alongside precision but is not the optimisation target — the model is tuned to minimise false positives (incorrectly scrapping repairable cars) rather than to maximise fraud/total-loss recall.
- **No calibration**: XGBoost outputs raw probability-like scores but they are not calibrated. Since the model is used purely for ranking/triaging claims (not for making expected-value decisions), well-calibrated probabilities are not required and the calibration step is omitted to keep the pipeline simple. Note: if scores are used as propensity weights (e.g. for IPS correction), poor calibration can distort debiasing — this is a known limitation flagged in the SFP mitigation analysis.
- **Decision threshold**: the scrapping policy applies an **absolute score cutoff** — a car is fast-tracked to scrap only when `model_score ≥ threshold`. This is **not** a percentile/top-N rule. Because the cutoff is fixed in score space, the *scrap rate* is free to move with the score distribution — which is precisely how score drift in a later model version becomes observable as a higher scrap rate (the headline SFP signal). See `src/data/synthetic/generate/model.py` (`SCRAP_THRESHOLD`).

  > **Threshold is per-model, not a universal constant.** What is invariant across model versions is the **business constraint (precision ≥ 0.985)** and the fact that the cutoff lives in **score space (an absolute cutoff, never a percentile)**. The actual cutoff *value* differs by version: each model's score distribution is different, so a **different** absolute threshold is required to hold the same ≥ 0.985 precision target. **`0.872` is specifically v2's tuned threshold** (the real Insurance Company. value used by the synthetic generator) — and only during **two** of v2's five threshold regimes (2021-06-03 → 2024-06-02 and 2026-02-25 → 2026-06-30); v2 has also run at **0.8915** and, currently, at **0.825**. **v1 and v3 were tuned to different threshold values**, each calibrated on validation to keep precision ≥ 0.985. So "the threshold" should always be read as "the precision-≥-0.985 cutoff *for that model version, in that period*".
  >
  > **⚠️ The policy *form* is NOT invariant** (confirmed 2026-07-29). v1 applies **two** cutoffs segmented by vehicle mobility (0.75 immobile / 0.85 mobile); v2 applies **one** global cutoff. Only the *absolute-cutoff-in-score-space* property is shared — the **arity** of the cutoff differs by version. See § "Model v1", § "Model v2" and § "Model v3". ✅ **v3 is also a single global cutoff** (confirmed 2026-07-29), so **v1 alone is segmented.**
  >
  > **Operational note — threshold change history (v2)** *(superseded 2026-07-29 — see below)*: the earlier account was that v2's threshold was briefly changed away from 0.872 at some point during production (exact value and dates not confirmed), that performance degraded, and that it was **promptly reverted** to 0.872 — so v2's threshold could be treated as constant at 0.872 throughout.
  >
  > **✅ Update (2026-08-18 — full change history supplied; supersedes the 2026-07-29 and 2026-07-31 readings):** v2's threshold **moved four times**, so the production log is **five policy regimes** deep, and the value **alternates**: `0.8915 → 0.872` (2021-06-03) `→ 0.825` (2024-06-02) `→ 0.872` (2026-02-25 16:26 UK) `→ 0.825` (2026-06-30 14:30 UK, currently in force). **v2's threshold is therefore very far from constant** — and because 0.872 and 0.825 each name *two* deployments years apart, **a threshold value does not identify a regime**. See § "Threshold history" for the full specification and consequences.
  >
  > **✅ Resolved (2026-08-18) — four changes, not one or two.** The question "how far back did the pre-2026-02-25 era of 0.825 run?" now has an answer: **it began 2024-06-02**, and before that 0.872 ran from **2021-06-03**. The record reaches back to 2021; only the *start* of the opening 0.8915 era is still unknown, and that era is left open on the left so it cannot mislabel anything. The old "briefly changed, promptly reverted" account is **wrong in both parts** — the 0.825 episode ran ~20 months, and it was not a revert.
  >
  > **❌ CONFIRMS the researcher concern (2026-06-25) — it is not closed, it happened.** The original worry was that a threshold change might have overlapped a *retraining data window*, injecting decisions made under a different cutoff into the training labels. With the full history in hand, **v3's window (`2023-06-01 → 2026-05-01`) contains two changes** — 2024-06-02 and 2026-02-25 — so v3's labels were generated under `0.872`, then `0.825`, then `0.872` again. The earlier "closed for v3 / v3 trains on a policy-homogeneous log" statement (2026-07-29) is **retracted**: it was true only of the June break, which was the only one then known. The concern is live for v3, for the current v2 log, and for any future retrain.

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


# Model v1 — Actual Data Split & Target Specification (real, confirmed 2026-07-28)

> The methodology section above describes the **generic/idealised** training scheme (maturation buffer + single 6-month OOT). The block below is the **actual v1 split as implemented at Insurance Company.**, transcribed from the real training code. Where the two differ, this section is authoritative *for v1*. The key real-world differences: v1 uses **explicit `lossdate` cut-off dates** (not a rolling "last N months" rule), the **Test set is an in-period random split — not an OOT holdout**, and there are **two** date-bounded validation sets rather than one OOT block.

## Source extract & cleaning

```sql
SELECT * FROM uc005.cc_fttl_train_v8 WHERE lossdate >= '2016-01-01'
```

- **Original dataset shape:** `(311591, 198)` — 311,591 claims × 198 raw columns.
- **No corrupted-row drop applies to v1.** The four splits sum to the **full 311,591** (178,435 + 44,609 + 36,049 + 52,498), i.e. no row is removed before splitting. *(The `ReportedDate = 2020-08-24` corrupted-record exclusion is a **v2** cleaning detail, not v1 — see the Cross-Version Dataset Summary.)*
- **Model-ready column count:** **39 columns** (after preprocessing/selection from the 198 raw columns). ⚠️ This **39 includes the `target` column and the claim-number (ID) column**, which are *not* predictive inputs — so the actual model **input feature count is 37** (39 − target − claim number). Read "39" as the width of the model-ready table, not the number of features the model learns from.

## Training-set exclusion — centre-flagged FTTL claims (`cc_fttl`)

**A key characteristic of v1's training data:** claims flagged by the claims centre as FTTL (`cc_fttl`) are **removed from the training set**. v1 is trained on the *remaining* claims only.

**`cc_fttl` is a rule-based flag — not discretionary handler judgment.** It is computed at First Notification of Loss (FNOL) from four "obvious total loss" physical indicators, OR'd together:

```python
# Source comments (verbatim from the real training code):
#   "adding claim centre fttl flag"
#   "find claims flagged by centre as fttl: we are removing those from the training set"
def claimCentreIndicators(df):
    cc_fttl = np.where(
        (df['FNOLRECOVERYAGENTTOTALLOSS_EXT'] == True) |   # recovery agent called it a total loss
        (df['FNOLEXTRICATION'] == True) |                  # occupant had to be cut/extricated from the vehicle
        (df['vehiclerollover'] == 1) |                     # vehicle rolled over / flipped
        (df['FNOLWATERDEPTHTYPE_EXT'].isin(['Into Cabin (above seat base)'])),  # flood water above seat base
        1, 0
    )
    return cc_fttl   # numpy ndarray, shape (len(df),), values 0/1 — NOT a pandas Series (no index)
```

The intent is stated directly in the code comments: the flag is *added* ("adding claim centre fttl flag") and then used purely as an **exclusion filter** ("find claims flagged by centre as fttl: we are removing those from the training set"). So `cc_fttl` never enters v1 as a *feature* — it exists only to drop those rows before training.

- **What it captures:** the **"predefined rules" component of the pre-ML era** (e.g. *"car flips over → automatic total loss"*) made concrete. Any one of {recovery-agent total-loss call, extrication, rollover, water into cabin} → `cc_fttl = 1`. These are unambiguous, physically-evident write-offs identified at intake — *not* a handler's subjective severity call.
- **Return value:** `np.where(cond, 1, 0)` returns a **numpy `ndarray`** of 0/1 (length = number of rows), **not** a Series and **not** a slice of `df`. It has no index and aligns to `df` positionally.
- **Why removed (rationale):** a `cc_fttl` flag is a **rule-forced label with no garage verification**. Training on it would teach the model to *reproduce the intake rule* rather than learn to discriminate — the human-based SFP loop v1 is meant to beat (cf. "v1 was built to *outperform* the call handler", § "Why there is no ground truth"). Excluding them keeps the model from simply re-learning the obvious-total-loss rules.
- **⚠️ Two-edged consequence (selective-labels bias).** These rule-flagged claims are exactly the **obvious total losses**. Removing them means v1 trains on a **truncated distribution** with the clear-cut write-offs excluded — yet in production it must still score exactly those cases. This is a **train/serve distribution mismatch** and a concrete instance of the selective-labels problem (P27): training distribution ≠ application distribution. So the exclusion *reduces rule-forced contamination in training* but *introduces a training-set selection bias* — it is not unambiguously "cleaner".

**Impact on dataset size — applied to ALL splits, not just training.** Although the code comment says "removing those from the *training* set", the row counts show the exclusion hits **every** split (Test and both validation sets shrink too) — i.e. `cc_fttl` is filtered from the whole modelling population, not just the training rows:

| Split | Before `cc_fttl` exclusion | After | Removed | % removed |
|---|---:|---:|---:|---:|
| Train | 178,435 | 173,758 | 4,677 | 2.62% |
| Test | 44,609 | 43,471 | 1,138 | 2.55% |
| Validation set 1 | 36,049 | 35,254 | 795 | 2.21% |
| Validation set 2 | 52,498 | 51,189 | 1,309 | 2.49% |
| **Total** | **311,591** | **303,672** | **7,919** | **2.54%** |

Two consequences: **(i)** because **Test and Validation are also filtered**, the precision ≥ 0.985 target is measured on a population with the obvious total losses **already removed** — the reported metric excludes the easiest positives, so it is *not* a precision over all claims the model faces in production. **(ii)** the removal rate is concrete: **~2.54% of all rows** (7,919 of 311,591).

> **Open items to reconcile (confirm against real data before treating as settled):**
> 1. **How `cc_fttl` relates to the ~43% forced-label contamination** documented in the Pre-ML Baseline section (a figure **provided by the Allianz team**, defined as the % of *scrapped* cars handler-fast-tracked with no garage visit ≈ 6.45% of *all* pre-ML cars; to be reconciled with real logs). **The new size evidence points to subset, not equality:** `cc_fttl` removes only **~2.54% of all rows**, well below the ~6.45% forced-label rate — consistent with the rule-based flag (rollover/extrication/water/recovery-agent) catching only the *physically obvious* fraction of the broader handler-forced population, leaving the softer subjective-judgment forced labels still in the data. (Caveat: 6.45% is an early-era estimate and 2.54% is the real v1-window count — this is **directional evidence for `cc_fttl ⊆ forced-labels`, not a like-for-like proof**.) So v1's training set is **partly** de-contaminated, not fully: the "trained on contaminated `pre_ml_label`" narrative still stands but must be qualified with "minus the rule-obvious total losses".
> 2. **Relationship to `veh_fast_track`** (the target-construction source): is `cc_fttl` the same signal, a subset, or independent? The target rule (`fast_track=1 & total_loss=0 → target=1`) applies to whatever rows *remain* after the `cc_fttl` exclusion — the interaction needs confirming.
> 3. **Scope:** does this exclusion apply to **v1 only**, or also to v2/v3 training?

## Date boundaries (all applied on `lossdate`)

| Boundary | Value |
|---|---|
| `EndTrainDate` | **2017-04-01** |
| `EndValidationDate` | **2017-06-30** |
| Dataset max `lossdate` | **2018-02-09** ✅ *(confirmed 2026-08-04 — full extract period is 2016-01-01 → 2018-02-09)* |

## The four splits

| Split | Definition (on `lossdate`) | Size | How produced | OOT? |
|---|---|---:|---|---|
| **Train** | `lossdate < EndTrainDate` (2017-04-01), then 80% of that pool | **178,435** | `train_test_split(..., test_size=0.2, random_state=0)` | No |
| **Test** | remaining 20% of the same `lossdate < EndTrainDate` pool | **44,609** | same `train_test_split` call (the held-out 20%) | **No — in-period random split** |
| **Validation set 1** | `EndTrainDate ≤ lossdate < EndValidationDate` (2017-04-01 … 2017-06-30) | **36,049** | date filter | Yes (near-term, ~3 months) |
| **Validation set 2** | `lossdate ≥ EndValidationDate` (2017-06-30 → 2018-02-09, dataset end) | **52,498** | date filter | Yes (longer horizon, ~7.5 months) |

> The **Size** column shows the raw `lossdate`-split counts **before** the `cc_fttl` exclusion above is applied. The `cc_fttl` filter then removes ~2.5% from **every** split — post-exclusion counts: **Train 173,758 · Test 43,471 · Val1 35,254 · Val2 51,189** (total 311,591 → 303,672). See the "Training-set exclusion — centre-flagged FTTL claims" section above.

**Test vs Val1 vs Val2 — the distinction that matters.**
- **Test (44,609)** is carved out of the *training period itself* by a random `train_test_split`. It shares the train time window, so it is **not** an out-of-time set — it measures in-distribution performance only.
- **Val1 / Val2** are **temporal holdouts** defined purely by `lossdate` cut-offs — data *later* than the training window. Val1 is the near-term future (~3 months after the train cut-off); Val2 is the longer horizon (2017-06-30 → 2018-02-09, the dataset end). These are what actually test generalisation to future claims (the role the generic "OOT" block plays).

> **⚠️ The Test split is NOT stratified.** The call is
> `train_test_split(train_test, train_test['target'], test_size=0.2, random_state=0)` — with **no `stratify=` argument**. Passing the label array as the second positional argument does **not** trigger stratification (a common misreading): scikit-learn only stratifies when `stratify=` is explicitly set. So v1's Train/Test split is a plain random split. Given the low positive rate (total-loss is the minority class), Train and Test can carry **slightly different class balances** — a real caveat under the tight precision ≥ 0.985 constraint, where the Test positive count directly drives the precision estimate's stability. `random_state=0` only fixes reproducibility (identical split on every re-run), not class balance. **This is v1-specific: v2 *does* pass `stratify=` explicitly** (see § "Model v2"), so the caveat does not carry across versions — and it is one more reason v1's and v2's in-period holdouts are not like-for-like.

## v1 target construction — `veh_fast_track` + `veh_total_loss` → `target`

v1's label is **not** a single raw column. It is derived from two raw fields and encodes the forced-label mechanism directly:

```python
df = data[[target_fttl, target_tl]].copy()          # target_fttl = veh_fast_track, target_tl = veh_total_loss
df[target_fttl] = df[target_fttl].map(...)           # veh_fast_track: non-FTTL → 0, FTTL → 1
df[target_tl]   = df[target_tl].map(...)             # veh_total_loss: non-TL → 0, TL → 1
df['target']    = df[target_tl]                      # base target = the total-loss flag
df.loc[(df[target_fttl] == 1) & (df[target_tl] == 0), 'target'] = 1   # fast-tracked but not TL → force to 1
```

Resulting truth table:

| `veh_fast_track` | `veh_total_loss` | `target` | Interpretation |
|:---:|:---:|:---:|---|
| 0 | 0 | **0** | not fast-tracked, not total loss → repairable |
| 0 | 1 | **1** | not fast-tracked, garage-confirmed total loss → genuine positive |
| 1 | 0 | **1** ⚠️ | **fast-tracked but no total-loss label → forced to 1** |
| 1 | 1 | **1** | fast-tracked and total loss → positive |

So `target` = "**total loss, OR fast-tracked**" by formula. **✅ Confirmed 2026-08-04 (from the real data): the recorded `veh_total_loss` is itself already set to 1 for fast-tracked vehicles** — the recording process treats fast-tracking as settling the total-loss question, with no garage visit ever taking place (an engineer-confirmed scrap also records 1). The forced positive therefore enters through the **data**, before any formula runs: the truth table's third row (`fast_track = 1 & total_loss = 0`) is a **defensive safeguard that rarely-to-never fires**, and the effective target is `veh_total_loss` alone. This is the **selective-labelling / forced-label SFP mechanism (§2.2 of `problem.md`) made concrete in the real v1 label** — fast-track routing itself becomes a recorded positive.

> ✅ **Resolved 2026-08-04 (supersedes the earlier open item on why `fast_track = 1 & total_loss = 0` occurs).** It does not occur as a stable data state: recorded `total_loss` is 1 whenever the vehicle is fast-tracked. v1's explicit disjunct guards against transient states (e.g. label maturation at snapshot), and is **not** the channel by which the forced label enters — that channel is the recording of `veh_total_loss` itself. Consequence: **the same forced-label semantics carry into every version's target** (v2: `veh_total_loss`; v3: `Fttl` = `veh_total_loss` renamed), so **P1 holds for all three versions through the data.**


## v1 model training call — real XGBoost configuration

Reconstructed from the real v1 training code:

```python
seed_val = 10**9   # = 1,000,000,000 — XGBoost random seed (distinct from the split's random_state=0)

eval_set  = [(X_test, y_test)]          # the held-out 20% Test split, passed as the eval set
xgb_model = xgb.XGBClassifier(
    eval_metric="mlogloss",             # multiclass log-loss
    eval_set=eval_set,
    silent=False,
    n_jobs=20,
    # seed_val wired in as the seed / random_state
)
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, ...)             # persist the fitted model artefact
return xgb_model
```

Key points + things to verify:

- **Two different seeds — don't conflate.** The Train/Test split uses `random_state=0`; the XGBoost model uses `seed_val = 10⁹`.
- **`eval_metric="mlogloss"` on a confirmed binary target.** `target` is **binary 0/1** (confirmed), and "**m**logloss" is the *multiclass* log-loss — so this is `mlogloss` applied to a 2-class problem (config choice, not evidence of >2 classes). XGBoost then runs a `multi:softprob`-style objective with `num_class=2`, so **`predict_proba` returns 2 columns** and `model_v1_score` is column `[:, 1]` (the total-loss probability). (Had `binary:logistic`/`logloss` been used, `predict_proba` would still be 2 columns — the score extraction is the same; the note matters only so no one assumes a k-column output.)
- **⚠️ `eval_set` is passed to the *constructor*, not to `.fit()`.** In the XGBoost sklearn API `eval_set` is normally a `.fit()` argument. Passed to the constructor it may be **stored but never actually used** — i.e. no real eval/early-stopping happens and the "Test as eval_set" monitoring silently no-ops. **Verify whether this eval_set takes effect.**
- **`silent=False` is a deprecated parameter** (superseded by `verbosity`), confirming v1 was built on an **older XGBoost release** — consistent with the per-version frozen-environment constraint (§ "Model artefacts and reproduction"): reloading this `joblib.dump` artefact requires v1's exact pinned XGBoost.
- **`joblib.dump(...)`** produces the serialised v1 artefact the analysis pipeline later reloads — the pickle that binds v1 to its exact library stack (the clone-&-run reproduction model).
- **`n_jobs=20`** — 20 parallel threads at fit time (infra detail, no modelling effect).

**v1 runtime environment (confirmed via `conda_dependencies_local.yml`): Python 3.5.2, pandas 0.22.0** (numpy pin not stated in the yml — 🔎 confirm). Two consequences:
- Pins v1 to a **very old stack** (2016–2017 era), consistent with the deprecated `silent=` XGBoost arg above and the per-version environment-isolation constraint (§ "Model artefacts and reproduction").
- **Even v1's *data* pickle is env-bound, not just its model pickle.** A DataFrame serialised by pandas 0.22.0 will very likely **not** unpickle under a modern pandas (2.x) — the internal BlockManager format changed across that gap — so v1's `inputs.pkl` (`…/Prod-Predictions/`) must be opened **inside v1's env**. `notebook/real/01_export_v1.ipynb` is written **Python-3.5-compatible on purpose** for exactly this, and exports the pickle to parquet for the analysis env (it recovered v1's unknown last date — see the Cross-Version Dataset Summary). The earlier one-off inspector `src/scoring/inspect_pickle.py` was deleted 2026-08-19, its job absorbed into that notebook.

## v1 scrapping threshold — segmented by vehicle mobility (creates a score-space overlap band)

**⚠️ v1's scrap decision is NOT a single absolute cutoff.** It is **segmented by vehicle mobility** — two different thresholds:

```
D = 1[ score > 0.75 ]   if the vehicle is IMMOBILE
D = 1[ score > 0.85 ]   if the vehicle is MOBILE
```

where **MOBILE** = vehicle mobility status ∈ {`Mobile`, `Mobile Not Roadworthy`, `Mobile Not Secure`}, and **IMMOBILE** = the complement. The immobile (undrivable → typically more damaged) car is scrapped more readily (lower bar, 0.75); the mobile (drivable → benefit of the doubt) car needs a higher score (0.85) to be scrapped.

> This means **`τ = 0.872` is not v1's rule** — 0.872 is v2's cutoff, and only in two of its five threshold regimes (see § "Threshold history"). v1's decision rule *form* is a **mobility-conditional (segmented) cutoff**, not one absolute threshold. The claim elsewhere that "the policy form (a single absolute cutoff) is invariant across versions" is therefore **false as stated**: v1 uses two cutoffs keyed on mobility, while **v2 is confirmed (2026-07-29) to use a single global cutoff with no segmentation**. The invariant is narrower — *an absolute cutoff in score space, never a percentile* — with the cutoff's **arity** differing by version. ✅ v3 is also a single global cutoff, so **v1 alone is segmented.**

### Does this create positivity in the v1 FTTL system?

**Short answer: partially — and importantly.** It does *not* restore strict positivity, but it opens a genuine **overlap band in score space** that the single-cutoff model said could not exist.

**Three score regions now, not two:**

| Score range | Immobile vehicle | Mobile vehicle | Oracle (garage outcome) observed? |
|---|---|---|---|
| `score < 0.75` | garage | garage | ✓ **both** groups garage-verified |
| `0.75 < score < 0.85` | **scrap** | **garage** | ⚠️ **mixed** — mobile cars ARE garage-verified up here |
| `score > 0.85` | scrap | scrap | ✗ neither — positivity dead |

**What this means formally:**

- **Conditional on the *full* feature vector (score + mobility): positivity still FAILS.** Given both the score and the mobility status, the decision is still a deterministic step function — the propensity `e(x) = P(D=1 | score, mobility)` is still 0 or 1. So the strict positivity violation (P4) is *not* dissolved.
- **Conditional on the *score alone*: positivity HOLDS in the band `(0.75, 0.85]`.** There, `e(score) = P(D=1 | score) = P(immobile | score) ∈ (0, 1)` — because at the *same* score a mobile car is garaged and an immobile car is scrapped. Mobility is the variable that breaks the perfect collinearity between score and treatment in that region.
- **Consequence for the data:** there ARE now **garage-verified outcomes above 0.75** (the mobile cars in the band). The blanket statement "**zero garage rows above `τ`**" is only true **above 0.85**; between 0.75 and 0.85 the oracle is partially observable via mobile vehicles.

**Why this is genuinely good news for the framework (with one caveat):**

- **Two-cutoff sharp RDD.** Each mobility group has its own sharp discontinuity (immobile at 0.75, mobile at 0.85), so the payout/outcome RDD can be run at **two** boundaries instead of one — more identification, not less.
- **Overlap band anchors counterfactual/reweighting models.** The mitigation's counterfactual-outcome model (what a scrapped car's garage result would have been) now has *real observed outcomes for high-scoring vehicles* (mobile, 0.75–0.85) to learn from, instead of extrapolating blindly above a single cutoff.
- **⚠️ Caveat — the overlap is confounded, not random.** Mobility is **not** independent of the true outcome: a `Mobile` car is drivable ⇒ physically less damaged ⇒ genuinely more likely repairable. So the mobile garage-verified cars in the band are a **systematically-more-repairable subgroup**. Using them directly to impute the counterfactual for immobile scrapped cars would **under-state** the immobile cars' true total-loss probability. The band gives *exploitable overlap*, but identification still needs a mobility adjustment — it is not clean random positivity.

**Bottom line:** the segmented threshold moves the system from "positivity globally dead above one cutoff" to "positivity dead only above 0.85, with a confounded-but-usable overlap band in 0.75–0.85." That is a materially better starting point for RDD and for the mitigation counterfactual — provided the mobility confound is handled explicitly.

> **Open items:** (1) confirm the exact `IMMOBILE` category set (complement of the three mobile statuses — list the actual values). (2) ~~Confirm whether v2/v3 also use a segmented mobility threshold or the single 0.872.~~ **✅ Resolved for v2 (2026-07-29): v2 is a single global cutoff with no mobility segmentation** — see § "Model v2". ✅ **Also resolved for v3 (2026-07-29): single global cutoff at 0.984.** So **v1 alone is segmented.** (3) This finding **qualifies the P4 "positivity dead" property** and the "zero garage rows above τ" assumption used in the mitigation/evaluation design — propagate to the dissertation (§2.4 P4, §3.6) and revisit the IPS-positivity caveat.


# Model v2 — Actual Split, Scoring & Decision Rule (real, confirmed 2026-07-29)

> **Scope of this section.** Read directly from v2's production **`train.py`** (split scheme, cleaning, data window) and **`score.py`** (decision rule). Still **🔎 TBD** for v2: target construction, the training call / XGBoost config, runtime environment, feature set, and whether v1's `cc_fttl` exclusion applies. Where this section and the generic methodology differ, this section is authoritative *for v2*.

## Split scheme & data window (from v2 `train.py`)

```python
dataset = dataset.sort_values('ReportedDate')
dataset = dataset[dataset['ReportedDate'] != '2020-08-24']      # corrupted — dropped
dataset = dataset[dataset['ReportedDate'] >  cutoff_date]       # cutoff_date = 2018-01-01

out_of_time_data = dataset.tail(math.floor(0.2 * len(dataset)))
out_of_time_data = out_of_time_data[out_of_time_data['ReportedDate'] > '2019-12-01']

train_test = dataset.head(math.floor(0.8 * len(dataset)))
# train_test_split(train_test, ...) → 0.8 = TRAIN, 0.2 = VALIDATION
# out_of_time_data                  → TEST
# all three stored as key/value entries in a dict
```

### ⚠️ The split names collide across versions — the biggest cross-version trap

With all three versions now confirmed, **v2 is the odd one out**:

| Actual role | v1 calls it | v2 calls it | v3 calls it |
|---|---|---|---|
| 80% of the in-period pool | **Train** | **Train** | **Train** |
| 20% random holdout, **same period** as train (in-distribution) | **Test** | **Validation** | **Test** |
| **Out-of-time** holdout, later than train | **Validation set 1 / 2** | **Test** | **OOT** |

**"Test" means an in-period random split in v1 and v3, but the out-of-time block in v2.** So comparing "v2 test precision" against v1's or v3's compares an **out-of-time** number against an **in-distribution** one: v2's is measured under a strictly harder condition and will look worse for reasons that have nothing to do with the model. **Any cross-version performance table must key on the *role*, not the name.** This document uses the role labels above throughout.

Two further points. **v2's naming is the misleading one** — v1 and v3 agree with each other, so a reader who checks only v1 and v3 will form the wrong expectation about v2. And **v3's convention is the clearest of the three**: it reserves the name "OOT" for the out-of-time block instead of overloading "Test" or "Validation".

### The splitting variable changed: `lossdate` (v1) → `ReportedDate` (v2)

v1 splits on **`lossdate`** (when the accident happened); v2 splits on **`ReportedDate`** (when the claim was notified). These differ by the **reporting lag**.

- **v2's choice is arguably the more honest one operationally:** only *reported* claims are knowable at training time, so ordering by `ReportedDate` reflects the information actually available at each point — a `lossdate` ordering can place a late-reported claim in the training window even though the insurer did not know about it yet.
- **But the two versions' boundaries are not comparable.** A v2 `ReportedDate` cut is not the same population slice as a v1 `lossdate` cut, so the split windows cannot be lined up directly across versions.
- **Reporting lag is not constant.** It plausibly varies by channel (FNOL phone vs ENOL online) and by severity, so sorting by `ReportedDate` **reshuffles** the loss-date ordering, and the OOT block can contain *older losses reported late*. 🔎 Quantify the lag distribution (and its drift) before treating v2's OOT block as cleanly "later" in loss terms.

### Data window — and why it matters for the SFP narrative

The cutoff filter is strict (`>`), so everything **on or before 2018-01-01** is removed: **v2's window is 2018-01-02 onward.**

> ✅ **Realised windows confirmed 2026-08-04:** the in-period pool (Train + Validation) runs **2018-01-02 → 2019-12-02**; the OOT block (v2's "Test") runs **2019-12-02 → 2020-09-30**. v2's data ends **2020-09-30** — the window does *not* extend to the 2021 training date, and the training pool closes **before COVID-19**.

> **⚠️ This window lies entirely *after* v1's.** v1 trained on `lossdate < 2017-04-01`. So essentially **all** of v2's training data was generated while **v1 was already in production**, meaning v2's labels are v1-influenced across the board. The SFP hand-off is therefore **not a partial contamination of v2's training set — it is close to total.** This is a concrete, code-level confirmation of the "v2a trained on `model_v1_observed_outcome`" narrative, and it is stronger than that narrative previously claimed. 🔎 Confirm v1's actual deployment date to state "all" rather than "essentially all".

### ✅ The corrupted-record exclusion — confirmed v2, and it drops a whole DAY

`dataset[dataset['ReportedDate'] != '2020-08-24']`.

- ✅ **Closes the open item** carried since the v1 section: this exclusion belongs to **v2**, not v1.
- ⚠️ **It is not "a corrupted record".** The filter tests `ReportedDate` equality, so it removes **every claim reported on 2020-08-24** — potentially a full day of claims, not one row. Both this document and `problem.md` previously described it as a single record. 🔎 **Count the rows actually dropped.** If it is a whole day, that is a non-trivial hole in the series — and with the realised windows now confirmed (2026-08-04), **2020-08-24 falls inside the OOT block** (2019-12-02 → 2020-09-30), i.e. the hole sits in v2's headline temporal holdout.

### ⚠️ A silent data-loss region between the train pool and the OOT block

The OOT set is taken as `tail(20%)` and **then** filtered to `ReportedDate > 2019-12-01`. Rows that are in the tail 20% **but** reported on or before 2019-12-01 are:

- **not** in `head(80%)` → excluded from Train and Validation, **and**
- filtered out of the tail → excluded from Test.

**They are dropped from the run entirely, and nothing records it.** Consequences:

- **The nominal 80/20 is not the realised split.** The OOT/Test block is **smaller than 20%** of the post-cutoff data by an unrecorded amount. Always report *realised* split sizes; never quote the nominal fractions.
- ✅ **Resolved in effect (2026-08-04).** The realised OOT block starts **2019-12-02**, immediately after the hard-coded filter date (2019-12-01) — so the 80th-percentile `ReportedDate` sits just above the filter and the filter was **non-binding (or marginal)**: the silent-loss region is empty or near-empty *for this run*. The reproducibility caveat below still applies to any re-run on refreshed data.
- **Reproducibility caveat:** a hard-coded date is being applied to a *proportional* tail. Re-running on an extended dataset moves the tail later in time, changing whether the filter binds — so **the split is not stable across re-runs**, and a re-run on refreshed data will not reproduce the original partition.

*(Minor, for completeness: `floor(0.8n) + floor(0.2n) ≤ n`, so head and tail never overlap — there is no leakage between the train/validation pool and the OOT block. They may leave a single row assigned to neither.)*

### ✅ v2's train/validation split IS stratified on the target — unlike v1

`train_test_split(..., stratify=data['veh_total_loss'])`. This **closes the open item** and resolves it in v2's favour: the concern flagged for v1 does **not** carry over.

- **v1 vs v2:** v1 passes the label array *positionally* with **no `stratify=`**, so v1's split is plain random and Train/Test class balances can drift apart. v2 passes `stratify=` explicitly, so Train and Validation carry **matched positive rates by construction**. Under precision ≥ 0.985 with a minority positive class, this materially stabilises v2's validation precision estimate.
- **⚠️ Stratification applies to the in-period pool only.** The OOT block (v2's "Test") is the temporal tail — its positive rate is whatever the later period actually contained, and nothing balances it against Train.

  > **This is quietly useful.** Because the in-period split *is* stratified, the Train/Validation positive rate is fixed by construction, so **any positive-rate gap between Validation and the OOT Test block is real temporal drift, not split noise.** That makes the comparison a clean, essentially free diagnostic — and under the SFP hypothesis the direction is predicted: as v1 scrapped more, forced positives accumulate, so the positive rate should be **rising** into the later window. 🔎 Compute the positive rate in Train / Validation / Test and compare — a cheap, high-value check that the stratification makes interpretable.

- **A modest sharpening, not a defect:** stratifying on the target means stratifying on a **contaminated** label, so the in-period holdout is built to mirror the training distribution *including its forced-positive share* — it therefore cannot serve as a check on the contamination. This is true of any random in-period split; stratification only makes the mirroring exact rather than approximate. Stratifying is the right call, and it is noted here only so the validation set is not mistaken for independent evidence about the label problem.
- 🔎 Still unrecorded: the split's `random_state`.

## v2 target — `veh_total_loss` alone (v1's `∨ fast_track` term is gone)

v2 trains on the raw **`veh_total_loss`** flag, with no derivation on top of it. Compared with v1:

| | v1 | v2 |
|---|---|---|
| Target | `veh_total_loss ∨ veh_fast_track` | **`veh_total_loss`** |
| Forced label in the **formula**? | ✅ yes — the `fast_track` disjunct (a safeguard) | ❌ no |
| Forced label in the **data**? | ✅ **yes** — recorded `total_loss` = 1 on fast-track | ✅ **yes — same recording semantics** |

**The formula difference is presentational, not substantive.** ✅ **Resolved 2026-08-04:** the recorded `veh_total_loss` is **set to 1 for fast-tracked vehicles** (no garage visit) under the same recording logic in both eras — v1's disjunct was a safeguard, not the contamination channel. So dropping the disjunct in v2 changes nothing about the label's content:

### ✅ Resolved (2026-08-04): `veh_total_loss` = 1 for a scrapped car — P1 holds for v2 through the data

The question this section previously carried — *what value does `veh_total_loss` hold for a vehicle fast-tracked to scrap?* — is resolved to the former reading **(a): it is recorded as 1**. Scrapping settles the total-loss question in the data. Consequences:

- **P1 holds for the currently-live model (v2)**, and for v3 (whose `Fttl` is the same column renamed — see § "Model v3"). It lives in the **data-recording process** rather than in any label formula; the SFP mechanism is unchanged, only its location.
- The alternative reading (b) — scrapped cars recorded as 0, i.e. trained-to-call-scrapped-negative — is **excluded**, as is the null-and-dropped selection reading.
- The cross-tab (`veh_total_loss` × fast-track flag on v2's surviving training data) is no longer a blocking open item; it is kept only as a cheap **regression test** of this confirmation when the data is at hand.

## v2 model training call — real XGBoost configuration (**xgboost 1.4.2**, model trained **2021**)

```python
xgb.XGBClassifier(
    objective="binary:logistic",   # clean binary — unlike v1's mlogloss-on-binary
    eval_metric="auc",             # ⚠️ not precision — see below
    colsample_bytree=0.6,
    eta=0.147686...,               # ⚠️ alias of learning_rate — BOTH are set, values differ
    gamma=15.0,                    # very high min split-loss (default 0)
    grow_policy="depthwise",       # the default
    learning_rate=0.0887667...,    # ⚠️ conflicts with eta above
    max_delta_step=10,
    max_depth=10,
    min_child_weight=1,            # default — permissive, tension with gamma=15
    n_estimators=450,
    random_state=42,
    reg_alpha=20.0,                # ⚠️ very strong L1 (default 0)
    reg_lambda=0.0123626...,       # ⚠️ L2 ~80× BELOW default (1.0) — effectively off
    scale_pos_weight=4.5,          # ⚠️ upweights the contaminated positive class
    subsample=1.0,                 # no row subsampling
    n_jobs=-1,
)
```

**Trained in 2021**, deployed ~2022, still live in 2026 — see "A five-year-old model" below.

### ⚠️ 1. `eta` and `learning_rate` are both set, and they disagree — the effective learning rate is ambiguous

`eta` is XGBoost's native name for `learning_rate`; they are **the same hyperparameter**. Here both are passed with **different values**: `0.147686` vs `0.0887667` — a **1.66×** difference. One silently wins.

- In the sklearn wrapper, `learning_rate` is an explicit constructor argument while `eta` arrives via `**kwargs`, and kwargs are merged *over* the explicit params — so **`eta = 0.147686` is the more likely effective value**. But this precedence is **XGBoost-version-dependent**, so it **must be verified, not assumed**. (v2 pins **xgboost 1.4.2**; later 2.x/3.x releases may warn or error on duplicate aliases instead — which is consistent with v3, on 3.2.0, setting `learning_rate` alone.)
- The difference is not cosmetic: with `n_estimators = 450`, the learning-rate × rounds budget is **66.5 vs 39.9** — materially different amounts of fitting.
- 🔎 **Definitive check:** load the saved v2 artefact and read the booster's own config —
  `model.get_booster().save_config()` returns JSON containing the `eta` actually used. Do this before quoting any v2 hyperparameter in the dissertation. **This is a reproducibility defect in the production code, and worth reporting to the team regardless of which value won.**

### ⚠️ 2. `eval_metric="auc"` is misaligned with the business constraint — and is computed on contaminated labels

Two separate problems, both material:

- **AUC does not measure what the deployment constraint measures.** The business rule is **precision ≥ 0.985 at a very high cutoff** — a property of the extreme top tail of the score distribution. AUC is rank-based and **threshold-free**, weighting the whole score range roughly equally. A model can post an excellent AUC while behaving poorly exactly where the threshold sits. Optimising/monitoring AUC therefore gives **almost no assurance about precision at τ**. *(v1 used `mlogloss`; neither version's training metric is the precision the business actually enforces.)*
- **The AUC is measured against SFP-contaminated labels.** v2 trains on v1's production log, where every v1-scrapped vehicle is recorded as a total loss (the forced label). So the metric rewards v2 for **ranking highly exactly those cases v1 chose to scrap**. Because v1 and v2 share much of the feature space, a high AUC partly certifies **agreement with the previous model's decisions** rather than agreement with reality. This is the contaminated-metric trap operating **inside the training objective**, not merely in the reported precision.

> The long non-round decimals on `eta`, `learning_rate` and `reg_lambda` are characteristic of an **automated hyperparameter search** (Bayesian optimisation / random search) rather than hand-tuning; the round values (`gamma=15.0`, `reg_alpha=20.0`, `scale_pos_weight=4.5`, `max_delta_step=10`) look like fixed or grid choices. If a search was run, it was **optimising AUC** — so the misalignment above propagates into the entire hyperparameter selection, not just the reported metric. 🔎 Confirm whether a search was run and what it optimised.

### ⚠️ 3. `reg_alpha = 20.0` vs v1's default `0` — this CONFOUNDS the SHAP concentration analysis

**This is the finding with the largest consequence for the dissertation's central contribution.**

The regularisation is strikingly asymmetric: **L1 `reg_alpha = 20.0`** (default 0 — very strong) against **L2 `reg_lambda = 0.0123626`** (default 1.0 — roughly **80× below** default, i.e. effectively switched off). Strong L1 drives **sparsity in leaf weights**, which mechanically **concentrates the model onto fewer features**. `gamma = 15.0` (default 0) pushes the same way by pruning all but the most productive splits.

Now compare with v1, whose training call sets **none of these** — v1 runs with `reg_alpha = 0`, `reg_lambda = 1.0`, `gamma = 0`.

> **⚠️ Therefore a v1 → v2 rise in feature-importance concentration is NOT, on its own, evidence of an SFP loop.** The thesis's central statistic — SHAP-importance concentration (Simpson / Hill / Shannon) rising across model generations — has a **competing mechanical explanation**: v2 is far more heavily L1-regularised than v1, and L1 concentrates importance by construction. The two hypotheses predict the same direction of change.
>
> **This must be addressed head-on, not left implicit.** Options, in rough order of strength: **(i)** run the concentration comparison on the **v2a → v3a** pair rather than v1 → v2a, *provided* v3 shares v2's hyperparameters (🔎 obtain v3's config — this is now a high-priority item); **(ii)** re-train a v1-configured model on v2's data (and vice versa) to separate the regularisation effect from the data effect; **(iii)** report concentration under matched hyperparameters; **(iv)** at minimum, state the confound explicitly as a limitation and bound its plausible size. Note that option (i) is already the specified design for a *different* reason (parallel-trends violation at the pre-ML→v1 era boundary) — so the two arguments reinforce each other.

### ⚠️ 4. `scale_pos_weight = 4.5` — upweights the contaminated class, and guarantees miscalibration

- **It amplifies the SFP loop directly.** The positive class *includes the forced positives*. Weighting positives 4.5× means the manufactured labels are given 4.5× the influence on the fitted model — a concrete amplification channel from v1's decisions into v2's parameters.
- **Implied class balance:** if set by the conventional `n_neg / n_pos` rule, it implies a positive rate near **18%**. 🔎 Confirm whether it was computed from the data or tuned by the search — the two readings support different claims about the label distribution.
- **⚠️ The score is NOT a probability, and `0.872` must never be read as "87.2% chance of total loss."** `scale_pos_weight` inflates predicted odds by roughly the weight factor, and the pipeline applies **no calibration step** (§ "Training Process"). Naively dividing the predicted odds by 4.5:

  | Score | Implied true probability *(illustrative only)* |
  |---|---|
  | 0.872 (pre-2026-07 threshold) | ≈ **0.60** |
  | 0.825 (current threshold) | ≈ **0.51** |

  *These are order-of-magnitude illustrations, not calibrated estimates* — the correction assumes the model is otherwise well-calibrated, which an uncalibrated tree ensemble is not. But the direction is unambiguous and the implication is serious: **the current threshold may sit close to a coin-flip in true-probability terms.** 🔎 This deserves a proper calibration curve on garage-verified rows (where the true outcome is observed) as a priority piece of analysis — it bears directly on the false-positive cost that the ≥ 0.985 precision floor exists to control.
- **Consequence for IPS:** the known caveat that "uncalibrated scores distort propensity weighting" now has a **concrete, quantified cause** rather than being a generic worry. Any use of v2 scores as propensities must account for the `scale_pos_weight` inflation.

### 5. Other parameters — and what they say about the fit

- **`objective="binary:logistic"`** — a clean binary setup. This **resolves for v2 the ambiguity flagged for v1**, which applied *multiclass* `mlogloss` to a binary target. `predict_proba` returns 2 columns; the score is column `[:, 1]`.
- **`gamma=15.0` vs `min_child_weight=1`** — these pull in opposite directions: gamma prunes aggressively, while min_child_weight is left at its permissive default (tiny leaves allowed). Combined with `max_depth=10`, the trees may be deep but every split must clear a high bar. Consistent with an automated search that found strong pruning + strong L1 rather than a hand-designed configuration.
- **`subsample=1.0`** — **no row subsampling**; randomisation comes only from `colsample_bytree=0.6`. Less variance reduction than the usual setup, and with 450 deep trees the overfitting pressure is held back mainly by `gamma`/`reg_alpha`.
- **`max_delta_step=10`** — the parameter conventionally used for logistic objectives under class imbalance; its presence alongside `scale_pos_weight` is coherent, though at 10 it is permissive enough to rarely bind.
- **`random_state=42`** — reproducible (v1 used seed 10⁹ for the model and `random_state=0` for the split). 🔎 Whether v2's `train_test_split` also uses 42 is still unconfirmed.
- **`n_jobs=-1`** — infra only (v1 used 20).

### ⚠️ 6. A five-year-old model: trained 2021, deployed ~2022, still live in 2026

The training year fixes the timeline and raises two issues the dissertation should state plainly:

- **v2 has been scoring live claims for ~4–5 years without retraining** — and its **training pool closed on 2019-12-02** (data window 2018-01-02 → 2020-09-30, the tail being OOT only — ✅ confirmed 2026-08-04). It is scoring 2026 claims from patterns learned on **2018–2019 data**: six-plus years of drift (vehicle values, parts costs, repair economics, channel mix) unaddressed, with the SFP loop compounding throughout. This also sharpens why v3's failure matters: the only attempted refresh did not ship.
- **⚠️ COVID-19 sits in the OOT block, not the training pool** *(corrects the earlier "training window spans COVID" reading)*. The in-period pool ends 2019-12-02 — entirely **pre-COVID** — while the OOT block (2019-12-02 → 2020-09-30) spans the first UK lockdown, when claim volumes collapsed and the incident mix changed. Two consequences: **(a)** v2 never saw COVID-era or post-COVID claims in training at all — the drift exposure above starts immediately at deployment; **(b)** v2's headline OOT figures were measured partly on an **anomalous period**, so its reported out-of-time performance is not a clean read of normal-times generalisation. *(The earlier volume-interaction worry about the proportional tail is superseded — the realised windows show the date filter was non-binding, see the silent-loss note above.)*
- ✅ The earlier placeholder describing v2's window as "≈ 2022–2024" is **wrong**, and now exactly resolved: the data window ends **2020-09-30**; training happened in 2021.

## Data availability — ✅ CORRECTED 2026-08-05: all three training sets survive; v1's *log* is what is gone

> **⚠️ This section supersedes the previous account, which stated that v1's training dataset had been destroyed under the data-retention period. That was wrong.** The correction **inverts** v1's availability: its *training data* is present, its *production log* is not. Every claim elsewhere in this document that rests on "v1's training data is gone" is retracted; every claim that rests on "infer it from the v1 log" is now the one that fails.

| Version | Training data | Production log | Status |
|---|---|---|---|
| **v1** | ✅ **exists — including the `pre_ml_label` target** | ❌ **gone — scored inputs *and* observed outcomes** (outcome recovery being attempted, not assumed) | deployed, superseded |
| **v2** | ✅ exists (`Z:` drive) | ✅ exists | **currently live** |
| **v3** | ✅ exists (training + OOT) | ❌ **cannot exist — never deployed** | never deployed |

What this changes, in both directions:

- **v1 is now re-derivable, not merely documented.** The split scheme, the `cc_fttl` exclusion and the `veh_fast_track` × `veh_total_loss` target construction can be **audited against real source rows** rather than read off code. The forced-positive rate *within v1's training labels* — previously recorded as "not recoverable" — is now **directly measurable**.
- **`pre_ml_label` survives.** It is v1's training target, so the human-era labels were **not** disposed of after all. **⚠️ This does *not* point-identify the class prior α, and the bound α ∈ [8.55%, 15%] stands.** Data availability is not oracle availability: for the 6.45% of pre-ML cars that were fast-tracked, the recorded `pre_ml_label` is the *forced* 1, and their true status was destroyed by the scrapping itself — no surviving dataset can recover it. What *does* change is that the bound's **inputs become empirical**: the 15% scrap rate, the 43% fast-track share and the 6.45% unverified slice were team-provided figures, and can now be **computed directly from v1's training rows** rather than taken on trust. The bound may therefore tighten or move, but it remains a bound.
- **v2 remains re-derivable end-to-end** and, because its log also survives, remains the anchor for anything requiring observed production behaviour.
- **⚠️ The v1 → v2 hand-off loses its observational side.** The previous fallback — "infer it from v1's production log" — is **no longer available**: that log is gone. v1's production scores and decisions must now be **reconstructed** by re-scoring v1's surviving artefact on surviving feature rows, which makes them *reconstructed* rather than *observed* quantities. Any analysis needing genuine v1 production behaviour (the mobility overlap band of § "v1 scrapping threshold", the error-inheritance detector) must state this explicitly and justify the reconstruction. 🔎 Whether v1's model pickle can be re-scored on v2's training rows depends on feature-space compatibility — see `features/`.
- **⚠️ The `…/Prod-Predictions/inputs.pkl` references in this document are stale** and should be read as historical: that artefact is part of the missing log. (`src/scoring/inspect_pickle.py`, the Python-3.5-compatible inspector, was deleted 2026-08-19 — `notebook/real/01_export_v1.ipynb` reads v1's training pickle inside env-v1 and exports it, under the same pandas-0.22.0 constraint.)
- **v2b is still not reproducible on real data — but for the opposite reason.** It mixes `pre_ml_label` with the v1 log; the pre-ML half is now available and the **v1-log half is not**. The conclusion (synthetic-only) is unchanged; its justification is inverted.

## The decision rule — a single global cutoff (no mobility segmentation)

```python
predictions["FastTrackerDecision"] = (
    predictions["FasterTrackerProbability"] > threshold
).astype(int)      # 1 → fast-track to scrap, 0 → garage
```

Three things this settles:

1. **v2 uses ONE global threshold.** There is no mobility segmentation, no per-segment cutoff, no percentile rule — a single scalar `threshold` compared against a single score column. **This closes the open item left by the v1 section**, which recorded "whether v2/v3 also segment by mobility is unconfirmed".
2. **Therefore the policy *form* is genuinely NOT invariant across versions.** v1 = **two** cutoffs keyed on vehicle mobility (0.75 immobile / 0.85 mobile); v2 = **one** global cutoff. Any statement that "the policy form (a single absolute cutoff) is invariant across versions" is **false as stated** — it holds for v2, not for v1. The invariant is narrower: *an absolute cutoff in score space* (never a percentile), with the **arity** of that cutoff differing by version.
3. **The comparison is strict `>`, not `≥`.** This documentation (and the synthetic generator) writes the rule as `score ≥ τ`; production uses `score > τ`. With continuous scores the difference is measure-zero and practically irrelevant, but it should be **`>` in any code that reproduces the production decision** — an exact-tie row would be classified differently. 🔎 Minor: confirm the exact spelling of the two columns (`FastTrackerDecision` vs `FasterTrackerProbability` — the "Fast"/"Faster" mismatch is transcribed as read and looks like a genuine inconsistency in the production code, not a typo introduced here).

## Threshold history — v2 has FIVE regimes, and the threshold alternates

*(Full history supplied 2026-08-18. Supersedes every earlier account: "constant at 0.872", "one break", and "two changes with an unrecoverable prior era" are all wrong.)*

| Regime | Period | Threshold | Note |
|---|---|---|---|
| 1 | … → 2021-06-03 | **0.8915** | era *start* unknown — the record does not reach back past it |
| 2 | 2021-06-03 → 2024-06-02 | **0.872** | the value most of this repo used to call "v2's threshold" |
| 3 | 2024-06-02 → 2026-02-25 | **0.825** | ~20 months, **inside v3's training window** |
| 4 | 2026-02-25 → 2026-06-30 | **0.872** | raised back; 16:26 UK local (GMT) |
| 5 | 2026-06-30 → present | **0.825** | **currently in force**; 14:30 UK local (BST) = 13:30 UTC |

Four changes, in **alternating directions** (down, down, up, down in value terms: 0.8915 → 0.872 → 0.825 → 0.872 → 0.825). Three consequences follow immediately:

- **A threshold value does not identify a regime.** 0.872 names regimes 2 *and* 4; 0.825 names regimes 3 *and* 5. Two eras sharing a number are still two deployments years apart, with different portfolios, case-mix and model behaviour between them. Never select rows by `τ == 0.872`; select by date span.
- **Never index `regimes` positionally.** `regimes[0]` is now the open-ended 0.8915 era, not "the documented cutoff". Use `config.regime_on("v2", date)`, `config.threshold_on("v2", date)`, or `config.spans_a_break("v2", start, end)` (an empty list is the *only* evidence that a window is single-regime).
- **Only two of the four instants are known to the minute** (2026-02-25 16:26 and 2026-06-30 14:30 UK local); 2021-06-03 and 2024-06-02 are known to the day. All four are therefore treated at **whole-day** resolution, with the change date belonging to the **new** regime. The cost is at most one morning of claims per break sitting in the wrong regime. If an analysis is sensitive to a single day at a break — a narrow-bandwidth RDD at the cutoff, for instance — **drop the break day itself** rather than trusting the assignment.

Recorded once, in `src/config.py::DECISION_RULES["v2"]["regimes"]` — the five spans, and nothing else. A break *is* the boundary between two consecutive spans, so the change list is **derived** (`config.breaks("v2")`) rather than stored: one source of truth, no second copy to fall out of sync. `src/threshold.py::apply()` tiles the half-open spans `[from, until)` and **raises** if any row falls outside every regime. Nothing else may hard-code a date or a threshold.

**Direction is not evidence.** Three of the four changes lowered the bar and one raised it; a lower bar scraps more claims and so mechanically produces more forced positives per unit volume. That is a *mechanism*, not a finding — the SFP reading of the lowering was retired 2026-08-02 (see below).

### ❌ RETRACTED: "v3's training data is single-regime"

**This section previously claimed that all of v3's training logs were generated under `threshold = 0.872`, and that v3 therefore trains on a policy-homogeneous log. That claim is false and is withdrawn (2026-08-18).** It was correct only about the June 2026 break, which was the only change known when it was written.

v3's window is `2023-06-01 → 2026-05-01`. `config.spans_a_break("v2", "2023-06-01", "2026-05-01")` returns **two** breaks:

| v3 window segment | Threshold in force |
|---|---|
| 2023-06-01 → 2024-06-02 | 0.872 |
| 2024-06-02 → 2026-02-25 | **0.825** — ~20 months, the largest part of the window |
| 2026-02-25 → 2026-05-01 | 0.872 |

And v3's **OOT slice** (`2025-12-01 → 2026-05-01`) straddles 2026-02-25 by itself: ~3 months at 0.825 followed by ~2 months at 0.872. Consequences:

- The **2026-06-25 researcher concern** — that a threshold change might have overlapped a retraining window and injected decisions made under a different cutoff into the training labels — is **confirmed, not closed**. It is what happened.
- **v3's labels span three policy segments**, and the middle one (0.825, the loosest bar in the whole history bar none) contributed the majority of the window. A looser bar scraps more claims, so *more* of v3's positive labels are forced positives than a single-regime reading assumed. The direction of this error is not neutral: it makes v3's training data **more** contaminated than previously documented, not less.
- **v3's recall-collapse result was measured on a holdout that straddles a policy break.** That result is load-bearing in the SFP narrative, so it must now be reported with the regime split stated, or re-measured within one regime. 🔎 Re-run the recall-collapse evaluation restricted to `2025-12-01 → 2026-02-25` (0.825) and to `2026-02-25 → 2026-05-01` (0.872) separately, and report both.
- **Practical rule for this project:** any per-row analysis on the real v2 log must **either** restrict to a single regime **or** carry the regime as an explicit covariate — and "single regime" now means one of five date spans, not one side of one break. Silently pooling conflates up to five different treatment assignments, and because the value alternates it can also produce a *false* homogeneity: a pooled slice of regimes 2 and 4 looks like one policy at 0.872 and is not.

> ✅ **This retraction is not itself provisional.** The earlier version of this section was hedged because the change record was incomplete; the record now reaches back to 2021-06-03 and the two breaks inside v3's window are confirmed. What remains unknown is only the *start* of the opening 0.8915 era (before 2021-06-03), which lies ~2 years before v3's window opens and cannot affect it.

### ~~Why the *lowering* is itself SFP evidence~~ — reading RETIRED (2026-08-02)

**Superseded by user confirmation: the 0.872 → 0.825 change was a deliberate manual re-tuning by the team.** The candidate reading previously carried here — that a contaminated precision metric showed false headroom and licensed the lowering (P6 closing into an operational loop) — is **retired and removed from the dissertation** (`report/paper/paper.mid.draft.md` §2.3.2). The break is treated purely as an **exogenous operational policy action**: usable as a natural experiment (the temporal overlap band and cutoff-shift RDD below), with **no SFP inference drawn from the direction of the change**. Retiring it also removes an unfalsifiability tension: P8 predicts contamination surfaces as the threshold being forced *up* (v3's 0.984), so a lowering cannot simultaneously count as loop evidence.

## The threshold changes create a temporal overlap band — (0.825, 0.872], now crossed FOUR times

This is the analytically important consequence, and it **parallels the v1 mobility band exactly, but in time rather than cross-section**. With the full history the band is crossed at every one of the four breaks, in alternating directions:

| Regime | Period | τ | Decision for a score in (0.825, 0.872] | Garage outcome observed? |
|---|---|---|---|---|
| 1 | … → 2021-06-03 | 0.8915 | **garage** | ✓ yes |
| 2 | 2021-06-03 → 2024-06-02 | 0.872 | **garage** | ✓ yes |
| 3 | 2024-06-02 → 2026-02-25 | 0.825 | **scrap** | ✗ no — forced positive |
| 4 | 2026-02-25 → 2026-06-30 | 0.872 | **garage** | ✓ yes |
| 5 | 2026-06-30 → present | 0.825 | **scrap** | ✗ no — forced positive |

**A second, higher band also exists: (0.872, 0.8915].** Those scores were garaged in regime 1 and scrapped in every regime after it, so verified outcomes reach up to **0.8915** — higher than any threshold v2 ever ran at except its first.

**What this gives the framework:**

- **Garage-verified outcomes exist for scores up to 0.872**, drawn from the pre-change era. The blanket premise "**zero garage rows above τ**" is, for v2, only true above **0.872** — not above the *current* 0.825.
- **Positivity holds in the band when pooling across the change date.** Conditional on the score alone, `e(s) = Pr(D=1 | s) = Pr(post-change | s) ∈ (0,1)` for `s ∈ (0.825, 0.872]` — at the same score, a pre-change claim is garaged and a post-change claim is scrapped. Within *either* regime taken alone, positivity is still dead (deterministic step function).
- **A cutoff-shift design becomes available:** the same claims are treated differently before and after an exogenous policy break — a DiD / RDD-with-moving-cutoff, with **two** discontinuity locations (0.872 pre, 0.825 post) instead of one.
- **It anchors the mitigation counterfactual** with *real observed outcomes for high-scoring vehicles* (pre-change, 0.825–0.872), instead of blind extrapolation above a single cutoff.

**⚠️ Two caveats — this band is weaker than v1's:**

- **The overlap is confounded by time, not by an observed covariate.** v1's band is confounded by *mobility*, which is measured and can be adjusted for. v2's band is confounded by **calendar time** — and everything that moves with it: score drift, case-mix, FNOL/ENOL channel mix, seasonality, portfolio composition. There is no clean "mobility adjustment" analogue; identification needs a parallel-trends-style assumption across the break, which is exactly the assumption this project has already found violated at era boundaries elsewhere.
- ~~**The post-change window is ~4.3 weeks**~~ — **this caveat is largely lifted (2026-08-18).** It assumed the only 0.825 era was the one that began 2026-06-30. In fact regime 3 ran **~20 months at 0.825** (2024-06-02 → 2026-02-25), so the treated side of the band has years of rows, not weeks. The power objection now applies only to regimes 4 and 5 individually. 🔎 Still count the actual rows in (0.825, 0.872] per regime *first* and gate on that count — but expect regime 3 to carry the analysis.
- **⚠️ The alternation is a falsification test, and it should be used as one.** The band flips treated→control→treated across the four breaks. A genuine cutoff effect must **change sign with the direction of the change**; anything that moves the same way at every break is time trend, not policy. This is a stronger design than a single break affords, and it costs nothing to run.
- **Longer spans buy power and spend comparability.** Regime 3 gives sample size, but 20 months of drift, case-mix and channel-mix movement sit inside it. Pairing regime 3 with its *adjacent* regimes (2 before, 4 after) keeps the comparison local in time; pairing regime 2 with regime 4 — both at 0.872, ~5 years apart — does not.

## ✅ Resolved (2026-08-18) — the change record now reaches back to 2021-06-03

*(This section previously read "Limitation — the threshold change record is incomplete before 2026-02-25". The gap it described is closed; what the gap was hedging turned out to be true.)*

The unanswered question was how long 0.825 had been in force *before* 2026-02-25. **Answer: since 2024-06-02**, and before that 0.872 since **2021-06-03**, and 0.8915 before that. All four changes are confirmed and encoded in `config.DECISION_RULES["v2"]["regimes"]`.

**What is still unknown, and why it does not matter here.** The *start* of the opening 0.8915 era. That era ends 2021-06-03 — over two years before v3's window opens and more than three years after v2's own training data ends — so no analysis in this project reaches into it. It is encoded as a regime with no `from` bound (open on the left), which means `threshold.apply()` assigns every early row 0.8915 without a guessed start date and without leaving a gap in the timeline.

**What the closed gap cost.** The hedge was one-directional and it broke against the project: the prior 0.825 era **did** reach into v3's window, by ~20 months. Every claim in this document that depended on "v3 is policy-homogeneous" is retracted above. Two further items follow:

- **v3's recall-collapse result** — load-bearing in the SFP narrative — was measured on a holdout straddling 2026-02-25. Report it with the regime split, or re-measure within a regime.
- **The empirical check proposed for the broken record is now a validation, not a substitute.** Running `read_off()` month-by-month across the v2 log should show steps in `min{score | scrapped}` at exactly the four dates above. If it shows a step anywhere else, there is a *fifth* change nobody recorded — run it before trusting the table.

**On the reasoning that failed.** The earlier version of this section argued that the exposure was conditional on an unmeasurable fact, that the risk was one-way, and that the finding should therefore stand with the possibility stated. The risk being one-way was correct; treating "unmeasurable" as grounds to keep the claim was not — the fact turned out to be perfectly measurable, it simply had not been asked for. **Where a hedge is one-directional and the missing fact is obtainable from a person, obtain it before publishing the claim it protects.**

> **Open items for v2.** *Threshold:* ~~(1) exact change date~~ ✅ **closed**; ~~(2) one change or two~~ ✅ **closed 2026-08-18 — FOUR changes, five regimes**; ~~(2b) when the first 0.825 era began~~ ✅ **2024-06-02**; (1b) **the timezone and time-granularity of the log's date column** — decides whether any break can be cut intra-day (all four are handled at whole-day resolution until it is answered); (2c) **monthly `read_off()` across the v2 log** — no longer a substitute for the broken record but a *validation* of the four dates, and the only way to catch a fifth unrecorded change; (2d) **when the opening 0.8915 era began** — still unknown, and harmless: it ends 2021-06-03, before any window this project analyses; (3) stated rationale for each of the four changes; (4) exact column spellings in `score.py`; (5) row counts in (0.825, 0.872] **per regime** — with regime 3 (~20 months at 0.825) expected to dominate, and each break day excluded if the column is date-only.
> *Split:* (6) the **80th-percentile `ReportedDate`** — decides whether the silent data-loss region above is empty or large; (7) realised split sizes (never the nominal 80/20); (8) rows dropped by the 2020-08-24 filter; (9) the train/validation split's `random_state` (`stratify=` now ✅ confirmed); (9b) **positive rate in Train / Validation / OOT Test** — a cheap drift check the stratification makes interpretable; (10) reporting-lag distribution (`ReportedDate − lossdate`) and its drift.
> *Training config (⚠️ highest priority first):* (11) **which of `eta` / `learning_rate` actually took effect** — read the saved booster's config; (12) **v3's hyperparameters**, needed to know whether the SHAP-concentration comparison can be run on a regularisation-matched pair; (13) whether `scale_pos_weight=4.5` was computed from the data or tuned; (14) whether a hyperparameter search was run and what it optimised; (15) a **calibration curve on garage-verified rows** (does 0.825 sit near a coin-flip in true probability?); (16) `train_test_split`'s `random_state`.
> *Highest value of all:* (17) **cross-tab `veh_total_loss` × the fast-track/scrap flag on v2's training data** — decides whether P1 applies to the live model (see § "v2 target").
> *Still no v2 counterpart:* (18) extract SQL and raw shape; (19) whether v1's `cc_fttl` exclusion applies to v2; (20) runtime environment / library pins; (21) feature set. *(Target and XGBoost config now ✅ confirmed.)*


# Model v3 — Data Window (real, confirmed 2026-07-29)

> **Scope: the data window only.** This is the first confirmed real detail for v3. Its **split scheme, target construction, training configuration, runtime environment and feature set are all still 🔎 TBD** — do not assume it follows v2's.

v3 selects its dataset with a `subset_by_date` function applied to **`ReportedDate_CLAIM`**:

```python
old_date      = DATA_START_DATE   # 2023-06-01
immature_date = DATA_END_DATE     # 2026-05-01

df.filter((df['ReportedDate_CLAIM'] > old_date) & (df['ReportedDate_CLAIM'] < immature_date))
```

Both bounds are strict, giving a window of **2 years 11 months**: `2023-06-01 < ReportedDate_CLAIM < 2026-05-01`.

## Split scheme — a date-bounded OOT block plus a shuffled in-period split

v3 divides the window into **two kinds**: an out-of-time block and a non-OOT pool that is then split.

```python
# OOT      : ReportedDate_CLAIM >= 2025-12-01
# non-OOT  : ReportedDate_CLAIM <  2025-12-01

shuffled = non_oot.sample(fraction=1, shuffle=True, seed=123)   # full permutation
test  = shuffled.head(0.2 * len(non_oot))     # first 20%
train = shuffled.tail(-0.2 * len(non_oot))    # "all but the first 20%" → the other 80%
```

| Role | Definition | Span |
|---|---|---|
| **Train** | 80% of the shuffled non-OOT pool | 2023-06-01 → 2025-12-01, interleaved |
| **Test** *(in-period)* | 20% of the same shuffled pool | same span, interleaved |
| **OOT** | `ReportedDate_CLAIM ≥ 2025-12-01` | 2025-12-01 → 2026-05-01 — **5 months** |

**This is the cleanest split of the three versions.** The OOT boundary is an explicit date (not a proportional tail), the 5-month holdout closely matches the documented "~6 months OOT" methodology, and `head(0.2n)` / `tail(-0.2n)` are **exactly complementary** — no overlap, no gap, and none of v2's silent data-loss problem. v3 also names the out-of-time block **"OOT"** rather than reusing "Test" or "Validation", which is the clearest of the three conventions.

*(Implementation note: `sample(fraction=…, shuffle=…, seed=…)`, `filter`, and `tail(-k)` are **Polars**, not pandas — v3 is a modern rewrite. `tail(-k)` means "drop the first k rows". Third split seed convention across versions: v1 `random_state=0`, v2 `42` for the model, v3 `seed=123`.)*

### Is shuffling the in-period pool a good choice?

**Yes — it is a reasonable and standard design.** The temporal question is handled by the separate, date-bounded OOT block, so the in-period split is free to measure in-distribution fit. v1 and v2 both use in-period random splits too.

One clarification on the wording: shuffling does not *remove* time bias — it makes the in-period test set match train in time, so that test set simply **cannot show drift**. Drift is measured by the OOT block instead. So the in-period number is an in-distribution figure, and generalisation claims should rest on OOT. Nothing wrong with the code; just don't quote the in-period number as evidence of generalisation.

Two smaller notes:

- **No stratification.** `sample(fraction=1, shuffle=True, seed=123)` + `head`/`tail` has no `stratify=`. v2 stratified on the target; v3 does not. With a minority positive class and a precision ≥ 0.985 target, the 20% test set's positive count affects how stable the precision estimate is. 🔎 Confirm stratification isn't applied elsewhere in v3's pipeline.
- 🔎 **If a claim can contain more than one vehicle**, a random split can put two vehicles from the same claim on opposite sides, which would flatter the in-period test score. A quick check of the vehicles-per-claim distribution settles it; if multi-vehicle claims are common, group the split by claim. The OOT block is unaffected either way.

> **One question worth settling: which split produced v3's recall collapse?**
>
> "Recall collapsed when precision was held at ≥ 0.985" is a load-bearing fact, and it means different things depending on where it was measured:
>
> - **On the OOT block:** some of the collapse could just be ordinary drift over those 5 months, not SFP contamination — so the finding is confounded.
> - **On the in-period Test:** drift is ruled out by construction, so the collapse is harder to explain away — a stronger result for the SFP argument.
>
> 🔎 Worth confirming which one it was.

## v3 decision rule & threshold — single cutoff, and far stricter than v2

**Target column: `Fttl`.** Threshold is tuned to a precision target, and two calibrations are recorded:

| Precision target | v3 threshold | v2's threshold for comparison |
|---|---|---|
| **0.985** (the business floor) | **0.984** | 0.872 |
| 0.97 | **0.970** | — |

Three things this settles or raises.

**1. ✅ v3's decision-rule form is a single absolute cutoff** — same form as v2, *not* v1's mobility-segmented pair. This closes the last open item on decision-rule form across the three versions: **v1 alone is segmented.**

**2. ✅ It explains the recall collapse quantitatively.** To reach precision 0.985, v3 needs a cutoff of **0.984** — pressed right against the top of its score range. Almost nothing clears that bar, which is exactly what "recall collapsed when precision was held at ≥ 0.985" means. Relaxing the target to 0.97 only buys back a cutoff of 0.970, so the recall recovery from that relaxation is modest: the scores are **densely packed at the top**, and precision is extremely sensitive to the cutoff in that region. The two-point pair (0.985→0.984, 0.97→0.970) is itself the evidence for that density.

**3. ⚠️ But raw thresholds are NOT comparable across versions.** "0.984 vs 0.872" does not by itself mean v3 is stricter, because the two models' score distributions and `scale_pos_weight` values differ. Applying the same crude odds correction used for v2 (divide predicted odds by `scale_pos_weight`):

| | threshold | `scale_pos_weight` | implied true probability *(illustrative)* |
|---|---|---|---|
| v2 @ precision 0.985 | 0.872 | 4.5 | ≈ **0.60** |
| v2 current | 0.825 | 4.5 | ≈ **0.51** |
| **v3 @ precision 0.985** | 0.984 | 5.552301 | ≈ **0.92** |
| v3 @ precision 0.97 | 0.970 | 5.552301 | ≈ **0.85** |

On this (rough) like-for-like footing v3 really is far more conservative — it demands roughly a 0.92 chance of total loss where v2 demands about 0.60. **That is the shape of the collapse:** v3 could only hold the precision floor by scrapping almost nothing. As always these are illustrations, not calibrated estimates.

### ✅ `Fttl` resolved (2026-08-04): it is `veh_total_loss` under a new name

**User-confirmed from the real data: `Fttl` carries exactly the same values as `veh_total_loss`** — a rename, not a new construction (consistent with the `ReportedDate` → `ReportedDate_CLAIM` rename in the same rewrite). The earlier decision-vs-outcome ambiguity therefore dissolves in a specific way: the column is nominally the **outcome as recorded**, but — per the 2026-08-04 confirmation in the v1/v2 sections — the recorded value is **set to 1 for fast-tracked vehicles without garage verification**, so for fast-tracked rows its recorded value *is* the decision. **P1 holds for v3's target through the data, exactly as for v1 and v2.**

**Cross-version status of the target definition:**

| Version | Target | Status |
|---|---|---|
| **v1** | `target` = `veh_total_loss OR veh_fast_track` (disjunct = safeguard) | ✅ **Full derivation confirmed; forced 1s already in the recorded column** |
| **v2** | **`veh_total_loss`** — the raw flag alone | ✅ **Confirmed — carries the forced 1s in the data** |
| **v3** | `Fttl` **= `veh_total_loss` renamed** | ✅ **Confirmed 2026-08-04 — identical values; carries the forced 1s** |

## v3 model training call — real XGBoost configuration (**xgboost 3.2.0**)

```python
xgb.XGBClassifier(
    colsample_bytree=0.887008...,
    subsample=0.980460...,
    max_depth=3,                     # v2 used 10
    gamma=0.0004847861...,           # v2 used 15.0
    learning_rate=0.09973689...,     # single value — no `eta` clash (v2's bug is gone)
    min_child_weight=44,             # v2 used 1
    n_estimators=802,                # v2 used 450
    reg_lambda=1.182373785...,       # v2 used 0.0123626 — near-default now
    scale_pos_weight=5.552301...,    # v2 used 4.5
    max_delta_step=61,               # v2 used 10
    max_leaves=18,                   # v2 did not set this
    grow_policy="depthwise",
    n_jobs=-1,
    random_state=123,
)
```

*(Decimals are reproduced exactly as supplied; the trailing `...` marks values truncated at source, not by rounding here.)*

### ⚠️ v3 does NOT share v2's regularisation — the confound is not escaped

The v2 section flagged v2's `reg_alpha = 20.0` as a **competing mechanical explanation** for any rise in SHAP feature-importance concentration, and named "obtain v3's hyperparameters" a **blocking prerequisite** — because the SHAP–DiD design falls back to the **v2a → v3a** pair, which only works if that pair is regularisation-matched.

**It is not matched. The regularisation is effectively inverted:**

| | v2 | v3 |
|---|---|---|
| `reg_alpha` (L1) | **20.0** — very strong | **not set ⇒ 0 (default)** |
| `reg_lambda` (L2) | **0.0123626** — ~80× below default | **1.182373785** — near default |
| `gamma` | **15.0** — heavy pruning | **0.0004847861** — effectively none |

And the capacity controls differ just as much: v3 uses **shallow, tightly-constrained trees** (`max_depth=3`, `max_leaves=18`, `min_child_weight=44`, 802 rounds), where v2 used **deep trees under heavy penalties** (`max_depth=10`, `min_child_weight=1`, 450 rounds). The two models control complexity by **entirely different mechanisms** — v2 via penalty terms, v3 via tree structure.

> **Consequence: the v2a → v3a pair is confounded too.** Every capacity and regularisation knob differs, so a difference in feature-importance concentration between v2 and v3 **cannot be attributed to the data (and hence to SFP) without controlling for the configuration.** The "run it on v2a → v3a instead" escape route from the v1 → v2a parallel-trends problem is therefore **not available as-is**. Both candidate pairs are now confounded, each for a different reason.
>
> The remaining options are the ones already listed in the v2 section, now **required** rather than optional: **(i)** matched-hyperparameter re-training — ✅ **now feasible for all three versions** (corrected 2026-08-05: v1's and v3's training data both survive, not just v2's), so a configuration can be held fixed while the training *data* is varied across every generation, which is the clean separation this confound needs; **(ii)** a regularisation/capacity sensitivity sweep bounding how much of the observed ΔC the configuration alone can produce; **(iii)** report the concentration result as *consistent with* the loop rather than as identifying it. Whichever route, **every version's configuration must be printed beside its concentration figures.**
>
> **✅ This is the largest single gain from the 2026-08-05 correction.** The confound was previously irreducible on the v1 → v2 pair because v1's data was believed destroyed, leaving only option (iii). With all three training sets in hand, the design can now run **the same configuration on v1's, v2's and v3's data** and read the concentration difference as attributable to the data rather than the regularisation. 🔎 Library-version matching (xgboost 1.4.2 vs 3.2.0) remains a separate obstacle and is *not* solved by this.

> 🔎 **Confirm `reg_alpha` is genuinely absent from v3's call.** The list above does not include it, nor `objective` or `eval_metric`, so the listing may be partial. This matters: if v3 in fact sets `reg_alpha` near v2's value, the confound narrows sharply. Also confirm v3's `eval_metric` — v2 used `auc`, which the v2 section flags as misaligned with the precision ≥ 0.985 constraint.

### What v3 fixed, and one thing it did not

- ✅ **The `eta` / `learning_rate` clash is gone.** v3 sets `learning_rate` only (`0.09973689`), so its effective learning rate is unambiguous — unlike v2, where two aliases carry different values.
- ✅ **`reg_lambda` is back near its default** (1.18 vs 1.0), instead of v2's effectively-disabled 0.0123626.
- ⚠️ **`scale_pos_weight` rose from 4.5 to 5.552301.** The score is therefore *further* from a probability than v2's, and the caveat that v2's cutoff must not be read as a probability applies to v3 at least as strongly.
- ⚠️ **`max_delta_step` jumped from 10 to 61.** At that magnitude the cap is unlikely ever to bind, so it is effectively inert — 🔎 worth confirming it was deliberate rather than a search artefact.

### ⚠️ The xgboost version jump also breaks comparability: 1.4.2 → 3.2.0

v2 ran on **xgboost 1.4.2**; v3 runs on **3.2.0** — spanning two major releases. Identical nominal hyperparameters do **not** guarantee identical models across that gap, because library defaults and internals changed. In particular, the default `tree_method` moved to `hist` in the 2.x line, so v2 and v3 may be building trees by **different algorithms** even where their parameters agree. 🔎 Confirm the effective `tree_method` for each version.

This compounds the point above: any "matched-hyperparameter re-training" remedy must match the **library version** as well as the parameters, or it will not isolate the data effect. It also fits the per-version frozen-environment constraint (§ "Model artefacts and reproduction").

## ✅ This completes the SFP inheritance chain — and it is unbroken

v3's window opens in June 2023, **after v2 went live (~2022)**. So v3's training data was generated **entirely under v2's scrapping policy** — exactly the structure already confirmed one generation earlier, where v2's window (2018+) fell entirely after v1's deployment.

| Generation | Training window | Policy in force during that window | Contamination |
|---|---|---|---|
| v2 | `2018-01-02 → 2019-12-02` (pool; OOT → 2020-09-30) | v1 (deployed ~2017) | essentially total |
| v3 | `2023-06-01 → 2026-05-01` | v2 (deployed ~2022) | **total** |

> **This is a stronger statement than the thesis has been making.** The forced-label hand-off is not "partial contamination that accumulates over generations" — at the code level **each model generation trains exclusively on its predecessor's forced labels**, with no clean-label component mixed in at any step. The chain `pre-ML → v1 → v2 → v3` is unbroken, and both hand-offs are confirmed from the production window definitions — and, as of 2026-08-04, at the **label level** as well: the recorded `veh_total_loss` / `Fttl` is 1 for fast-tracked vehicles in every era, so the forced value demonstrably sits in each generation's target data.

## ❌ WITHDRAWN: "v3's window ends before the threshold break"

*(This section previously argued that because v3's window closes 2026-05-01 and the threshold changed on 2026-06-30, v3's labels were all generated under `τ = 0.872` — and treated the agreement of two sources as confirmation.)*

**The premise was true and the conclusion was false.** v3's window does end ~2 months before the **June** break. It does not end before the **February 2026** or **June 2024** breaks, both of which fall *inside* it. The two agreeing sources agreed only because both were reasoning from an incomplete change record — a shared blind spot, not independent corroboration.

The methodological lesson is worth keeping: **"no break after the window" is not the same test as "no break inside the window".** The check to run is `config.spans_a_break("v2", window_start, window_end)`, which for v3 returns two breaks, not zero.

## ⚠️ `immature_date` — the maturation buffer made explicit, and a dating puzzle

The end bound is named **`immature_date`**: claims reported after it are excluded because their labels have not yet matured (the outcome is not final). This is the generic "~1–2 months excluded from training" rule (§ "Training Process") appearing as a concrete constant — and it is an **improvement over v2**, whose window as read shows no comparable upper bound. 🔎 Confirm whether v2 had a maturation exclusion elsewhere; if not, v2's most recent training rows carried immature labels, which is an independent source of label noise in v2 that v3 corrected.

> **⚠️ Dating discrepancy — needs resolving, because it is load-bearing.** This document records v3 as **"attempted 2025"**. But a maturation buffer of the documented 1–2 months against `DATA_END_DATE = 2026-05-01` implies the run happened around **mid-2026**, not 2025. Three readings: (a) v3 was **re-attempted in 2026** and the 2025 date refers to an earlier attempt; (b) there have been **multiple v3 attempts**; (c) the constants were updated in the repo without a corresponding retraining run. This matters because "v3 was attempted, recall collapsed at precision ≥ 0.985" is a load-bearing fact in the SFP narrative — **which attempt that finding came from changes what it is evidence about.** 🔎 Confirm the actual v3 training run date(s) and which one produced the recall-collapse result.

## Date columns across versions

| Version | Date column | Meaning |
|---|---|---|
| v1 | `lossdate` | accident date |
| v2 | `ReportedDate` | notification date |
| v3 | `ReportedDate_CLAIM` | ✅ **same field as v2's `ReportedDate`** — renamed only (confirmed 2026-07-29) |

So **v2 and v3 split on the same variable** and their windows are directly comparable. Only **v1 differs**, splitting on `lossdate` (accident date) rather than notification date — the two are separated by the reporting lag, so v1's boundaries are not directly comparable with v2's or v3's.

## ⚠️ v2's and v3's training windows do NOT overlap — a ~2¾-year gap between them

v2's data ends **2020-09-30** (and its training pool already on **2019-12-02** — ✅ confirmed 2026-08-04); v3's window opens **2023-06-01**. **The windows are disjoint: ~2 years 8 months from v2's last data to v3's first, and ~3.5 years from the close of v2's training pool.** The gap contains both the COVID period and the 2021–2022 price surge below.

This bears directly on the **SHAP–DiD design, which is specified on the v2a → v3a pair** (chosen because parallel trends visibly fails across the pre-ML → v1 era boundary). The gap is not empty of events — it spans the **2021–2022 UK used-vehicle price surge**, which is not a neutral background shift for this problem: a vehicle is a total loss when repair cost exceeds a fraction of its market value, so a sharp rise in market values **mechanically reduces the total-loss rate** at unchanged damage severity. Parts costs and repair-labour inflation over the same period push the other way.

> 🔎 **Check this against the actual DiD specification before relying on the v2a → v3a estimate.** The concern is not automatically fatal — the SHAP–DiD compares attribution concentration across contaminated/clean partitions, which is not the same object as a raw outcome trend, and the estimator may be computed on a common evaluation set rather than on the disjoint training windows. But the two versions learned from economically different eras with a two-year unobserved gap between them, and the design's justification rests on parallel trends holding for **this** pair. Establish explicitly which quantities are compared over which time support, and whether the gap enters. Note this is now the **second** structural threat to that pair, alongside the `reg_alpha` regularisation confound (§ "Model v2").

## Window lengths across versions

| Version | Window | Span |
|---|---|---|
| v1 | `2016-01-01 → 2018-02-09` (full extract); train `< 2017-04-01` | ~1 yr 3 mo (training portion) / 2 yr 1 mo (full) |
| v2 | `2018-01-02 → 2020-09-30` (data); in-period pool `→ 2019-12-02`; trained 2021 | **1 yr 11 mo (pool)** / 2 yr 9 mo incl. OOT |
| v3 | `2023-06-01 → 2026-05-01` | 2 yr 11 mo |

v3's window is the longest; v2's training pool is under two years (the earlier "~3 yr 4 mo" figure counted up to the 2021 training date and is superseded); v1's training portion is ~1¼ years. v3's start date also implements the previously-noted intent to **drop the pre-COVID period** — now confirmed as an explicit constant rather than an approximate description.

> **Open items for v3.** *Highest value first:* (1) **which split produced the recall-collapse figure** — in-period Test or OOT (changes how much weight the finding carries, see above); (2) **v3's hyperparameters**, a blocking prerequisite for the SHAP-concentration design (§ "Model v2", `reg_alpha` confound); (3) the training run date(s), given the 2025-vs-2026 discrepancy above.
> (4) ~~what `Fttl` actually means~~ ✅ **resolved 2026-08-04** — `Fttl` = `veh_total_loss` renamed, identical values, carries the forced 1s (P1 in the data; see above).
> *Remaining:* (5) realised split sizes; (6) vehicles-per-claim distribution (decides whether the shuffle needs grouping); (7) whether stratification is applied elsewhere; (8) whether the `cc_fttl` exclusion applies; (9) cleaning exclusions; (10) runtime environment; (11) feature set; (12) whether v3's training data still exists. *(Decision-rule form and threshold now ✅ confirmed.)*


# Cross-Version Dataset Summary (v1–v3)

Consolidated view of each version's train / test / validation / OOT splits — sizes and periods.
**v1 is real (confirmed 2026-07-28); v2 and v3 are placeholder templates awaiting the real numbers.**

> **Reading notes.** (1) The **split *scheme* differs by version, and so does the splitting variable** —
> v1 uses **`lossdate`** cut-offs with a non-OOT random Test plus two date-bounded validation sets (see
> "Model v1 — Actual Data Split"); v2 uses **`ReportedDate`** with a proportional head/tail split (see
> "Model v2"); v3's scheme is **unconfirmed**. (2) **The split names are inverted between v1 and v2** —
> rows below are therefore labelled by **role**, with each version's own name in brackets. (3) The
> v1 sizes below are the **raw `lossdate`-split counts *before* the `cc_fttl` exclusion**; post-exclusion
> counts (Train 173,758 · Test 43,471 · Val1 35,254 · Val2 51,189) are in the v1 section above.

| Version | Role *(version's own name)* | Condition / window | Size | Split type / OOT? |
|---|---|---|---:|---|
| **v1** | Train *(Train)* | `lossdate < 2017-04-01`, then 80% | 178,435 | — |
| **v1** | In-period holdout *(**Test**)* | random split from Train (80/20, `random_state=0`) | 44,609 | Random (non-stratified); **not OOT** |
| **v1** | Temporal holdout 1 *(Val1)* | `2017-04-01 ≤ lossdate < 2017-06-30` | 36,049 | Temporal (OOT, ~3 months) |
| **v1** | Temporal holdout 2 *(Val2)* | `2017-06-30 ≤ lossdate ≤ 2018-02-09` (dataset end) | 52,498 | Temporal (OOT, ~7.5 months) |
| **v2** | Train *(Train)* | 80% of the `ReportedDate` `2018-01-02 → 2019-12-02` pool (sorted head) | 🔎 TBD | Random, **stratified on target**; `random_state` 🔎 |
| **v2** | In-period holdout *(**Validation**)* | remaining 20% of the same `2018-01-02 → 2019-12-02` pool | 🔎 TBD | Random, **stratified**; **not OOT** |
| **v2** | Temporal holdout *(**Test**)* | `2019-12-02 → 2020-09-30` (realised ✅ 2026-08-04; tail then `> 2019-12-01` filter, non-binding) | 🔎 TBD sizes | Temporal (OOT, ~10 months) |
| **v3** | Train *(Train)* | 80% of the shuffled `ReportedDate_CLAIM < 2025-12-01` pool | 🔎 TBD | Random shuffle (`seed=123`), **not stratified** |
| **v3** | In-period holdout *(**Test**)* | first 20% of the same shuffled pool | 🔎 TBD | Random; **not OOT** |
| **v3** | Temporal holdout *(**OOT**)* | `2025-12-01 ≤ ReportedDate_CLAIM < 2026-05-01` | 🔎 TBD | Temporal (OOT, **5 months**) |

**v2 window:** in-period pool `2018-01-02 → 2019-12-02`, OOT `2019-12-02 → 2020-09-30` ✅ (realised windows confirmed 2026-08-04), minus all rows with `ReportedDate == 2020-08-24` (which falls **inside the OOT block**). The realised OOT start (2019-12-02) shows the hard-coded `> 2019-12-01` filter was **non-binding**, so the silent-loss region is empty or near-empty for this run. Sizes still 🔎 — report realised sizes only.

**v1 date boundaries:** `EndTrainDate = 2017-04-01`, `EndValidationDate = 2017-06-30` (both on `lossdate`). Source extract shape `(311,591 × 198)`; splits sum to 311,591 (no corrupted-row drop in v1). v2/v3 boundaries: 🔎 TBD.

**✅ v2 cleaning detail — confirmed 2026-07-29.** v2's pipeline drops rows with `ReportedDate == 2020-08-24` as corrupted. This belongs to **v2**, not v1 (open item now closed). ⚠️ Because the filter is an equality test on the date, it removes **every claim reported that day**, not a single record. 🔎 Still TBD: v2's raw extract shape, the row count dropped, and whether any other corrupted rows are removed.

> **⚠️ The table above is keyed on `lossdate`, which is v1's splitting variable only.** v2 splits on **`ReportedDate`** (see § "Model v2"). The two are separated by the reporting lag, so **v1 and v2 window boundaries are not directly comparable** and must not be plotted on a common axis without first converting or quantifying the lag.

## Split scheme by version

| | **v1** | **v2** | **v3** |
|---|---|---|---|
| Splitting variable | `lossdate` (accident date) | **`ReportedDate`** | `ReportedDate_CLAIM` — **same field as v2's**, renamed |
| How the OOT boundary is set | **explicit cut-off dates** (`EndTrainDate`, `EndValidationDate`) | **proportional** `tail(⌊0.2n⌋)`, then date-filtered | **explicit date** (`≥ 2025-12-01`) |
| Data window | `2016-01-01 → 2018-02-09` ✅ | `2018-01-02 → 2020-09-30` (pool → 2019-12-02) ✅ | `2023-06-01 → 2026-05-01` |
| Cleaning exclusions | `cc_fttl` rule-flagged rows (~2.54%, all splits) | all rows with `ReportedDate == 2020-08-24` | 🔎 TBD |
| Maturation buffer? | 🔎 not visible | 🔎 not visible | ✅ **Yes** — `immature_date` |
| In-period split stratified? | ❌ **No** — plain random | ✅ **Yes** — `stratify=target` | ❌ **No** — plain shuffle |
| Genuine OOT holdout? | Yes — named **Val1 / Val2** | Yes — named **Test** | Yes — named **OOT** |
| Headline "Test" set is OOT? | ❌ **No** — in-period | ✅ **Yes** | ❌ **No** — in-period |
| Realised sizes derivable from n? | Yes | ❌ **No** — post-hoc filter shrinks OOT silently | ✅ Yes — head/tail exactly complementary |
| Library | pandas 0.22 / Py 3.5.2 | pandas | **Polars** |
| Split seed | `random_state=0` | 🔎 TBD (model: 42) | `seed=123` |
| Training data still exists? | ✅ **Yes** — incl. `pre_ml_label` ✅ *(corrected 2026-08-05)* | ✅ **Yes** (`Z:` drive) | ✅ **Yes** (training + OOT) |
| Production log still exists? | ❌ **Gone** — inputs *and* outcomes | ✅ **Yes** | ❌ n/a — never deployed |

**Overall, v3's split design is the soundest of the three** — explicit OOT date, a ~5-month holdout matching the documented methodology, exactly complementary head/tail, an explicit maturation buffer, and unambiguous naming. Its two weaknesses are the **loss of v2's stratification** and the **claim-level leakage exposure** created by shuffling (see § "Model v3").

## Model configuration by version — hyperparameters, package versions, training data

Consolidated reference. **Decimals are exactly as supplied from the real code; trailing `…` marks values truncated at source.** Blank `—` means the parameter is not set in that version's call (so the library default applies); 🔎 means not yet confirmed.

### Environment & training data

| | **v1** | **v2** | **v3** |
|---|---|---|---|
| **xgboost version** | 🔎 — pre-1.0 (uses deprecated `silent=`) | **1.4.2** | **3.2.0** |
| Python / dataframe stack | Python 3.5.2, pandas 0.22.0 | pandas 🔎 | **Polars** |
| Model trained | ~2017 | **2021** | 🔎 (2025 or 2026 — unresolved) |
| Deployed? | Yes, superseded | **Yes — currently live** | ❌ Never |
| Training window | `lossdate < 2017-04-01` (extract → 2018-02-09) | `2018-01-02 → 2019-12-02` (pool; OOT → 2020-09-30) ✅ | `2023-06-01 → 2026-05-01` |
| Window span | ~1 yr 3 mo (train) / 2 yr 1 mo (extract) | **1 yr 11 mo (pool)** / 2 yr 9 mo incl. OOT | 2 yr 11 mo |
| Split variable | `lossdate` (accident date) | `ReportedDate` | `ReportedDate_CLAIM` *(= v2's field, renamed)* |
| Label source | `pre_ml_label` (human era) | v1 production log | v2 production log |
| Contamination | partial (human forced labels) | **essentially total** | **total** |
| Rows (raw extract) | 311,591 × 198 | 🔎 | 🔎 |
| In-period split stratified? | ❌ No | ✅ Yes (`stratify=target`) | ❌ No |
| Maturation buffer | 🔎 not visible | 🔎 not visible | ✅ `immature_date` |
| Training data still exists? | ✅ Yes — incl. `pre_ml_label` ✅ | ✅ Yes (`Z:`) | ✅ Yes (training + OOT) |
| Production log still exists? | ❌ **Gone** (inputs + outcomes) | ✅ Yes | ❌ n/a — never deployed |

### Hyperparameters

| Parameter | **v1** | **v2** | **v3** |
|---|---|---|---|
| **target column** | `target` = `veh_total_loss ∨ veh_fast_track` ✅ *(disjunct = safeguard)* | **`veh_total_loss`** alone ✅ | `Fttl` **= `veh_total_loss` renamed** ✅ 2026-08-04 |
| **scrap threshold** | 0.75 immobile / 0.85 mobile | 0.872 → **0.825** (≈2026-07) | **0.984** @ prec 0.985; 0.970 @ prec 0.97 |
| `objective` | — *(implied multiclass)* | `binary:logistic` | 🔎 |
| `eval_metric` | `mlogloss` | `auc` | 🔎 |
| `n_estimators` | — *(default)* | 450 | **802** |
| `max_depth` | — *(default)* | **10** | **3** |
| `max_leaves` | — | — | **18** |
| `grow_policy` | — | `depthwise` | `depthwise` |
| `learning_rate` | — *(default)* | **0.0887667…** ⚠️ | **0.09973689…** |
| `eta` *(alias of above)* | — | **0.147686…** ⚠️ clash | — |
| `gamma` | — *(default 0)* | **15.0** | **0.0004847861…** |
| `min_child_weight` | — *(default 1)* | 1 | **44** |
| `max_delta_step` | — | 10 | **61** |
| `subsample` | — *(default 1.0)* | 1.0 | **0.980460…** |
| `colsample_bytree` | — *(default 1.0)* | 0.6 | **0.887008…** |
| **`reg_alpha` (L1)** | — *(default 0)* | **20.0** | **— ⇒ 0** 🔎 |
| **`reg_lambda` (L2)** | — *(default 1.0)* | **0.0123626…** | **1.182373785…** |
| `scale_pos_weight` | — *(default 1)* | 4.5 | **5.552301…** |
| `random_state` / seed | `10**9` (model), `0` (split) | 42 | **123** |
| `n_jobs` | 20 | −1 | −1 |
| *(other)* | `silent=False`, `eval_set` passed to constructor ⚠️ | — | — |

**How to read this table.** v1 sets almost nothing — it is essentially stock defaults with a metric, seed and thread count. v2 and v3 are both clearly the output of automated hyperparameter searches (the long decimals), but they **regularise by opposite mechanisms**: v2 leans on **penalty terms** (`reg_alpha=20`, `gamma=15`, L2 switched off) with deep trees; v3 leans on **tree structure** (`max_depth=3`, `max_leaves=18`, `min_child_weight=44`) with penalties near default. ⚠️ Because *no* adjacent pair of versions shares a configuration, **no cross-version comparison of feature-importance concentration is clean** — see § "Model v3" for what this does to the SHAP–DiD design.

## Decision rule by version

Split *sizes* are still TBD for v2/v3, but the **decision rules** are now partly confirmed and differ in form — worth tabulating separately so the difference is not lost inside the size table above.

| Version | Decision rule | Arity | Confirmed? |
|---|---|---|---|
| **v1** | `D = 1[s > 0.75]` if immobile; `D = 1[s > 0.85]` if mobile | **two** cutoffs, segmented by vehicle mobility | ✅ 2026-07-28 |
| **v2** (→ 2021-06-03) | `D = 1[s > 0.8915]` | **one** global cutoff | ✅ 2026-08-18 — era *start* unknown |
| **v2** (2021-06-03 → 2024-06-02) | `D = 1[s > 0.872]` | **one** global cutoff | ✅ 2026-08-18 |
| **v2** (2024-06-02 → 2026-02-25) | `D = 1[s > 0.825]` | **one** global cutoff | ✅ 2026-08-18 — **inside v3's window** |
| **v2** (2026-02-25 → 2026-06-30) | `D = 1[s > 0.872]` | **one** global cutoff | ✅ 2026-08-18 — **inside v3's window** |
| **v2** (2026-06-30 →) | `D = 1[s > 0.825]` | **one** global cutoff | ✅ 2026-07-29 (`score.py`), currently in force |
| **v3** | `D = 1[s > 0.984]` at precision 0.985 (`> 0.970` at precision 0.97) | **one** global cutoff | ✅ 2026-07-29 — but **never deployed** |

**Where garage-verified outcomes exist above a scrap cutoff** (i.e. where the "zero garage rows above τ" premise breaks):

| Version | Overlap band | Source of overlap | Confound |
|---|---|---|---|
| **v1** | `(0.75, 0.85]` | **cross-sectional** — mobile vehicles are garaged where immobile ones are scrapped | **mobility** (observed → adjustable) |
| **v2** | `(0.825, 0.872]`, crossed at **all four** breaks; plus `(0.872, 0.8915]` from regime 1 | **temporal** — claims at the same score are garaged in the 0.872/0.8915 regimes and scrapped in the 0.825 ones | **calendar time** (not a single observed covariate → needs a parallel-trends-style assumption) |

Both bands are **confounded, not random**, so neither restores clean positivity — but both are exploitable, and both mean the premise "no garage-verified outcome exists above the scrap cutoff" is **too strong as stated**. v1's band is the better-identified of the two (its confounder is measured). v2's is confounded by time, but with the full history it is far larger than the ~4-week figure recorded here before 2026-08-18: regime 3 alone ran ~20 months at 0.825, and the band's treated/control status **flips four times**, which is itself a falsification test — a real cutoff effect must change sign with the direction of the change.


# Training Process (Insurance Company.-aligned)

## Data & Model Timeline

### 1. Production log — what was happening and when

| Period | Model running | Who made scrap decisions | Label generated | Researcher access |
|---|---|---|---|---|
| Before ~2018 | None (pre-ML era) | Handlers + engineers | `pre_ml_label` | ✅ **Available** — survives as v1's training target *(corrected 2026-08-05)* |
| ~2018 – pre-v2 deployment | None → v1 (transition date unknown) | Handlers/engineers → Model v1 | `pre_ml_label` → `model_v1_observed_outcome` | ⚠️ **Split:** `pre_ml_label` ✅ available; **v1 production log ❌ gone** (inputs + outcomes) |
| ~2022 – present | **v2** *(currently live)* | Model v2 | `model_v2_observed_outcome` | ✓ v2 scores + decisions + training data |

> ✅ **Corrected 2026-08-05 — this table previously had both v1 rows backwards** (`pre_ml_label` marked disposed, v1 log marked available). The truth is the reverse: the pre-ML labels survive inside v1's training data, and it is the v1 *production log* that is gone. See § "Data availability".
>
> v1's exact deployment and end dates are not confirmed. The ~2022 figure for v2 deployment is approximate.

---

### 2. Model training — what each model learned from

| Model | Trained on | Training label | Training window | Deployment |
|---|---|---|---|---|
| **v1** | Pre-ML era production log | `pre_ml_label` (human handler/engineer decisions) | `lossdate < 2017-04-01` (extract → 2018-02-09) ✅ — **training data survives, `pre_ml_label` included** | ✓ Deployed (dates TBC) |
| **v2** *(currently live)* | v1 production log only | `model_v1_observed_outcome` | `2018-01-02 → 2019-12-02` pool ✅. ⚠️ *(corrected 2026-08-05)* `pre_ml_label` was **not** disposed at retraining time — why v2 was trained on the v1 log alone is therefore an open question, not a data-availability necessity | ✓ Currently active |
| **v3** *(not deployed)* | v2 production log | `model_v2_observed_outcome` | ✅ Window confirmed: `2023-06-01 < ReportedDate_CLAIM < 2026-05-01` (drops the pre-COVID period, as previously described). ⚠️ "Attempted 2025" conflicts with an end date of 2026-05-01 — see § "Model v3" | ✗ Recall collapsed when precision held at ≥ 0.985 |

> The "drop pre-COVID data" consideration came from the v3 retraining attempt — it was not part of v2's training design.

---

### 3. What this research can and cannot access

| | Pre-ML + v1 era | v2 production period (~2022 – present) |
|---|---|---|
| **Claim features** | ✅ **Yes** — v1's training data survives | ✓ |
| **Model scores** | ❌ `model_v1_score` **gone** — reconstructable only by re-scoring v1's artefact | ✓ `model_v2_score` |
| **Model decisions** | ❌ `model_v1_decision` **gone** — same | ✓ `model_v2_decision` |
| **Observed outcome label** | ❌ `model_v1_observed_outcome` **gone** *(recovery being attempted — do not assume)* | ✓ `model_v2_observed_outcome` |
| **Pre-ML targets** | ✅ **`pre_ml_label` available** — survives as v1's training target | ✗ N/A |
| **Ground truth / oracle** | ✗ None — scrapped cars never garage-verified; handler vs engineer indistinguishable in records | ✗ None |

> ✅ **Corrected 2026-08-05 — the first column was previously inverted.** It recorded v1's scores/decisions/outcomes as available and `pre_ml_label` as disposed; the reverse is true. Note that the **oracle row is unchanged**: it was never a data-retention matter. The oracle is absent because scrapped cars are physically destroyed without assessment, which no surviving dataset can undo.

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

> **✅ RETRACTED 2026-08-05 — this constraint was stated backwards.** The block previously asserted that v1's training data and the `pre_ml_label` dataset had been permanently disposed of under retention obligations, and that v1 could therefore only be studied through its production log. **Both halves are wrong, and in opposite directions.** See § "Data availability" for the corrected statement. What actually holds:
>
> - **v1's training data survives, `pre_ml_label` included.** It *can* be re-scored, audited and re-analysed. Designs may assume access to it and to a pre-ML holdout.
> - **v1's production log is what is gone** — scored inputs *and* observed outcomes (`model_v1_score`, `model_v1_decision`, `model_v1_observed_outcome`). The old instruction to "study v1 through the artefacts it left behind, never through its inputs" is now **exactly inverted**: the inputs survive and the artefacts do not. *(Outcome recovery is being attempted but must not be assumed.)*
> - **The class prior α is still only bounded** — but for a different reason than stated above. Not because the data is missing: because for the 6.45% fast-tracked slice the *oracle* was destroyed by the scrapping itself, and the surviving `pre_ml_label` records the forced 1 rather than the true status. **Data availability is not oracle availability.** The bound's input figures do, however, become directly measurable.
> - **v2b remains synthetic-only, with its justification inverted.** It mixes `pre_ml_label` with the v1 log; the pre-ML half is now available and the **v1-log half is not**, so it still cannot be reproduced on real data.
>
> **The design constraint survives in weakened, relocated form.** Methods must now be designed to work without **v1's production log**, not without its training data. Anything requiring genuine observed v1 production behaviour — the mobility overlap band, the error-inheritance detector — must either reconstruct v1's scores by re-scoring the surviving model artefact on surviving feature rows (and label them *reconstructed*, not *observed*), or rely on garage-verified rows and bounding arguments as before.
>
> The chain still cannot be anchored to uncontaminated ground truth — but the reason is the destroyed *oracle*, not a destroyed *dataset*.

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

> **⚠️ The row-year ranges above are the SYNTHETIC generator's timeline, not the real one.** The real v2 window is **`2018-01-02 → 2020-09-30` (in-period pool → 2019-12-02), with the model trained in 2021** (§ "Model v2 — Actual Split, Scoring & Decision Rule"). Do not quote "2022–2024" as v2's real training window.
>
> **✅ The claim in this section is now confirmed and is in fact stronger than stated.** Because v2's window opens in 2018 — entirely after v1's training window (`lossdate < 2017-04-01`) and after v1's deployment — essentially **all** of v2's training data was generated while v1 was already making scrap decisions. "Entirely contaminated by the SFP loop" is not an approximation here; it is close to literally true.

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

> **✅ CORRECTED 2026-08-08 (read off the training code):** the preprocessing and the model are
> **two separate pickles**, in every version — `fttl_pipeline.pkl` + `fasstacker_xgb.pkl` (v1),
> `pipeline.pkl` + `model.pkl` (v2), `p146_pipeline.pkl` + `p146_model.pkl` (v3), all under the
> repo's `./outputs/`. The 2026-07-01 "one combined Pipeline pickle" description below is wrong on
> that point. Everything else in this section stands: preprocessing still lives only inside a
> fitted pickle (not re-implementable), unpickling still needs that version's own env, and the
> estimator's `predict_proba` takes the **already-preprocessed** matrix. See "Training flow &
> artefact storage by version" below for the full per-version flows.

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


# Training Flow & Artefact Storage by Version (real, read off the training code 2026-08-08)

What each version's training run actually does, start to finish, and **where every artefact
lands**. Three storage tiers appear, and they have different reachability:

| tier | what lives there | reachable from |
|---|---|---|
| repo `./outputs/` | the two fitted pickles (pipeline + model), per version | any clone, inside that version's env |
| `Z:` network drive | raw extracts, transformed matrices, predictions (pandas pickles) | company laptop only |
| company database | v3's raw extract (`datascience_lab.prod.p146_extract_v3`) | company laptop only, re-queryable |

A `Z:` pandas pickle only opens under the pandas that wrote it — so v1's data pickles are readable
**only inside env-v1** (pandas 0.x / Python 3.5), and must be re-exported to parquet there before
the analysis env can touch them.

## v1 — file-based, everything on `Z:`, four splits appended into single files

```
Z:/P10_…/inputs.pkl                                    # raw extract (pandas pickle)
  → cc_fttl flag (ClaimCentreIndicators): handler-flagged FTTL rows (~2.54%) DROPPED from training
  → target = veh_total_loss   (veh_fast_track=1 forces veh_total_loss=1 — see "v1 target construction")
  → merge car_table on abicode_ext = abicode           # vehicle enrichment join
  → incident_time = hour(lossdate)                     # engineered feature
  → SplitData(data, …) on lossdate:
        train+test  2016-01-01 ≤ x < 2017-04-01, split 80/20
        val1        2017-04-01 ≤ x < 2017-06-30
        val2        2017-06-30 ≤ x ≤ 2018-02-09  (extract end; user-confirmed 2026-08-08)
  → features = param.MODEL_FEATURES + NOTROADWORTHY + DAMAGE_FEATURES + ADMIN_FEATURES
  → claims_pipe.fit(train)          → ./outputs/fttl_pipeline.pkl        # fitted on TRAIN only
  → transform all four splits       → Z:/…/inputs_transformed.pkl        # ONE appended file
  → trainXGB(…)                     → ./outputs/fasstacker_xgb.pkl       # eval_set=[(X_test,y_test)]
  → predict_proba on all splits     → Z:/…/predictions.pkl               # [claimnumber, predictions]
  → per-split roc_auc printed
```

- **Claim id = `claimnumber`** — carried in the predictions file, so v1's train-time scores ARE
  joinable back to claims.
- The transformed file appends train+test+val1+val2 **in that order** with no split column — split
  membership must be reconstructed from `lossdate` against the boundaries above.
- Spelling of `fasstacker_xgb.pkl` is as transcribed; verify against `dir outputs` (a typo fails
  loudly with FileNotFoundError, so trying it as-is is safe).

## v2 — pre-cleaned `Z:` extract, tubular pipeline, timestamped split files

```
{DATA_FOLDER_PROD}/Data/clean_dataset.pkl              # pre-cleaned extract on Z: (exact folder TBC)
  → split_data(data, 0.2, 2018-01-01)  on ReportedDate # see "Model v2 — Actual Split"
  → raw splits saved:      Z: Data/train_raw_{ts}.pkl / val_raw_{ts}.pkl / test_raw_{ts}.pkl
  → fit_and_save_input_checker(train)  → ./outputs/input_checker.pkl
  → build_fit_save_pipeline(train)     → ./outputs/pipeline.pkl          # tubular transformers
  → transform all splits               → Z: Data/train_transf_{ts}.pkl / val_transf_ / test_transf_
  → train_model_save(train[MODEL_FEATURES], train[TARGET])
                                       → ./outputs/model.pkl             # XGBClassifier, joblib
```

- Target = `par.TARGET` = **`veh_total_loss`** (v1's `∨ veh_fast_track` term gone — see "v2 target").
- The split files carry a **training-time timestamp in the filename** — a fresh run writes new
  files rather than overwriting, so the timestamp identifies *which* training run's data survives.
  The timestamp of the production run must be read off the `Z:` folder before `paths.features`
  can be declared.
- `input_checker.pkl` is a fitted input-schema validator — an artefact v1 and v3 do not have.
- Data window 2018-01-02 → 2020-09-30 (train/val pool → 2019-12-02); see "Model v2 — Actual Split".

## v3 — database extract, in-repo enrichment joins, nothing but the two pickles saved

```
DatabaseConnector: SELECT * FROM datascience_lab.prod.p146_extract_v3    # polars, NO raw file
  → add_target(): vehicle status ∈ {fttl, total_loss, unrecovered} → Fttl = 1
  → subset_by_date(...)                                # maturation cut (immature_date)
  → enrichments from ./fttl/dependencies/:             # hpi + thatcham parquets
        join on project_params.ENRICHMENT_KEY, how=left
  → cc_rule flags from ./fttl/dependencies/cc_rule_lookup.csv
        (FTTL YEARS / FTTL YEARS airbag deployed / MAKE / SHORT MODEL;
         intermediate _cc_rule_* columns computed, then dropped along with the lookup features)
  → split_into_train_test_oot_by_date on ReportedDate_CLAIM, OOT ≥ 2025-12-01   # saves NOTHING
  → stateless_pipeline(each split)                     # code, not a pickle — rerun from the repo
  → build_pipeline(train, stateful_pipeline)           → ./outputs/p146_pipeline.pkl
  → model fields = [Fttl, project_params.KEY, ReportedDate_CLAIM, *WORKING_MODEL_FEATURES]
  → build_model(train, WORKING_MODEL_FEATURES, target=Fttl, HYPERPARAMETERS)
                                       → ./outputs/p146_model.pkl
  → get_predictions(): fttl_predicted_prob, fttl_predicted_labels (> threshold)   # in memory only
  → recall / precision / roc_auc computed per split
```

- **The training script persists nothing — but the data survives anyway.** ✅ CORRECTED
  2026-08-10: v3's raw extract and transformed data **do exist on the `Z:` drive** (saved outside
  the transcribed script), and the transformed files carry the **train-time predictions as
  columns** (`fttl_predicted_prob`, thresholded `fttl_predicted_label`) — so v3's own scores are
  recoverable without re-scoring, like v1's `predictions.pkl`. Only the production log genuinely
  does not exist (never deployed); a predicted label is not a decision — no v3 car was actioned.
  So v3 exports like v1/v2 (`01_export_v3.ipynb` reads the Z: files), and the full regeneration
  chain — DB export → enrichment joins → cc-rule → stateless → `p146_pipeline.pkl.transform` —
  is a **fallback**, needed only if the Z: files prove stale.
- **The as-of-now enrichment caveat applies to the regeneration route only**: a DB re-extract
  today joins *current* hpi/thatcham rows (see "Enrichment Table — Update Cycle"). The Z: files
  are what training actually saw, so exports from them carry no such caveat.
- Claim id = whatever string `project_params.KEY` holds — to be read off the repo.
- Data window 2023-06 → 2026-05 on `ReportedDate_CLAIM`; OOT block ≥ 2025-12-01.

## What differs across the three flows — the SFP-relevant summary

| | v1 | v2 | v3 |
|---|---|---|---|
| raw source | `Z:` pickle | `Z:` pickle (pre-cleaned) | database query |
| dataframe library | pandas | pandas | **polars** |
| target column | `veh_total_loss` (∨ fast_track) | `veh_total_loss` | `Fttl` (same label renamed) |
| split variable | `lossdate` (accident) | `ReportedDate` | `ReportedDate_CLAIM` |
| enrichment | car_table on `abicode_ext` | (inside the cleaned extract) | hpi + thatcham + cc-rule lookup, in-repo |
| pipeline style | bespoke `claims_pipe` | `tubular` transformers | stateless (code) + stateful (pickle) |
| transformed data saved? | yes — one appended file | yes — three timestamped files | not by the script — but **present on Z:** (✅ 2026-08-10) |
| train-time scores saved? | yes — `predictions.pkl` (`claimnumber`) | no | yes — **columns in the transformed data** (`fttl_predicted_prob`/`_label`, ✅ 2026-08-10) |
| production log today | **destroyed** | exists (live) | **never existed** (not deployed) |

Every row of that table is a *pipeline-divergence* axis in the sense of "Preprocessing and
Training-Window Divergence Across Versions" — none of it is SFP, all of it can move scores, and
all of it must be held out of any cross-version claim.

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