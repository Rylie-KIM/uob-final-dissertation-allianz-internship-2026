# Problem Definition — Machine Learning Self-Fulfilling Prophecy Loop

> This document is the **central reference** consulted when reading each paper. When writing a paper note (`literatures/notes/pXX.md`), answer the following questions:
> 1. Which component of §2 (Problem Formalisation) does this paper address?
> 2. Can this paper's methodology actually be applied under the constraints in §3, or do its preconditions break down?
> 3. How does this paper classify our problem within §2.4 (Problem Type Taxonomy) — and is that classification correct?

---

## 1. Business Logic Spec (source: `README.md`, transcribed verbatim 2026-06-16)

### 1.1 Service Overview — Fast Track Total Loss Model

Internal service name: **Fast Track Total Loss**. Without this model, every damaged vehicle is sent to a garage where an engineer determines whether it can be repaired. This process is costly — the insurer pays for the garage assessment time and must provide the customer with a replacement vehicle during that period.

The model's purpose is to **bypass the garage process entirely for obvious total-loss cases** — vehicles so severely damaged that write-off is certain. By fast-tracking these cases directly to salvage, Insurance Company. reduces garage costs and delivers a faster settlement to the customer.

### 1.1a Pre-ML Baseline — the human era, before any model (⭐ key reference figures)

Before any ML model existed, scrapping (write-off as total loss) was decided by a mix of **predefined rules** (e.g. *car flips over → automatic total loss*) and **handler judgement**. Two figures anchor the pre-ML human-based SFP loop:

| Pre-ML figure | Value | Relevance to §2 |
|---|---|---|
| **% of all cars scrapped** | **15%** | Human-era baseline scrap rate — the reference against which real ML-era inflation (≈ 19% → 21.5% post-deployment) is judged. Scrap-rate inflation above 15% is the headline (but confounded, §2.5 #2/#6/#10) symptom. (**Real Allianz pre-model figures**; the synthetic DGP's analogue is ~18.4% → 18.6%, §1.5 table.) |
| **% of scrapped cars fast-tracked for TL** (handler-identified, **no garage visit**) | **43%** | ~43% of pre-ML write-offs are **forced, garage-unverified labels** — the pre-ML human-based SFP loop (§2.2, §1.5 "Model v1") **quantified**. These land in `pre_ml_label` as contaminated positives; the remaining ~57% of scrapped cars reached a garage and carry an engineer (ground-truth) outcome. |

This is the concrete size of the contamination `pre_ml_label` carried *before* the model-based loop began — the human loop v1 was trained on and the model loop then amplified (§2.3). Source: Allianz team figures; to be reconciled with the real logs when available. Mirrors `README.md` §"Pre-ML Baseline".

**Implied class prior α = P(y=1) — sharp bound [8.55%, 15%], point estimate ≈ 15%.** Decomposing all pre-ML cars: **93.55% are garage-observed ground truth** (85% confirmed non-TL + 8.55% confirmed TL) and **only 6.45% are fast-tracked** (oracle destroyed, unverifiable). α is partially identified by that 6.45% slice alone: 8.55% (all forced labels actually repairable) → 15% (all genuine TL). The bound is **sharp**, not merely conservative — the 85% non-scrapped region is fully garage-observed so its missed-TL error is structurally **0**; all uncertainty is confined to the 6.45% fast-track slice. Point estimate ≈ upper bound because fast-tracking is rule-based on obvious write-offs (e.g. flipped car). **Calibration check:** the pre-ML scrap rate (15%) ≈ α → the human era was well-calibrated *before* contamination; the SFP fingerprint is the post-deployment scrap rate inflating *above* α (≈ 19% → 21.5%, cf. §1.5). This is the *marginal* class prior α, distinct from scrap precision π_scrap = P(y=1 | scrap) (see `literatures/notes/p28.md`).

> **✅ Update 2026-08-05 — the bound's inputs are now measurable, the bound itself stands.** v1's training data survives **including `pre_ml_label`** (§1.4b "Data availability"), so 15% / 43% / 6.45% can be **computed from real rows** rather than taken as team-provided figures; the endpoints may shift. **⚠️ α remains bounded, not identified** — the surviving label records the *forced* 1 for fast-tracked cars, and their oracle was destroyed by the scrapping itself. Data availability is not oracle availability. 🔎 Recompute the three figures and restate the interval.

### 1.2 Expected Benefits (2)

1. Reduced process time, cost, and effort for handling total-loss vehicles
2. Accurate prediction of total-loss status is critical — if classified as total loss, the insurer must pay the full vehicle value

### 1.3 Cost Structure — Why Precision is the Binding Constraint

| Decision | Outcome | Cost Implication |
|----------|---------|-----------------|
| `predict = total loss` → scrap (correct) | Genuine total loss bypasses garage | Saves garage assessment fee + replacement vehicle cost |
| `predict = total loss` → scrap (**wrong**) | Repairable car is scrapped | Insurer pays full vehicle value instead of repair cost — **large financial loss** |
| `predict = repairable` → garage | Garage confirms actual outcome | Garage + replacement costs incurred, but no catastrophic loss |

**False Positives (misclassifying a repairable car as total loss)** are asymmetrically far more expensive. This makes **precision ≥ 0.985 a business-critical constraint**. The model does not need to catch every total loss — a missed total loss (False Negative) simply proceeds normally through the garage, so recall is monitored but is not an optimisation target.

### 1.4 Model Training Methodology

- **Target maturation time:** ~1–2 months. Because it takes time to confirm whether a vehicle is genuinely unrepairable, the most recent 2 months of data are excluded from training (labels may be unresolved).
- **OOT (Out-of-time) holdout:** Approximately the most recent 6 months (after the exclusion buffer) are held out as a time-separated validation set to test generalisation to future data.
- **Train/Test split:** 80/20 random split on the remaining data after removing the maturation buffer and OOT period.
- **Evaluation metric:** Primary training metric is **precision**, with a target threshold of **≥ 0.985**. Recall is computed alongside but is not an optimisation target — the model is tuned to minimise False Positives, not to maximise total-loss recall.
- **No calibration:** XGBoost outputs probability-like scores but these are not calibrated. Since the model is used purely for ranking/triage (not expected-value-based decisions), well-calibrated probabilities are deemed unnecessary and the calibration step is omitted. **Note:** If scores are used as propensity weights (e.g. for IPS correction), the lack of calibration may distort debiasing — this is flagged as a known limitation in the SFP mitigation analysis.
- **Decision threshold:** The scrap policy applies an **absolute score cutoff** — a vehicle is fast-tracked to salvage only when `model_score ≥ τ_v`, where `τ_v` is the cutoff tuned on the validation set to satisfy precision ≥ 0.985 **for that model version**. This is **not** a percentile/top-N rule. Because the cutoff is fixed in score space, the *scrap rate* moves freely with the score distribution — this is precisely the mechanism by which score drift in later model versions becomes observable as an increased scrap rate (= the key SFP signal).

  > **Threshold is per-version, not a universal constant.** What is invariant across versions is the **business constraint (precision ≥ 0.985)** and the fact that the cutoff lives in **score space (absolute, never a percentile)**. The cutoff *value* `τ_v` differs by version because each model's score distribution differs, so a different absolute threshold is needed to hold the same precision target. **`0.872` is the documented real-world value for v2** — and only until **2026-06-30 14:30 UK time**, when v2 moved to **0.825** (§1.4b); v1 uses a segmented pair (0.75/0.85, §1.4a) and v3 is unconfirmed. The synthetic generator now **tunes `τ_v` per version** at deployment — the lowest cutoff holding precision ≥ 0.985 on a held-out validation slice, scored against that version's (SFP-contaminated) training label — with `0.872` retained only as the fallback when the target is unreachable (`generate/model.py::_tune_threshold`). Synthetic tuned values land near the real anchor (v1 ≈ 0.852, v2a ≈ 0.906, distribution-dependent). Wherever this document writes the threshold as `0.872`, read it as the v2 instance of `τ_v`. See `README.md` for the canonical statement.

  > **⚠️ The policy *form* is NOT invariant across versions** (confirmed 2026-07-29). v1 applies **two** cutoffs segmented by vehicle mobility (§1.4a); v2 applies **one** global cutoff (§1.4b). Only the *absolute-cutoff-in-score-space* property is shared; the cutoff's **arity** differs by version. Wherever this document assumes a single scalar `τ_v` per version, that holds for v2 but **not** for v1.

  > **Operational note — threshold change history (v2)** *(superseded 2026-07-29)*: the earlier account was that v2's threshold was briefly changed away from 0.872, that performance degraded, and that it was **promptly reverted** — licensing the working assumption that v2's threshold is constant at 0.872 throughout production.
  >
  > **✅ Update (2026-07-29, read from v2 `score.py`):** the threshold is **currently `0.825`**, changed from `0.872` and **still in force** — not reverted. ✅ **Exact break confirmed 2026-07-31: 2026-06-30 14:30 UK local (BST) = 13:30 UTC** (supersedes the earlier "≈ 2026-07-01" estimate). **v2's threshold is piecewise-constant with one documented break, not constant.** Full specification and consequences in §1.4b. ⚠️ An **earlier** change is also confirmed (`0.825 → 0.872` at 2026-02-25 16:26 UK), so there are at least two — but how far back the 0.825 era before it ran **cannot be recovered**; the record is broken there. Carried as a limitation in §1.4b-lim, not resolved.
  >
  > **✅ The 2026-06-25 researcher concern is closed for v3.** The worry was that a threshold change might have overlapped a retraining window, injecting decisions made under a different cutoff into the training labels. **v3's training logs are confirmed to come entirely from the τ = 0.872 regime** — v3 trains on a policy-homogeneous log. The concern remains live for analysis on the **current** v2 production log (which now spans both regimes) and for any **future** retrain.

```
Full data timeline
──────────────────────────────────────────────────────────────────────►
│        Training + Test (80/20 split)        │   OOT (6m)  │ excl. │
│                                             │             │  (2m) │
```

The OOT holdout is temporally separated — it comes *after* the training data and is not a random sample. This reflects the actual deployment condition where the model is applied to future, unseen claims.

#### Data Split Roles

| Split | When Used | Purpose |
|---|---|---|
| Train | During training | Learn model parameters |
| Validation | During training | Hyperparameter tuning, early stopping |
| Test | After training | Report final performance metrics |
| OOT | After training | Validate model robustness on future data |

### 1.4a Model v1 — Actual Data Split & Target Specification (real, confirmed 2026-07-28)

§1.4 above is the **generic/idealised** scheme (maturation buffer + single 6-month OOT). This subsection is the **actual v1 split as implemented at Insurance Company.**, transcribed from the real training code — authoritative *for v1* where the two diverge. Three real-world differences from §1.4: v1 uses **explicit `lossdate` cut-off dates** (not a rolling "last N months" rule); the **Test set is an in-period random split, not an OOT holdout**; and there are **two** date-bounded validation sets, not one OOT block. Mirrors `README.md` § "Model v1 — Actual Data Split & Target Specification".

#### Source extract & cleaning

```sql
SELECT * FROM uc005.cc_fttl_train_v8 WHERE lossdate >= '2016-01-01'
```

- **Original dataset shape:** `(311591, 198)` (311,591 claims × 198 raw columns).
- **No corrupted-row drop applies to v1** — the four splits sum to the **full 311,591** (178,435 + 44,609 + 36,049 + 52,498), so no row is removed before splitting. *(The `ReportedDate = 2020-08-24` corrupted-record exclusion is a **v2** cleaning detail, not v1 — see §1.4b.)*
- **Model-ready column count:** **39 columns** (selected/preprocessed from the 198 raw columns). ⚠️ This **39 includes the `target` column and the claim-number (ID) column**, which are *not* predictive inputs — so the actual model **input feature count is 37** (39 − target − claim number). Read "39" as the width of the model-ready table, not the number of features the model learns from.

#### Training-set exclusion — centre-flagged FTTL claims (`cc_fttl`)

**A key characteristic of v1's training data:** claims flagged by the claims centre as FTTL (`cc_fttl`) are **removed from the training set**; v1 trains on the remaining claims only.

**`cc_fttl` is a rule-based flag — not discretionary handler judgment.** Computed at FNOL from four "obvious total loss" physical indicators, OR'd together:

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

- **What it captures:** the **"predefined rules" component of the pre-ML era** (§1.1a — e.g. *"car flips over → automatic total loss"*) made concrete. Any one of {recovery-agent total-loss call, extrication, rollover, water into cabin} → `cc_fttl = 1`. Unambiguous, physically-evident write-offs identified at intake — *not* a handler's subjective severity call. (Syntax note: each OR-clause must be parenthesised — pandas `|` binds tighter than `==`.)
- **Return value:** `np.where(cond, 1, 0)` returns a **numpy `ndarray`** of 0/1 (length = #rows), **not** a Series and **not** a slice of `df`; no index, aligns to `df` positionally.
- **Why removed (rationale):** a `cc_fttl` flag is a **rule-forced label with no garage verification**. Training on it would teach the model to *reproduce the intake rule* rather than discriminate — the human-based SFP loop v1 is meant to beat (§1.5, "outperform the call handler"). Excluding them stops the model from simply re-learning the obvious-total-loss rules.
- **⚠️ Two-edged consequence (selective-labels bias).** These rule-flagged claims are exactly the **obvious total losses**. Removing them trains v1 on a **truncated distribution** (clear-cut write-offs excluded), yet in production it must still score those cases → a **train/serve distribution mismatch**, a concrete instance of the selective-labels problem (P27, §2.4): training distribution ≠ application distribution. The exclusion *reduces rule-forced contamination in training* but *introduces a training-set selection bias* — not unambiguously "cleaner".

**Impact on dataset size — applied to ALL splits, not just training.** Although the code comment says "removing those from the *training* set", the row counts show the exclusion hits **every** split (Test and both validation sets shrink too) — `cc_fttl` is filtered from the whole modelling population:

| Split | Before `cc_fttl` exclusion | After | Removed | % removed |
|---|---:|---:|---:|---:|
| Train | 178,435 | 173,758 | 4,677 | 2.62% |
| Test | 44,609 | 43,471 | 1,138 | 2.55% |
| Validation set 1 | 36,049 | 35,254 | 795 | 2.21% |
| Validation set 2 | 52,498 | 51,189 | 1,309 | 2.49% |
| **Total** | **311,591** | **303,672** | **7,919** | **2.54%** |

Two consequences: **(i)** because **Test and Validation are also filtered**, the precision ≥ 0.985 target is measured on a population with the obvious total losses **already removed** — the reported metric excludes the easiest positives, so it is *not* a precision over all claims the model faces in production. **(ii)** the removal rate is concrete: **~2.54% of all rows** (7,919 of 311,591).

> **Open items to reconcile (confirm against real data before treating as settled):**
> 1. **How `cc_fttl` relates to the ~43% forced-label contamination** (§1.1a Pre-ML Baseline — a figure **provided by the Allianz team**, defined as the % of *scrapped* cars handler-fast-tracked with no garage visit ≈ 6.45% of *all* pre-ML cars; to be reconciled with real logs). **The new size evidence points to subset, not equality:** `cc_fttl` removes only **~2.54% of all rows**, well below the ~6.45% forced-label rate — consistent with the rule-based flag (rollover/extrication/water/recovery-agent) catching only the *physically obvious* fraction of the broader handler-forced population, leaving softer subjective-judgment forced labels in the data. (Caveat: 6.45% is an early-era estimate and 2.54% is the real v1-window count — **directional evidence for `cc_fttl ⊆ forced-labels`, not a like-for-like proof**.) So v1's training set is **partly** de-contaminated, not fully: the "trained on contaminated `pre_ml_label`" narrative (§1.5, §2.3) still stands but must be qualified with "minus the rule-obvious total losses".
> 2. **Relationship to `veh_fast_track`** (target source, §1.4a): same signal, subset, or independent? The target rule `fast_track=1 & total_loss=0 → target=1` applies to whatever rows *remain* after the `cc_fttl` exclusion — confirm the interaction.
> 3. **Scope:** v1-only, or also applied to v2/v3 training?

#### Date boundaries (all on `lossdate`)

`EndTrainDate = 2017-04-01`, `EndValidationDate = 2017-06-30`; **dataset max `lossdate` = 2018-02-09** ✅ (confirmed 2026-08-04 — full extract period 2016-01-01 → 2018-02-09).

#### The four splits

| Split | Definition (on `lossdate`) | Size | How produced | OOT? |
|---|---|---:|---|---|
| **Train** | `lossdate < 2017-04-01`, then 80% of that pool | **178,435** | `train_test_split(..., test_size=0.2, random_state=0)` | No |
| **Test** | remaining 20% of the same `lossdate < 2017-04-01` pool | **44,609** | same `train_test_split` call | **No — in-period random split** |
| **Validation set 1** | `2017-04-01 ≤ lossdate < 2017-06-30` | **36,049** | date filter | Yes (near-term, ~3 months) |
| **Validation set 2** | `2017-06-30 ≤ lossdate ≤ 2018-02-09` (dataset end) | **52,498** | date filter | Yes (longer horizon, ~7.5 months) |

> The **Size** column is the raw `lossdate`-split count **before** the `cc_fttl` exclusion above. Post-exclusion: **Train 173,758 · Test 43,471 · Val1 35,254 · Val2 51,189** (311,591 → 303,672). See "Training-set exclusion — centre-flagged FTTL claims" above.

**Test vs Val1 vs Val2.** Test is carved from the *training period* by a random split → shares the train window → **not** out-of-time (in-distribution only). Val1/Val2 are **temporal holdouts** defined by `lossdate` cut-offs (data later than training) — these play the generalisation-testing role that §1.4's generic "OOT" block describes. Val1 = near-term future (~3 months post cut-off); Val2 = longer horizon (2017-06-30 → 2018-02-09, the dataset end).

> **⚠️ Test is NOT stratified.** The call is `train_test_split(train_test, train_test['target'], test_size=0.2, random_state=0)` — **no `stratify=` argument**. Passing the label as the second positional array does **not** trigger stratification (scikit-learn stratifies only when `stratify=` is set explicitly); this is a plain random split. With total-loss as the minority class, Train and Test can carry **slightly different class balances** — a real caveat under precision ≥ 0.985, where the Test positive count drives the precision estimate's stability. `random_state=0` fixes reproducibility only, not class balance. **This is v1-specific: v2 *does* pass `stratify=` explicitly** (§1.4b), so the caveat does not carry across versions — one more reason v1's and v2's in-period holdouts are not like-for-like.

#### v1 target construction — `veh_fast_track` + `veh_total_loss` → `target`

v1's label is derived from two raw fields, and the derivation encodes the forced-label mechanism directly:

```python
df = data[[target_fttl, target_tl]].copy()          # target_fttl = veh_fast_track, target_tl = veh_total_loss
df[target_fttl] = df[target_fttl].map(...)           # veh_fast_track: non-FTTL → 0, FTTL → 1
df[target_tl]   = df[target_tl].map(...)             # veh_total_loss: non-TL → 0, TL → 1
df['target']    = df[target_tl]                      # base target = the total-loss flag
df.loc[(df[target_fttl] == 1) & (df[target_tl] == 0), 'target'] = 1   # fast-tracked but not TL → force to 1
```

| `veh_fast_track` | `veh_total_loss` | `target` | Interpretation |
|:---:|:---:|:---:|---|
| 0 | 0 | **0** | not fast-tracked, not total loss → repairable |
| 0 | 1 | **1** | not fast-tracked, garage-confirmed total loss → genuine positive |
| 1 | 0 | **1** ⚠️ | **fast-tracked but no total-loss label → forced to 1** |
| 1 | 1 | **1** | fast-tracked and total loss → positive |

`target` = "**total loss OR fast-tracked**" by formula. **✅ Confirmed 2026-08-04: the recorded `veh_total_loss` is itself already 1 for fast-tracked vehicles** — the recording process treats fast-tracking as settling the total-loss question with no garage visit (an engineer-confirmed scrap also records 1). The forced positive therefore enters through the **data**, before any formula runs: the truth table's third row is a **defensive safeguard that rarely-to-never fires**, and the effective target is `veh_total_loss` alone. This is **§2.2's label-generation mechanism made concrete in the real v1 label** — fast-track routing itself becomes a recorded positive, i.e. $\tilde{Y}_i^{v1}=1$ whenever $D_i^{v1}=1$ regardless of true $Y_i$.

> ✅ **Resolved 2026-08-04 (supersedes the open item on why `fast_track = 1 & total_loss = 0` occurs).** It does not occur as a stable data state: recorded `total_loss` is 1 whenever the vehicle is fast-tracked; v1's disjunct guards against transient states (e.g. maturation at snapshot). Consequence: **the same forced-label semantics carry into every version's target** (v2: `veh_total_loss`; v3: `Fttl` = `veh_total_loss` renamed) — **P1 holds for all three versions through the data.**

#### v1 model training call — real XGBoost configuration

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
- **`eval_metric="mlogloss"` on a confirmed binary target.** `target` is **binary 0/1** (confirmed), and "**m**logloss" is the *multiclass* log-loss — so this is `mlogloss` applied to a 2-class problem (config choice, not evidence of >2 classes). XGBoost then runs a `multi:softprob`-style objective with `num_class=2`, so **`predict_proba` returns 2 columns** and $S^{v1}$ (`model_v1_score`) is column `[:, 1]` (the total-loss probability).
- **⚠️ `eval_set` is passed to the *constructor*, not to `.fit()`.** In the XGBoost sklearn API `eval_set` is normally a `.fit()` argument; passed to the constructor it may be **stored but never used** — no real eval/early-stopping happens and the "Test as eval_set" monitoring silently no-ops. **Verify whether it takes effect.**
- **`silent=False` is a deprecated parameter** (superseded by `verbosity`) → v1 built on an **older XGBoost release**, consistent with the per-version frozen-environment constraint (§2.5 #7): reloading this `joblib.dump` artefact requires v1's exact pinned XGBoost.
- **`joblib.dump(...)`** produces the serialised v1 artefact the analysis pipeline later reloads — the pickle binding v1 to its exact library stack (the clone-&-run reproduction model, §2.5 #7).
- **`n_jobs=20`** — 20 parallel threads at fit time (infra detail, no modelling effect).

**v1 runtime environment (confirmed via `conda_dependencies_local.yml`): Python 3.5.2, pandas 0.22.0** (numpy pin not stated in the yml — 🔎 confirm). Two consequences:
- Pins v1 to a **very old stack** (2016–2017 era), consistent with the deprecated `silent=` XGBoost arg above and the per-version environment-isolation constraint (§2.5 #7).
- **Even v1's *data* pickle is env-bound, not just its model pickle.** A DataFrame serialised by pandas 0.22.0 will very likely **not** unpickle under a modern pandas (2.x) — the internal BlockManager format changed across that gap — so v1's `inputs.pkl` (`…/Prod-Predictions/`) must be opened **inside v1's env**. The real-data onboarding util `src/scoring/inspect_pickle.py` is written **Python-3.5.2-compatible on purpose** for exactly this (used to recover v1's unknown last date — see §1.4d).

#### v1 scrapping threshold — segmented by vehicle mobility (creates a score-space overlap band)

**⚠️ v1's scrap decision is NOT a single absolute cutoff** — it is **segmented by vehicle mobility**, two thresholds:

```
D = 1[ score > 0.75 ]   if the vehicle is IMMOBILE
D = 1[ score > 0.85 ]   if the vehicle is MOBILE
```

**MOBILE** = vehicle mobility status ∈ {`Mobile`, `Mobile Not Roadworthy`, `Mobile Not Secure`}; **IMMOBILE** = the complement. Immobile (undrivable → typically more damaged) is scrapped more readily (lower bar 0.75); mobile (drivable → benefit of the doubt) needs a higher score (0.85).

> **`τ = 0.872` is not v1's rule** (0.872 is v2's documented single cutoff). v1's decision-rule *form* is a **mobility-conditional (segmented) cutoff**, not one absolute threshold — so the claim in §1.4 that "the policy *form* (single absolute cutoff) is invariant across versions" must be **qualified**: at least v1 uses two cutoffs keyed on mobility. Whether v2/v3 also segment (or collapse to 0.872) is **unconfirmed — to check**. This directly bears on **§2.5 P4 / §2.6 (positivity)** and the mitigation/evaluation "zero garage rows above τ" assumption.

**Does this create positivity in the v1 FTTL system?** *Partially — and importantly.* It does not restore strict positivity, but it opens a genuine **overlap band in score space** the single-cutoff model said could not exist.

**Three score regions now, not two:**

| Score range | Immobile | Mobile | Oracle (garage outcome) observed? |
|---|---|---|---|
| `score < 0.75` | garage | garage | ✓ **both** garage-verified |
| `0.75 < score < 0.85` | **scrap** | **garage** | ⚠️ **mixed** — mobile cars ARE garage-verified up here |
| `score > 0.85` | scrap | scrap | ✗ neither — positivity dead |

- **Conditional on the *full* feature vector (score + mobility): positivity still FAILS.** Given score *and* mobility the decision is a deterministic step function; `e(x) = Pr(D=1 | score, mobility) ∈ {0,1}`. The strict P4 violation is **not** dissolved.
- **Conditional on the *score alone*: positivity HOLDS in `(0.75, 0.85]`.** There `e(score) = Pr(D=1|score) = Pr(immobile|score) ∈ (0,1)` — at the same score a mobile car is garaged and an immobile car is scrapped. Mobility breaks the score↔treatment collinearity in that region.
- **Data consequence:** there are now **garage-verified outcomes above 0.75** (mobile cars in the band). "**Zero garage rows above τ**" is only true **above 0.85**; in 0.75–0.85 the oracle is partially observable via mobile vehicles.

**Why this helps the framework (with one caveat):**
- **Two-cutoff sharp RDD** — each mobility group has its own discontinuity (immobile @0.75, mobile @0.85) → the payout/outcome RDD (§2.6 RDD) runs at **two** boundaries, more identification.
- **Primary Layer-1 detector (added 2026-08-02): the error-inheritance test.** Mobile band rows garage-verified as *repairable* are rows where a high v1 score is **known** to have been wrong; if v2 trained on v1's forced labels it should **over-score exactly those rows**. Estimated as a local-linear jump in E[s_v2 | s_v1] at 0.75 on the mobile verified Y=0 population, with the Y=1 population as a second difference and placebo cutoffs as falsification. Needs no common feature space (scores joined by claim id only). Spec: `paper.mid.draft.md` §3.3.1; implementation: `notebook/real/01_error_inheritance.ipynb`. ψ (p29 residual density) is demoted to a secondary/corroborating signal.
- **Overlap band anchors the mitigation counterfactual** — the counterfactual-outcome model (§ mitigation) now has *real observed outcomes for high-scoring vehicles* (mobile, 0.75–0.85) to learn from, not blind extrapolation above a single cutoff.
- **⚠️ Confounded overlap, not random.** Mobility is **not** independent of the true outcome: `Mobile` ⇒ drivable ⇒ less damaged ⇒ genuinely more repairable. Mobile garage-verified cars in the band are a **systematically-more-repairable subgroup**; using them directly to impute the immobile-scrapped counterfactual would **under-state** immobile cars' true total-loss probability. Exploitable overlap, but identification still needs a **mobility adjustment** — not clean random positivity.

**Bottom line:** the segmented threshold moves the system from "positivity globally dead above one cutoff" to "positivity dead only above 0.85, with a confounded-but-usable overlap band in 0.75–0.85" — a materially better starting point for RDD and the mitigation counterfactual, provided the mobility confound is handled explicitly.

> **Open items:** (1) confirm the exact `IMMOBILE` category set (complement of the three mobile statuses). (2) ~~Confirm whether v2/v3 segment by mobility or use the single 0.872.~~ **✅ Resolved for v2 and v3 (2026-07-29): both are single global cutoffs, no segmentation (§1.4b, §1.4c). v1 alone is segmented.** (3) This **qualifies P4 (§2.5/§2.6) and the "zero garage rows above τ" assumption** (§2.6 IPS discussion, mitigation/eval design) — propagate to the dissertation (§2.4 P4, §3.6) and the IPS-positivity caveat.

### 1.4b Model v2 — Actual Split, Scoring & Decision Rule (real, confirmed 2026-07-29)

Read directly from v2's production **`train.py`** (split scheme, cleaning, data window) and **`score.py`** (decision rule). Still 🔎 TBD for v2: target construction, training call / XGBoost config, runtime environment, feature set, and whether v1's `cc_fttl` exclusion applies. The v1 spec is §1.4a. Mirrors `README.md` § "Model v2 — Actual Split, Scoring & Decision Rule".

#### Split scheme & data window (from v2 `train.py`)

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

##### ⚠️ The split names collide across versions — the biggest cross-version trap

With all three versions confirmed, **v2 is the odd one out**:

| Actual role | v1 calls it | v2 calls it | v3 calls it |
|---|---|---|---|
| 80% of the in-period pool | **Train** | **Train** | **Train** |
| 20% random holdout, **same period** as train (in-distribution) | **Test** | **Validation** | **Test** |
| **Out-of-time** holdout, later than train | **Validation set 1 / 2** | **Test** | **OOT** |

**"Test" means an in-period random split in v1 and v3, but the out-of-time block in v2.** Comparing "v2 test precision" with v1's or v3's therefore sets an **out-of-time** number against an **in-distribution** one: v2's is measured under a strictly harder condition and will look worse for reasons unrelated to the model. **Any cross-version performance table must key on the *role*, not the name.** Two further points: **v2's naming is the misleading one** — v1 and v3 agree with each other, so checking only those two builds the wrong expectation about v2; and **v3's convention is clearest**, reserving "OOT" for the out-of-time block (§1.4c).

##### The splitting variable changed: `lossdate` (v1) → `ReportedDate` (v2)

v1 splits on **`lossdate`** (accident date); v2 on **`ReportedDate`** (notification date). These differ by the **reporting lag**.

- **v2's choice is arguably more honest operationally:** only *reported* claims are knowable at training time, so ordering by `ReportedDate` reflects the information actually available — a `lossdate` ordering can place a late-reported claim in the training window before the insurer knew of it.
- **But the versions' boundaries are not comparable.** A v2 `ReportedDate` cut is not the same population slice as a v1 `lossdate` cut; the windows cannot be lined up across versions or plotted on a common axis without converting.
- **Reporting lag is not constant** — plausibly varying by channel (FNOL vs ENOL, §1.2) and severity — so sorting by `ReportedDate` **reshuffles** the loss-date ordering, and the OOT block can contain *older losses reported late*. 🔎 Quantify the lag distribution and its drift before treating v2's OOT as cleanly "later" in loss terms.

##### Data window — and why it matters for the SFP narrative

The cutoff filter is strict (`>`), so everything on or before 2018-01-01 is removed: **v2's window is 2018-01-02 onward.**

> ✅ **Realised windows confirmed 2026-08-04:** in-period pool (Train + Validation) **2018-01-02 → 2019-12-02**; OOT block (v2's "Test") **2019-12-02 → 2020-09-30**. v2's data ends **2020-09-30** — the window does *not* extend to the 2021 training date, and the training pool closes **before COVID-19**.

> **⚠️ This window lies entirely *after* v1's.** v1 trained on `lossdate < 2017-04-01` (§1.4a). So essentially **all** of v2's training data was generated while **v1 was already in production**, i.e. v2's labels are v1-influenced across the board. The SFP hand-off is therefore **not a partial contamination of v2's training set — it is close to total.** This is a code-level confirmation of the "v2a trained on `model_v1_observed_outcome`" narrative (§1.5, §2.3), and it is **stronger** than that narrative previously claimed. 🔎 Confirm v1's actual deployment date to state "all" rather than "essentially all".

##### ✅ The corrupted-record exclusion — confirmed v2, and it drops a whole DAY

`dataset[dataset['ReportedDate'] != '2020-08-24']`.

- ✅ **Closes the open item** carried since §1.4a: the exclusion belongs to **v2**, not v1.
- ⚠️ **It is not "a corrupted record".** The filter is an equality test on `ReportedDate`, so it removes **every claim reported on 2020-08-24** — potentially a full day, not one row. Both this document and `README.md` previously described it as a single record. 🔎 **Count the rows dropped.** A whole-day hole in the middle of the series is non-trivial — and with the realised windows confirmed (2026-08-04), **2020-08-24 falls inside the OOT block** (2019-12-02 → 2020-09-30), i.e. inside v2's headline temporal holdout.

##### ⚠️ A silent data-loss region between the train pool and the OOT block

OOT is taken as `tail(20%)` and **then** filtered to `ReportedDate > 2019-12-01`. Rows in the tail 20% but reported on or before 2019-12-01 are **not** in `head(80%)` (→ excluded from Train and Validation) **and** are filtered out of the tail (→ excluded from Test). **They are dropped from the run entirely, unrecorded.** Consequences:

- **The nominal 80/20 is not the realised split.** The OOT/Test block is **smaller than 20%** of the post-cutoff data by an unrecorded amount. Report *realised* sizes; never quote nominal fractions.
- ✅ **Resolved in effect (2026-08-04).** The realised OOT block starts **2019-12-02**, immediately after the hard-coded filter date (2019-12-01), so the 80th-percentile `ReportedDate` sits just above the filter: it was **non-binding (or marginal)** and the silent-loss region is empty or near-empty *for this run*. The reproducibility caveat below still applies to any re-run on refreshed data.
- **Reproducibility:** a hard-coded date applied to a *proportional* tail. Re-running on extended data moves the tail later, changing whether the filter binds — **the split is not stable across re-runs**, so a refresh will not reproduce the original partition.

*(Minor: `⌊0.8n⌋ + ⌊0.2n⌋ ≤ n`, so head and tail never overlap — no leakage between the train/validation pool and the OOT block; they may leave one row assigned to neither.)*

##### ✅ v2's train/validation split IS stratified on the target — unlike v1

`train_test_split(..., stratify=data['veh_total_loss'])`. This **closes the open item**, in v2's favour: the concern flagged for v1 (§1.4a) does not carry over.

- **v1 vs v2:** v1 passes the label positionally with **no `stratify=`** → plain random split, Train/Test class balances can drift apart. v2 passes `stratify=` explicitly → Train and Validation carry **matched positive rates by construction**, materially stabilising v2's validation precision estimate under precision ≥ 0.985 with a minority class.
- **⚠️ Stratification covers the in-period pool only.** The OOT block (v2's "Test") is the temporal tail; its positive rate is whatever the later period contained, and nothing balances it against Train.

  > **Quietly useful:** because the in-period split *is* stratified, the Train/Validation positive rate is fixed by construction, so **any positive-rate gap between Validation and the OOT Test block is real temporal drift, not split noise** — a clean, essentially free diagnostic. Under the SFP hypothesis the direction is predicted: as v1 scrapped more, forced positives accumulate, so the positive rate should be **rising** into the later window. 🔎 Compute the positive rate in Train / Validation / Test and compare.

- **A sharpening, not a defect:** stratifying on the target stratifies on a **contaminated** label, so the in-period holdout mirrors the training distribution *including its forced-positive share* and cannot serve as a check on the contamination. This holds for any random in-period split — stratification only makes the mirroring exact. Stratifying is the right call; it is noted so the validation set is not mistaken for independent evidence about the label problem.
- 🔎 Still unrecorded: the split's `random_state`.

#### v2 target — `veh_total_loss` alone (v1's `∨ fast_track` term is gone)

v2 trains on the raw **`veh_total_loss`** flag, with no derivation on top. Compared with v1 (§1.4a):

| | v1 | v2 |
|---|---|---|
| Target | `veh_total_loss ∨ veh_fast_track` | **`veh_total_loss`** |
| Forced label in the **formula**? | ✅ yes — the `fast_track` disjunct (a safeguard) | ❌ no |
| Forced label in the **data**? | ✅ **yes** — recorded `total_loss` = 1 on fast-track | ✅ **yes — same recording semantics** |

**The formula difference is presentational, not substantive.** ✅ **Resolved 2026-08-04:** the recorded `veh_total_loss` is **set to 1 for fast-tracked vehicles** (no garage visit) under the same recording logic in both eras — v1's disjunct was a safeguard, not the contamination channel.

##### ✅ Resolved (2026-08-04): `veh_total_loss` = 1 for a scrapped car — P1 holds for v2 through the data

The question previously carried here — what value `veh_total_loss` holds for a vehicle fast-tracked to scrap — resolves to the former reading **(a): recorded as 1**; scrapping settles the total-loss question in the data. Consequences:

- **P1 holds for the currently-live model (v2)**, and for v3 (`Fttl` = `veh_total_loss` renamed — §1.4c). It lives in the **data-recording process**, not in any label formula; the SFP mechanism is unchanged, only its location.
- The alternative (b) — scrapped cars recorded 0 — is **excluded**, as is the null-and-dropped selection reading.
- The cross-tab (`veh_total_loss` × fast-track flag on v2's surviving data) is no longer a blocking open item; it is kept only as a cheap **regression test** of this confirmation.

#### v2 model training call — real XGBoost configuration (**xgboost 1.4.2**, model trained **2021**)

```python
xgb.XGBClassifier(
    objective="binary:logistic",   # clean binary — unlike v1's mlogloss-on-binary
    eval_metric="auc",             # ⚠️ not precision
    colsample_bytree=0.6,
    eta=0.147686...,               # ⚠️ alias of learning_rate — BOTH set, values differ
    gamma=15.0,                    # very high min split-loss (default 0)
    grow_policy="depthwise",       # the default
    learning_rate=0.0887667...,    # ⚠️ conflicts with eta
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

##### ⚠️ 1. `eta` and `learning_rate` are both set and disagree — effective learning rate ambiguous

They are **the same hyperparameter** (`eta` is the native name), passed with **different values**: `0.147686` vs `0.0887667`, a **1.66×** gap. One silently wins.

- In the sklearn wrapper `learning_rate` is an explicit argument while `eta` arrives via `**kwargs`, and kwargs merge *over* explicit params — so **`eta = 0.147686` most likely wins**. Precedence is **XGBoost-version-dependent**, so this must be **verified, not assumed**. (v2 pins **xgboost 1.4.2**; later 2.x/3.x releases may warn or error on duplicate aliases instead — consistent with v3, on 3.2.0, setting `learning_rate` alone.)
- Not cosmetic: with `n_estimators = 450` the learning-rate × rounds budget is **66.5 vs 39.9**.
- 🔎 **Definitive check:** `model.get_booster().save_config()` on the saved artefact returns the `eta` actually used. Do this before quoting any v2 hyperparameter. **This is a reproducibility defect in production code**, worth reporting regardless of which value won.

##### ⚠️ 2. `eval_metric="auc"` is misaligned with the constraint — and computed on contaminated labels

- **AUC does not measure the deployment constraint.** The rule is **precision ≥ 0.985 at a high cutoff** — a property of the extreme top tail. AUC is rank-based and **threshold-free**, weighting the whole score range roughly equally, so it gives **almost no assurance about precision at τ**. (v1 used `mlogloss`; neither version trains on the metric the business enforces.)
- **The AUC is measured against SFP-contaminated labels.** v2 trains on v1's log, where every v1-scrapped vehicle carries a forced positive (§2.2, P1). The metric therefore rewards v2 for **ranking highly exactly those cases v1 chose to scrap**; since v1 and v2 share much of the feature space, a high AUC partly certifies **agreement with the previous model's decisions**. This is the contaminated-metric trap (P6) operating **inside the training objective**, not only in the reported precision.

> The long non-round decimals (`eta`, `learning_rate`, `reg_lambda`) are characteristic of an **automated hyperparameter search**; the round values (`gamma=15.0`, `reg_alpha=20.0`, `scale_pos_weight=4.5`, `max_delta_step=10`) look fixed or grid-chosen. If a search was run it was **optimising AUC**, so the misalignment propagates into the whole hyperparameter selection. 🔎 Confirm.

##### ⚠️ 3. `reg_alpha = 20.0` vs v1's default `0` — this CONFOUNDS the SHAP concentration analysis

**The finding with the largest consequence for the dissertation's central contribution.**

Regularisation is strikingly asymmetric: **L1 `reg_alpha = 20.0`** (default 0 — very strong) against **L2 `reg_lambda = 0.0123626`** (default 1.0 — ~**80× below** default, effectively off). Strong L1 drives **sparsity in leaf weights**, mechanically **concentrating the model onto fewer features**; `gamma = 15.0` (default 0) pushes the same way. v1's call sets **none of these** — v1 runs at `reg_alpha = 0`, `reg_lambda = 1.0`, `gamma = 0` (§1.4a).

> **⚠️ A v1 → v2 rise in feature-importance concentration is therefore NOT, on its own, evidence of an SFP loop.** The central statistic — SHAP concentration (Simpson / Hill / Shannon) rising across generations — has a **competing mechanical explanation**: v2 is far more heavily L1-regularised, and L1 concentrates importance by construction. Both hypotheses predict the same direction.
>
> **To address, in rough order of strength:** **(i)** run the comparison on **v2a → v3a** rather than v1 → v2a, *provided* v3 shares v2's hyperparameters (🔎 obtain v3's config — now high priority); **(ii)** re-train a v1-configured model on v2's data and vice versa, separating the regularisation effect from the data effect; **(iii)** report concentration under matched hyperparameters; **(iv)** at minimum state the confound as an explicit limitation and bound its size. Option (i) is **already** the specified design for an independent reason (parallel-trends violation at the era boundary), so the two arguments reinforce each other.

##### ⚠️ 4. `scale_pos_weight = 4.5` — upweights the contaminated class, guarantees miscalibration

- **It amplifies the SFP loop directly.** The positive class *includes the forced positives*; weighting positives 4.5× gives the manufactured labels 4.5× the influence on the fitted model — a concrete amplification channel from v1's decisions into v2's parameters.
- **Implied balance:** under the conventional `n_neg / n_pos` rule this implies a positive rate near **18%**. 🔎 Confirm whether computed or tuned.
- **⚠️ The score is NOT a probability — `0.872` must never be read as "87.2% chance of total loss."** `scale_pos_weight` inflates predicted odds by roughly the weight factor, and no calibration step is applied (§1.4). Naively dividing predicted odds by 4.5:

  | Score | Implied true probability *(illustrative only)* |
  |---|---|
  | 0.872 (pre-2026-07 threshold) | ≈ **0.60** |
  | 0.825 (current threshold) | ≈ **0.51** |

  *Order-of-magnitude illustrations, not calibrated estimates* — the correction assumes the model is otherwise calibrated, which an uncalibrated tree ensemble is not. But the direction is unambiguous: **the current threshold may sit close to a coin-flip in true-probability terms.** 🔎 A calibration curve on garage-verified rows (where the outcome *is* observed) is a priority — it bears directly on the false-positive cost the ≥ 0.985 floor exists to control.
- **Consequence for IPS (§2.6):** the caveat that "uncalibrated scores distort propensity weighting" now has a **concrete, quantified cause**. Any use of v2 scores as propensities must account for the `scale_pos_weight` inflation.

##### 5. Other parameters

- **`objective="binary:logistic"`** — clean binary; **resolves for v2 the ambiguity flagged for v1** (multiclass `mlogloss` on a binary target). `predict_proba` returns 2 columns; $S^{v2}$ is column `[:, 1]`.
- **`gamma=15.0` vs `min_child_weight=1`** — opposite directions: aggressive pruning alongside a permissive default allowing tiny leaves. With `max_depth=10`, trees may be deep but every split must clear a high bar — consistent with an automated search rather than hand design.
- **`subsample=1.0`** — no row subsampling; randomisation comes only from `colsample_bytree=0.6`. With 450 deep trees, overfitting pressure is held back mainly by `gamma`/`reg_alpha`.
- **`max_delta_step=10`** — conventionally used for logistic objectives under imbalance; coherent alongside `scale_pos_weight`, though permissive enough at 10 to rarely bind.
- **`random_state=42`** (v1 used 10⁹ for the model, 0 for the split). 🔎 v2's `train_test_split` seed still unconfirmed.
- **`n_jobs=-1`** — infra only.

##### ⚠️ 6. A five-year-old model: trained 2021, deployed ~2022, still live in 2026

- **v2 has scored live claims for ~4–5 years without retraining**, and its **training pool closed on 2019-12-02** (data window 2018-01-02 → 2020-09-30, tail = OOT only — ✅ 2026-08-04): 2026 claims scored from **2018–2019 patterns**, six-plus years of drift (vehicle values, parts costs, repair economics, channel mix) unaddressed, the SFP loop compounding throughout. This sharpens why v3's failure matters: the only attempted refresh did not ship.
- **⚠️ COVID-19 sits in the OOT block, not the training pool** *(corrects the earlier "training window spans COVID" reading)*. The in-period pool ends 2019-12-02 — entirely **pre-COVID** — while the OOT block (2019-12-02 → 2020-09-30) spans the first UK lockdown. Consequences: **(a)** v2 never saw COVID-era or post-COVID claims in training at all; **(b)** v2's headline OOT figures were measured partly on an **anomalous period**, so its reported out-of-time performance is not a clean read of normal-times generalisation. *(The earlier proportional-tail volume worry is superseded — the realised windows show the date filter was non-binding.)*
- ✅ Any earlier placeholder describing v2's window as "≈ 2022–2024" is **wrong**, and now exactly resolved: the data window ends **2020-09-30**; training happened in 2021.

#### Data availability — ✅ CORRECTED 2026-08-05: all three training sets survive; v1's *log* is what is gone

> **⚠️ Supersedes the previous account** (v1's training dataset destroyed under the data-retention period). That was wrong, and the correction **inverts** v1: *training data* present, *production log* absent. Mirrors `README.md` § "Data availability".

| Version | Training data | Production log | Status |
|---|---|---|---|
| **v1** | ✅ **exists — including the `pre_ml_label` target** | ❌ **gone — scored inputs *and* observed outcomes** (outcome recovery attempted, not assumed) | deployed, superseded |
| **v2** | ✅ exists (`Z:` drive) | ✅ exists | **currently live** |
| **v3** | ✅ exists (training + OOT) | ❌ **cannot exist — never deployed** | never deployed |

- **v1 is now re-derivable, not merely documented.** §1.4a's split scheme, the `cc_fttl` exclusion and the `veh_fast_track` × `veh_total_loss` target construction can be **audited against real source rows**. The forced-positive rate *within* v1's training labels — previously "not recoverable" — is **directly measurable**, which settles open item 1 of §1.4a (`cc_fttl` ⊆ forced-labels) empirically rather than directionally.
- **`pre_ml_label` survives.** ⚠️ This does **not** point-identify α, and the §1.1a bound [8.55%, 15%] stands: for the 6.45% fast-tracked slice the recorded label is the *forced* 1, and the oracle was destroyed by the scrapping itself. **Data availability is not oracle availability.** What changes is that the bound's **inputs** (15% / 43% / 6.45%) become computable from real rows instead of team-provided.
- **v2 remains re-derivable end-to-end** and, its log also surviving, stays the anchor for anything needing observed production behaviour.
- **⚠️ The v1 → v2 hand-off loses its observational side.** The old fallback — "infer it from v1's production log" — is **gone with that log**. v1's production scores/decisions must now be **reconstructed** by re-scoring v1's surviving artefact on surviving feature rows, making them *reconstructed*, not *observed*. This bears directly on §1.4a's mobility overlap band and on the error-inheritance detector, both of which assume v1-side production behaviour. 🔎 Feasibility depends on feature-space compatibility — see `features/`.
  - **✅ Partial mitigation found 2026-08-08:** v1's **train-time scored file survives** on the `Z:` drive — `predictions.pkl`, columns `[claimnumber, predictions]`, covering all four splits (2016-01-01 → 2018-02-09 on `lossdate`). These are scores the *actual* fitted v1 produced at training time, so for those rows no re-scoring is needed — only the production **decisions** and post-2019 scores remain reconstruction targets. Caveat: train-split rows are in-sample; treat them accordingly. See `README.md` § "Training Flow & Artefact Storage by Version".
- **⚠️ The `…/Prod-Predictions/inputs.pkl` reference in §1.4a is stale** — that artefact is part of the missing log. `src/scoring/inspect_pickle.py` remains useful for v1's *training* pickle, under the same pandas-0.22.0 constraint.
- **v2b is still not reproducible on real data — justification inverted.** It mixes `pre_ml_label` (now ✅ available) with the v1 log (now ❌ gone). Conclusion unchanged, reason reversed.

#### The decision rule — a single global cutoff (no mobility segmentation)

```python
predictions["FastTrackerDecision"] = (
    predictions["FasterTrackerProbability"] > threshold
).astype(int)      # 1 → fast-track to scrap, 0 → garage
```

1. **v2 uses ONE global threshold** — no mobility segmentation, no per-segment cutoff, no percentile rule: a single scalar compared against a single score column. This **closes the open item** left by §1.4a ("whether v2/v3 also segment by mobility is unconfirmed").
2. **The policy *form* is therefore genuinely NOT invariant.** v1 = **two** mobility-keyed cutoffs (0.75/0.85); v2 = **one** global cutoff. The claim in §1.4 that the form is invariant is **false as stated** — the invariant is narrower: *an absolute cutoff in score space, never a percentile*, with the cutoff's **arity** differing by version.
3. **The comparison is strict `>`, not `≥`.** This document and the synthetic generator write `D = 1[s ≥ τ]`; production is `D = 1[s > τ]`. Measure-zero for continuous scores, but reproduction code should use `>`. 🔎 Confirm the exact column spellings (`FastTrackerDecision` vs `FasterTrackerProbability` — the "Fast"/"Faster" mismatch is transcribed as read and appears to be a real inconsistency in the production code).

#### Threshold history — v2 is piecewise-constant, not constant

| Period | Threshold | Status |
|---|---|---|
| Deployment → 2026-06-30 14:30 | **0.872** | superseded |
| 2026-06-30 14:30 → present | **0.825** | **currently in force** |

- ✅ **Break instant confirmed (2026-07-31): 2026-06-30, 14:30 UK local time** (BST, UTC+1) = **2026-06-30 13:30 UTC**. Supersedes the earlier "≈ four weeks before 2026-07-29 / ≈ 2026-07-01" estimate.
- **It is an instant, not a date — 2026-06-30 straddles both regimes.** Two consequences for date-cut analysis: (i) if the log's date column is **date-only**, every 2026-06-30 row parses to 00:00 and is assigned pre-change even where the decision was actually made at 0.825 → **drop 2026-06-30** unless a timestamp exists; (ii) if the timestamps are **UTC**, the boundary is **13:30**, not 14:30, and a one-hour band of rows flips regime. 🔎 Confirm the log column's timezone and time granularity (new open item 1b).
- Recorded once in `src/config.py::DECISION_RULES["v2"]` (`break_at_local` / `break_at_utc` / `break_tz`); `src/threshold.py::apply()` converts tz-aware date columns to `Europe/London` before comparing. No other file may hard-code the date.
- The threshold was **lowered**: a lower bar scraps **more** claims, so the post-change regime mechanically produces **more forced positives** per unit volume.
- Supersedes the "briefly changed, promptly reverted, treat as constant" note in §1.4. 🔎 Unconfirmed whether this is that same episode or a **second** change (→ three regimes).

**✅ v3's training data is single-regime.** All v3 training logs were generated under `threshold = 0.872`; the 0.825 regime lies entirely after v3's training window. This **closes the 2026-06-25 researcher concern for v3**. It stays open for analysis on the current v2 log (which spans both regimes) and for any future retrain. **Practical rule:** any per-row analysis on the real v2 log must either restrict to the pre-change regime (matching v3's training data) or carry the regime as an explicit covariate — silently pooling conflates two different treatment assignments.

> ⚠️ **Conditional on the change record being complete — it is not.** See §1.4b-lim below. The claim is **retained, not withdrawn**; the exposure is carried as a limitation.

#### 1.4b-lim Limitation — the threshold change record is incomplete before 2026-02-25

**A known gap in the source record, stated rather than fixed.** A *second, earlier* change is confirmed:

| Instant (UK local) | Change | UTC |
|---|---|---|
| **2026-02-25 16:26** | `0.825 → 0.872` | 16:26 UTC (February = GMT, UTC+0) |
| **2026-06-30 14:30** | `0.872 → 0.825` | 13:30 UTC (June = BST, UTC+1) |

So 0.825 was already in force for some period *before* 2026-02-25, but **how long is unrecoverable — the deployment/change record does not reach back past that point.** (Note the two instants sit in different UTC offsets; one fixed offset applied to both is wrong.)

**Not encoded, on purpose.** The February change is recorded in `config.DECISION_RULES["v2"]["known_unmodelled_break"]` but is **not** a `regimes` entry. Encoding it needs a start date for the prior era; inventing one would silently relabel years of v2 log rows on a guess — worse than a documented gap.

**What it exposes.** v3's window (2023-06-01 → 2026-05-01) and its OOT slice (2025-12-01 → 2026-05-01) both **contain** 2026-02-25. If the prior 0.825 era reached into that window: (a) v3's training labels span two policies, not one; (b) v3's recall-collapse result — load-bearing in the SFP narrative (§2.3) — was measured on a holdout straddling a policy break; (c) the 2026-06-25 researcher concern is **live** for v3, not closed.

**Why the claim still stands.** The exposure is conditional on a fact the record cannot supply, and retracting a documented finding on an unmeasurable possibility swaps one unsupported claim for another. What *is* certain is the **direction**: an unrecorded regime can only add heterogeneity, never remove it, so every result resting on v3's single-regime status is an **upper bound** on policy homogeneity.

**Empirical substitute (does not need the record).** Assignment is a hard cutoff, so `min{score | scrapped}` in any period estimates the τ then in force. Running `read_off()` month-by-month across v3's window would make an unrecorded regime **visible as a step** in that series. 🔎 Cheap — run it first.

Carried in the dissertation as `report/paper/paper.mid.draft.md` §4.6.1 and README § "Limitation — the threshold change record is incomplete before 2026-02-25".

#### ~~The *lowering* as SFP evidence~~ — reading RETIRED (2026-08-02)

**Superseded by user confirmation (2026-08-02): the 0.872 → 0.825 change was a deliberate manual
re-tuning by the team.** The earlier candidate reading — that a contaminated precision metric showed
false headroom and licensed the lowering (P6 closing into an operational loop) — has been **removed
from the dissertation** (`paper.mid.draft.md` §2.3.2) and is retired here. The break is treated
purely as an **exogenous operational policy action**: usable as a natural experiment (temporal
overlap band, cutoff-shift RDD), carrying **no SFP inference from the direction of the change**.
It also removed an internal tension: P8 predicts contamination surfaces as the threshold being
forced *up* (v3's 0.984), so treating a *lowering* as loop evidence alongside would have made the
theory unfalsifiable.

#### The threshold change creates a temporal overlap band — (0.825, 0.872]

This **parallels the v1 mobility band (§1.4a) exactly, but in time rather than cross-section.** A claim scoring in (0.825, 0.872] was:

| Regime | Decision for `s ∈ (0.825, 0.872]` | Garage outcome observed? |
|---|---|---|
| **Pre-change** (τ = 0.872) | **garage** | ✓ **yes — verified outcome exists** |
| **Post-change** (τ = 0.825) | **scrap** | ✗ no — forced positive |

- **Garage-verified outcomes exist for scores up to 0.872** (pre-change era). "**Zero garage rows above τ**" is, for v2, true only above **0.872** — not above the *current* 0.825.
- **Positivity holds in the band when pooling across the break:** conditional on the score alone, `e(s) = Pr(D=1|s) = Pr(post-change | s) ∈ (0,1)` for `s ∈ (0.825, 0.872]`. Within *either* regime alone it is still a deterministic step function, so strict P4 is not dissolved.
- **A cutoff-shift design becomes available** — the same claims treated differently across an exogenous policy break: DiD / RDD-with-moving-cutoff, with **two** discontinuity locations (0.872 pre, 0.825 post).
- **It anchors the mitigation counterfactual** with real observed outcomes for high-scoring vehicles (pre-change, 0.825–0.872).

**⚠️ Two caveats — this band is weaker than v1's:**

- **Confounded by time, not by an observed covariate.** v1's band is confounded by *mobility*, which is measured and adjustable. v2's is confounded by **calendar time** and everything moving with it — score drift, case-mix, FNOL/ENOL channel mix, seasonality, portfolio composition. There is no clean adjustment analogue; identification needs a parallel-trends-style assumption across the break — the assumption this project has already found violated at era boundaries.
- **The post-change window is ~4.3 weeks** (2026-06-30 14:30 → 2026-07-31). Given the documented thinness of the scrapped partition, the band may be **underpowered before any modelling begins**. 🔎 Count rows in (0.825, 0.872] on each side of the break *first* and gate the analysis on that count. A date-only column forces dropping 2026-06-30 — the single most valuable day, sitting exactly on the break.

> **Open items for v2.** *Threshold:* ~~(1) exact change date~~ ✅ **closed 2026-07-31 — 2026-06-30 14:30 UK local (13:30 UTC)**; (1b) **timezone + time granularity of the log's date column** — decides whether the break is cuttable intra-day and whether the boundary is 14:30 or 13:30 (new); (2) ~~one change or two~~ ✅ **at least two — 2026-02-25 16:26 confirmed**; what remains is (2b) **when the first 0.825 era began — record broken, see §1.4b-lim** and (2c) **monthly `read_off()` across v3's window** as the empirical substitute; (3) stated rationale; (4) exact column spellings; (5) row counts in (0.825, 0.872] each side of the break, **excluding 2026-06-30 if the column is date-only**.
> *Split:* (6) ~~the 80th-percentile `ReportedDate`~~ ✅ **resolved in effect 2026-08-04** — realised OOT starts 2019-12-02, filter non-binding; (7) realised split sizes (never nominal 80/20; windows now ✅, sizes still 🔎); (8) rows dropped by the 2020-08-24 filter (now known to sit inside the OOT block); (9) the train/validation split's `random_state` (`stratify=` now ✅ confirmed); (9b) **positive rate in Train / Validation / OOT Test** — a cheap drift check the stratification makes interpretable (note the OOT block spans COVID); (10) reporting-lag distribution (`ReportedDate − lossdate`) and its drift.
> *Training config (⚠️ highest priority first):* (11) **which of `eta` / `learning_rate` actually took effect** — read the saved booster's config; (12) **v3's hyperparameters**, needed to know whether the SHAP-concentration comparison can run on a regularisation-matched pair; (13) whether `scale_pos_weight=4.5` was computed or tuned; (14) whether a hyperparameter search was run and what it optimised; (15) a **calibration curve on garage-verified rows**; (16) `train_test_split`'s `random_state`.
> *Highest value of all:* (17) ~~cross-tab `veh_total_loss` × fast-track flag~~ ✅ **resolved 2026-08-04** — recorded `veh_total_loss` = 1 for fast-tracked vehicles, so **P1 applies to the live model through the data** (§1.4b); the cross-tab is retained only as a regression test.
> *Still no v2 counterpart:* (18) extract SQL and raw shape; (19) whether §1.4a's `cc_fttl` exclusion applies to v2; (20) runtime environment / library pins; (21) feature set. *(Target and XGBoost config now ✅ confirmed.)*

### 1.4c Model v3 — Data Window (real, confirmed 2026-07-29)

**Scope: the data window only.** First confirmed real detail for v3; its split scheme, target construction, training configuration, runtime environment and feature set are all 🔎 TBD — do not assume they follow v2 (§1.4b). Mirrors `README.md` § "Model v3 — Data Window".

v3 selects its dataset with a `subset_by_date` function on **`ReportedDate_CLAIM`**:

```python
old_date      = DATA_START_DATE   # 2023-06-01
immature_date = DATA_END_DATE     # 2026-05-01

df.filter((df['ReportedDate_CLAIM'] > old_date) & (df['ReportedDate_CLAIM'] < immature_date))
```

Both bounds strict → a window of **2 years 11 months**: `2023-06-01 < ReportedDate_CLAIM < 2026-05-01`.

#### Split scheme — a date-bounded OOT block plus a shuffled in-period split

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

**This is the cleanest split of the three versions.** The OOT boundary is an explicit date (not a proportional tail), the 5-month holdout closely matches §1.4's "~6 months OOT", and `head(0.2n)` / `tail(-0.2n)` are **exactly complementary** — no overlap, no gap, none of v2's silent data-loss problem (§1.4b). v3 also names the block **"OOT"** rather than overloading "Test" or "Validation" — the clearest of the three conventions.

*(Implementation: `sample(fraction=…, shuffle=…, seed=…)`, `filter`, `tail(-k)` are **Polars**, not pandas — v3 is a modern rewrite; `tail(-k)` drops the first k rows. Third seed convention: v1 `random_state=0`, v2 `42` (model), v3 `seed=123`.)*

##### Is shuffling the in-period pool a good choice?

**Yes — a reasonable and standard design.** The temporal question is handled by the separate, date-bounded OOT block, so the in-period split is free to measure in-distribution fit. v1 and v2 also use in-period random splits.

One clarification on wording: shuffling does not *remove* time bias — it makes the in-period test set match train in time, so that test set simply **cannot show drift**. Drift is measured by the OOT block instead. The in-period number is an in-distribution figure; generalisation claims should rest on OOT.

Two smaller notes:

- **No stratification.** `sample(fraction=1, shuffle=True, seed=123)` + `head`/`tail` has no `stratify=`. v2 stratified on the target (§1.4b); v3 does not. With a minority positive class and a precision ≥ 0.985 target, the 20% test set's positive count affects how stable the precision estimate is. 🔎 Confirm stratification isn't applied elsewhere in v3's pipeline.
- 🔎 **If a claim can contain more than one vehicle**, a random split can put two vehicles from the same claim on opposite sides, which would flatter the in-period test score. A quick check of the vehicles-per-claim distribution settles it; if multi-vehicle claims are common, group the split by claim. The OOT block is unaffected.

> **One question worth settling: which split produced v3's recall collapse?**
>
> "Recall collapsed when precision was held at ≥ 0.985" is load-bearing (§1.5, §2.3), and it means different things depending on where it was measured:
>
> - **On the OOT block:** some of the collapse could be ordinary drift over those 5 months rather than SFP contamination — so the finding is confounded.
> - **On the in-period Test:** drift is ruled out by construction, so the collapse is harder to explain away — a stronger result for the SFP argument.
>
> 🔎 Worth confirming which one it was.

#### v3 decision rule & threshold — single cutoff, far stricter than v2

**Target column: `Fttl`.** The threshold is tuned to a precision target; two calibrations are recorded:

| Precision target | v3 threshold | v2 for comparison |
|---|---|---|
| **0.985** (business floor) | **0.984** | 0.872 |
| 0.97 | **0.970** | — |

**1. ✅ v3's decision-rule form is a single absolute cutoff** — same form as v2, *not* v1's mobility-segmented pair (§1.4a). This closes the last open item on decision-rule form: **v1 alone is segmented.**

**2. ✅ It explains the recall collapse quantitatively.** Reaching precision 0.985 requires a cutoff of **0.984**, pressed against the top of v3's score range — almost nothing clears it, which is what "recall collapsed when precision was held at ≥ 0.985" means. Relaxing the target to 0.97 only buys a cutoff of 0.970, so the recall recovered is modest: scores are **densely packed at the top** and precision is extremely sensitive to the cutoff there. The two-point pair (0.985→0.984, 0.97→0.970) is itself the evidence for that density.

**3. ⚠️ Raw thresholds are NOT comparable across versions.** "0.984 vs 0.872" does not alone mean v3 is stricter — the score distributions and `scale_pos_weight` values differ. Applying the crude odds correction of §1.4b (divide predicted odds by `scale_pos_weight`):

| | threshold | `scale_pos_weight` | implied true probability *(illustrative)* |
|---|---|---|---|
| v2 @ precision 0.985 | 0.872 | 4.5 | ≈ **0.60** |
| v2 current | 0.825 | 4.5 | ≈ **0.51** |
| **v3 @ precision 0.985** | 0.984 | 5.552301 | ≈ **0.92** |
| v3 @ precision 0.97 | 0.970 | 5.552301 | ≈ **0.85** |

On this rough like-for-like footing v3 is genuinely far more conservative — demanding roughly a 0.92 chance of total loss where v2 demands about 0.60. **That is the shape of the collapse:** v3 could hold the precision floor only by scrapping almost nothing. Illustrations, not calibrated estimates.

##### ✅ `Fttl` resolved (2026-08-04): it is `veh_total_loss` under a new name

**User-confirmed from the real data: `Fttl` carries exactly the same values as `veh_total_loss`** — a rename, not a new construction (consistent with `ReportedDate` → `ReportedDate_CLAIM` in the same rewrite). The earlier decision-vs-outcome ambiguity dissolves in a specific way: the column is nominally the **outcome as recorded**, but — per the 2026-08-04 confirmation in §1.4a/§1.4b — the recorded value is **set to 1 for fast-tracked vehicles without garage verification**, so for fast-tracked rows its recorded value *is* the decision. **P1 holds for v3's target through the data, exactly as for v1 and v2.**

**Cross-version status of the target definition:**

| Version | Target | Status |
|---|---|---|
| **v1** | `target` = `veh_total_loss OR veh_fast_track` (disjunct = safeguard) | ✅ **Full derivation confirmed (§1.4a); forced 1s already in the recorded column** |
| **v2** | **`veh_total_loss`** — the raw flag alone | ✅ **Confirmed — carries the forced 1s in the data (§1.4b)** |
| **v3** | `Fttl` **= `veh_total_loss` renamed** | ✅ **Confirmed 2026-08-04 — identical values; carries the forced 1s** |

#### v3 model training call — real XGBoost configuration (**xgboost 3.2.0**)

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

*(Decimals reproduced exactly as supplied; trailing `...` marks values truncated at source, not rounded here.)*

##### ⚠️ v3 does NOT share v2's regularisation — the confound is not escaped

§1.4b flagged v2's `reg_alpha = 20.0` as a **competing mechanical explanation** for any rise in SHAP concentration, and made "obtain v3's hyperparameters" a **blocking prerequisite** — because the SHAP–DiD design falls back to the **v2a → v3a** pair, which only works if that pair is regularisation-matched.

**It is not matched. The regularisation is effectively inverted:**

| | v2 | v3 |
|---|---|---|
| `reg_alpha` (L1) | **20.0** — very strong | **not set ⇒ 0 (default)** |
| `reg_lambda` (L2) | **0.0123626** — ~80× below default | **1.182373785** — near default |
| `gamma` | **15.0** — heavy pruning | **0.0004847861** — effectively none |

Capacity controls differ as much: v3 uses **shallow, tightly-constrained trees** (`max_depth=3`, `max_leaves=18`, `min_child_weight=44`, 802 rounds) where v2 used **deep trees under heavy penalties** (`max_depth=10`, `min_child_weight=1`, 450 rounds). The two control complexity by **entirely different mechanisms** — v2 via penalties, v3 via tree structure.

> **Consequence: the v2a → v3a pair is confounded too.** Every capacity and regularisation knob differs, so a concentration difference between v2 and v3 **cannot be attributed to the data (hence to SFP) without controlling for configuration.** The "use v2a → v3a instead" escape from the v1 → v2a parallel-trends problem is **not available as-is** — both candidate pairs are now confounded, each for a different reason.
>
> The options in §1.4b become **required** rather than optional: **(i)** matched-hyperparameter re-training — ✅ **now feasible for all three versions** (corrected 2026-08-05: v1's and v3's training data both survive, not only v2's), so configuration can be held fixed while the training *data* varies across every generation; **(ii)** a regularisation/capacity sensitivity sweep bounding how much of the observed ΔC configuration alone can produce; **(iii)** report concentration as *consistent with* the loop rather than identifying it. Whichever route, **every version's configuration must be printed beside its concentration figures.**
>
> **✅ The largest single gain from the 2026-08-05 correction.** This confound was previously irreducible on the v1 → v2 pair — v1's data was believed destroyed, leaving only option (iii). With all three training sets in hand, the same configuration can be run on v1's, v2's and v3's data and the concentration difference read as attributable to the **data** rather than the regularisation. 🔎 Library-version matching (xgboost 1.4.2 vs 3.2.0) is a separate obstacle and is *not* solved by this.
>
> **Update 2026-08-02 — option (i) promoted to the primary design, as a dose–response experiment.** Because the feature spaces also diverge across versions (L0 sets/encodings differ — one-hot granularity moves the Simpson index mechanically, and 1/n normalisation removes only the floor), the cross-version contrast is confounded three ways (configuration, library, feature space). The identifying design is therefore a **controlled dose–response re-training inside v2's surviving data**: one pipeline / feature set / config / library, vary only the forced-positive content of the labels (dose 0 → observed → elevated via lowered cutoffs applied to garage rows), trace C(p) against dose. The observational v2 → v3 contrast is reported as corroboration only. Spec: `paper.mid.draft.md` §3.4.1 (threat (c) + remedy 1).
>
> **Enforced in the tooling (2026-08-01).** That last requirement is no longer a discipline to remember: `src/scoring/attribute.py` captures the estimator's hyperparameters into `detection/<v>_attributions_meta.json` — the only process that can open the pickle — and `notebook/real/00_shap_attribution.ipynb` §4b renders them beside the concentration table, flagging exactly which knobs differ across the versions being compared. It does **not** resolve the confound; options (i)–(iii) still stand. It only makes it impossible to report the concentration numbers without the configuration next to them.

> 🔎 **Confirm `reg_alpha` is genuinely absent from v3's call.** The list omits it, and also `objective` and `eval_metric`, so it may be partial. If v3 in fact sets `reg_alpha` near v2's value the confound narrows sharply. Also confirm v3's `eval_metric` — v2's `auc` is flagged in §1.4b as misaligned with the precision ≥ 0.985 constraint.

##### What v3 fixed, and one thing it did not

- ✅ **The `eta` / `learning_rate` clash is gone** — v3 sets `learning_rate` only (`0.09973689`), so its effective learning rate is unambiguous.
- ✅ **`reg_lambda` is back near default** (1.18 vs 1.0), not v2's effectively-disabled 0.0123626.
- ⚠️ **`scale_pos_weight` rose 4.5 → 5.552301**, so v3's score is *further* from a probability than v2's; §1.4b's warning against reading the cutoff as a probability applies at least as strongly.
- ⚠️ **`max_delta_step` jumped 10 → 61** — at that magnitude the cap is unlikely to bind, i.e. effectively inert. 🔎 Confirm it was deliberate rather than a search artefact.

##### ⚠️ The xgboost version jump also breaks comparability: 1.4.2 → 3.2.0

v2 ran on **xgboost 1.4.2**, v3 on **3.2.0** — two major releases apart. Identical nominal hyperparameters do **not** guarantee identical models across that gap, since defaults and internals changed; in particular the default `tree_method` moved to `hist` in the 2.x line, so v2 and v3 may build trees by **different algorithms** even where parameters agree. 🔎 Confirm the effective `tree_method` per version.

This compounds the confound: any "matched-hyperparameter re-training" remedy must match the **library version** too, or it will not isolate the data effect. It also fits the per-version frozen-environment constraint (§2.5 #7).

#### ✅ This completes the SFP inheritance chain — and it is unbroken

v3's window opens June 2023, **after v2 went live (~2022)**, so v3's training data was generated **entirely under v2's scrapping policy** — the same structure confirmed one generation earlier, where v2's window (2018+) fell entirely after v1's deployment (§1.4b).

| Generation | Training window | Policy in force | Contamination |
|---|---|---|---|
| v2 | `2018-01-02 → 2019-12-02` (pool; OOT → 2020-09-30) | v1 (deployed ~2017) | essentially total |
| v3 | `2023-06-01 → 2026-05-01` | v2 (deployed ~2022) | **total** |

> **Stronger than §1.5/§2.3 have been claiming.** The forced-label hand-off is not "partial contamination accumulating over generations" — at code level **each generation trains exclusively on its predecessor's forced labels**, with no clean-label component at any step. The chain `pre-ML → v1 → v2 → v3` is unbroken, and both hand-offs are confirmed from production window definitions — and, as of 2026-08-04, at the **label level** as well: the recorded `veh_total_loss` / `Fttl` is 1 for fast-tracked vehicles in every era, so the forced value demonstrably sits in each generation's target data.

#### ✅ Independent corroboration: v3's window ends before the threshold break

v3's window closes **2026-05-01**; v2's threshold moved 0.872 → 0.825 at **2026-06-30 14:30 UK time** (§1.4b). The window ends **~2 months before the break**, **independently confirming** — from the data definition rather than the earlier report — that **all v3 training labels come from the `τ = 0.872` regime**. Two independent sources agree, so "v3 is policy-homogeneous" can be stated with confidence.

#### ⚠️ `immature_date` — the maturation buffer made explicit, and a dating puzzle

The end bound is named **`immature_date`**: claims reported after it are excluded because labels have not matured. This is §1.4's generic "~1–2 months excluded" rule as a concrete constant — and an **improvement over v2**, whose window as read shows no comparable upper bound. 🔎 Confirm whether v2 had a maturation exclusion elsewhere; if not, v2's most recent training rows carried immature labels — an independent source of label noise in v2 that v3 corrected.

> **⚠️ Dating discrepancy — load-bearing.** This document records v3 as **"attempted 2025"**, but a 1–2 month maturation buffer against `DATA_END_DATE = 2026-05-01` implies a run around **mid-2026**. Readings: (a) v3 was **re-attempted in 2026**, the 2025 date referring to an earlier attempt; (b) **multiple v3 attempts**; (c) constants updated without a retraining run. This matters because "v3 attempted, recall collapsed at precision ≥ 0.985" is load-bearing in the SFP narrative — **which attempt produced that finding changes what it is evidence about.** 🔎 Confirm the actual v3 run date(s) and which produced the recall collapse.

#### Date columns across versions

| Version | Date column | Meaning |
|---|---|---|
| v1 | `lossdate` | accident date |
| v2 | `ReportedDate` | notification date |
| v3 | `ReportedDate_CLAIM` | ✅ **same field as v2's `ReportedDate`** — renamed only (confirmed 2026-07-29) |

So **v2 and v3 split on the same variable** and their windows are directly comparable. Only **v1 differs**, splitting on `lossdate` (accident date) rather than notification date — the two are separated by the reporting lag, so v1's boundaries are not directly comparable with v2's or v3's.

#### ⚠️ v2's and v3's training windows do NOT overlap — a ~2¾-year gap

v2's data ends **2020-09-30** (training pool already on **2019-12-02** — ✅ 2026-08-04); v3's window opens **2023-06-01**. **The windows are disjoint: ~2 years 8 months from v2's last data to v3's first, ~3.5 years from the close of v2's training pool.** The gap contains both the COVID period and the price surge below.

This bears on the **SHAP–DiD design, specified on the v2a → v3a pair** (chosen because parallel trends visibly fails across the pre-ML → v1 boundary). The gap spans the **2021–2022 UK used-vehicle price surge**, which is not neutral here: a vehicle is a total loss when repair cost exceeds a fraction of market value, so rising market values **mechanically reduce the total-loss rate** at unchanged damage severity, while parts and labour inflation push the other way.

> 🔎 **Check against the actual DiD specification before relying on the v2a → v3a estimate.** Not automatically fatal — the SHAP–DiD compares attribution concentration across contaminated/clean partitions, not a raw outcome trend, and may be computed on a common evaluation set rather than the disjoint training windows. But the two versions learned from economically different eras with a two-year unobserved gap, and the design's justification rests on parallel trends holding for **this** pair. Establish which quantities are compared over which time support, and whether the gap enters. This is the **second** structural threat to that pair, alongside the `reg_alpha` regularisation confound (§1.4b).

#### Window lengths across versions

| Version | Window | Span |
|---|---|---|
| v1 | `2016-01-01 → 2018-02-09` (full extract); train `< 2017-04-01` | ~1 yr 3 mo (training portion) / 2 yr 1 mo (full) |
| v2 | `2018-01-02 → 2020-09-30` (data); in-period pool `→ 2019-12-02`; trained 2021 | **1 yr 11 mo (pool)** / 2 yr 9 mo incl. OOT |
| v3 | `2023-06-01 → 2026-05-01` | 2 yr 11 mo |

v3's window is the longest; v2's training pool is under two years (the earlier "~3 yr 4 mo" figure counted up to the 2021 training date and is superseded); v1's training portion is ~1¼ years. v3's start date also implements the previously-noted intent to **drop the pre-COVID period** — now an explicit constant rather than an approximate description.

> **Open items for v3.** *Highest value first:* (1) **which split produced the recall-collapse figure** — in-period Test or OOT (changes how much weight the finding carries); (2) **v3's hyperparameters**, a blocking prerequisite for the SHAP-concentration design (§1.4b, `reg_alpha` confound); (3) training run date(s), given the 2025-vs-2026 discrepancy above.
> (4) ~~what `Fttl` actually means~~ ✅ **resolved 2026-08-04** — `Fttl` = `veh_total_loss` renamed, identical values, carries the forced 1s (P1 in the data; see above).
> *Remaining:* (5) realised split sizes; (6) vehicles-per-claim distribution; (7) whether stratification appears elsewhere; (8) whether `cc_fttl` applies; (9) cleaning exclusions; (10) runtime environment; (11) feature set; (12) whether v3's training data still exists. *(Decision-rule form and threshold now ✅ confirmed.)*

### 1.4d Cross-version dataset summary (v1–v3)

Consolidated view of each version's train / test / validation / OOT splits — sizes and periods.
**v1 is real (confirmed 2026-07-28); v2 and v3 are placeholder templates awaiting the real numbers.**

> **Reading notes.** (1) The **split *scheme* differs by version, and so does the splitting variable** —
> v1 uses **`lossdate`** cut-offs with a non-OOT random Test plus two date-bounded validation sets
> (§1.4a); v2 uses **`ReportedDate`** with a proportional head/tail split (§1.4b); v3's scheme is
> **unconfirmed**. (2) **The split names are inverted between v1 and v2** (§1.4b) — rows below are
> therefore labelled by **role**, with each version's own name in brackets. (3) The v1 sizes below are
> the **raw `lossdate`-split counts *before* the `cc_fttl` exclusion**; post-exclusion counts
> (Train 173,758 · Test 43,471 · Val1 35,254 · Val2 51,189) are in §1.4a.

| Version | Role *(version's own name)* | Condition / window | Size | Split type / OOT? |
|---|---|---|---:|---|
| **v1** | Train *(Train)* | `lossdate < 2017-04-01`, then 80% | 178,435 | — |
| **v1** | In-period holdout *(**Test**)* | random split from Train (80/20, `random_state=0`) | 44,609 | Random (non-stratified); **not OOT** |
| **v1** | Temporal holdout 1 *(Val1)* | `2017-04-01 ≤ lossdate < 2017-06-30` | 36,049 | Temporal (OOT, ~3 months) |
| **v1** | Temporal holdout 2 *(Val2)* | `2017-06-30 ≤ lossdate ≤ 2018-02-09` (dataset end ✅ 2026-08-04) | 52,498 | Temporal (OOT, ~7.5 months) |
| **v2** | Train *(Train)* | 80% of the `ReportedDate` `2018-01-02 → 2019-12-02` pool (sorted head) ✅ | 🔎 TBD | Random, **stratified on target**; `random_state` 🔎 |
| **v2** | In-period holdout *(**Validation**)* | remaining 20% of the same `2018-01-02 → 2019-12-02` pool ✅ | 🔎 TBD | Random, **stratified**; **not OOT** |
| **v2** | Temporal holdout *(**Test**)* | `2019-12-02 → 2020-09-30` (realised ✅ 2026-08-04; tail then `> 2019-12-01` filter, non-binding) | 🔎 TBD sizes | Temporal (OOT, ~10 months) |
| **v3** | Train *(Train)* | 80% of the shuffled `ReportedDate_CLAIM < 2025-12-01` pool | 🔎 TBD | Random shuffle (`seed=123`), **not stratified** |
| **v3** | In-period holdout *(**Test**)* | first 20% of the same shuffled pool | 🔎 TBD | Random; **not OOT** |
| **v3** | Temporal holdout *(**OOT**)* | `2025-12-01 ≤ ReportedDate_CLAIM < 2026-05-01` | 🔎 TBD | Temporal (OOT, **5 months**) |

**v1 date boundaries:** `EndTrainDate = 2017-04-01`, `EndValidationDate = 2017-06-30` (both on `lossdate`); **dataset max `lossdate` = 2018-02-09** ✅ (2026-08-04 — full extract period 2016-01-01 → 2018-02-09). Source extract shape `(311,591 × 198)`; splits sum to 311,591 (§1.4a), i.e. no corrupted-row drop in v1.

**v2 window:** in-period pool `2018-01-02 → 2019-12-02`, OOT `2019-12-02 → 2020-09-30` ✅ (realised windows confirmed 2026-08-04), minus all rows with `ReportedDate == 2020-08-24` (which falls **inside the OOT block**). The realised OOT start (2019-12-02) sits immediately after the hard-coded `> 2019-12-01` filter, so the filter was **non-binding** and the silent-loss region is empty or near-empty for this run (§1.4b). Sizes still 🔎 — report realised sizes only.

**✅ v2 cleaning detail — confirmed 2026-07-29.** v2 drops rows with `ReportedDate == 2020-08-24` as corrupted; this belongs to **v2**, not v1 (open item closed). ⚠️ The equality filter removes **every claim reported that day**, not a single record (§1.4b). 🔎 Still TBD: v2's raw extract shape, rows dropped, and whether other corrupted rows are removed.

#### Split scheme by version

| | **v1** | **v2** | **v3** |
|---|---|---|---|
| Splitting variable | `lossdate` (accident date) | **`ReportedDate`** | `ReportedDate_CLAIM` — **same field as v2's**, renamed |
| How the OOT boundary is set | **explicit cut-off dates** | **proportional** `tail(⌊0.2n⌋)`, then date-filtered | **explicit date** (`≥ 2025-12-01`) |
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

**v3's split design is the soundest of the three** — explicit OOT date, ~5-month holdout matching §1.4, exactly complementary head/tail, an explicit maturation buffer, unambiguous naming. Its two weaknesses are the **loss of v2's stratification** and the **claim-level leakage exposure** from shuffling (§1.4c).

#### Model configuration by version — hyperparameters, package versions, training data

Consolidated reference. **Decimals are exactly as supplied from the real code; trailing `…` marks values truncated at source.** `—` means the parameter is not set in that version's call (library default applies); 🔎 means not yet confirmed.

##### Environment & training data

| | **v1** | **v2** | **v3** |
|---|---|---|---|
| **xgboost version** | **0.72** ✅ *(read off env-v1 2026-08-08: the vendored cp35 wheel reports `0.72`, correcting the earlier "0.72.1"; resolves the "pre-1.0, uses deprecated `silent=`" 🔎)* | **1.4.2** | **3.2.0** |
| Python / dataframe stack | Python 3.5.2, pandas 0.22.0 | pandas 🔎 | **Polars** |
| Model trained | ~2017 | **2021** | 🔎 (2025 or 2026 — unresolved) |
| Deployed? | Yes, superseded | **Yes — currently live** | ❌ Never |
| Training window | `lossdate < 2017-04-01` (extract → 2018-02-09 ✅) | `2018-01-02 → 2019-12-02` pool; OOT → 2020-09-30 ✅ | `2023-06-01 → 2026-05-01` |
| Window span | ~1 yr 3 mo (train) / 2 yr 1 mo (extract) | **1 yr 11 mo (pool)** / 2 yr 9 mo incl. OOT | 2 yr 11 mo |
| Split variable | `lossdate` (accident date) | `ReportedDate` | `ReportedDate_CLAIM` *(= v2's field, renamed)* |
| Label source | `pre_ml_label` (human era) | v1 production log | v2 production log |
| Contamination | partial (human forced labels) | **essentially total** | **total** |
| Rows (raw extract) | 311,591 × 198 | 🔎 | 🔎 |
| In-period split stratified? | ❌ No | ✅ Yes (`stratify=target`) | ❌ No |
| Maturation buffer | 🔎 not visible | 🔎 not visible | ✅ `immature_date` |
| Training data still exists? | ✅ Yes — incl. `pre_ml_label` ✅ | ✅ Yes (`Z:`) | ✅ Yes (training + OOT) |
| Production log still exists? | ❌ **Gone** (inputs + outcomes) | ✅ Yes | ❌ n/a — never deployed |

##### Hyperparameters

| Parameter | **v1** | **v2** | **v3** |
|---|---|---|---|
| **target column** | `target` = `veh_total_loss ∨ veh_fast_track` ✅ *(disjunct = safeguard)* | **`veh_total_loss`** alone ✅ | `Fttl` **= `veh_total_loss` renamed** ✅ 2026-08-04 — all three carry the forced 1s in the data |
| **scrap threshold** | 0.75 immobile / 0.85 mobile | 0.872 → **0.825** (2026-06-30 14:30 UK) | **0.984** @ prec 0.985; 0.970 @ prec 0.97 |
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

**How to read this.** v1 sets almost nothing — stock defaults plus a metric, seed and thread count. ⚠️ **But "default" is not one thing across these three models**: v1 ran on **xgboost 0.72** (read off env-v1 2026-08-08), v2 on 1.4.2, v3 on 3.2.0, and the sklearn wrapper's defaults moved in between — the 0.72-era `XGBClassifier` defaulted to `max_depth=3`, `n_estimators=100`, `learning_rate=0.1`, not the values a reader of modern xgboost assumes. So the em-dashes in v1's column encode a **2018 library**, and any "v1 was a simpler model" reading is partly a statement about the library, not about v1's authors. This compounds, rather than replaces, the release-gap problem already raised for v2 → v3 in §"The xgboost version jump also breaks comparability". v2 and v3 are both outputs of automated searches (the long decimals), but they **regularise by opposite mechanisms**: v2 via **penalty terms** (`reg_alpha=20`, `gamma=15`, L2 off) with deep trees; v3 via **tree structure** (`max_depth=3`, `max_leaves=18`, `min_child_weight=44`) with penalties near default. ⚠️ Since **no adjacent pair shares a configuration**, **no cross-version comparison of feature-importance concentration is clean** — see §1.4c for the effect on the SHAP–DiD design.

#### Decision rule by version

Split *sizes* remain TBD for v2/v3, but the **decision rules** are now partly confirmed and **differ in form** — tabulated separately so the difference is not lost inside the size table.

| Version | Decision rule | Arity | Confirmed? |
|---|---|---|---|
| **v1** | `D = 1[s > 0.75]` if immobile; `D = 1[s > 0.85]` if mobile | **two** cutoffs, segmented by mobility | ✅ 2026-07-28 (§1.4a) |
| **v2** (→ 2026-06-30 14:30 UK) | `D = 1[s > 0.872]` | **one** global cutoff | ✅ 2026-07-29 (§1.4b); break instant ✅ 2026-07-31 |
| **v2** (2026-06-30 14:30 UK →) | `D = 1[s > 0.825]` | **one** global cutoff | ✅ 2026-07-29 (§1.4b); break instant ✅ 2026-07-31 |
| **v3** | `D = 1[s > 0.984]` at precision 0.985 (`> 0.970` at precision 0.97) | **one** global cutoff | ✅ 2026-07-29 — but **never deployed** |

**Where garage-verified outcomes exist above a scrap cutoff** — i.e. where the "zero garage rows above τ" premise (§2.6, mitigation/eval design) breaks:

| Version | Overlap band | Source of overlap | Confounder |
|---|---|---|---|
| **v1** | `(0.75, 0.85]` | **cross-sectional** — mobile vehicles garaged where immobile ones are scrapped | **mobility** (observed → adjustable) |
| **v2** | `(0.825, 0.872]` | **temporal** — pre-change claims garaged where post-change ones are scrapped | **calendar time** (no single observed covariate → parallel-trends-style assumption) |

Both bands are **confounded, not random**, so neither restores clean positivity — but both are exploitable, and both show the premise "no garage-verified outcome exists above the scrap cutoff" is **too strong as stated**. v1's band is the better identified (measured confounder); v2's is confounded by time and, post-change, spans only ~4 weeks.

### 1.5 Training Process (Insurance Company. alignment, 2-generation structure)

The synthetic data generator faithfully replicates the **2-generation training process** that actually occurred at Insurance Company. The SFP loop is **baked into** the data — it is not asserted separately. Code: `src/data/synthetic/generate/model.py`.

#### Scrapping policy (form shared by all model versions; threshold value is per-version)

```python
# τ_v is TUNED per version at deployment (not hardcoded): the lowest score cutoff holding
# precision ≥ TARGET_PRECISION (0.985) on a held-out validation slice, scored against that
# version's (SFP-contaminated) training label. 0.872 = documented v2 real value + fallback.
# See generate/model.py: _tune_threshold() + fit/validation split in train_and_apply().

def apply_policy(scores, tau_v):     # scores = model.predict_proba(X)[:, 1]
    return (scores >= tau_v).astype(int)   # 1 → scrap, 0 → garage
```

- **The *form* (absolute cutoff tuned to precision ≥ 0.985) is shared; the threshold *value* τ_v is tuned per-version** — `0.872` is v2's documented real instance; v1/v3 differ because each model's score distribution requires a different cutoff to hold ≥ 0.985. Synthetic generator tunes τ_v per version, landing near the anchor (v1 ≈ 0.852, v2a ≈ 0.906).
- **Absolute cutoff, not percentile.** The vehicle is scrapped only when the model is highly confident. The scrap *volume* flows with the score distribution — this is the mechanism by which v2's upward score drift manifests as a higher scrap rate.
- `decision = 1` → scrap → `observed_outcome` **forced to 1** (the car is gone — the garage never sees it → **self-fulfilling label**).
- `decision = 0` → garage → `observed_outcome` = **true** repair outcome.

#### Model v1 — trained on the pre-ML (human) era

**Purpose of v1: to outperform the call handler**, not to reproduce it (supervisor decision, 2026-07-06). In the pre-ML era, fast-track write-offs were decided by call handlers, who lack the engineering knowledge to judge repair feasibility. **Only the garage engineer's physical assessment is ground truth**; the **call handler is treated unconditionally as a biased data generator — never as ground truth, regardless of individual confidence.** A handler who writes a car off without a garage inspection produces a forced, unverified label — the original **human-based SFP loop** that predates the model. v1 was introduced to make these decisions more accurately than a handler, yet it is *trained* on `pre_ml_label` (labels that biased handler generator produced), so v1 inherits the very human-based loop it was meant to improve on. The full FTTL problem is therefore **two nested SFP loops**: the human-based loop embedded in `pre_ml_label`, and the model-based forced-label loop (§2.2–2.3) that inherits and amplifies it.

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

> **✅ RETRACTED 2026-08-05 — this constraint was stated backwards.** It previously asserted that v1's training data and `pre_ml_label` were permanently disposed of under retention obligations, and that v1 was observable only through its production log. **Both halves are wrong, in opposite directions.** See §1.4b "Data availability". What actually holds:
>
> - **v1's training data survives, `pre_ml_label` included** — it *can* be re-scored, audited and re-analysed, and methods may assume access to it and to a pre-ML holdout.
> - **v1's production log is what is gone** (`model_v1_score` / `_decision` / `_observed_outcome`, plus the scored inputs). The old instruction — observe v1 "only through the artefacts it left behind, never through its inputs" — is now **exactly inverted**: the inputs survive, the artefacts do not. *(Outcome recovery is being attempted; do not assume it.)*
> - **α (§1.1a) is still only bounded**, but not for the reason given above. Not missing data: for the 6.45% fast-tracked slice the *oracle* was destroyed by the scrapping itself, and the surviving `pre_ml_label` holds the forced 1 rather than the true status. **Data availability is not oracle availability.** The bound's inputs do become directly measurable.
> - **v2b stays synthetic-only, justification inverted** — the `pre_ml_label` half is now available, the v1-log half is not.
>
> **The design constraint survives in weakened, relocated form.** Builds 02–06 must be designed to run without **v1's production log**, not without its training data. Anything requiring genuine observed v1 production behaviour — the §1.4a mobility overlap band, the error-inheritance detector — must either reconstruct v1's scores by re-scoring the surviving artefact (labelling them *reconstructed*, not *observed*) or fall back on garage-verified rows and bounding arguments. The chain still cannot be anchored to uncontaminated ground truth, but the reason is the destroyed **oracle**, not a destroyed **dataset**. Mirrors `README.md` § "Model v1".

#### Model v2 — currently deployed; trained exclusively on v1-generated data

v2 is the **currently active model** at Insurance Company.. It was retrained using only claims data generated under v1's scrapping policy. This makes v2 the first model version whose training labels are entirely contaminated by the SFP loop.

> ⚠️ **Corrected 2026-08-05 — the stated *reason* no longer holds.** This passage previously explained v2's v1-log-only training by "the pre-ML dataset was no longer available at the time of retraining". Since `pre_ml_label` survives (§1.4b "Data availability"), that explanation is **unavailable**: excluding the pre-ML data was evidently a **choice**, not a data-availability necessity. The contamination *fact* is unchanged and confirmed at window level (§1.4b); only its causal account is now open. 🔎 Ask the team why the pre-ML data was not used in v2's retrain — the answer changes whether the loop is best framed as *imposed by constraint* or *entered by decision*, which matters for the mitigation argument.

In the synthetic data two variants are generated side by side:

- **v2a** — the real Insurance Company. scenario: v1 log only, no pre-ML signal.
- **v2b** — a research-only counterfactual: shows what v2 would have looked like if pre-ML data had still been available. **Does not represent anything that happened at Insurance Company..**

```python
# v2a — REAL scenario (v1 log only; pre_ml_label not used at retraining time
#       — note: NOT because it was unavailable, see the correction above)
#   rows 2022–2024, target = model_v1_observed_outcome
model_v2a.fit(X_2022_2024, y=model_v1_observed_outcome)

# v2b — RESEARCH COMPARISON ONLY (counterfactual; not a real Allianz model)
#   rows 2020–2021 → target = pre_ml_label
#   rows 2022–2024 → target = model_v1_observed_outcome
model_v2b.fit(X_2020_2024, y=combined_label)
```

Both variants are scored on all rows and stored in their respective columns (`model_v2a_score`, `model_v2a_decision`, `model_v2b_score`, `model_v2b_decision`).

#### Model v3 — refresh attempted but not deployed (now simulated)

A v3 refresh was attempted (end of 2024 / early 2025) but not put into production; v2 remains the active model. The failure mode: precision could be held at ≥ 0.985 by tightening the threshold, but doing so caused recall to collapse to an operationally unacceptable level. This is the expected SFP signature — positive labels in training are inflated by v2's false positives (scrapped repairable cars), so the decision boundary becomes imprecise, and forcing precision high suppresses true positives alongside them. Two synthetic v3 variants are now generated: **v3a** (trained on v2a log, 2023+, shows SFP deepening) and **v3b** (trained on v2b log, 2023+, counterfactual diluted path).

Additionally, v3 was shelved for a business reason independent of the SFP problem: Allianz acquired **Control Expert**, a third-party platform that includes its own total loss prediction capability. The decision was taken to retire the in-house FTTL model in favour of Control Expert — without conducting a proper comparative evaluation. Control Expert is not yet integrated (expected **early 2027**). This means v2 remains live with no near-term retraining planned, and any SFP deepening in the interim accumulates unaddressed. The dissertation should note this as a real-world constraint: the operational window for applying SFP mitigation to the current model is limited by an external business timeline, not by technical readiness.

#### How the loop manifests in the generated data

Each version tunes its own threshold τ_v to hold **precision ≥ 0.985 against the SFP-contaminated
label** (see §1.4). Measured on each version's OOT window (`verify_sfp_oracle.py`, synthetic-only
oracle check; `evaluate.py` for the contaminated view):

| Model | Training target | Scrap rate | Contaminated prec | **Oracle prec (true)** | Gap | Status |
|---|---|---|---|---|---|---|
| v1 | `pre_ml_label` (human era) | ~12.5% | 0.976 | 0.974 | 0.002 | Deployed (superseded) |
| **v2a** *(real)* | `model_v1_observed_outcome` | **~18.4%** ↑ | 0.988 | 0.980 | 0.008 ↑ | **Currently deployed** |
| v2b *(counterfactual)* | mixed (pre-ML + v1 log) | ~18.3% | 0.992 | 0.985 | 0.007 | Synthetic only — not real |
| v3a *(SFP deepening)* | `model_v2a_observed_outcome` | **~18.6%** ↑ | 0.988 | 0.972 | **0.016** ↑ | Attempted; not deployed (real) |
| v3b *(counterfactual)* | `model_v2b_observed_outcome` | ~18.8% | 0.983 | 0.972 | 0.011 | Synthetic only — not real |

**Where the loop hides — threshold absorbs the score drift.** The contaminated labels force
`scrap → label 1`, so the **contaminated precision the business monitors stays pinned near 0.985**;
tuning τ_v to that easy target does not force the threshold up enough to cut recall, so **recall
does not collapse in this DGP** (it rises with scrap rate). Instead the SFP surfaces where the
contaminated view is blind:
- **The contaminated−oracle precision gap widens**: 0.002 → 0.008 → 0.016 (v1→v2a→v3a) — contaminated precision stays high while true (oracle) precision is dragged down by hidden false positives. **Read the gap, not the oracle-precision level:** within a fixed OOT window the gap is the clean signal, but the oracle-precision *level* is not directly comparable across versions because the windows differ in true-TL base rate (see next bullet).
- **Scrap rate inflates** (~12.5% → ~18.4% → ~18.6%), but this is **partly confounded**: the true-TL base rate itself drifts up across the OOT windows (≈16.6% at v1's 2021 window → ≈22.1% at the 2024 windows), a year-over-year feature-drift effect **separate from the SFP loop** (under investigation in `notebook/00_feature_drift_EDA.ipynb`). Scrap inflation alone therefore cannot prove SFP — the widening gap is what isolates the loop's hidden false-positive harm.

Consequence for detection — what each candidate signal is worth here:
- **τ_v (threshold trajectory): unreliable.** Endogenous (a *response* to the contaminated data), non-monotonic (0.852 → 0.906 → 0.891). Cannot be read as an SFP indicator.
- **Contaminated precision: uninformative by construction.** Pinned near 0.985 because tuning targets it and `scrap → label 1` makes it easy. It absorbs the drift and hides the harm.
- **Scrap-rate inflation: a visible symptom, but confounded.** It *does* rise (~12.5% → ~18.6%) because holding contaminated precision does not constrain scrap *volume* — so it is not "absorbed". But rising scrap rate can equally come from case-mix / score-distribution shift unrelated to SFP (here the true-TL base rate itself drifts ≈16.6% → ≈22.1% across the OOT windows — see `notebook/00_feature_drift_EDA.ipynb`), so it flags a symptom without proving harm.
- **The contaminated-vs-oracle gap: the dispositive signal.** Directly measures the SFP harm (repairable cars wrongly scrapped). Available only in the synthetic DGP (`verify_sfp_oracle.py`), never operationally — which is exactly why the real detection framework (Build 02/03) needs the IPS / selective-labels machinery to *estimate* this gap rather than read it directly.

> **Note — real vs synthetic v3.** The real v3 was shelved because recall collapsed when precision
> was held (README/§Model v3). The current synthetic DGP produces a *milder* SFP that manifests as
> a widening contaminated−oracle precision gap + scrap inflation rather than recall collapse, because the loop's
> contamination (~1.5–2.8% false positives among scrapped, anchored by the true garage labels of the
> ~80% non-scrapped majority) is too weak to compound into a recall collapse. Reproducing the real
> recall-collapse shape would require a stronger contamination/amplification mechanism in the DGP —
> logged as an open modelling task.

v2b and v3b are counterfactual analytical baselines only; they do not reflect what happened at Insurance Company..

---

## 2. Formal Problem Statement

### 2.1 Notation

| Symbol | Meaning |
|--------|---------|
| $X_i$ | Observable claim features — base claim fields (damage severity, vehicle age, mileage, etc.) plus enrichment-derived fields (repair-to-value ratio, vehicle value, part cost index, used-car price index) and vehicle physical specs (BHP, kerb weight, height, acceleration, number of gears) joined from `vehicle_enrichment.parquet` |
| $\hat{f}_v$ | Model version $v$ (v1, v2a, v2b, ...) |
| $S_i^v = \hat{f}_v(X_i)$ | Score assigned by version $v$ |
| $D_i^v \in \{0,1\}$ | Decision by version $v$: $D_i^v = \mathbb{1}[S_i^v \geq \tau_v]$, where $\tau_v$ is version $v$'s absolute threshold, each tuned to hold precision ≥ 0.985 on a held-out validation slice (against the contaminated training label). $\tau_{v2} = 0.872$ (documented real value); $\tau_{v1}, \tau_{v3}$ differ. The synthetic generator tunes $\tau_v$ per version (`generate/model.py::_tune_threshold`), with $0.872$ as fallback |
| $Y_i \in \{0,1\}$ | **True** repairability — the oracle. Exists only in the data-generating process; never stored in operational data |
| $\tilde{Y}_i^v$ | **Observed label** produced by version $v$ |

### 2.2 Label Generation Mechanism — The Core Issue (distinction from P27)

$$\tilde{Y}_i^v = \begin{cases} 1, & \text{if } D_i^v = 1 \quad (\text{forced, regardless of true } Y_i) \\ Y_i, & \text{if } D_i^v = 0 \quad (\text{true observed outcome}) \end{cases}$$

**Key point:** When $D_i^v=1$, $\tilde{Y}_i^v$ is not missing (NA) but **forced to 1**. The classical selective labels problem (P27, Lakkaraju et al. 2017) assumes labels are *missing* when $D_i=1$; here the label **exists but is false**. This makes the closer problem type **PU learning (labelled positives + contaminated unlabelled, P28)** or a **forced/contaminated positive label** problem.

### 2.3 Loop Mechanism (Self-Reinforcing Cycle)

$$\hat{f}_{v1} \to S^{v1} \to D^{v1} \to \tilde{Y}^{v1} \to \hat{f}_{v2}.\text{fit}(X, y=\tilde{Y}^{v1}) \to S^{v2} \to D^{v2} \to \cdots$$

1. $\hat{f}_{v1}$ produces score $S^{v1}$; threshold $\tau$ generates decision $D^{v1}$.
2. Rows where $D_i^{v1}=1$ are **forced** to $\tilde{Y}_i^{v1}=1$ (regardless of true $Y_i$).
3. $\hat{f}_{v2}$ is trained using $\tilde{Y}^{v1}$ (contaminated labels) as the target.
4. If $\hat{f}_{v1}$ was over-confident in some region (e.g. a particular damage pattern), all rows in that region receive $\tilde{Y}=1$, and $\hat{f}_{v2}$ learns that pattern as "confirmed fact" → $S^{v2}$ is higher in that region → $D^{v2}$ scraps more cases → the loop intensifies (observed: v1 ~12.5% → v2a ~18.4% scrap rate).
5. **This is the self-fulfilling prophecy:** the model's prediction ($D^{v1}=1$) eliminates any means of verifying its own accuracy (the vehicle is scrapped so $Y_i$ can never be observed), while simultaneously providing the next generation model with false evidence ($\tilde{Y}^{v1}=1$) that "the prediction was correct."

**Formal unification (P29 — Veprikov et al. 2025):** The cycle above is an instance of the repeated learning map $E_{t+1} = F(E_t, M_t)$, where $E_t$ is the claim feature distribution at model generation $t$ and $M_t = \hat{f}_{v(t)}$ is the deployed model. The loop is "hidden" when standard OOT AUC is not a monotone function of the true performative risk — precisely the Insurance Company. situation where v2a's OOT AUC looked acceptable while the scrap rate inflated.

**Convergence condition (P31 — Mendler-Dünner et al. 2020):** Let $\varepsilon$ be the distribution sensitivity (Wasserstein distance shift per unit change in model parameters) and $\beta$ the strong convexity constant of the XGBoost loss. The loop converges to a biased-but-stable fixed point when $\varepsilon / \beta < 1$; it diverges (runaway amplification, as in P12 Pólya urn) when $\varepsilon / \beta \geq 1$. Build 02 estimates this ratio empirically from the v1→v2a score distribution shift as a single-number **SFP loop severity score**.

**Loop type classification (P30 — Pagan et al. 2023):** The total loss loop is a **positive-gain, short-delay amplifying feedback loop** in the control-theory taxonomy — the forced label is applied immediately at the scrapping decision (not after a lag), and the gain is positive (higher score → more scrapping → more forced-positive labels → higher score next generation). This type provably converges to a maximally biased fixed point, consistent with Build 01's simulation results.

**Echo chamber vs. filter bubble (P32 — Jiang et al. AIES 2019):** The loop has two separable components. The *echo chamber* is the model amplifying its own past decisions (rising cross-version Spearman rank correlation — Build 02 Step 1). The *filter bubble* is the model becoming blind to true repairability of certain vehicle segments as their scrap rate approaches 100% (segment blind spots — Build 02 Step 4). Formal degeneracy condition: if the Jacobian spectral radius of the v1→v2a score-mapping exceeds 1 in a score band, that band is on a diverging path toward information collapse. These two signals together constitute the full observable signature of the SFP loop and motivate Steps 1 and 4 of Build 02 SFPDetector as distinct, complementary tests.

**Stateful performativity and why v3 retraining failed (P33 — Brown, Hod & Kalemaj, AISTATS 2022):** P15's map D(θ) depends only on the current model θ. P33 extends this to D(θ, s_t), where s_t is the *accumulated state* of forced-positive labels across all prior model versions. State evolution: s_{t+1} = g(s_t, θ_t). Convergence to a good equilibrium requires the state sensitivity ε_s (how much the contamination state shifts per model generation) to satisfy a joint contraction condition with ε_θ. The failure of v3 retraining is consistent with this: after v1 and v2a both operated under the 0.872 threshold, the contamination state s was sufficiently entrenched that retraining on s-contaminated labels could not converge to the performatively optimal classifier. v2b's partial resistance to SFP is also explained — including pre-ML labels in training effectively reduces ε_s by partially resetting the contaminated state. Build 01's simulation of v1→v2a→v3 trajectories should model the stateful dynamics explicitly, not just single-step transitions.

**Model-class-agnostic anchoring (P34 — Taori & Hashimoto ICML 2023; P35 — Adam et al. CHIL 2022):** The convergence results above (P31/P33) assume a strongly convex loss, which the production XGBoost model does **not** satisfy — so for FTTL their ε/β-style guarantees are approximations, not theorems (cf. §2.5 and the 2026-06-25 framing decision in `literatures/reading_list.md`). The loop is nonetheless on firm ground because its driver is the **forced-label mechanism (§2.2), which is independent of model class**. P34 demonstrates empirically — on deep, non-convex models, with no convexity assumption — that retraining on a model's own outputs amplifies bias, and that amplification grows with the fraction of model-labelled data (v2a is trained *entirely* on v1-generated labels — the worst case). P34's *uniform faithfulness* criterion gives Build 02 a model-agnostic stability diagnostic: compare the realised v2a forced-label distribution against the v1 score distribution that generated it. P35 supplies the applied precedent: in a real ICU system, a deployed non-linear model's false positives propagated into its next training set and the error amplified — structurally identical to a false-positive scrap forcing $\tilde{Y}=1$. The dissertation's formal claim is therefore scoped to the *mechanism* (label generation + retraining), not to a convergence theorem for tree ensembles.

### 2.4 Problem Type Taxonomy — Which Existing Framework Fits?

→ Moved to [`literatures/reading_list.md` — Problem Type Taxonomy section](literatures/reading_list.md#problem-type-taxonomy--which-existing-framework-fits).

### 2.5 Additional Difficulties Created by Operational Constraints

1. **Precision ≥ 0.985 constraint:** Any mitigation technique (randomisation, threshold adjustment, etc.) that violates this constraint is commercially non-viable. A simple "explore more" solution cannot be proposed without a cost-benefit analysis.
2. **Absolute threshold (not percentile):** Because the scrap rate is a function of the score distribution, score drift alone can shift the scrap rate — when detecting the loop, one must distinguish whether "increasing scrap rate" is a genuine loop signal or merely distributional shift.
3. **No calibration:** Any correction technique that uses $S_i^v$ as a propensity score (IPS weight) may be distorted by this lack of calibration — affects all IPW/PSM-class methods used in Build 03/06.
4. **Permanent absence of oracle $Y_i$:** Scrapped vehicles are physically destroyed, so post-hoc audits cannot recover $Y_i$. This is a stricter constraint than the typical judicial/medical selective-labels setting covered by P27 (where, for example, bail-denied individuals can be observed later).
5. **Contamination of the OOT holdout itself:** The OOT period (most recent 6 months) is drawn from logs in which v1 was already in production, so OOT evaluation may itself be contaminated by $\tilde{Y}^{v1}$ — the assumption that "we validated on future data, so we're safe" may not hold.
6. **ENOL/FNOL channel-mix shift as a confound (confirmed 2026-06-22):** Claims are filed either by phone (FNOL — First Notification of Loss) or online (ENOL — Electronic Notification of Loss, recently introduced). The two channels produce structurally different feature distributions: FNOL claims are handler-mediated and may capture more severe incidents (claimants in worse situations tend to phone rather than self-serve online), while ENOL claims follow a fixed form path with different missing-value and error patterns. If the proportion of ENOL claims grows over time as online filing becomes more common, this induces a feature distribution shift that is independent of any SFP loop. Score drift attributed to SFP amplification must therefore be checked against the ENOL/FNOL mix change as an alternative explanation. See `README.md` § "Claim Intake Channels — FNOL vs ENOL" for the full operational context.
7. **Model environment incompatibility — versions cannot share a runtime:** All three model versions (v1, v2, v3) are preserved as serialised files within Allianz's internal systems. **Every version has a genuinely different, mutually incompatible library environment — this is confirmed, not a "likely".** Each version's pickle is bound to the *exact* third-party library versions it was serialised with (its own XGBoost / scikit-learn / numpy releases), and those versions differ across v1/v2/v3 and cannot coexist in one Python process. The project therefore manages **a separate, independently pinned environment per version** (`env-v1` / `env-v2` / `env-v3`), so that retraining or upgrading one version can never silently mutate another's frozen environment. Any framework that needs to load and compare model outputs across versions must isolate each version in its own process or environment. This is not just an infrastructure concern: it constrains which detection methods can be applied interactively across model generations without manual environment switching. See `src/DESIGN.md` and `src/ENV_MANAGEMENT.md` for the isolation strategy (offline per-version scoring; analysis runtime loads no model).

   **Why "just install every version's repo into one env" does not work (the recurring misunderstanding):** the blocker is **not** the repos' source code — that installs fine — but the **third-party numeric stack** each pickle is bound to. A pickle stores class *references* + fitted *state*, never source code, so loading re-imports both the repo's custom classes **and** the exact library classes the pipeline was built from (`xgboost.sklearn.XGBClassifier`, `sklearn.pipeline.Pipeline`, numpy dtypes); all must resolve at a compatible version or the load fails. A Python environment is a **flat** library pool — *one version of each library, shared by everything installed in it* — so installing packages does **not** isolate their dependencies (unlike npm's nested `node_modules`). Installing v1+v2+v3 into one `.venv` forces `xgboost`/`scikit-learn`/`numpy` to a single resolved version; when v1 and v3 need incompatible versions (which they do), that is physically unsatisfiable — pip errors, or one version's pickle silently fails to load at runtime. Physically separating the environments is the **only** thing that provides isolation; hence the per-version env split is mandatory, not a tidiness choice.

   **Reproduction model — clone & run, not re-implement (confirmed 2026-07-01; ✅ corrected 2026-08-08):** each version is preserved as **two separate pickles in the repo's `./outputs/`** — a fitted preprocessing pipeline (`fttl_pipeline.pkl` / `pipeline.pkl` / `p146_pipeline.pkl`) and a fitted estimator (`fasstacker_xgb.pkl` / `model.pkl` / `p146_model.pkl`), whose `predict_proba` takes the **already-preprocessed** matrix. The original 2026-07-01 description ("preprocessing *and* model frozen together in one `.pkl`") was wrong on the packaging; every consequence below is unchanged, because the preprocessing still exists only as a fitted pickle. Three consequences. (a) A version's preprocessing is **not independently re-implementable and must not be guessed**: the correct way to obtain its features/scores is to **clone that version's model repo and run its own pipeline** in a matching env (guessing would inject a spurious #10 artefact). (b) `joblib.load` **reconstructs custom transformer classes by importing them**, so that version's code must be on the import path at its pinned library versions — which is *why* the per-version env is mandatory, not merely tidy; the pickle format makes it non-optional. Version repos are pulled in as a **git dependency of each `env-vX` pinned to a commit SHA**, not as git submodules (submodules tried and abandoned for detached-HEAD / recursive-clone friction). (c) **A pickle can `predict` but cannot re-train itself:** the mitigation re-evaluation (retrain on corrected labels) requires each version's **training script from its repo**, run in that version's env. **A version available only as a prediction pickle, with no runnable training code, is scoreable but not re-trainable** → it drops to symptom tracking and is excluded from the quantitative before/after mitigation comparison (compounds difficulty 9). Confirming runnable training code (not just the pickle) exists per version is therefore a **prerequisite** for the mitigation experiment. This also unifies real and synthetic onto a single contract — *a version is always a pipeline artefact loaded and run* (`predict_proba(raw)`); to honour "always code for real, else reproduce real conditions in synthetic", the synthetic generator likewise emits per-version pipeline pickles consumed by the same `predict.py`/`retrain.py` (agreed direction; refactor of `run.py::export_version_features` pending). See `README.md` § "Model artefacts and reproduction — clone & run, not re-implement".
8. **Enrichment table update mechanism unconfirmed:** The enrichment table (vehicle values, part cost indices) is refreshed approximately every 6, 9, or 12 months, independently of model retraining. However, it is **not yet confirmed** whether: (a) per-ABI-code entries are static once added, (b) existing value fields are refreshed to track market prices across update cycles, or (c) only new manufacture-year rows are appended. If the real enrichment table does update existing values, then `repair_to_value_ratio` — the strongest DGP predictor — can shift for the same vehicle across training windows purely due to enrichment changes, independently of any SFP loop. This is a confound for SFP score drift detection that cannot be resolved without confirming the update mechanics with the Allianz data team. See `README.md` § "Enrichment Table — Update Cycle and Open Questions" for full detail.
9. **Leakage-free common window is required for cross-version *quantitative* comparison (but not for symptom tracking):** The P29 dynamical-systems mapping (§2.3) treats the model generations $v(t)$ as the time axis of $f_{t+1}=D_t(f_t)$, with the loop summary $\psi_t = f_t(0)$ (residual density at 0). For $\psi_t$ to reflect the retraining operator $D_t$ alone — rather than case-mix or market drift — **all versions must be scored on the same rows, and those rows must be out-of-time for every version simultaneously**; otherwise a version evaluated on its own training rows has optimistically biased residuals that fake a positive loop. Two consequences: (i) *symptom* analysis (score drift, decision-rate inflation) tolerates training rows and needs no leakage-free window, but the $\psi_t$ trajectory and any performance metric (AUC, precision) require the locked window; (ii) in the **synthetic** data this is automatic — all ML cutoffs are aligned at 2024-04-30 by `_train_cutoff()`, so May–Oct 2024 is OOT for all five versions with zero train/eval overlap by construction. In **real** data it is far tighter: the binding constraint is the latest-trained model (v3, 2025 on 2023+ data), so a v1/v2/v3 common leakage-free window sits after v3's later cutoff and is much narrower (compounded by label maturation, the oracle-free restriction to garage-verified rows, and whether the v1 artefact can still be re-scored — cf. difficulty 7). The pragmatic fallback is to restrict the quantitative comparison to v1 ↔ v2a, whose common window is wider. See `notebook/experiment.design.md` (§2.2 principle, §3 synthetic-vs-real table, §5.6 real-data fallback) and `src/data/synthetic/synth_data_structure.md`.

10. **Preprocessing and training-window divergence across versions (confirmed 2026-06-25):** v1, v2, and v3 differ not only in their training label source and window *size* but in their **data preprocessing pipelines**, which were re-implemented separately for each version rather than shared as one versioned pipeline. The same raw claim therefore maps to different model-ready features $X_i$ under v1 vs v2 vs v3. This is a **third score-drift confound** alongside enrichment updates (#8) and channel-mix shift (#6): an upward shift in $\text{mean}(S^{v2}) - \text{mean}(S^{v1})$ can arise from preprocessing differences alone, with no SFP loop present. Consequence: *symptom* tracking (score / decision-rate drift) on the real logs cannot be read as SFP evidence until the preprocessing contribution is differenced out — which is impossible to do retrospectively, because each version's preprocessing is baked into its serialised artefact. The clean separation (hold preprocessing fixed, vary only label/window) is available **only in the synthetic DGP**, which is a further reason the Build 01 simulation — not the real-log comparison — must carry the burden of proof for the mechanism. This compounds difficulty 9: the leakage-free common window is necessary but not sufficient; even on a shared OOT window the three versions' scores are not directly comparable unless re-scored through a single preprocessing pipeline (feasible only if the v1 artefact can be re-run, cf. difficulty 7). See `README.md` § "Preprocessing and Training-Window Divergence Across Versions." **Concretised 2026-08-08** by reading the three training flows off the repos: the divergence axes now have names — raw source (`Z:` pickle / pre-cleaned `Z:` pickle / database query), dataframe library (pandas / pandas / **polars**), enrichment route (car_table on `abicode_ext` / baked into the cleaned extract / in-repo hpi+thatcham+cc-rule joins), pipeline style (bespoke `claims_pipe` / `tubular` transformers / stateless-code + stateful-pickle) — see `README.md` § "Training Flow & Artefact Storage by Version", closing table.

   **Concrete mechanisms + the synthetic knob (added 2026-07-01):** "re-implemented differently" is not abstract — worked examples in `README.md`: (A) `damage_severity` as `OneHotEncoder` (3 cols) vs `OrdinalEncoder` (1 col) → the same `severe` claim is `[0,0,1]` under one version and `[2]` under another (different schema *and* different semantics — no-order vs linear-order); (B) `vehicle_make` rare-category bucketing → 55 one-hot columns (v1, every make its own) vs 41 (v2, rare makes → `OTHER`), so the same Ferrari claim is a unique feature under v1 but the misc bucket under v2. Divergence has three levels: (a) same schema + same logic, (b) **same schema + different logic** (e.g. different missing-value fill), (c) different schema + different logic (the examples above). On **real** data the level is not chosen (the repos fixed it, typically c); on **synthetic** data it is a **deliberate knob** — level (a) gives Build 01 the clean isolation to *prove* the mechanism (only label/window varies), level (c) *reproduces* this confound to stress-test the detector against encoding noise. The synthetic generator supports both, defaulting to (a).

   **Pipeline response (adopted 2026-06-26):** the scoring stage now uses a **per-version feature contract** — each version is scored on its own `features_<version>.parquet` rather than one shared matrix, so each model always sees the features *it* would actually produce. This does **not** dissolve the confound on real data (it still requires each version's preprocessing to be reconstructable — difficulty 7), but it removes the *single-shared-matrix* error mode by construction and makes the analysis pipeline structurally honest about per-version divergence. The synthetic DGP adopts the identical contract, but its per-version files are **identical by construction** (one preprocessing pipeline + one fitted imputer), which is precisely the held-fixed-preprocessing property that lets Build 01 carry the burden of proof. See `src/DESIGN.md` § "Per-version feature matrices", `src/STRUCTURE.md`, and `export_version_features()` in `src/data/synthetic/run.py`.

11. **Serving-layer migration (AML → FastAPI) and train/serve skew (noted 2026-06-25):** The model's *serving* layer is being migrated from Azure Machine Learning (AML) managed endpoints to a FastAPI service, while *training* is expected to remain in AML (the latter is the current expectation, not yet confirmed). Training and serving are linked through a model registry: AML registers a versioned artefact and FastAPI loads that version for inference. This is primarily MLOps context, not a direct SFP mechanism, but it interacts with two difficulties above. (a) It adds a **train/serve skew** dimension to the preprocessing-divergence confound (#10): if the FastAPI serving path re-implements preprocessing separately from the AML training path, the *same* raw claim can be scored on features that differ from those seen at training time — a within-version analogue of the cross-version preprocessing divergence, again capable of moving scores with no SFP loop present. (b) The registry-based split is precisely where production scores and scrap decisions are emitted, i.e. where the very log data the SFP analysis consumes is generated, so the integrity of that emission path bears on the trustworthiness of the real decision logs (cf. the model-environment isolation constraint, #7). The "training stays in AML" assumption should be re-checked against what the team means; "moving training to FastAPI" would be a loose use of terms, since FastAPI is a serving framework, not a training tool. See `README.md` § "Deployment & Serving Infrastructure — AML → FastAPI."

12. **Application architecture — split by concern (two layers), not by model version (decided 2026-07-01):** Because each version diverges in environment (#7) and preprocessing (#10), a proposal was raised to split the project into three separate applications, one per model version, each with its own `pyproject.toml`. **Resolved against:** the correct cut is by *concern (layer)*, not by *version*. **Version Layer (per-version model worker)** — preprocessing → (re)train → score for one version's artefact — is genuinely isolated one env/`pyproject.toml` per version (`src/model/envs/v1│v2│v3`); this is what "each model on a separate application" correctly refers to, and it already exists as the per-version scoring/feature contract (#10 pipeline response, difficulty 7 isolation). **Analysis Layer (analysis)** — detector + mitigator + re-evaluation — is a **single, version-agnostic application** and must **not** be triplicated, for two substantive (not merely engineering) reasons: (i) the SFP signal lives *between* versions — the cross-version score-drift / temporal-prediction-correlation test (§2.3, the $\psi_t$ trajectory of difficulty 9) is computed on the join of all versions and cannot exist inside any single-version app; (ii) validity of the comparison requires *identical* detection code applied to every version, so three diverging copies would themselves become a confound. The two layers never share a process (Version Layer emits `features_<v>.parquet` + `*_scores.parquet`; Analysis Layer reads and merges on `claim_id`) — the same decoupling already established for environment isolation. **Re-evaluation invariant (answers the natural worry that per-version preprocessing breaks the after-mitigation comparison):** mitigation alters the *training labels / sample weights / training data*, which is **downstream of preprocessing**, so retraining version $v$ on the corrected data reuses $v$'s **same** preprocessing both times — the within-version pre→post Δ is attributable to the mitigation, not to a preprocessing change. This is a distinct axis from the *cross-version* preprocessing divergence of #10: the latter is a between-version confound already differenced away by the per-version feature contract, the former a within-version before/after where preprocessing is held fixed by construction. Retraining is therefore a Version Layer activity (version's own env + preprocessing, on the Analysis-Layer-corrected training set); the before/after score comparison is an Analysis Layer activity — which is why the Version Layer owns *(re)train + score*, not merely *score*. See `README.md` § "Application Architecture — Split by Concern (Two Layers), Not by Model Version", `src/DESIGN.md`, `src/STRUCTURE.md`.

---

### 2.6 SCAR Violation — Formal Analysis (basis: `synth_data_structure.md` + P28 §3.1)

#### What SCAR requires

P28 (Bekker & Davis 2020) Definition 1:

$$e(x) = \Pr(s=1 \mid x, y=1) = \Pr(s=1 \mid y=1) = c \quad \text{(constant)}$$

In our notation: among **true total-loss vehicles** ($Y_i=1$), the probability of being scrapped ($D_i=1$) must be the same regardless of their features $X_i$.

#### Why SCAR fails here

The scrapping decision is a deterministic function of $X_i$:

$$D_i = \mathbb{1}[\hat{f}(X_i) \geq 0.872]$$

So the propensity score is:

$$e(x) = \Pr(D=1 \mid X=x, Y=1) = \mathbb{1}[\hat{f}(x) \geq 0.872]$$

This is a **step function** — not a constant. Among true total losses:

| Claim type | Features | Model score | $e(x)$ |
|------------|----------|-------------|---------|
| Obvious total loss | BMW, severe, multiple, RTV=1.2, 15yr, 150k miles | ≈ 0.97 | **1.0** (always scrapped) |
| Borderline total loss | Vauxhall, moderate, rear, RTV=0.72, 7yr, 60k miles | ≈ 0.60 | **0.0** (always sent to garage) |

Two genuine total-loss vehicles with completely different labelling probabilities — SCAR is violated by construction.

This holds in **both eras**:
- Pre-ML: rules `RTV > 0.9` or `(severe AND age > 15)` are deterministic $X_i$ functions → SCAR violated
- ML era: absolute threshold on model score → SCAR violated

#### What holds instead: structured SAR

P28 §3.1.2 (Selected At Random):

$$e(x) = \Pr(s=1 \mid x, y=1) \quad \text{(varies with } x\text{)}$$

Our case is a **structured** (known mechanism) SAR because $e(x)$ is not unknown — it is exactly recoverable from the model:

$$e(x) = \mathbb{1}[\hat{f}(x) \geq 0.872]$$

This is **better than the typical unknown-SAR setting** in the PU literature. Most SAR methods need to estimate $e(x)$ from data; we can compute it directly.

#### What e(x) is, concretely

| Term | General PU meaning | Our domain |
|------|-------------------|------------|
| $e(x)$ | Pr(this true positive gets labelled) | Pr(this true total-loss car gets scrapped) |
| $s=1$ | labelled positive | $D_i=1$: model decided → scrap |
| $y=1$ | true positive | $Y_i=1$: car genuinely unrepairable |
| $e(x)=1$ | certain to be labelled | score ≥ 0.872 → always scrapped |
| $e(x)=0$ | certain to remain unlabelled | score < 0.872 → garage, true outcome observed |

The complement $1-e(x) = \Pr(D=0 \mid X=x)$ is the **propensity of being sent to garage** — this is the weight used in IPS correction.

#### What "knowing e(x) enables IPS correction" means

**IPS (Inverse Propensity Scoring)** — also called IPW (Inverse Probability Weighting) — is the standard tool for correcting selection bias in observational data (Horvitz & Thompson 1952; cross-reference P3).

**The problem IPS solves:**

We want to measure model performance (or compute a training loss) on the full claim population. But we can only observe true outcomes $Y_i$ for garage claims ($D_i=0$). These are not a random sample — they are claims the model was *least* confident about (score < 0.872). High-damage, high-RTV claims are systematically missing from our observation set.

**The IPS fix:**

$$\hat{\mu}_{IPS} = \frac{1}{n} \sum_{i:\, D_i=0} \frac{Y_i}{\Pr(D_i=0 \mid X_i)}$$

Each observed garage claim is upweighted by $1 / \Pr(\text{sent to garage} \mid X_i)$. Claims that were *unlikely* to reach the garage (high score, just below threshold) are rare in the data and get high weights; claims that were *certain* to go to garage (low score) are plentiful and get weight ≈ 1.

**In our domain:**

$$\Pr(D_i=0 \mid X_i) = 1 - e(x_i) = \mathbb{1}[\hat{f}(x_i) < 0.872]$$

Since the threshold is deterministic, this is exactly 1 for all D=0 claims by definition — IPS weights are all 1 under a hard threshold. This is the degenerate case: **deterministic selection makes IPS uninformative at the threshold boundary**.

The practical workaround used in Build 03/06 is to treat the **model score itself** as a soft propensity:

$$\widehat{\Pr}(D=0 \mid X=x) \approx 1 - \hat{f}(x)$$

This gives non-trivial IPS weights that reflect *how confidently* a claim was near or far from the threshold. Claims with score 0.80 (just below 0.872) get high weights because similar claims just above the boundary were scrapped and are absent from observation.

**Why the no-calibration constraint matters here:**

$\hat{f}(x)$ is an uncalibrated XGBoost score — it is not a true probability. Using it as $\Pr(\text{total loss} \mid X)$ in the IPS denominator introduces bias. If the model systematically over-estimates scores (as SFP causes), the weights will be wrong, and the IPS correction will itself be distorted. This is why calibration absence is flagged as a known limitation for Build 03/06.

**If SCAR were true, IPS would not be needed:**

Under SCAR, $e(x) = c$ (constant), so $\Pr(D=0 \mid X) = 1-c$ for every claim. The IPS estimator reduces to:

$$\hat{\mu}_{IPS} = \frac{1}{n(1-c)} \sum_{i:\, D_i=0} Y_i$$

which is just a scaled version of the garage-observed mean — no feature-dependent reweighting required. The fact that we *need* IPS and that it *matters* is itself a consequence of SCAR being violated.

---

## 3. Paper Reading Checklist

When reading each paper and writing a note (`literatures/notes/pXX.md`), complete the following table and link it back to this document:

| Check Item | Question |
|------------|----------|
| **§2.1–2.3 Mechanism** | Does this paper address the "label generation mechanism" (§2.2) or the "loop mechanism" (§2.3)? Both? Or does it formalise only one of them? |
| **§2.4 Problem Type** | Under this paper's taxonomy, which type does our problem belong to? How does that classification differ from or align with the existing candidates in the §2.4 table? |
| **Precondition Validation** | Are the preconditions for this paper's methodology (e.g. P27's "multiple simultaneous decision-makers + random assignment") satisfied in our data? If not, what can substitute? |
| **Conflict with §2.5 Constraints** | Does this paper's solution conflict with any of: precision ≥ 0.985, absolute threshold, lack of calibration, permanent absence of oracle? |
| **Detection vs Mitigation** | Is this paper useful for **detecting** the loop, **mitigating** it, or both? (Build 02 vs Build 04–06 mapping) |

---

*First written: 2026-06-16. Last updated: 2026-07-06 (Model v1 section — supervisor decision: v1's purpose was to **outperform the call handler**, not reproduce it; the **garage engineer is the only ground truth**, the **call handler is unconditionally a biased data generator (never ground truth)** because it lacks engineering knowledge and writes off cars without garage verification — the original human-based SFP loop. The FTTL problem is framed as **two nested SFP loops** (human-based in `pre_ml_label` + model-based forced-label). Mirrors `README.md` §"Why there is no ground truth (oracle) label" and §"Model v1", which drop the earlier implication that a handler label could be treated as an engineer/ground-truth equivalent). Previous update 2026-07-02 (§2.5 #7 hardened — per-version environment incompatibility is now stated as **confirmed, not "likely"**: every version's pickle is bound to the exact XGBoost/scikit-learn/numpy versions it was serialised with, and those differ and cannot coexist in one process. Added the recurring-misunderstanding rebuttal: installing all repos into one env fails not because of the repo code (that installs fine) but because a Python env is a **flat** library pool — one version of each library, no per-package dependency isolation (unlike npm's nested node_modules) — so v1/v2/v3's incompatible numeric stacks are physically unsatisfiable in a single `.venv`; physical env separation is the only isolation. Mirrors edits to `README.md` § "Model artefacts and reproduction" point 2 and `src/DESIGN.md` § "Model Scoring & Environment Isolation → Why 'just install every version's repo into one env' does not work". Previous update 2026-07-01 (§2.5 #7 extended — clone-&-run reproduction model: versions are pickled sklearn Pipelines (preprocessing+model frozen together), so preprocessing is not re-implementable and must be run from each version's cloned repo in its own env; joblib.load imports custom transformer classes → per-version env mandatory; version repos as git dependency pinned to SHA, not submodules; pickle predicts but cannot retrain → a predict-only version is scoreable but not re-trainable (excluded from mitigation comparison, compounds #9); real+synthetic unified onto one "load pipeline artefact → predict_proba(raw)" contract, synthetic to emit per-version pipeline pickles (run.py refactor pending). §2.5 #10 extended — concrete worked examples (damage_severity one-hot vs ordinal; vehicle_make 55 vs 41 one-hot cols) and the three-level divergence knob (a/b/c) for synthetic. Both mirror new `README.md` §§ "Model artefacts and reproduction — clone & run" and the preprocessing worked examples. Earlier same-day: §2.5 difficulty 12 added — application-architecture decision: split by concern into two layers, not three per-version apps. Version Layer (per-version: preprocess→retrain→score, own env/`pyproject.toml`) isolated per version; Analysis Layer (detector+mitigator+re-evaluation) a single version-agnostic app that must not be triplicated because the SFP signal lives between versions and the comparison needs identical code. Includes the re-evaluation invariant: mitigation acts downstream of preprocessing, so within-version pre→post retraining holds preprocessing fixed — distinct from the cross-version preprocessing confound #10; mirrors new `README.md` § "Application Architecture — Split by Concern (Two Layers), Not by Model Version"). Previous update 2026-06-26 (§2.5 difficulty 10 extended — per-version feature contract `features_<version>.parquet` adopted as the scoring-pipeline response to preprocessing divergence; removes the single-shared-matrix error mode by construction, synthetic files identical-by-construction; cross-refs `src/DESIGN.md` § "Per-version feature matrices", `src/STRUCTURE.md`, `export_version_features()` in `src/data/synthetic/run.py`. Same-day: §2.5 difficulty 11 added — serving-layer migration AML→FastAPI; training expected to stay in AML; flagged as a within-version train/serve skew dimension of the preprocessing confound #10 and tied to log-emission integrity #7; mirrors new `README.md` § "Deployment & Serving Infrastructure — AML → FastAPI". Same-day: §2.5 difficulty 10 added — preprocessing + training-window divergence across v1/v2/v3 as a third score-drift confound; §1.4 operational note extended with the researcher concern that the brief threshold change may not be safely ignorable; both per the 2026-06-25 meeting. Earlier same-day: §2.5 difficulty 9 added — leakage-free common cross-version evaluation window required for the P29 ψ_t mapping and performance comparison but not for symptom tracking; synthetic cutoffs aligned at 2024-04-30 make this automatic, real v3 (2025) makes it the binding constraint; cross-refs to `notebook/experiment.design.md` §2.2/§3/§5.6). Previous update 2026-06-23 (§2.3 extended with P33 Brown et al. AISTATS 2022 stateful performativity D(θ,s_t) and explanation of v3 retraining failure; P32 Jiang et al. AIES 2019 echo chamber / filter bubble distinction and degeneracy condition; P29 Veprikov et al. 2025 repeated-learning map E_{t+1}=F(E_t,M_t); P31 Mendler-Dünner et al. 2020 convergence condition ε/β<1 as loop severity score; P30 Pagan et al. 2023 control-theory classification as positive-gain short-delay amplifying loop. Previous update 2026-06-22: enrichment table extended with vehicle physical specs — BHP, acceleration, gears, kerb weight, height — per Luna meeting 2026-06-22; §2.1 $X_i$ notation updated; v3 section extended with Control Expert replacement context and early-2027 implementation timeline; §2.5 difficulty 6 added — ENOL/FNOL channel-mix shift as a confound for SFP score drift detection; `used_car_price_index` added to synthetic data enrichment step). Source: `README.md` (verbatim transcription, §1), direct formalisation (§2). Working assumptions may be updated as further papers are read. Cross-references: [`literatures/notes/p27.md`](literatures/notes/p27.md) (basis for selective-labels candidate evaluation in §2.4 table), `src/data/synthetic/synth_data_structure.md`, `src/data/synthetic/generate/model.py`.*