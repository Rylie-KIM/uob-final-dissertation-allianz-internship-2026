# Problem Definition — Fast Track Total Loss SFP Loop

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
| **% of all cars scrapped** | **15%** | Human-era baseline scrap rate — the reference against which ML-era inflation (v2a/v3a ~18–19%, §1.5 table) is judged. Scrap-rate inflation above 15% is the headline (but confounded, §2.5 #2/#6/#10) symptom. |
| **% of scrapped cars fast-tracked for TL** (handler-identified, **no garage visit**) | **43%** | ~43% of pre-ML write-offs are **forced, garage-unverified labels** — the pre-ML human-based SFP loop (§2.2, §1.5 "Model v1") **quantified**. These land in `pre_ml_label` as contaminated positives; the remaining ~57% of scrapped cars reached a garage and carry an engineer (ground-truth) outcome. |

This is the concrete size of the contamination `pre_ml_label` carried *before* the model-based loop began — the human loop v1 was trained on and the model loop then amplified (§2.3). Source: Allianz team figures; to be reconciled with the real logs when available. Mirrors `README.md` §"Pre-ML Baseline".

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

  > **Threshold is per-version, not a universal constant.** What is invariant across versions is the **business constraint (precision ≥ 0.985)** and the **policy *form*** (absolute cutoff, not percentile). The cutoff *value* `τ_v` differs by version because each model's score distribution differs, so a different absolute threshold is needed to hold the same precision target. **`0.872` is the documented real-world value for v2**; v1 and v3 were tuned to different values (not confirmed). The synthetic generator now **tunes `τ_v` per version** at deployment — the lowest cutoff holding precision ≥ 0.985 on a held-out validation slice, scored against that version's (SFP-contaminated) training label — with `0.872` retained only as the fallback when the target is unreachable (`generate/model.py::_tune_threshold`). Synthetic tuned values land near the real anchor (v1 ≈ 0.852, v2a ≈ 0.906, distribution-dependent). Wherever this document writes the threshold as `0.872`, read it as the v2 instance of `τ_v`. See `README.md` for the canonical statement.

  > **Operational note — threshold change history (v2):** v2's production threshold was briefly changed away from 0.872 at some point during deployment (exact value and dates not confirmed). Performance degraded and the threshold was promptly reverted to 0.872. **For all analysis in this dissertation, v2's threshold is treated as constant at 0.872 throughout the production period.** The brief deviation is not modelled and is not reflected in the decision columns in the dataset. See `README.md` for the canonical statement of this assumption. **Researcher concern (2026-06-25):** this "treat as constant" assumption is *operational, not verified* — if the brief threshold change overlapped a retraining window, the decisions/labels generated under the changed cutoff would inject an undocumented inconsistency into exactly the label data the SFP analysis depends on. Logged as an open risk to revisit with the real decision logs, not a settled simplification.

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

#### Model v2 — currently deployed; trained exclusively on v1-generated data

v2 is the **currently active model** at Insurance Company.. It was retrained using only claims data generated under v1's scrapping policy — the pre-ML dataset was no longer available at the time of retraining. This makes v2 the first model version whose training labels are entirely contaminated by the SFP loop. In the synthetic data two variants are generated side by side:

- **v2a** — the real Insurance Company. scenario: v1 log only, no pre-ML signal.
- **v2b** — a research-only counterfactual: shows what v2 would have looked like if pre-ML data had still been available. **Does not represent anything that happened at Insurance Company..**

```python
# v2a — REAL scenario (v1 log only; pre_ml_label unavailable at retraining time)
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

   **Reproduction model — clone & run, not re-implement (confirmed 2026-07-01):** each version is preserved as a **pickled scikit-learn `Pipeline`** — preprocessing steps *and* model frozen together in one `.pkl`. Three consequences. (a) A version's preprocessing is **not independently re-implementable and must not be guessed**: the correct way to obtain its features/scores is to **clone that version's model repo and run its own pipeline** in a matching env (guessing would inject a spurious #10 artefact). (b) `joblib.load` **reconstructs custom transformer classes by importing them**, so that version's code must be on the import path at its pinned library versions — which is *why* the per-version env is mandatory, not merely tidy; the pickle format makes it non-optional. Version repos are pulled in as a **git dependency of each `env-vX` pinned to a commit SHA**, not as git submodules (submodules tried and abandoned for detached-HEAD / recursive-clone friction). (c) **A pickle can `predict` but cannot re-train itself:** the mitigation re-evaluation (retrain on corrected labels) requires each version's **training script from its repo**, run in that version's env. **A version available only as a prediction pickle, with no runnable training code, is scoreable but not re-trainable** → it drops to symptom tracking and is excluded from the quantitative before/after mitigation comparison (compounds difficulty 9). Confirming runnable training code (not just the pickle) exists per version is therefore a **prerequisite** for the mitigation experiment. This also unifies real and synthetic onto a single contract — *a version is always a pipeline artefact loaded and run* (`predict_proba(raw)`); to honour "always code for real, else reproduce real conditions in synthetic", the synthetic generator likewise emits per-version pipeline pickles consumed by the same `predict.py`/`retrain.py` (agreed direction; refactor of `run.py::export_version_features` pending). See `README.md` § "Model artefacts and reproduction — clone & run, not re-implement".
8. **Enrichment table update mechanism unconfirmed:** The enrichment table (vehicle values, part cost indices) is refreshed approximately every 6, 9, or 12 months, independently of model retraining. However, it is **not yet confirmed** whether: (a) per-ABI-code entries are static once added, (b) existing value fields are refreshed to track market prices across update cycles, or (c) only new manufacture-year rows are appended. If the real enrichment table does update existing values, then `repair_to_value_ratio` — the strongest DGP predictor — can shift for the same vehicle across training windows purely due to enrichment changes, independently of any SFP loop. This is a confound for SFP score drift detection that cannot be resolved without confirming the update mechanics with the Allianz data team. See `README.md` § "Enrichment Table — Update Cycle and Open Questions" for full detail.
9. **Leakage-free common window is required for cross-version *quantitative* comparison (but not for symptom tracking):** The P29 dynamical-systems mapping (§2.3) treats the model generations $v(t)$ as the time axis of $f_{t+1}=D_t(f_t)$, with the loop summary $\psi_t = f_t(0)$ (residual density at 0). For $\psi_t$ to reflect the retraining operator $D_t$ alone — rather than case-mix or market drift — **all versions must be scored on the same rows, and those rows must be out-of-time for every version simultaneously**; otherwise a version evaluated on its own training rows has optimistically biased residuals that fake a positive loop. Two consequences: (i) *symptom* analysis (score drift, decision-rate inflation) tolerates training rows and needs no leakage-free window, but the $\psi_t$ trajectory and any performance metric (AUC, precision) require the locked window; (ii) in the **synthetic** data this is automatic — all ML cutoffs are aligned at 2024-04-30 by `_train_cutoff()`, so May–Oct 2024 is OOT for all five versions with zero train/eval overlap by construction. In **real** data it is far tighter: the binding constraint is the latest-trained model (v3, 2025 on 2023+ data), so a v1/v2/v3 common leakage-free window sits after v3's later cutoff and is much narrower (compounded by label maturation, the oracle-free restriction to garage-verified rows, and whether the v1 artefact can still be re-scored — cf. difficulty 7). The pragmatic fallback is to restrict the quantitative comparison to v1 ↔ v2a, whose common window is wider. See `notebook/experiment.design.md` (§2.2 principle, §3 synthetic-vs-real table, §5.6 real-data fallback) and `src/data/synthetic/synth_data_structure.md`.

10. **Preprocessing and training-window divergence across versions (confirmed 2026-06-25):** v1, v2, and v3 differ not only in their training label source and window *size* but in their **data preprocessing pipelines**, which were re-implemented separately for each version rather than shared as one versioned pipeline. The same raw claim therefore maps to different model-ready features $X_i$ under v1 vs v2 vs v3. This is a **third score-drift confound** alongside enrichment updates (#8) and channel-mix shift (#6): an upward shift in $\text{mean}(S^{v2}) - \text{mean}(S^{v1})$ can arise from preprocessing differences alone, with no SFP loop present. Consequence: *symptom* tracking (score / decision-rate drift) on the real logs cannot be read as SFP evidence until the preprocessing contribution is differenced out — which is impossible to do retrospectively, because each version's preprocessing is baked into its serialised artefact. The clean separation (hold preprocessing fixed, vary only label/window) is available **only in the synthetic DGP**, which is a further reason the Build 01 simulation — not the real-log comparison — must carry the burden of proof for the mechanism. This compounds difficulty 9: the leakage-free common window is necessary but not sufficient; even on a shared OOT window the three versions' scores are not directly comparable unless re-scored through a single preprocessing pipeline (feasible only if the v1 artefact can be re-run, cf. difficulty 7). See `README.md` § "Preprocessing and Training-Window Divergence Across Versions."

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