# Problem Definition — Fast Track Total Loss SFP Loop

> This document is the **central reference** consulted when reading each paper. When writing a paper note (`literatures/notes/pXX.md`), answer the following questions:
> 1. Which component of §2 (Problem Formalisation) does this paper address?
> 2. Can this paper's methodology actually be applied under the constraints in §3, or do its preconditions break down?
> 3. How does this paper classify our problem within §2.4 (Problem Type Taxonomy) — and is that classification correct?

---

## 1. Business Logic Spec (source: `README.md`, transcribed verbatim 2026-06-16)

### 1.1 Service Overview — Fast Track Total Loss Model

Internal service name: **Fast Track Total Loss**. Without this model, every damaged vehicle is sent to a garage where an engineer determines whether it can be repaired. This process is costly — the insurer pays for the garage assessment time and must provide the customer with a replacement vehicle during that period.

The model's purpose is to **bypass the garage process entirely for obvious total-loss cases** — vehicles so severely damaged that write-off is certain. By fast-tracking these cases directly to salvage, Insurance A Cop. reduces garage costs and delivers a faster settlement to the customer.

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
- **Decision threshold:** The scrap policy applies an **absolute score cutoff** — a vehicle is fast-tracked to salvage only when `model_score ≥ 0.872` (tuned on the validation set to satisfy precision ≥ 0.985). This is **not** a percentile/top-N rule. Because the cutoff is fixed in score space, the *scrap rate* moves freely with the score distribution — this is precisely the mechanism by which score drift in later model versions becomes observable as an increased scrap rate (= the key SFP signal).

  > **Operational note — threshold change history:** The production threshold was briefly changed away from 0.872 at some point during deployment (exact value and dates not confirmed). Performance degraded and the threshold was promptly reverted to 0.872. **For all analysis in this dissertation, the threshold is treated as constant at 0.872 throughout the entire production period.** The brief deviation is not modelled and is not reflected in the decision columns in the dataset. See `README.md` for the canonical statement of this assumption.

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

### 1.5 Training Process (Insurance A Cop. alignment, 2-generation structure)

The synthetic data generator faithfully replicates the **2-generation training process** that actually occurred at Insurance A Cop. The SFP loop is **baked into** the data — it is not asserted separately. Code: `src/data/synthetic/generate/model.py`.

#### Scrapping policy (shared by all model versions)

```python
SCRAP_THRESHOLD = 0.872          # absolute P(total_loss) cutoff (real Insurance A Cop. value)

def apply_policy(scores):        # scores = model.predict_proba(X)[:, 1]
    return (scores >= SCRAP_THRESHOLD).astype(int)   # 1 → scrap, 0 → garage
```

- **Absolute cutoff, not percentile.** The vehicle is scrapped only when the model is highly confident. The scrap *volume* flows with the score distribution — this is the mechanism by which v2's upward score drift manifests as a higher scrap rate.
- `decision = 1` → scrap → `observed_outcome` **forced to 1** (the car is gone — the garage never sees it → **self-fulfilling label**).
- `decision = 0` → garage → `observed_outcome` = **true** repair outcome.

#### Model v1 — trained on the pre-ML (human) era

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

v2 is the **currently active model** at Insurance A Cop.. It was retrained using only claims data generated under v1's scrapping policy — the pre-ML dataset was no longer available at the time of retraining. This makes v2 the first model version whose training labels are entirely contaminated by the SFP loop. In the synthetic data two variants are generated side by side:

- **v2a** — the real Insurance A Cop. scenario: v1 log only, no pre-ML signal.
- **v2b** — a research-only counterfactual: shows what v2 would have looked like if pre-ML data had still been available. **Does not represent anything that happened at Insurance A Cop..**

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

| Model | Training target | Scrap rate | Status |
|---|---|---|---|
| v1 | `pre_ml_label` (human era) | ~13.1% | Deployed (superseded) |
| **v2a** *(real)* | `model_v1_observed_outcome` (v1 log only) | **~14.1%** ↑ | **Currently deployed** |
| v2b *(counterfactual)* | mixed (pre-ML + v1 log) | ~13.4% | Synthetic only — not real |
| v3a *(SFP deepening)* | `model_v2a_observed_outcome` (2023+) | **~14.2%** ↑ | Attempted; not deployed (recall collapsed) |
| v3b *(counterfactual)* | `model_v2b_observed_outcome` (2023+) | ~14.0% | Synthetic only — not real |

v2a's scrap rate inflates even though the true repairability of vehicles has not changed — v1's self-fulfilling labels push v2 to over-predict total loss. v3a continues the same deepening pattern. This is the observable signature that the detection framework (Build 02) must capture. v2b and v3b are included in the synthetic data purely as counterfactual analytical baselines; they do not reflect what happened at Insurance A Cop..

---

## 2. Formal Problem Statement

### 2.1 Notation

| Symbol | Meaning |
|--------|---------|
| $X_i$ | Observable claim features — base claim fields (damage severity, vehicle age, mileage, etc.) plus enrichment-derived fields (repair-to-value ratio, vehicle value, part cost index, used-car price index) and vehicle physical specs (BHP, kerb weight, height, acceleration, number of gears) joined from `vehicle_enrichment.parquet` |
| $\hat{f}_v$ | Model version $v$ (v1, v2a, v2b, ...) |
| $S_i^v = \hat{f}_v(X_i)$ | Score assigned by version $v$ |
| $D_i^v \in \{0,1\}$ | Decision by version $v$: $D_i^v = \mathbb{1}[S_i^v \geq \tau]$, $\tau = 0.872$ (fixed absolute threshold) |
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
4. If $\hat{f}_{v1}$ was over-confident in some region (e.g. a particular damage pattern), all rows in that region receive $\tilde{Y}=1$, and $\hat{f}_{v2}$ learns that pattern as "confirmed fact" → $S^{v2}$ is higher in that region → $D^{v2}$ scraps more cases → the loop intensifies (observed: v1 19% → v2a 21.5% scrap rate).
5. **This is the self-fulfilling prophecy:** the model's prediction ($D^{v1}=1$) eliminates any means of verifying its own accuracy (the vehicle is scrapped so $Y_i$ can never be observed), while simultaneously providing the next generation model with false evidence ($\tilde{Y}^{v1}=1$) that "the prediction was correct."

**Formal unification (P31 — Veprikov et al. 2025):** The cycle above is an instance of the repeated learning map $E_{t+1} = F(E_t, M_t)$, where $E_t$ is the claim feature distribution at model generation $t$ and $M_t = \hat{f}_{v(t)}$ is the deployed model. The loop is "hidden" when standard OOT AUC is not a monotone function of the true performative risk — precisely the Insurance A Cop. situation where v2a's OOT AUC looked acceptable while the scrap rate inflated.

**Convergence condition (P29 — Mendler-Dünner et al. 2020):** Let $\varepsilon$ be the distribution sensitivity (Wasserstein distance shift per unit change in model parameters) and $\beta$ the strong convexity constant of the XGBoost loss. The loop converges to a biased-but-stable fixed point when $\varepsilon / \beta < 1$; it diverges (runaway amplification, as in P12 Pólya urn) when $\varepsilon / \beta \geq 1$. Build 02 estimates this ratio empirically from the v1→v2a score distribution shift as a single-number **SFP loop severity score**.

**Loop type classification (P30 — Pagan et al. 2023):** The total loss loop is a **positive-gain, short-delay amplifying feedback loop** in the control-theory taxonomy — the forced label is applied immediately at the scrapping decision (not after a lag), and the gain is positive (higher score → more scrapping → more forced-positive labels → higher score next generation). This type provably converges to a maximally biased fixed point, consistent with Build 01's simulation results.

**Echo chamber vs. filter bubble (P32 — Jiang et al. AIES 2019):** The loop has two separable components. The *echo chamber* is the model amplifying its own past decisions (rising cross-version Spearman rank correlation — Build 02 Step 1). The *filter bubble* is the model becoming blind to true repairability of certain vehicle segments as their scrap rate approaches 100% (segment blind spots — Build 02 Step 4). Formal degeneracy condition: if the Jacobian spectral radius of the v1→v2a score-mapping exceeds 1 in a score band, that band is on a diverging path toward information collapse. These two signals together constitute the full observable signature of the SFP loop and motivate Steps 1 and 4 of Build 02 SFPDetector as distinct, complementary tests.

**Stateful performativity and why v3 retraining failed (P33 — Brown, Hod & Kalemaj, AISTATS 2022):** P15's map D(θ) depends only on the current model θ. P33 extends this to D(θ, s_t), where s_t is the *accumulated state* of forced-positive labels across all prior model versions. State evolution: s_{t+1} = g(s_t, θ_t). Convergence to a good equilibrium requires the state sensitivity ε_s (how much the contamination state shifts per model generation) to satisfy a joint contraction condition with ε_θ. The failure of v3 retraining is consistent with this: after v1 and v2a both operated under the 0.872 threshold, the contamination state s was sufficiently entrenched that retraining on s-contaminated labels could not converge to the performatively optimal classifier. v2b's partial resistance to SFP is also explained — including pre-ML labels in training effectively reduces ε_s by partially resetting the contaminated state. Build 01's simulation of v1→v2a→v3 trajectories should model the stateful dynamics explicitly, not just single-step transitions.

### 2.4 Problem Type Taxonomy — Which Existing Framework Fits?

→ Moved to [`literatures/reading_list.md` — Problem Type Taxonomy section](literatures/reading_list.md#problem-type-taxonomy--which-existing-framework-fits).

### 2.5 Additional Difficulties Created by Operational Constraints

1. **Precision ≥ 0.985 constraint:** Any mitigation technique (randomisation, threshold adjustment, etc.) that violates this constraint is commercially non-viable. A simple "explore more" solution cannot be proposed without a cost-benefit analysis.
2. **Absolute threshold (not percentile):** Because the scrap rate is a function of the score distribution, score drift alone can shift the scrap rate — when detecting the loop, one must distinguish whether "increasing scrap rate" is a genuine loop signal or merely distributional shift.
3. **No calibration:** Any correction technique that uses $S_i^v$ as a propensity score (IPS weight) may be distorted by this lack of calibration — affects all IPW/PSM-class methods used in Build 03/06.
4. **Permanent absence of oracle $Y_i$:** Scrapped vehicles are physically destroyed, so post-hoc audits cannot recover $Y_i$. This is a stricter constraint than the typical judicial/medical selective-labels setting covered by P27 (where, for example, bail-denied individuals can be observed later).
5. **Contamination of the OOT holdout itself:** The OOT period (most recent 6 months) is drawn from logs in which v1 was already in production, so OOT evaluation may itself be contaminated by $\tilde{Y}^{v1}$ — the assumption that "we validated on future data, so we're safe" may not hold.
6. **ENOL/FNOL channel-mix shift as a confound (confirmed 2026-06-22):** Claims are filed either by phone (FNOL — First Notification of Loss) or online (ENOL — Electronic Notification of Loss, recently introduced). The two channels produce structurally different feature distributions: FNOL claims are handler-mediated and may capture more severe incidents (claimants in worse situations tend to phone rather than self-serve online), while ENOL claims follow a fixed form path with different missing-value and error patterns. If the proportion of ENOL claims grows over time as online filing becomes more common, this induces a feature distribution shift that is independent of any SFP loop. Score drift attributed to SFP amplification must therefore be checked against the ENOL/FNOL mix change as an alternative explanation. See `README.md` § "Claim Intake Channels — FNOL vs ENOL" for the full operational context.
7. **Model environment incompatibility — v1 cannot share a runtime with v2/v3:** All three model versions (v1, v2, v3) are preserved as serialised files within Allianz's internal systems, but v1 has different library dependencies from v2 and v3 (exact versions not yet documented — likely a different XGBoost or scikit-learn version). v2 and v3 share the same environment. Any framework that needs to load and compare model outputs across versions must isolate v1 in a separate process or environment. This is not just an infrastructure concern: it constrains which detection methods can be applied interactively across model generations without manual environment switching. See `src/DESIGN.md` for the isolation strategy.
8. **Enrichment table update mechanism unconfirmed:** The enrichment table (vehicle values, part cost indices) is refreshed approximately every 6, 9, or 12 months, independently of model retraining. However, it is **not yet confirmed** whether: (a) per-ABI-code entries are static once added, (b) existing value fields are refreshed to track market prices across update cycles, or (c) only new manufacture-year rows are appended. If the real enrichment table does update existing values, then `repair_to_value_ratio` — the strongest DGP predictor — can shift for the same vehicle across training windows purely due to enrichment changes, independently of any SFP loop. This is a confound for SFP score drift detection that cannot be resolved without confirming the update mechanics with the Allianz data team. See `README.md` § "Enrichment Table — Update Cycle and Open Questions" for full detail.

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

*First written: 2026-06-16. Last updated: 2026-06-23 (§2.3 extended with P33 Brown et al. AISTATS 2022 stateful performativity D(θ,s_t) and explanation of v3 retraining failure; P32 Jiang et al. AIES 2019 echo chamber / filter bubble distinction and degeneracy condition; P31 Veprikov et al. 2025 repeated-learning map E_{t+1}=F(E_t,M_t); P29 Mendler-Dünner et al. 2020 convergence condition ε/β<1 as loop severity score; P30 Pagan et al. 2023 control-theory classification as positive-gain short-delay amplifying loop. Previous update 2026-06-22: enrichment table extended with vehicle physical specs — BHP, acceleration, gears, kerb weight, height — per Luna meeting 2026-06-22; §2.1 $X_i$ notation updated; v3 section extended with Control Expert replacement context and early-2027 implementation timeline; §2.5 difficulty 6 added — ENOL/FNOL channel-mix shift as a confound for SFP score drift detection; `used_car_price_index` added to synthetic data enrichment step). Source: `README.md` (verbatim transcription, §1), direct formalisation (§2). Working assumptions may be updated as further papers are read. Cross-references: [`literatures/notes/p27.md`](literatures/notes/p27.md) (basis for selective-labels candidate evaluation in §2.4 table), `src/data/synthetic/synth_data_structure.md`, `src/data/synthetic/generate/model.py`.*