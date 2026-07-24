# Experiment Design — SFP Loop Evaluation

> Governs how SFP detection and evaluation experiments are set up, for **both** the synthetic
> framework (current) and real Allianz UK production data (on arrival). §2 records the design
> *principles* that hold regardless of data source. §3 is the side-by-side of how synthetic and
> real data differ in practice. §4 / §5 hold the source-specific details, and §6 the real-data
> clarifications still outstanding.
>
> When real data arrives, swap the parquet files in `src/data/synthetic/` and apply §5.
>
> *Companion documents: `README.md` and `problem.md` (canonical business + problem spec, the
> source of truth for every figure below), `src/data/synthetic/synth_data_structure.md`,
> `src/data/synthetic/eval_design.md`,
> `notebook/02_01_p29_sfp_detection_dynamical_systems.ipynb`,
> `notebook/00_feature_drift_EDA.ipynb` (true-TL base-rate drift confound, §2.4).*

---

## 1. Scope

- **Synthetic (now):** `claims_all.csv` (70,000 rows, 2016–2024) carries scores from five model
  versions — v1, v2a, v2b, v3a, v3b — each scored retroactively on **every** row regardless of
  its own training window. This is the placeholder framework the Builds are developed against.
- **Real (on arrival):** the production log (parquet + DB tables) replaces the synthetic files.
  Real data has no garage oracle, disposed pre-ML labels, GDPR retention limits, and model
  versions trained at non-aligned cutoffs. The principles in §2 still apply; the constraints in
  §5 take over the source-specific decisions.

---

## 2. Shared Design Principles

These hold for synthetic and real data alike. The recurring theme: **what is valid for tracking
that a loop *exists* (symptom analysis) is more permissive than what is valid for *quantifying* it
(performance metrics, the p29 ψ_t trajectory).**

### 2.1 Evaluation dataset — symptom analysis vs performance metrics

A model version's scores exist on rows that were used to train that same version (synthetic:
`claims_all.csv` covers all 70k rows; real: the log spans each model's own training window).
Using those rows depends on the goal:

| Use case | Training rows OK? | Reason |
|---|---|---|
| **SFP symptom analysis** — score drift, decision-rate inflation, label-mechanism bias | **Yes** | Comparing version scores on the *same* claims is required to trace how the loop evolves across generations |
| **Model performance evaluation** — AUC, precision, recall, calibration | **No — leakage** | Must restrict to each model's out-of-time (OOT) holdout window |

**Rule:** the full dataset is appropriate for SFP *pattern* analysis only. For any performance
metric, restrict to the relevant OOT window (synthetic windows in §4; real windows in §5).

### 2.2 Leakage-free common cross-version window (the p29 ψ_t mapping)

`notebook/02_01_p29_sfp_detection_dynamical_systems.ipynb` maps the FTTL retraining loop onto Veprikov et al. (2024)'s
discrete dynamical system **f_{t+1} = D_t(f_t)**, with the time axis t = **model versions**
(pre-ML → v1 → v2a → v3a). The scalar loop summary is **ψ_t = f_t(0)**, the residual density at 0.

**Problem.** For the ψ_t trajectory to mean anything, the *only* thing allowed to change across t
is the retraining operator D_t. If each version were scored on a different population (its own
window), differences in ψ_t could come from case-mix / market drift rather than the loop — exactly
the failure the notebook flags in E6 (applying a fixed model across calendar quarters moves the
*input* distribution f_in, not D_t, so it is not f_{t+1}=D_t(f_t)).

**Decision — the mapping is valid only under three locked conditions:**

| Condition | Why it is mandatory |
|---|---|
| **Common window** (same rows for all versions) | holds f_in fixed → ψ_t change is pure D_t signal |
| **Leakage-free** (window OOT for *every* version simultaneously) | a version scored on its own training rows has optimistically biased residuals → spurious ψ_t inflation → fake positive loop |
| **Oracle-free genuine labels** | scrapped rows carry forced label = 1; their residual is meaningless — restrict to garage-verified rows (`decision == 0`) |

**Principle vs specific dates.** The *principle* (common + leakage-free + oracle-free) is
non-negotiable: violate any one and ψ_t stops being interpretable as the D_t dynamical system. The
*specific window* is dataset-dependent — see §3 for how it differs between synthetic and real.

**Single-row leakage is not the real concern.** "A version must not be evaluated on its training
rows" is about **systematic** overlap, not a single stray row. One row out of thousands does not
move ψ_t = f_t(0). A temporal holdout sidesteps the question entirely: if the window is wholly
after every version's cutoff, overlap is zero, so there is nothing to count.

**Ties back to §2.1.** Symptom tracking tolerates training rows; the ψ_t trajectory and any
performance metric require the locked window above.

### 2.3 Enrichment / market-inflation confound

`used_car_price_index` (and any refreshed enrichment value field) is time-varying and feeds
`vehicle_value` → `repair_to_value_ratio`, the strongest DGP predictor. Market movement therefore
shifts genuine total-loss rates **independently of any SFP loop**.

Directionally, price *deflation* raises the genuine TL rate — the **opposite** direction to SFP
(which inflates the *predicted* scrap rate). The two partially offset, so a detected SFP symptom
on top of deflation is a **conservative lower bound**, not an artefact. The real risk is the
reverse: mis-attributing a price-driven rate change to SFP.

**Required controls (both data sources):**
1. Include `used_car_price_index` as a control when comparing decision rates across time bands;
   never compare raw scrap rates across year bands without adjusting for it.
2. When stratifying OOT vs training results, check the gap is consistent across price-index bins —
   a gap that holds at constant index is cleaner SFP evidence than one concentrated at extremes.
3. Document the confound explicitly; the index is an observed model feature, so partial
   self-correction occurs at inference, but it was not designed as a debiasing mechanism.

### 2.4 Cross-version signal reliability — read the gap, not the level

The market/enrichment confound of §2.3 is **not the only** thing that can move
`mean(S^{v2}) − mean(S^{v1})` with no SFP loop present. Three further confounds bear on any
cross-version comparison (full detail in `problem.md` §2.5 and `README.md`):

| Confound | Mechanism | Where handled |
|---|---|---|
| Market / enrichment inflation | price index shifts `repair_to_value_ratio` | §2.3 — control for `used_car_price_index` |
| **Preprocessing divergence (#10)** | same raw claim → different features under each version's own pipeline | per-version feature contract (`features_<v>.parquet`); on real data reconstructable only if the v1 artefact can be re-run (cf. §5.4) |
| **Channel-mix shift FNOL/ENOL (#6)** | case-mix changes the score distribution over calendar time | stratify by channel before attributing drift to the loop |
| **True-TL base-rate drift** | genuine TL rate drifts across OOT windows (≈16.6% at v1's 2021 window → ≈22.1% at 2024) — a feature-drift effect, **not** the loop | `notebook/00_feature_drift_EDA.ipynb`; compare **within** a fixed window |

Because of these, the candidate cross-version signals are **not** equally trustworthy — this governs
what an experiment is allowed to conclude:

| Signal | Worth | Why |
|---|---|---|
| τ_v threshold trajectory | **unreliable** | endogenous response to the contaminated data; non-monotonic (synthetic v1 ≈ 0.852 → v2a ≈ 0.906 → v3a ≈ 0.891) |
| Contaminated precision | **uninformative** | pinned near 0.985 by construction (tuning targets it; `scrap → label 1` makes it easy) — it absorbs the drift and hides the harm |
| Scrap-rate inflation | **symptom, confounded** | rises (~12.5% → ~18.6%) but the base-rate drift above can explain part of it — flags a symptom, does not prove harm |
| **Contaminated − oracle precision gap** | **dispositive** | directly measures the SFP harm (repairable cars wrongly scrapped): **0.002 → 0.008 → 0.016** across v1→v2a→v3a. **Synthetic-only** (`verify_sfp_oracle.py`); on real data Build 03 must *estimate* it via IPS / selective-labels because no oracle exists |

**Design consequence.** Within a fixed OOT window the contaminated−oracle gap isolates the loop; the
oracle-precision *level* is **not** comparable across versions (the windows differ in true-TL base
rate). Any quantitative SFP claim rests on the gap (or its IPS estimate), never on scrap-rate or
contaminated-precision alone. This is also what the p29 ψ_t trajectory (§2.2) operationalises: with
f_in and the label set held fixed, a rising ψ_t = f_t(0) is the residual-density face of the same
widening gap.

---

## 3. Synthetic vs Real — Side by Side

| Aspect | **Synthetic** | **Real (Allianz production)** |
|---|---|---|
| Source | `claims_all.csv`, 70k rows, all 5 versions scored on every row | Production log (parquet + DB); each version scored over its own operating period |
| ML training cutoffs | **all aligned at 2024-04-30** (generator forces `end_year=2024` for v2a/v2b/v3a/v3b) | **not aligned** — v3 trained 2025 on 2023+ data, ~1y later than v2a |
| Common leakage-free window | May–Oct 2024 (~4,051 rows) — OOT for **all five** versions at once | must sit **after v3's later cutoff** → late-2025/2026, **much narrower** |
| Train/eval overlap in that window | **exactly zero by construction** (window wholly after all cutoffs) | zero only within the narrow v3-onward window; wider windows reintroduce v3 leakage |
| Oracle (`garage_outcome`) | exists in the generator but **not used** (oracle-free constraint enforced) | **permanently absent** — scrapped cars destroyed, never garage-verified |
| Genuine-label rows for ψ_t | garage-verified rows (`v1_decision==0`), ~3,296 in the window | garage-verified subset of the narrow window — potentially very small |
| Labels | simulated forced-positive mechanism | operationally contaminated (forced label = 1 for scrapped); maturation longer than 2m |
| Re-scoring all versions | one per-version **pipeline pickle** → `predict_proba(raw)`; the synthetic pipelines are identical by construction (agreed direction — `run.py::export_version_features` still writes feature parquets until the refactor lands) | `clone repo → build env-vX → joblib.load(vX_pipeline.pkl) → predict_proba(raw)`; **each version needs its own pinned env** (incompatible numeric stacks, cf. §5.4) and the decommissioned **v1 artefact must still run** |
| Dates | fixed and known | v1/v2 deploy dates, log start (~2018), v3 cutoff all **TBC** |
| Practical ψ_t comparison | v1→v2a→v3a directly on the common window | often restrict to **v1 ↔ v2a** (wider window); v3 not deployed |

**Bottom line:** the synthetic data satisfies §2.2 automatically because all cutoffs are aligned.
Real data does not — the binding constraint becomes the latest-trained model (v3, 2025), so the
common leakage-free window shrinks and the pragmatic fallback (v1 ↔ v2a) is usually preferred.

---

## 4. Synthetic-Specific Details

### 4.1 When is `claims_all.csv` safe as an evaluation set? (instantiates §2.1)

Every version's scores exist on its own training rows — e.g. v2a was trained on 2022–Apr 2024 but
`model_v2a_score` is present for all 70k rows including that window. Evaluating v2a performance
there is leakage. Use the full file for SFP pattern analysis only; for performance metrics use the
per-model OOT windows below.

### 4.2 Cutoff alignment → a five-version common OOT window

`_train_cutoff()` in `generate/model.py` computes
`cutoff = (end_year Dec 31) − (MATURATION_BUFFER 2m + OOT 6m)`. Because v2a, v2b, v3a and v3b all
use `end_year = 2024`, every ML version terminates training at the **same cutoff, 2024-04-30**:

| Version | Training + test (80/20) | OOT holdout | Train cutoff | Window 2024-05→10 |
|---|---|---|---|---|
| v1 | 2016-01 → 2021-04 | 2021-05 → 2021-10 | 2021-04-30 | OOT |
| v2a (real scenario) | 2022-01 → 2024-04 | 2024-05 → 2024-10 | 2024-04-30 | OOT |
| v2b (research only) | 2020-01 → 2024-04 | 2024-05 → 2024-10 | 2024-04-30 | OOT |
| v3a / v3b | 2023+ → 2024-04 | 2024-05 → 2024-10 | 2024-04-30 | OOT |

The **May–Oct 2024 window (~4,051 rows)** is therefore simultaneously OOT for all five versions,
with **zero train/eval overlap by construction** — no per-row filtering needed. This is the time
axis t for the p29 ψ_t mapping (§2.2). Maturation buffer (excluded): Nov–Dec of the cutoff year.

**Threshold is tuned per version, not fixed at 0.872.** Each version tunes its own absolute cutoff
τ_v on a validation slice to hold precision ≥ 0.985 against its (contaminated) training label
(`model.py::_tune_threshold`), so the synthetic cutoffs come out per-version (v1 ≈ 0.852,
v2a ≈ 0.906, v3a ≈ 0.891) — `0.872` is only the documented **v2 real-world anchor / fallback**, not
a universal constant. τ_v is thus endogenous and is **not** itself a usable SFP signal (§2.4).

> **Contaminated labels in the window:** labels are `model_v1_observed_outcome`; scrapped cars
> carry forced label = 1 (never garage-verified). Standard AUC against this label is biased upward
> for models mimicking v1's scrap decisions. **Build 03 (Unbiased Evaluation) must apply
> selective-labels (IPS) correction before any cross-version performance comparison.** For the
> oracle-free ψ_t computation, the notebook instead restricts to garage-verified rows.

### 4.3 `used_car_price_index` across the synthetic timeline (instantiates §2.3)

| Year | Index | Market context |
|---|---|---|
| 2016 | 1.00 | Baseline |
| 2020 | 0.97 | COVID-19 demand drop |
| 2021 | 1.18 | Post-lockdown rebound |
| 2022 | 1.28 | Semiconductor shortage peak |
| 2023 | 1.18 | Normalising |
| 2024 | 1.10 | Further normalisation — OOT period |

v2a's training window (2022–Apr 2024) straddles the price peak and partial recovery; the OOT
(May–Oct 2024) sits at index ≈ 1.10, **below** the training-period average. Per §2.3 the resulting
deflation moves the DGP opposite to SFP, making any detected symptom conservative. Using the most
recent data as the OOT holdout is therefore methodologically sound — it also mirrors the real
setting where enrichment tables refresh independently of retraining.

---

## 5. Real-Data-Specific Constraints

> Apply once real Allianz UK production data access is granted.

### 5.1 Threshold — v2's τ_v, constant at 0.872

`0.872` is specifically **v2's tuned τ_v** (v1 and v3 were tuned to different values, per §4.2 /
`README.md`); on real data v2 is the live model, so its cutoff is the one that matters here.
The production scrap threshold was briefly changed away from 0.872 (exact value/dates
unconfirmed); performance degraded and it was promptly reverted. **For all real-data experiments
the threshold is treated as constant at 0.872 throughout.** The brief deviation is not modelled
separately and is not reflected in the decision columns. Anomalous decision patterns inconsistent
with 0.872 should be noted but not used to infer a different value. This matches Allianz's
operational understanding of the dataset.

### 5.2 Labels — contaminated, not missing

Scrapped cars (`decision = 1`) have `observed_outcome = 1` by construction — a **forced** label,
not a missing one. Do not treat these rows as unlabelled or impute them. The correct frame is
**selective labelling / label contamination** (see `problem.md` §2.2).

### 5.3 No oracle

There is no ground-truth label for any claim:
- `pre_ml_label` (handler/engineer decisions, pre-2022) disposed under data retention policy
- scrapped cars never garage-verified — true repairability permanently unknown
- all detection/evaluation methods must operate without oracle access

### 5.4 Data access constraints

| Constraint | Detail |
|---|---|
| Format | Parquet files + database tables (not CSV) |
| PII | Customer data columns require PII handling once NDA is signed |
| Retention | GDPR: data older than 8 years at model build time excluded (internal policy) |
| Pre-ML labels | `pre_ml_label` disposed — unavailable for real-data analysis |
| Model runtime | v1 has different library deps from v2/v3 (cf. `problem.md` §2.5.7) — must isolate v1 in a separate process to re-score; confirm the v1 artefact still runs |
| Exact dates | v1/v2 deploy dates and log start (~2018) approximate — to be confirmed |

### 5.5 Experiment mapping (Builds → real data)

| Build | What changes on real data |
|---|---|
| **00 — EDA** | Run on real log; check ENOL/FNOL split, missing-value patterns, per-channel score distributions |
| **02 — Loop Detection** | Same `SFPDetector` logic; the **real** ML-era scrap rates (≈19% → 21.5%, vs 15% pre-ML — README) replace the **synthetic** analogue (~12.5% → ~18.4% → ~18.6% across v1→v2a→v3a). Read the contaminated−oracle gap, not scrap rate (§2.4) |
| **03 — Unbiased Evaluation** | IPS correction on real OOT holdout; note the OOT is also SFP-contaminated |
| **04 — Intervention Analysis** | DiD / RDD on real score drift; ENOL/FNOL mix shift is the primary confound |
| **05 — Randomisation** | Cost-benefit uses real garage assessment + hire car costs (confirm with ops) |
| **06 — Causal Mitigation** | DoWhy DAG unchanged; real propensity scores replace synthetic |

### 5.6 Cross-version window — why §2.2 is tighter on real data

Split by goal (this is §2.1 / §2.2 applied to real data):

| Goal | Leakage-free common window needed? |
|---|---|
| **SFP symptom tracking** | **No** — only needs all versions scored on the same claims; training rows fine |
| **p29 ψ_t / performance comparison** | **Yes — OOT for v1, v2 *and* v3 simultaneously** |

Why tighter than simulation (see §3):
1. **Binding constraint = latest-trained model (v3, 2025 on 2023+ data).** A window OOT for v1, v2
   *and* v3 must lie after v3's cutoff → roughly late-2025 / 2026 claims, far narrower than v2a
   alone would allow.
2. **Label maturation eats into it** — recent claims may lack a finalised outcome (the synthetic
   2-month buffer is likely longer in production).
3. **Oracle-free shrinks it again** — genuine residuals exist only for garage-verified rows
   (`decision == 0`); within an already-narrow window this subset can be very small.
4. **Re-scoring must be possible at all** — comparison requires scoring v1/v2/v3 artefacts on the
   same held-out claims; confirm the decommissioned v1 model still runs (cf. §5.4).

**Practical implication.** If the goal is only to *show the loop is operating* (symptom tracking),
no leakage-free window is needed — use the full log with all versions scored on common claims. For
a *quantitative* comparison (ψ_t, IPS-corrected AUC), either (a) accept the narrow v3-onward window
with its small-n / maturation caveats, or (b) restrict to **v1 ↔ v2a**, whose common leakage-free
window opens after v2a's cutoff and is much wider. Since v3 is not deployed, (b) is usually the
pragmatic choice.

---

## 6. Outstanding Clarifications Needed (real data)

- [ ] Exact v1 and v2 deployment dates
- [ ] **v3 training cutoff date** — sets the binding constraint for any v1/v2/v3 common leakage-free window (§5.6)
- [ ] **v1 model artefact availability** — can the decommissioned v1 model still be re-run to score a held-out window? (required for cross-version ψ_t / performance comparison)
- [ ] Real log start date (approximate: ~2018)
- [ ] Threshold change — exact value it was changed to, and the date range it was active
- [ ] Garage assessment cost and hire car cost (for Build 05 cost-benefit)
- [ ] ENOL introduction date (needed to separate channel-mix shift from SFP signal)
- [ ] **Enrichment table update mechanics** — when refreshed (~6/9/12 months), what actually changes?
  - Are existing per-ABI-code value fields (`typical_market_value_gbp`, `part_cost_index`) revised to current market prices?
  - Or are only new make/model/year rows appended, existing rows unchanged?
  - Or a combination (physical specs static; value fields periodically refreshed)?
  - Matters because if values are refreshed, `repair_to_value_ratio` can drift for the same vehicle across training windows purely due to enrichment changes — a confound for SFP score drift detection (§2.3).

---

## 7. Summary

| Design question | Decision | Key constraint |
|---|---|---|
| Use full dataset / log for evaluation? | SFP pattern analysis only; not for performance metrics | Training rows cause leakage if used for AUC (§2.1) |
| Common holdout for cross-version comparison? | Synthetic: v2a OOT (May–Oct 2024, ~4k, all 5 versions). Real: after v3 cutoff, narrow — fallback v1↔v2a | Labels SFP-contaminated → Build 03 IPS correction (§4.2, §5.6) |
| What does the p29 ψ_t mapping require? | Common + leakage-free + oracle-free window (locked) | Principle mandatory; window is dataset-specific. Symptom analysis exempt; performance/ψ_t not (§2.2) |
| Which cross-version signal proves SFP? | The contaminated−oracle precision gap (0.002 → 0.008 → 0.016), within a fixed window | τ_v / contaminated precision / scrap rate each confounded; gap is synthetic-only → real data estimates it via IPS (§2.4) |
| Most recent data as OOT despite inflation? | Yes — appropriate and conservative | Control for `used_car_price_index` in all cross-time comparisons (§2.3) |
