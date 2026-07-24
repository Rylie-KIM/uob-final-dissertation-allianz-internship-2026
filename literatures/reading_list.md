# Reading List — Identifying and Mitigating Self-Fulfilling Prophecy Loops in ML
**MSc Data Science Dissertation · University of Bristol · Insurance Company. UK (Operations team)**
All claims in scope: **motor insurance claims (UK personal lines auto)**

---

## Paper Category Framework

These papers are selected to address the **motor insurance total loss prediction service's Self-Fulfilling Prophecy (SFP) Loop problem** — where the model's own scrapping decisions force labels on future training data, causing each retrained model version to amplify the previous version's over-confident predictions.

Each paper is classified into one or more of the following categories. A single paper may belong to multiple categories.

**1. Define** — Papers that explain *what* the SFP Loop problem is and *why* it occurs at a mathematical or structural level. These help the Claims Operations team understand why the refreshed model's performance degrades over successive retraining cycles.

**2. Detect** — Papers that provide a method or observable signal for detecting the SFP Loop in a deployed service. These methods are candidates for implementation in the detection framework (Build 02).

**3. Mitigate** — Papers that propose a concrete method for reducing or breaking the SFP Loop. These methods are candidates for implementation in the mitigation pipeline (Builds 05–06).

---

## SFP Loop: Fraud Detection vs. Total Loss Prediction — Domain Comparison

*This table was written to clarify domain-specific methodological choices for the paper.
The SFP loop structure differs fundamentally between the two domains — methods valid for fraud investigation cannot be transplanted directly into total loss prediction without modification.*

### Core Mechanism

| Dimension | Fraud Detection (assumed initially) | Total Loss Prediction (actual domain) |
|-----------|-------------------------------------|---------------------------------------|
| **Prediction target** | Fraud probability — is this claim fraudulent? | Repair feasibility — is this car a total loss? |
| **Consequential action** | Investigate the claim (send to specialist) | Scrap the car OR send to garage |
| **Who/what executes the action** | Human investigator (reversible: claim file persists) | Physical disposal of the vehicle (irreversible) |
| **SFP trigger** | `high score → investigate → fraud found only where looked → biased label` | `high score → scrap immediately → label forced = 1 without verification` |
| **Direction of bias in labels** | **Under-labelling** — uninvestigated claims have no fraud label; true fraud rate is underestimated in low-score segments | **Over-labelling** — scrapped cars always receive label = 1 regardless of true repair feasibility; total loss rate is inflated |
| **Direction of model drift** | v2 under-predicts fraud in segments that v1 under-investigated | v2 over-predicts total loss across the board; scrapping rate inflates over versions |

### Oracle and Ground Truth

| Dimension | Fraud Detection | Total Loss Prediction |
|-----------|----------------|----------------------|
| **Oracle definition** | True fraud status of a claim | True repair feasibility of a damaged car |
| **Oracle availability for actioned cases** | Potentially recoverable — can audit a "closed" claim file after the fact | **Permanently unobservable** — the car is physically destroyed; no audit can recover the true repair outcome |
| **Oracle availability for unactioned cases** | Available for investigated claims (label exists); missing for uninvestigated | Available for garage cases (`garage_outcome`); structurally absent for scrapped cases |
| **Ground truth recovery mechanism** | Random audit — investigate a sample of low-score claims post-hoc | Must intervene *before* scrapping — send high-score cars to garage; no post-hoc recovery path |
| **Formal framework** | Missing labels / semi-supervised learning | Selective labels (Lakkaraju et al., KDD'17, **P27**); PU learning (Bekker & Davis, 2020, **P28**) |

### Label Structure

| Dimension | Fraud Detection | Total Loss Prediction |
|-----------|----------------|----------------------|
| **Label for actioned cases** | Confirmed fraud = 1 (reliable); not confirmed = 0 or missing | Scrapped = **forced 1** (unreliable — may be repairable false positive) |
| **Label for unactioned cases** | No label (uninvestigated = unobserved; treated as 0 implicitly) | Garage outcome = **true label** (reliable 0 or 1 from actual repair assessment) |
| **Noise type** | Missing labels (one-sided: positives may be hidden in uninvestigated set) | Asymmetric forced positives (one-sided: false negatives impossible for `decision=1`; false positives possible) |
| **PU (Positive-Unlabeled) Learning applicability** | **PU Learning** = learning when you have reliably labeled positives (P) + a set of unlabeled examples (U) that contain a mix of true positives and true negatives, but you cannot tell which. — **Fraud: Weak applicability.** Uninvestigated claims simply have no label at all. They are absent from the dataset, not present with a wrong label. The training set is a biased subset, but every label in it is reliable (investigated fraud = real 1, investigated non-fraud = real 0). This is closer to semi-supervised learning than PU learning. | **Total loss: Strong applicability.** `decision=0, outcome=1` rows (garage-confirmed total losses) are the **labeled positives (P)** — these are genuine, verified positives. `decision=1` rows (scrapped cars) are the **unlabeled (U)** — they carry a forced label of 1, but their *true* repair feasibility is unknown. Some fraction $\pi$ were genuine total losses; the rest were repairable cars incorrectly scrapped. This is exactly the PU setting: reliable P from garage assessments + U with unknown true labels from the scrapping pool. The danger is that the forced label=1 makes U look like additional positives, so v2 trains as if *all* scrapped cars were confirmed total losses — the most harmful variant of PU learning (treating U as all-positive). |
| **Heckman selection direction** | Selected into observation by high model score (investigated) | Selected into observation by **low** model score (sent to garage; high-score = scrapped, never labelled truthfully) |

### Detection Signals

| Dimension | Fraud Detection | Total Loss Prediction |
|-----------|----------------|----------------------|
| **Primary observable SFP signal** | Increasing score correlation across model versions in historically investigated segments | Score drift upward (`mean(v2_score) > mean(v1_score)`); decision rate inflation (`rate(v2=1) > rate(v1=1)`) |
| **Secondary signal** | Segment blind spots — postcode/product-line groups with systematically low investigation rates | `P(observed_outcome=1 \| decision=1) = 1.0` (tautological forced positive); large gap vs. `P(outcome=1 \| decision=0)` |
| **Causal identification** | Investigation propensity as treatment variable; DiD/RDD on model version deployment | Scrapping decision as treatment variable; RDD around the **absolute score cutoff τ_v** (0.872 for v2 — *not* a percentile) is directly applicable |
| **Pólya urn analogy** | Investigation-count urn per claim segment — over-investigated segments dominate | Scrapping-count urn — once a score band is heavily scrapped, v2 learns higher scores for that band, deepening the loop |
| **OOT holdout signal** | OOT fraud labels come from actual investigation results — the model's score does not determine whether a claim has a label in the holdout. Labels are reliable (either confirmed fraud=1 or confirmed non-fraud=0, independently of the model). **OOT AUC is an unbiased estimate of true fraud detection performance.** | The v2 OOT holdout (May–Oct 2024) is a period when v1 was already in production. During this period, v1 scrapped cars it scored high → those cars received label=1 (not from a garage, but from the scrapping act itself). When you evaluate v2 on this OOT set: a claim scrapped by v1 has label=1; if v2 also scores it high, the prediction looks "correct." **But that label was created by v1's own decision, not by an independent garage assessment.** OOT AUC is rewarding v2 for imitating v1's scrapping behaviour, not for correctly identifying genuine total losses. Selective-labels-corrected AUC (**P27**) required on the OOT set too. |
| **Why P27 (Selective Labels) applies here but NOT in fraud** | In fraud, uninvestigated claims simply have **no label** (absent). You can compute AUC on investigated claims — those labels are real. The oracle is also recoverable: you could in principle audit any uninvestigated claim later (the file still exists, the person is still observable). P27's framework applies, but the damage is less severe because labels you do have are reliable, and missing ones can be recovered. | In total loss, scrapped cars have a **forced label=1** that looks like a real label but is not verified by a garage. The car is physically destroyed — the oracle is permanently unobservable. You cannot recover the true repair feasibility of a scrapped car by any post-hoc method. P27 is essential here because: (a) the forced label actively misleads the model into treating scrapped = confirmed total loss; (b) no audit path exists; (c) even the OOT evaluation is contaminated (see above). The structural irreversibility is the decisive reason P27 becomes a top-3 paper for this domain. |
| **Maturation buffer** | 2-month label lag has no structural effect on the SFP — fraud labels mature through investigation completion | 2-month exclusion is operationally necessary (repair outcomes not yet confirmed); but also means the most recent SFP signal is invisible during training. The gap between label maturation and model deployment is a window where the loop continues unobserved. |

### Mitigation Approach

| Dimension | Fraud Detection | Total Loss Prediction |
|-----------|----------------|----------------------|
| **Exploration mechanism** | Randomly investigate a fraction of low-score claims to recover missing fraud labels | Send a fraction of high-score claims to garage (instead of scrapping) to recover true repair outcomes |
| **Exploration cost** | Low-to-moderate — investigator time; claim file already exists | **High** — garage transport, storage, assessment fees; delayed settlement; potentially paying repair cost for a genuine total loss |
| **Thompson Sampling arm definition** | Claim segment (postcode, product line) — arm = under-investigated group | Score decile or vehicle category — arm = over-scrapped group needing oracle verification |
| **IPW/IPS reweighting target** | Upweight uninvestigated claims when estimating true fraud rate | Upweight garage-outcome observations when estimating true total loss rate; downweight forced-positive scrapped rows |
| **Debiased training data** | Include randomly investigated low-score claims in next training cycle | Replace forced `outcome=1` for scrapped cars with imputed counterfactual outcomes (PU imputation, **P28**) |
| **Long-run fairness concern** | Under-investigated segments (often correlated with demographics) receive systematically lower fraud scores — disparate impact (Barocas & Selbst, **P10**) | Under-scrapped vehicle types receive systematically lower total loss scores — bias against certain makes/damage profiles |

### Cost Structure (from README.md business context)

| Dimension | Fraud Detection | Total Loss Prediction |
|-----------|----------------|----------------------|
| **False positive cost** | Unnecessary investigation cost (investigator time) — relatively low | **Pay full car value** for a car that could have been repaired at lower cost — high direct financial loss |
| **False negative cost** | Missed fraud payout — high financial loss | Unnecessary garage visit + delayed settlement for a genuine total loss — moderate operational cost |
| **SFP impact on costs** | SFP → missed fraud in under-investigated segments → financial loss accumulates silently | SFP → over-scrapping → more false positives → **systematic increase in unnecessary total loss payouts** |
| **Model purpose (business)** | Prioritise limited investigation resources toward highest-risk claims | Speed up settlement for genuine total losses; avoid costly garage visits for cars that will be declared total loss anyway |
| **When model fails (v2 underperformance)** | Fraud slips through uninvestigated | Repairable cars get scrapped → full value payout instead of repair → direct cost inflation |

### Key Methodological Implications for the Paper

| Research Question | Fraud Detection | Total Loss Prediction — what changes |
|------------------|----------------|--------------------------------------|
| How to detect the SFP loop? | Look for score inflation in historically investigated segments | Look for score inflation + decision rate inflation across versions; tautological label check (`P(outcome=1\|decision=1)=1.0`) |
| How to evaluate model performance without bias? | IPS-weighted AUC over all claims (upweight uninvestigated) | IPS-weighted AUC over garage observations only — **selective-labels-corrected AUC** (**P27**) |
| How to estimate the treatment effect of the action? | DiD: model deployment date as natural experiment; RDD: investigation threshold | RDD: the **absolute scrapping cutoff τ_v** (0.872 for v2) is a **natural sharp regression discontinuity** — near-identical claims just above/below threshold |
| How to debias training data? | Add randomly investigated claims; IPW re-weighting | PU class-prior estimation for scrapped rows; IPW for garage rows; counterfactual imputation (**P27, P28**) |
| How to break the loop going forward? | Random investigation of low-score claims (cheap exploration) | Deliberate garage routing of high-score claims (costly exploration — must model cost-benefit explicitly) |
| What paper is the formal backbone? | Ensign et al. (**P12**) + Perdomo et al. (**P15**) | All of the above **plus** Lakkaraju et al. (**P27**) for oracle-absence + Bekker & Davis (**P28**) for forced-positive labels |
| How to handle OOT evaluation under SFP? | Standard OOT AUC is valid — labels are eventually confirmed | OOT labels are SFP-contaminated; apply selective-labels-corrected AUC (**P27**) to OOT set. Also: 2-month maturation buffer creates a blind spot at deployment boundary — the most recently SFP-affected data is invisible during both training and OOT evaluation. |

---

## Document Status

> **⚠️ Domain correction — first made 2026-06-15; language migration completed 2026-07-07**
>
> Originally drafted with a **fraud detection / investigation-based SFP** assumption. After reviewing `src/data/synthetic/synth_data_structure.md` (business logic confirmed via internal meetings), the actual domain is total loss prediction. **As of 2026-07-07 the fraud → total-loss language migration flagged below has been applied throughout this document** (all individual paper "How to apply / What to write" sections), and this file is now aligned with the canonical `README.md` and `problem.md`. The Fraud-vs-Total-Loss comparison tables above are retained *deliberately* — they document why the domain shift changes the methodology; they are not a leftover of the old framing.

| Field | Detail |
|-------|--------|
| **Confirmed domain** | **Total Loss Prediction** — model predicts whether a damaged car should be scrapped (`total_loss=1`) or sent to garage (`total_loss=0`). **Not fraud detection.** |
| **Actual SFP mechanism** | `total_loss=1` → car scrapped immediately → label forced to 1 (self-fulfilling). `total_loss=0` → garage → true repair outcome observed. |
| **Structural difference from fraud** | Oracle (`garage_outcome`) is **permanently unobservable** for scrapped cars — the car is physically gone. This is a **selective labels** problem (→ P27), not an investigation-bias problem. |
| **Label noise structure** | `decision=1` rows always receive label 1 (forced positive; true repair outcome unknown). `decision=0` rows receive true labels. This asymmetric noise is the SFP mechanism. |
| **Language note — ✅ DONE 2026-07-07** | "fraud label" → "total loss label"; "investigation" → "scrapping decision"; "investigated claims" → "scrapped / sent-to-garage"; "postcode risk" → "vehicle make / damage profile / `repair_to_value_ratio`" — **now applied throughout every "How to apply / What to write" section** (previously only flagged as a TODO). |
| **Pre-ML baseline & class prior α — NEW 2026-07-07 (README §Pre-ML Baseline / problem.md §1.1a)** | Pre-ML human era: **15% of all cars scrapped**, of which **43% were handler fast-tracked with no garage visit** (forced, unverifiable labels in `pre_ml_label`). This pins the **true total-loss rate (class prior α = P(y=1)) to a sharp bound α ∈ [8.55%, 15%], point estimate ≈ 15%** — sharp because the 93.55% garage-observed region has structurally zero missed-TL error; all uncertainty is in the 6.45% fast-track slice. **Calibration anchor:** pre-ML scrap rate (15%) ≈ α ⇒ the human era was well-calibrated *before* contamination; the SFP fingerprint is the post-deployment scrap rate inflating *above* α (real ≈ 19% → 21.5%; synthetic analogue ~12.5% → 18.4% → 18.6% across v1→v2a→v3a). α is the *marginal* prior, distinct from scrap precision π_scrap = P(y=1 \| scrap). |
| **Two nested SFP loops — NEW, supervisor decision 2026-07-06** | The FTTL problem is **two nested loops**: (1) a **human-based loop** embedded in `pre_ml_label` — call handlers write cars off without a garage inspection, producing forced labels — and (2) the **model-based forced-label loop** (§ P15/P27) that v1 inherited and each version amplifies. **Only the garage engineer's physical assessment is ground truth; the call handler is treated *unconditionally* as a biased data generator, never as a weaker oracle** (lacks the engineering knowledge to judge repair feasibility). v1's purpose was to *outperform* the handler, yet it was trained on handler-generated labels. |
| **Positivity / SCAR violation — NEW (problem.md §2.6, P28)** | The scrapping decision `D = 𝟙[f(X) ≥ τ_v]` is a **deterministic step function of X**, so the labelling propensity `e(x) = Pr(D=1 \| X, Y=1) ∈ {0,1}` — **SCAR is violated by construction** and **positivity/overlap fails** wherever `e(x)=1` (high-score cars are always scrapped → no garage counterfactual). Consequences: SCAR-based PU prior estimators are invalid; IPS weights are degenerate exactly at the threshold, so Build 03/06 use the (uncalibrated) model score as a *soft* propensity — a documented limitation. The saving grace vs. generic SAR is that the mechanism is *known* (the model + τ_v), so `e(x)` is computable rather than estimated. |
| **Per-version threshold τ_v — NEW (README/problem.md §1.4)** | The scrap policy is an **absolute score cutoff, not a percentile**, but the cutoff is **not a fixed constant of the pipeline** — it is the *output* of a **mandatory per-version threshold-tuning step**: for each new model version the team sweeps the score cutoff and picks the smallest `τ_v` that still holds **precision ≥ 0.985** on validation (against the contaminated label). Precision ≥ 0.985 is the *binding constraint*; `τ_v` is whatever value delivers it for that version's score distribution. **`0.872` is specifically the value this tuning produced for v2** — it is not portable to v1/v3, which have their own tuned cutoffs (synthetic tuned values v1 ≈ 0.852, v2a ≈ 0.906). So "the threshold" always means "the precision-≥-0.985 cutoff **tuned for that version**", and a bare "0.872" is shorthand for "v2's tuned cutoff", never a universal number. This is why score drift shows up as scrap-rate inflation (a percentile rule would hide it), why the SFP fingerprint is visible even though precision is *held fixed by construction* (the constraint is satisfied by re-tuning τ_v each version, masking drift in the metric), and why v2's cutoff is a *sharp RDD*. |
| **New papers P29–P35** | P29–P33 added 2026-06-23 (mathematical SFP-evaluation gap); P34–P35 added 2026-07-07 (model-class-agnostic amplification — the non-convexity defence for XGBoost). **Numbering corrected 2026-07-07: P29 = Veprikov (dynamical systems), P31 = Mendler-Dünner (convergence)** — previously swapped in this file and in `literatures/compare.md`. |
| **Priority changes** | P27 (Selective Labels — NEW, KDD'17) enters **#3**; P6 (Heckman) rises to **#4**; P28 (PU Learning survey — NEW) enters **#8**; P11 (Lum & Isaac) drops to **#6**; P21, P7, P5 drop out of top 10 |
| **Business cost structure (README)** | False positive = scrapping a repairable car → insurer pays **full car value** (vs. lower repair cost). SFP loop deepening → systematic false positive increase → direct financial loss. Company prefers repairable claims. v2 underperformed → SFP suspected → project parked (also superseded by the Control Expert acquisition, integration expected early 2027). This reframes SFP as a **cost containment failure**, not just a technical bias problem. |
| **Model training methodology (README)** | **Maturation buffer**: last 2 months of data excluded from training (labels not yet fully confirmed). **OOT holdout**: last 6 months of non-excluded data held out temporally. **Train/test**: 80/20 random split on remaining data. Critical SFP implication: the OOT holdout is drawn from the v1 log period — its labels are already SFP-contaminated (`model_v1_observed_outcome`). Standard OOT AUC is therefore also a biased metric under selective labels (P27). |

---

Papers are ordered historically to show how the concepts evolved — from causal inference foundations through econometric quasi-experimental methods, feedback-loop theory, causal ML, exploration strategies, fairness regulation, and finally the domain-specific label-observability frameworks that are unique to the total loss prediction context.

---

## Top 10 Core Papers — Read These First

Ranked by how central each paper is to the dissertation's argument and implementation.
If you only have time for 10, read these in this order.

| Rank | Paper | Why it's #N |
|------|-------|-------------|
| **1** | **P15 · Perdomo et al. (2020) — Performative Prediction** · [arXiv:2002.06673](https://arxiv.org/abs/2002.06673) | The formal mathematical definition of the entire dissertation concept. Introduces performative risk $PR(\theta) = \mathbb{E}_{z \sim D(\theta)}[\ell(z;\theta)]$ — the gap between training-time and deployment-time objectives. Every other paper connects to this one. |
| **2** | **P12 · Ensign et al. (2018) — Runaway Feedback Loops** · [arXiv:1706.09847](https://arxiv.org/abs/1706.09847) | The mathematical proof that prediction-driven scrapping decisions *provably* converge to a biased fixed point (Pólya urn). Build 01 simulates this; Build 02 detects it. The "FAT'18" paper in the project plan. |
| **3** | **P27 · Lakkaraju et al. (2017) — The Selective Labels Problem** · [ACM DL](https://dl.acm.org/doi/10.1145/3097983.3098066) | **NEW — elevated to #3.** The total loss domain is a textbook selective-labels problem: repair outcomes are observable only for cars sent to garage (`decision=0`); for scrapped cars (`decision=1`) the outcome is permanently unobservable. This paper provides the exact formal framework for evaluating and correcting models under this structural constraint. |
| **4** | **P6 · Heckman (1979) — Sample Selection Bias** · [DOI](https://doi.org/10.2307/1912352) | **Elevated from previous list.** In the total loss domain, partial observability of repair outcomes is more severe than in the fraud analogue: scrapped cars never reach a garage, so the oracle is structurally absent. Heckman's correction is the direct precursor to IPW debiasing in Build 06. |
| **5** | **P4 · Rosenbaum & Rubin (1983) — Propensity Score** · [DOI](https://doi.org/10.1093/biomet/70.1.41) | Was #4. Statistical engine of Builds 04 and 06. Propensity of being scrapped (vs. sent to garage) — conditioned on `repair_to_value_ratio`, `damage_severity`, `vehicle_age_years` — is the key confound to control. |
| **6** | **P11 · Lum & Isaac (2016) — To Predict and Serve?** · [DOI](https://doi.org/10.1111/j.1740-9713.2016.00960.x) | Was #3. Still the closest empirical analogue — replace patrol with scrapping decision, drug arrests with total-loss labels. Note: analogy now maps to *irreversible operational decisions* rather than *investigation*, so update framing accordingly. |
| **7** | **P16 · Chernozhukov et al. (2018) — Double/Debiased ML** · [arXiv:1608.00060](https://arxiv.org/abs/1608.00060) | Was #5. Cross-fitted propensity scores for the scrapping decision; valid causal effect estimates despite high-dimensional vehicle features (`vehicle_make`, `damage_severity`, `repair_to_value_ratio`). |
| **8** | **P28 · Bekker & Davis (2020) — Learning from Positive and Unlabeled Data** · [arXiv:1811.04820](https://arxiv.org/abs/1811.04820) | **NEW — enters #8.** The label structure for scrapped cars (forced 1, true repair outcome unknown) maps to PU learning: we observe confirmed total losses from garage outcomes (`decision=0, label=1`) but cannot verify labels for scrapped cars. Provides theory for learning under asymmetric label observability. |
| **9** | **P3 · Horvitz & Thompson (1952) — IPS Estimator** · [DOI](https://doi.org/10.1080/01621459.1952.10483446) | Was #6. Mathematical engine behind Build 06's debiasing — reweighting garage-outcome observations back to the full claims distribution by the inverse probability of being sent to garage (vs. scrapped). |
| **10** | **P13 · Corbett-Davies et al. (2017) — Cost of Fairness (KDD '17)** · [arXiv:1701.08230](https://arxiv.org/abs/1701.08230) | Formalises the exploitation-exploration trade-off. Directly motivates why Build 05 uses randomisation (sending some high-score cars to garage to verify true repair outcome) rather than pure exploitation. The "KDD'17" paper in the project plan. |

> **Reading strategy:** Read P15 → P12 → P27 in sequence first (2–3 hours total). That gives the full problem framing — performative prediction, feedback loop mechanics, and the selective-labels structural constraint. Then read P6 + P4 + P16 as a block (the estimation toolkit). Then P28 + P3 for label noise and debiasing. P11 and P13 can be read last as contextual anchors.

---

## Part 1 — Causal Inference Foundations (1950s–1990s)

---

### P1 · Neyman (1923/1990) — Potential Outcomes Framework

**Citation**
Neyman, J. (1923/1990). "On the Application of Probability Theory to Agricultural Experiments: Essay on Principles." *Statistical Science*, 5(4), 465–472 (English translation).

**Link** → https://doi.org/10.1214/ss/1177012031
**Citations** ≈ 3,000+ (Google Scholar) · **Journal** *Statistical Science* (IMS flagship, IF ≈ 5)

**Why this paper matters**
Every causal claim in the dissertation rests on the potential-outcomes (PO) notation introduced here. Without PO, there is no rigorous way to say "what would the true repair outcome have been had this car been sent to a garage instead of scrapped?"

**Summary**
Neyman introduced the notation Y(1) and Y(0) for the outcome a unit *would* have under treatment and control. The Average Treatment Effect (ATE) = E[Y(1) − Y(0)] is the quantity the dissertation is ultimately trying to estimate when asking: does the scrapping decision *cause* the total-loss label (forcing observed outcome = 1) rather than merely reflect a genuine total loss?

**Key concept / formula**
$$\tau = \mathbb{E}[Y_i(1) - Y_i(0)]$$
The fundamental problem of causal inference: we observe at most one potential outcome per unit. All identification strategies (IPW, DiD, RDD) are solutions to this problem.

**How to apply at Insurance Company.**
Frame every analysis as a PO problem: for each damaged car, define Y(1) = observed outcome if scrapped (structurally forced to 1, oracle destroyed), Y(0) = true repair outcome if sent to the garage. The fundamental problem is acute here — for scrapped cars Y(0) is *permanently* unobservable. Unobserved counterfactuals are estimated via the methods in Builds 04–06.

**What to write in the dissertation**
Use this paper in the causal framework chapter (Build 02 background) to formally justify why naive accuracy metrics are biased. Cite as the origin of the PO framework.

**Additional note**
The paper was in Polish until the 1990 translation; cite the 1990 *Statistical Science* version unless discussing the history of statistics.

---

### P2 · Rubin (1974) — Estimating Causal Effects

**Citation**
Rubin, D. B. (1974). "Estimating Causal Effects of Treatments in Randomized and Nonrandomized Studies." *Journal of Educational Psychology*, 66(5), 688–701.

**Link** → https://doi.org/10.1037/h0037350
**Citations** ≈ 9,800 (Semantic Scholar) · **Journal** *Journal of Educational Psychology* (IF ≈ 5)

**Why this paper matters**
Rubin operationalised Neyman's notation into a usable framework for observational data and formalised SUTVA — the assumption that one car's scrapping decision does not affect another car's outcome. This is the "Rubin" in Neyman–Rubin.

**Summary**
Defines the Stable Unit Treatment Value Assumption (SUTVA): no interference between units and no hidden versions of treatment. Introduces ignorability (unconfoundedness): treatment assignment is independent of potential outcomes given observed covariates. If ignorability holds, we can estimate ATE from observational data.

**Key concept / formula**
SUTVA: $Y_i = Y_i(W_i)$ — each unit's outcome depends only on its own treatment.
Ignorability: $(Y(0), Y(1)) \perp W \mid X$ — conditional on features, who gets scrapped is "as good as random." (In FTTL this is violated by construction — the scrapping decision is a deterministic step function of the score — so unconfoundedness must be argued on the pre-decision covariates, not assumed.)

**How to apply at Insurance Company.**
SUTVA is plausible for total-loss claims (one car being scrapped should not directly affect another car's repair outcome). Ignorability is the key assumption to defend in Build 04 — argue that `repair_to_value_ratio`, `damage_severity`, and `vehicle_age_years` are sufficient to satisfy it (conditional on these, which cars get scrapped is "as good as random").

**What to write in the dissertation**
Cite in the identification strategy section. State the SUTVA assumption explicitly, explain why it is reasonable for total-loss claims, and discuss what might violate it (e.g., a shared enrichment table or salvage-market conditions that couple scrapping decisions across cars, or a threshold change that shifts the whole score distribution at once).

---

### P3 · Horvitz & Thompson (1952) — Inverse Probability Weighting

**Citation**
Horvitz, D. G. & Thompson, D. J. (1952). "A Generalization of Sampling Without Replacement from a Finite Universe." *Journal of the American Statistical Association*, 47(260), 663–685.

**Link** → https://doi.org/10.1080/01621459.1952.10483446
**Citations** ≈ 5,000+ (Google Scholar) · **Journal** *JASA* (flagship ASA journal, IF ≈ 5)

**Why this paper matters**
The Horvitz–Thompson estimator is the mathematical engine behind Inverse Probability Weighting (IPW) — the core technique in Build 06 (causal mitigation via DoWhy). Every IPW claim in the dissertation traces back here.

**Summary**
When units are sampled with unequal probabilities, naively averaging observed outcomes gives a biased population estimate. Horvitz and Thompson showed that dividing each unit's outcome by its sampling probability gives an unbiased estimator of the population total.

**Key concept / formula**
$$\hat{\mu}_{IPW} = \frac{1}{n} \sum_{i=1}^{n} \frac{W_i \cdot Y_i}{\hat{e}(X_i)} + \frac{(1 - W_i) \cdot Y_i}{1 - \hat{e}(X_i)}$$
where $\hat{e}(X_i) = P(W_i = 1 \mid X_i)$ is the propensity score — here the estimated probability of being **sent to the garage** (i.e. *not* scrapped). Upweights garage observations that were unlikely to be observed (high-score, near-threshold cars); downweights routinely-garaged low-score cars.

**How to apply at Insurance Company.**
Build 06 re-weights each garage-observed car by the inverse of its propensity to be sent to the garage. Cars with high model scores that were nonetheless sent to garage (rare, just below v2's precision-tuned cutoff τ_v = 0.872) get high weights; low-score cars (routinely garaged) get weight ≈ 1. This re-balances the garage-only observed set back toward the full claims distribution, then trains a bias-corrected total-loss model. **Caveat:** under a hard threshold the propensity is degenerate (0/1) exactly at the boundary, so the practical workaround treats the uncalibrated model score itself as a soft propensity — see `problem.md` §2.6.

**What to write in the dissertation**
Cite as the theoretical foundation of IPW debiasing. State that Build 06 implements a DoWhy-based IPW estimator whose statistical properties trace to this paper.

---

### P4 · Rosenbaum & Rubin (1983) — Propensity Score

**Citation**
Rosenbaum, P. R. & Rubin, D. B. (1983). "The Central Role of the Propensity Score in Observational Studies for Causal Effects." *Biometrika*, 70(1), 41–55.

**Link** → https://doi.org/10.1093/biomet/70.1.41
**Citations** ≈ 25,000 (Google Scholar — one of the most-cited statistics papers of all time) · **Journal** *Biometrika* (elite methodological journal, IF ≈ 2.4)

**Why this paper matters**
Proves that matching or weighting on the scalar propensity score alone removes all observed confounding — you do not need to match on every covariate. Directly used in Build 04 (propensity-score matching) and Build 06 (IPW).

**Summary**
The propensity score $e(x) = P(W=1 \mid X=x)$ is a balancing score: within strata of equal propensity, treatment is independent of all observed covariates. Therefore, conditioning on $e(X)$ instead of all of $X$ is sufficient for removing confounding. This dimensionality reduction is critical when claims have many features.

**Key concept / formula**
Balancing property: $W \perp X \mid e(X)$
Unconfoundedness given score: $(Y(0), Y(1)) \perp W \mid e(X)$

**How to apply at Insurance Company.**
Estimate propensity scores (probability a car was **scrapped** given its pre-decision features) using logistic regression or XGBoost. Match scrapped cars to sent-to-garage "controls" with similar propensity scores. Differences in observed outcome between matched pairs estimate the scrapping decision's effect on the forced-positive label. **Overlap caveat:** cars with very high scores are scrapped at near-100% rate, so common support fails in the tail — matching cannot recover the oracle there (→ P27).

**What to write in the dissertation**
Cite this as the theoretical justification for propensity-score matching in Build 04. State that the score is estimated with logistic regression and that common support (overlap) is checked to validate the matching.

**Additional note**
~25,000 citations makes this one of the safest methodological citations in the thesis — reviewers will recognise and respect it.

---

### P5 · Pearl (1995) — Causal Diagrams (DAGs) and Do-Calculus

**Citation**
Pearl, J. (1995). "Causal Diagrams for Empirical Research." *Biometrika*, 82(4), 669–710.

**Link** → https://doi.org/10.1093/biomet/82.4.669
**Citations** ≈ 5,000 (Semantic Scholar) · **Journal** *Biometrika* (elite, IF ≈ 2.4)

**Why this paper matters**
Establishes the DAG (directed acyclic graph) language for expressing causal assumptions and the do-calculus for identifying causal effects. DoWhy (Build 06) implements Pearl's framework directly; the SFP loop itself is a cycle in a dynamic causal graph.

**Summary**
Pearl showed that causal assumptions (which variables confound which, which are mediators) can be encoded as a DAG. The do-operator $P(Y \mid do(X=x))$ is distinct from conditional probability $P(Y \mid X=x)$. The do-calculus provides rules for computing causal effects from observational data when certain graphical conditions hold (back-door criterion, front-door criterion).

**Key concept / formula**
Back-door criterion: a set $Z$ blocks all back-door paths from $X$ to $Y$ → $P(Y \mid do(X)) = \sum_z P(Y \mid X, Z=z) P(Z=z)$
This is exactly what Build 06 does: adjust for the confounder set $Z$ = {model score, pre-decision claim features} to estimate the effect of the scrapping decision on the observed (forced-positive) outcome.

**How to apply at Insurance Company.**
Draw the causal DAG for the Insurance Company. FTTL pipeline: model score → scrapping decision → forced label (outcome=1) → retrain next version → model score (the loop). The SFP loop is the cyclic path. Use the back-door criterion to identify which variables need to be controlled when debiasing. **Note:** because the loop is a *cycle*, standard acyclic identification breaks — this is why the dynamical-systems view (P29) and dynamic causal modelling are needed alongside a static DAG.

**What to write in the dissertation**
Include the causal DAG as a figure in the methodology chapter. Cite Pearl (1995) when justifying the identification strategy and when explaining how DoWhy specifies causal assumptions.

---

## Part 2 — Econometric Quasi-Experimental Methods (1990s–2000s)

---

### P6 · Heckman (1979) — Sample Selection Bias

**Citation**
Heckman, J. J. (1979). "Sample Selection Bias as a Specification Error." *Econometrica*, 47(1), 153–161.

**Link** → https://doi.org/10.2307/1912352
**Citations** ≈ 29,000 (Semantic Scholar — one of the most-cited papers in economics) · **Journal** *Econometrica* (top econometrics journal, IF ≈ 6.5)

**Why this paper matters**
The FTTL SFP problem is structurally a sample selection problem: *true* repair outcomes are observed only for cars sent to the garage (`decision=0`, the selected sample); scrapped cars carry a forced label, not a verified outcome. Heckman's paper shows this selection creates bias in any model trained on these labels — and provides a correction strategy. (Caveat: Heckman corrects for labels that are *missing* under selection; here the scrapped-car label is not missing but *forced to 1*, so the correction addresses the garage-selection bias, while the forced-positive contamination additionally needs the PU treatment of P28.)

**Summary**
When the sample used for estimation is selected non-randomly (e.g., only garaged cars carry a verified repair outcome), OLS estimates are biased. Heckman derived a two-stage correction: first model the selection probability, then include the inverse Mills ratio as a control variable in the outcome equation.

**Key concept / formula**
Inverse Mills ratio: $\lambda(z_i) = \frac{\phi(\hat{z}_i)}{\Phi(\hat{z}_i)}$
Adding $\lambda$ as a regressor corrects for selection bias. The modern IPW approach in Build 06 is a re-parameterisation of the same correction.

**How to apply at Insurance Company.**
The FTTL model's verified outcomes come only from garaged cars, while scrapped cars are forced to outcome=1. Heckman's result implies the observed total-loss rate is biased (the garaged set is systematically the *lower-score* cars). Quantify this bias in Build 03 (Unbiased Evaluation) and correct it in Build 06.

**What to write in the dissertation**
Cite in the problem framing section: "the partial observability of true repair outcomes — verified only for garaged cars, never for scrapped ones — constitutes a sample selection problem in the sense of Heckman (1979), which induces systematic bias in any model trained on the production log."

---

### P7 · Imbens & Wooldridge (2009) — Econometrics of Program Evaluation

**Citation**
Imbens, G. W. & Wooldridge, J. M. (2009). "Recent Developments in the Econometrics of Program Evaluation." *Journal of Economic Literature*, 47(1), 5–86.

**Link** → https://doi.org/10.1257/jel.47.1.5
**Citations** ≈ 8,000+ (Semantic Scholar) · **Journal** *Journal of Economic Literature* (highest-impact economics survey journal, IF ≈ 13)

**Why this paper matters**
The definitive survey of DiD, RDD, IV, and matching — the complete toolkit used in Build 04 (Intervention Analysis). Examiners familiar with econometrics will expect this citation when seeing these methods.

**Summary**
Covers: (a) randomised experiments and their design; (b) DiD — comparing before/after treatment for treated vs. control groups; (c) IV — using an exogenous instrument to estimate local ATE; (d) regression discontinuity — exploiting threshold-based assignment; (e) matching and propensity score methods. Each estimator's assumptions, identification conditions, and limitations are clearly laid out.

**Key concept / formula**
DiD estimator: $\hat{\tau}_{DiD} = (\bar{Y}_{treated,post} - \bar{Y}_{treated,pre}) - (\bar{Y}_{control,post} - \bar{Y}_{control,pre})$
Parallel trends assumption: in the absence of treatment, both groups would have evolved in parallel.

**How to apply at Insurance Company.**
Use DiD to estimate the causal effect of a model update (the "treatment") on scrap / observed total-loss rates. The treated group = cars scored by the new model version; control group = cars still scored by the old version (if a phased rollout happened). Parallel trends is checked by plotting pre-period trends. RDD at the absolute τ_v cutoff (0.872 for v2) is the complementary design — near-identical cars just above/below the scrap threshold.

**What to write in the dissertation**
Cite as the methodological authority for Build 04. State which identifying assumptions are invoked for each quasi-experimental design and cite the relevant section of this survey.

---

### P8 · Angrist & Pischke (2009) — Mostly Harmless Econometrics

**Citation**
Angrist, J. D. & Pischke, J.-S. (2009). *Mostly Harmless Econometrics: An Empiricist's Companion*. Princeton University Press. ISBN: 9780691120355.

**Link** → https://press.princeton.edu/books/paperback/9780691120355/mostly-harmless-econometrics
**Citations** ≈ 30,000+ (Google Scholar) · **Publisher** Princeton University Press (most-cited applied econometrics textbook)

**Why this paper matters**
The most-cited applied econometrics textbook; defines the "credibility revolution" standard for causal identification. Using DiD and RDD in a dissertation without citing this book will raise eyebrows.

**Summary**
Chapters cover: OLS and its limitations, IV and two-stage least squares, DiD, RDD. The core lesson: credible causal inference requires clear identification strategies (natural experiments, quasi-randomness), not just controlling for covariates. Each method's key assumption is made explicit and testable where possible.

**Key concept / formula**
The "regression discontinuity" design around a score threshold $c$:
$\tau_{RDD} = \lim_{x \downarrow c} E[Y \mid X=x] - \lim_{x \uparrow c} E[Y \mid X=x]$
Applicable when the model switches from "scrap" to "send to garage" at a score threshold.

**How to apply at Insurance Company.**
FTTL uses exactly such a fixed absolute cutoff — a car is scrapped iff `score ≥ τ_v` (0.872 for v2). This is a **textbook sharp RDD**: RDD estimates the causal effect of the scrapping decision by comparing cars just above and just below 0.872 — near-identical except for whether they were scrapped (and thus whether their outcome was forced to 1 or verified at the garage). This is the cleanest natural experiment in the whole project.

**What to write in the dissertation**
Cite alongside Imbens & Wooldridge (2009) as the applied econometrics standard. Use the RDD design explicitly if a score threshold exists in the Insurance Company. pipeline; document the bandwidth selection and local linear regression approach as specified in Chapter 6 of Angrist & Pischke.

---

## Part 3 — Statistical / Data-Mining Fraud Detection (2000s–2010s)

---

### P9 · Phua et al. (2010) — Comprehensive Survey of Fraud Detection

**Citation**
Phua, C., Lee, V., Smith, K. & Gayler, R. (2010). "A Comprehensive Survey of Data Mining-Based Fraud Detection Research." *arXiv preprint*, arXiv:1009.6119.

**Link** → https://arxiv.org/abs/1009.6119
**Citations** ≈ 795 (Semantic Scholar) · **Venue** Widely cited arXiv preprint; canonical fraud-detection survey

**Why this paper matters**
Establishes the taxonomy of fraud types, detection architectures, and evaluation metrics used across the insurance fraud detection literature. Provides the vocabulary (supervised, unsupervised, network-based) against which this dissertation's contribution is positioned.

**Summary**
Reviews 49 publicly available fraud detection papers across banking, insurance, telecommunications, and healthcare. Identifies three levels of adversarial behaviour (opportunistic, planned, professional). Surveys classification, clustering, and graph-based methods. Highlights the evaluation problem: ground truth labels are incomplete because undetected fraud is never labelled.

**Key concept / formula**
The "tip of the iceberg" problem: observed fraud rate $\hat{p}$ underestimates true fraud rate $p^*$ because the denominator includes uninvestigated claims:
$$\hat{p} = \frac{\text{confirmed fraud}}{\text{investigated claims}} \gg \frac{\text{confirmed fraud}}{\text{all claims}} = p^*$$
The SFP loop magnifies this gap over successive model versions.

**How to apply at Insurance Company.**
**Note — this is an *adjacent-field* paper, not our domain.** FTTL is **total loss prediction, not fraud detection.** P9 is included only for (a) the methodological vocabulary (supervised / unsupervised / network-based) against which the contribution is positioned, and (b) the "tip of the iceberg" observed-rate-bias analogy, which transfers directly: our observed total-loss rate is inflated because scrapped cars are forced to outcome=1 without garage verification, just as observed fraud rate is inflated by looking only where you investigated. Cite the observed-rate-bias problem as empirical motivation for Build 03 (Unbiased Evaluation) — but state the mechanism in total-loss terms (forced positives), not fraud terms.

**What to write in the dissertation**
Cite in the literature review only to position the SFP contribution against the wider ML-for-insurance field, explicitly flagging that FTTL is a total-loss (repair-feasibility) problem rather than a fraud problem. Note that unlike the papers surveyed here, this dissertation addresses the *feedback mechanism* rather than the *detection algorithm* alone.

---

### P10 · Barocas & Selbst (2016) — Big Data's Disparate Impact

**Citation**
Barocas, S. & Selbst, A. D. (2016). "Big Data's Disparate Impact." *California Law Review*, 104, 671–732.

**Link** → https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2477899  (DOI: https://doi.org/10.15779/Z38BG31)
**Citations** ≈ 2,500 (Semantic Scholar) · **Journal** *California Law Review* (top-5 US law review)

**Why this paper matters**
Provides the legal and regulatory framing for why the SFP loop is not merely a technical problem but a potential compliance liability — especially relevant to Insurance Company. UK under FCA guidelines on fair treatment of customers and the EU AI Act.

**Summary**
Argues that even facially neutral ML models trained on historical data can violate anti-discrimination law by perpetuating past biases. Identifies five pathways from biased training data to discriminatory outcomes: target variable definition, feature selection, proxies for protected characteristics, sample bias, and feedback effects. The last pathway is precisely the SFP loop.

**Key concept / formula**
The disparate impact standard: a selection rate for a protected group that is less than 4/5 (80%) of the rate for the group with the highest rate is considered prima facie discriminatory (US EEOC; analogous to FCA proportionality rules in the UK).

**How to apply at Insurance Company.**
Check whether the FTTL model's **scrap rate** varies significantly across vehicle make, damage profile, or customer segments that proxy for protected characteristics. If systematically over-scrapped segments are correlated with protected characteristics, the SFP loop may have disparate-impact implications under FCA PRIN 6 (fair treatment of customers) — e.g. certain makes are written off (and paid out at full value, or wrongly scrapped) at disproportionate rates.

**What to write in the dissertation**
Cite in the ethics and regulatory chapter. Frame the SFP loop as simultaneously a technical problem (model bias) and a legal risk (disparate impact). Note Insurance Company. UK's obligations under FCA rules as a real-world motivation for the research.

---

## Part 4 — Feedback Loops and Performative Prediction (2016–2020)

---

### P11 · Lum & Isaac (2016) — To Predict and Serve?

**Citation**
Lum, K. & Isaac, W. (2016). "To Predict and Serve?" *Significance*, 13(5), 14–19.

**Link** → https://doi.org/10.1111/j.1740-9713.2016.00960.x
**Citations** ≈ 509 (Semantic Scholar) · **Journal** *Significance* (joint RSS/ASA practitioner magazine, high visibility)

**Why this paper matters**
First empirical demonstration — in a domain analogous to insurance — that a model trained on biased data reinforces the patrol patterns that generated the bias. The closest published analogue to the Insurance Company. motor insurance SFP loop.

**Summary**
Applies PredPol (predictive policing software) to Oakland, CA crime data. Shows that because drug arrests reflect where police patrol (not where drugs are actually used), re-training on arrest data sends police back to the same neighbourhoods, creating a self-reinforcing loop. Communities with high historical arrest rates are systematically over-policed.

**Key concept / formula**
Feedback amplification: if the scrapping decision $D_t(x)=\mathbb{1}[f_t(x)\ge\tau]$ tracks model score $f_t(x)$, and $f_{t+1}$ is trained on the forced labels $\{\tilde{y}_i : D_t(x_i)=1 \Rightarrow \tilde{y}_i=1\}$, then in expectation $f_{t+1}(x) \geq f_t(x)$ for high-score regions — the model becomes increasingly confident about already-scrapped vehicle types.

**How to apply at Insurance Company.**
Map "patrol area → vehicle/damage segment" and "drug arrests → forced total-loss labels." **One structural difference to state explicitly:** in policing the label of an unpatrolled area is *missing* (recoverable by later audit); in FTTL the label of a scrapped car is *forced to 1* and the oracle is *destroyed*. Cars of certain makes / damage profiles are scrapped more; those forced positives train the next version to score the segment even higher — regardless of the true underlying total-loss rate — so the segment's scrap rate climbs toward 100% and its true repairability becomes unobservable.

**What to write in the dissertation**
Cite as the primary motivating analogy. State: "Lum & Isaac (2016) demonstrate an empirically observed SFP loop in predictive policing; this dissertation applies the same detection and mitigation framework to the total loss prediction (FTTL) setting, adapting it for the harder case where the action is irreversible and the label is *forced* rather than merely *missing*."

---

### P12 · Ensign et al. (2018) — Runaway Feedback Loops in Predictive Policing

**Citation**
Ensign, D., Friedler, S. A., Neville, S., Scheidegger, C. & Venkatasubramanian, S. (2018). "Runaway Feedback Loops in Predictive Policing." *Proceedings of the 1st ACM FAccT Conference*, PMLR 81:160–171.

**Link** → https://arxiv.org/abs/1706.09847 | https://proceedings.mlr.press/v81/ensign18a.html
**Citations** ≈ 650+ (Semantic Scholar) · **Venue** *FAccT* (A* conference for fairness/accountability in ML)

**Why this paper matters**
Provides the mathematical proof that a prediction-driven decision system (for us, the scrapping policy) converges to a fixed point that ignores true underlying rates — i.e., the feedback loop is not just a risk but a provable inevitability under standard Pólya urn dynamics. Cited in the CLAUDE.md as "runaway feedback FAT'18."

**Summary**
Models the discretised decision (in the original paper, where to patrol; for us, which score band to scrap) as a Pólya urn process. Proves that without exploration, all probability mass concentrates on the cells where incidents were initially recorded, regardless of the true underlying rate elsewhere. Derives the convergence rate as a function of the initial "unfairness" and shows it is not self-correcting.

**Key concept / formula**
Pólya urn dynamic: at each step, the probability of scrapping in score band $r$ is $\propto n_r$ (number of past scraps → forced positives there). As $n_r \to \infty$, the scrapping distribution converges almost surely to a fixed composition determined by initial conditions — not by true total-loss rates.

**How to apply at Insurance Company.**
This is the theoretical model the SFP simulation (Build 01) implements. Parameterise the urn with FTTL's initial per-band scrap rates; show how the distribution converges toward over-scrapping. The randomisation strategies in Build 05 (deliberate garage routing of some high-score cars) are the interventions that break the urn dynamic by re-injecting oracle labels.

**What to write in the dissertation**
Cite as the primary mathematical foundation for the SFP loop mechanism. Reproduce the Pólya urn convergence result in the theory section; show how the simulation in Build 01 recovers it empirically.

---

### P13 · Corbett-Davies et al. (2017) — Algorithmic Decision Making and the Cost of Fairness (KDD '17)

**Citation**
Corbett-Davies, S., Pierson, E., Feller, A., Goel, S. & Huq, A. (2017). "Algorithmic Decision Making and the Cost of Fairness." *Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 797–806.

**Link** → https://dl.acm.org/doi/10.1145/3097983.3098095 | arXiv: https://arxiv.org/abs/1701.08230
**Citations** ≈ 1,445 (Semantic Scholar) · **Venue** *KDD* (A* CORE ranking — top data-mining conference) — **the "feedback loops KDD'17" paper referenced in the project plan**

**Why this paper matters**
Shows formally that imposing fairness constraints on a prediction system can — through feedback effects — harm the very groups the constraints were designed to protect. Establishes the tension between short-term accuracy and long-term equity that motivates the exploration strategies in Build 05.

**Summary**
Studies the trade-off between prediction accuracy and demographic parity in algorithmic decision-making (bail, stop-and-frisk). Shows that depending on which fairness criterion is imposed, the optimal decision boundary changes, and different groups bear different costs. Critically, enforcing parity without addressing the underlying data bias can worsen outcomes over time.

**Key concept / formula**
Optimal threshold under unconstrained utility: $\hat{t} = \arg\min_t \text{Cost}(t)$ where Cost reflects false positive and false negative costs. Under a fairness constraint (e.g., equal FPR), the constrained optimum $\hat{t}^*$ may have higher total cost — the "price of fairness."

**How to apply at Insurance Company.**
When designing the randomisation policy (Build 05), consider the long-run equity impact: a policy that aggressively scraps every high-model-score car may be short-run accurate but deepens the blind spots (segments scrapped at ~100% with zero surviving oracle labels). Use this framework to justify why ε-greedy / Thompson Sampling (budgeted garage routing to recover oracle labels) are preferred over pure exploitation — subject to the precision ≥ 0.985 floor.

**What to write in the dissertation**
Cite in the discussion of the exploration-exploitation trade-off. Note that KDD '17 established the theoretical cost structure; the dissertation implements a practical randomisation policy that navigates this trade-off in the specific context of motor insurance.

---

### P14 · Liu et al. (2018) — Delayed Impact of Fair Machine Learning

**Citation**
Liu, L. T., Dean, S., Rolf, E., Simchowitz, M. & Hardt, M. (2018). "Delayed Impact of Fair Machine Learning." *Proceedings of the 35th ICML*, PMLR 80:3150–3158.

**Link** → https://arxiv.org/abs/1803.04383 | https://proceedings.mlr.press/v80/liu18c
**Citations** ≈ 491 (Semantic Scholar) · **Venue** *ICML* (A* CORE ranking)

**Why this paper matters**
Proves that static fairness criteria can produce worse long-term outcomes for protected groups — because they ignore the temporal dynamics of feedback loops. Directly motivates why Build 05's randomisation strategy must account for long-run distributional shift.

**Summary**
Models a loan decision system where the bank's decisions affect borrowers' future creditworthiness. Shows that "demographic parity" (equal selection rates) can help or harm minority groups depending on the true underlying benefit from selection. Identifies "active harm," "stagnation," "improvement," and "equity" zones based on the relationship between true positive rates and long-term outcome distributions.

**Key concept / formula**
Long-run outcome for group $g$: $\mu_g^{(t+1)} = f(\mu_g^{(t)}, \pi_g^{(t)})$ — the next period's mean outcome is a function of the current mean and the selection rate. Feedback creates a dynamical system; static fairness criteria analyse only the current period's snapshot.

**How to apply at Insurance Company.**
Extend the SFP simulation (Build 01) beyond 3 model versions to show the long-run trajectory of scrap / observed total-loss rates for different vehicle segments under different scrapping/garage-routing policies. Use this framework to argue that a randomisation policy evaluated only at period $t=1$ may look worse (extra garage cost) but converge to a better long-run equilibrium (recovered oracle labels, less over-scrapping).

**What to write in the dissertation**
Cite in the long-run analysis section of Build 05. Frame the randomisation strategy as a dynamic policy rather than a static reweighting — the dissertation's contribution goes beyond identifying the loop to designing a policy with proven long-run properties.

---

### P15 · Perdomo et al. (2020) — Performative Prediction

**Citation**
Perdomo, J. C., Zrnic, T., Mendler-Dünner, C. & Hardt, M. (2020). "Performative Prediction." *Proceedings of the 37th ICML*, PMLR 119:7599–7609.

**Link** → https://arxiv.org/abs/2002.06673 | https://proceedings.mlr.press/v119/perdomo20a
**Citations** ≈ 325+ (Semantic Scholar) · **Venue** *ICML* (A* CORE ranking)

**Why this paper matters**
Provides the most rigorous mathematical formalisation of the self-fulfilling prophecy concept in ML. Introduces "performative risk" and characterises the conditions under which a deployed model converges to a stable fixed point — or diverges. This is the theoretical backbone of the entire dissertation.

**Summary**
A prediction is "performative" if deploying it changes the distribution it was trained to predict. Defines performative risk: $PR(\theta) = \mathbb{E}_{z \sim D(\theta)}[\ell(z; \theta)]$ where $D(\theta)$ is the distribution induced by deploying model $\theta$. Identifies "performatively stable" models (fixed points of repeated retraining) and "performatively optimal" models (minimisers of performative risk). These two are generally different.

**Key concept / formula**
Performative risk: $PR(\theta) = \mathbb{E}_{z \sim D(\theta)}[\ell(z; \theta)]$
Standard ERM minimises $E_{z \sim D_0}[\ell(z;\theta)]$ on fixed data $D_0$ — ignoring that $D$ shifts with $\theta$. The SFP loop is the gap between these two objectives.

**How to apply at Insurance Company.**
Argue that Insurance Company.'s FTTL model is performative: its scores determine which cars are scrapped, which forces their labels to 1, which changes what data trains the next version. Build 01 simulates this performative dynamic; Build 06 estimates the gap between performative risk and standard training risk.

**What to write in the dissertation**
Cite in the theory section as the formal definition of the dissertation's central concept. Include the performative risk formula in the notation table. State explicitly: "the Insurance Company. Fast Track Total Loss pipeline exhibits performative prediction in the sense of Perdomo et al. (2020)."

---

## Part 5 — Causal ML and Debiasing / Double ML (2018–present)

---

### P16 · Chernozhukov et al. (2018) — Double/Debiased Machine Learning

**Citation**
Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W. & Robins, J. (2018). "Double/Debiased Machine Learning for Treatment and Structural Parameters." *The Econometrics Journal*, 21(1), C1–C68.

**Link** → https://doi.org/10.1111/ectj.12097 | arXiv: https://arxiv.org/abs/1608.00060
**Citations** ≈ 6,000+ (Google Scholar — one of the most-cited econometrics papers of the past decade) · **Journal** *The Econometrics Journal* (IF ≈ 3, landmark methodological contribution)

**Why this paper matters**
Provides the Neyman-orthogonal score / cross-fitting framework that allows ML-estimated nuisance functions (propensity score, outcome model) to be used in causal inference without regularisation bias contaminating the causal effect estimate. Essential for Build 04 and 06 when features are high-dimensional.

**Summary**
When using ML to estimate propensity scores or outcome functions (needed for causal inference), regularisation introduces bias that inflates standard errors and invalidates inference. Double ML solves this via: (1) Neyman orthogonality — deriving score functions with zero first-order sensitivity to nuisance estimation error; (2) cross-fitting — estimating nuisances on a separate fold from the final effect estimate. Together these allow $\sqrt{n}$-consistent, asymptotically normal treatment effect estimates even when ML nuisance estimators converge at slower rates.

**Key concept / formula**
The Neyman-orthogonal score $\psi(W, \theta, \eta)$ satisfies: $\partial_\eta E[\psi] = 0$ at $\eta = \eta_0$. Cross-fitted estimator: estimate $\hat{\eta}_k$ on $\mathcal{D} \setminus \mathcal{D}_k$, then evaluate $\psi$ on $\mathcal{D}_k$, average across folds.

**How to apply at Insurance Company.**
When estimating the effect of the scrapping decision on the forced-positive outcome (Build 04), use cross-fitted propensity-to-scrap scores (from LightGBM) in a double ML estimator. This gives valid confidence intervals even though the scrapping propensity has a high-dimensional feature set (`vehicle_make`, `damage_severity`, `repair_to_value_ratio`, `vehicle_age_years`, enrichment specs, etc.). **Overlap caveat:** DML needs common support, but high-score cars are scrapped at near-100% rate → the estimator breaks in that tail (→ P27).

**What to write in the dissertation**
Cite in the methodology section for Build 04. State that the propensity score is estimated with LightGBM using 5-fold cross-fitting following Chernozhukov et al. (2018), and that this ensures the causal effect estimate is $\sqrt{n}$-consistent even with flexible ML nuisance estimators.

---

### P17 · Sculley et al. (2015) — Hidden Technical Debt in ML Systems

**Citation**
Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., Chaudhary, V., Young, M., Crespo, J.-F. & Dennison, D. (2015). "Hidden Technical Debt in Machine Learning Systems." *Advances in Neural Information Processing Systems (NeurIPS) 28*, pp. 2503–2511.

**Link** → https://proceedings.neurips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html
**Citations** ≈ 4,000+ (Google Scholar) · **Venue** *NeurIPS* (A* CORE ranking)

**Why this paper matters**
Identifies feedback loops as a first-class form of technical debt in production ML systems. Provides the systems-engineering framing for why the SFP loop is hard to detect and correct in an operational pipeline like Insurance Company.'s.

**Summary**
Categorises ML technical debt as: entanglement (correlated features), hidden feedback loops, undeclared consumers, data dependency debt, and configuration debt. Feedback loops are singled out as particularly dangerous because they can cause slow but compounding degradation that is invisible in standard monitoring metrics. A model's outputs influence the world, which influences future training data.

**Key concept / formula**
The "data dependency debt" formulation: if model output $f(x)$ feeds into any process that generates future training data $D_{t+1}$, then the model has a "hidden feedback loop." Formally: $D_{t+1} = g(D_t, f_t)$ where $g$ is the data-generating process influenced by the model. The SFP loop is exactly this.

**How to apply at Insurance Company.**
Use the Sculley et al. taxonomy to audit the FTTL pipeline: identify all places where model outputs influence future data (the scrapping decision → forced label being the primary one; also settlement routing and enrichment-join effects). Each is a potential SFP entry point. Document these as part of the Build 00 data exploration.

**What to write in the dissertation**
Cite in the problem framing section alongside Perdomo et al. (2020). Position the dissertation as applying the SFP detection framework to a specific instance of the hidden feedback loop problem identified by Sculley et al.

---

### P18 · D'Amour et al. (2022) — Underspecification in Modern ML

**Citation**
D'Amour, A., et al. (2022). "Underspecification Presents Challenges for Credibility in Modern Machine Learning." *Journal of Machine Learning Research*, 23(226), 1–61.

**Link** → https://arxiv.org/abs/2011.03395 | https://www.jmlr.org/papers/v23/20-1335.html
**Citations** ≈ 900+ (Semantic Scholar) · **Journal** *JMLR* (IF ≈ 6, top open-access ML journal)

**Why this paper matters**
Shows that multiple models with identical in-distribution performance can behave very differently under distribution shift — precisely the problem when a model trained/evaluated on the garage-selected (and forced-label-contaminated) subset is judged against the full claims population (Build 03).

**Summary**
A training pipeline is "underspecified" when many models achieve equivalent training performance but diverge under distribution shift. Experiments across NLP, computer vision, medical imaging, and genomics show that standard training and evaluation pipelines do not select for models that generalise reliably. The solution requires stress tests that expose distribution shift.

**Key concept / formula**
Underspecification: $\exists \theta_1 \neq \theta_2$ such that $R_{train}(\theta_1) \approx R_{train}(\theta_2)$ but $R_{test}(\theta_1) \gg R_{test}(\theta_2)$ for some OOD test distribution. For the SFP problem: the in-distribution performance is on garage-observed cars; the OOD distribution is all cars (including the scrapped tail with no verified label).

**How to apply at Insurance Company.**
Build 03 (Unbiased Evaluation) is directly motivated by this paper: the FTTL model's performance on garage-observed cars (where true labels exist) is not representative of its performance on all cars (the full portfolio, including the scrapped high-score tail). Selective-labels/IPS-corrected metrics estimate the true OOD performance.

**What to write in the dissertation**
Cite in Build 03. State that standard in-sample AUC is an underspecified metric for the FTTL model because the test distribution (garage-observed cars) is a non-random subset of the deployment distribution (all cars), and the missing tail is systematically the high-score cars the model is most confident about.

---

## Part 6 — Exploration-Exploitation / Bandit Strategies (classical to modern)

---

### P19 · Thompson (1933) — Thompson Sampling

**Citation**
Thompson, W. R. (1933). "On the Likelihood That One Unknown Probability Exceeds Another in View of the Evidence of Two Samples." *Biometrika*, 25(3–4), 285–294.

**Link** → https://doi.org/10.1093/biomet/25.3-4.285
**Citations** ≈ 3,000+ (Google Scholar) · **Journal** *Biometrika* (elite — originating paper of Thompson Sampling)

**Why this paper matters**
The original paper introducing Thompson Sampling — the primary randomisation algorithm evaluated in Build 05 for escaping the SFP loop.

**Summary**
Thompson posed the question: given two Bernoulli arms with unknown parameters, how should one sequentially choose which arm to pull to maximise expected reward? His solution was to draw a sample from the posterior distribution over each arm's parameter and choose the arm whose sample is highest — now called posterior sampling or Thompson Sampling.

**Key concept / formula**
At step $t$: draw $\tilde{\theta}_k \sim \text{Beta}(\alpha_k, \beta_k)$ for each arm $k$; pull arm $k^* = \arg\max_k \tilde{\theta}_k$; update $\alpha_{k^*}$ or $\beta_{k^*}$ based on outcome.
For FTTL exploration: arm = vehicle/score segment; reward = a *recovered oracle label* (the true repair outcome learned by routing a car to garage instead of scrapping it); prior updated with each garage outcome.

**How to apply at Insurance Company.**
Partition cars into segments (by score decile, vehicle category, damage profile). Run Thompson Sampling across segments: occasionally **route a high-model-score car to the garage instead of scrapping it**, to update the Beta posterior for that segment's true total-loss rate. Over time, this ensures no over-scrapped segment is permanently a blind spot. Unlike cheap re-investigation, each exploration here costs a garage assessment (and risks paying a genuine total loss's full value), so the budget must be modelled explicitly (→ P13) and floored by the precision ≥ 0.985 constraint.

**What to write in the dissertation**
Cite as the origin of the algorithm. Describe the Beta-Binomial model as the practical instantiation for the binary total-loss label recovered at the garage. Compare to ε-greedy in Build 05 using regret (against a garage-cost budget) as the evaluation metric.

---

### P20 · Auer, Cesa-Bianchi & Fischer (2002) — UCB1

**Citation**
Auer, P., Cesa-Bianchi, N. & Fischer, P. (2002). "Finite-Time Analysis of the Multiarmed Bandit Problem." *Machine Learning*, 47(2–3), 235–256.

**Link** → https://doi.org/10.1023/A:1013689704352
**Citations** ≈ 7,000 (Semantic Scholar) · **Journal** *Machine Learning* (Springer, IF ≈ 7.5 — one of the most-cited bandit papers)

**Why this paper matters**
Derives UCB1 and its $O(\log T)$ regret guarantee — the theoretical benchmark against which Thompson Sampling and ε-greedy are compared in Build 05.

**Summary**
Proves that the UCB1 algorithm achieves logarithmic regret in the stochastic bandit setting, which is optimal up to constants. UCB1 selects the arm with the highest upper confidence bound: $\bar{x}_k + \sqrt{2 \ln t / n_k}$. This is the "optimism in the face of uncertainty" principle — explore under-pulled arms until you are confident they are inferior.

**Key concept / formula**
UCB1 index: $\text{UCB}_k(t) = \bar{x}_k + \sqrt{\frac{2 \ln t}{n_k}}$
Regret bound: $E[R_T] \leq \sum_{k: \mu_k < \mu^*} \left(\frac{8 \ln T}{\Delta_k} + \left(1 + \frac{\pi^2}{3}\right) \Delta_k\right)$
where $\Delta_k = \mu^* - \mu_k$ is the gap between arm $k$ and the best arm.

**How to apply at Insurance Company.**
UCB1 applied to FTTL garage-routing: each vehicle/score segment is an arm; $\bar{x}_k$ is the empirical *true* total-loss rate for that segment (from garage outcomes); $n_k$ is the number of cars routed to garage there. Segments with few surviving oracle labels (heavily scrapped) get a UCB bonus, encouraging garage routing even when the model scores them as near-certain total losses.

**What to write in the dissertation**
Cite when introducing the UCB baseline in Build 05. State the regret bound explicitly and contrast with Thompson Sampling's empirical performance. Note that UCB is frequentist (no prior needed) while Thompson Sampling is Bayesian — both are evaluated on the synthetic dataset.

---

### P21 · Russo et al. (2018) — Tutorial on Thompson Sampling

**Citation**
Russo, D. J., Van Roy, B., Kazerouni, A., Osband, I. & Wen, Z. (2018). "A Tutorial on Thompson Sampling." *Foundations and Trends in Machine Learning*, 11(1), 1–96.

**Link** → https://doi.org/10.1561/2200000070 | arXiv: https://arxiv.org/abs/1707.02038
**Citations** ≈ 3,000+ (Google Scholar) · **Journal** *Foundations and Trends in ML* (NOW Publishers, high-citation review journal)

**Why this paper matters**
The standard practical reference for implementing Thompson Sampling. Covers Bernoulli bandits (directly applicable to the binary total-loss label recovered at the garage), contextual extensions, and convergence guarantees.

**Summary**
Provides a thorough tutorial from the Beta-Bernoulli case through Gaussian, contextual, and combinatorial bandits. Derives regret bounds, discusses computational approximations (Langevin, variational), and surveys empirical results across online advertising, medical trials, and recommendation systems.

**Key concept / formula**
Bayesian regret bound for Thompson Sampling:
$\text{BayesRegret}(T) \leq \sqrt{\frac{T K \ln K}{2}}$
where $K$ is the number of arms. This improves on UCB1's $O(\sqrt{KT \ln T})$ bound in many practical settings.

**How to apply at Insurance Company.**
The contextual bandit extension (Section 5 of the tutorial) is directly applicable: use car features (`repair_to_value_ratio`, `damage_severity`, `vehicle_age_years`, make) as context to form a per-car garage-routing exploration policy rather than a single Beta distribution per segment.

**What to write in the dissertation**
Cite as the practical implementation reference for Build 05. If the contextual extension is implemented, cite the specific section. Include the regret bound in the evaluation section and compare empirically to UCB1 on the synthetic dataset.

---

## Part 7 — Fairness in ML and Insurance AI Regulation (2016–2024)

---

### P22 · Hardt, Price & Srebro (2016) — Equality of Opportunity

**Citation**
Hardt, M., Price, E. & Srebro, N. (2016). "Equality of Opportunity in Supervised Learning." *Advances in Neural Information Processing Systems (NeurIPS) 29*.

**Link** → https://arxiv.org/abs/1610.02413 | https://proceedings.neurips.cc/paper/2016/hash/9d2682367c3935defcb1f9e247a97c0d-Abstract.html
**Citations** ≈ 5,000+ (Google Scholar) · **Venue** *NeurIPS* (A* CORE ranking)

**Why this paper matters**
Introduces the equalised-odds and equal opportunity fairness criteria. In the FTTL context: the model should not have systematically different error rates (e.g. wrongly scrapping repairable cars) for particular vehicle segments because those segments accumulated more forced-positive labels historically.

**Summary**
Proposes that a fair classifier should have equal true positive rates and equal false positive rates across demographic groups — "equalised odds." A weaker condition, "equal opportunity," requires only equal true positive rates. Provides a post-processing algorithm to achieve either criterion by adjusting decision thresholds per group.

**Key concept / formula**
Equalised odds: $\hat{Y} \perp A \mid Y$ — prediction is independent of the protected attribute $A$ given the true label $Y$.
Equal opportunity: $P(\hat{Y}=1 \mid A=0, Y=1) = P(\hat{Y}=1 \mid A=1, Y=1)$ — equal TPR across groups.

**How to apply at Insurance Company.**
Check whether the SFP-corrected model achieves equal opportunity across vehicle make / damage-profile groups. A model that wrongly scraps repairable cars disproportionately in certain segments (because those segments accumulated more forced positives) fails this criterion. Build 06's IPW/PU debiasing should improve equal opportunity.

**What to write in the dissertation**
Cite in the evaluation section of Build 06. Use equal opportunity as one of the post-mitigation fairness metrics alongside standard AUC. Note: the production XGBoost model is not calibrated (see README — Model Training Methodology), so raw score outputs cannot be interpreted as true probabilities. Calibration as a formal metric is therefore not evaluated unless a post-hoc calibration step (e.g., Platt scaling) is applied first. Show before/after comparison on TPR parity across segments.

---

### P23 · Barocas, Hardt & Narayanan (2023) — Fairness and Machine Learning

**Citation**
Barocas, S., Hardt, M. & Narayanan, A. (2023). *Fairness and Machine Learning: Limitations and Opportunities*. MIT Press. (Draft freely available since 2019.)

**Link** → https://fairmlbook.org | MIT Press: https://mitpress.mit.edu/9780262048613/fairness-and-machine-learning/
**Citations** ≈ 4,000+ (online draft; Google Scholar) · **Publisher** MIT Press

**Why this paper matters**
The definitive textbook on algorithmic fairness; Chapter 4 covers feedback loops explicitly, Chapter 6 covers causal reasoning for fairness. Examiners familiar with the FAccT literature will expect this citation.

**Summary**
Covers: measurement, classification, causality and fairness, equal opportunity, individual fairness, and the impossibility results showing that multiple fairness criteria cannot be simultaneously satisfied. Chapter 4 ("Causality") explains why fairness interventions must address the data-generating process (the SFP loop) rather than just post-hoc adjustments to outputs.

**Key concept / formula**
The impossibility theorem (Chouldechova 2017, formalised here): calibration + equal FPR + equal FNR cannot all hold simultaneously when base rates differ across groups. This implies any fairness criterion for the Insurance Company. model involves trade-offs that must be explicitly documented.

**How to apply at Insurance Company.**
Use the impossibility result to justify why a single metric (e.g., AUC alone) is insufficient for evaluating the post-mitigation model. Present the fairness trade-off frontier in Build 06 to show which criterion is prioritised and why.

**What to write in the dissertation**
Cite as the primary fairness reference. Use the classification in Chapter 2 to organise the fairness evaluation section. Acknowledge the impossibility result and state explicitly which fairness criterion is adopted and why.

---

### P24 · Corbett-Davies & Goel (2023) — The Measure and Mismeasure of Fairness

**Citation**
Corbett-Davies, S. & Goel, S. (2023). "The Measure and Mismeasure of Fairness." *Journal of Machine Learning Research*, 24(312), 1–117.

**Link** → https://arxiv.org/abs/1808.00023 | JMLR: https://dl.acm.org/doi/10.5555/3648699.3649011
**Citations** ≈ 700+ (Semantic Scholar) · **Journal** *JMLR* (IF ≈ 6)

**Why this paper matters**
Critical review showing that calibration, anti-classification, and classification parity all conflict with each other and with social welfare maximisation. Particularly relevant when justifying the specific fairness metric chosen for the Insurance Company. model evaluation.

**Summary**
Argues that commonly used fairness metrics have unintended consequences when applied naively. Proposes "conditional use accuracy equality" as a more principled criterion. Shows through the bail and lending examples that equality constraints on error rates can lead to worse outcomes for the groups being "protected."

**Key concept / formula**
Calibration: $P(Y=1 \mid \hat{p}(X) = p) = p$ — model probabilities match true probabilities. Anti-classification: the model does not use protected attributes. Classification parity: equal error rates. Theorem: all three cannot hold simultaneously when $P(Y=1 \mid A=0) \neq P(Y=1 \mid A=1)$.

**How to apply at Insurance Company.**
True total-loss base rates vary by vehicle make and damage profile. Document these base rate differences and show that perfect calibration implies different error rates across groups — this is expected and not a failure of the model. Use this to defend against naive criticism of disparate error rates. **Important caveat**: the production XGBoost model is not calibrated (see README — Model Training Methodology). Raw scores are used for ranking/triage only, not as probability estimates. This means calibration cannot be directly assessed from model outputs unless Platt scaling or isotonic regression is applied post-hoc. This also affects IPS/IPW debiasing in Build 06: propensity weights derived from uncalibrated scores introduce additional bias into the reweighting — a limitation to acknowledge explicitly.

**What to write in the dissertation**
Cite in the fairness evaluation section. Use the impossibility result to frame the discussion: "following Corbett-Davies & Goel (2023), we acknowledge that calibration and classification parity cannot be simultaneously achieved given differential base rates across claim segments." Note that because the production model is not calibrated, formal calibration assessment requires a post-hoc calibration step not present in the current pipeline — flag this as a limitation and a direction for future work.

---

### P25 · European Parliament & Council of the EU (2024) — EU AI Act

**Citation**
European Parliament & Council of the EU (2024). "Regulation (EU) 2024/1689 of the European Parliament and of the Council — Laying Down Harmonised Rules on Artificial Intelligence (Artificial Intelligence Act)." *Official Journal of the European Union*, L 2024/1689. Entered into force: 1 August 2024.

**Link** → https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=OJ:L_202401689
EIOPA factsheet: https://www.eiopa.europa.eu/document/download/b53a3b92-08cc-4079-a4f7-606cf309a34a_en
**Impact** — Primary EU legislation (no IF; regulatory force)

**Why this paper matters**
Classifies AI systems used in insurance risk assessment and claims handling as **high-risk** (Annex III). Requires technical documentation, bias testing, human oversight, and post-deployment monitoring. Provides the regulatory mandate for solving the SFP loop: it is not merely academic but a compliance obligation for Insurance Company. UK.

**Summary**
The AI Act creates a risk-based framework: prohibited AI (social scoring, biometric surveillance), high-risk AI (credit, insurance, employment, law enforcement), and limited/minimal risk. For Insurance Company. UK, the total loss scoring system is high-risk under Annex III point 5(b) (AI in insurance pricing and risk assessment) and point 6 (AI in law enforcement-adjacent tasks). Requirements include: risk management systems, data governance, transparency documentation, human oversight, accuracy and robustness requirements, and post-market monitoring.

**Key concept / formula**
Article 9 (Risk management system): continuous risk management cycle required throughout the lifecycle. Article 10 (Data governance): training data must be representative, free from errors, and complete. The SFP loop directly violates Article 10 — biased labels from the scrapping decision are not representative.

**How to apply at Insurance Company.**
The dissertation's SFP detection framework (Build 02) and mitigation methods (Builds 05–06) can be positioned as the technical documentation and bias-testing component required by Articles 9–10 of the AI Act. Build 03 (Unbiased Evaluation) maps to Article 15 (accuracy requirements).

**What to write in the dissertation**
Cite in the ethics and regulatory chapter. State: "the total loss scoring system at Insurance Company. UK falls within the high-risk category under Annex III of the EU AI Act (Regulation 2024/1689), which mandates bias testing and post-deployment monitoring. This dissertation provides a technical framework for satisfying those requirements."

---

### P26 · Lattimore & Szepesvári (2020) — Bandit Algorithms

**Citation**
Lattimore, T. & Szepesvári, C. (2020). *Bandit Algorithms*. Cambridge University Press. ISBN: 9781108486828.

**Link** → https://tor-lattimore.com/downloads/book/book.pdf (free PDF) | Cambridge: https://doi.org/10.1017/9781108571401
**Citations** ≈ 1,500+ · **Publisher** Cambridge University Press

**Why this paper matters**
Definitive graduate textbook on bandit algorithms; the primary theoretical reference for Build 05. Covers UCB, Thompson Sampling, contextual bandits, and exploration strategies with rigorous proofs.

**Summary**
Part I: the stochastic bandit framework and regret bounds. Part II: adversarial bandits. Part III: contextual and structured bandits. Chapter 36 covers linear contextual bandits directly applicable when claim features are used to personalise the exploration policy.

**Key concept / formula**
Regret decomposition: $R_T = \sum_{t=1}^T \Delta_{A_t}$ where $\Delta_k = \mu^* - \mu_k$.
The goal of any bandit algorithm is to minimise expected cumulative regret $E[R_T]$ by balancing exploration (learning the true total-loss rate of under-observed arms via garage routing) and exploitation (scrapping high-score cars to save garage cost).

**How to apply at Insurance Company.**
Use Chapter 36 to implement a contextual bandit that takes car features as context and outputs a garage-routing (vs. scrap) probability, subject to the precision ≥ 0.985 floor and a garage-cost budget. This is strictly more powerful than the segment-level Thompson Sampling in Build 05 and can be presented as a future extension.

**What to write in the dissertation**
Cite in the Build 05 methodology section. Reference Chapter 4 (UCB1) and Chapter 36 (contextual bandits) as the theoretical grounding. Note the contextual extension as a direction for future work.

---

## Part 8 — Selective Labels, Label Noise & Partial Observability

*Added 2026-06-15. These papers address the structural observability gap unique to the total loss prediction domain: the model's own scrapping decision determines which repair outcomes are ever observed.*

---

### P27 · Lakkaraju et al. (2017) — The Selective Labels Problem

**Citation**
Lakkaraju, H., Kleinberg, J., Leskovec, J., Ludwig, J. & Mullainathan, S. (2017). "The Selective Labels Problem: Evaluating Algorithmic Predictions in the Presence of Unobservable Outcomes." *Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 275–284.

**Link** → https://dl.acm.org/doi/10.1145/3097983.3098066
**Citations** ≈ 450+ (Semantic Scholar) · **Venue** *KDD* (A* CORE ranking)

**Why this paper matters**
The Insurance Company. total loss model is a textbook selective-labels system: the model decides whether to scrap (`decision=1`) or send to garage (`decision=0`). Repair outcomes (`garage_outcome`) are observable only for `decision=0` rows. For scrapped cars the true repair outcome is structurally absent — not missing at random. This paper provides the exact formal framework for evaluating and learning from such systems, and is the direct citation for this class of problems in the fairness and causal ML literature.

**Summary**
Studies the problem of evaluating ML models when outcomes are only observed for a subset of cases determined by the model's own decisions (or a human predecessor). Formalises **selective labels bias**: the observed accuracy on the selected subset systematically overestimates true accuracy on the full population. Proposes evaluation strategies that account for structural missingness of outcomes in the unselected group, and derives conditions under which counterfactual performance can be bounded or estimated from observational data.

**Key concept / formula**
Let $\hat{Y}_i$ be the model prediction and $D_i \in \{0,1\}$ the binary decision (0 = garage, 1 = scrap). Outcome $Y_i$ is observable only when $D_i = 0$:
$$Y_i \text{ observed} \iff D_i = 0$$
Standard accuracy evaluated on $\{i : D_i = 0\}$ is biased because $D_i$ is a function of $\hat{Y}_i$. The paper derives conditions under which counterfactual performance on $\{i : D_i = 1\}$ can be bounded or estimated.

**How to apply at Insurance Company.**
`model_v1_decision = 1` → car scrapped → `garage_outcome` permanently unobservable. All model evaluation in Build 03 is implicitly selective-labels evaluation on the `decision=0` subset. Cite this paper when explaining why standard AUC on the observed log data is a biased estimate of true total loss prediction accuracy. Use the framework to formalise the oracle-absence limitation described in `synth_data_structure.md`.

**What to write in the dissertation**
Cite in Build 02 (loop detection) and Build 03 (unbiased evaluation) as the formal characterisation of the structural observability constraint. State: "the total loss prediction pipeline constitutes a selective-labels system in the sense of Lakkaraju et al. (2017): repair outcomes are observed only for claims where the model decided not to scrap the vehicle, creating a structural missing-data problem that cannot be resolved post-hoc without external intervention such as targeted garage audits."

---

### P28 · Bekker & Davis (2020) — Learning from Positive and Unlabeled Data

**Citation**
Bekker, J. & Davis, J. (2020). "Learning from Positive and Unlabeled Data: A Survey." *Machine Learning*, 109(4), 719–760.

**Link** → https://doi.org/10.1007/s10994-020-05877-5 | arXiv: https://arxiv.org/abs/1811.04820
**Citations** ≈ 1,000+ (Google Scholar) · **Journal** *Machine Learning* (Springer, IF ≈ 7.5)

**Why this paper matters**
The label structure of the Insurance Company. total loss dataset maps onto the PU learning setting. Cars sent to garage with confirmed total losses are **labeled positives** (`Y=1`, observed). Cars sent to garage that were successfully repaired are **labeled negatives** (`Y=0`). Scrapped cars have a forced label of 1 but their true repair outcome is unknown — they may have been repairable false positives. PU learning provides theory for training and evaluating classifiers under exactly this asymmetric observability structure.

**Summary**
Comprehensive survey of learning algorithms when training data consists of labeled positives and unlabeled examples (containing both true positives and true negatives). Covers: (a) the two main assumptions — single-training-set (SCAR) vs. selected-completely-at-random; (b) methods for estimating the class prior $\pi = P(Y=1)$ from unlabeled data; (c) algorithms including biased SVM, EM-based methods, two-step methods (spy technique), and cost-sensitive re-weighting; (d) evaluation criteria under PU assumptions.

**Key concept / formula**
PU risk decomposition: given labeled positives $\mathcal{P}$ and unlabeled $\mathcal{U}$ (true positive rate $\pi$), the risk of classifier $g$ is:
$$R(g) = \pi \cdot R^+(g) + (1-\pi) \cdot R^-(g)$$
where $R^+(g)$ and $R^-(g)$ are false-negative and false-positive risks. Estimating $\pi$ — the proportion of true total losses among scrapped cars — is the core estimation problem in the Insurance Company. context.

**How to apply at Insurance Company.**
Treat `model_v1_decision=1` rows (scrapped) as the "unlabeled" group: their observed label is 1, but the true fraction of genuine total losses $\pi$ is unknown. The `decision=0` rows with `outcome=1` are the labeled positives. Use PU learning methods to estimate $\hat{\pi}$, which quantifies the false-positive rate of the scrapping policy. This estimate directly informs the magnitude of SFP bias quantified in Builds 01 and 03.

**What to write in the dissertation**
Cite in Build 03 (Unbiased Evaluation) and Build 06 (Causal Mitigation). Frame the label contamination in `model_v1_observed_outcome` as a PU learning problem: "following Bekker & Davis (2020), we treat scrapped-car rows as unlabeled under the PU assumption, since their observed label of 1 reflects the scrapping decision rather than a verified repair outcome. We estimate the class prior $\hat{\pi}$ to quantify the magnitude of label contamination introduced by the SFP mechanism."

---

## Part 9 — Mathematical Evaluation of SFP / Feedback Loops

*Added 2026-06-23. These papers provide formal tools for **quantifying** how strong an SFP loop is — as opposed to proving it exists (P12) or defining its framework (P15). They are the gap-filling literature for Build 02's detection logic.*

---

### P29 · Veprikov, Afanasiev & Khritankov (2025) — A Mathematical Model of the Hidden Feedback Loop Effect in Machine Learning Systems

> **Numbering note (corrected 2026-07-07):** P29 is Veprikov et al. (the dynamical-systems / repeated-learning map) and P31 is Mendler-Dünner et al. (the convergence-rate paper). This matches the canonical numbering used everywhere else in the repository — `problem.md` §2.3, `literatures/notes/p29.md`, `notebook/02_01_p29_sfp_detection_dynamical_systems.ipynb`, and `report/project_plan/*`. Earlier drafts of this file (and `literatures/compare.md`) had the two swapped.

**Citation**
Veprikov, A., Afanasiev, A. & Khritankov, A. (2025). "A Mathematical Model of the Hidden Feedback Loop Effect in Machine Learning Systems." *Knowledge and Information Systems* (Springer). (arXiv preprint: arXiv:2405.02726, May 2024.)

**Link** → https://arxiv.org/abs/2405.02726 | https://link.springer.com/article/10.1007/s10115-025-02560-w
**Citations** ≈ 5 (early; published 2025) · **Journal** *Knowledge and Information Systems* (Springer, IF ≈ 2.5)

**Why this paper matters**
The only paper identified (as of 2026) that provides a single mathematical model unifying **error amplification**, **induced concept drift**, and **echo chambers** as special cases of the same repeated-learning feedback loop. This is the most directly applicable paper for formalising what the SFP simulation (Build 01) implements and what the detector (Build 02) is searching for. It supplies the notation for `problem.md` §2.3's loop mechanism and is the anchor of the Build 02 dynamical-systems diagnostic (`notebook/02_01_p29_sfp_detection_dynamical_systems.ipynb`).

**Summary**
Formalises the entire "data collection → training → deployment → environment influence → data collection" cycle as a single dynamical system, treating one retraining generation as an evolution operator $D_t$ acting on the PDF $f_t$ of the model's residuals. The key insight: the state of the environment at time t+1 is a deterministic function of the environment at time t *and* the predictions made at t. This causally couples the learner to the data-generating process, violating i.i.d. assumptions and producing a zoo of observable phenomena (error amplification, concept drift, echo chambers) depending on the feedback gain coefficient. Provides a theorem on the limiting set of distributions the system can converge to — a positive loop drives $f_t \to \delta$ (Dirac delta at residual zero), a negative loop blows up/collapses the variance — and sufficient conditions for the loop to be "hidden" (undetectable by standard train/test splitting).

**Key concept / formula**
Repeated learning map: $E_{t+1} = F(E_t, M_t)$ where $E_t$ is the environment distribution and $M_t$ is the deployed model. Loop summary statistic $\psi_t = f_t(0)$ (residual density at zero): a rising $\psi_t$ trajectory is the positive-loop fingerprint. A feedback loop is "hidden" when the standard empirical risk on the held-out split is not a monotone function of the true performative risk — i.e., the model appears to improve on the test set while the loop worsens. This directly explains why v2a's OOT AUC at Insurance Company. looked acceptable while the scrap rate inflated.

**How to apply at Insurance Company.**
Build 01 (SFP Simulation) implements the map $E_{t+1} = F(E_t, M_t)$ with the scrapping decision (`repair_decision`) as the coupling mechanism, and reproduces the $\psi_t = f_t(0)$ rise → $\delta$-convergence in a ground-truth environment. Build 02's dynamical-systems module tracks $\psi_t$ across versions v1 → v2a → v3, plus the decreasing-even-moments check (Lemma 1) and the ln $f_t(0)$-vs-$t$ linearity test for the **autonomy** criterion. Crucial caveat: FTTL is a **non-autonomous** system — $D_t$ changes across versions (different training window, re-implemented preprocessing, one threshold change at v2), so $\psi_t$ is not a clean power series and must be read as a per-version signal, not a smooth trend. Build 03 (Unbiased Evaluation) is needed precisely because the OOT AUC is an example of the "hidden loop" condition — it masks performative risk growth. With only three model versions the $\psi_t$ trajectory is treated as a directional signal, reinforced by synthetic validation, not as proof.

**What to write in the dissertation**
Cite in the theory chapter as the unified mathematical model and the dynamical-systems backbone of Build 02. State: "Veprikov et al. (2025) formalise the repeated learning process as $E_{t+1} = F(E_t, M_t)$ and prove conditions under which feedback effects are 'hidden' from standard evaluation metrics. Our Build 03 unbiased evaluation addresses exactly this hiding condition: the OOT AUC on v1-logged data is a biased proxy for performative risk, consistent with their Theorem 3. Because FTTL is non-autonomous (the retraining operator changes per version), we diagnose the loop with per-version summary statistics rather than assuming the autonomy criterion of their Theorem 4."

---

### P30 · Pagan et al. (2023) — A Classification of Feedback Loops and Their Relation to Biases in Automated Decision-Making

**Citation**
Pagan, N., Baumann, J., Elokda, E., De Pasquale, G., Bolognani, S. & Hannák, A. (2023). "A Classification of Feedback Loops and Their Relation to Biases in Automated Decision-Making Systems." *arXiv preprint*, arXiv:2305.06055.

**Link** → https://arxiv.org/abs/2305.06055
**Citations** ≈ 30+ (Semantic Scholar, 2026) · **Venue** arXiv preprint (cs.LG / FAccT-adjacent)

**Why this paper matters**
The paper's own abstract states: "a rigorous theoretical understanding of the feedback dynamics in ML-based decision-making systems is currently missing." It then provides exactly that via control theory. For Build 02 (Loop Detection), knowing **which type** of feedback loop is operating determines which detection test is most powerful.

**Summary**
Uses dynamical-systems / control-theory language to classify ML feedback loops into formal types based on the direction and timescale of the feedback signal. Shows that different loop types produce characteristically different bias signatures in the learned model and in outcome distributions. Derives conditions under which each loop type leads to persistent bias amplification vs. convergence to an unbiased equilibrium. Provides a taxonomy that researchers can use to identify which type of loop applies to their system from observable data.

**Key concept / formula**
Classifies loops by the sign and delay of the feedback gain: a loop with positive gain and short delay (the total loss case — scrapping immediately forces label=1) is the "amplifying feedback" type, which the paper proves converges to a maximally biased fixed point under any learning rate. A loop with negative gain (error-correcting) is self-stabilising. The gain sign is identifiable from the cross-correlation between model score and label at the next timestep.

**How to apply at Insurance Company.**
Step 1 of SFPDetector (temporal prediction correlation) can be re-grounded in this taxonomy: rising cross-version Spearman rank correlation is the empirical signature of a positive-gain amplifying loop. Step 4 (segment blind spots) corresponds to the paper's "one-sided selection" loop subtype. Cite this to give the 4-step detector a unified theoretical basis rather than presenting each step as an ad-hoc heuristic.

**What to write in the dissertation**
Cite in the Build 02 methodology section when introducing the four-step SFP detection algorithm. State: "following Pagan et al. (2023), we classify the total loss SFP mechanism as an amplifying positive-gain feedback loop; each detection step targets the observable signature of this loop type."

---

### P31 · Mendler-Dünner et al. (2020) — Stochastic Optimization for Performative Prediction

**Citation**
Mendler-Dünner, C., Perdomo, J. C., Zrnic, T. & Hardt, M. (2020). "Stochastic Optimization for Performative Prediction." *Advances in Neural Information Processing Systems (NeurIPS) 33*.

**Link** → https://proceedings.neurips.cc/paper/2020/hash/33e75ff09dd601bbe69f351039152189-Abstract.html | arXiv: https://arxiv.org/abs/2002.09058
**Citations** ≈ 250+ (Semantic Scholar) · **Venue** *NeurIPS* (A* CORE ranking)

**Why this paper matters**
P15 (Perdomo et al.) defines performative risk and proves the gap between standard ERM and the performative optimum. This companion paper asks the next question: *how quickly does repeated retraining converge to the biased performative-stable point?* The convergence rate is a direct mathematical measure of **how fast** the SFP loop locks in.

**Summary**
Distinguishes two natural deployment strategies: (a) **greedy deploy** — deploy immediately after each stochastic gradient step; (b) **lazy deploy** — accumulate gradients on multiple samples before redeploying. Derives necessary and sufficient conditions for convergence to a performatively stable (PS) point under each strategy. Shows that sensitivity (how much the data distribution shifts per unit change in model parameters) and strong convexity jointly determine whether the loop stabilises or diverges. Generalises Perdomo et al. to the non-i.i.d., stochastic gradient regime. The lazy-deploy regime matches Insurance Company.'s batch (per-cycle) retraining, not per-claim updating.

**Key concept / formula**
Let ε be the sensitivity (ε = max‖θ₁−θ₂‖→0 W₂(D(θ₁), D(θ₂))/‖θ₁−θ₂‖) and β the strong convexity constant of the loss. Convergence condition: **ε/β < 1** — the distribution shift per parameter change must be smaller than the loss curvature. When ε/β ≥ 1 the loop diverges (the SFP amplification outpaces the model's self-correcting tendency).

**How to apply at Insurance Company.**
The ratio ε/β is the quantitative SFP loop coefficient for the total loss pipeline. Estimate ε empirically by measuring how much the scrapping-decision distribution shifts between v1 and v2a (Wasserstein distance on propensity scores). Estimate β from the Hessian of the v2a loss. If ε/β is close to or exceeds 1, the simulation (Build 01) and real-data evaluation (Build 03) will show runaway drift; if ε/β < 1, convergence to a biased-but-stable fixed point is expected. **Caveat:** the production model is a gradient-boosted tree (XGBoost), which is not globally strongly convex, so the ε/β guarantee is an **approximation, not a theorem** for FTTL — the loop's firmer footing is the model-class-agnostic forced-label mechanism (see P34/P35).

**What to write in the dissertation**
Cite alongside P15 in the theory section. State: "Mendler-Dünner et al. (2020) prove that convergence to a performatively stable point requires ε/β < 1, where ε is the distribution sensitivity and β is the loss curvature. We estimate this ratio empirically in Build 02 as a single-number loop severity score, while noting that the strong-convexity premise holds only approximately for the production XGBoost model."

---

### P32 · Jiang et al. (2019) — Degenerate Feedback Loops in Recommender Systems

**Citation**
Jiang, R., Chiappa, S., Lattimore, T., György, A. & Kohli, P. (2019). "Degenerate Feedback Loops in Recommender Systems." *Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society (AIES '19)*, pp. 383–390.

**Link** → https://dl.acm.org/doi/10.1145/3306618.3314288 | arXiv: https://arxiv.org/abs/1902.10730
**Citations** ≈ 200+ (Semantic Scholar, 2026) · **Venue** *AIES 2019* (AAAI/ACM Conference on AI, Ethics, and Society)

**Why this paper matters**
Provides formal **degeneracy conditions** — sufficient conditions under which a feedback loop provably converges to a state where the system only ever acts on a collapsed subset of its input space. In the total loss setting, degeneracy = the model eventually scraps every high-score vehicle regardless of true repairability, collapsing the observed label distribution to all-1 in the high-score band. Crucially, the paper also **disentangles two distinct phenomena** that are often conflated: the *echo chamber* effect (model amplifies its own past decisions) and the *filter bubble* effect (model becomes blind to items/claims outside the already-explored region). This maps directly onto Build 02's Step 1 (echo chamber: temporal score inflation) and Step 4 (filter bubble: segment blind spots).

**Summary**
Models the recommender-system feedback loop as a dynamical system where the recommendation policy influences user interest states, which in turn determine future feedback to the model. Formalises *system degeneracy* as convergence of the user-interest distribution to a degenerate point (all mass on one item type). Derives sufficient conditions for degeneracy under both deterministic and stochastic update dynamics. Disentangles the echo chamber (caused by the recommender's own bias reinforcing itself) from the filter bubble (caused by user preference drift under personalisation). Proposes practical mitigation: injecting diversity into recommendations to slow degeneracy onset — analogous to Build 05's garage-routing exploration budget.

**Key concept / formula**
Degeneracy condition (deterministic case): the Jacobian of the feedback map F at the fixed point has spectral radius > 1 — the system is locally unstable away from the degenerate attractor. Under stochastic dynamics, degeneracy occurs almost surely when the noise magnitude is below a threshold determined by the feedback gain. The echo chamber / filter bubble split: echo chamber arises when F amplifies the model's own output signal; filter bubble arises when F restricts the input distribution regardless of model output.

**How to apply at Insurance Company.**
Build 02 Step 1 (temporal score correlation) measures the echo chamber: rising Spearman rank correlation between v1 and v2a scores signals that the model is amplifying its own past decisions. Build 02 Step 4 (segment blind spots) measures the filter bubble: vehicle segments with near-100% scrap rates are the degenerate attractor — the model has "filtered out" any information about true repairability there. The degeneracy conditions provide a formal check: if the Jacobian spectral radius of the v1→v2a score-mapping exceeds 1 in any score band, that band is on a diverging path and is the priority target for Build 05's garage-routing exploration.

**What to write in the dissertation**
Cite in the Build 02 methodology section alongside P30 (Pagan et al.) to ground the echo-chamber vs. filter-bubble distinction formally. State: "following Jiang et al. (2019), we distinguish the echo chamber component of the SFP loop (captured by Step 1's temporal score correlation) from the filter bubble component (captured by Step 4's segment blind-spot analysis). The paper's degeneracy conditions provide a formal criterion for identifying which score bands are at risk of irreversible information collapse."

---

### P33 · Brown, Hod & Kalemaj (2022) — Performative Prediction in a Stateful World

**Citation**
Brown, G., Hod, S. & Kalemaj, I. (2022). "Performative Prediction in a Stateful World." *Proceedings of the 25th International Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR 151:6045–6061.

**Link** → https://proceedings.mlr.press/v151/brown22a.html | arXiv: https://arxiv.org/abs/2011.03885
**Citations** ≈ 90+ (Semantic Scholar, 2026) · **Venue** *AISTATS 2022* (A-ranked, PMLR)

**Why this paper matters**
P15 (Perdomo et al.) defines performative risk under the map D(θ) — the data distribution depends only on the **current** model θ. This is an incomplete model of the total loss SFP loop: in practice the distribution at retraining time depends not only on v2's model parameters but also on the **accumulated state** of forced-positive labels generated by all prior versions (v1 → v2). This paper extends performative prediction to D(θ, s_t), where s_t is the state of the population at time t. The state evolves across model generations, and convergence conditions now depend on **both** distribution sensitivity and state-transition dynamics. This directly explains why the Insurance Company. v3 retraining failed: the state (v1+v2 label contamination accumulated over the training window) was already entrenched, and retraining on contaminated labels could not escape the biased equilibrium regardless of v3's architecture.

**Summary**
Proposes a framework where the response of the target population to the deployed classifier is a function of both the classifier θ and the current state s_t (the distribution of the population itself). The state evolves according to a transition function g: s_{t+1} = g(s_t, θ_t). Two retraining algorithms are analysed: (1) **repeated risk minimisation** — retrain on the current state's data distribution; (2) **lazy variant** — retrain less frequently, allowing the state to settle. Derives necessary and sufficient conditions for convergence to a stable equilibrium near the performatively optimal classifier. Captures the phenomenon that distinct groups accumulate information and resources at different rates in response to the deployed classifier — translating to vehicle segments accumulating forced-positive labels at different rates under the scrapping policy.

**Key concept / formula**
Stateful performative map: D(θ, s_t), with state transition s_{t+1} = g(s_t, θ_t).
Convergence to equilibrium (θ*, s*) requires: sensitivity ε_θ (distribution shift per parameter change) and sensitivity ε_s (state shift per state change) jointly satisfy a contraction condition. When ε_s is large — i.e. the state itself is highly reactive to past model decisions — standard repeated retraining cannot escape the biased fixed point even if the model's per-step update is small.

**How to apply at Insurance Company.**
The state s_t is the accumulated label-contamination profile across vehicle segments: how many forced-positive labels have been added per segment across all prior model versions. After v1 and v2a both ran under precision-≥-0.985-tuned absolute cutoffs (each version's own τ_v; 0.872 is v2's), s is heavily contaminated in high-RTV / high-damage segments. Build 01 should simulate the stateful dynamics explicitly — not just the one-step v1→v2 transition, but the multi-step v1→v2→v3 trajectory — to show that even a well-specified v3 cannot escape the biased equilibrium once the state accumulation has reached a threshold. The state-dependent framework also explains why v2b (counterfactual with pre-ML data) partially resists SFP: including pre-ML labels in training effectively resets part of the contaminated state.

**What to write in the dissertation**
Cite alongside P15 in the theory section. State: "Perdomo et al. (2020) characterise the loop in terms of the current model alone. Brown et al. (2022) generalise this to a stateful setting where s_{t+1} = g(s_t, θ_t): the distribution depends on both the model and the accumulated history of prior scrapping decisions. This is the correct model for the Insurance Company. pipeline, where v1's forced-positive labels became part of the training state for v2a, and v2a's labels in turn become the state for v3. The failure of v3 retraining is consistent with Brown et al.'s result that a large state sensitivity ε_s can prevent convergence to the performatively optimal classifier."

---

### P34 · Taori & Hashimoto (2023) — Data Feedback Loops: Model-driven Amplification of Dataset Biases

**Citation**
Taori, R. & Hashimoto, T. B. (2023). "Data Feedback Loops: Model-driven Amplification of Dataset Biases." *Proceedings of the 40th International Conference on Machine Learning (ICML)*, PMLR 202:33883–33920.

**Link** → https://proceedings.mlr.press/v202/taori23a.html | arXiv: https://arxiv.org/abs/2209.03942
**Citations** ≈ 60+ (Semantic Scholar, 2026) · **Venue** *ICML 2023* (A* CORE ranking)

**Why this paper matters**
The convergence results in P29/P31/P33 assume a strongly convex loss, which the production XGBoost model does **not** satisfy — so their ε/β-style guarantees are approximations, not theorems, for FTTL. This paper is the empirical bridge that keeps the SFP claim on firm ground **without** any convexity assumption: it demonstrates directly, on deep non-convex models, that retraining a model on its own outputs amplifies bias, and that the amplification grows with the fraction of model-labelled data. v2a is trained *entirely* on v1-generated labels — the worst case this paper characterises.

**Summary**
Studies systems where a model's predictions are (partly) fed back as training labels for the next generation. Introduces a **uniform faithfulness** criterion: bias is stable across retraining generations only if the model's sampling of new labels is calibrated/faithful to the true distribution; when it is not, dataset biases compound generation over generation. Runs controlled experiments across vision and language tasks showing bias amplification that scales with the proportion of model-generated data in the training set, and shows that faithfulness-restoring interventions (resampling, calibration) slow or halt the amplification.

**Key concept / formula**
Amplification grows monotonically with the model-labelled fraction ρ of the training set; the stable (non-amplifying) case requires *uniform faithfulness* of the model's induced label distribution to the true one. For FTTL, ρ = 1 for v2a (all training labels are v1-generated), the regime with maximal amplification and no faithfulness guarantee.

**How to apply at Insurance Company.**
Gives Build 02 a **model-agnostic stability diagnostic** that does not depend on strong convexity: compare the realised v2a forced-label distribution against the v1 score distribution that generated it — a faithfulness gap is the amplification signature. Because FTTL's ρ = 1, cite this as the reason the loop is expected to amplify regardless of the tree-ensemble's non-convexity, scoping the dissertation's formal claim to the **label-generation mechanism** (§2.2 of `problem.md`) rather than to a convergence theorem for XGBoost.

**What to write in the dissertation**
Cite alongside P15/P29/P31 when handling the non-convexity objection. State: "Because the production model is a gradient-boosted tree, the strong-convexity premise of the performative-prediction convergence results holds only approximately. Taori & Hashimoto (2023) show empirically, with no convexity assumption, that retraining on model-generated labels amplifies bias in proportion to the model-labelled fraction — which for v2a is unity — so our formal claim is scoped to the model-class-agnostic forced-label mechanism rather than to a convergence guarantee for the specific estimator."

---

### P35 · Adam et al. (2022) — Error Amplification When Updating Deployed Machine Learning Models

**Citation**
Adam, G. A., Chang, C.-H. K., Haibe-Kains, B. & Goldenberg, A. (2022). "Error Amplification When Updating Deployed Machine Learning Models in Human-in-the-Loop Systems." *Proceedings of the Conference on Health, Inference, and Learning (CHIL)*. (Related work: "Hidden Risks of Machine Learning Applied to Healthcare," MLHC 2020.)

**Link** → CHIL 2022 proceedings (PMLR); related MLHC 2020 paper "Hidden Risks of Machine Learning Applied to Healthcare" → https://arxiv.org/abs/1909.03095 · *(confirm exact CHIL 2022 PMLR/DOI entry via UoB library — citation details to be verified)*
**Citations** ≈ 40+ (Semantic Scholar, 2026) · **Venue** *CHIL 2022* (health ML)

**Why this paper matters**
The applied precedent closest in structure to FTTL. In a real clinical (ICU) decision system, a deployed non-linear model's **false positives propagated into its next training set** and the error amplified across updates — structurally identical to a false-positive scrap forcing $\tilde{Y}=1$ that then trains v2a. It demonstrates that the SFP mechanism is not a theoretical artefact of convex analysis but an *observed* failure mode of deployed non-convex models updated on their own influenced data.

**Summary**
Shows that when a deployed model influences the data it is later retrained on (via human-in-the-loop decisions that follow the model's recommendations), naive periodic updating amplifies the model's existing errors rather than correcting them. Characterises the conditions under which this "error amplification" occurs, and evaluates mitigation via holding out un-influenced data or explicitly modelling the selection induced by the deployed model.

**Key concept / formula**
Error amplification: retraining on data whose labels/selection were shaped by the previous model version increases, rather than decreases, the model's error on the true distribution — the empirical analogue of a positive-gain performative loop (cf. P30's amplifying loop type).

**How to apply at Insurance Company.**
Cite as the real-world evidence that FTTL's forced-label loop is a recognised, documented failure mode — not a synthetic-only concern. Motivates Build 03's insistence on evaluating against garage-verified (un-influenced) rows and Build 06's use of un-influenced signal (garage outcomes, bounding arguments) rather than naive retraining on the contaminated log.

**What to write in the dissertation**
Cite in the problem-framing and related-work sections as the applied precedent. State: "Adam et al. (2022) document error amplification in a deployed clinical model whose recommendations shaped its own future training data — the same structure as the FTTL forced-label loop, and evidence that the mechanism manifests in real non-convex systems, not only under the convex assumptions of the performative-prediction theory."

---

## Part 10 — Model Interpretability / Feature Attribution (2017–present)

---

### P39 · Lundberg & Lee (2017) — A Unified Approach to Interpreting Model Predictions (SHAP)

**Citation**
Lundberg, S. M. & Lee, S.-I. (2017). "A Unified Approach to Interpreting Model Predictions." *Advances in Neural Information Processing Systems (NeurIPS/NIPS)* 30, 4765–4774.

**Link** → https://arxiv.org/abs/1705.07874 | https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html
**Citations** ≈ 30,000+ (Google Scholar, 2026) · **Venue** *NeurIPS 2017* (A* CORE ranking) — **the canonical SHAP origin paper**

**Why this paper matters**
This is the source paper for **SHAP (SHapley Additive exPlanations)** — the feature-attribution method used in the SHAP-DiD analysis (notebook `04_02`). The dissertation does not merely *use* SHAP as a black box; the SHAP-DiD probe treats each feature's SHAP contribution as the outcome variable in a difference-in-differences design across model versions, so the theoretical guarantees established here (local accuracy, consistency, uniqueness) are what make it valid to compare a feature's attribution *across* v2a and v3a. Without a consistent attribution method, a shift in a feature's importance between versions could be an artefact of the explainer rather than a real change in model reliance.

**Summary**
Unifies six existing feature-attribution methods (LIME, DeepLIFT, Layer-Wise Relevance Propagation, Shapley regression/sampling values, QII) under a single class of **additive feature-attribution** models. Proves that within this class there is a **unique** solution satisfying three desirable properties — local accuracy, missingness, and consistency — and that this solution is the classical Shapley value from cooperative game theory. Introduces SHAP values as this unique attribution and gives model-agnostic (KernelSHAP) and model-specific approximations. The tree-specific fast exact algorithm (TreeSHAP), relevant to the production XGBoost model, is developed in the follow-up Lundberg et al. (2020), *Nature Machine Intelligence*.

**Key concept / formula**
Additive feature-attribution explanation model:
$$g(z') = \phi_0 + \sum_{i=1}^{M} \phi_i z'_i$$
where the SHAP value for feature $i$ is the Shapley value
$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!\,(|F| - |S| - 1)!}{|F|!}\,\big[f_x(S \cup \{i\}) - f_x(S)\big].$$
**Consistency** guarantees that if a model changes so that a feature's marginal contribution increases in every coalition, its SHAP value cannot decrease — the property that licenses cross-version comparison in SHAP-DiD.

**How to apply at Insurance Company.**
In Build 04 (`04_02`), decompose each model version's score into per-feature SHAP contributions, then track how a feature's mean contribution shifts across versions (**v2a → v3a**, following the SHAP-DiD specification) as the DiD outcome — testing whether the SFP loop concentrates the model's reliance onto particular features. Two caveats to state: (a) the production scores are **uncalibrated**, so SHAP contributions are attributions on the raw score scale, not on a probability; (b) the real v2 top feature is `location_Home` (from actual feature importance, not SHAP), and versions differ in feature *set*, so a like-for-like SHAP-DiD comparison must restrict to the common feature set.

**What to write in the dissertation**
Cite as the methodological source for every SHAP result. State: "Feature attributions are computed with SHAP (Lundberg & Lee, 2017), the unique additive attribution satisfying local accuracy and consistency; the consistency property is what permits comparing a feature's attribution across model versions in the SHAP-DiD probe. For the gradient-boosted production model we use the exact tree estimator of Lundberg et al. (2020)." Flag the parallel-trends assumption of the DiD layer as the limitation to defend, not model autonomy.

---

### P40 · Lundberg et al. (2020) — From Local Explanations to Global Understanding with Explainable AI for Trees (TreeSHAP)

**Citation**
Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N. & Lee, S.-I. (2020). "From Local Explanations to Global Understanding with Explainable AI for Trees." *Nature Machine Intelligence*, 2(1), 56–67.

**Link** → https://doi.org/10.1038/s42256-019-0138-9 | arXiv: https://arxiv.org/abs/1905.04610
**Citations** ≈ 10,000+ (Google Scholar, 2026) · **Journal** *Nature Machine Intelligence* (IF ≈ 18) — **the TreeSHAP paper; the exact estimator used on the production XGBoost model**

**Why this paper matters**
This is the paper that makes SHAP **usable on the actual FTTL model**. The general SHAP formulation (P39) requires summing over all feature coalitions — exponential in the number of features, hence intractable to compute exactly for a real model. This paper gives **TreeSHAP**: a polynomial-time algorithm that computes *exact* SHAP values for tree ensembles (XGBoost, LightGBM, random forests). Every SHAP number produced in notebook `04_02` comes from this algorithm, not from the KernelSHAP approximation. It also supplies the two things the SHAP-DiD probe actually consumes: (1) **global** feature importance built up consistently from per-row local attributions, and (2) **SHAP interaction values**, which let the analysis check whether a version's reliance shift is a main effect or a feature-interaction effect.

**Summary**
Introduces an algorithm to compute exact Shapley values for trees in $O(TLD^2)$ time (T trees, L leaves, D depth) instead of the exponential general case. Shows that summarising many local explanations yields faithful **global** model understanding — global feature importance, dependence plots, interaction effects, and clustering of explanations — while retaining the local-accuracy and consistency guarantees of P39. Demonstrates the approach on medical-risk models, including a finding that consistent local attributions correct the known inconsistencies of standard tree feature-importance measures (gain, split count).

**Key concept / formula**
Exact tree Shapley values in polynomial time; global importance is the mean absolute SHAP value per feature,
$$I_j = \frac{1}{n} \sum_{i=1}^{n} \big| \phi_j^{(i)} \big|,$$
and **SHAP interaction values** $\phi_{i,j}$ decompose a prediction into main effects plus pairwise interactions (a Shapley-interaction extension of the P39 additive model). Crucially, the paper shows classical tree importances (gain/split-count) are **inconsistent** — a feature's importance can *fall* when the model is changed to rely on it more — which is exactly the failure mode that would invalidate cross-version comparison; TreeSHAP does not have it.

**How to apply at Insurance Company.**
This is the estimator behind Build 04 (`04_02`). Compute exact TreeSHAP values on each version's XGBoost model, then (a) build the global importance ranking $I_j$ to identify which features the model relies on (real v2 top feature: `location_Home`), and (b) feed per-feature SHAP contributions into the **v2a → v3a** SHAP-DiD design. Use SHAP **interaction values** to test whether an SFP-driven reliance shift concentrates on a single feature or on a feature pair. Cite this paper — not P39 alone — whenever a SHAP number is reported, because the exactness and consistency guarantees for *trees* specifically are what license the cross-version DiD comparison. Same two caveats carry over: uncalibrated raw-score scale, and restrict SHAP-DiD to the common feature set across versions.

**What to write in the dissertation**
Cite as the source of the actual attribution algorithm. State: "SHAP values for the gradient-boosted production models are computed exactly with the polynomial-time tree estimator of Lundberg et al. (2020). This paper also establishes that classical tree feature-importance measures (gain, split count) are inconsistent, whereas TreeSHAP is not — the property on which the validity of the cross-version SHAP-DiD comparison rests. Global importance is reported as the mean absolute SHAP value per feature, and SHAP interaction values are used to distinguish main-effect from interaction-driven reliance shifts."

---

## Part 11 — Concentration & Diversity Measures for Feature Importance (borrowed foundations)

> These four papers are **not about SFP**. They are the measurement foundation for the *concentration
> scalar* in the SHAP-DiD probe (notebook `04_02`). Once SHAP gives an importance-**share** vector
> `s` over features, one number must answer *"has importance piled onto a few features?"*. That number
> — the **Simpson index** `D = Σ sⱼ²` (headline), with **Shannon entropy** and **Gini** as audits —
> is borrowed from **ecology (diversity)** and **economics / information theory (concentration)**, the
> fields that rigorously solved *"how concentrated is a share vector?"*. **Rationale for reaching
> outside XAI:** the SHAP literature has no axiomatic theory of importance concentration (P41 is the
> only XAI-native source and is the weakest — a workshop note with no formula); the *legitimacy*
> criterion (which measures are even valid → Pigou–Dalton / Schur-convexity, still to be catalogued)
> lives in welfare economics, and the *unification* (why entropy and Simpson are one family) lives in
> ecology. The axioms transfer because all they require is a probability/share vector — no economic
> assumption (cardinal utility, interpersonal comparison) is imported. See the provenance note in
> `04_02` cell `md09`. **In the dissertation** the Simpson index is formally specified in
> `paper.mid.draft.md` §2.7 (definition, sample-size floor $1/n$, normalised $\text{Simpson}^\ast$ and
> the inverse-Simpson / $\exp H$ numbers-equivalents) and consumed by the SHAP-DiD statistic in §3.4.1.

---

### P41 · Saadallah (2025) — SHAP-Guided Regularization in Machine Learning Models

**Citation**
Saadallah, A. (2025). "SHAP-Guided Regularization in Machine Learning Models." *Late-breaking work, 3rd World Conference on eXplainable Artificial Intelligence (XAI 2025)*, Istanbul, 9–11 July 2025. Lamarr Institute for ML and AI, Dortmund. (Proceedings forthcoming.)

**Link** → local PDF `literatures/p41_SHAP_feature_concentration.pdf` (no stable DOI yet; XAI 2025 late-breaking track)
**Citations** ≈ negligible (2025 late-breaking) · **Venue** *XAI 2025* late-breaking / doctoral consortium — **weak anchor: workshop note, not peer-reviewed proceedings**

**Why this paper matters**
The only source from the **XAI literature itself** that names *importance concentration* as a SHAP diagnostic — it proposes a **SHAP entropy penalty** (sparsify attributions during training) and reports **Top-k Concentration** as an interpretability metric. The notebook's top-k audit statistic (`04_02`, cell `cd28b`) traces to it. It is cited as **precedent, not foundation**: it gives no formula for Top-k, k is an arbitrary cutoff, and the statistic is non-smooth — which is exactly why `04_02` demotes top-k to an audit and takes the Simpson index (P42) as headline.

**Summary**
Adds two SHAP-based regularisation terms to a LightGBM objective — an **entropy penalty** `L_entropy = −(1/N) Σ_i Σ_j p̂_ij log p̂_ij` (encourages sparse, concentrated attributions) and a **stability penalty** (consistent attributions across similar samples). Evaluates on 10 benchmark datasets, reporting RMSE/F1/AUC alongside SHAP Entropy, Top-k Concentration and Stability.

**Key concept / formula**
SHAP entropy penalty as above; **Top-k Concentration** = mass held by the k largest importance shares (no explicit formula given in the paper).

**How to apply at Insurance Company.**
Cite as the precedent that "importance concentration" is an established SHAP diagnostic. In `04_02` the top-k statistic is **re-implemented with an explicit normalisation the paper does not provide**, `(C_k − k/n)/(1 − k/n)` (floor is `k/n`, not `1/n`), and swept over k = 1..8 as a robustness audit only.

**What to write in the dissertation**
"Top-k concentration has been used as a SHAP interpretability metric (Saadallah 2025); we adopt it only as a k-robustness audit and take the Simpson index as the headline, because k is an arbitrary cutoff and top-k is non-smooth (flat to reallocations within the top-k set or within the tail)."

---

### P42 · Simpson (1949) — Measurement of Diversity

**Citation**
Simpson, E. H. (1949). "Measurement of Diversity." *Nature*, 163, 688.

**Link** → https://doi.org/10.1038/163688a0 | free reprint: https://people.wku.edu/charles.smith/biogeog/SIMP1949.htm
**Citations** ≈ 20,000+ (Google Scholar) · **Journal** *Nature* — **the source of `Σ pᵢ²` as a concentration index on a share vector**

**Why this paper matters**
The origin of `D = Σ pᵢ²` as a concentration / diversity index — the headline scalar in `04_02`. Gives the "why square?" answer: `D` is a **collision probability**, not squaring-for-emphasis.

**Summary**
Defines a measure of concentration for a population classified into groups: the probability that two individuals drawn at random belong to the *same* group. High `D` = concentrated (one group dominates); low `D` = diverse.

**Key concept / formula**
`D = Σ_j p_j²` = P(two independent draws hit the same class); range `[1/n, 1]`; **`1/D` = effective number of classes** (the "numbers equivalent").

**How to apply at Insurance Company.**
`D` is the headline concentration scalar `simpson` in `scalars()` (`04_02`, `cd10`); the SHAP-DiD footprint is computed on it. Its dual `inv_simpson = 1/D` is reported as the effective number of features.

**What to write in the dissertation**
"Importance concentration is summarised by the Simpson index `D = Σ sⱼ²` (Simpson 1949) — the probability that two importance draws land on the same feature — whose inverse `1/D` is the effective number of features."

---

### P43 · Hill (1973) — Diversity and Evenness: A Unifying Notation

**Citation**
Hill, M. O. (1973). "Diversity and Evenness: A Unifying Notation and Its Consequences." *Ecology*, 54(2), 427–432.

**Link** → https://doi.org/10.2307/1934352 | free PDF: https://biocomparison.ucoz.ru/_ld/0/78_hill_obzor.pdf
**Citations** ≈ 6,000+ (Google Scholar) · **Journal** *Ecology* (ESA flagship) — **the paper that unifies Simpson, Shannon and richness as one Rényi/Hill family**

**Why this paper matters**
This is what **licenses the claim that Shannon entropy and the Simpson index are two points of a single family**, not an arbitrary grab-bag. Hill relates Simpson (order q=2), Shannon (q=1) and species richness (q=0) to Rényi's generalized entropy, giving each an **effective-number** form indexed by an *order q* that dials how much weight the dominant classes receive.

**Summary**
Introduces the **Hill numbers** `ᴺq = (Σ_j p_j^q)^{1/(1−q)}` — a continuum of "effective number of species" measures. q=0 counts nonzero classes; q=1 → exp(Shannon); q=2 → 1/Simpson. Larger q weights common classes more.

**Key concept / formula**
`ᴺq = (Σ_j p_j^q)^{1/(1−q)}`; the order `q` is a sensitivity dial (q→∞ is governed by the single largest share; q→0 ignores magnitudes).

**How to apply at Insurance Company.**
Justifies reporting **entropy (q=1) and Simpson (q=2) together** in `04_02`; `eff_features = exp(H)` and `inv_simpson = 1/D` are the Hill effective numbers at q=1 and q=2. Top-k is *not* a member of this family (it is an order-statistic sum), which is a further reason it is audit-only.

**What to write in the dissertation**
"Shannon entropy and the Simpson index are the q=1 and q=2 members of the Hill/Rényi family of concentration measures (Hill 1973); each is reported with its effective-number dual, `exp(H)` and `1/D`."

---

### P44 · Jost (2006) — Entropy and Diversity

**Citation**
Jost, L. (2006). "Entropy and diversity." *Oikos*, 113(2), 363–375.

**Link** → https://doi.org/10.1111/j.2006.0030-1299.14714.x
**Citations** ≈ 4,000+ (Google Scholar) · **Journal** *Oikos* — **the modern standard on why to report effective numbers, not raw entropies**

**Why this paper matters**
Clarifies the distinction the notebook relies on: **entropies (Shannon `H`, Simpson `D`) are not themselves on a ratio scale** — their **numbers-equivalents** (`exp(H)`, `1/D`) are. This is why `04_02` reports `eff_features` and `inv_simpson` as the human-readable outputs rather than the raw indices.

**Summary**
Shows that raw diversity indices behave non-intuitively (doubling "true" diversity does not double `H`), whereas their numbers-equivalents (the Hill numbers of P43) do. Recommends converting any entropy to its effective number before interpretation or comparison.

**Key concept / formula**
"True diversity" = `exp(Shannon)` or `1/Simpson` (numbers equivalents); the raw entropy is only a transform of it, not a diversity itself.

**How to apply at Insurance Company.**
Motivates reporting `eff_features = exp(H)` and `inv_simpson = 1/D` alongside the raw scalars in `04_02`, and interpreting concentration shifts on the effective-number scale.

**What to write in the dissertation**
"Following Jost (2006), effective numbers of features (`exp(H)`, `1/D`) are reported alongside the raw indices, since entropies are not themselves on a ratio scale and only their numbers-equivalents are directly comparable."

---

## Paper Role Classification

> A paper can belong to multiple categories. See the **Paper Category Framework** section at the top for category definitions.
> "Key limitation for Allianz" explains why the paper alone cannot fully solve the Fast Track Total Loss SFP problem.

| # | Short Title | Define | Detect | Mitigate | Key limitation for Allianz total loss problem |
|---|-------------|:------:|:------:|:--------:|------------------------------------------------|
| P1 | Potential Outcomes | ✓ | | | Abstract notation only — no estimation strategy; requires a companion estimator (→ P3, P4) |
| P2 | Estimating Causal Effects | ✓ | | | Assumes ignorability holds; cannot fix it when unobservables drive scrapping (→ P6) |
| P3 | IPS Estimator | | ✓ | ✓ | Assumes known propensity; under hard threshold e(x)∈{0,1} → degenerate weights; soft-score workaround needed |
| P4 | Propensity Score | | ✓ | ✓ | Matching on propensity doesn't recover oracle for scrapped cars — overlap fails where e(x)=1 (→ P27) |
| P5 | Causal Diagrams / DAGs | ✓ | | | Acyclic assumption; SFP loop is a *cycle* — standard DAG identification breaks; dynamic causal modelling needed |
| P6 | Sample Selection Bias | ✓ | ✓ | ✓ | Assumes label is *missing* when unselected; in our case it is *forced to 1* — Heckman correction addresses missingness, not contamination |
| P7 | Econometrics of Program Eval. | | ✓ | | Survey — parallel trends (DiD) and continuity (RDD) assumptions must be separately verified on our data |
| P8 | Mostly Harmless Econometrics | | ✓ | | Textbook — RDD requires smooth covariate distribution around 0.872; score bunching near threshold may violate this |
| P9 | Fraud Detection Survey | ✓ | | | Domain is fraud, not total loss; under-labelling framing doesn't map to our over-labelling / forced-positive structure |
| P10 | Big Data's Disparate Impact | ✓ | | | Legal/regulatory framing only; no algorithmic correction; UK FCA rules differ from US EEOC disparate impact standard |
| P11 | To Predict and Serve? | ✓ | ✓ | | Empirical analogy (policing ≈ scrapping) but labels are *missing* there vs. *forced* here; oracle is recoverable in policing, not in total loss |
| P12 | Runaway Feedback Loops | ✓ | ✓ | ✓ | Remedies assume cheap exploration; garage routing is expensive — cost-benefit not modelled (→ P13, P19) |
| P13 | Cost of Fairness (KDD '17) | ✓ | | ✓ | Formalises the trade-off but no closed-form solution for our precision ≥ 0.985 hard constraint |
| P14 | Delayed Impact of Fair ML | ✓ | ✓ | ✓ | Requires knowing the "benefit function" mapping selection → future outcome; permanently scrapped cars have no observable future outcome |
| P15 | Performative Prediction | ✓ | ✓ | | Theory only — proves the loop exists and characterises convergence; provides no detection test or debiasing algorithm |
| P16 | Double/Debiased ML | | ✓ | ✓ | Requires overlap (common support); high-score cars scrapped at near-100% rate → no overlap → DML breaks in the tail |
| P17 | Hidden Technical Debt in ML | ✓ | | | Taxonomy paper — identifies feedback loops as tech debt but provides no quantification or correction method |
| P18 | Underspecification in ML | ✓ | ✓ | | Diagnoses the problem but solution ("stress tests") is underspecified for structured selection bias with forced labels |
| P19 | Thompson Sampling | | | ✓ | Original Beta-Bernoulli formulation; no cost model — routing a car to garage is not free (→ P13 for cost-benefit) |
| P20 | UCB1 | | | ✓ | Frequentist, no prior — cannot incorporate domain knowledge about cost asymmetry; precision floor not modelled |
| P21 | Tutorial on Thompson Sampling | | | ✓ | Contextual extension assumes rewards are observed; scrapped cars' true outcomes are never observed even after routing |
| P22 | Equality of Opportunity | ✓ | ✓ | ✓ | Post-processing threshold adjustment assumes calibrated scores; production model is uncalibrated → cannot apply directly |
| P23 | Fairness and ML (book) | ✓ | | | Impossibility theorem shows no single metric resolves all fairness criteria — does not resolve which criterion to prioritise |
| P24 | Measure and Mismeasure of Fairness | ✓ | ✓ | | Proposes conditional use accuracy equality but assumes calibrated model; our uncalibrated XGBoost violates this precondition |
| P25 | EU AI Act | ✓ | | | Regulatory mandate — specifies *what* must be achieved (Art. 9–10), not *how* to achieve it technically |
| P26 | Bandit Algorithms (book) | | | ✓ | General theory — contextual bandit (Ch. 36) assumes reward is always observed; we permanently lose oracle for scrapped cases |
| P27 | Selective Labels Problem | ✓ | ✓ | ✓ | Contraction requires multiple concurrent heterogeneous decision-makers; we have a single sequential model pipeline — precondition fails |
| P28 | PU Learning Survey | ✓ | ✓ | ✓ | SCAR assumption definitively violated; SAR methods need to estimate e(x) — our advantage is e(x) is known, but SCAR-based prior estimators (e.g. c = Pr(s=1)/α) are invalid |
| P29 | Mathematical Model of Hidden Feedback Loop (Veprikov) | ✓ | ✓ | | Provides the dynamical-systems unification (ψ_t = f_t(0) trajectory); no off-the-shelf implementation — must be instantiated for the total loss domain, and FTTL's non-autonomy breaks the clean power-series (autonomy) case |
| P30 | Classification of Feedback Loops | ✓ | ✓ | | Taxonomy and gain-sign classification; does not provide a correction or debiasing algorithm |
| P31 | Stochastic Opt. for Performative Pred. (Mendler-Dünner) | ✓ | ✓ | | Convergence condition ε/β < 1 requires estimating Wasserstein sensitivity ε between model versions (needs two deployed versions); strong-convexity premise only approximate for XGBoost |
| P32 | Degenerate Feedback Loops in Recommender Systems | ✓ | ✓ | | Recommender-system setting (user preference drift); forced-label structure unique to total loss must be analogised, not directly applied |
| P33 | Performative Prediction in a Stateful World | ✓ | ✓ | | Convergence conditions require estimating state sensitivity ε_s from multi-generation logs (needs v1→v2→v3 data); no debiasing or correction method |
| P34 | Data Feedback Loops (Taori & Hashimoto) | ✓ | ✓ | | Empirical, model-agnostic bias-amplification result; gives a faithfulness diagnostic but no debiasing algorithm — quantifies amplification, does not remove it |
| P35 | Error Amplification When Updating Deployed Models (Adam et al.) | ✓ | ✓ | | Applied precedent (clinical); demonstrates the failure mode in a real non-convex system but supplies no total-loss-specific correction |
| P39 | A Unified Approach to Interpreting Model Predictions — SHAP (Lundberg & Lee) | | ✓ | | Attribution tool, not a loop theory or correction; explains *what* the model relies on, not *why* the loop forms or how to break it. SHAP-DiD inherits the DiD parallel-trends assumption; scores are uncalibrated so contributions are on the raw-score scale |
| P40 | Explainable AI for Trees — TreeSHAP (Lundberg et al.) | | ✓ | | Exact tree estimator that makes SHAP tractable on the XGBoost model, but still attribution only — no loop correction. Consistency fixes the cross-version comparison; same uncalibrated-score / common-feature-set caveats as P39 |

---

## Problem Type Taxonomy — Which Existing Framework Fits?

*Cross-reference: `problem.md` §2.4. For the full formal derivation (notation, label generation mechanism, SCAR violation proof), see `problem.md` §2–2.6.*

| Candidate Framework | What Fits | What Doesn't / What's Missing |
|---------------------|-----------|-------------------------------|
| **Selective Labels (P27, Lakkaraju et al. 2017)** | The structure where $D_i$ determines label availability; evaluation on the $\{D_i=0\}$ subset being biased | When $D_i=1$, the label is not NA but forced to 1 — this is "contamination," not "missingness." The Contraction technique requires multiple simultaneous heterogeneous decision-makers and does not apply directly to the sequential model-version structure (see `notes/p27.md` §3-2) |
| **PU Learning (P28, Bekker & Davis 2020)** | The "contaminated label" structure where it is unknown whether $D_i=1$ rows are true positives — a more accurate mapping | **SCAR is definitively violated** (see `problem.md` §2.6). The correct assumption is **structured SAR**: the propensity score $e(x) = \Pr(D=1 \mid X=x, Y=1) \approx \mathbb{1}[\hat{f}(x) \geq 0.872]$ is a deterministic step function of $x$ — the labelling probability is 0 or 1 depending entirely on features, not constant. SCAR-based prior estimation ($c = \Pr(s=1)/\alpha$) is therefore invalid. However, because the labelling mechanism is *known* (the model threshold), $e(x)$ can be computed directly, enabling IPS-corrected evaluation and debiased training despite the violation. |
| **Performative Prediction (P15, Perdomo et al. 2020)** | Precisely matches the core mechanism where the model's predictions change the data distribution itself — the best theoretical backbone for explaining the loop dynamics | Does not provide concrete evaluation or correction algorithms — answers "why the loop forms," not "how to fix it" |
| **Runaway Feedback Loops (P12, Ensign et al. 2018)** | Structurally identical to the "see more → find more → send more" loop in predictive policing | The proposed remedies (random exploration/audits) are highly costly in our domain (sending a car to the garage is itself a cost) — cost-benefit analysis required in Build 05 |

**Working conclusion:** Our problem is most accurately characterised as a **performative prediction loop operating over PU-contaminated labels**. P27 remains a valid supporting tool for explaining why evaluation is biased, but it is not the primary explanation of the core mechanism (label contamination + self-reinforcement). This working assumption may be revised as further papers are read.

---

## Quick Reference Table

| # | First Author | Year | Short Title | Role | Venue | URL | ~Citations |
|---|-------------|------|-------------|------|-------|-----|-----------|
| P1 | Neyman | 1923/1990 | Potential outcomes | Define | Stat. Science | doi:10.1214/ss/1177012031 | 3,000 |
| P2 | Rubin | 1974 | Estimating causal effects | Define | J. Educ. Psychol. | doi:10.1037/h0037350 | 9,800 |
| P3 | Horvitz & Thompson | 1952 | IPS estimator | Detect + Mitigate | JASA | doi:10.1080/01621459.1952.10483446 | 5,000 |
| P4 | Rosenbaum & Rubin | 1983 | Propensity score | Detect + Mitigate | Biometrika | doi:10.1093/biomet/70.1.41 | 25,000 |
| P5 | Pearl | 1995 | Causal diagrams / DAGs | Define | Biometrika | doi:10.1093/biomet/82.4.669 | 5,000 |
| P6 | Heckman | 1979 | Sample selection bias | Define + Detect + Mitigate | Econometrica | doi:10.2307/1912352 | 29,000 |
| P7 | Imbens & Wooldridge | 2009 | Econometrics of program eval. | Detect | J. Econ. Lit. | doi:10.1257/jel.47.1.5 | 8,000 |
| P8 | Angrist & Pischke | 2009 | Mostly Harmless Econometrics | Detect | Princeton UP | press.princeton.edu | 30,000 |
| P9 | Phua et al. | 2010 | Fraud detection survey | Define | arXiv | arxiv.org/abs/1009.6119 | 795 |
| P10 | Barocas & Selbst | 2016 | Big Data's Disparate Impact | Define | CA Law Rev. | ssrn.com/abstract=2477899 | 2,500 |
| P11 | Lum & Isaac | 2016 | To Predict and Serve? | Define + Detect | Significance | doi:10.1111/j.1740-9713.2016.00960.x | 509 |
| P12 | Ensign et al. | 2018 | Runaway Feedback Loops | Define + Detect + Mitigate | FAccT | arxiv.org/abs/1706.09847 | 650 |
| P13 | Corbett-Davies et al. | 2017 | Cost of Fairness (KDD '17) | Define + Mitigate | KDD | dl.acm.org/doi/10.1145/3097983.3098095 | 1,445 |
| P14 | Liu et al. | 2018 | Delayed Impact of Fair ML | Define + Detect + Mitigate | ICML | arxiv.org/abs/1803.04383 | 491 |
| P15 | Perdomo et al. | 2020 | Performative Prediction | Define + Detect | ICML | arxiv.org/abs/2002.06673 | 325 |
| P16 | Chernozhukov et al. | 2018 | Double/Debiased ML | Detect + Mitigate | Econometrics J. | doi:10.1111/ectj.12097 | 6,000 |
| P17 | Sculley et al. | 2015 | Hidden Technical Debt in ML | Define | NeurIPS | proceedings.neurips.cc | 4,000 |
| P18 | D'Amour et al. | 2022 | Underspecification in ML | Define + Detect | JMLR | arxiv.org/abs/2011.03395 | 900 |
| P19 | Thompson | 1933 | Thompson Sampling | Mitigate | Biometrika | doi:10.1093/biomet/25.3-4.285 | 3,000 |
| P20 | Auer et al. | 2002 | UCB1 bandit algorithm | Mitigate | ML journal | doi:10.1023/A:1013689704352 | 7,000 |
| P21 | Russo et al. | 2018 | Tutorial on Thompson Sampling | Mitigate | FnT-ML | doi:10.1561/2200000070 | 3,000 |
| P22 | Hardt et al. | 2016 | Equality of Opportunity | Define + Detect + Mitigate | NeurIPS | arxiv.org/abs/1610.02413 | 5,000 |
| P23 | Barocas, Hardt & Narayanan | 2023 | Fairness and ML (book) | Define | MIT Press | fairmlbook.org | 4,000 |
| P24 | Corbett-Davies & Goel | 2023 | Measure and Mismeasure of Fairness | Define + Detect | JMLR | arxiv.org/abs/1808.00023 | 700 |
| P25 | EU Parliament | 2024 | EU AI Act | Define | EU OJ | eur-lex.europa.eu | — |
| P26 | Lattimore & Szepesvári | 2020 | Bandit Algorithms (book) | Mitigate | Cambridge UP | doi:10.1017/9781108571401 | 1,500 |
| P27 | Lakkaraju et al. | 2017 | Selective Labels Problem | Define + Detect + Mitigate | KDD | dl.acm.org/doi/10.1145/3097983.3098066 | 450 |
| P28 | Bekker & Davis | 2020 | PU Learning survey | Define + Detect + Mitigate | Machine Learning | doi:10.1007/s10994-020-05877-5 | 1,000 |
| P29 | Veprikov et al. | 2025 | Mathematical Model of Hidden Feedback Loop | Define + Detect | KAIS (Springer) | arxiv.org/abs/2405.02726 | 5 |
| P30 | Pagan et al. | 2023 | Classification of Feedback Loops | Define + Detect | arXiv | arxiv.org/abs/2305.06055 | 30 |
| P31 | Mendler-Dünner et al. | 2020 | Stochastic Opt. for Performative Pred. | Define + Detect | NeurIPS | arxiv.org/abs/2002.09058 | 250 |
| P32 | Jiang et al. | 2019 | Degenerate Feedback Loops in Recommender Systems | Define + Detect | AIES | dl.acm.org/doi/10.1145/3306618.3314288 | 200 |
| P33 | Brown et al. | 2022 | Performative Prediction in a Stateful World | Define + Detect | AISTATS | arxiv.org/abs/2011.03885 | 90 |
| P34 | Taori & Hashimoto | 2023 | Data Feedback Loops (bias amplification) | Define + Detect | ICML | arxiv.org/abs/2209.03942 | 60 |
| P35 | Adam et al. | 2022 | Error Amplification When Updating Deployed Models | Define + Detect | CHIL | (PMLR; MLHC'20 arxiv.org/abs/1909.03095) | 40 |
| P39 | Lundberg & Lee | 2017 | A Unified Approach to Interpreting Model Predictions (SHAP) | Detect | NeurIPS | arxiv.org/abs/1705.07874 | 30,000 |
| P40 | Lundberg et al. | 2020 | Explainable AI for Trees (TreeSHAP) | Detect | Nature Mach. Intell. | doi:10.1038/s42256-019-0138-9 | 10,000 |

---

*All citation counts are approximate Google Scholar / Semantic Scholar figures as of mid-2026. Regulatory documents (P25) do not have citation counts. P27 and P28 added 2026-06-15 following domain confirmation (total loss prediction). P29–P33 added 2026-06-23 to fill the mathematical SFP evaluation gap — a unified repeated-learning / dynamical-systems model (P29, Veprikov), loop-type classification (P30), convergence rates (P31, Mendler-Dünner), degenerate-loop / echo-chamber-vs-filter-bubble conditions (P32), and stateful performativity (P33). P34–P35 added 2026-07-07 — model-class-agnostic empirical bias amplification (P34) and an applied clinical precedent (P35) that keep the SFP claim on firm ground despite the production XGBoost model's non-convexity.*

**Numbering corrected 2026-07-07:** P29 = Veprikov (dynamical systems), P31 = Mendler-Dünner (convergence) — earlier drafts of this file and `literatures/compare.md` had these two swapped relative to the canonical numbering in `problem.md`, `literatures/notes/p29.md`, and the notebooks.

---
**Quick access for paywalled papers via UoB library:**
1. Go to https://www.bristol.ac.uk/library/
2. Search by DOI or title in the "Find a resource" search bar
3. Sign in with your UoB student credentials
4. Download PDF and save as `p{N}.pdf` in this folder

---
---

# Application Logic Grounded in the Research Papers

*Draft for review. This section turns the reading list into the analytical design for the
application — what each component must compute and **which paper licenses each choice** —
for the **total loss prediction** domain (not the original fraud framing). It is written
against the real dataset (`src/data/synthetic/`) and the real policy
(absolute scrap cutoff `score ≥ τ_v`, where `τ_v` is **tuned per version to hold precision ≥ 0.985** — 0.872 is v2's tuned value; two-generation training v1 → v2a/v2b). The
companion section "Application Implementation" below maps this onto the `src/` code.*

## 0. The one structural fact everything else follows from

The total loss pipeline is a **selective-labels system with irreversible, over-labelling
actions**:

```
score ≥ τ_v  → scrap     → observed_outcome forced to 1   (car gone; garage NEVER verifies → oracle permanently absent)
score <  τ_v → garage    → observed_outcome = true result  (reliable 0/1)

τ_v = per-version cutoff, tuned to hold precision ≥ 0.985 on validation; τ_v(v2) = 0.872
```

Two consequences drive the entire application:

1. **The label is a function of the decision, which is a function of the score** — so any
   metric computed on the production log is contaminated (Lakkaraju et al., **P27**;
   Heckman, **P6**). This is *over*-labelling (forced positives), the mirror image of the
   under-labelling in the fraud/policing analogues (Lum & Isaac **P11**, Ensign et al. **P12**).
2. **The oracle is destroyed, not merely missing.** No post-hoc audit can recover the true
   repair feasibility of a scrapped car. The only way to learn it is to *not scrap* — i.e.
   route some high-score cars to the garage on purpose (the exploration cost in **P13/P19**).

Every build below is one of four moves: **measure** the problem (00, 03), **prove** the
mechanism (01), **estimate** clean effects despite the bias (02, 04), or **break** the loop
(05, 06).

## 1. Paper → application-logic map (total loss framing)

| Build | Application must compute | Core papers | What changes vs. the fraud framing |
|---|---|---|---|
| **00 Audit** | `repair_decision` mix; scrap rate by score band; outcome observability mask (`decision==0`); enrichment join integrity | P17 (feedback as tech debt), P9 | Diagnostic is **scrap-rate monotonicity in score**, not investigation-rate; the "selection" is the irreversible scrap, recorded in `repair_decision` |
| **01 Simulation** | Re-run the v1→v2 loop on synthetic data; show scrap rate inflates (19%→21.5%) while true feasibility is fixed | P12 (Pólya urn), P15 (performative risk), P14 (long-run) | Urn "ball" = a scrapped car forcing label=1; the fixed point is **over-scrapping**, not over-patrolling. Because the cutoff is absolute, drift shows up as *more* scrapping (not a held-constant rate) |
| **02 Detection** | 4 falsifiable tests (below) on `model_v1_*` vs `model_v2a_*` | P15, P12, P4, P16, P27 | Step 2 becomes the **tautology check** `P(observed=1\|decision=1)=1.0`; Step 3 treats *scrapping* as the treatment |
| **02-02 Unbiased eval** | Selective-labels-corrected AUC on the **garage-only** subset, reweighted to the full population; PU class-prior `π̂` for scrapped cars | P27, P3 (IPS), P6, P28 (PU), P18 | Evaluate on `decision==0` rows (true labels) and reweight by inverse P(garage); **also correct the OOT set** (it is inside the v1 log → contaminated) |
| **02-03 Intervention** | Causal effect of *scrapping* on the forced-positive label; RDD at the **per-version cutoff τ_v** (v2 = 0.872); PSM/DML on propensity-to-scrap | P7, P8 (RDD/DiD), P4, P16, P5 (DAG) | The absolute cutoff τ_v (precision-tuned; 0.872 for v2) is a **textbook sharp RDD** — near-identical cars just above/below it. This is the cleanest natural experiment in the whole project |
| **02-04 Randomisation** | Policy that sends a budgeted fraction of high-score cars to the garage to recover oracle labels; regret vs cost | P19/P21 (Thompson), P20/P26 (UCB), P13 (cost of fairness), P14 | Exploration is **expensive and risky** (garage fee + possibly paying a true total loss's full value) — cost-benefit must be modelled explicitly, unlike cheap re-investigation |
| **03 Mitigation** | Debias next-gen training data: downweight/relabel forced positives; IPW for garage rows; PU-imputed counterfactuals for scrapped rows | P3, P4, P16, P5, P27, P28 | Don't just reweight — the forced `outcome=1` on scrapped rows is *wrong*, so PU relabelling/`π̂`-correction matters as much as IPW |
| **Ethics/Reg** | Disparate-impact check by `vehicle_make`/`damage_profile`; AI Act Art. 10 data-governance argument | P10, P22, P23, P24, P25 | "Protected segment" proxy is vehicle/damage profile, not postcode; the harm is **systematically over-scrapping** certain makes |

## 2. The detection logic (Build 02), restated for total loss

The four steps are the application's diagnostic core. Each is a falsifiable hypothesis on the
real columns:

1. **Temporal score correlation** (P15, P12). `spearman(model_v1_score, model_v2a_score)`
   trending → 1 across versions is the signature of approaching a *biased* performative fixed
   point — the models agree because they share v1's self-fulfilling history, not because they
   detect total loss better.
2. **Label-mechanism tautology** (P6, P27). Verify `P(model_v1_observed_outcome=1 | model_v1_decision=1) = 1.0`
   exactly (forced positive), and contrast with `P(outcome=1 | decision=0)` (garage truth).
   The gap **is** the label noise; it is not estimable away without intervention.
3. **Action–outcome confounding** (P4, P16). Treatment = *scrapping* (`decision`), not
   investigation. Estimate the propensity-to-scrap `e(X)=P(decision=1|X)` from pre-decision
   covariates (`repair_to_value_ratio`, `damage_severity`, `vehicle_age_years`, …), then the
   naive vs. IPW/DML gap quantifies how much "scrapping → outcome=1" is mechanical rather
   than real.
4. **Segment blind spots** (P11, P12). Find vehicle/damage segments scrapped at near-100%:
   these have *zero* surviving oracle labels, so the model can never be corrected there
   without Build 05's deliberate garage routing.

## 3. Why the absolute cutoff τ_v (v2 = 0.872) matters to the logic

Choosing an **absolute** cutoff (not a percentile) is what makes Builds 01–04 work. Note the
cutoff is not a fixed pipeline constant: each version is **re-tuned** to the smallest `τ_v`
that holds **precision ≥ 0.985** on validation, and 0.872 is the value that tuning produced
for v2. The precision constraint is what is held fixed; τ_v is what moves.

- **Build 01/02**: score drift is *visible* as scrap-rate inflation (a percentile rule would
  pin the rate and hide it). Because precision is re-pinned to ≥ 0.985 each version by
  re-tuning τ_v, the drift **cannot** show up in the precision metric — it surfaces only as
  the cutoff shifting and the scrap rate inflating.
- **Build 04 RDD**: a fixed score cutoff is a sharp discontinuity in treatment assignment —
  Imbens & Wooldridge (**P7**) / Angrist & Pischke (**P8**) RDD applies almost verbatim, with
  bandwidth around the analysed version's τ_v (0.872 for v2).
- **Caveat to carry into Build 03/06** (P24, P22): scores are **uncalibrated** (README), so
  using them as propensity weights injects bias. The application must either calibrate
  (Platt/isotonic) before reweighting or report this as a limitation.

---

# Application Implementation

*Draft for review. The recommended build order is: (1) validate each research step's logic
in the per-step notebooks under `test/builds/00…06` (the prototypes), then (2) promote the
stabilised logic into the `src/` application as Strategy classes. This section describes step
(2) — how the paper-grounded logic above is realised in `src/` (see `src/DESIGN.md`,
`src/STRUCTURE.md`).*

## 1. Why a Strategy pattern, and what it buys us

The application is built so the **only** thing that changes between the dissertation
(synthetic) and Insurance Company. (real) runs is the injected class — never the core logic. Three swap
axes (`src/DESIGN.md`):

| Axis | Interface | Synthetic now | Real later |
|---|---|---|---|
| Data loading | `DataLoader` | `SyntheticDataLoader` (reads `src/data/synthetic/parquet`) | `RealDataLoader` (Insurance Company. Parquet + DB tables) |
| Detection | `DetectionAlgorithm` | the 4 Build-02 steps | same algorithms, real columns |
| Policy / correction | `InvestigationPolicy`, `TrainingDataCorrector` | Thompson / IPW+PU | same, with real cost parameters |

This directly serves EU AI Act Art. 10/15 (**P25**): the detector + corrector become the
documented, re-runnable **post-market monitoring** component, swappable per model version.

## 2. Mapping the four "moves" onto the class skeleton

```
SFPPipeline (pipeline.py) — orchestrates one model-version transition (v1 → v2a)
  │
  ├─ DataLoader.load()                    → claims_v1_log (+ pre_v1 for v2b window)
  │
  ├─ SFPDetector.run(report)              → MEASURE + PROVE
  │     algorithm/temporal_corr.py        (P15, P12)  Step 1
  │     algorithm/label_mechanism.py      (P6, P27)   Step 2 — tautology check
  │     algorithm/action_outcome.py       (P4, P16)   Step 3 — propensity-to-scrap
  │     algorithm/blind_spots.py          (P11, P12)  Step 4
  │   → DetectionReport { sfp_detected, scrap_rate_drift, auc_inflation, blind_segments }
  │
  └─ if report.sfp_detected:  SFPMitigator.run(report)   → BREAK
        policy/thompson.py        (P19, P21, P13)  budgeted garage-routing of high-score cars
        corrector/ips_pu.py       (P3, P28, P27)   IPW garage rows + PU relabel scrapped rows
      → corrected training frame for the *next* model version
```

`ESTIMATE` (Build 04: RDD at the version's τ_v — 0.872 for v2, DiD, DML) lives as analysis modules the detector can call
to quantify effect sizes for the report; it is not on the live mitigation path.

## 3. Implementation decisions that matter

- **Observability mask is a first-class column, not a `fillna`.** Build the boolean
  `outcome_observed = (model_v1_decision == 0)` once at load time. Every estimator either
  restricts to it (Build 03 AUC) or models it (propensity, IPW). Never impute scrapped rows'
  outcome to 0 — and treat the forced 1 as *unverified*, per **P27/P28**.
- **Propensity = P(scrap | X) from pre-decision covariates only.** Estimate with
  cross-fitted LightGBM (**P16**); store `e_hat` on the frame so detector, evaluator and
  corrector share one definition. Check overlap before any matching (**P4**).
- **Two correctors, composed.** (a) IPW reweight garage rows back to the full distribution
  (**P3**); (b) PU class-prior `π̂` to relabel/soft-label scrapped rows (**P28**). Report
  `π̂` itself — it is the headline "how many repairable cars were scrapped" number for the viva.
- **The randomisation policy needs a cost model and a kill-switch.** Garage-routing a
  high-score car costs a real assessment + hire-car fee and risks paying a true total loss's
  full value; the budget `B` (cars/month the business will divert) is an input, and a hard
  floor ("never divert below score X") guards precision (**P13**, README precision ≥ 0.985).
- **Calibrate before you weight.** Wrap the raw model in a calibration step (Platt/isotonic)
  *inside* `DataLoader` or the corrector, since uncalibrated scores bias both IPW and any
  probability read of the per-version cutoff τ_v (0.872 for v2) (**P24**).

## 4. Synthetic → real cutover checklist

1. Implement `RealDataLoader` to yield the same schema (`synth_data_structure.md`) — map
   Insurance Company. columns to `model_v{1,2}_*`, `repair_decision`, `*_observed_outcome`.
2. Confirm the real scrap policy is an absolute per-version cutoff τ_v tuned to precision
   ≥ 0.985 (0.872 for v2) and recover each version's tuned value into `SCRAP_THRESHOLD`; the
   RDD bandwidth in Build 04 keys off the analysed version's τ_v.
3. Re-fit propensity/calibration on real data; re-check overlap and `π̂` plausibility.
4. Everything downstream (detector, mitigator, report) runs unchanged — that is the whole
   point of the Strategy split.
