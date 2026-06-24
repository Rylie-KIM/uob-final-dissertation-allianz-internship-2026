# Reading List — Identifying and Mitigating Self-Fulfilling Prophecy Loops in ML
**MSc Data Science Dissertation · University of Bristol · Insurance A Cop. UK (Operations team)**
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
| **Causal identification** | Investigation propensity as treatment variable; DiD/RDD on model version deployment | Scrapping decision as treatment variable; RDD around 90th percentile score threshold is directly applicable |
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
| How to estimate the treatment effect of the action? | DiD: model deployment date as natural experiment; RDD: investigation threshold | RDD: 90th percentile scrapping threshold is a **natural regression discontinuity** — near-identical claims just above/below threshold |
| How to debias training data? | Add randomly investigated claims; IPW re-weighting | PU class-prior estimation for scrapped rows; IPW for garage rows; counterfactual imputation (**P27, P28**) |
| How to break the loop going forward? | Random investigation of low-score claims (cheap exploration) | Deliberate garage routing of high-score claims (costly exploration — must model cost-benefit explicitly) |
| What paper is the formal backbone? | Ensign et al. (**P12**) + Perdomo et al. (**P15**) | All of the above **plus** Lakkaraju et al. (**P27**) for oracle-absence + Bekker & Davis (**P28**) for forced-positive labels |
| How to handle OOT evaluation under SFP? | Standard OOT AUC is valid — labels are eventually confirmed | OOT labels are SFP-contaminated; apply selective-labels-corrected AUC (**P27**) to OOT set. Also: 2-month maturation buffer creates a blind spot at deployment boundary — the most recently SFP-affected data is invisible during both training and OOT evaluation. |

---

## Document Status

> **⚠️ Domain correction — updated 2026-06-15**
>
> Originally drafted with a **fraud detection / investigation-based SFP** assumption. After reviewing `src/data/synthetic/synth_data_structure.md` (business logic confirmed via internal meetings), the actual domain is:

| Field | Detail |
|-------|--------|
| **Confirmed domain** | **Total Loss Prediction** — model predicts whether a damaged car should be scrapped (`total_loss=1`) or sent to garage (`total_loss=0`) |
| **Actual SFP mechanism** | `total_loss=1` → car scrapped immediately → label forced to 1 (self-fulfilling). `total_loss=0` → garage → true repair outcome observed. |
| **Structural difference from fraud** | Oracle (`garage_outcome`) is **permanently unobservable** for scrapped cars — the car is physically gone. This is a **selective labels** problem (→ P27), not an investigation-bias problem. |
| **Label noise structure** | `decision=1` rows always receive label 1 (forced positive; true repair outcome unknown). `decision=0` rows receive true labels. This asymmetric noise is the SFP mechanism. |
| **Language note** | "fraud label" → "total loss label"; "investigation" → "scrapping decision"; "investigated claims" → "scrapped / sent-to-garage"; "postcode risk" → "vehicle make / damage profile / `repair_to_value_ratio`" — update throughout "How to apply" sections |
| **Priority changes** | P27 (Selective Labels — NEW, KDD'17) enters **#3**; P6 (Heckman) rises to **#4**; P28 (PU Learning survey — NEW) enters **#8**; P11 (Lum & Isaac) drops to **#6**; P21, P7, P5 drop out of top 10 |
| **Business cost structure (README)** | False positive = scrapping a repairable car → insurer pays **full car value** (vs. lower repair cost). SFP loop deepening → systematic false positive increase → direct financial loss. Company prefers repairable claims. v2 underperformed → SFP suspected → project parked. This reframes SFP as a **cost containment failure**, not just a technical bias problem. |
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
Every causal claim in the dissertation rests on the potential-outcomes (PO) notation introduced here. Without PO, there is no rigorous way to say "what would the fraud rate have been had this claim been investigated?"

**Summary**
Neyman introduced the notation Y(1) and Y(0) for the outcome a unit *would* have under treatment and control. The Average Treatment Effect (ATE) = E[Y(1) − Y(0)] is the quantity the dissertation is ultimately trying to estimate when asking: does investigation cause more fraud to be discovered (or merely reflect it)?

**Key concept / formula**
$$\tau = \mathbb{E}[Y_i(1) - Y_i(0)]$$
The fundamental problem of causal inference: we observe at most one potential outcome per unit. All identification strategies (IPW, DiD, RDD) are solutions to this problem.

**How to apply at Insurance A Cop.**
Frame every analysis as a PO problem: for each motor claim, define Y(1) = fraud discovered if investigated, Y(0) = fraud discovered if not investigated. Unobserved counterfactuals are estimated via the methods in Builds 04–06.

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
Rubin operationalised Neyman's notation into a usable framework for observational data and formalised SUTVA — the assumption that one claim's investigation does not affect another's outcome. This is the "Rubin" in Neyman–Rubin.

**Summary**
Defines the Stable Unit Treatment Value Assumption (SUTVA): no interference between units and no hidden versions of treatment. Introduces ignorability (unconfoundedness): treatment assignment is independent of potential outcomes given observed covariates. If ignorability holds, we can estimate ATE from observational data.

**Key concept / formula**
SUTVA: $Y_i = Y_i(W_i)$ — each unit's outcome depends only on its own treatment.
Ignorability: $(Y(0), Y(1)) \perp W \mid X$ — conditional on features, who gets investigated is "as good as random."

**How to apply at Insurance A Cop.**
SUTVA is plausible for motor claims (one claim being investigated should not directly affect another). Ignorability is the key assumption to defend in Build 04 — argue that postcode risk, prior claims, and product line are sufficient to satisfy it.

**What to write in the dissertation**
Cite in the identification strategy section. State the SUTVA assumption explicitly, explain why it is reasonable for motor claims, and discuss what might violate it (e.g., organised fraud rings where investigating one claim affects another).

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
where $\hat{e}(X_i) = P(W_i = 1 \mid X_i)$ is the propensity score (estimated investigation probability). Upweights under-investigated claims; downweights over-investigated ones.

**How to apply at Insurance A Cop.**
Build 06 re-weights each motor claim by the inverse of its propensity to be investigated. Claims with low model scores that were accidentally investigated get high weights; high-score claims (routinely investigated) get low weights. This re-balanced dataset then trains a bias-corrected fraud model.

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

**How to apply at Insurance A Cop.**
Estimate propensity scores (probability a claim was investigated given its features) using logistic regression or XGBoost. Match investigated claims to un-investigated "controls" with similar propensity scores. Differences in fraud rate between matched pairs estimate the investigation effect.

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
This is exactly what Build 06 does: adjust for the confounder set $Z$ = {model score, claim features} to estimate the effect of investigation on fraud discovery.

**How to apply at Insurance A Cop.**
Draw the causal DAG for the Insurance A Cop. claims pipeline: model score → investigation decision → fraud discovery → label → retrain → model score (the loop). The SFP loop is the cyclic path. Use the back-door criterion to identify which variables need to be controlled when debiasing.

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
The insurance SFP problem is structurally a sample selection problem: fraud labels are only observed for investigated claims (the selected sample). Heckman's paper shows this creates bias in any model trained on these labels — and provides a correction strategy.

**Summary**
When the sample used for estimation is selected non-randomly (e.g., only investigated claims have fraud labels), OLS estimates are biased. Heckman derived a two-stage correction: first model the selection probability, then include the inverse Mills ratio as a control variable in the outcome equation.

**Key concept / formula**
Inverse Mills ratio: $\lambda(z_i) = \frac{\phi(\hat{z}_i)}{\Phi(\hat{z}_i)}$
Adding $\lambda$ as a regressor corrects for selection bias. The modern IPW approach in Build 06 is a re-parameterisation of the same correction.

**How to apply at Insurance A Cop.**
The Insurance A Cop. fraud model is trained only on investigated claims. Heckman's result implies every fraud rate estimate is upward-biased (investigated claims are pre-selected as likely fraudulent). Quantify this bias in Build 03 (Unbiased Evaluation) and correct it in Build 06.

**What to write in the dissertation**
Cite in the problem framing section: "the partial observability of fraud labels constitutes a sample selection problem in the sense of Heckman (1979), which induces systematic bias in any model trained on these labels."

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

**How to apply at Insurance A Cop.**
Use DiD to estimate the causal effect of a model update (the "treatment") on fraud discovery rates. The treated group = claims scored by the new model; control group = claims still scored by the old model (if a phased rollout happened). Parallel trends is checked by plotting pre-period trends.

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
Applicable when the model switches from "investigate" to "don't investigate" at a score threshold.

**How to apply at Insurance A Cop.**
If Insurance A Cop. uses a fixed model-score threshold to trigger investigation (e.g., score > 0.5 → investigate), RDD estimates the causal investigation effect by comparing claims just above and just below the threshold — they are near-identical except for their investigation status.

**What to write in the dissertation**
Cite alongside Imbens & Wooldridge (2009) as the applied econometrics standard. Use the RDD design explicitly if a score threshold exists in the Insurance A Cop. pipeline; document the bandwidth selection and local linear regression approach as specified in Chapter 6 of Angrist & Pischke.

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

**How to apply at Insurance A Cop.**
Use the taxonomy to classify the motor insurance fraud types in scope (staged accidents, inflated repairs, phantom injuries). Cite the "tip of the iceberg" problem as the empirical motivation for Build 03 (Unbiased Evaluation).

**What to write in the dissertation**
Cite in the literature review to position the work within the fraud detection field. Note that unlike most papers surveyed, this dissertation addresses the *feedback mechanism* rather than the *detection algorithm* alone.

---

### P10 · Barocas & Selbst (2016) — Big Data's Disparate Impact

**Citation**
Barocas, S. & Selbst, A. D. (2016). "Big Data's Disparate Impact." *California Law Review*, 104, 671–732.

**Link** → https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2477899  (DOI: https://doi.org/10.15779/Z38BG31)
**Citations** ≈ 2,500 (Semantic Scholar) · **Journal** *California Law Review* (top-5 US law review)

**Why this paper matters**
Provides the legal and regulatory framing for why the SFP loop is not merely a technical problem but a potential compliance liability — especially relevant to Insurance A Cop. UK under FCA guidelines on fair treatment of customers and the EU AI Act.

**Summary**
Argues that even facially neutral ML models trained on historical data can violate anti-discrimination law by perpetuating past biases. Identifies five pathways from biased training data to discriminatory outcomes: target variable definition, feature selection, proxies for protected characteristics, sample bias, and feedback effects. The last pathway is precisely the SFP loop.

**Key concept / formula**
The disparate impact standard: a selection rate for a protected group that is less than 4/5 (80%) of the rate for the group with the highest rate is considered prima facie discriminatory (US EEOC; analogous to FCA proportionality rules in the UK).

**How to apply at Insurance A Cop.**
Check whether the motor fraud model's investigation rate varies significantly by postcode (proxy for demographics) or product line. If under-investigated segments are correlated with protected characteristics, the SFP loop may have disparate impact implications under FCA PRIN 6 (fair treatment of customers).

**What to write in the dissertation**
Cite in the ethics and regulatory chapter. Frame the SFP loop as simultaneously a technical problem (model bias) and a legal risk (disparate impact). Note Insurance A Cop. UK's obligations under FCA rules as a real-world motivation for the research.

---

## Part 4 — Feedback Loops and Performative Prediction (2016–2020)

---

### P11 · Lum & Isaac (2016) — To Predict and Serve?

**Citation**
Lum, K. & Isaac, W. (2016). "To Predict and Serve?" *Significance*, 13(5), 14–19.

**Link** → https://doi.org/10.1111/j.1740-9713.2016.00960.x
**Citations** ≈ 509 (Semantic Scholar) · **Journal** *Significance* (joint RSS/ASA practitioner magazine, high visibility)

**Why this paper matters**
First empirical demonstration — in a domain analogous to insurance — that a model trained on biased data reinforces the patrol patterns that generated the bias. The closest published analogue to the Insurance A Cop. motor insurance SFP loop.

**Summary**
Applies PredPol (predictive policing software) to Oakland, CA crime data. Shows that because drug arrests reflect where police patrol (not where drugs are actually used), re-training on arrest data sends police back to the same neighbourhoods, creating a self-reinforcing loop. Communities with high historical arrest rates are systematically over-policed.

**Key concept / formula**
Feedback amplification: if investigation probability $\pi_t(x)$ is proportional to model score $f_t(x)$, and $f_{t+1}$ is trained on $\{y_i : \pi_t(x_i) = 1\}$, then in expectation $f_{t+1}(x) \geq f_t(x)$ for high-score regions — the model becomes increasingly confident about already-investigated areas.

**How to apply at Insurance A Cop.**
Replace "drug arrests" with "fraud confirmations" and "patrol area" with "claim segment." Motor claims from certain postcodes or product lines are investigated more; fraud is discovered there; the next model version treats those segments as higher risk — regardless of the true underlying fraud rate.

**What to write in the dissertation**
Cite as the primary motivating analogy. State: "Lum & Isaac (2016) demonstrate an empirically observed SFP loop in predictive policing; this dissertation applies the same detection and mitigation framework to motor insurance fraud detection."

---

### P12 · Ensign et al. (2018) — Runaway Feedback Loops in Predictive Policing

**Citation**
Ensign, D., Friedler, S. A., Neville, S., Scheidegger, C. & Venkatasubramanian, S. (2018). "Runaway Feedback Loops in Predictive Policing." *Proceedings of the 1st ACM FAccT Conference*, PMLR 81:160–171.

**Link** → https://arxiv.org/abs/1706.09847 | https://proceedings.mlr.press/v81/ensign18a.html
**Citations** ≈ 650+ (Semantic Scholar) · **Venue** *FAccT* (A* conference for fairness/accountability in ML)

**Why this paper matters**
Provides the mathematical proof that a prediction-driven investigation system converges to a fixed point that ignores true underlying rates — i.e., the feedback loop is not just a risk but a provable inevitability under standard Pólya urn dynamics. Cited in the CLAUDE.md as "runaway feedback FAT'18."

**Summary**
Models the investigation decision as a Pólya urn process. Proves that without exploration, all probability mass concentrates on the cells where fraud was initially detected, regardless of the true fraud rate elsewhere. Derives the convergence rate as a function of the initial "unfairness" and shows it is not self-correcting.

**Key concept / formula**
Pólya urn dynamic: at each step, the probability of investigating region $r$ is $\propto n_r$ (number of past investigations there). As $n_r \to \infty$, the investigation distribution converges almost surely to a fixed composition determined by initial conditions — not by true fraud rates.

**How to apply at Insurance A Cop.**
This is the theoretical model the SFP simulation (Build 01) implements. Parameterise the urn with Insurance A Cop.'s initial investigation rates; show how the distribution converges. The randomisation strategies in Build 05 are the interventions that break the urn dynamic.

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

**How to apply at Insurance A Cop.**
When designing the randomisation policy (Build 05), consider the long-run equity impact: a policy that aggressively targets high-model-score claims may be short-run accurate but reinforces under-investigation of low-score segments. Use this framework to justify why ε-greedy / Thompson Sampling are preferred over pure exploitation.

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

**How to apply at Insurance A Cop.**
Extend the SFP simulation (Build 01) beyond 3 model versions to show the long-run trajectory of fraud discovery rates for different claim segments under different investigation policies. Use this framework to argue that a randomisation policy evaluated only at period $t=1$ may look worse but converge to a better long-run equilibrium.

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

**How to apply at Insurance A Cop.**
Argue that Insurance A Cop.'s fraud model is performative: its scores determine which claims are investigated, changing what fraud is discovered, changing what data trains the next version. Build 01 simulates this performative dynamic; Build 06 estimates the gap between performative risk and standard training risk.

**What to write in the dissertation**
Cite in the theory section as the formal definition of the dissertation's central concept. Include the performative risk formula in the notation table. State explicitly: "the Insurance A Cop. fraud detection pipeline exhibits performative prediction in the sense of Perdomo et al. (2020)."

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

**How to apply at Insurance A Cop.**
When estimating the effect of investigation on fraud discovery (Build 04), use cross-fitted propensity scores (from LightGBM) in a double ML estimator. This gives valid confidence intervals even though investigation propensity has a high-dimensional feature set (postcode, prior claims, product line, claim type, etc.).

**What to write in the dissertation**
Cite in the methodology section for Build 04. State that the propensity score is estimated with LightGBM using 5-fold cross-fitting following Chernozhukov et al. (2018), and that this ensures the causal effect estimate is $\sqrt{n}$-consistent even with flexible ML nuisance estimators.

---

### P17 · Sculley et al. (2015) — Hidden Technical Debt in ML Systems

**Citation**
Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., Chaudhary, V., Young, M., Crespo, J.-F. & Dennison, D. (2015). "Hidden Technical Debt in Machine Learning Systems." *Advances in Neural Information Processing Systems (NeurIPS) 28*, pp. 2503–2511.

**Link** → https://proceedings.neurips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html
**Citations** ≈ 4,000+ (Google Scholar) · **Venue** *NeurIPS* (A* CORE ranking)

**Why this paper matters**
Identifies feedback loops as a first-class form of technical debt in production ML systems. Provides the systems-engineering framing for why the SFP loop is hard to detect and correct in an operational pipeline like Insurance A Cop.'s.

**Summary**
Categorises ML technical debt as: entanglement (correlated features), hidden feedback loops, undeclared consumers, data dependency debt, and configuration debt. Feedback loops are singled out as particularly dangerous because they can cause slow but compounding degradation that is invisible in standard monitoring metrics. A model's outputs influence the world, which influences future training data.

**Key concept / formula**
The "data dependency debt" formulation: if model output $f(x)$ feeds into any process that generates future training data $D_{t+1}$, then the model has a "hidden feedback loop." Formally: $D_{t+1} = g(D_t, f_t)$ where $g$ is the data-generating process influenced by the model. The SFP loop is exactly this.

**How to apply at Insurance A Cop.**
Use the Sculley et al. taxonomy to audit the Insurance A Cop. motor claims pipeline: identify all places where model outputs influence future data (investigation decisions, adjuster prioritisation, reserve setting). Each is a potential SFP entry point. Document these as part of the Build 00 data exploration.

**What to write in the dissertation**
Cite in the problem framing section alongside Perdomo et al. (2020). Position the dissertation as applying the SFP detection framework to a specific instance of the hidden feedback loop problem identified by Sculley et al.

---

### P18 · D'Amour et al. (2022) — Underspecification in Modern ML

**Citation**
D'Amour, A., et al. (2022). "Underspecification Presents Challenges for Credibility in Modern Machine Learning." *Journal of Machine Learning Research*, 23(226), 1–61.

**Link** → https://arxiv.org/abs/2011.03395 | https://www.jmlr.org/papers/v23/20-1335.html
**Citations** ≈ 900+ (Semantic Scholar) · **Journal** *JMLR* (IF ≈ 6, top open-access ML journal)

**Why this paper matters**
Shows that multiple models with identical in-distribution performance can behave very differently under distribution shift — precisely the problem when a model trained on investigation-biased data is evaluated on the full claims population (Build 03).

**Summary**
A training pipeline is "underspecified" when many models achieve equivalent training performance but diverge under distribution shift. Experiments across NLP, computer vision, medical imaging, and genomics show that standard training and evaluation pipelines do not select for models that generalise reliably. The solution requires stress tests that expose distribution shift.

**Key concept / formula**
Underspecification: $\exists \theta_1 \neq \theta_2$ such that $R_{train}(\theta_1) \approx R_{train}(\theta_2)$ but $R_{test}(\theta_1) \gg R_{test}(\theta_2)$ for some OOD test distribution. For the SFP problem: the in-distribution performance is on investigated claims; the OOD distribution is all claims.

**How to apply at Insurance A Cop.**
Build 03 (Unbiased Evaluation) is directly motivated by this paper: the fraud model's performance on investigated claims (where labels exist) is not representative of its performance on all claims (the full portfolio). IPS-corrected metrics estimate the true OOD performance.

**What to write in the dissertation**
Cite in Build 03. State that standard in-sample AUC is an underspecified metric for the Insurance A Cop. fraud model because the test distribution (investigated claims) is a non-random subset of the deployment distribution (all claims).

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
For claim investigation: arm = claim segment; reward = fraud discovery; prior updated with each investigation outcome.

**How to apply at Insurance A Cop.**
Partition motor claims into segments (by product line, postcode risk, claim type). Run Thompson Sampling across segments: occasionally investigate low-model-score claims to update the Beta posterior for that segment. Over time, this ensures no segment is permanently under-investigated due to a bad initial model.

**What to write in the dissertation**
Cite as the origin of the algorithm. Describe the Beta-Binomial model as the practical instantiation for a binary fraud label. Compare to ε-greedy in Build 05 using regret as the evaluation metric.

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

**How to apply at Insurance A Cop.**
UCB1 applied to claim investigation: each claim segment is an arm; $\bar{x}_k$ is the empirical fraud rate for that segment; $n_k$ is the number of past investigations. Claims from under-investigated segments get a UCB bonus, encouraging exploration even when the model scores them as low risk.

**What to write in the dissertation**
Cite when introducing the UCB baseline in Build 05. State the regret bound explicitly and contrast with Thompson Sampling's empirical performance. Note that UCB is frequentist (no prior needed) while Thompson Sampling is Bayesian — both are evaluated on the synthetic dataset.

---

### P21 · Russo et al. (2018) — Tutorial on Thompson Sampling

**Citation**
Russo, D. J., Van Roy, B., Kazerouni, A., Osband, I. & Wen, Z. (2018). "A Tutorial on Thompson Sampling." *Foundations and Trends in Machine Learning*, 11(1), 1–96.

**Link** → https://doi.org/10.1561/2200000070 | arXiv: https://arxiv.org/abs/1707.02038
**Citations** ≈ 3,000+ (Google Scholar) · **Journal** *Foundations and Trends in ML* (NOW Publishers, high-citation review journal)

**Why this paper matters**
The standard practical reference for implementing Thompson Sampling. Covers Bernoulli bandits (directly applicable to binary fraud labels), contextual extensions, and convergence guarantees.

**Summary**
Provides a thorough tutorial from the Beta-Bernoulli case through Gaussian, contextual, and combinatorial bandits. Derives regret bounds, discusses computational approximations (Langevin, variational), and surveys empirical results across online advertising, medical trials, and recommendation systems.

**Key concept / formula**
Bayesian regret bound for Thompson Sampling:
$\text{BayesRegret}(T) \leq \sqrt{\frac{T K \ln K}{2}}$
where $K$ is the number of arms. This improves on UCB1's $O(\sqrt{KT \ln T})$ bound in many practical settings.

**How to apply at Insurance A Cop.**
The contextual bandit extension (Section 5 of the tutorial) is directly applicable: use claim features (amount, product line, prior claims) as context to form a personalised exploration policy rather than a single Beta distribution per segment.

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
Introduces the equalised-odds and equal opportunity fairness criteria. In the insurance context: the model should not have systematically higher false-negative rates (missed fraud) for particular claim segments because those segments were historically under-investigated.

**Summary**
Proposes that a fair classifier should have equal true positive rates and equal false positive rates across demographic groups — "equalised odds." A weaker condition, "equal opportunity," requires only equal true positive rates. Provides a post-processing algorithm to achieve either criterion by adjusting decision thresholds per group.

**Key concept / formula**
Equalised odds: $\hat{Y} \perp A \mid Y$ — prediction is independent of the protected attribute $A$ given the true label $Y$.
Equal opportunity: $P(\hat{Y}=1 \mid A=0, Y=1) = P(\hat{Y}=1 \mid A=1, Y=1)$ — equal TPR across groups.

**How to apply at Insurance A Cop.**
Check whether the SFP-corrected model achieves equal opportunity across product lines or postcode risk groups. A model that misses fraud disproportionately in certain segments (because those segments were historically under-investigated) fails this criterion. Build 06's IPW debiasing should improve equal opportunity.

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
The impossibility theorem (Chouldechova 2017, formalised here): calibration + equal FPR + equal FNR cannot all hold simultaneously when base rates differ across groups. This implies any fairness criterion for the Insurance A Cop. model involves trade-offs that must be explicitly documented.

**How to apply at Insurance A Cop.**
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
Critical review showing that calibration, anti-classification, and classification parity all conflict with each other and with social welfare maximisation. Particularly relevant when justifying the specific fairness metric chosen for the Insurance A Cop. model evaluation.

**Summary**
Argues that commonly used fairness metrics have unintended consequences when applied naively. Proposes "conditional use accuracy equality" as a more principled criterion. Shows through the bail and lending examples that equality constraints on error rates can lead to worse outcomes for the groups being "protected."

**Key concept / formula**
Calibration: $P(Y=1 \mid \hat{p}(X) = p) = p$ — model probabilities match true probabilities. Anti-classification: the model does not use protected attributes. Classification parity: equal error rates. Theorem: all three cannot hold simultaneously when $P(Y=1 \mid A=0) \neq P(Y=1 \mid A=1)$.

**How to apply at Insurance A Cop.**
Motor fraud base rates vary by product line and postcode. Document these base rate differences and show that perfect calibration implies different error rates across groups — this is expected and not a failure of the model. Use this to defend against naive criticism of disparate error rates. **Important caveat**: the production XGBoost model is not calibrated (see README — Model Training Methodology). Raw scores are used for ranking/triage only, not as probability estimates. This means calibration cannot be directly assessed from model outputs unless Platt scaling or isotonic regression is applied post-hoc. This also affects IPS/IPW debiasing in Build 06: propensity weights derived from uncalibrated scores introduce additional bias into the reweighting — a limitation to acknowledge explicitly.

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
Classifies AI systems used in insurance risk assessment and claims handling as **high-risk** (Annex III). Requires technical documentation, bias testing, human oversight, and post-deployment monitoring. Provides the regulatory mandate for solving the SFP loop: it is not merely academic but a compliance obligation for Insurance A Cop. UK.

**Summary**
The AI Act creates a risk-based framework: prohibited AI (social scoring, biometric surveillance), high-risk AI (credit, insurance, employment, law enforcement), and limited/minimal risk. For Insurance A Cop. UK, the total loss scoring system is high-risk under Annex III point 5(b) (AI in insurance pricing and risk assessment) and point 6 (AI in law enforcement-adjacent tasks). Requirements include: risk management systems, data governance, transparency documentation, human oversight, accuracy and robustness requirements, and post-market monitoring.

**Key concept / formula**
Article 9 (Risk management system): continuous risk management cycle required throughout the lifecycle. Article 10 (Data governance): training data must be representative, free from errors, and complete. The SFP loop directly violates Article 10 — biased labels from the scrapping decision are not representative.

**How to apply at Insurance A Cop.**
The dissertation's SFP detection framework (Build 02) and mitigation methods (Builds 05–06) can be positioned as the technical documentation and bias-testing component required by Articles 9–10 of the AI Act. Build 03 (Unbiased Evaluation) maps to Article 15 (accuracy requirements).

**What to write in the dissertation**
Cite in the ethics and regulatory chapter. State: "the total loss scoring system at Insurance A Cop. UK falls within the high-risk category under Annex III of the EU AI Act (Regulation 2024/1689), which mandates bias testing and post-deployment monitoring. This dissertation provides a technical framework for satisfying those requirements."

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
The goal of any bandit algorithm is to minimise expected cumulative regret $E[R_T]$ by balancing exploration (learning about under-investigated arms) and exploitation (investigating high-fraud-probability claims).

**How to apply at Insurance A Cop.**
Use Chapter 36 to implement a contextual bandit that takes claim features as context and outputs an investigation probability. This is strictly more powerful than the segment-level Thompson Sampling in Build 05 and can be presented as a future extension.

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
The Insurance A Cop. total loss model is a textbook selective-labels system: the model decides whether to scrap (`decision=1`) or send to garage (`decision=0`). Repair outcomes (`garage_outcome`) are observable only for `decision=0` rows. For scrapped cars the true repair outcome is structurally absent — not missing at random. This paper provides the exact formal framework for evaluating and learning from such systems, and is the direct citation for this class of problems in the fairness and causal ML literature.

**Summary**
Studies the problem of evaluating ML models when outcomes are only observed for a subset of cases determined by the model's own decisions (or a human predecessor). Formalises **selective labels bias**: the observed accuracy on the selected subset systematically overestimates true accuracy on the full population. Proposes evaluation strategies that account for structural missingness of outcomes in the unselected group, and derives conditions under which counterfactual performance can be bounded or estimated from observational data.

**Key concept / formula**
Let $\hat{Y}_i$ be the model prediction and $D_i \in \{0,1\}$ the binary decision (0 = garage, 1 = scrap). Outcome $Y_i$ is observable only when $D_i = 0$:
$$Y_i \text{ observed} \iff D_i = 0$$
Standard accuracy evaluated on $\{i : D_i = 0\}$ is biased because $D_i$ is a function of $\hat{Y}_i$. The paper derives conditions under which counterfactual performance on $\{i : D_i = 1\}$ can be bounded or estimated.

**How to apply at Insurance A Cop.**
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
The label structure of the Insurance A Cop. total loss dataset maps onto the PU learning setting. Cars sent to garage with confirmed total losses are **labeled positives** (`Y=1`, observed). Cars sent to garage that were successfully repaired are **labeled negatives** (`Y=0`). Scrapped cars have a forced label of 1 but their true repair outcome is unknown — they may have been repairable false positives. PU learning provides theory for training and evaluating classifiers under exactly this asymmetric observability structure.

**Summary**
Comprehensive survey of learning algorithms when training data consists of labeled positives and unlabeled examples (containing both true positives and true negatives). Covers: (a) the two main assumptions — single-training-set (SCAR) vs. selected-completely-at-random; (b) methods for estimating the class prior $\pi = P(Y=1)$ from unlabeled data; (c) algorithms including biased SVM, EM-based methods, two-step methods (spy technique), and cost-sensitive re-weighting; (d) evaluation criteria under PU assumptions.

**Key concept / formula**
PU risk decomposition: given labeled positives $\mathcal{P}$ and unlabeled $\mathcal{U}$ (true positive rate $\pi$), the risk of classifier $g$ is:
$$R(g) = \pi \cdot R^+(g) + (1-\pi) \cdot R^-(g)$$
where $R^+(g)$ and $R^-(g)$ are false-negative and false-positive risks. Estimating $\pi$ — the proportion of true total losses among scrapped cars — is the core estimation problem in the Insurance A Cop. context.

**How to apply at Insurance A Cop.**
Treat `model_v1_decision=1` rows (scrapped) as the "unlabeled" group: their observed label is 1, but the true fraction of genuine total losses $\pi$ is unknown. The `decision=0` rows with `outcome=1` are the labeled positives. Use PU learning methods to estimate $\hat{\pi}$, which quantifies the false-positive rate of the scrapping policy. This estimate directly informs the magnitude of SFP bias quantified in Builds 01 and 03.

**What to write in the dissertation**
Cite in Build 03 (Unbiased Evaluation) and Build 06 (Causal Mitigation). Frame the label contamination in `model_v1_observed_outcome` as a PU learning problem: "following Bekker & Davis (2020), we treat scrapped-car rows as unlabeled under the PU assumption, since their observed label of 1 reflects the scrapping decision rather than a verified repair outcome. We estimate the class prior $\hat{\pi}$ to quantify the magnitude of label contamination introduced by the SFP mechanism."

---

## Part 9 — Mathematical Evaluation of SFP / Feedback Loops

*Added 2026-06-23. These papers provide formal tools for **quantifying** how strong an SFP loop is — as opposed to proving it exists (P12) or defining its framework (P15). They are the gap-filling literature for Build 02's detection logic.*

---

### P29 · Mendler-Dünner et al. (2020) — Stochastic Optimization for Performative Prediction

**Citation**
Mendler-Dünner, C., Perdomo, J. C., Zrnic, T. & Hardt, M. (2020). "Stochastic Optimization for Performative Prediction." *Advances in Neural Information Processing Systems (NeurIPS) 33*.

**Link** → https://proceedings.neurips.cc/paper/2020/hash/33e75ff09dd601bbe69f351039152189-Abstract.html | arXiv: https://arxiv.org/abs/2002.09058
**Citations** ≈ 250+ (Semantic Scholar) · **Venue** *NeurIPS* (A* CORE ranking)

**Why this paper matters**
P15 (Perdomo et al.) defines performative risk and proves the gap between standard ERM and the performative optimum. This companion paper asks the next question: *how quickly does repeated retraining converge to the biased performative-stable point?* The convergence rate is a direct mathematical measure of **how fast** the SFP loop locks in.

**Summary**
Distinguishes two natural deployment strategies: (a) **greedy deploy** — deploy immediately after each stochastic gradient step; (b) **lazy deploy** — accumulate gradients on multiple samples before redeploying. Derives necessary and sufficient conditions for convergence to a performatively stable (PS) point under each strategy. Shows that sensitivity (how much the data distribution shifts per unit change in model parameters) and strong convexity jointly determine whether the loop stabilises or diverges. Generalises Perdomo et al. to the non-i.i.d., stochastic gradient regime.

**Key concept / formula**
Let ε be the sensitivity (ε = max‖θ₁−θ₂‖→0 W₂(D(θ₁), D(θ₂))/‖θ₁−θ₂‖) and β the strong convexity constant of the loss. Convergence condition: **ε/β < 1** — the distribution shift per parameter change must be smaller than the loss curvature. When ε/β ≥ 1 the loop diverges (the SFP amplification outpaces the model's self-correcting tendency).

**How to apply at Insurance A Cop.**
The ratio ε/β is the quantitative SFP loop coefficient for the total loss pipeline. Estimate ε empirically by measuring how much the scrapping-decision distribution shifts between v1 and v2a (Wasserstein distance on propensity scores). Estimate β from the Hessian of the v2a loss. If ε/β is close to or exceeds 1, the simulation (Build 01) and real-data evaluation (Build 03) will show runaway drift; if ε/β < 1, convergence to a biased-but-stable fixed point is expected.

**What to write in the dissertation**
Cite alongside P15 in the theory section. State: "Mendler-Dünner et al. (2020) prove that convergence to a performatively stable point requires ε/β < 1, where ε is the distribution sensitivity and β is the loss curvature. We estimate this ratio empirically in Build 02 as a single-number loop severity score."

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

**How to apply at Insurance A Cop.**
Step 1 of SFPDetector (temporal prediction correlation) can be re-grounded in this taxonomy: rising cross-version Spearman rank correlation is the empirical signature of a positive-gain amplifying loop. Step 4 (segment blind spots) corresponds to the paper's "one-sided selection" loop subtype. Cite this to give the 4-step detector a unified theoretical basis rather than presenting each step as an ad-hoc heuristic.

**What to write in the dissertation**
Cite in the Build 02 methodology section when introducing the four-step SFP detection algorithm. State: "following Pagan et al. (2023), we classify the total loss SFP mechanism as an amplifying positive-gain feedback loop; each detection step targets the observable signature of this loop type."

---

### P31 · Veprikov, Afanasiev & Khritankov (2025) — A Mathematical Model of the Hidden Feedback Loop Effect in Machine Learning Systems

**Citation**
Veprikov, A., Afanasiev, A. & Khritankov, A. (2025). "A Mathematical Model of the Hidden Feedback Loop Effect in Machine Learning Systems." *Knowledge and Information Systems* (Springer). (arXiv preprint: arXiv:2405.02726, May 2024.)

**Link** → https://arxiv.org/abs/2405.02726 | https://link.springer.com/article/10.1007/s10115-025-02560-w
**Citations** ≈ 5 (early; published 2025) · **Journal** *Knowledge and Information Systems* (Springer, IF ≈ 2.5)

**Why this paper matters**
The only paper identified (as of 2026) that provides a single mathematical model unifying **error amplification**, **induced concept drift**, and **echo chambers** as special cases of the same repeated-learning feedback loop. This is the most directly applicable paper for formalising what the SFP simulation (Build 01) implements and what the detector (Build 02) is searching for.

**Summary**
Formalises the entire "data collection → training → deployment → environment influence → data collection" cycle as a single dynamical system. The key insight: the state of the environment at time t+1 is a deterministic function of the environment at time t *and* the predictions made at t. This causally couples the learner to the data-generating process, violating i.i.d. assumptions and producing a zoo of observable phenomena (error amplification, concept drift, echo chambers) depending on the feedback gain coefficient. Provides a theorem on the limiting set of distributions the system can converge to and sufficient conditions for the loop to be "hidden" (undetectable by standard train/test splitting).

**Key concept / formula**
Repeated learning map: E_{t+1} = F(E_t, M_t) where E_t is the environment distribution and M_t is the deployed model. A feedback loop is "hidden" when the standard empirical risk on the held-out split is not a monotone function of the true performative risk — i.e., the model appears to improve on the test set while the loop worsens. This directly explains why v2a's OOT AUC at Insurance A Cop. looked acceptable while the scrap rate inflated.

**How to apply at Insurance A Cop.**
Build 01 (SFP Simulation) is implementing the map E_{t+1} = F(E_t, M_t) with `repair_decision` as the coupling mechanism. Build 02 Step 1 (temporal score correlation) detects the signature of F being non-trivial. Build 03 (Unbiased Evaluation) is needed precisely because the OOT AUC is an example of the "hidden loop" condition — it masks performative risk growth. Cite this paper as the mathematical unification of all four detection steps.

**What to write in the dissertation**
Cite in the theory chapter as the unified mathematical model. State: "Veprikov et al. (2025) formalise the repeated learning process as E_{t+1} = F(E_t, M_t) and prove conditions under which feedback effects are 'hidden' from standard evaluation metrics. Our Build 03 unbiased evaluation addresses exactly this hiding condition: the OOT AUC on v1-logged data is a biased proxy for performative risk, consistent with their Theorem 3."

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

**How to apply at Insurance A Cop.**
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
P15 (Perdomo et al.) defines performative risk under the map D(θ) — the data distribution depends only on the **current** model θ. This is an incomplete model of the total loss SFP loop: in practice the distribution at retraining time depends not only on v2's model parameters but also on the **accumulated state** of forced-positive labels generated by all prior versions (v1 → v2). This paper extends performative prediction to D(θ, s_t), where s_t is the state of the population at time t. The state evolves across model generations, and convergence conditions now depend on **both** distribution sensitivity and state-transition dynamics. This directly explains why the Insurance A Cop. v3 retraining failed: the state (v1+v2 label contamination accumulated over the training window) was already entrenched, and retraining on contaminated labels could not escape the biased equilibrium regardless of v3's architecture.

**Summary**
Proposes a framework where the response of the target population to the deployed classifier is a function of both the classifier θ and the current state s_t (the distribution of the population itself). The state evolves according to a transition function g: s_{t+1} = g(s_t, θ_t). Two retraining algorithms are analysed: (1) **repeated risk minimisation** — retrain on the current state's data distribution; (2) **lazy variant** — retrain less frequently, allowing the state to settle. Derives necessary and sufficient conditions for convergence to a stable equilibrium near the performatively optimal classifier. Captures the phenomenon that distinct groups accumulate information and resources at different rates in response to the deployed classifier — translating to vehicle segments accumulating forced-positive labels at different rates under the scrapping policy.

**Key concept / formula**
Stateful performative map: D(θ, s_t), with state transition s_{t+1} = g(s_t, θ_t).
Convergence to equilibrium (θ*, s*) requires: sensitivity ε_θ (distribution shift per parameter change) and sensitivity ε_s (state shift per state change) jointly satisfy a contraction condition. When ε_s is large — i.e. the state itself is highly reactive to past model decisions — standard repeated retraining cannot escape the biased fixed point even if the model's per-step update is small.

**How to apply at Insurance A Cop.**
The state s_t is the accumulated label-contamination profile across vehicle segments: how many forced-positive labels have been added per segment across all prior model versions. After v1 and v2a both ran with the absolute 0.872 threshold, s is heavily contaminated in high-RTV / high-damage segments. Build 01 should simulate the stateful dynamics explicitly — not just the one-step v1→v2 transition, but the multi-step v1→v2→v3 trajectory — to show that even a well-specified v3 cannot escape the biased equilibrium once the state accumulation has reached a threshold. The state-dependent framework also explains why v2b (counterfactual with pre-ML data) partially resists SFP: including pre-ML labels in training effectively resets part of the contaminated state.

**What to write in the dissertation**
Cite alongside P15 in the theory section. State: "Perdomo et al. (2020) characterise the loop in terms of the current model alone. Brown et al. (2022) generalise this to a stateful setting where s_{t+1} = g(s_t, θ_t): the distribution depends on both the model and the accumulated history of prior scrapping decisions. This is the correct model for the Insurance A Cop. pipeline, where v1's forced-positive labels became part of the training state for v2a, and v2a's labels in turn become the state for v3. The failure of v3 retraining is consistent with Brown et al.'s result that a large state sensitivity ε_s can prevent convergence to the performatively optimal classifier."

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
| P29 | Stochastic Opt. for Performative Pred. | ✓ | ✓ | | Convergence condition ε/β < 1 requires estimating Wasserstein sensitivity ε between model versions — needs two deployed versions to compute empirically |
| P30 | Classification of Feedback Loops | ✓ | ✓ | | Taxonomy and gain-sign classification; does not provide a correction or debiasing algorithm |
| P31 | Mathematical Model of Hidden Feedback Loop | ✓ | ✓ | | Provides the theoretical unification; no off-the-shelf implementation — must be instantiated for the total loss domain |
| P32 | Degenerate Feedback Loops in Recommender Systems | ✓ | ✓ | | Recommender-system setting (user preference drift); forced-label structure unique to total loss must be analogised, not directly applied |
| P33 | Performative Prediction in a Stateful World | ✓ | ✓ | | Convergence conditions require estimating state sensitivity ε_s from multi-generation logs (needs v1→v2→v3 data); no debiasing or correction method |

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
| P29 | Mendler-Dünner et al. | 2020 | Stochastic Opt. for Performative Pred. | Define + Detect | NeurIPS | arxiv.org/abs/2002.09058 | 250 |
| P30 | Pagan et al. | 2023 | Classification of Feedback Loops | Define + Detect | arXiv | arxiv.org/abs/2305.06055 | 30 |
| P31 | Veprikov et al. | 2025 | Mathematical Model of Hidden Feedback Loop | Define + Detect | KAIS (Springer) | arxiv.org/abs/2405.02726 | 5 |
| P32 | Jiang et al. | 2019 | Degenerate Feedback Loops in Recommender Systems | Define + Detect | AIES | dl.acm.org/doi/10.1145/3306618.3314288 | 200 |
| P33 | Brown et al. | 2022 | Performative Prediction in a Stateful World | Define + Detect | AISTATS | arxiv.org/abs/2011.03885 | 90 |

---

*All citation counts are approximate Google Scholar / Semantic Scholar figures as of mid-2026. Regulatory documents (P25) do not have citation counts. P27 and P28 added 2026-06-15 following domain confirmation (total loss prediction). P29, P30, P31 added 2026-06-23 to fill the mathematical SFP evaluation gap — providing convergence rates (P29), loop-type classification (P30), and a unified repeated-learning model (P31) that the existing library lacked.*

---

## PDF Download Status

18 of 28 papers downloaded automatically to `literatures/p{N}.pdf`.
The remaining 10 are behind institutional paywalls or are paid books — download via your **University of Bristol library** login or **Insurance A Cop. VPN**.

| # | File | Status | Note |
|---|------|--------|------|
| P1 | `p1.pdf` ✓ | Downloaded | Statistical Science (Project Euclid open archive) |
| P2 | — | **UoB library needed** | APA paywall · DOI: 10.1037/h0037350 |
| P3 | — | **UoB library needed** | JASA 1952 · DOI: 10.1080/01621459.1952.10483446 |
| P4 | — | **UoB library needed** | Biometrika/Oxford paywall · DOI: 10.1093/biomet/70.1.41 |
| P5 | `p5.pdf` ✓ | Downloaded | Pearl's UCLA preprint server |
| P6 | — | **UoB library needed** | Econometrica paywall · DOI: 10.2307/1912352 |
| P7 | `p7.pdf` ✓ | Downloaded | NBER working paper version (WP 14251) |
| P8 | — | **Purchase / library** | Princeton UP book · ISBN: 9780691120355 |
| P9 | `p9.pdf` ✓ | Downloaded | arXiv:1009.6119 |
| P10 | — | **UoB library needed** | California Law Review · DOI: 10.15779/Z38BG31 |
| P11 | — | **UoB library needed** | Significance/Wiley paywall · DOI: 10.1111/j.1740-9713.2016.00960.x |
| P12 | `p12.pdf` ✓ | Downloaded | arXiv:1706.09847 |
| P13 | `p13.pdf` ✓ | Downloaded | arXiv:1701.08230 |
| P14 | `p14.pdf` ✓ | Downloaded | arXiv:1803.04383 |
| P15 | `p15.pdf` ✓ | Downloaded | arXiv:2002.06673 |
| P16 | `p16.pdf` ✓ | Downloaded | arXiv:1608.00060 |
| P17 | `p17.pdf` ✓ | Downloaded | NeurIPS 2015 proceedings |
| P18 | `p18.pdf` ✓ | Downloaded | arXiv:2011.03395 |
| P19 | — | **UoB library needed** | Biometrika 1933 · DOI: 10.1093/biomet/25.3-4.285 |
| P20 | `p20.pdf` ✓ | Downloaded | Author's (Cesa-Bianchi) personal page |
| P21 | `p21.pdf` ✓ | Downloaded | arXiv:1707.02038 |
| P22 | `p22.pdf` ✓ | Downloaded | arXiv:1610.02413 |
| P23 | `p23.pdf` ✓ | Downloaded | fairmlbook.org (open access book) |
| P24 | `p24.pdf` ✓ | Downloaded | arXiv:1808.00023 |
| P25 | `p25.pdf` ✓ | Downloaded | eur-lex.europa.eu (full 144-page regulation) |
| P26 | `p26.pdf` ✓ | Downloaded | tor-lattimore.com (open access book, 597pp) |
| P27 | — | **ACM DL / UoB library** | KDD 2017 · ACM DL: dl.acm.org/doi/10.1145/3097983.3098066 (no arXiv preprint) |
| P28 | — | **arXiv free / UoB library** | arXiv:1811.04820 (free preprint); Springer DOI: 10.1007/s10994-020-05877-5 |
| P29 | — | **arXiv free** | arXiv:2002.09058 · NeurIPS 2020 proceedings also open access |
| P30 | — | **arXiv free** | arXiv:2305.06055 |
| P31 | — | **arXiv free / UoB library** | arXiv:2405.02726 (free preprint); Springer KAIS DOI: 10.1007/s10115-025-02560-w |
| P32 | — | **arXiv free / ACM DL** | arXiv:1902.10730 (free preprint); ACM DL: dl.acm.org/doi/10.1145/3306618.3314288 |
| P33 | — | **arXiv free** | arXiv:2011.03885 · AISTATS 2022 proceedings also open access (PMLR 151) |

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
(absolute scrap cutoff `score ≥ 0.872`, two-generation training v1 → v2a/v2b). The
companion section "Application Implementation" below maps this onto the `src/` code.*

## 0. The one structural fact everything else follows from

The total loss pipeline is a **selective-labels system with irreversible, over-labelling
actions**:

```
score ≥ 0.872  → scrap     → observed_outcome forced to 1   (car gone; garage NEVER verifies → oracle permanently absent)
score <  0.872 → garage    → observed_outcome = true result  (reliable 0/1)
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
| **03 Unbiased eval** | Selective-labels-corrected AUC on the **garage-only** subset, reweighted to the full population; PU class-prior `π̂` for scrapped cars | P27, P3 (IPS), P6, P28 (PU), P18 | Evaluate on `decision==0` rows (true labels) and reweight by inverse P(garage); **also correct the OOT set** (it is inside the v1 log → contaminated) |
| **04 Intervention** | Causal effect of *scrapping* on the forced-positive label; RDD at the **0.872 cutoff**; PSM/DML on propensity-to-scrap | P7, P8 (RDD/DiD), P4, P16, P5 (DAG) | The 0.872 absolute cutoff is a **textbook sharp RDD** — near-identical cars just above/below it. This is the cleanest natural experiment in the whole project |
| **05 Randomisation** | Policy that sends a budgeted fraction of high-score cars to the garage to recover oracle labels; regret vs cost | P19/P21 (Thompson), P20/P26 (UCB), P13 (cost of fairness), P14 | Exploration is **expensive and risky** (garage fee + possibly paying a true total loss's full value) — cost-benefit must be modelled explicitly, unlike cheap re-investigation |
| **06 Mitigation** | Debias next-gen training data: downweight/relabel forced positives; IPW for garage rows; PU-imputed counterfactuals for scrapped rows | P3, P4, P16, P5, P27, P28 | Don't just reweight — the forced `outcome=1` on scrapped rows is *wrong*, so PU relabelling/`π̂`-correction matters as much as IPW |
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

## 3. Why the absolute 0.872 cutoff matters to the logic

Choosing an **absolute** cutoff (not a percentile) is what makes Builds 01–04 work:

- **Build 01/02**: score drift is *visible* as scrap-rate inflation (a percentile rule would
  pin the rate and hide it).
- **Build 04 RDD**: a fixed score cutoff is a sharp discontinuity in treatment assignment —
  Imbens & Wooldridge (**P7**) / Angrist & Pischke (**P8**) RDD applies almost verbatim, with
  bandwidth around 0.872.
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
(synthetic) and Insurance A Cop. (real) runs is the injected class — never the core logic. Three swap
axes (`src/DESIGN.md`):

| Axis | Interface | Synthetic now | Real later |
|---|---|---|---|
| Data loading | `DataLoader` | `SyntheticDataLoader` (reads `src/data/synthetic/parquet`) | `RealDataLoader` (Insurance A Cop. Parquet + DB tables) |
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

`ESTIMATE` (Build 04: RDD at 0.872, DiD, DML) lives as analysis modules the detector can call
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
  probability read of the 0.872 cutoff (**P24**).

## 4. Synthetic → real cutover checklist

1. Implement `RealDataLoader` to yield the same schema (`synth_data_structure.md`) — map
   Insurance A Cop. columns to `model_v{1,2}_*`, `repair_decision`, `*_observed_outcome`.
2. Confirm the real scrap policy is the absolute 0.872 cutoff (or parameterise
   `SCRAP_THRESHOLD`); the RDD bandwidth in Build 04 keys off it.
3. Re-fit propensity/calibration on real data; re-check overlap and `π̂` plausibility.
4. Everything downstream (detector, mitigator, report) runs unchanged — that is the whole
   point of the Strategy split.
