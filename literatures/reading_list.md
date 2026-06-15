# Reading List — Identifying and Mitigating Self-Fulfilling Prophecy Loops in ML
**MSc Data Science Dissertation · University of Bristol · Allianz UK (Operations team)**
All claims in scope: **motor insurance claims (UK personal lines auto)**

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

**How to apply at Allianz**
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

**How to apply at Allianz**
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

**How to apply at Allianz**
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

**How to apply at Allianz**
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

**How to apply at Allianz**
Draw the causal DAG for the Allianz claims pipeline: model score → investigation decision → fraud discovery → label → retrain → model score (the loop). The SFP loop is the cyclic path. Use the back-door criterion to identify which variables need to be controlled when debiasing.

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

**How to apply at Allianz**
The Allianz fraud model is trained only on investigated claims. Heckman's result implies every fraud rate estimate is upward-biased (investigated claims are pre-selected as likely fraudulent). Quantify this bias in Build 03 (Unbiased Evaluation) and correct it in Build 06.

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

**How to apply at Allianz**
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

**How to apply at Allianz**
If Allianz uses a fixed model-score threshold to trigger investigation (e.g., score > 0.5 → investigate), RDD estimates the causal investigation effect by comparing claims just above and just below the threshold — they are near-identical except for their investigation status.

**What to write in the dissertation**
Cite alongside Imbens & Wooldridge (2009) as the applied econometrics standard. Use the RDD design explicitly if a score threshold exists in the Allianz pipeline; document the bandwidth selection and local linear regression approach as specified in Chapter 6 of Angrist & Pischke.

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

**How to apply at Allianz**
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
Provides the legal and regulatory framing for why the SFP loop is not merely a technical problem but a potential compliance liability — especially relevant to Allianz UK under FCA guidelines on fair treatment of customers and the EU AI Act.

**Summary**
Argues that even facially neutral ML models trained on historical data can violate anti-discrimination law by perpetuating past biases. Identifies five pathways from biased training data to discriminatory outcomes: target variable definition, feature selection, proxies for protected characteristics, sample bias, and feedback effects. The last pathway is precisely the SFP loop.

**Key concept / formula**
The disparate impact standard: a selection rate for a protected group that is less than 4/5 (80%) of the rate for the group with the highest rate is considered prima facie discriminatory (US EEOC; analogous to FCA proportionality rules in the UK).

**How to apply at Allianz**
Check whether the motor fraud model's investigation rate varies significantly by postcode (proxy for demographics) or product line. If under-investigated segments are correlated with protected characteristics, the SFP loop may have disparate impact implications under FCA PRIN 6 (fair treatment of customers).

**What to write in the dissertation**
Cite in the ethics and regulatory chapter. Frame the SFP loop as simultaneously a technical problem (model bias) and a legal risk (disparate impact). Note Allianz UK's obligations under FCA rules as a real-world motivation for the research.

---

## Part 4 — Feedback Loops and Performative Prediction (2016–2020)

---

### P11 · Lum & Isaac (2016) — To Predict and Serve?

**Citation**
Lum, K. & Isaac, W. (2016). "To Predict and Serve?" *Significance*, 13(5), 14–19.

**Link** → https://doi.org/10.1111/j.1740-9713.2016.00960.x
**Citations** ≈ 509 (Semantic Scholar) · **Journal** *Significance* (joint RSS/ASA practitioner magazine, high visibility)

**Why this paper matters**
First empirical demonstration — in a domain analogous to insurance — that a model trained on biased data reinforces the patrol patterns that generated the bias. The closest published analogue to the Allianz motor insurance SFP loop.

**Summary**
Applies PredPol (predictive policing software) to Oakland, CA crime data. Shows that because drug arrests reflect where police patrol (not where drugs are actually used), re-training on arrest data sends police back to the same neighbourhoods, creating a self-reinforcing loop. Communities with high historical arrest rates are systematically over-policed.

**Key concept / formula**
Feedback amplification: if investigation probability $\pi_t(x)$ is proportional to model score $f_t(x)$, and $f_{t+1}$ is trained on $\{y_i : \pi_t(x_i) = 1\}$, then in expectation $f_{t+1}(x) \geq f_t(x)$ for high-score regions — the model becomes increasingly confident about already-investigated areas.

**How to apply at Allianz**
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

**How to apply at Allianz**
This is the theoretical model the SFP simulation (Build 01) implements. Parameterise the urn with Allianz's initial investigation rates; show how the distribution converges. The randomisation strategies in Build 05 are the interventions that break the urn dynamic.

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

**How to apply at Allianz**
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

**How to apply at Allianz**
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

**How to apply at Allianz**
Argue that Allianz's fraud model is performative: its scores determine which claims are investigated, changing what fraud is discovered, changing what data trains the next version. Build 01 simulates this performative dynamic; Build 06 estimates the gap between performative risk and standard training risk.

**What to write in the dissertation**
Cite in the theory section as the formal definition of the dissertation's central concept. Include the performative risk formula in the notation table. State explicitly: "the Allianz fraud detection pipeline exhibits performative prediction in the sense of Perdomo et al. (2020)."

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

**How to apply at Allianz**
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
Identifies feedback loops as a first-class form of technical debt in production ML systems. Provides the systems-engineering framing for why the SFP loop is hard to detect and correct in an operational pipeline like Allianz's.

**Summary**
Categorises ML technical debt as: entanglement (correlated features), hidden feedback loops, undeclared consumers, data dependency debt, and configuration debt. Feedback loops are singled out as particularly dangerous because they can cause slow but compounding degradation that is invisible in standard monitoring metrics. A model's outputs influence the world, which influences future training data.

**Key concept / formula**
The "data dependency debt" formulation: if model output $f(x)$ feeds into any process that generates future training data $D_{t+1}$, then the model has a "hidden feedback loop." Formally: $D_{t+1} = g(D_t, f_t)$ where $g$ is the data-generating process influenced by the model. The SFP loop is exactly this.

**How to apply at Allianz**
Use the Sculley et al. taxonomy to audit the Allianz motor claims pipeline: identify all places where model outputs influence future data (investigation decisions, adjuster prioritisation, reserve setting). Each is a potential SFP entry point. Document these as part of the Build 00 data exploration.

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

**How to apply at Allianz**
Build 03 (Unbiased Evaluation) is directly motivated by this paper: the fraud model's performance on investigated claims (where labels exist) is not representative of its performance on all claims (the full portfolio). IPS-corrected metrics estimate the true OOD performance.

**What to write in the dissertation**
Cite in Build 03. State that standard in-sample AUC is an underspecified metric for the Allianz fraud model because the test distribution (investigated claims) is a non-random subset of the deployment distribution (all claims).

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

**How to apply at Allianz**
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

**How to apply at Allianz**
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

**How to apply at Allianz**
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

**How to apply at Allianz**
Check whether the SFP-corrected model achieves equal opportunity across product lines or postcode risk groups. A model that misses fraud disproportionately in certain segments (because those segments were historically under-investigated) fails this criterion. Build 06's IPW debiasing should improve equal opportunity.

**What to write in the dissertation**
Cite in the evaluation section of Build 06. Use equal opportunity as one of the post-mitigation fairness metrics alongside standard AUC and calibration. Show before/after comparison.

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
The impossibility theorem (Chouldechova 2017, formalised here): calibration + equal FPR + equal FNR cannot all hold simultaneously when base rates differ across groups. This implies any fairness criterion for the Allianz model involves trade-offs that must be explicitly documented.

**How to apply at Allianz**
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
Critical review showing that calibration, anti-classification, and classification parity all conflict with each other and with social welfare maximisation. Particularly relevant when justifying the specific fairness metric chosen for the Allianz model evaluation.

**Summary**
Argues that commonly used fairness metrics have unintended consequences when applied naively. Proposes "conditional use accuracy equality" as a more principled criterion. Shows through the bail and lending examples that equality constraints on error rates can lead to worse outcomes for the groups being "protected."

**Key concept / formula**
Calibration: $P(Y=1 \mid \hat{p}(X) = p) = p$ — model probabilities match true probabilities. Anti-classification: the model does not use protected attributes. Classification parity: equal error rates. Theorem: all three cannot hold simultaneously when $P(Y=1 \mid A=0) \neq P(Y=1 \mid A=1)$.

**How to apply at Allianz**
Motor fraud base rates vary by product line and postcode. Document these base rate differences and show that perfect calibration implies different error rates across groups — this is expected and not a failure of the model. Use this to defend against naive criticism of disparate error rates.

**What to write in the dissertation**
Cite in the fairness evaluation section. Use the impossibility result to frame the discussion: "following Corbett-Davies & Goel (2023), we acknowledge that calibration and classification parity cannot be simultaneously achieved given differential base rates across claim segments, and we prioritise calibration as the primary criterion."

---

### P25 · European Parliament & Council of the EU (2024) — EU AI Act

**Citation**
European Parliament & Council of the EU (2024). "Regulation (EU) 2024/1689 of the European Parliament and of the Council — Laying Down Harmonised Rules on Artificial Intelligence (Artificial Intelligence Act)." *Official Journal of the European Union*, L 2024/1689. Entered into force: 1 August 2024.

**Link** → https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=OJ:L_202401689
EIOPA factsheet: https://www.eiopa.europa.eu/document/download/b53a3b92-08cc-4079-a4f7-606cf309a34a_en
**Impact** — Primary EU legislation (no IF; regulatory force)

**Why this paper matters**
Classifies AI systems used in insurance risk assessment and claims handling as **high-risk** (Annex III). Requires technical documentation, bias testing, human oversight, and post-deployment monitoring. Provides the regulatory mandate for solving the SFP loop: it is not merely academic but a compliance obligation for Allianz UK.

**Summary**
The AI Act creates a risk-based framework: prohibited AI (social scoring, biometric surveillance), high-risk AI (credit, insurance, employment, law enforcement), and limited/minimal risk. For Allianz UK, the fraud scoring system is high-risk under Annex III point 5(b) (AI in insurance pricing and risk assessment) and point 6 (AI in law enforcement-adjacent tasks). Requirements include: risk management systems, data governance, transparency documentation, human oversight, accuracy and robustness requirements, and post-market monitoring.

**Key concept / formula**
Article 9 (Risk management system): continuous risk management cycle required throughout the lifecycle. Article 10 (Data governance): training data must be representative, free from errors, and complete. The SFP loop directly violates Article 10 — biased labels from selective investigation are not representative.

**How to apply at Allianz**
The dissertation's SFP detection framework (Build 02) and mitigation methods (Builds 05–06) can be positioned as the technical documentation and bias-testing component required by Articles 9–10 of the AI Act. Build 03 (Unbiased Evaluation) maps to Article 15 (accuracy requirements).

**What to write in the dissertation**
Cite in the ethics and regulatory chapter. State: "the fraud scoring system at Allianz UK falls within the high-risk category under Annex III of the EU AI Act (Regulation 2024/1689), which mandates bias testing and post-deployment monitoring. This dissertation provides a technical framework for satisfying those requirements."

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

**How to apply at Allianz**
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
The Allianz total loss model is a textbook selective-labels system: the model decides whether to scrap (`decision=1`) or send to garage (`decision=0`). Repair outcomes (`garage_outcome`) are observable only for `decision=0` rows. For scrapped cars the true repair outcome is structurally absent — not missing at random. This paper provides the exact formal framework for evaluating and learning from such systems, and is the direct citation for this class of problems in the fairness and causal ML literature.

**Summary**
Studies the problem of evaluating ML models when outcomes are only observed for a subset of cases determined by the model's own decisions (or a human predecessor). Formalises **selective labels bias**: the observed accuracy on the selected subset systematically overestimates true accuracy on the full population. Proposes evaluation strategies that account for structural missingness of outcomes in the unselected group, and derives conditions under which counterfactual performance can be bounded or estimated from observational data.

**Key concept / formula**
Let $\hat{Y}_i$ be the model prediction and $D_i \in \{0,1\}$ the binary decision (0 = garage, 1 = scrap). Outcome $Y_i$ is observable only when $D_i = 0$:
$$Y_i \text{ observed} \iff D_i = 0$$
Standard accuracy evaluated on $\{i : D_i = 0\}$ is biased because $D_i$ is a function of $\hat{Y}_i$. The paper derives conditions under which counterfactual performance on $\{i : D_i = 1\}$ can be bounded or estimated.

**How to apply at Allianz**
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
The label structure of the Allianz total loss dataset maps onto the PU learning setting. Cars sent to garage with confirmed total losses are **labeled positives** (`Y=1`, observed). Cars sent to garage that were successfully repaired are **labeled negatives** (`Y=0`). Scrapped cars have a forced label of 1 but their true repair outcome is unknown — they may have been repairable false positives. PU learning provides theory for training and evaluating classifiers under exactly this asymmetric observability structure.

**Summary**
Comprehensive survey of learning algorithms when training data consists of labeled positives and unlabeled examples (containing both true positives and true negatives). Covers: (a) the two main assumptions — single-training-set (SCAR) vs. selected-completely-at-random; (b) methods for estimating the class prior $\pi = P(Y=1)$ from unlabeled data; (c) algorithms including biased SVM, EM-based methods, two-step methods (spy technique), and cost-sensitive re-weighting; (d) evaluation criteria under PU assumptions.

**Key concept / formula**
PU risk decomposition: given labeled positives $\mathcal{P}$ and unlabeled $\mathcal{U}$ (true positive rate $\pi$), the risk of classifier $g$ is:
$$R(g) = \pi \cdot R^+(g) + (1-\pi) \cdot R^-(g)$$
where $R^+(g)$ and $R^-(g)$ are false-negative and false-positive risks. Estimating $\pi$ — the proportion of true total losses among scrapped cars — is the core estimation problem in the Allianz context.

**How to apply at Allianz**
Treat `model_v1_decision=1` rows (scrapped) as the "unlabeled" group: their observed label is 1, but the true fraction of genuine total losses $\pi$ is unknown. The `decision=0` rows with `outcome=1` are the labeled positives. Use PU learning methods to estimate $\hat{\pi}$, which quantifies the false-positive rate of the scrapping policy. This estimate directly informs the magnitude of SFP bias quantified in Builds 01 and 03.

**What to write in the dissertation**
Cite in Build 03 (Unbiased Evaluation) and Build 06 (Causal Mitigation). Frame the label contamination in `model_v1_observed_outcome` as a PU learning problem: "following Bekker & Davis (2020), we treat scrapped-car rows as unlabeled under the PU assumption, since their observed label of 1 reflects the scrapping decision rather than a verified repair outcome. We estimate the class prior $\hat{\pi}$ to quantify the magnitude of label contamination introduced by the SFP mechanism."

---

## Quick Reference Table

| # | First Author | Year | Short Title | Venue | URL | ~Citations |
|---|-------------|------|-------------|-------|-----|-----------|
| P1 | Neyman | 1923/1990 | Potential outcomes | Stat. Science | doi:10.1214/ss/1177012031 | 3,000 |
| P2 | Rubin | 1974 | Estimating causal effects | J. Educ. Psychol. | doi:10.1037/h0037350 | 9,800 |
| P3 | Horvitz & Thompson | 1952 | IPS estimator | JASA | doi:10.1080/01621459.1952.10483446 | 5,000 |
| P4 | Rosenbaum & Rubin | 1983 | Propensity score | Biometrika | doi:10.1093/biomet/70.1.41 | 25,000 |
| P5 | Pearl | 1995 | Causal diagrams / DAGs | Biometrika | doi:10.1093/biomet/82.4.669 | 5,000 |
| P6 | Heckman | 1979 | Sample selection bias | Econometrica | doi:10.2307/1912352 | 29,000 |
| P7 | Imbens & Wooldridge | 2009 | Econometrics of program eval. | J. Econ. Lit. | doi:10.1257/jel.47.1.5 | 8,000 |
| P8 | Angrist & Pischke | 2009 | Mostly Harmless Econometrics | Princeton UP | press.princeton.edu | 30,000 |
| P9 | Phua et al. | 2010 | Fraud detection survey | arXiv | arxiv.org/abs/1009.6119 | 795 |
| P10 | Barocas & Selbst | 2016 | Big Data's Disparate Impact | CA Law Rev. | ssrn.com/abstract=2477899 | 2,500 |
| P11 | Lum & Isaac | 2016 | To Predict and Serve? | Significance | doi:10.1111/j.1740-9713.2016.00960.x | 509 |
| P12 | Ensign et al. | 2018 | Runaway Feedback Loops | FAccT | arxiv.org/abs/1706.09847 | 650 |
| P13 | Corbett-Davies et al. | 2017 | Cost of Fairness (KDD '17) | KDD | dl.acm.org/doi/10.1145/3097983.3098095 | 1,445 |
| P14 | Liu et al. | 2018 | Delayed Impact of Fair ML | ICML | arxiv.org/abs/1803.04383 | 491 |
| P15 | Perdomo et al. | 2020 | Performative Prediction | ICML | arxiv.org/abs/2002.06673 | 325 |
| P16 | Chernozhukov et al. | 2018 | Double/Debiased ML | Econometrics J. | doi:10.1111/ectj.12097 | 6,000 |
| P17 | Sculley et al. | 2015 | Hidden Technical Debt in ML | NeurIPS | proceedings.neurips.cc | 4,000 |
| P18 | D'Amour et al. | 2022 | Underspecification in ML | JMLR | arxiv.org/abs/2011.03395 | 900 |
| P19 | Thompson | 1933 | Thompson Sampling | Biometrika | doi:10.1093/biomet/25.3-4.285 | 3,000 |
| P20 | Auer et al. | 2002 | UCB1 bandit algorithm | ML journal | doi:10.1023/A:1013689704352 | 7,000 |
| P21 | Russo et al. | 2018 | Tutorial on Thompson Sampling | FnT-ML | doi:10.1561/2200000070 | 3,000 |
| P22 | Hardt et al. | 2016 | Equality of Opportunity | NeurIPS | arxiv.org/abs/1610.02413 | 5,000 |
| P23 | Barocas, Hardt & Narayanan | 2023 | Fairness and ML (book) | MIT Press | fairmlbook.org | 4,000 |
| P24 | Corbett-Davies & Goel | 2023 | Measure and Mismeasure of Fairness | JMLR | arxiv.org/abs/1808.00023 | 700 |
| P25 | EU Parliament | 2024 | EU AI Act | EU OJ | eur-lex.europa.eu | — |
| P26 | Lattimore & Szepesvári | 2020 | Bandit Algorithms (book) | Cambridge UP | doi:10.1017/9781108571401 | 1,500 |
| P27 | Lakkaraju et al. | 2017 | Selective Labels Problem | KDD | dl.acm.org/doi/10.1145/3097983.3098066 | 450 |
| P28 | Bekker & Davis | 2020 | PU Learning survey | Machine Learning | doi:10.1007/s10994-020-05877-5 | 1,000 |

---

*All citation counts are approximate Google Scholar / Semantic Scholar figures as of mid-2026. Regulatory documents (P25) do not have citation counts. P27 and P28 added 2026-06-15 following domain confirmation (total loss prediction).*

---

## PDF Download Status

18 of 28 papers downloaded automatically to `literatures/p{N}.pdf`.
The remaining 10 are behind institutional paywalls or are paid books — download via your **University of Bristol library** login or **Allianz VPN**.

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

**Quick access for paywalled papers via UoB library:**
1. Go to https://www.bristol.ac.uk/library/
2. Search by DOI or title in the "Find a resource" search bar
3. Sign in with your UoB student credentials
4. Download PDF and save as `p{N}.pdf` in this folder
