# Detailed Project Plan: Identifying and Mitigating Self-Fulfilling Prophecy Loops in ML
*Allianz UK Data Science Internship — Research Project Plan*
*Seoyeon, UoB MSc Data Science, 2026*

---

## 1. Project Overview

### 1.1 Problem Statement

A self-fulfilling prophecy (SFP) loop in machine learning occurs when:

```
Model Prediction → Decision/Action → Observed Outcome → Training Label → Model Retrained
       ↑_______________________________________________________________↑
                        (loop: output corrupts input)
```

In insurance, this is not hypothetical — it is structural. A fraud detection model decides *which claims to investigate*, meaning fraud is only ever found where the model looked. This means:
- The training label "fraud = 1" is not ground truth — it is *investigator-chosen*
- The model's recall on uninvestigated claims is unknowable from standard evaluation
- Over time, model versions reinforce each other's biases

### 1.2 Research Questions

1. **Detection:** Can we statistically detect the presence and magnitude of an SFP loop in insurance claims data with multiple model versions?
2. **Evaluation:** How much does standard train/test evaluation overestimate model performance when a feedback loop is present?
3. **Randomisation:** Does introducing random exploration (epsilon-greedy policy) recover unbiased fraud rate estimates over time?
4. **Causal Mitigation:** Can we use causal inference techniques to produce a debiased training dataset that breaks the loop in future model versions?

### 1.3 Hypothesis

If model version N predictions causally influence version N+1 training labels (beyond what true underlying risk explains), then:
- Standard metrics will overestimate precision
- Recall on unexplored segments will be systematically underestimated
- Re-weighting or random exploration will recover closer-to-true performance

---

## 2. Dataset Description

### 2.1 Expected Structure (Allianz-provided)

| Column | Type | Description |
|--------|------|-------------|
| `claim_id` | str | Unique claim identifier |
| `claim_date` | date | Date claim was submitted |
| `product_line` | cat | motor, home, pet, commercial, legal |
| `claim_amount` | float | Claimed amount (GBP) |
| `claim_type` | cat | theft, damage, injury, etc. |
| `claimant_postcode_area` | cat | First 2-4 chars of postcode |
| `investigated` | bool | Was the claim investigated? |
| `fraud_label` | int | 0/1 — fraud confirmed (ONLY for investigated claims) |
| `model_v1_score` | float | Fraud risk score from model version 1 |
| `model_v2_score` | float | Fraud risk score from model version 2 |
| `model_v3_score` | float | Fraud risk score from model version 3 |
| `investigation_threshold_v1` | float | Score threshold used in v1 deployment |
| `settlement_amount` | float | Actual payout (may differ from claimed amount) |

### 2.2 Key Properties of the Data

- **Missing fraud labels on uninvestigated claims** — this is fundamental; it's not missing at random
- **Multiple model versions** — natural experiment; different versions made different investigation decisions
- **Temporal ordering** — v1 deployed first, v2 built on v1's training data, v3 built on v2's
- **Censored outcomes** — any claim never investigated has no fraud label (survives uncensored only if scored above threshold)

### 2.3 Synthetic Data Schema (for builds before Allianz data arrives)

```python
# builds/00_data_exploration/synthetic_schema.py
{
    'n_claims': 50_000,
    'true_fraud_rate': 0.08,  # true population rate, unknown to model
    'investigation_rate_v1': 0.25,  # model v1 investigates top 25%
    'investigation_rate_random': 0.05,  # additional random sample
    'label_noise': 0.02,  # investigator error rate
    'loop_strength': 0.6  # correlation between v1 score and v2 training label
}
```

---

## 3. Methodology

### 3.1 Phase 0: Data Exploration & Audit (Build 00)

**Goal:** Understand the dataset structure, identify missingness patterns, validate that the loop structure exists.

**Steps:**
1. Load and profile dataset (shape, dtypes, missing values)
2. Compute fraud rate by model version — expect it to *increase* over versions if loop is present
3. Plot investigation rate by model score quartile — expect strong correlation
4. Compute coverage: % of claims with fraud label vs not
5. Temporal trend analysis: fraud rate over time, by model version deployment date
6. Segment analysis: fraud rate by postcode area, claim type, product line

**Output:** EDA report as Jupyter notebook + summary statistics CSV

---

### 3.2 Phase 1: SFP Simulation (Build 01)

**Goal:** Prove the loop mechanism on synthetic data before applying to real data.

**Simulation Design:**

```
1. Generate 50,000 synthetic claims with TRUE fraud labels (ground truth we know)
2. Deploy Model v1 (logistic regression trained on initial unbiased batch)
3. Apply Model v1: investigate only top-25% scored claims
4. Observe fraud ONLY in investigated claims → biased training set for v2
5. Train Model v2 on biased labels
6. Compare v1 vs v2 performance against ground truth
7. Repeat for v3, v4 → show degradation
8. Show: recall on uninvestigated segments drops each generation
```

**Key Metric:** `loop_amplification_factor` = (model_vN_fraud_rate_in_top_segment) / (true_fraud_rate)

**Expected Result:** By v3, the model is excellent at finding fraud where it already looks (high precision), but has systematically missed fraud in segments it never investigated (recall collapses for those segments).

---

### 3.3 Phase 2: Loop Detection Framework (Build 02)

**Goal:** Build a reusable Python module that inputs claims + model versions and outputs a loop risk score.

**Four-Step Detection Algorithm:**

```
Step 1: Temporal Prediction Correlation
  - Compute Spearman correlation between model_vN_score and model_v(N+1)_score
  - If rho > 0.85 across versions: flag potential loop
  - Control for claim features (partial correlation)

Step 2: Label Generation Mechanism Test
  - Test whether P(fraud_label=1 | investigated=True, score=s) depends on s
  - If P(investigated=True | score=s) >> P(investigated=True | score<threshold):
    → investigation is score-driven (necessary condition for loop)

Step 3: Action-Outcome Confounding
  - Test: does investigation_decision Granger-cause fraud_label?
  - Compute: P(fraud_found | investigated) vs P(fraud_found | random_sample)
  - If P(fraud_found|investigated) >> P(fraud_found|random): loop is active

Step 4: Segment Blind Spot Analysis
  - For each claim segment (postcode, type, value band):
    compute "investigation gap" = |model_score_rank - investigation_rate_rank|
  - High gap segments = blind spots in the model
  - Output: ranked list of blind spot segments + estimated missed fraud
```

**Module Interface:**

```python
from builds.loop_detection_framework import SFPDetector

detector = SFPDetector(claims_df, model_versions=['v1', 'v2', 'v3'])
report = detector.run_detection()
# Returns: loop_risk_score (0-1), flagged_segments, temporal_correlation_matrix
```

---

### 3.4 Phase 3: Unbiased Performance Evaluation (Build 03)

**Goal:** Show how much standard evaluation overestimates performance; implement IPS-corrected metrics.

**Standard vs Unbiased Evaluation:**

| Metric | Standard (biased) | Unbiased method |
|--------|-------------------|-----------------|
| Precision | Computed on investigated claims only | Same (unaffected — fraud rate within investigated) |
| Recall | Unknown — denominator (total fraud) is unknown | IPS-weighted recall estimate |
| AUC-ROC | Computed on investigated claims | Temporal holdout on random-investigation period |
| F1 | Overestimated (high recall bias) | IPS-corrected F1 |

**Inverse Propensity Score (IPS) Correction:**

```
IPS weight for claim i = 1 / P(investigated=1 | features_i)

where P(investigated=1 | features_i) estimated by:
  - Logistic regression on investigation decision
  - Features: model score, claim features (NOT fraud label)

IPS-corrected fraud rate:
  = sum(fraud_label_i * IPS_weight_i for investigated claims) / sum(IPS_weight_i)
```

**Temporal Holdout Strategy:**
- Split data by model deployment date
- Never use future model version's data to evaluate past versions
- Use random investigation period (if available) as unbiased test set

---

### 3.5 Phase 4: Intervention Analysis (Build 04)

**Goal:** Use multiple model versions as a natural experiment to measure causal effect of model decisions.

**Difference-in-Differences Setup:**

```
Treatment: claims scored above threshold by model vN → investigated
Control: claims scored below threshold → not investigated

Pre-period: before model vN deployment (model v(N-1) in use)
Post-period: after model vN deployment

DiD estimate: (fraud_rate_treated_post - fraud_rate_treated_pre) -
              (fraud_rate_control_post - fraud_rate_control_pre)

If DiD ≠ 0: model version change causally shifted fraud detection pattern
```

**Regression Discontinuity Design (RDD):**
- Use the investigation score threshold as a cutoff
- Claims just above threshold vs just below = quasi-random assignment
- Estimate: local average treatment effect of investigation on fraud label
- This cleanly isolates: does investigation *cause* fraud discovery, or did those claims have more fraud anyway?

**Expected Finding:** Near the threshold, claims just above and below have similar fraud risk but massively different investigation rates → any difference in fraud labels at the threshold is due to the investigation, not underlying risk.

---

### 3.6 Phase 5: Randomisation Strategy (Build 05)

**Goal:** Show that epsilon-greedy random exploration breaks the loop.

**Policy Comparison:**

| Policy | Description | Exploration Rate |
|--------|-------------|-----------------|
| Pure model | Investigate top-k by model score | 0% random |
| Epsilon-greedy (ε=0.05) | 95% model-driven, 5% random | 5% |
| Epsilon-greedy (ε=0.10) | 90% model-driven, 10% random | 10% |
| Pure random | Random sample | 100% |

**Simulation Steps:**
1. Run each policy for 10,000 claims (10 time periods, 1000 claims each)
2. At each period: retrain model on accumulated data from that policy
3. Track: fraud rate convergence to true rate, recall recovery, blind spot reduction
4. Show: with ε=0.05, loop breaks within 5 periods while maintaining ~90% of model precision

**Cost-Benefit Analysis:**
- Cost of random investigation: investigating some low-fraud claims
- Benefit: recovering blind spots, unbiased retraining data, regulatory compliance
- Optimal ε: where marginal benefit of bias reduction = marginal cost of wasted investigation

---

### 3.7 Phase 6: Causal Mitigation (Build 06)

**Goal:** Apply DoWhy to estimate causal effect; build debiasing pipeline.

**Causal Graph (DAG):**

```
True_Risk → Fraud_Label
True_Risk → Model_Score
Model_Score → Investigation_Decision
Investigation_Decision → Fraud_Label  ← THIS IS THE CONFOUNDER
Investigation_Decision → Training_Data → Next_Model_Score
True_Risk → Claim_Features
Claim_Features → Model_Score
```

**Identification Strategy:**
- `Investigation_Decision` confounds `Model_Score → Fraud_Label` relationship
- Backdoor criterion: condition on `Investigation_Decision` to block confounding path
- IPW re-weighting: weight each training example by `1/P(investigated | features)`

**DoWhy Pipeline:**

```python
import dowhy
model = dowhy.CausalModel(
    data=claims_df,
    treatment="investigation_decision",
    outcome="fraud_label",
    graph=causal_dag_string
)
identified_estimand = model.identify_effect()
causal_estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.propensity_score_weighting"
)
```

**Debiased Training Dataset:**
1. Estimate propensity score: P(investigated | claim_features)
2. Assign IPW weights to all investigated claims
3. Retrain model on IPW-weighted dataset
4. Compare model v_debiased vs model vN on held-out random investigation period
5. Show: debiased model has better calibration on previously unseen segments

---

## 4. Evaluation Metrics

| Metric | Description | Why It Matters |
|--------|-------------|----------------|
| Loop Amplification Factor | fraud_rate_top_quartile / true_fraud_rate | Quantifies loop strength |
| IPS-Corrected Recall | Recall estimated across full population | True measure of fraud detection coverage |
| Blind Spot Score | % of fraud in never-investigated segments | Regulatory risk indicator |
| Calibration Error (ECE) | Difference between predicted and actual fraud rate | Model reliability |
| DiD Estimate | Causal effect of model deployment on fraud patterns | Intervention effectiveness |
| Convergence Rate (ε-greedy) | Periods to recover ±1% of true fraud rate | Exploration efficiency |

---

## 5. Expected Outcomes

### 5.1 Research Outputs

1. **SFP Detection Framework** — Python module: `SFPDetector` class that produces loop risk scores and blind spot maps for any claims dataset with model versions
2. **Evaluation Benchmark** — Quantified gap between standard and IPS-corrected AUC/F1 on the Allianz dataset
3. **Intervention Recommendation** — Evidence-based recommendation on optimal ε for epsilon-greedy exploration policy, with cost-benefit analysis
4. **Debiasing Pipeline** — Reusable IPW re-weighting pipeline that produces training datasets with reduced loop effect

### 5.2 Business Deliverables for Allianz

1. A loop risk score for each product line in the current dataset
2. A ranked list of blind spot segments (postcodes, claim types, value bands) with estimated missed fraud volume
3. A policy recommendation with cost estimates (investigation budget required to break the loop)
4. A debiased retrained model with evaluation showing improvement vs current version

---

## 6. Timeline (3-Month Internship)

| Month | Phase | Deliverable |
|-------|-------|-------------|
| Month 1, Week 1-2 | Data onboarding + EDA (Build 00) | EDA report, data quality assessment |
| Month 1, Week 3-4 | SFP Detection Framework (Build 02) | Loop risk scores across product lines |
| Month 2, Week 1-2 | Unbiased Evaluation (Build 03) | Evaluation benchmark report |
| Month 2, Week 3-4 | Intervention Analysis (Build 04) | DiD/RDD causal estimates |
| Month 3, Week 1-2 | Randomisation Strategy (Build 05) | Policy recommendation + simulation results |
| Month 3, Week 3 | Causal Mitigation (Build 06) | Debiased training pipeline + retrained model |
| Month 3, Week 4 | Write-up + presentation | Final report + stakeholder presentation |

---

## 7. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Allianz dataset has no random investigation sample | High | High | Use RDD at threshold + simulation to validate methods |
| Loop signal is weak in the data | Medium | Medium | Extend to multiple product lines to aggregate signal |
| DoWhy DAG misspecified | Medium | High | Sensitivity analysis; test multiple DAG structures |
| Data too small for causal methods | Low | High | Use simulation to validate methods before applying to real data |
| FCA/GDPR data access constraints | Medium | Medium | Work within pseudonymised data; no individual-level output |

---

## 8. Technical Stack

```
Language:     Python 3.11+
Data:         pandas, numpy, polars (large data)
ML:           scikit-learn, lightgbm, xgboost
Causal:       dowhy, econml, causalml
Viz:          matplotlib, seaborn, plotly
Stats:        scipy, statsmodels, pingouin
Notebooks:    Jupyter, nbconvert
Version ctrl: git + GitHub
Environment:  conda / venv
```

---

## 9. Folder Structure

```
uob-ds-final-project-internship-allianz/
├── PLAN.md
├── README.md
├── docs/
│   ├── project_plan_detailed.md     ← this file
│   ├── motivation_statement.md      ← 500-word application statement
│   └── workflow_overview.md         ← build workflow diagram
├── domain/
│   ├── sfp_loops_in_insurance.md         ✅ done
│   ├── claims_types_and_sfp_by_product.md ✅ done
│   ├── sfp_identification_framework.md   ← Pillar 2
│   ├── model_evaluation_bias.md          ← Pillar 3
│   ├── intervention_analysis.md          ← Pillar 4
│   ├── randomisation_strategies.md       ← Pillar 5
│   ├── causal_inference.md               ← Pillar 6
│   └── insurance_uk_overview.md          ← Pillar 7
└── builds/
    ├── 00_data_exploration/
    │   ├── eda_template.py
    │   └── eda_notebook.ipynb
    ├── 01_sfp_simulation/
    │   ├── simulate_sfp_loop.py
    │   └── run_simulation.ipynb
    ├── 02_loop_detection_framework/
    │   ├── loop_detector.py
    │   └── demo_detection.ipynb
    ├── 03_unbiased_evaluation/
    │   ├── unbiased_eval.py
    │   └── evaluation_comparison.ipynb
    ├── 04_intervention_analysis/
    │   ├── intervention_analysis.py
    │   └── causal_experiment.ipynb
    ├── 05_randomisation_strategy/
    │   ├── randomisation_policy.py
    │   └── policy_comparison.ipynb
    └── 06_causal_mitigation/
        ├── causal_mitigation.py
        └── debiasing_pipeline.ipynb
```
