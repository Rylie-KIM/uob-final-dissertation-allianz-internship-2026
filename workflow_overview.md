# Project Workflow Overview
*End-to-end workflow: from data receipt to production recommendation*

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ALLIANZ CLAIMS DATASET                           │
│         claims + model v1/v2/v3 scores + investigation flags        │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  BUILD 00: Data Exploration & Audit                                 │
│  • Profile dataset (missing values, distributions, temporal range)  │
│  • Fraud rate analysis by model version and product line            │
│  • Investigation coverage map (which segments are over/under-       │
│    investigated relative to model score)                            │
│  OUTPUT: EDA report, data quality assessment, loop presence flag    │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  BUILD 02: Loop Detection Framework                                 │
│  • Step 1: Temporal prediction correlation (ρ across versions)      │
│  • Step 2: Label mechanism test (MNAR test, investigation AUC)      │
│  • Step 3: Action-outcome confounding (rate ratio, Granger test)    │
│  • Step 4: Segment blind spot analysis (investigation gap map)      │
│  OUTPUT: Loop risk score (0–1), severity, blind spot map            │
└──────────────┬──────────────────────────┬───────────────────────────┘
               │                          │
               ▼                          ▼
┌──────────────────────────┐  ┌──────────────────────────────────────┐
│  BUILD 03: Unbiased      │  │  BUILD 04: Intervention Analysis     │
│  Evaluation              │  │                                      │
│  • IPS-corrected metrics │  │  • DiD: model version as natural     │
│  • Temporal holdout      │  │    experiment                        │
│  • True vs biased AUC    │  │  • RDD: causal effect at threshold   │
│  OUTPUT: Evaluation gap  │  │  • PSM: ATE of investigation         │
│  report                  │  │  OUTPUT: Causal estimates of loop    │
└──────────────────────────┘  └──────────────────────────────────────┘
               │                          │
               └──────────┬───────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  BUILD 05: Randomisation Strategy                                   │
│  • Simulate epsilon-greedy vs pure model vs Thompson sampling       │
│  • Track recall recovery and loop break over time                   │
│  • Cost-benefit analysis for each ε value                           │
│  OUTPUT: Optimal exploration policy recommendation with cost        │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  BUILD 06: Causal Mitigation                                        │
│  • DoWhy DAG specification                                          │
│  • IPW re-weighted debiased model training                          │
│  • Segment-level evaluation: debiased vs standard model             │
│  OUTPUT: Debiased model + evaluation benchmark + debiasing pipeline │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  FINAL OUTPUTS                                                      │
│  1. Loop risk score per product line                                │
│  2. Ranked blind spot map (segments + estimated missed fraud)       │
│  3. Policy recommendation (optimal ε for exploration)              │
│  4. Debiased retrained model (better calibrated on blind spots)     │
│  5. Written report + stakeholder presentation                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Build-by-Build Workflow Detail

### Build 00: Data Exploration (`builds/00_data_exploration/`)

**Input:** Raw Allianz dataset (CSV / Parquet)
**Script:** `eda_template.py`

**Run order:**
```bash
# 1. Load and profile
python eda_template.py --filepath data/allianz_claims.csv

# 2. Review output:
#    - temporal_trends.png  (fraud/investigation rate over time)
#    - coverage_by_decile.csv  (investigation rate by score decile)
#    - missing_value_report.csv
```

**Decision gate:** If fraud label coverage < 20% (i.e., >80% claims uninvestigated), IPS correction is essential (proceed to Build 03). If coverage is < 5%, simulation-only approach may be more reliable.

---

### Build 01: SFP Simulation (`builds/01_sfp_simulation/`)

**Input:** None (synthetic data generated internally)
**Script:** `simulate_sfp_loop.py`

**Run order:**
```bash
# 1. Run full simulation with no exploration
python simulate_sfp_loop.py

# 2. Key outputs:
#    - sfp_loop_degradation.png  (4 panels: AUC, recall, blind spot, amplification)
#    - Printed metrics table per model version

# 3. Run policy comparison
# (calls compare_policies() inside the script)
```

**What to interpret:**
- AUC on investigated claims: should appear to *increase* over versions (loop makes model look better)
- True recall: should *decrease* over versions (loop creates blind spots)
- Loop amplification: should *increase* (fraud rate in top quartile grows relative to true rate)
- With ε=0.05: recall should stabilise or recover

---

### Build 02: Loop Detection (`builds/02_loop_detection_framework/`)

**Input:** Claims DataFrame with model scores, investigation flags, fraud labels
**Script:** `loop_detector.py`

**Run order:**
```python
from loop_detector import SFPDetector

detector = SFPDetector(
    claims_df=df,
    model_versions=['v1', 'v2', 'v3'],
    investigation_col='investigated',
    fraud_label_col='fraud_label',
    feature_cols=['amount', 'night_claim', 'postcode_risk', 'prior_claims'],
)
report = detector.run_detection(segment_cols=['product_line', 'claim_type', 'postcode_area'])
detector.print_report(report)
```

**Output:**
```
Loop Risk Score: 75%  [HIGH]
Steps Flagged: 3 / 4
Flags:
  1. Versions v1 and v2 are highly correlated (ρ=0.891) — loop may be reinforcing scores
  2. Version v2: Model score predicts investigation with AUC=0.847 — MNAR violation
  3. Fraud rate in high-score investigated claims (0.342) is 8.6× higher than low-score (0.040)
Blind Spots:
  → product_line=commercial | risk=0.612 | investigated=0.12 | gap=+7 ranks
  → claim_type=theft | risk=0.584 | investigated=0.18 | gap=+5 ranks
```

---

### Build 03: Unbiased Evaluation (`builds/03_unbiased_evaluation/`)

**Input:** Claims DataFrame + model score column + investigation + fraud labels
**Script:** `unbiased_eval.py`

**Key function:**
```python
comparison_df, propensity = full_evaluation_comparison(
    df=df,
    score_col='model_v3_score',
    fraud_col='fraud_label',
    investigation_col='investigated',
    feature_cols=feature_cols,
    true_fraud_col='true_fraud',  # omit if no ground truth
)
print(comparison_df.to_string(index=False))
```

**Expected output shape:**
```
Metric               Standard (biased)  IPS-Corrected   True (ground truth)
Fraud Rate Estimate  0.1850             0.0820          0.0800
AUC-ROC              0.8934             0.7612          0.7450
Recall               0.7820             0.4230          0.4180
```

Note the gap between standard and IPS-corrected/true — this IS the loop effect.

---

### Build 04: Intervention Analysis (`builds/04_intervention_analysis/`)

**Input:** Claims DataFrame + model versions as natural experiment
**Script:** `intervention_analysis.py`

**Key analysis:**
1. RDD at investigation threshold: Is there a discontinuous jump in fraud discovery?
2. PSM: Does investigation cause fraud labels, or just reveal them?

**Interpretation guidance:**
- RDD estimate ≈ 0.15+ : investigation is creating fraud labels (strong loop)
- PSM ATE ≈ 0.05–0.10 : some causal effect of investigation on label
- PSM ATE ≈ 0 : investigation reveals genuine fraud (model is working correctly)

---

### Build 05: Randomisation Strategy (`builds/05_randomisation_strategy/`)

**Input:** Claims DataFrame (synthetic or real)
**Script:** `randomisation_policy.py`

**Comparison output table:**
```
Policy              Final Recall  Avg Recall  Total Investigations  Net Benefit (£)  ROI
Pure Model (ε=0)    0.2341        0.2689      12,500                £125,000         42%
ε-Greedy (ε=0.05)   0.3820        0.3240      12,625                £195,000         64%
ε-Greedy (ε=0.10)   0.4430        0.3820      12,750                £235,000         76%
ε-Greedy (ε=0.20)   0.5120        0.4490      13,000                £265,000         84%
Thompson Sampling   0.4680        0.4020      12,750                £248,000         79%
```

**Recommendation rule:**
- If ROI improves by ≥ 10% going from ε=0 to ε=0.05 → implement ε=0.05
- If investigation budget is constrained → use Thompson Sampling (more efficient exploration)

---

### Build 06: Causal Mitigation (`builds/06_causal_mitigation/`)

**Input:** Claims training DataFrame + test DataFrame
**Script:** `causal_mitigation.py`

**Debiased vs Standard model comparison:**
```
Model                AUC (vs true)  Recall (vs true)  Precision  Brier  ECE
Standard (biased)    0.7450         0.4180             0.7234     0.087  0.042
IPW Debiased         0.7812         0.5340             0.6980     0.079  0.031
```

Key improvements in debiased model:
- Higher recall (finds more of the true fraud, including in blind spots)
- Lower calibration error (better-calibrated scores)
- Modest precision decrease (acceptable trade-off for the recall gain)

---

## Files Created / Checklist

### Domain Knowledge
- [x] `domain/sfp_loops_in_insurance.md`
- [x] `domain/claims_types_and_sfp_by_product.md`
- [x] `domain/sfp_identification_framework.md`
- [x] `domain/model_evaluation_bias.md`
- [x] `domain/intervention_analysis.md`
- [x] `domain/randomisation_strategies.md`
- [x] `domain/causal_inference.md`
- [x] `domain/insurance_uk_overview.md`

### Builds
- [x] `builds/00_data_exploration/eda_template.py`
- [x] `builds/01_sfp_simulation/simulate_sfp_loop.py`
- [x] `builds/02_loop_detection_framework/loop_detector.py`
- [x] `builds/03_unbiased_evaluation/unbiased_eval.py`
- [x] `builds/04_intervention_analysis/intervention_analysis.py`
- [x] `builds/05_randomisation_strategy/randomisation_policy.py`
- [x] `builds/06_causal_mitigation/causal_mitigation.py`

### Documents
- [x] `docs/motivation_statement.md`
- [x] `docs/project_plan_detailed.md`
- [x] `docs/workflow_overview.md`

---

## Python Environment Setup

```bash
# Create environment
conda create -n allianz-sfp python=3.11
conda activate allianz-sfp

# Install dependencies
pip install numpy pandas scikit-learn scipy matplotlib seaborn
pip install dowhy econml causalml
pip install lightgbm xgboost
pip install jupyter nbconvert
pip install statsmodels pingouin

# Optional (large datasets)
pip install polars pyarrow
```

---

## Interview Demo Script

When asked "show me what you've built", walk through in this order:

1. **Explain the loop** using `domain/sfp_loops_in_insurance.md` fraud detection diagram
2. **Show the simulation**: run `simulate_sfp_loop.py`, show the recall degradation plot
3. **Show the detection framework**: `loop_detector.py` on simulated data, explain the 4 steps
4. **Show IPS-corrected evaluation**: side-by-side table from `unbiased_eval.py`
5. **Show randomisation comparison**: policy comparison plot from `randomisation_policy.py`
6. **Show causal mitigation**: before/after table from `causal_mitigation.py`

Total demo time: ~10 minutes. Each build has a `if __name__ == '__main__'` block you can run live.
