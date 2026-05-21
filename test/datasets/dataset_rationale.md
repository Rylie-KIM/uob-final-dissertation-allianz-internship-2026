# Dataset Selection Rationale
*Why we use each dataset and what it proves*

---

## Summary Decision

| Stage | Dataset | Purpose | Why This Dataset |
|-------|---------|---------|-----------------|
| **Core loop demonstration** | Synthetic (`simulate_sfp_loop.py`) | Prove loop mechanism + test all methods | Ground truth known → can measure loop strength directly |
| **Realistic feature validation** | IEEE-CIS Fraud Detection (Kaggle) | Show methods work on real transaction data | Largest public fraud dataset; tabular features close to insurance claims |
| **Insurance-specific validation** | Porto Seguro Safe Driver (Kaggle) | Show methods on real insurance risk features | Only large public insurance dataset with structured features |
| **Benchmark** | COIL 2000 (UCI) | Classical insurance benchmark | Actual Dutch insurance policies; referenced in academic literature |

---

## Dataset 1: Synthetic Data — Primary Dataset

**Source:** `builds/01_sfp_simulation/simulate_sfp_loop.py` (self-generated)

**Size:** Configurable; default 50,000 claims, 5 model versions

**Why synthetic is ESSENTIAL for this project:**

The SFP loop research question is fundamentally about a **counterfactual**: what would the fraud rate be in claims we never investigated? This is unknowable in real data — by definition, uninvestigated claims have no fraud labels. Without ground truth, we cannot measure:
- True recall (we don't know how much fraud the model misses)
- Loop amplification factor (we don't know the true fraud rate)
- Whether IPS correction actually improves recall (we have nothing to compare to)

The synthetic dataset gives us access to:
```
true_fraud_prob  ← the DGP we designed (oracle)
true_fraud       ← ground truth label for ALL claims
observed_fraud   ← biased label only for investigated claims
model_vN_score   ← N generations of score-driven selection
```

This lets us MEASURE the gap between standard and IPS-corrected metrics against a known truth — which is the entire scientific contribution of the project.

**How the DGP is designed:**
```python
logit(P(fraud)) = -2.5                          # base rate ~8%
                + 0.8 * high_claim_amount        # large claims are riskier
                + 0.6 * night_time_claim         # night claims riskier
                + 1.2 * high_postcode_risk        # postcode correlates with fraud
                + 0.4 * prior_claims_count        # repeat claimants riskier
                + N(0, 0.3)                       # individual noise
```

This mimics real insurance fraud signal:
- Fraud rate ~8% (typical UK motor fraud rate: 5–12%)
- Multiple correlated features (not just one dominant predictor)
- Noise term → no perfect prediction → model is never 100% AUC

**Limitation:** Results on synthetic data overstate how clearly the loop can be detected, because the DGP is exactly what our model assumes. Real data has additional confounders, non-linearities, and temporal effects.

---

## Dataset 2: IEEE-CIS Fraud Detection — Real-World Validation

**Source:** Kaggle Competition "IEEE-CIS Fraud Detection" (2019)
**URL:** https://www.kaggle.com/c/ieee-fraud-detection/data
**Size:** 590,540 transactions × 434 features
**Fraud rate:** 3.5%

**Why this dataset:**

1. **Scale:** 590k transactions is large enough to simulate multiple model versions and still have statistical power for causal estimation
2. **Temporal structure:** Transactions are ordered in time → natural model version splits
3. **Tabular structure:** V1–V339 anonymised Vesta features, plus card, address, email features → similar to insurance claim features
4. **Realistic fraud signal:** Real-world fraud, not synthetic → methods must be robust to non-linearity and confounding

**How to adapt for SFP loop research:**

The IEEE-CIS dataset does not come with investigation flags. We **simulate the investigation policy** on top of the real fraud labels:

```python
# Step 1: Train model v1 on first 40% of data (unbiased — treat all true labels as known)
# Step 2: Score all remaining transactions with v1
# Step 3: Simulate investigation: investigate top-25% by score ONLY
# Step 4: Observe fraud only in investigated claims
# Step 5: Train v2 on biased labels from step 4
# Step 6: Repeat for v3, v4
```

This gives a real-feature dataset WITH a simulated loop — the best of both worlds.

**Features most similar to insurance claims:**
| IEEE-CIS Feature | Insurance Analogue |
|-----------------|-------------------|
| `TransactionAmt` | Claim amount |
| `card4` (visa/mastercard) | Payment method at inception |
| `addr1`, `addr2` | Postcode area |
| `D1`–`D15` (time deltas) | Time between events (prior claims) |
| `M1`–`M9` (match flags) | Identity verification flags |

**Access:** Free download via Kaggle account (requires account creation).

**File to load:**
```python
import pandas as pd
train = pd.read_csv('data/ieee_cis/train_transaction.csv')
identity = pd.read_csv('data/ieee_cis/train_identity.csv')
df = train.merge(identity, on='TransactionID', how='left')
```

---

## Dataset 3: Porto Seguro Safe Driver Prediction — Insurance-Native

**Source:** Kaggle Competition "Porto Seguro's Safe Driver Prediction" (2017)
**URL:** https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data
**Size:** 595,212 policies × 57 features
**Event rate (claim made):** 3.6%

**Why this dataset:**

1. **Actual insurance data:** Porto Seguro is a Brazilian insurer — this is real policy-level data, not simulated
2. **Insurance feature naming convention:** Features are grouped by type (`ps_ind_` = individual, `ps_reg_` = regional, `ps_car_` = car, `ps_calc_` = calculated)
3. **Demonstrates the pricing feedback loop:** The target is "did this policyholder file a claim?" — this is the exact outcome variable for the adverse selection SFP loop in motor insurance pricing
4. **Used in academic literature:** Several papers on fairness and feedback loops in insurance cite this dataset

**Limitation:** The target is claim-filed (yes/no), not fraud. This makes it better for demonstrating the **pricing/adverse selection loop** (Pillar 1, motor insurance) than the fraud detection loop.

**How to use for SFP research:**

```python
# The SFP loop here is the PRICING ADVERSE SELECTION loop:
# Model predicts high risk → higher premium → high-risk driver stays (adverse selection)
# → more claims in that segment → model reinforces "high risk"

# Simulate the loop:
# Step 1: Train risk model on first 60% of policies
# Step 2: Policies above risk threshold → apply "high premium" (simulate churn)
# Step 3: High-risk policies that would have churned are removed from next cohort
# Step 4: Retrain on remaining (biased) portfolio
# Step 5: Show that model overestimates risk for policies that stayed
```

**Key features for the loop:**
| Feature | Relevance |
|---------|-----------|
| `ps_reg_01`, `ps_reg_02`, `ps_reg_03` | Regional risk (postcode equivalent) |
| `ps_car_13` | Car risk factor (most predictive single feature) |
| `ps_ind_06_bin`–`ps_ind_18_bin` | Driver characteristics |

---

## Dataset 4: COIL 2000 — Academic Benchmark

**Source:** UCI Machine Learning Repository / CoIL Challenge 2000
**URL:** https://archive.ics.uci.edu/ml/datasets/Insurance+Company+Benchmark+%28COIL+2000%29
**Size:** 9,822 customers × 86 features

**Why this dataset:**

1. **Academic credibility:** Published in a challenge format; widely cited in insurance ML papers
2. **Dutch insurance policies:** Actual product data from a Dutch insurer (Tirion)
3. **Features:** Includes policy types (motor, home, bicycle, disability), premium amounts, and 44 contribution variables

**Limitation:** Very small by modern standards. However, it is the only publicly available insurance benchmark with multiple product lines — useful for demonstrating that the SFP framework generalises across Allianz's product range.

**Best use:** Quick validation that the loop detection framework produces sensible segment rankings on real insurance demographic data.

---

## Why NOT These Datasets

| Dataset | Reason Not Primary |
|---------|-------------------|
| Credit Card Fraud (Kaggle/ULB) | Only 284k rows, binary features, no temporal structure — too simple |
| Home Credit Default | Loan, not insurance; loop structure is different (credit denial ≠ investigation bias) |
| Fraud Detection in Financial Payments | Very small (6.3 million rows but synthetic financial, not insurance) |
| UK Claims Data (CUE, IFB) | Not publicly available — internal to insurers |

---

## Download Instructions

```bash
# Create data directory
mkdir -p data/ieee_cis data/porto_seguro data/coil2000

# IEEE-CIS (requires Kaggle CLI)
pip install kaggle
kaggle competitions download -c ieee-fraud-detection -p data/ieee_cis
cd data/ieee_cis && unzip "*.zip"

# Porto Seguro (requires Kaggle CLI)
kaggle competitions download -c porto-seguro-safe-driver-prediction -p data/porto_seguro
cd data/porto_seguro && unzip "*.zip"

# COIL 2000 (free, no login)
# Download from: https://archive.ics.uci.edu/ml/datasets/Insurance+Company+Benchmark+(COIL+2000)
# Or use ucimlrepo:
pip install ucimlrepo
python -c "
from ucimlrepo import fetch_ucirepo
coil = fetch_ucirepo(id=125)
coil.data.features.to_csv('data/coil2000/features.csv', index=False)
coil.data.targets.to_csv('data/coil2000/targets.csv', index=False)
"
```

---

## Data Usage Flow (Builds → Datasets)

```
Build 00 (EDA)
    └── Primary: Allianz dataset (when received)
    └── Fallback: Synthetic (for template testing)

Build 01 (Simulation)
    └── Synthetic ONLY (ground truth required)

Build 02 (Loop Detection)
    └── Synthetic (for development)
    └── IEEE-CIS (for real-feature validation)
    └── Porto Seguro (for insurance-native validation)

Build 03 (Unbiased Evaluation)
    └── Synthetic (for measuring IPS gap vs ground truth)
    └── IEEE-CIS with simulated investigation policy

Build 04 (Intervention Analysis)
    └── Synthetic (for RDD validation at known threshold)
    └── IEEE-CIS (for real-scale DiD)

Build 05 (Randomisation)
    └── Synthetic (for loop break demonstration)
    └── Porto Seguro (for pricing loop simulation)

Build 06 (Causal Mitigation)
    └── Synthetic (for debiased model validation vs ground truth)
    └── IEEE-CIS (for realistic feature debiasing demonstration)
```

---

## Licensing

| Dataset | Licence | Commercial Use |
|---------|---------|----------------|
| Synthetic (ours) | MIT | Yes |
| IEEE-CIS | Kaggle competition rules | Research only |
| Porto Seguro | Kaggle competition rules | Research only |
| COIL 2000 | UCI public domain | Yes (academic) |

All external datasets are for **research and educational purposes only** in the context of this internship project. They must not be used for commercial model deployment.
