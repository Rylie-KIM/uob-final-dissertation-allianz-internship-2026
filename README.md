# uob-final-dissertation-allianz-internship-2026


Identifying and Mitigating Self-Fulfilling Prophecy Loops in Machine Learning 

- A self-fulfilling prophecy in machine learning occurs when a model's predictions influence the outcome in a way that reinforces the model's original prediction, creating a feedback loop.
- This can lead to biased or inaccurate results, especially in systems that interact with human behaviour or decision-making processes. 
- Identifying these loops can be tricky. 
- Can we create a framework to identify these loops and suggest for improvements in the future?
- method 1) We can start from unbiased performance evaluation and intervention analysis to assess model driven impact.
- method 2) Then potentially exploring randomisation strategies and causal inference technics for mitigation.
- A dataset containing insurance claims, 
- along with several versions of model predictions, is prepared and available for analysis.


# Business Context 
여

# Model Training Methodology
- **Target maturation time**: ~1–2 months. The total loss outcome (whether a car is genuinely repairable or not) takes time to be confirmed. The most recent 2 months of data are excluded from training to ensure labels are fully matured and not provisional.
- **Out-of-time (OOT) holdout**: ~6 months of the most recent (non-excluded) data is held out as an out-of-time validation set. This tests whether the model generalises to future data and is not just overfitting to historical patterns.
- **Train / test split**: 80-20 random split on the remaining data (after removing the maturation buffer and OOT holdout).

```
Full data timeline
──────────────────────────────────────────────────────────────────────►
│        Training + Test (80/20 split)        │   OOT (6m)  │ excl. │
│                                             │             │  (2m) │
```

- The OOT holdout is temporally separated — it always comes *after* training data, not randomly sampled from it. This reflects real deployment conditions where the model is applied to future unseen claims.

## Data Split Roles

| Split | When used | Purpose |
|---|---|---|
| **Train** | During training | Learn model parameters |
| **Validation** | During training | Hyperparameter tuning, early stopping |
| **Test** | After training | Report final performance metrics |
| **OOT** | After training | Verify the model holds up on future data |

```
Full dataset
├── Train       ┐
├── Validation  ┼── same time period, random split
├── Test        ┘
└── OOT             ← temporally later data
```