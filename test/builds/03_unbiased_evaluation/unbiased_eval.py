"""
Build 03: Unbiased Performance Evaluation
==========================================
Demonstrates how standard evaluation overestimates model performance
when a self-fulfilling prophecy loop is present.

Implements:
    1. Standard biased evaluation (train/test on investigated claims only)
    2. IPS-corrected evaluation (Inverse Propensity Score weighting)
    3. Temporal holdout evaluation
    4. Side-by-side comparison

Key insight:
    Standard train/test gives high AUC because the test set is also
    biased by the same investigation policy that created the training labels.
    IPS correction re-weights claims by 1/P(investigated|features) to
    approximate evaluation on the full (unobserved) population.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. Propensity Score Estimation
# ─────────────────────────────────────────────

def estimate_propensity_scores(
    df: pd.DataFrame,
    feature_cols: list[str],
    investigation_col: str = 'investigated'
) -> np.ndarray:
    """
    Estimate P(investigated=1 | features) using logistic regression.

    This propensity score is used to:
        (a) Weight investigated claims so they represent the full population
        (b) Correct evaluation metrics for selection bias

    Returns:
        Array of propensity scores for all claims (including uninvestigated)
    """
    X = df[feature_cols].values
    y = df[investigation_col].values.astype(int)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    propensity_scores = model.predict_proba(X)[:, 1]

    # Clip to avoid extreme weights
    propensity_scores = np.clip(propensity_scores, 0.01, 0.99)

    print(f"Propensity score estimation:")
    print(f"  AUC (how well score predicts investigation): "
          f"{roc_auc_score(y, propensity_scores):.4f}")
    print(f"  Mean propensity: {propensity_scores.mean():.4f}")
    print(f"  Min/Max: {propensity_scores.min():.4f} / {propensity_scores.max():.4f}")

    return propensity_scores


# ─────────────────────────────────────────────
# 2. IPS-Corrected Metrics
# ─────────────────────────────────────────────

def compute_ips_corrected_fraud_rate(
    df: pd.DataFrame,
    fraud_col: str,
    investigation_col: str,
    propensity_scores: np.ndarray,
) -> float:
    """
    Estimate the true population fraud rate using IPS weighting.

    Standard estimate: P(fraud | investigated)  ← BIASED (higher than true rate)
    IPS estimate:      Σ(fraud_i * w_i) / N     where w_i = 1/propensity_i

    This is the Horvitz-Thompson estimator for missing data.
    """
    investigated_mask = df[investigation_col] == 1
    fraud_labels = df.loc[investigated_mask, fraud_col].values.astype(float)
    weights = 1.0 / propensity_scores[investigated_mask]

    # Horvitz-Thompson estimator
    ips_fraud_rate = np.sum(fraud_labels * weights) / len(df)

    return ips_fraud_rate


def compute_ips_corrected_auc(
    df: pd.DataFrame,
    fraud_col: str,
    score_col: str,
    investigation_col: str,
    propensity_scores: np.ndarray,
) -> float:
    """
    Compute IPS-corrected AUC using weighted ROC.

    Each investigated claim is up-weighted by 1/propensity to simulate
    evaluating on the full (unbiased) population.
    """
    investigated_mask = df[investigation_col] == 1
    sub = df[investigated_mask].copy()
    sub_scores = sub[score_col].values
    sub_labels = sub[fraud_col].values.astype(int)
    sub_weights = 1.0 / propensity_scores[investigated_mask]

    # Weighted AUC (approximate using sklearn with sample_weight)
    try:
        # sklearn's roc_auc_score does not support sample weights directly
        # Use weighted resampling as approximation
        rng = np.random.default_rng(42)
        normalized_weights = sub_weights / sub_weights.sum()
        resample_idx = rng.choice(len(sub), size=len(sub) * 3, replace=True, p=normalized_weights)
        auc = roc_auc_score(sub_labels[resample_idx], sub_scores[resample_idx])
    except Exception:
        auc = np.nan

    return auc


def compute_ips_corrected_recall(
    df: pd.DataFrame,
    fraud_col: str,
    score_col: str,
    investigation_col: str,
    propensity_scores: np.ndarray,
    threshold: float = 0.5,
    true_fraud_col: str = None,
) -> dict:
    """
    Estimate true population recall using IPS weighting.

    Standard recall = TP / (TP + FN)
        - Denominator is computed only on investigated claims → biased
        - Assumes no fraud in uninvestigated claims → wrong

    IPS recall:
        - Estimates total fraud in population using IPS
        - Computes recall against that estimated total
    """
    investigated_mask = df[investigation_col] == 1
    sub = df[investigated_mask].copy()
    weights = 1.0 / propensity_scores[investigated_mask]

    fraud_labels = sub[fraud_col].values.astype(int)
    model_preds = (sub[score_col].values >= threshold).astype(int)

    # Standard recall (biased — computed only on investigated)
    standard_recall = recall_score(fraud_labels, model_preds, zero_division=0)

    # IPS-estimated total fraud in population
    ips_total_fraud = np.sum(fraud_labels * weights)
    ips_detected_fraud = np.sum((fraud_labels * model_preds) * weights)
    ips_recall = ips_detected_fraud / ips_total_fraud if ips_total_fraud > 0 else np.nan

    result = {
        'standard_recall': round(standard_recall, 4),
        'ips_corrected_recall': round(ips_recall, 4) if not np.isnan(ips_recall) else None,
        'standard_precision': round(precision_score(fraud_labels, model_preds, zero_division=0), 4),
    }

    if true_fraud_col and true_fraud_col in df.columns:
        true_recall = recall_score(
            df[true_fraud_col].values,
            (df[score_col].values >= threshold).astype(int),
            zero_division=0
        )
        result['true_recall_ground_truth'] = round(true_recall, 4)

    return result


# ─────────────────────────────────────────────
# 3. Full Evaluation Comparison
# ─────────────────────────────────────────────

def full_evaluation_comparison(
    df: pd.DataFrame,
    score_col: str,
    fraud_col: str,
    investigation_col: str,
    feature_cols: list[str],
    true_fraud_col: str = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Compare standard biased metrics vs IPS-corrected metrics.

    Returns a summary DataFrame showing the gap between them.
    """
    # Estimate propensity scores
    propensity = estimate_propensity_scores(df, feature_cols, investigation_col)

    # Standard biased evaluation (on investigated claims only)
    investigated_mask = df[investigation_col] == 1
    sub = df[investigated_mask].dropna(subset=[fraud_col])
    labels = sub[fraud_col].astype(int)
    scores = sub[score_col]

    standard_auc = roc_auc_score(labels, scores)
    standard_precision = precision_score(labels, (scores >= threshold).astype(int), zero_division=0)
    standard_recall = recall_score(labels, (scores >= threshold).astype(int), zero_division=0)
    standard_brier = brier_score_loss(labels, scores)
    standard_fraud_rate = labels.mean()

    # IPS-corrected metrics
    ips_fraud_rate = compute_ips_corrected_fraud_rate(df, fraud_col, investigation_col, propensity)
    ips_auc = compute_ips_corrected_auc(df, fraud_col, score_col, investigation_col, propensity)
    ips_recall_results = compute_ips_corrected_recall(
        df, fraud_col, score_col, investigation_col, propensity, threshold, true_fraud_col
    )

    results = {
        'Metric': [
            'Fraud Rate Estimate',
            'AUC-ROC',
            'Precision',
            'Recall',
        ],
        'Standard (biased)': [
            f"{standard_fraud_rate:.4f}",
            f"{standard_auc:.4f}",
            f"{standard_precision:.4f}",
            f"{standard_recall:.4f}",
        ],
        'IPS-Corrected': [
            f"{ips_fraud_rate:.4f}",
            f"{ips_auc:.4f}" if not np.isnan(ips_auc) else "N/A",
            "N/A (unaffected by selection)",
            f"{ips_recall_results['ips_corrected_recall']}",
        ],
    }

    if true_fraud_col and true_fraud_col in df.columns:
        true_labels = df[true_fraud_col]
        true_fraud_rate = true_labels.mean()
        true_auc = roc_auc_score(true_labels, df[score_col])
        true_recall = recall_score(true_labels, (df[score_col] >= threshold).astype(int), zero_division=0)
        results['True (ground truth)'] = [
            f"{true_fraud_rate:.4f}",
            f"{true_auc:.4f}",
            "N/A",
            f"{true_recall:.4f}",
        ]

    comparison_df = pd.DataFrame(results)
    return comparison_df, propensity


# ─────────────────────────────────────────────
# 4. Temporal Holdout Evaluation
# ─────────────────────────────────────────────

def temporal_holdout_split(
    df: pd.DataFrame,
    date_col: str,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by time rather than randomly.

    Why this matters:
        Random split leaks future information into training.
        Temporal split respects the causal order of model deployment.
        If model v1 influenced labels in month 3, we must NOT use month 3
        data to train the model we're evaluating as if it were independent.
    """
    df = df.sort_values(date_col)
    cutoff_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:cutoff_idx].copy()
    test = df.iloc[cutoff_idx:].copy()

    print(f"Temporal split: train={len(train):,} claims, test={len(test):,} claims")
    if date_col in df.columns:
        print(f"  Train period: {df[date_col].iloc[0]} → {df[date_col].iloc[cutoff_idx-1]}")
        print(f"  Test period:  {df[date_col].iloc[cutoff_idx]} → {df[date_col].iloc[-1]}")

    return train, test


# ─────────────────────────────────────────────
# 5. Main Demo
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.append('../..')
    from builds.sfp_simulation.simulate_sfp_loop import generate_claims, run_full_simulation

    print("Generating simulated data (no exploration — loop active)...")
    df, metrics = run_full_simulation(n_claims=30_000, n_versions=3, epsilon=0.0)

    # Use version 3 as the current model under evaluation
    df['investigated'] = df['model_v3_investigated']
    df['observed_fraud'] = pd.to_numeric(df['model_v3_observed_fraud'], errors='coerce')
    df['observed_fraud'] = df['observed_fraud'].fillna(0).astype(int)
    # For uninvestigated, set to 0 but mask them
    invest_mask = df['model_v3_investigated'] == 1
    df.loc[~invest_mask, 'observed_fraud'] = np.nan

    feature_cols = ['high_amount', 'night_claim', 'high_postcode', 'prior_claims']

    print("\n" + "=" * 60)
    print("EVALUATION COMPARISON: Standard vs IPS-Corrected vs Ground Truth")
    print("=" * 60)

    comparison_df, propensity = full_evaluation_comparison(
        df=df,
        score_col='model_v3_score',
        fraud_col='observed_fraud',
        investigation_col='investigated',
        feature_cols=feature_cols,
        true_fraud_col='true_fraud',
        threshold=0.5,
    )

    print("\n")
    print(comparison_df.to_string(index=False))

    print("\n\n📌 KEY INSIGHT:")
    print("  Standard AUC appears high because the test set is biased by")
    print("  the same investigation policy that created the training labels.")
    print("  IPS-corrected and ground-truth metrics reveal the true performance gap.")
