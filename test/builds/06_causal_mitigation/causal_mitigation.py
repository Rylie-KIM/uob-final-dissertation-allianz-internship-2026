"""
Build 06: Causal Inference Mitigation
=======================================
Applies causal inference techniques to break the SFP loop in training data.

Methods:
    1. Causal DAG specification (using DoWhy)
    2. Backdoor adjustment via IPW re-weighting
    3. Propensity score re-weighted model training
    4. Debiased model vs standard model comparison

Key insight:
    The investigation decision confounds the relationship between
    claim features and fraud labels. By conditioning on (weighting by)
    the investigation propensity, we remove this confounding and
    recover a training signal that reflects true fraud probability,
    not investigator-driven label bias.

Causal DAG:
    True_Risk → Fraud_Label
    True_Risk → Claim_Features → Model_Score
    Model_Score → Investigation_Decision
    Investigation_Decision → Fraud_Label (confounding path!)
    Investigation_Decision → Next_Training_Data → Next_Model_Score

Identification: Backdoor criterion
    Block the path: Investigation_Decision → Fraud_Label
    by conditioning on Investigation_Decision (weighting by propensity)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

try:
    import dowhy
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False
    print("DoWhy not installed. Running IPW pipeline without DoWhy. Install with: pip install dowhy")


FEATURE_COLS = ['high_amount', 'night_claim', 'high_postcode', 'prior_claims']


# ─────────────────────────────────────────────
# 1. Causal Graph (DAG) Specification
# ─────────────────────────────────────────────

CAUSAL_DAG_GML = """
graph [
    directed 1
    node [ id "claim_features" label "claim_features" ]
    node [ id "true_risk" label "true_risk" ]
    node [ id "model_score" label "model_score" ]
    node [ id "investigation" label "investigation" ]
    node [ id "fraud_label" label "fraud_label" ]
    edge [ source "true_risk" target "fraud_label" ]
    edge [ source "true_risk" target "claim_features" ]
    edge [ source "claim_features" target "model_score" ]
    edge [ source "model_score" target "investigation" ]
    edge [ source "investigation" target "fraud_label" ]
]
"""

# DoWhy dot notation (alternative format)
CAUSAL_DAG_DOT = """
digraph {
    true_risk -> fraud_label;
    true_risk -> claim_features;
    claim_features -> model_score;
    model_score -> investigation;
    investigation -> fraud_label;
}
"""


def run_dowhy_analysis(
    df: pd.DataFrame,
    treatment_col: str = 'investigated',
    outcome_col: str = 'observed_fraud',
    feature_cols: list[str] = None,
) -> dict:
    """
    Use DoWhy to formally identify and estimate the causal effect of
    investigation on fraud labels.

    This separates:
        - True causal effect: investigation CAUSES fraud discovery (= loop signal)
        - Spurious correlation: high-risk claims are both more likely investigated
          AND more likely fraudulent (= legitimate model signal)
    """
    if not DOWHY_AVAILABLE:
        return {'error': 'DoWhy not available. Run: pip install dowhy'}

    if feature_cols is None:
        feature_cols = FEATURE_COLS

    # Only use investigated claims (where outcome is observed)
    invest_mask = df[treatment_col] == 1
    analysis_df = df[invest_mask].dropna(subset=[outcome_col]).copy()
    analysis_df[outcome_col] = analysis_df[outcome_col].astype(int)
    analysis_df[treatment_col] = analysis_df[treatment_col].astype(int)

    # Columns needed for DoWhy
    needed_cols = [treatment_col, outcome_col] + feature_cols
    analysis_df = analysis_df[needed_cols].dropna()

    try:
        model = CausalModel(
            data=analysis_df,
            treatment=treatment_col,
            outcome=outcome_col,
            graph=CAUSAL_DAG_DOT,
            common_causes=feature_cols,
        )

        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

        # Estimate using propensity score weighting
        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.propensity_score_weighting",
            confidence_intervals=True,
        )

        # Refutation: random common cause test
        refute = model.refute_estimate(
            identified_estimand, estimate,
            method_name="random_common_cause",
            num_simulations=10,
        )

        return {
            'causal_estimate': round(estimate.value, 4),
            'confidence_interval': estimate.get_confidence_intervals(),
            'refutation_p_value': round(refute.new_effect, 4) if hasattr(refute, 'new_effect') else None,
            'interpretation': (
                f"Causal effect of investigation on fraud label: {estimate.value:.4f}. "
                f"This is the portion of the observed fraud rate attributable to the "
                f"investigation decision itself (loop signal), not underlying claim risk."
            )
        }

    except Exception as e:
        return {'error': f'DoWhy estimation failed: {str(e)}'}


# ─────────────────────────────────────────────
# 2. IPW Re-weighted Training
# ─────────────────────────────────────────────

def train_ipw_debiased_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    investigation_col: str = 'investigated',
    fraud_col: str = 'observed_fraud',
    score_col: str = None,
    clip_weights: float = 10.0,
) -> tuple[LogisticRegression, np.ndarray]:
    """
    Train a logistic regression model on IPW-weighted training data.

    IPW weight for claim i:
        w_i = 1 / P(investigated=1 | features_i)

    This gives higher weight to claims that were investigated DESPITE
    having low model scores (random or near-threshold investigations),
    making the training set more representative of the full population.

    Args:
        clip_weights: Maximum allowed weight (prevents extreme influence)

    Returns:
        (debiased_model, ipw_weights)
    """
    # Estimate propensity scores
    X_ps = train_df[feature_cols].values
    y_ps = train_df[investigation_col].values.astype(int)
    ps_model = LogisticRegression(max_iter=1000)
    ps_model.fit(X_ps, y_ps)
    propensity = ps_model.predict_proba(X_ps)[:, 1]
    propensity = np.clip(propensity, 0.01, 0.99)

    # IPW weights
    ipw_weights = 1.0 / propensity
    ipw_weights = np.clip(ipw_weights, 1.0, clip_weights)  # clip extremes

    # Train debiased model on INVESTIGATED claims with IPW weights
    invest_mask = train_df[investigation_col] == 1
    fraud_labels = train_df.loc[invest_mask, fraud_col].dropna()
    invest_idx = train_df[invest_mask].index

    X_train = train_df.loc[invest_idx, feature_cols].values
    y_train = fraud_labels.values.astype(int)
    w_train = ipw_weights[invest_mask][:len(y_train)]

    if len(np.unique(y_train)) < 2:
        raise ValueError("Training labels have only one class after filtering — need more data")

    debiased_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    debiased_model.fit(X_train, y_train, sample_weight=w_train)

    print(f"IPW debiased model trained on {len(y_train):,} investigated claims")
    print(f"  Mean weight: {w_train.mean():.3f}")
    print(f"  Max weight (clipped at {clip_weights}): {w_train.max():.3f}")
    print(f"  Weight Gini (dispersion): {np.std(w_train) / np.mean(w_train):.3f}")

    return debiased_model, ipw_weights


def train_standard_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    investigation_col: str = 'investigated',
    fraud_col: str = 'observed_fraud',
) -> LogisticRegression:
    """Train standard (biased) model on investigated claims without re-weighting."""
    invest_mask = train_df[investigation_col] == 1
    sub = train_df[invest_mask].dropna(subset=[fraud_col])

    X = sub[feature_cols].values
    y = sub[fraud_col].values.astype(int)

    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X, y)
    return model


# ─────────────────────────────────────────────
# 3. Evaluation on Debiased vs Standard Model
# ─────────────────────────────────────────────

def evaluate_models(
    test_df: pd.DataFrame,
    standard_model: LogisticRegression,
    debiased_model: LogisticRegression,
    feature_cols: list[str],
    true_fraud_col: str = 'true_fraud',
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Compare debiased vs standard model against true fraud labels (ground truth).

    In production, true_fraud_col would not exist — we'd use IPS-corrected metrics.
    Here we use ground truth for clean validation.
    """
    X_test = test_df[feature_cols].values
    true_labels = test_df[true_fraud_col].values

    scores_standard = standard_model.predict_proba(X_test)[:, 1]
    scores_debiased = debiased_model.predict_proba(X_test)[:, 1]

    preds_standard = (scores_standard >= threshold).astype(int)
    preds_debiased = (scores_debiased >= threshold).astype(int)

    results = []
    for model_name, scores, preds in [
        ('Standard (biased)', scores_standard, preds_standard),
        ('IPW Debiased', scores_debiased, preds_debiased),
    ]:
        auc = roc_auc_score(true_labels, scores)
        recall = recall_score(true_labels, preds, zero_division=0)
        precision = (
            np.sum((preds == 1) & (true_labels == 1)) /
            max(np.sum(preds == 1), 1)
        )
        brier = brier_score_loss(true_labels, scores)

        # Calibration: compare predicted rate vs actual rate in deciles
        deciles = pd.qcut(scores, 10, duplicates='drop', labels=False)
        calib_df = pd.DataFrame({'score': scores, 'true_fraud': true_labels, 'decile': deciles})
        calib = calib_df.groupby('decile').agg(
            pred_rate=('score', 'mean'),
            actual_rate=('true_fraud', 'mean')
        )
        calibration_error = np.mean(np.abs(calib['pred_rate'] - calib['actual_rate']))

        results.append({
            'Model': model_name,
            'AUC (vs true fraud)': round(auc, 4),
            'Recall (vs true fraud)': round(recall, 4),
            'Precision (vs true fraud)': round(precision, 4),
            'Brier Score': round(brier, 4),
            'Calibration Error (ECE)': round(calibration_error, 4),
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# 4. Segment-Level Evaluation
# ─────────────────────────────────────────────

def segment_level_comparison(
    test_df: pd.DataFrame,
    standard_model: LogisticRegression,
    debiased_model: LogisticRegression,
    feature_cols: list[str],
    segment_col: str,
    true_fraud_col: str = 'true_fraud',
) -> pd.DataFrame:
    """
    Compare models at segment level.

    The debiased model should show larger improvements in:
    - Segments that were historically under-investigated (blind spots)
    - High-postcode, night-time claims
    """
    X_test = test_df[feature_cols].values
    scores_standard = standard_model.predict_proba(X_test)[:, 1]
    scores_debiased = debiased_model.predict_proba(X_test)[:, 1]

    test_df = test_df.copy()
    test_df['score_standard'] = scores_standard
    test_df['score_debiased'] = scores_debiased

    agg = test_df.groupby(segment_col).apply(
        lambda g: pd.Series({
            'n_claims': len(g),
            'true_fraud_rate': g[true_fraud_col].mean(),
            'predicted_fraud_rate_standard': g['score_standard'].mean(),
            'predicted_fraud_rate_debiased': g['score_debiased'].mean(),
            'auc_standard': roc_auc_score(g[true_fraud_col], g['score_standard']) if g[true_fraud_col].nunique() > 1 else np.nan,
            'auc_debiased': roc_auc_score(g[true_fraud_col], g['score_debiased']) if g[true_fraud_col].nunique() > 1 else np.nan,
        })
    ).reset_index()

    agg['auc_improvement'] = agg['auc_debiased'] - agg['auc_standard']
    agg = agg.sort_values('auc_improvement', ascending=False)

    return agg


# ─────────────────────────────────────────────
# 5. Main Demo
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.append('../..')
    from builds.sfp_simulation.simulate_sfp_loop import generate_claims, run_full_simulation

    print("Generating data (loop active, no exploration)...")
    df, _ = run_full_simulation(n_claims=30_000, n_versions=3, epsilon=0.0)

    # Prepare data
    df['investigated'] = df['model_v3_investigated'].astype(int)
    df['observed_fraud'] = pd.to_numeric(df['model_v3_observed_fraud'], errors='coerce')

    # Train/test split (temporal-style: use first 70% for training)
    cutoff = int(len(df) * 0.7)
    train_df = df.iloc[:cutoff].copy()
    test_df = df.iloc[cutoff:].copy()

    print(f"\nTrain: {len(train_df):,} | Test: {len(test_df):,}")
    print(f"Train investigated: {train_df['investigated'].sum():,} ({train_df['investigated'].mean():.1%})")

    # Train standard (biased) model
    print("\n── Training Standard (Biased) Model ───────────────────")
    standard_model = train_standard_model(
        train_df, FEATURE_COLS, 'investigated', 'observed_fraud'
    )

    # Train IPW debiased model
    print("\n── Training IPW Debiased Model ────────────────────────")
    debiased_model, ipw_weights = train_ipw_debiased_model(
        train_df, FEATURE_COLS, 'investigated', 'observed_fraud'
    )

    # Evaluate both models against ground truth
    print("\n── Evaluation vs Ground Truth ─────────────────────────")
    eval_df = evaluate_models(
        test_df=test_df,
        standard_model=standard_model,
        debiased_model=debiased_model,
        feature_cols=FEATURE_COLS,
        true_fraud_col='true_fraud',
    )
    print(eval_df.to_string(index=False))

    # Segment-level analysis
    print("\n── Segment-Level Improvement (by high_postcode) ───────")
    seg_df = segment_level_comparison(
        test_df=test_df,
        standard_model=standard_model,
        debiased_model=debiased_model,
        feature_cols=FEATURE_COLS,
        segment_col='high_postcode',
    )
    print(seg_df[['high_postcode', 'true_fraud_rate', 'auc_standard', 'auc_debiased', 'auc_improvement']].to_string(index=False))

    # DoWhy analysis (if available)
    if DOWHY_AVAILABLE:
        print("\n── DoWhy Causal Analysis ────────────────────────────")
        result = run_dowhy_analysis(train_df, 'investigated', 'observed_fraud', FEATURE_COLS)
        if 'error' not in result:
            print(result['interpretation'])
        else:
            print(result['error'])
