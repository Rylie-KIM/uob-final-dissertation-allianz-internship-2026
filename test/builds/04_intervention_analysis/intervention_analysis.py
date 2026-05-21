"""
Build 04: Intervention Analysis
================================
Uses multiple model versions as a natural experiment to measure the
causal effect of model deployment on fraud detection patterns.

Methods:
    1. Difference-in-Differences (DiD)
       — Compare claim outcomes before/after a new model version deployment
       — Treatment: claims above threshold (model-investigated)
       — Control: claims below threshold (not investigated)

    2. Regression Discontinuity Design (RDD)
       — Exploit the sharp investigation threshold as a quasi-random cutoff
       — Claims just above vs just below threshold = near-random assignment
       — Estimates: does investigation *cause* fraud discovery or just correlate?

    3. Propensity Score Matching
       — Match investigated vs non-investigated claims on observable features
       — Estimate Average Treatment Effect (ATE) of investigation

Key question: Is the observed fraud rate in high-scored claims a property of
the claims themselves, or an artifact of the investigation policy?
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. Difference-in-Differences (DiD)
# ─────────────────────────────────────────────

def difference_in_differences(
    df: pd.DataFrame,
    version_col: str,        # Column indicating model version period (e.g. 'model_period')
    pre_version: str,        # Value of version_col for pre-period
    post_version: str,       # Value of version_col for post-period
    treatment_col: str,      # Binary: above/below threshold
    outcome_col: str,        # Fraud label (binary)
) -> dict:
    """
    Difference-in-Differences estimator.

    Setup:
        Pre-period  = model v1 in deployment
        Post-period = model v2 in deployment
        Treatment   = claim scored above investigation threshold
        Control     = claim scored below threshold
        Outcome     = fraud label

    DiD estimate = (E[Y|treated,post] - E[Y|treated,pre])
                  - (E[Y|control,post] - E[Y|control,pre])

    Interpretation:
        DiD ≠ 0 means: the model version change causally shifted fraud
        detection patterns (beyond what would have changed anyway in
        the control group)
    """
    pre_mask = df[version_col] == pre_version
    post_mask = df[version_col] == post_version
    treated_mask = df[treatment_col] == 1
    control_mask = df[treatment_col] == 0

    # Four cells
    y_treated_pre = df.loc[pre_mask & treated_mask, outcome_col].dropna().mean()
    y_treated_post = df.loc[post_mask & treated_mask, outcome_col].dropna().mean()
    y_control_pre = df.loc[pre_mask & control_mask, outcome_col].dropna().mean()
    y_control_post = df.loc[post_mask & control_mask, outcome_col].dropna().mean()

    # DiD estimate
    did_estimate = (y_treated_post - y_treated_pre) - (y_control_post - y_control_pre)

    # Standard error via delta method (approximate)
    n_tp = (pre_mask & treated_mask).sum()
    n_tpost = (post_mask & treated_mask).sum()
    n_cp = (pre_mask & control_mask).sum()
    n_cpost = (post_mask & control_mask).sum()

    # SE using binomial variance approximation
    se = np.sqrt(
        y_treated_pre * (1 - y_treated_pre) / max(n_tp, 1) +
        y_treated_post * (1 - y_treated_post) / max(n_tpost, 1) +
        y_control_pre * (1 - y_control_pre) / max(n_cp, 1) +
        y_control_post * (1 - y_control_post) / max(n_cpost, 1)
    )
    z_stat = did_estimate / se if se > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return {
        'y_treated_pre': round(y_treated_pre, 4),
        'y_treated_post': round(y_treated_post, 4),
        'y_control_pre': round(y_control_pre, 4),
        'y_control_post': round(y_control_post, 4),
        'did_estimate': round(did_estimate, 4),
        'std_error': round(se, 4),
        'z_statistic': round(z_stat, 3),
        'p_value': round(p_value, 4),
        'significant': p_value < 0.05,
        'interpretation': (
            f"Model version change {'causally shifted' if p_value < 0.05 else 'did NOT significantly shift'} "
            f"fraud detection by {did_estimate:+.3f} (p={p_value:.3f})"
        )
    }


# ─────────────────────────────────────────────
# 2. Regression Discontinuity Design (RDD)
# ─────────────────────────────────────────────

def regression_discontinuity(
    df: pd.DataFrame,
    score_col: str,
    threshold: float,
    outcome_col: str,
    bandwidth: float = None,
    n_bins: int = 20,
) -> dict:
    """
    Regression Discontinuity Design at the investigation score threshold.

    Intuition:
        Claims scored just above the threshold are VERY similar to claims
        just below it (same true risk, same features), but:
        - Just above: investigated → fraud label observed
        - Just below: not investigated → fraud label NOT observed

        The jump in fraud discovery rate at the threshold is:
            (a) Due to investigation causing fraud discovery, OR
            (b) Due to genuinely riskier claims being just above threshold

        RDD measures the LOCAL causal effect of investigation.

    If the jump is large relative to the overall trend,
    investigation is CAUSING the observed fraud label, not just
    selecting for genuinely riskier claims.
    """
    df = df.copy()
    df['running_var'] = df[score_col] - threshold  # centered at threshold

    if bandwidth is None:
        bandwidth = df['running_var'].std() * 0.5

    # Local sample around threshold
    local_mask = df['running_var'].abs() <= bandwidth
    local_df = df[local_mask].copy()

    if len(local_df) < 50:
        return {'error': f'Insufficient data near threshold (n={len(local_df)}). Increase bandwidth.'}

    treated = local_df['running_var'] >= 0   # above threshold
    control = local_df['running_var'] < 0    # below threshold

    # Outcomes near threshold
    y_just_above = local_df.loc[treated, outcome_col].dropna().mean()
    y_just_below = local_df.loc[control, outcome_col].dropna().mean()
    rdd_estimate = y_just_above - y_just_below

    # Test significance
    above_outcomes = local_df.loc[treated, outcome_col].dropna()
    below_outcomes = local_df.loc[control, outcome_col].dropna()

    t_stat, p_value = stats.ttest_ind(above_outcomes, below_outcomes)

    # Binned visualization data
    df['score_bin'] = pd.cut(df[score_col], bins=n_bins)
    bin_stats = df.groupby('score_bin', observed=True).agg(
        mean_outcome=(outcome_col, 'mean'),
        n_claims=(outcome_col, 'count'),
        mean_score=(score_col, 'mean')
    ).reset_index().dropna()

    return {
        'threshold': threshold,
        'bandwidth': round(bandwidth, 4),
        'n_local_sample': len(local_df),
        'n_above': int(treated.sum()),
        'n_below': int(control.sum()),
        'y_just_above_threshold': round(y_just_above, 4),
        'y_just_below_threshold': round(y_just_below, 4),
        'rdd_estimate': round(rdd_estimate, 4),
        't_statistic': round(t_stat, 3),
        'p_value': round(p_value, 4),
        'significant': p_value < 0.05,
        'bin_stats': bin_stats,
        'interpretation': (
            f"At the investigation threshold ({threshold:.3f}), fraud discovery rate "
            f"jumps by {rdd_estimate:+.3f} (p={p_value:.3f}). "
            f"{'This jump CANNOT be explained by claim features alone — investigation causes fraud discovery.' if p_value < 0.05 else 'No significant discontinuity at threshold.'}"
        )
    }


def plot_rdd(rdd_result: dict, title: str = "RDD: Fraud Rate Around Investigation Threshold"):
    """Visualise the RDD discontinuity."""
    if 'error' in rdd_result:
        print(f"Cannot plot: {rdd_result['error']}")
        return

    bin_stats = rdd_result['bin_stats']
    threshold = rdd_result['threshold']

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(
        bin_stats['mean_score'], bin_stats['mean_outcome'],
        s=bin_stats['n_claims'] / bin_stats['n_claims'].max() * 100,
        alpha=0.7, color='steelblue', label='Fraud rate (bin average)'
    )
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Investigation threshold ({threshold:.3f})')

    # Add annotation for jump
    ax.annotate(
        f'RDD estimate: {rdd_result["rdd_estimate"]:+.3f}\n(p={rdd_result["p_value"]:.3f})',
        xy=(threshold, rdd_result['y_just_above_threshold']),
        xytext=(threshold + 0.1, rdd_result['y_just_above_threshold'] + 0.05),
        arrowprops=dict(arrowstyle='->', color='red'),
        fontsize=10, color='red'
    )

    ax.set_xlabel('Model Risk Score')
    ax.set_ylabel('Observed Fraud Rate')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig('rdd_discontinuity.png', dpi=150)
    plt.show()
    print("Saved: rdd_discontinuity.png")


# ─────────────────────────────────────────────
# 3. Propensity Score Matching
# ─────────────────────────────────────────────

def propensity_score_matching(
    df: pd.DataFrame,
    treatment_col: str,      # 'investigated' (1=treated, 0=control)
    outcome_col: str,        # 'observed_fraud'
    feature_cols: list[str],
    n_matches: int = 1,
    caliper: float = 0.05,
) -> dict:
    """
    Propensity Score Matching to estimate Average Treatment Effect (ATE)
    of investigation on fraud discovery.

    If investigation causes fraud discovery (investigation reveals fraud that
    would not otherwise have been detected), then:
        ATE = E[Y(1)] - E[Y(0)] > 0

    But if fraud is truly higher in the investigated group (model is correct),
    then ATE should be approximately 0 (investigation just reveals existing fraud).

    A high ATE means: investigation is CREATING the fraud label, not finding it.
    """
    treated_mask = df[treatment_col] == 1
    control_mask = df[treatment_col] == 0

    # Estimate propensity scores
    X = df[feature_cols].values
    y = df[treatment_col].values.astype(int)
    ps_model = LogisticRegression(max_iter=1000)
    ps_model.fit(X, y)
    ps_scores = ps_model.predict_proba(X)[:, 1]
    df = df.copy()
    df['propensity_score'] = ps_scores

    treated_df = df[treated_mask].copy().reset_index(drop=True)
    control_df = df[control_mask].copy().reset_index(drop=True)

    # Nearest-neighbour matching within caliper
    treated_ps = treated_df['propensity_score'].values.reshape(-1, 1)
    control_ps = control_df['propensity_score'].values.reshape(-1, 1)

    nn = NearestNeighbors(n_neighbors=n_matches, metric='euclidean')
    nn.fit(control_ps)
    distances, indices = nn.kneighbors(treated_ps)

    # Filter by caliper
    matched_treated_outcomes = []
    matched_control_outcomes = []
    n_matched = 0

    for i, (dist, idx) in enumerate(zip(distances, indices)):
        if dist[0] <= caliper:
            t_outcome = treated_df[outcome_col].iloc[i]
            c_outcomes = control_df[outcome_col].iloc[idx]
            if not (np.isnan(t_outcome) or c_outcomes.isna().any()):
                matched_treated_outcomes.append(t_outcome)
                matched_control_outcomes.extend(c_outcomes.tolist())
                n_matched += 1

    if n_matched == 0:
        return {'error': 'No matches found within caliper. Try increasing caliper or caliper=None.'}

    matched_treated_outcomes = np.array(matched_treated_outcomes)
    matched_control_outcomes = np.array(matched_control_outcomes)

    ate = matched_treated_outcomes.mean() - matched_control_outcomes.mean()
    t_stat, p_value = stats.ttest_ind(matched_treated_outcomes, matched_control_outcomes)

    return {
        'n_total_treated': int(treated_mask.sum()),
        'n_matched': n_matched,
        'match_rate': round(n_matched / treated_mask.sum(), 4),
        'mean_outcome_treated': round(matched_treated_outcomes.mean(), 4),
        'mean_outcome_control': round(matched_control_outcomes.mean(), 4),
        'ate_estimate': round(ate, 4),
        't_statistic': round(t_stat, 3),
        'p_value': round(p_value, 4),
        'significant': p_value < 0.05,
        'interpretation': (
            f"ATE of investigation on fraud label: {ate:+.4f} (p={p_value:.3f}). "
            f"{'Investigation is CAUSING fraud discovery (not just selecting for genuine fraud).' if p_value < 0.05 and ate > 0.05 else 'Investigation does not appear to inflate fraud labels beyond true underlying risk.'}"
        )
    }


# ─────────────────────────────────────────────
# 4. Main Demo
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.append('../..')
    from builds.sfp_simulation.simulate_sfp_loop import generate_claims, run_full_simulation

    print("Generating data for intervention analysis...")
    df, _ = run_full_simulation(n_claims=20_000, n_versions=3, epsilon=0.0)

    feature_cols = ['high_amount', 'night_claim', 'high_postcode', 'prior_claims']

    # Set up for PSM: use v3 investigated claims
    df['investigated'] = df['model_v3_investigated'].astype(int)
    df['observed_fraud'] = pd.to_numeric(df['model_v3_observed_fraud'], errors='coerce')

    # Proxy investigation threshold
    threshold = df['model_v3_score'].quantile(0.75)
    df['above_threshold'] = (df['model_v3_score'] >= threshold).astype(int)

    print("\n── RDD Analysis ─────────────────────────────────────────────")
    # Uninvestigated claims have NaN observed_fraud — fill with 0 ("not found").
    # This is correct for RDD: below the threshold, fraud was never looked for,
    # so the observed rate is 0 by construction of the investigation policy.
    df['observed_fraud_filled'] = df['observed_fraud'].fillna(0)
    df['true_fraud_float'] = df['true_fraud'].astype(float)

    # RDD 1: true fraud — does genuine risk actually jump at the threshold?
    rdd_true = regression_discontinuity(
        df=df,
        score_col='model_v3_score',
        threshold=threshold,
        outcome_col='true_fraud_float',
        bandwidth=0.1,
    )
    print("True fraud jump (genuine risk at threshold):")
    print(f"  {rdd_true['interpretation']}")
    print(f"  Jump: {rdd_true['rdd_estimate']:+.4f}  (p={rdd_true['p_value']:.4f})")

    # RDD 2: observed fraud — how much does *discovered* fraud jump?
    # This includes both genuine risk AND the investigation-caused label effect.
    rdd_observed = regression_discontinuity(
        df=df,
        score_col='model_v3_score',
        threshold=threshold,
        outcome_col='observed_fraud_filled',
        bandwidth=0.1,
    )
    print("\nObserved fraud jump (discovery rate at threshold):")
    print(f"  {rdd_observed['interpretation']}")
    print(f"  Jump: {rdd_observed['rdd_estimate']:+.4f}  (p={rdd_observed['p_value']:.4f})")

    # SFP inflation = the part of the observed jump NOT explained by true risk.
    # A large positive number means investigation is creating labels, not just finding fraud.
    sfp_inflation = rdd_observed['rdd_estimate'] - rdd_true['rdd_estimate']
    print(f"\n  SFP inflation at threshold: {sfp_inflation:+.4f}")
    print(f"  (observed jump − true jump = label inflation caused by investigation policy)")

    plot_rdd(rdd_observed, title="RDD: Observed Fraud Rate Around v3 Investigation Threshold")
    plot_rdd(rdd_true,     title="RDD: True Fraud Rate Around v3 Investigation Threshold")

    print("\n── Propensity Score Matching ────────────────────────────────")
    psm_result = propensity_score_matching(
        df=df,
        treatment_col='investigated',
        outcome_col='true_fraud_float',
        feature_cols=feature_cols,
        caliper=0.05
    )
    if 'error' not in psm_result:
        print(psm_result['interpretation'])
        print(f"  Matched {psm_result['n_matched']:,} of {psm_result['n_total_treated']:,} treated claims")
        print(f"  Mean fraud (treated): {psm_result['mean_outcome_treated']:.4f}")
        print(f"  Mean fraud (control): {psm_result['mean_outcome_control']:.4f}")
    else:
        print(psm_result['error'])
