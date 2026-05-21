"""
Build 01: SFP Loop Simulation
==============================
Simulates a self-fulfilling prophecy loop in insurance fraud detection.

Scenario:
  - 50,000 synthetic insurance claims with TRUE fraud labels (ground truth)
  - Model v1 trained on a small unbiased seed batch
  - Each version investigates only top-k% scored claims
  - Fraud label is ONLY observed in investigated claims
  - Model vN+1 is trained on biased labels from vN
  - We track loop amplification and recall degradation across versions

Usage:
    python simulate_sfp_loop.py
    OR import and call run_full_simulation()
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. Data Generation
# ─────────────────────────────────────────────

def generate_claims(n_claims: int = 50_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic insurance claims with TRUE fraud labels.

    Features:
        - claim_amount: log-normal (fraud claims tend to be larger)
        - claim_hour: time of day (night claims higher fraud)
        - postcode_risk: postcode-level risk (0=low, 1=high)
        - prior_claims: number of prior claims by claimant
        - claim_type: categorical (theft, damage, injury)

    True fraud model (latent):
        logit(P(fraud)) = -2.5 + 0.8*high_amount + 0.6*night +
                          1.2*high_postcode_risk + 0.4*prior_claims
    """
    rng = np.random.default_rng(seed)

    n = n_claims

    # Claim features
    claim_amount_raw = rng.lognormal(mean=7.5, sigma=1.2, size=n)  # GBP
    claim_hour = rng.integers(0, 24, size=n)
    postcode_risk = rng.beta(1.5, 3.0, size=n)  # 0-1, right-skewed (most low risk)
    prior_claims = rng.poisson(0.5, size=n).clip(0, 5)
    claim_type = rng.choice(['theft', 'damage', 'injury'], p=[0.3, 0.5, 0.2], size=n)

    # Derived features
    high_amount = (claim_amount_raw > np.percentile(claim_amount_raw, 75)).astype(float)
    night_claim = ((claim_hour >= 22) | (claim_hour <= 5)).astype(float)
    high_postcode = (postcode_risk > 0.6).astype(float)

    # True fraud probability (the ground truth we know but the model doesn't)
    log_odds = (-2.5
                + 0.8 * high_amount
                + 0.6 * night_claim
                + 1.2 * high_postcode
                + 0.4 * prior_claims
                + rng.normal(0, 0.3, n))  # individual noise

    true_fraud_prob = 1 / (1 + np.exp(-log_odds))
    true_fraud = rng.binomial(1, true_fraud_prob).astype(int)

    df = pd.DataFrame({
        'claim_id': [f'CLM{i:06d}' for i in range(n)],
        'claim_amount': claim_amount_raw.round(2),
        'claim_hour': claim_hour,
        'postcode_risk': postcode_risk.round(4),
        'prior_claims': prior_claims,
        'claim_type': claim_type,
        'high_amount': high_amount,
        'night_claim': night_claim,
        'high_postcode': high_postcode,
        'true_fraud': true_fraud,          # GROUND TRUTH — not available in production
        'true_fraud_prob': true_fraud_prob.round(4),
    })

    print(f"Generated {n:,} claims | True fraud rate: {true_fraud.mean():.3f} ({true_fraud.sum():,} frauds)")
    return df


# ─────────────────────────────────────────────
# 2. Model Training
# ─────────────────────────────────────────────

FEATURE_COLS = ['high_amount', 'night_claim', 'high_postcode', 'prior_claims']


def train_model(training_df: pd.DataFrame, weights: np.ndarray = None) -> LogisticRegression:
    """Train logistic regression fraud model on OBSERVED (biased) labels."""
    X = training_df[FEATURE_COLS].values
    y = training_df['observed_fraud'].values  # NOT true_fraud — this is the biased label
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X, y, sample_weight=weights)
    return model


def score_claims(model: LogisticRegression, df: pd.DataFrame) -> np.ndarray:
    """Return fraud risk scores (probability of fraud) for all claims."""
    X = df[FEATURE_COLS].values
    return model.predict_proba(X)[:, 1]


# ─────────────────────────────────────────────
# 3. Investigation Policy
# ─────────────────────────────────────────────

def apply_investigation_policy(
    df: pd.DataFrame,
    model_scores: np.ndarray,
    investigation_rate: float = 0.25,
    random_exploration_rate: float = 0.0,
    rng: np.random.Generator = None
) -> pd.DataFrame:
    """
    Assign investigation decisions based on model scores.

    Args:
        investigation_rate: Fraction of claims to investigate (model-driven)
        random_exploration_rate: Additional random sample (epsilon in epsilon-greedy)

    Returns:
        DataFrame with 'investigated' and 'observed_fraud' columns added.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n = len(df)
    threshold = np.percentile(model_scores, 100 * (1 - investigation_rate))

    # Model-driven investigations
    model_investigate = (model_scores >= threshold)

    # Random exploration (epsilon-greedy)
    random_investigate = rng.random(n) < random_exploration_rate

    investigated = (model_investigate | random_investigate)

    # Fraud is ONLY observable in investigated claims
    # Add small investigator error (2% false negative rate)
    investigator_fnr = 0.02
    observed_fraud = np.where(
        investigated,
        np.where(df['true_fraud'].values == 1,
                 (rng.random(n) > investigator_fnr).astype(int),  # true fraud detected with 98% prob
                 0),
        np.nan  # fraud label UNKNOWN for uninvestigated claims
    )

    result = df.copy()
    result['model_score'] = model_scores.round(4)
    result['investigated'] = investigated.astype(int)
    result['observed_fraud'] = observed_fraud

    return result


# ─────────────────────────────────────────────
# 4. Loop Simulation — Multiple Generations
# ─────────────────────────────────────────────

def run_full_simulation(
    n_claims: int = 50_000,
    n_versions: int = 5,
    investigation_rate: float = 0.25,
    epsilon: float = 0.0,  # random exploration rate
    seed: int = 42
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Simulate the full SFP loop across multiple model versions.

    Returns:
        full_df: All claims with columns for each model version's scores
        metrics: List of per-version performance metrics
    """
    rng = np.random.default_rng(seed)

    # Generate ground truth
    df = generate_claims(n_claims=n_claims, seed=seed)

    # Seed model: train on a small RANDOM sample (unbiased initial seed)
    seed_size = int(n_claims * 0.05)  # 5% random seed
    seed_idx = rng.choice(len(df), size=seed_size, replace=False)
    seed_df = df.iloc[seed_idx].copy()
    seed_df['observed_fraud'] = seed_df['true_fraud']  # unbiased seed

    current_model = train_model(seed_df)

    metrics_per_version = []
    all_training_dfs = []

    for version in range(1, n_versions + 1):
        # Score all claims with current model
        scores = score_claims(current_model, df)

        # Apply investigation policy
        versioned_df = apply_investigation_policy(
            df, scores,
            investigation_rate=investigation_rate,
            random_exploration_rate=epsilon,
            rng=rng
        )

        # Store version scores
        df[f'model_v{version}_score'] = scores
        df[f'model_v{version}_investigated'] = versioned_df['investigated']
        df[f'model_v{version}_observed_fraud'] = versioned_df['observed_fraud']

        # ── Performance metrics (against TRUE fraud labels — ground truth) ──
        investigated_mask = versioned_df['investigated'] == 1
        investigated_df = versioned_df[investigated_mask].copy()

        # AUC on investigated claims only (biased view)
        auc_biased = roc_auc_score(
            investigated_df['true_fraud'],
            investigated_df['model_score']
        )

        # True recall: what fraction of ALL true fraud did we find?
        true_fraud_found = (
            investigated_df['true_fraud'] == 1
        ).sum()
        true_recall = true_fraud_found / df['true_fraud'].sum()

        # Precision on investigated claims
        precision = precision_score(
            investigated_df['true_fraud'],
            (investigated_df['model_score'] > 0.5).astype(int),
            zero_division=0
        )

        # Loop amplification: fraud rate in top quartile vs true population rate
        top_quartile_mask = scores >= np.percentile(scores, 75)
        fraud_rate_top = df.loc[top_quartile_mask, 'true_fraud'].mean()
        true_fraud_rate = df['true_fraud'].mean()
        loop_amplification = fraud_rate_top / true_fraud_rate if true_fraud_rate > 0 else 1.0

        # Blind spot: fraction of true fraud in NEVER-investigated segments
        # Segment by high_postcode x night_claim
        never_investigated_mask = versioned_df['investigated'] == 0
        fraud_in_blind_spot = (
            df.loc[never_investigated_mask, 'true_fraud'] == 1
        ).sum()
        blind_spot_fraction = fraud_in_blind_spot / df['true_fraud'].sum()

        metrics = {
            'version': version,
            'n_investigated': investigated_mask.sum(),
            'investigation_rate': investigated_mask.mean(),
            'observed_fraud_rate': investigated_df['observed_fraud'].mean(),
            'auc_on_investigated': round(auc_biased, 4),
            'true_recall': round(true_recall, 4),
            'precision': round(precision, 4),
            'loop_amplification_factor': round(loop_amplification, 3),
            'blind_spot_fraud_fraction': round(blind_spot_fraction, 4),
        }
        metrics_per_version.append(metrics)

        print(f"\n── Model v{version} ──────────────────────────────")
        print(f"  Investigated:       {metrics['n_investigated']:,} ({metrics['investigation_rate']:.1%})")
        print(f"  Observed fraud rate:{metrics['observed_fraud_rate']:.3f}")
        print(f"  AUC (biased):       {metrics['auc_on_investigated']:.4f}")
        print(f"  TRUE Recall:        {metrics['true_recall']:.4f}   ← real coverage")
        print(f"  Precision:          {metrics['precision']:.4f}")
        print(f"  Loop Amplification: {metrics['loop_amplification_factor']:.2f}x")
        print(f"  Blind Spot Fraud:   {metrics['blind_spot_fraud_fraction']:.1%} of all fraud undetectable")

        # Train next version model on OBSERVED (biased) labels
        training_df = versioned_df[versioned_df['investigated'] == 1].copy()
        training_df = training_df.dropna(subset=['observed_fraud'])
        training_df['observed_fraud'] = training_df['observed_fraud'].astype(int)
        all_training_dfs.append(training_df)

        # Combine all historical training data
        combined_training = pd.concat(all_training_dfs, ignore_index=True)
        current_model = train_model(combined_training)

    return df, metrics_per_version


# ─────────────────────────────────────────────
# 5. Visualisation
# ─────────────────────────────────────────────

def plot_loop_degradation(metrics_list: list[dict], title_suffix: str = "") -> None:
    """Plot how metrics evolve across model versions."""
    versions = [m['version'] for m in metrics_list]
    auc_scores = [m['auc_on_investigated'] for m in metrics_list]
    true_recalls = [m['true_recall'] for m in metrics_list]
    blind_spots = [m['blind_spot_fraud_fraction'] for m in metrics_list]
    amplifications = [m['loop_amplification_factor'] for m in metrics_list]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'SFP Loop Effect Across Model Versions {title_suffix}', fontsize=14)

    axes[0, 0].plot(versions, auc_scores, 'o-', color='steelblue')
    axes[0, 0].set_title('AUC (on investigated claims — BIASED)')
    axes[0, 0].set_xlabel('Model Version')
    axes[0, 0].set_ylabel('AUC-ROC')
    axes[0, 0].set_ylim(0.5, 1.0)

    axes[0, 1].plot(versions, true_recalls, 'o-', color='tomato')
    axes[0, 1].set_title('TRUE Recall (against ground truth)')
    axes[0, 1].set_xlabel('Model Version')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].set_ylim(0, 1.0)

    axes[1, 0].plot(versions, blind_spots, 'o-', color='darkorange')
    axes[1, 0].set_title('Blind Spot Fraction (fraud never found)')
    axes[1, 0].set_xlabel('Model Version')
    axes[1, 0].set_ylabel('Fraction of True Fraud')
    axes[1, 0].set_ylim(0, 1.0)

    axes[1, 1].plot(versions, amplifications, 'o-', color='purple')
    axes[1, 1].axhline(1.0, color='grey', linestyle='--', label='No amplification')
    axes[1, 1].set_title('Loop Amplification Factor')
    axes[1, 1].set_xlabel('Model Version')
    axes[1, 1].set_ylabel('Amplification (×)')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('sfp_loop_degradation.png', dpi=150)
    plt.show()
    print("Saved: sfp_loop_degradation.png")


def compare_policies(
    n_claims: int = 20_000,
    n_versions: int = 5,
    epsilons: list[float] = [0.0, 0.05, 0.10, 0.20]
) -> pd.DataFrame:
    """Compare SFP loop under different epsilon-greedy exploration rates."""
    results = []
    for eps in epsilons:
        print(f"\n{'='*50}")
        print(f"Policy: ε = {eps} ({'pure model' if eps == 0 else f'{eps:.0%} random exploration'})")
        print(f"{'='*50}")
        _, metrics = run_full_simulation(
            n_claims=n_claims, n_versions=n_versions,
            investigation_rate=0.25, epsilon=eps, seed=42
        )
        for m in metrics:
            m['epsilon'] = eps
            results.append(m)

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("SFP Loop Simulation — Allianz UK Fraud Detection")
    print("=" * 60)

    # Run simulation without exploration (pure loop)
    df_no_explore, metrics_no_explore = run_full_simulation(
        n_claims=50_000,
        n_versions=5,
        investigation_rate=0.25,
        epsilon=0.0
    )

    # Run simulation with epsilon-greedy exploration
    df_epsilon, metrics_epsilon = run_full_simulation(
        n_claims=50_000,
        n_versions=5,
        investigation_rate=0.25,
        epsilon=0.05
    )

    # Compare
    print("\n\n📊 COMPARISON: Pure Loop vs ε=0.05 Exploration")
    print("─" * 60)
    print(f"{'Version':<10} {'Recall (no ε)':<18} {'Recall (ε=0.05)':<18} {'Improvement'}")
    for m1, m2 in zip(metrics_no_explore, metrics_epsilon):
        improvement = m2['true_recall'] - m1['true_recall']
        print(f"  v{m1['version']:<8} {m1['true_recall']:<18.4f} {m2['true_recall']:<18.4f} +{improvement:.4f}")

    # Plot
    plot_loop_degradation(metrics_no_explore, title_suffix="(No Exploration)")
