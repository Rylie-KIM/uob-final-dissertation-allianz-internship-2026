"""
Build 05: Randomisation Strategy — Epsilon-Greedy Exploration
=============================================================
Implements and compares investigation policies to break SFP loops.

Key insight:
    A purely model-driven investigation policy creates a loop because
    the model's selection determines what labels exist in training data.
    By randomly investigating a fraction of claims (epsilon), we generate
    counterfactual observations: what would we have found in claims the
    model would normally skip?

Policies compared:
    1. Pure model     (ε = 0.00) — baseline loop
    2. ε = 0.05       — 5% random exploration
    3. ε = 0.10       — 10% random exploration
    4. ε = 0.20       — 20% random exploration

Metrics tracked:
    - True recall recovery over time (vs ground truth)
    - Loop amplification factor over versions
    - Investigation cost (fraction of wasted investigations on non-fraud claims)
    - Blind spot reduction rate
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


FEATURE_COLS = ['high_amount', 'night_claim', 'high_postcode', 'prior_claims']


# ─────────────────────────────────────────────
# 1. Policy Definitions
# ─────────────────────────────────────────────

class InvestigationPolicy:
    """Base class for investigation policies."""
    name: str = "Base"

    def decide(
        self,
        scores: np.ndarray,
        investigation_budget: float,
        rng: np.random.Generator
    ) -> np.ndarray:
        """Return binary array: 1 = investigate this claim, 0 = skip."""
        raise NotImplementedError


class PureModelPolicy(InvestigationPolicy):
    """Investigate top-k% claims by model score. No exploration."""
    name = "Pure Model (ε=0)"

    def decide(self, scores, investigation_budget, rng):
        n = len(scores)
        n_investigate = int(n * investigation_budget)
        threshold = np.sort(scores)[::-1][n_investigate - 1]
        return (scores >= threshold).astype(int)


class EpsilonGreedyPolicy(InvestigationPolicy):
    """
    Epsilon-greedy: with probability ε, investigate randomly;
    with probability (1-ε), follow model ranking.
    """
    def __init__(self, epsilon: float):
        self.epsilon = epsilon
        self.name = f"ε-Greedy (ε={epsilon:.2f})"

    def decide(self, scores, investigation_budget, rng):
        n = len(scores)
        n_investigate = int(n * investigation_budget)

        # Number of random investigations
        n_random = int(n_investigate * self.epsilon)
        n_model = n_investigate - n_random

        # Model-driven: top n_model by score
        top_model_idx = np.argsort(scores)[::-1][:n_model]

        # Random: uniform sample from remaining claims
        remaining_idx = np.setdiff1d(np.arange(n), top_model_idx)
        random_idx = rng.choice(remaining_idx, size=min(n_random, len(remaining_idx)), replace=False)

        investigated = np.zeros(n, dtype=int)
        investigated[top_model_idx] = 1
        investigated[random_idx] = 1
        return investigated


class ThompsonSamplingPolicy(InvestigationPolicy):
    """
    Thompson Sampling: treat each claim as a Bernoulli bandit.
    Sample from Beta posterior for each claim; investigate top-k samples.
    (Simplified: uses model score as prior alpha parameter)
    """
    name = "Thompson Sampling"

    def decide(self, scores, investigation_budget, rng):
        n = len(scores)
        n_investigate = int(n * investigation_budget)

        # Beta parameters: alpha = score * 10 + 1, beta = (1-score) * 10 + 1
        alpha = scores * 10 + 1
        beta = (1 - scores) * 10 + 1

        # Sample from Beta distribution for each claim
        samples = rng.beta(alpha, beta)
        top_idx = np.argsort(samples)[::-1][:n_investigate]

        investigated = np.zeros(n, dtype=int)
        investigated[top_idx] = 1
        return investigated


# ─────────────────────────────────────────────
# 2. Online Simulation Engine
# ─────────────────────────────────────────────

def run_policy_simulation(
    df_full: pd.DataFrame,
    policy: InvestigationPolicy,
    n_periods: int = 10,
    investigation_budget: float = 0.25,
    initial_seed_size: float = 0.05,
    investigator_fnr: float = 0.02,
    seed: int = 42,
) -> list[dict]:
    """
    Simulate online deployment of an investigation policy over n_periods.

    Each period:
        1. Score all claims with current model
        2. Apply policy → get investigation decisions
        3. Observe fraud only in investigated claims (+ random subset if ε > 0)
        4. Retrain model on accumulated biased (or partially unbiased) data
        5. Record metrics against ground truth

    Returns:
        List of per-period metric dicts
    """
    rng = np.random.default_rng(seed)
    n = len(df_full)
    period_size = n // n_periods

    # Initial unbiased seed model
    seed_size = int(n * initial_seed_size)
    seed_idx = rng.choice(n, size=seed_size, replace=False)
    seed_df = df_full.iloc[seed_idx].copy()
    seed_df['observed_fraud'] = seed_df['true_fraud']

    current_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    X_seed = seed_df[FEATURE_COLS].values
    y_seed = seed_df['observed_fraud'].values
    current_model.fit(X_seed, y_seed)

    all_training_data = [seed_df]
    metrics_per_period = []

    for period in range(n_periods):
        start = period * period_size
        end = min(start + period_size, n)
        period_df = df_full.iloc[start:end].copy()

        if len(period_df) == 0:
            break

        # Score this period's claims
        X_period = period_df[FEATURE_COLS].values
        scores = current_model.predict_proba(X_period)[:, 1]

        # Apply policy
        investigated = policy.decide(scores, investigation_budget, rng)

        # Observe fraud in investigated claims
        observed_fraud = np.full(len(period_df), np.nan)
        invest_mask = investigated == 1
        true_fraud_period = period_df['true_fraud'].values

        observed_fraud[invest_mask] = np.where(
            true_fraud_period[invest_mask] == 1,
            (rng.random(invest_mask.sum()) > investigator_fnr).astype(int),
            0
        )

        period_df['model_score'] = scores
        period_df['investigated'] = investigated
        period_df['observed_fraud'] = observed_fraud

        # Compute metrics against TRUE ground truth
        true_fraud_all = df_full['true_fraud'].values
        model_scores_all = current_model.predict_proba(df_full[FEATURE_COLS].values)[:, 1]

        auc_true = roc_auc_score(true_fraud_all, model_scores_all)
        true_recall = recall_score(
            true_fraud_all,
            (model_scores_all >= 0.5).astype(int),
            zero_division=0
        )

        # Observed fraud rate in investigated claims (biased view)
        invest_mask_bool = invest_mask
        observed_fraud_rate = period_df.loc[invest_mask_bool, 'observed_fraud'].mean()

        # Blind spot: fraction of true fraud in uninvestigated claims (this period)
        blind_spot_fraud = (
            period_df.loc[~invest_mask_bool, 'true_fraud'] == 1
        ).sum()
        total_fraud_period = (period_df['true_fraud'] == 1).sum()
        blind_spot_fraction = blind_spot_fraud / max(total_fraud_period, 1)

        # Loop amplification
        top_q_mask = model_scores_all >= np.percentile(model_scores_all, 75)
        fraud_rate_top = true_fraud_all[top_q_mask].mean()
        true_fraud_rate = true_fraud_all.mean()
        loop_amp = fraud_rate_top / true_fraud_rate if true_fraud_rate > 0 else 1.0

        metrics_per_period.append({
            'period': period + 1,
            'policy': policy.name,
            'n_claims': len(period_df),
            'n_investigated': int(invest_mask.sum()),
            'investigation_rate': invest_mask.mean(),
            'observed_fraud_rate': round(observed_fraud_rate, 4),
            'auc_vs_true': round(auc_true, 4),
            'true_recall': round(true_recall, 4),
            'loop_amplification': round(loop_amp, 3),
            'blind_spot_fraction': round(blind_spot_fraction, 4),
        })

        # Retrain model on ALL accumulated data
        new_train = period_df.dropna(subset=['observed_fraud']).copy()
        new_train['observed_fraud'] = new_train['observed_fraud'].astype(int)
        all_training_data.append(new_train)
        combined = pd.concat(all_training_data, ignore_index=True)

        X_train = combined[FEATURE_COLS].values
        y_train = combined['observed_fraud'].values
        if len(np.unique(y_train)) > 1:
            current_model = LogisticRegression(max_iter=1000, class_weight='balanced')
            current_model.fit(X_train, y_train)

    return metrics_per_period


# ─────────────────────────────────────────────
# 3. Cost-Benefit Analysis
# ─────────────────────────────────────────────

def cost_benefit_analysis(
    metrics_dict: dict[str, list[dict]],
    investigation_cost_per_claim: float = 150.0,  # £ per investigation
    fraud_value_recovered: float = 3000.0,         # £ per fraud detected
) -> pd.DataFrame:
    """
    Compute the financial cost-benefit of each exploration policy.

    Cost:   n_investigations × cost_per_investigation
    Benefit: n_fraud_detected × value_recovered

    For exploration policies, random investigations have lower hit rate
    but generate training data that improves future model recall.
    """
    rows = []
    for policy_name, metrics in metrics_dict.items():
        final_period = metrics[-1]
        avg_recall = np.mean([m['true_recall'] for m in metrics])
        final_recall = final_period['true_recall']
        avg_investigate_rate = np.mean([m['investigation_rate'] for m in metrics])
        total_n = sum(m['n_claims'] for m in metrics)
        total_investigated = sum(m['n_investigated'] for m in metrics)

        # Approximate: assume 8% true fraud rate across 50k claims
        assumed_fraud_rate = 0.08
        total_fraud_estimated = int(total_n * assumed_fraud_rate)

        total_cost = total_investigated * investigation_cost_per_claim
        total_fraud_detected = int(total_fraud_estimated * avg_recall)
        total_benefit = total_fraud_detected * fraud_value_recovered
        net_benefit = total_benefit - total_cost
        roi = (net_benefit / total_cost) * 100 if total_cost > 0 else 0

        rows.append({
            'Policy': policy_name,
            'Final Recall': f"{final_recall:.3f}",
            'Avg Recall (all periods)': f"{avg_recall:.3f}",
            'Total Investigations': f"{total_investigated:,}",
            'Total Cost (£)': f"£{total_cost:,.0f}",
            'Est. Fraud Detected': f"{total_fraud_detected:,}",
            'Total Benefit (£)': f"£{total_benefit:,.0f}",
            'Net Benefit (£)': f"£{net_benefit:,.0f}",
            'ROI (%)': f"{roi:.1f}%",
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 4. Visualisation
# ─────────────────────────────────────────────

def plot_policy_comparison(
    metrics_dict: dict[str, list[dict]],
    metric: str = 'true_recall',
    title: str = None
) -> None:
    """Plot a metric over time for multiple policies."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['tomato', 'steelblue', 'seagreen', 'darkorchid', 'darkorange']

    for i, (policy_name, metrics) in enumerate(metrics_dict.items()):
        periods = [m['period'] for m in metrics]
        values = [m[metric] for m in metrics]
        ax.plot(periods, values, 'o-', label=policy_name,
                color=colors[i % len(colors)], linewidth=2, markersize=6)

    metric_labels = {
        'true_recall': 'True Recall (vs ground truth)',
        'loop_amplification': 'Loop Amplification Factor',
        'blind_spot_fraction': 'Blind Spot Fraction (this period)',
        'auc_vs_true': 'AUC vs True Fraud Labels',
    }
    ylabel = metric_labels.get(metric, metric)

    ax.set_xlabel('Period')
    ax.set_ylabel(ylabel)
    ax.set_title(title or f'{ylabel} Over Periods')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f'policy_comparison_{metric}.png'
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Saved: {fname}")


# ─────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.append('../..')
    from builds.sfp_simulation.simulate_sfp_loop import generate_claims

    print("Generating 30,000 claims for policy comparison...")
    df = generate_claims(n_claims=30_000)

    policies = [
        PureModelPolicy(),
        EpsilonGreedyPolicy(epsilon=0.05),
        EpsilonGreedyPolicy(epsilon=0.10),
        EpsilonGreedyPolicy(epsilon=0.20),
        ThompsonSamplingPolicy(),
    ]

    all_metrics = {}

    for policy in policies:
        print(f"\n{'─'*50}")
        print(f"Simulating policy: {policy.name}")
        print(f"{'─'*50}")
        metrics = run_policy_simulation(
            df_full=df,
            policy=policy,
            n_periods=8,
            investigation_budget=0.25,
        )
        all_metrics[policy.name] = metrics
        final = metrics[-1]
        print(f"  Final recall: {final['true_recall']:.4f}")
        print(f"  Final loop amplification: {final['loop_amplification']:.2f}×")
        print(f"  Final blind spot: {final['blind_spot_fraction']:.1%}")

    # Plot comparison
    plot_policy_comparison(all_metrics, metric='true_recall',
                           title='True Recall Recovery Under Different Exploration Policies')
    plot_policy_comparison(all_metrics, metric='loop_amplification',
                           title='Loop Amplification Factor Under Different Policies')

    # Cost-benefit
    print("\n\n📊 COST-BENEFIT ANALYSIS")
    print("=" * 80)
    cb_df = cost_benefit_analysis(all_metrics)
    print(cb_df.to_string(index=False))
