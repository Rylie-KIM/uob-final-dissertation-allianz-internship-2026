"""
Build 00: EDA Template for Allianz Insurance Claims Dataset
============================================================
Exploratory data analysis template designed for the actual Allianz dataset.

Run this first when you receive the dataset. It will:
    1. Profile the dataset (shape, dtypes, missing values)
    2. Detect potential SFP loop signals in the data
    3. Analyse fraud rates across model versions
    4. Identify temporal trends and coverage gaps
    5. Generate a structured EDA report

Adapt column names (CLAIM_ID_COL etc.) to match the actual dataset schema.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────
# CONFIGURATION — Adapt to actual dataset schema
# ─────────────────────────────────────────────

CONFIG = {
    'claim_id_col': 'claim_id',
    'date_col': 'claim_date',
    'product_col': 'product_line',
    'amount_col': 'claim_amount',
    'claim_type_col': 'claim_type',
    'investigated_col': 'investigated',
    'fraud_label_col': 'fraud_label',
    'model_score_cols': ['model_v1_score', 'model_v2_score', 'model_v3_score'],
    'categorical_cols': ['product_line', 'claim_type', 'claimant_postcode_area'],
    'numeric_cols': ['claim_amount', 'settlement_amount', 'prior_claims'],
}


# ─────────────────────────────────────────────
# 1. Data Loading
# ─────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load the Allianz claims dataset."""
    path = Path(filepath)
    if path.suffix == '.csv':
        df = pd.read_csv(filepath, parse_dates=[CONFIG['date_col']], low_memory=False)
    elif path.suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(filepath, parse_dates=[CONFIG['date_col']])
    elif path.suffix == '.parquet':
        df = pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    print(f"Loaded: {len(df):,} rows × {len(df.columns)} columns")
    return df


# ─────────────────────────────────────────────
# 2. Data Profiling
# ─────────────────────────────────────────────

def profile_dataset(df: pd.DataFrame) -> dict:
    """Comprehensive data profiling."""
    print("\n" + "=" * 60)
    print("DATA PROFILE")
    print("=" * 60)

    # Basic shape
    print(f"\n  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")

    # Column types
    dtype_counts = df.dtypes.value_counts()
    print(f"\n  Column types:")
    for dtype, count in dtype_counts.items():
        print(f"    {str(dtype):<12} {count} columns")

    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({'missing_count': missing, 'missing_pct': missing_pct})
    missing_df = missing_df[missing_df['missing_count'] > 0].sort_values('missing_pct', ascending=False)

    print(f"\n  Missing values (columns with any missing):")
    if len(missing_df) == 0:
        print("    None")
    else:
        for col, row in missing_df.head(10).iterrows():
            print(f"    {col:<35} {row['missing_count']:>8,} ({row['missing_pct']:.1f}%)")

    # Date range
    date_col = CONFIG['date_col']
    if date_col in df.columns:
        print(f"\n  Date range: {df[date_col].min()} → {df[date_col].max()}")
        print(f"  Duration: {(df[date_col].max() - df[date_col].min()).days} days")

    return {
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'missing_summary': missing_df,
    }


# ─────────────────────────────────────────────
# 3. Fraud Rate Analysis
# ─────────────────────────────────────────────

def analyse_fraud_rates(df: pd.DataFrame) -> dict:
    """
    Analyse fraud rates across model versions and segments.

    Key SFP loop signal: if fraud rate increases with model version,
    the loop is likely active (model is becoming better at finding fraud
    where it looks, while creating more blind spots elsewhere).
    """
    print("\n" + "=" * 60)
    print("FRAUD RATE ANALYSIS")
    print("=" * 60)

    fraud_col = CONFIG['fraud_label_col']
    invest_col = CONFIG['investigated_col']
    score_cols = [c for c in CONFIG['model_score_cols'] if c in df.columns]

    if fraud_col not in df.columns:
        print("  WARNING: Fraud label column not found. Check CONFIG.")
        return {}

    # Overall investigation and fraud rates
    invest_rate = df[invest_col].mean() if invest_col in df.columns else None
    invest_df = df[df[invest_col] == 1] if invest_col in df.columns else df
    fraud_rate_observed = invest_df[fraud_col].mean()

    print(f"\n  Investigation rate: {invest_rate:.1%}" if invest_rate else "  Investigation column not found")
    print(f"  Observed fraud rate (investigated only): {fraud_rate_observed:.3f}")
    print(f"  n investigated: {len(invest_df):,}")
    print(f"  n uninvestigated: {len(df) - len(invest_df):,}")

    # Fraud rate by product line
    if CONFIG['product_col'] in df.columns:
        print(f"\n  Fraud rate by product line (investigated claims):")
        fraud_by_product = (
            invest_df.groupby(CONFIG['product_col'])[fraud_col]
            .agg(['mean', 'count'])
            .rename(columns={'mean': 'fraud_rate', 'count': 'n_claims'})
            .sort_values('fraud_rate', ascending=False)
        )
        print(fraud_by_product.to_string())

    # Fraud rate by model score quartile (loop signal)
    results_by_version = {}
    for score_col in score_cols:
        if score_col not in df.columns:
            continue
        version = score_col.replace('_score', '').replace('model_', '')

        df_temp = df.copy()
        df_temp['score_quartile'] = pd.qcut(df_temp[score_col], 4, labels=['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'])

        # Among investigated claims only
        invest_temp = df_temp[df_temp[invest_col] == 1] if invest_col in df_temp.columns else df_temp
        fraud_by_quartile = invest_temp.groupby('score_quartile', observed=True)[fraud_col].agg(['mean', 'count'])
        fraud_by_quartile.columns = ['fraud_rate', 'n_investigated']

        # Investigation rate by quartile (should be high for top quartile in a loop)
        if invest_col in df_temp.columns:
            invest_by_quartile = df_temp.groupby('score_quartile', observed=True)[invest_col].mean()
            fraud_by_quartile['investigation_rate'] = invest_by_quartile

        print(f"\n  Version {version} — fraud rate and investigation rate by score quartile:")
        print(fraud_by_quartile.to_string())

        results_by_version[version] = fraud_by_quartile

    return {'results_by_version': results_by_version}


# ─────────────────────────────────────────────
# 4. Temporal Trend Analysis
# ─────────────────────────────────────────────

def analyse_temporal_trends(df: pd.DataFrame) -> None:
    """
    Plot fraud rate, investigation rate, and model score distributions over time.

    Key SFP signal:
        - Investigation rate should be stable (policy-driven)
        - Fraud rate in investigated claims should increase over versions if loop is active
        - Model score distribution should shift higher over versions
    """
    date_col = CONFIG['date_col']
    fraud_col = CONFIG['fraud_label_col']
    invest_col = CONFIG['investigated_col']
    score_cols = [c for c in CONFIG['model_score_cols'] if c in df.columns]

    if date_col not in df.columns:
        print("Date column not found — skipping temporal analysis")
        return

    df = df.copy()
    df['year_month'] = df[date_col].dt.to_period('M')

    monthly = df.groupby('year_month').agg(
        n_claims=(CONFIG['claim_id_col'], 'count'),
        investigation_rate=(invest_col, 'mean') if invest_col in df.columns else None,
        **{f'mean_{sc}': (sc, 'mean') for sc in score_cols}
    )

    if invest_col in df.columns:
        invest_df = df[df[invest_col] == 1]
        monthly_fraud = invest_df.groupby('year_month')[fraud_col].mean().rename('fraud_rate')
        monthly = monthly.join(monthly_fraud, how='left')

    # Plot
    n_plots = 2 + len(score_cols)
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots))
    x = range(len(monthly))

    axes[0].bar(x, monthly['n_claims'], color='steelblue', alpha=0.7)
    axes[0].set_title('Monthly Claim Volume')
    axes[0].set_ylabel('Count')

    if 'fraud_rate' in monthly.columns:
        axes[1].plot(x, monthly['fraud_rate'], 'o-', color='tomato')
        axes[1].set_title('Monthly Fraud Rate (investigated claims only)')
        axes[1].set_ylabel('Fraud Rate')

    for i, sc in enumerate(score_cols):
        col = f'mean_{sc}'
        if col in monthly.columns:
            axes[2 + i].plot(x, monthly[col], 'o-', color='purple')
            axes[2 + i].set_title(f'Mean Model Score — {sc}')

    plt.tight_layout()
    plt.savefig('temporal_trends.png', dpi=150)
    plt.show()
    print("Saved: temporal_trends.png")


# ─────────────────────────────────────────────
# 5. Investigation Coverage Analysis
# ─────────────────────────────────────────────

def coverage_analysis(df: pd.DataFrame) -> None:
    """
    Analyse which segments of claims are over/under-investigated.

    A segment with:
        - High model score but low investigation rate = potential false negative
        - Low model score but high investigation rate = potentially wasted resource
    """
    print("\n" + "=" * 60)
    print("INVESTIGATION COVERAGE ANALYSIS")
    print("=" * 60)

    invest_col = CONFIG['investigated_col']
    score_cols = [c for c in CONFIG['model_score_cols'] if c in df.columns]

    if not score_cols or invest_col not in df.columns:
        print("  Score or investigation columns not found.")
        return

    latest_score = score_cols[-1]
    df = df.copy()
    df['score_decile'] = pd.qcut(df[latest_score], 10, labels=False, duplicates='drop')

    coverage = df.groupby('score_decile').agg(
        n_claims=('score_decile', 'count'),
        investigation_rate=(invest_col, 'mean'),
        mean_score=(latest_score, 'mean'),
    ).reset_index()

    fraud_col = CONFIG['fraud_label_col']
    if fraud_col in df.columns:
        invest_fraud = df[df[invest_col] == 1].groupby('score_decile')[fraud_col].mean().rename('fraud_rate_observed')
        coverage = coverage.merge(invest_fraud, on='score_decile', how='left')

    print("\n  Investigation rate and fraud rate by model score decile:")
    print(coverage.to_string(index=False))

    # Expected: monotonically increasing investigation_rate with score_decile
    corr = coverage['score_decile'].corr(coverage['investigation_rate'])
    print(f"\n  Correlation (score decile vs investigation rate): {corr:.4f}")
    if corr > 0.8:
        print("  → STRONG score-driven investigation policy detected (loop risk HIGH)")
    elif corr > 0.5:
        print("  → Moderate score-driven investigation (loop risk MEDIUM)")
    else:
        print("  → Weak score-investigation correlation (loop risk LOW or random policy used)")


# ─────────────────────────────────────────────
# 6. Main EDA Runner
# ─────────────────────────────────────────────

def run_eda(filepath: str) -> None:
    """Run the full EDA pipeline on the Allianz dataset."""
    print("=" * 60)
    print("ALLIANZ CLAIMS DATASET — EDA")
    print("=" * 60)

    df = load_data(filepath)
    profile = profile_dataset(df)
    fraud_analysis = analyse_fraud_rates(df)
    coverage_analysis(df)
    analyse_temporal_trends(df)

    print("\n\n📋 EDA COMPLETE")
    print("Next steps:")
    print("  1. Review fraud rates across model versions for loop signal")
    print("  2. Check investigation rate vs score correlation")
    print("  3. Run loop_detector.py on this dataset")
    print("  4. Proceed to unbiased_eval.py with IPS correction")


# For testing with synthetic data
if __name__ == '__main__':
    import sys
    sys.path.append('../..')
    from builds.sfp_simulation.simulate_sfp_loop import generate_claims, run_full_simulation

    print("No dataset path provided — running with synthetic data for demo.")
    df, _ = run_full_simulation(n_claims=20_000, n_versions=3, epsilon=0.0)

    # Map to expected schema
    df = df.rename(columns={'model_v3_investigated': 'investigated'})
    df['fraud_label'] = pd.to_numeric(df['model_v3_observed_fraud'], errors='coerce')
    df['claim_date'] = pd.date_range('2022-01-01', periods=len(df), freq='H')
    df['product_line'] = np.random.choice(['motor', 'home', 'pet', 'commercial'], len(df), p=[0.4, 0.3, 0.15, 0.15])
    df['claim_type'] = np.random.choice(['theft', 'damage', 'injury'], len(df))
    df['claim_id'] = [f'CLM{i:06d}' for i in range(len(df))]

    profile_dataset(df)
    analyse_fraud_rates(df)
    coverage_analysis(df)
