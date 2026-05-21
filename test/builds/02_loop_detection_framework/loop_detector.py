"""
Build 02: SFP Loop Detection Framework
=======================================
Reusable Python module to detect self-fulfilling prophecy loops
in insurance claims data with multiple model versions.

Four-step detection algorithm:
    Step 1: Temporal prediction correlation
    Step 2: Label generation mechanism test
    Step 3: Action-outcome confounding test
    Step 4: Segment blind spot analysis

Usage:
    from loop_detector import SFPDetector

    detector = SFPDetector(
        claims_df=df,
        model_versions=['v1', 'v2', 'v3'],
        investigation_col='investigated',
        fraud_label_col='observed_fraud',
        true_fraud_col='true_fraud'  # optional — for validation only
    )
    report = detector.run_detection()
    detector.print_report(report)
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from typing import Optional


class SFPDetector:
    """
    Detects self-fulfilling prophecy loops in ML-driven insurance claim data.

    Args:
        claims_df: DataFrame with claims + model scores + investigation decisions + fraud labels
        model_versions: List of version identifiers, e.g. ['v1', 'v2', 'v3']
        score_prefix: Column name prefix for model score columns (e.g. 'model_' → 'model_v1_score')
        investigation_col: Column indicating whether a claim was investigated (binary)
        fraud_label_col: Observed fraud label (NaN for uninvestigated claims)
        feature_cols: Claim feature columns (used for partial correlation controls)
    """

    def __init__(
        self,
        claims_df: pd.DataFrame,
        model_versions: list[str],
        score_prefix: str = 'model_',
        investigation_col: str = 'investigated',
        fraud_label_col: str = 'observed_fraud',
        feature_cols: Optional[list[str]] = None,
        true_fraud_col: Optional[str] = None,
    ):
        self.df = claims_df.copy()
        self.versions = model_versions
        self.score_cols = [f'{score_prefix}{v}_score' for v in model_versions]
        self.invest_col = investigation_col
        self.fraud_col = fraud_label_col
        self.feature_cols = feature_cols or []
        self.true_fraud_col = true_fraud_col

        self._validate_inputs()

    def _validate_inputs(self):
        missing = [c for c in self.score_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing score columns: {missing}")
        if self.invest_col not in self.df.columns:
            raise ValueError(f"Missing investigation column: {self.invest_col}")

    # ─────────────────────────────────────────────────────────────
    # STEP 1: Temporal Prediction Correlation
    # ─────────────────────────────────────────────────────────────

    def step1_temporal_correlation(self) -> dict:
        """
        Measure Spearman correlation between consecutive model version scores.

        Interpretation:
            - High correlation (ρ > 0.85) is expected if models are similar
            - BUT: if correlation is high AND investigation rate is also score-driven,
              this suggests the loop is reinforcing predictions, not just similar data

        Returns:
            dict with correlation matrix and loop suspicion flags
        """
        corr_matrix = {}
        flags = []

        for i, v1 in enumerate(self.versions):
            for j, v2 in enumerate(self.versions):
                if i < j:
                    rho, pval = stats.spearmanr(
                        self.df[self.score_cols[i]],
                        self.df[self.score_cols[j]]
                    )
                    pair = f'{v1}→{v2}'
                    corr_matrix[pair] = {'spearman_rho': round(rho, 4), 'p_value': round(pval, 6)}

                    # Flag if consecutive versions have very high correlation
                    if j == i + 1 and rho > 0.85:
                        flags.append(
                            f"Versions {v1} and {v2} are highly correlated (ρ={rho:.3f}) — "
                            f"loop may be reinforcing scores"
                        )

        return {
            'step': 1,
            'name': 'Temporal Prediction Correlation',
            'correlation_matrix': corr_matrix,
            'flags': flags,
            'risk_signal': len(flags) > 0,
        }

    # ─────────────────────────────────────────────────────────────
    # STEP 2: Label Generation Mechanism Test
    # ─────────────────────────────────────────────────────────────

    def step2_label_mechanism(self) -> dict:
        """
        Test whether P(investigated=1) is strongly driven by model score.

        If investigation is determined by the model score (threshold policy),
        then fraud labels are MCAR-violated — they are missing NOT at random.

        Tests:
            1. Point-biserial correlation: model_score vs investigated
            2. Logistic regression AUC: can we predict investigated from score alone?
            3. Investigation rate by score decile
        """
        results_per_version = {}
        flags = []

        for v, score_col in zip(self.versions, self.score_cols):
            scores = self.df[score_col].values
            investigated = self.df[self.invest_col].values.astype(float)

            # Point-biserial correlation
            rho, pval = stats.pointbiserialr(investigated, scores)

            # AUC: can score predict investigation?
            auc = roc_auc_score(investigated, scores)

            # Investigation rate by decile
            decile_labels = pd.qcut(scores, 10, labels=False, duplicates='drop')
            invest_by_decile = (
                pd.Series(investigated).groupby(decile_labels).mean()
            ).to_dict()

            # Concentration: investigation in top 2 deciles
            top_decile_invest_rate = invest_by_decile.get(9, 0)
            bottom_decile_invest_rate = invest_by_decile.get(0, 0)
            concentration_ratio = (
                top_decile_invest_rate / bottom_decile_invest_rate
                if bottom_decile_invest_rate > 0 else float('inf')
            )

            results_per_version[v] = {
                'pointbiserial_rho': round(rho, 4),
                'investigation_auc': round(auc, 4),
                'concentration_ratio': round(concentration_ratio, 2),
                'invest_rate_top_decile': round(top_decile_invest_rate, 4),
                'invest_rate_bottom_decile': round(bottom_decile_invest_rate, 4),
            }

            if auc > 0.80:
                flags.append(
                    f"Version {v}: Model score predicts investigation with AUC={auc:.3f} — "
                    f"labels are NOT missing at random (MAR violation)"
                )
            if concentration_ratio > 10:
                flags.append(
                    f"Version {v}: Top decile investigated {concentration_ratio:.1f}× more than "
                    f"bottom decile — strong score-driven selection"
                )

        return {
            'step': 2,
            'name': 'Label Generation Mechanism Test',
            'results_per_version': results_per_version,
            'flags': flags,
            'risk_signal': len(flags) > 0,
        }

    # ─────────────────────────────────────────────────────────────
    # STEP 3: Action-Outcome Confounding Test
    # ─────────────────────────────────────────────────────────────

    def step3_action_outcome_confounding(self) -> dict:
        """
        Test whether investigation decision confounds the fraud label.

        Core test:
            Compare observed fraud rate in:
              (a) Model-driven investigated claims (score-selected)
              (b) Randomly investigated claims (if a random sample exists)

            If (a) >> (b): the investigation decision is a strong confounder
            → fraud labels are biased by the model's own selection

        Also computes:
            - Odds ratio: P(fraud | investigated) / P(fraud | random_sample)
            - Granger-style test: do vN scores predict v(N+1) LABELS after controlling for features?
        """
        flags = []

        # Check for random investigation sample (if available)
        # We use the first version's scores and assume claims in the bottom quartile
        # of the score were investigated "randomly" (they shouldn't have been)
        first_score_col = self.score_cols[0]
        scores_v1 = self.df[first_score_col].values
        investigated = self.df[self.invest_col].values

        fraud_labels = self.df[self.fraud_col].copy()
        investigated_mask = investigated == 1
        uninvestigated_mask = investigated == 0

        n_investigated = investigated_mask.sum()
        n_uninvestigated = uninvestigated_mask.sum()

        # Observed fraud rate in investigated claims
        investigated_fraud_labels = fraud_labels[investigated_mask].dropna()
        observed_fraud_rate = investigated_fraud_labels.mean() if len(investigated_fraud_labels) > 0 else np.nan

        # Compare: low-score investigated claims vs high-score investigated claims
        # (proxy for random vs model-driven)
        median_score = np.median(scores_v1[investigated_mask])
        low_score_invest_mask = investigated_mask & (scores_v1 < median_score)
        high_score_invest_mask = investigated_mask & (scores_v1 >= median_score)

        low_fraud_rate = fraud_labels[low_score_invest_mask].dropna().mean()
        high_fraud_rate = fraud_labels[high_score_invest_mask].dropna().mean()

        rate_ratio = high_fraud_rate / low_fraud_rate if low_fraud_rate > 0 else float('inf')

        if rate_ratio > 3:
            flags.append(
                f"Fraud rate in high-score investigated claims ({high_fraud_rate:.3f}) is "
                f"{rate_ratio:.1f}× higher than low-score claims ({low_fraud_rate:.3f}) — "
                f"investigation decision strongly confounds fraud labels"
            )

        # Granger-style test: does vN score predict v(N+1) labels (controlling for features)?
        granger_results = {}
        if len(self.versions) >= 2 and self.fraud_col in self.df.columns:
            for i in range(len(self.versions) - 1):
                v_curr = self.versions[i]
                v_next = self.versions[i + 1]
                score_col_curr = self.score_cols[i]
                invest_col_curr = f'model_{v_curr}_investigated' if f'model_{v_curr}_investigated' in self.df.columns else self.invest_col
                fraud_col_curr = f'model_{v_next}_observed_fraud' if f'model_{v_next}_observed_fraud' in self.df.columns else self.fraud_col

                if fraud_col_curr not in self.df.columns:
                    continue

                # Only look at claims investigated in v_next
                mask = self.df[invest_col_curr].fillna(0) == 1
                sub = self.df[mask].dropna(subset=[score_col_curr, fraud_col_curr])
                if len(sub) < 100:
                    continue

                y = sub[fraud_col_curr].astype(int)
                X_with_score = sub[[score_col_curr] + self.feature_cols].values
                X_without_score = sub[self.feature_cols].values if self.feature_cols else None

                try:
                    model_with = LogisticRegression(max_iter=500).fit(X_with_score, y)
                    auc_with = roc_auc_score(y, model_with.predict_proba(X_with_score)[:, 1])
                    granger_results[f'{v_curr}→{v_next}'] = {
                        'auc_with_prior_score': round(auc_with, 4),
                    }
                    if X_without_score is not None:
                        model_without = LogisticRegression(max_iter=500).fit(X_without_score, y)
                        auc_without = roc_auc_score(y, model_without.predict_proba(X_without_score)[:, 1])
                        incremental_auc = auc_with - auc_without
                        granger_results[f'{v_curr}→{v_next}']['auc_without_prior_score'] = round(auc_without, 4)
                        granger_results[f'{v_curr}→{v_next}']['incremental_auc_from_prior_score'] = round(incremental_auc, 4)
                        if incremental_auc > 0.03:
                            flags.append(
                                f"Version {v_curr} scores predict version {v_next} fraud labels "
                                f"with incremental AUC gain of {incremental_auc:.3f} — "
                                f"prior model is influencing future labels (loop signal)"
                            )
                except Exception:
                    pass

        return {
            'step': 3,
            'name': 'Action-Outcome Confounding Test',
            'n_investigated': int(n_investigated),
            'n_uninvestigated': int(n_uninvestigated),
            'observed_fraud_rate': round(observed_fraud_rate, 4) if not np.isnan(observed_fraud_rate) else None,
            'fraud_rate_high_score_investigated': round(high_fraud_rate, 4),
            'fraud_rate_low_score_investigated': round(low_fraud_rate, 4),
            'rate_ratio': round(rate_ratio, 2),
            'granger_style_results': granger_results,
            'flags': flags,
            'risk_signal': len(flags) > 0,
        }

    # ─────────────────────────────────────────────────────────────
    # STEP 4: Segment Blind Spot Analysis
    # ─────────────────────────────────────────────────────────────

    def step4_segment_blind_spots(
        self,
        segment_cols: Optional[list[str]] = None
    ) -> dict:
        """
        Identify claim segments with systematically low investigation rates
        relative to their estimated fraud risk (blind spots in the model).

        For each segment (e.g. postcode area, claim type, value band):
            - Compute average model score (proxy for estimated risk)
            - Compute actual investigation rate
            - Compute 'investigation gap': score rank - investigation rank
            - High gap = segment is rated risky but rarely investigated

        Returns ranked blind spot list with estimated missed fraud volume.
        """
        if segment_cols is None:
            # Auto-detect potential segment columns
            segment_cols = [
                c for c in self.df.columns
                if c not in self.score_cols
                and c != self.invest_col
                and c != self.fraud_col
                and self.df[c].dtype in ['object', 'category', 'int64']
                and self.df[c].nunique() < 50
            ]

        if not segment_cols:
            return {
                'step': 4,
                'name': 'Segment Blind Spot Analysis',
                'blind_spots': [],
                'flags': ['No suitable segment columns found — provide segment_cols parameter'],
                'risk_signal': False,
            }

        # Use last model version scores as the current "risk estimate"
        latest_score_col = self.score_cols[-1]
        scores = self.df[latest_score_col]
        investigated = self.df[self.invest_col]

        blind_spots = []

        for seg_col in segment_cols:
            try:
                group = self.df.groupby(seg_col).agg(
                    n_claims=(seg_col, 'count'),
                    mean_score=(latest_score_col, 'mean'),
                    investigation_rate=(self.invest_col, 'mean'),
                ).reset_index()

                group = group[group['n_claims'] >= 50]  # minimum segment size

                # Rank segments by score (high = risky) and by investigation rate
                group['score_rank'] = group['mean_score'].rank(ascending=False)
                group['invest_rank'] = group['investigation_rate'].rank(ascending=False)

                # Investigation gap: high positive = high risk but rarely investigated
                group['investigation_gap'] = group['score_rank'] - group['invest_rank']

                # Add observed fraud rate for investigated claims in this segment
                if self.fraud_col in self.df.columns:
                    invest_mask = self.df[self.invest_col] == 1
                    fraud_by_seg = (
                        self.df[invest_mask]
                        .groupby(seg_col)[self.fraud_col]
                        .mean()
                        .rename('observed_fraud_rate')
                        .reset_index()
                    )
                    group = group.merge(fraud_by_seg, on=seg_col, how='left')
                else:
                    group['observed_fraud_rate'] = np.nan

                group = group.sort_values('investigation_gap', ascending=False)
                top_blind_spots = group.head(5)

                for _, row in top_blind_spots.iterrows():
                    if row['investigation_gap'] > 3:
                        blind_spots.append({
                            'segment_col': seg_col,
                            'segment_value': row[seg_col],
                            'n_claims': int(row['n_claims']),
                            'mean_risk_score': round(row['mean_score'], 4),
                            'investigation_rate': round(row['investigation_rate'], 4),
                            'score_rank': int(row['score_rank']),
                            'invest_rank': int(row['invest_rank']),
                            'investigation_gap': round(row['investigation_gap'], 1),
                            'observed_fraud_rate': round(row.get('observed_fraud_rate', np.nan), 4),
                        })
            except Exception:
                continue

        blind_spots.sort(key=lambda x: x['investigation_gap'], reverse=True)
        flags = []
        if blind_spots:
            top = blind_spots[0]
            flags.append(
                f"Top blind spot: {top['segment_col']}={top['segment_value']} — "
                f"mean risk score {top['mean_risk_score']:.3f} but investigation rate only "
                f"{top['investigation_rate']:.1%}"
            )

        return {
            'step': 4,
            'name': 'Segment Blind Spot Analysis',
            'n_blind_spots_found': len(blind_spots),
            'blind_spots': blind_spots[:10],
            'flags': flags,
            'risk_signal': len(blind_spots) > 0,
        }

    # ─────────────────────────────────────────────────────────────
    # FULL DETECTION PIPELINE
    # ─────────────────────────────────────────────────────────────

    def run_detection(self, segment_cols: Optional[list[str]] = None) -> dict:
        """Run all four detection steps and return a complete loop risk report."""
        step1 = self.step1_temporal_correlation()
        step2 = self.step2_label_mechanism()
        step3 = self.step3_action_outcome_confounding()
        step4 = self.step4_segment_blind_spots(segment_cols=segment_cols)

        steps = [step1, step2, step3, step4]
        n_risk_signals = sum(s['risk_signal'] for s in steps)

        loop_risk_score = n_risk_signals / len(steps)  # 0–1 score
        severity = (
            'HIGH' if loop_risk_score >= 0.75 else
            'MEDIUM' if loop_risk_score >= 0.50 else
            'LOW'
        )

        all_flags = []
        for step in steps:
            all_flags.extend(step['flags'])

        return {
            'loop_risk_score': loop_risk_score,
            'severity': severity,
            'n_steps_flagged': n_risk_signals,
            'all_flags': all_flags,
            'step_results': {
                'step1_temporal_correlation': step1,
                'step2_label_mechanism': step2,
                'step3_action_confounding': step3,
                'step4_blind_spots': step4,
            }
        }

    def print_report(self, report: dict) -> None:
        """Pretty-print the detection report."""
        print("\n" + "=" * 65)
        print("  SFP Loop Detection Report")
        print("=" * 65)
        print(f"  Loop Risk Score : {report['loop_risk_score']:.0%}  [{report['severity']}]")
        print(f"  Steps Flagged   : {report['n_steps_flagged']} / 4")
        print()
        print("  Flags:")
        for i, flag in enumerate(report['all_flags'], 1):
            # Wrap long flags
            print(f"  {i:2d}. {flag[:100]}")
            if len(flag) > 100:
                print(f"       {flag[100:]}")
        print()
        print("  Blind Spots:")
        blind_spots = report['step_results']['step4_blind_spots']['blind_spots']
        if blind_spots:
            for bs in blind_spots[:5]:
                print(f"    → {bs['segment_col']}={bs['segment_value']} | "
                      f"risk={bs['mean_risk_score']:.3f} | "
                      f"investigated={bs['investigation_rate']:.1%} | "
                      f"gap={bs['investigation_gap']:.0f} ranks")
        else:
            print("    None detected (no segment columns provided or insufficient data)")
        print("=" * 65)


# ─────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.append('..')
    from builds.sfp_simulation.simulate_sfp_loop import generate_claims, run_full_simulation

    print("Running SFP simulation to generate demo data...")
    df, _ = run_full_simulation(n_claims=20_000, n_versions=3, epsilon=0.0)

    # Prep for detector (use v3 as current)
    df['investigated'] = df['model_v3_investigated']
    df['observed_fraud'] = df['model_v3_observed_fraud']
    df['observed_fraud'] = pd.to_numeric(df['observed_fraud'], errors='coerce')

    detector = SFPDetector(
        claims_df=df,
        model_versions=['v1', 'v2', 'v3'],
        score_prefix='model_',
        investigation_col='investigated',
        fraud_label_col='observed_fraud',
        feature_cols=['high_amount', 'night_claim', 'high_postcode', 'prior_claims'],
        true_fraud_col='true_fraud',
    )

    report = detector.run_detection(segment_cols=['night_claim', 'high_postcode', 'prior_claims'])
    detector.print_report(report)
