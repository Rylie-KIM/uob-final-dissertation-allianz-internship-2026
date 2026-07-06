"""IPS (Inverse Propensity Score) corrector for the forced-label problem.

The contamination: honest labels exist ONLY for garaged claims (decision=0); scrapped claims
(decision=1) have a FORCED observed_outcome=1. So the trustworthy-label subset (decision=0) is a
biased sample of the population — the scrap-like claims are missing. IPS corrects that bias:

  1. Fit a PROPENSITY model  e(x) = P(scrap=1 | x)  with logistic regression (NOT xgboost here).
  2. Keep only decision==0 rows (their observed_outcome IS the garage truth).
  3. Weight each kept row by  w = 1 / (1 - e(x))  = 1 / P(not scrapped | x), clipped for stability.
     Rows that looked scrap-like but were garaged anyway are RARE -> large weight -> amplified.
  4. Emit (claim_id, label, weight); retrain.py (version env, xgboost) fits with sample_weight.

  PYTHONPATH=src .venv/bin/python -m mitigator.corrector.ips \
      --features src/data/synthetic/inputs/features_v2.parquet \
      --labels   src/data/synthetic/inputs/labels_v2.parquet \
      --out      src/data/synthetic/mitigation/v2_corrected.parquet
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from mitigator.corrector.base import TrainingDataCorrector


class IPSCorrector(TrainingDataCorrector):
    """Inverse-propensity re-weighting of the honestly-labelled (decision==0) subset."""

    def __init__(
        self,
        weight_clip: float = 50.0,
        id_col: str = "claim_id",
        decision_col: str = "decision",
        outcome_col: str = "observed_outcome",
    ) -> None:
        self.weight_clip = weight_clip
        self.id_col = id_col
        self.decision_col = decision_col
        self.outcome_col = outcome_col

    def correct(self, features: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        id_col, dec_col, out_col = self.id_col, self.decision_col, self.outcome_col
        m = features.merge(labels[[id_col, dec_col, out_col]], on=id_col)
        feat_cols = [c for c in features.columns if c != id_col]

        # 1. propensity e(x) = P(scrap=1 | x) — logistic regression, NOT xgboost
        num = m[feat_cols].select_dtypes("number")
        cat = m[feat_cols].select_dtypes(exclude="number")
        parts = [num.to_numpy()] if not num.empty else []
        if not cat.empty:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            parts.append(enc.fit_transform(cat))
        Xp = np.hstack(parts)
        e = LogisticRegression(max_iter=1000).fit(Xp, m[dec_col]).predict_proba(Xp)[:, 1]
        m["propensity"] = e

        # 2. keep honestly-labelled rows only (decision == 0 -> observed_outcome is garage truth)
        honest = m[m[dec_col] == 0].copy()

        # 3. IPS weight = 1 / P(not scrapped | x), clipped for stability
        honest["weight"] = np.clip(
            1.0 / np.clip(1.0 - honest["propensity"], 1e-3, 1.0), 1.0, self.weight_clip
        )
        honest["label"] = honest[out_col].astype(int)

        diag = {
            "n_total": int(len(m)),
            "n_scrapped_dropped": int((m[dec_col] == 1).sum()),
            "n_kept_honest": int(len(honest)),
            "weight_mean": round(float(honest["weight"].mean()), 3),
            "weight_max": round(float(honest["weight"].max()), 3),
            "propensity_mean": round(float(e.mean()), 3),
        }
        return honest[[id_col, "label", "weight"]], diag


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--id-col", default="claim_id")
    a = p.parse_args()

    corrected, diag = IPSCorrector(id_col=a.id_col).correct(
        pd.read_parquet(a.features), pd.read_parquet(a.labels)
    )
    corrected.to_parquet(a.out, index=False)
    print(f"IPS: dropped {diag['n_scrapped_dropped']} forced rows, kept {diag['n_kept_honest']} "
          f"honest (mean w={diag['weight_mean']}, max w={diag['weight_max']}) -> {a.out}")


if __name__ == "__main__":
    main()
