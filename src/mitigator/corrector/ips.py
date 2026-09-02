"""IPS (Inverse Propensity Score) corrector for the forced-label problem.

Column names below are the CANONICAL ones (src/schema.py), already translated at ingest.

The contamination: honest outcomes exist ONLY for garaged claims (decision=0); scrapped claims
(decision=1) have a FORCED observed=1. So the trustworthy subset (decision=0) is a biased sample
of the population — the scrap-like claims are missing. IPS corrects that bias:

  1. Fit a PROPENSITY model  e(x) = P(scrap=1 | x)  with logistic regression (NOT xgboost here).
  2. Keep only decision==0 rows (their `observed` IS the garage outcome).
  3. Weight each kept row by  w = 1 / (1 - e(x))  = 1 / P(not scrapped | x), clipped for stability.
     Rows that looked scrap-like but were garaged anyway are RARE -> large weight -> amplified.
  4. Emit (claim_id, label, weight); retrain.py (version env, xgboost) fits with sample_weight.

KNOWN TO BE INSUFFICIENT ON THIS PROBLEM, and kept deliberately. Step 2 discards every row above
the cutoff, so the corrected set has ZERO rows there and no weighting can represent a region with
no rows: positivity fails by construction, not by sample size. Its value is as the comparison that
shows WHY — the standard estimator, applied honestly, and the point at which it breaks. See
src/docs/STRUCTURE.md "Positivity is dead at tau".

WHICH COLUMNS THE PROPENSITY MODEL SEES. `--version`, not the matrix: the exported feature file
carries the TARGET beside the model inputs (and v3's its own `predicted_prob`), so "every column
except claim_id" would fit e(x) = P(scrap|x) on the outcome the scrap decision determines — a
near-perfect propensity, weights pinned at the clip, and a correction that corrects nothing.
The column list comes from config.model_features(version), i.e. that version's own registry.

  PYTHONPATH=src .venv/bin/python -m mitigator.corrector.ips --version v2 \
      --features src/data/real/inputs/features_v2.parquet \
      --targets  src/data/real/inputs/targets_v2.parquet \
      --out      src/data/real/mitigation/v2_corrected.parquet
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

import config
from mitigator.corrector.base import TrainingDataCorrector

# @TODO - turn to deprecated if not used 
class IPSCorrector(TrainingDataCorrector):
    """Inverse-propensity re-weighting of the honestly-labelled (decision==0) subset."""

    def __init__(
        self,
        weight_clip: float = 50.0,
        id_col: str = "claim_id",
        decision_col: str = "decision",
        outcome_col: str = "observed",     # canonical name — see src/schema.py
    ) -> None:
        self.weight_clip = weight_clip
        self.id_col = id_col
        self.decision_col = decision_col
        self.outcome_col = outcome_col

    def _feature_cols(self, features: pd.DataFrame, given: list[str] | None) -> list[str]:
        """The model-input columns of `features`, never inferred from the frame.

        The caller supplies them (config.model_features(version)); this only checks they are all
        present. Refusing when they are absent is deliberate: the silent fallback — every column
        but claim_id — puts the target into e(x), which is the one column that must stay out.
        """
        if not given:
            raise ValueError(
                "IPSCorrector needs feature_cols: which columns of `features` are model inputs. "
                "Pass config.model_features(version) — the exported matrix also carries the "
                "target (and v3's its own predictions), so they cannot be read off the frame."
            )
        missing = [c for c in given if c not in features.columns]
        if missing:
            raise ValueError(
                f"features is missing {len(missing)} model column(s), e.g. {missing[:8]}. "
                f"Point --features at that version's processed_inputs matrix."
            )
        extra = [c for c in features.columns if c not in given and c != self.id_col]
        if extra:
            print(f"  IPS: set aside {len(extra)} non-model column(s): {extra[:8]}"
                  + (" ..." if len(extra) > 8 else ""))
        return list(given)

    def correct(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        feature_cols: list[str] | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        id_col, dec_col, out_col = self.id_col, self.decision_col, self.outcome_col
        feat_cols = self._feature_cols(features, feature_cols)
        m = features.merge(labels[[id_col, dec_col, out_col]], on=id_col)

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

        # 2. keep honestly-labelled rows only (decision == 0 -> `observed` is the garage outcome)
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
    p.add_argument("--targets", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--version", required=True,
                   help="which version's registry names the model-input columns, e.g. v2")
    p.add_argument("--id-col", default="claim_id")
    a = p.parse_args()

    corrected, diag = IPSCorrector(id_col=a.id_col).correct(
        pd.read_parquet(a.features), pd.read_parquet(a.targets),
        feature_cols=config.model_features(a.version),
    )
    corrected.to_parquet(a.out, index=False)
    print(f"IPS: dropped {diag['n_scrapped_dropped']} forced rows, kept {diag['n_kept_honest']} "
          f"honest (mean w={diag['weight_mean']}, max w={diag['weight_max']}) -> {a.out}")


if __name__ == "__main__":
    main()
