"""ScoreDriftMeter — how much a version's OWN score diverges from the PRIOR version's recorded
score, on claims the two versions share. `corrector_targets` already joins `version`'s claims to
the prior version's log score + decision by claim_id (`03_01_corrector_inputs.ipynb`) — this is
not a new join, just a comparison on top of it. A detection-layer metric at the same "minor
detection" level as `DoseMeter` (`detector/dose.py`): no scores/labels/score_col pair and no
boolean verdict, so it does not fit `DetectionAlgorithm` either. Sits directly under `detector/`,
not `algorithm/`.

Extracted 2026-09-17 from `notebook/real/shap_did/04_02_shap_did_concentration.ipynb` §4, which
still owns the plotting/table-saving around it.
"""
from __future__ import annotations

import pandas as pd

import config
import schema
from loaders import load


class ScoreDriftMeter:
    """Compares `version`'s own score against the PRIOR version's recorded score, on the claims
    `corrector_targets` links between them."""

    def measure(self, version: str, split: str, own_score_col: str) -> dict | None:
        """Returns None ("no shared claims measured") when `version` has no `corrector_targets`
        file for `split` yet. Otherwise a dict with n_shared, score_corr, mean_diff (own score
        minus prior version's score), and the merged claim-level `table` (columns: claim_id,
        prior_score, prior_decision, own_score) for the caller to plot/save.
        """
        p = config.split_path("corrector_targets", version, split)
        if not p.exists():
            return None
        prior = pd.read_parquet(p)[[schema.CLAIM_ID, "score", schema.DECISION]].rename(
            columns={"score": "prior_score", schema.DECISION: "prior_decision"})
        own = load(version, split=split).frame[[schema.CLAIM_ID, own_score_col]].rename(
            columns={own_score_col: "own_score"})
        merged = prior.merge(own, on=schema.CLAIM_ID, how="inner")
        n = len(merged)
        return {
            "n_shared": int(n),
            "score_corr": float(merged["prior_score"].corr(merged["own_score"])) if n else float("nan"),
            "mean_diff": float((merged["own_score"] - merged["prior_score"]).mean()) if n else float("nan"),
            "table": merged,
        }
