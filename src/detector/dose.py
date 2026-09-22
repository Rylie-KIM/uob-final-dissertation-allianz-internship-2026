"""DoseMeter — how much of a version's training label was forced by the PRIOR version's scrap
decision. A detection-layer metric (model-scrap share of the population), not a strategy
implementation — it does not fit `DetectionAlgorithm` (no scores/labels/score_col, and
deliberately reports no boolean verdict; see `estimator.effect.shap_did`'s use of it). A plain
class (no shared base, minor detection utility), sits directly under `detector/`, the same way
`report.py` does, rather than in `algorithm/`.

Moved here 2026-09-16 from `estimator/shap_did.py` (now `estimator/effect/shap_did.py`), where it
was only ever a mitigator/estimator
diagnostic borrowed from a detection quantity; turned into a class 2026-09-17 to match the
instantiate-then-call style of `detector.algorithm`'s strategies. `ShapDiDEstimator.falsify()`
still imports it from here for its parallel-trends proxy.
"""
from __future__ import annotations

import pandas as pd

import config
import schema


class DoseMeter:
    """Measures dose(v): among corrector_targets rows (already join- and mismatch-filtered,
    see 03_01_corrector_inputs), the share forced total-loss by the PRIOR version's scrap
    decision -- n_scrapped / n_out, conditioned on claims with a known treatment status.

    Different quantity from Table tab:data-composition's scrap rate, which divides by ALL
    target rows instead (n_scrapped / n_targets) -- so dose(v) >= that table's figure
    whenever the join dropped rows (unmatched claim_ids or observed/status mismatches)."""

    def measure(self, version: str, split: str) -> float | None:
        """`measure("v1", ...)` is 0 by construction: no prior model exists to have forced a v1
        label. Returns None ("unmeasured", never 0) when no `corrector_targets` file exists yet.
        """
        if version == "v1":
            return 0.0
        p = config.split_path("corrector_targets", version, split)
        if not p.exists():
            return None
        ct = pd.read_parquet(p)
        return float((ct[schema.DECISION] == 1).mean())
