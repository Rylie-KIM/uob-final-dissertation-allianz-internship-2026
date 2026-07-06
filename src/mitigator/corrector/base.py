"""TrainingDataCorrector — the pluggable strategy interface for de-contaminating the target.

A corrector turns a version's (features, contaminated labels) into a corrected training set
(claim_id + label + weight) that `retrain.py` fits in the version env. It runs in the Analysis Layer
and loads no model. `SFPMitigator` holds one of these and delegates to it (Strategy pattern).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class TrainingDataCorrector(ABC):
    """Interface for producing a de-contaminated, weighted training target."""

    @abstractmethod
    def correct(self, features: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Return (corrected [claim_id, label, weight], diagnostics dict).

        features : claim_id + raw features
        labels   : claim_id + decision + observed_outcome (+ …)
        """
        raise NotImplementedError
