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
    def correct(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        feature_cols: list[str] | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        """Return (corrected [claim_id, label, weight], diagnostics dict).

        features     : claim_id + the version's preprocessed feature matrix
        targets      : claim_id + decision + observed (+ …)  — canonical names, see src/schema.py
        feature_cols : which columns of `features` are MODEL INPUTS — `config.model_features(v)`,
                       read off that version's registry. Not optional in practice: the exported
                       matrix carries the target beside the inputs (and v3's its own predictions),
                       so a corrector that took "every column except claim_id" would fit its
                       nuisance model on the outcome it is trying to correct for.
        """
        raise NotImplementedError
