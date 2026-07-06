from __future__ import annotations

import pandas as pd

from mitigator.corrector.base import TrainingDataCorrector
from mitigator.policy.base import InvestigationPolicy


class SFPMitigator:
    def __init__(
        self,
        corrector: TrainingDataCorrector,
        policy: InvestigationPolicy | None = None,
    ) -> None:
        self.corrector = corrector
        self.policy = policy

    def correct_training_data(
        self, features: pd.DataFrame, labels: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict]:
        """Delegate to the injected corrector → (corrected [claim_id, label, weight], diagnostics)."""
        return self.corrector.correct(features, labels)

    def select_for_investigation(
        self, scores: pd.DataFrame, score_col: str
    ) -> pd.Series | None:
        """Delegate to the investigation policy if one is configured, else None."""
        if self.policy is None:
            return None
        return self.policy.select(scores, score_col)
