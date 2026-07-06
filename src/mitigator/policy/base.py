"""InvestigationPolicy — pluggable strategy for WHICH claims to route to independent verification.

The second mitigation lever (alongside training-data correction): decide which scrap-eligible claims
should still be sent to a garage so the forced-label blind spot gets real observations over time.
Concrete policies (e.g. epsilon-greedy randomisation, uncertainty-based) are TBD (see
src/docs/DESIGN.md "Three Strategy Axes"); this ABC fixes the interface so `SFPMitigator` can hold one
without depending on any specific implementation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class InvestigationPolicy(ABC):
    """Interface for selecting claims to send for independent (garage) verification."""

    @abstractmethod
    def select(self, scores: pd.DataFrame, score_col: str) -> pd.Series:
        """Return a boolean Series (indexed like `scores`): True = route to garage instead of scrap."""
        raise NotImplementedError
