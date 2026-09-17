"""DetectionAlgorithm — the pluggable strategy interface for SFP detection.

A detection algorithm is a pure, oracle-free function of (scores, observed labels): it never loads a
model and never touches the hidden true garage outcome. Callers instantiate a concrete algorithm
(e.g. `ResidualPeakAlgorithm()`) and call `.detect(...)` directly, so swapping the detection method
is a one-line change (Strategy pattern, see src/docs/DESIGN.md "Strategy Axes"). No context/wrapper
class holds one — `SFPDetector` played that role until deleted 2026-09-16 (unused by any real
caller; see this package's `__init__.py`).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class DetectionAlgorithm(ABC):
    """Interface for a single SFP-detection test. Implementations live in this package."""

    #: human-readable name of the metric this algorithm reports (e.g. "peak0").
    metric_name: str = "metric"

    @abstractmethod
    def detect(self, scores: pd.DataFrame, labels: pd.DataFrame, score_col: str) -> dict:
        """Return a metrics dict; it MUST include a boolean key ``sfp_detected``.

        scores : claim_id + <score_col>       (produced offline by predict.py)
        targets: claim_id + observed (+ …)    (canonical names; contaminated target, NO oracle)
        """
        raise NotImplementedError
