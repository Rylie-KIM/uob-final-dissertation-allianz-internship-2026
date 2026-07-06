"""DetectionAlgorithm — the pluggable strategy interface for SFP detection.

A detection algorithm is a pure, oracle-free function of (scores, observed labels): it never loads a
model and never touches the hidden true garage outcome. `SFPDetector` holds one of these and
delegates to it, so swapping the detection method is a one-line strategy change (Strategy pattern,
see src/docs/DESIGN.md "Three Strategy Axes").
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

        scores : claim_id + <score_col>            (produced offline by predict.py)
        labels : claim_id + observed_outcome (+ …)  (the contaminated production label; NO oracle)
        """
        raise NotImplementedError
