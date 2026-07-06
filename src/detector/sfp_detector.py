from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from detector.algorithm.base import DetectionAlgorithm


@dataclass
class DetectionReport:
    """Result of running one detection algorithm on one model version."""

    version: str
    metrics: dict
    sfp_detected: bool

    @property
    def score(self) -> float:
        """The headline metric value (e.g. peak0), for convenient before/after comparison."""
        return self.metrics.get(self.metrics.get("_metric_name", ""), float("nan"))


class SFPDetector:
    """Runs its injected detection algorithm and packages the outcome as a DetectionReport."""

    def __init__(self, algorithm: DetectionAlgorithm) -> None:
        self.algorithm = algorithm

    def run(self, scores: pd.DataFrame, labels: pd.DataFrame, version: str) -> DetectionReport:
        metrics = self.algorithm.detect(scores, labels, score_col=f"model_{version}_score")
        metrics["_metric_name"] = self.algorithm.metric_name
        return DetectionReport(
            version=version,
            metrics=metrics,
            sfp_detected=bool(metrics["sfp_detected"]),
        )
