"""SFP detection (Analysis Layer — loads no model).

Public API: the pluggable `DetectionAlgorithm` strategies under `detector.algorithm`, called
directly (`ResidualPeakAlgorithm().detect(scores, labels, score_col=...)`). No context/wrapper
class holds them — `SFPDetector` was deleted 2026-09-16: it was imported by no real-data
notebook (they call `detector.algorithm.residual_peak.peak0`/`residual` directly), and its only
caller was `pipeline/pipeline.py`, which now calls the algorithm directly too.

`DoseMeter` and `ScoreDriftMeter` — minor detection-layer metrics (model-scrap share / own-vs-
prior score drift on a version's training population), neither a `DetectionAlgorithm`
implementation — sit directly here rather than under `algorithm/`, the same way `report.py` does.
`DoseMeter` moved from `estimator/shap_did.py` 2026-09-16 (`ShapDiDEstimator` still imports it
from here); `ScoreDriftMeter` extracted 2026-09-17 from `04_02_shap_did_concentration.ipynb` §4.
"""

# project scope changed -> detection algorithm on-hold. Only DoseMeter/ScoreDriftMeter are used.
# from detector.algorithm import DetectionAlgorithm, ResidualPeakAlgorithm
from detector.dose import DoseMeter
from detector.score_drift import ScoreDriftMeter

# __all__ = ["DetectionAlgorithm", "ResidualPeakAlgorithm", "DoseMeter", "ScoreDriftMeter"]
__all__ = ["DoseMeter", "ScoreDriftMeter"]

