"""SFP detection (Analysis Layer — loads no model).

Public API: `SFPDetector` (holds a strategy, returns `DetectionReport`) + the pluggable
`DetectionAlgorithm` strategies under `detector.algorithm`.
"""
from detector.algorithm import DetectionAlgorithm, ResidualPeakAlgorithm
from detector.sfp_detector import DetectionReport, SFPDetector

__all__ = ["SFPDetector", "DetectionReport", "DetectionAlgorithm", "ResidualPeakAlgorithm"]
