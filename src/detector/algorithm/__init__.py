"""Pluggable SFP detection strategies (Analysis Layer — loads no model)."""
from detector.algorithm.base import DetectionAlgorithm
from detector.algorithm.residual_peak import ResidualPeakAlgorithm

__all__ = ["DetectionAlgorithm", "ResidualPeakAlgorithm"]
